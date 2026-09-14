# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import T

from ...tensor_shim import _run_compiled
from .gemm_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    _buffer_load,
    _make_buffer,
    wait_lds_barrier,
)


class _LdsF32View:
    def __init__(self, ptr, pong=None, half_elements=0):
        self.ptr = ptr
        self.pong = pong
        self.half_elements = half_elements

    def at(self, idx, upper=False):
        if const_expr(self.pong is not None and upper):
            return fx.add_offset(self.pong, fx.make_int_tuple(idx - fx.Int32(self.half_elements)))
        return fx.add_offset(self.ptr, fx.make_int_tuple(idx))


class _SplitABuffer:
    def __init__(self, ping, pong):
        self.ptr = ping.ptr
        self.ping = ping
        self.pong = pong


# fmt: off
@flyc.jit
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc, unroll_a_pingpong=False, split_a_lds=False, fp8_b_waitcnt=False, scalar_tile_row_base=False,
    tvr_rsrc=None, tvr_row_bytes=0,
    wait_payload=None, work=None, return_acc_only=False):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = (
        N_ACC + NUM_ACC_N * _PACK
    )
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    a_tile_bound = None
    a_valid_rows = None
    if const_expr(tvr_rsrc is not None):
        # This tile's real height. num_records must be wave-uniform or the descriptor
        # waterfalls, and every lane of a wave works the same m_tile, so read it once.
        a_valid_rows = fx.Int32(rocdl.readfirstlane(
            T.i32, _buffer_load(tvr_rsrc, m_tile, fx.Int32).ir_value()))
        a_tile_bound = a_valid_rows * fx.Int32(tvr_row_bytes)
    if const_expr(scalar_tile_row_base):
        # Every active lane executes the same scheduled M tile. Keep its buffer
        # base scalar so direct-to-LDS loads need no descriptor waterfall.
        tile_row_base = fx.Int32(rocdl.readfirstlane(T.i32, tile_row_base.ir_value()))
    b_row = sched.gate_base_row(expert) + n_tile_base
    if const_expr(wait_payload is not None):
        # Local weights and their scales do not depend on remote A readiness.
        # Reuse the first pipeline stage across the payload wait.
        b0 = b_loader.load_step(b_row, fx.Int32(0))
        sb0 = b_scale.load_step(b_row, fx.Int32(0))
        wait_payload(work)
    a_gather.for_tile(tile_row_base, a_tile_bound)
    if const_expr(pipe_weights):
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base, a_valid_rows)
        wait_lds_barrier(0 if async_a_copy else 63)
        if const_expr(wait_payload is None):
            b0 = b_loader.load_step(b_row, fx.Int32(0))
        init = [mfma.zero_value for _ in range(N_ACC)]
        init += [h for ni_list in b0 for h in ni_list]
        if const_expr(async_a_copy):
            if const_expr(wait_payload is not None):
                init += sb0
            else:
                init += b_scale.load_step(
                    b_row,
                    fx.Int32(0),
                )
            init += a_scale.load_step(
                a_scale_lds,
                fx.Int32(0),
            )
        if const_expr(unroll_a_pingpong):
            assert async_a_copy and mfma_amajor and K_ITERS % 2 == 0

            @flyc.jit
            def fixed_step(values, step: fx.Int32, phase):
                accum = [Vec(v) for v in values[:N_ACC]]
                weights = [[Vec(values[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
                           for ni in range(NUM_ACC_N)]
                sb_cur = [fx.Int32(values[B_STATE_END + g]) for g in range_constexpr(NUM_B_SCALE)]
                sa_cur = [fx.Int32(values[SB_STATE_END + g]) for g in range_constexpr(NUM_A_SCALE)]
                next_step = step + fx.Int32(1)
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(next_step * fx.Int32(A_K_STEP_BYTES),
                                         (a_buf.pong if phase == 0 else a_buf.ping) if split_a_lds else a_buf,
                                         fx.Int32(0 if split_a_lds else (1 - phase) * a_lds_i32))
                rocdl.sched_barrier(0)
                sb_next = b_scale.load_step(b_row, next_step)
                sa_next = a_scale.load_step(a_scale_lds, next_step)

                def fixed_a_load(mi, ks):
                    return a_s2r.load_operand(
                        (a_buf.ping if phase == 0 else a_buf.pong) if split_a_lds else a_buf,
                        mi, ks, fx.Int32(0 if split_a_lds else phase * a_lds_i32))

                def next_b(ni):
                    return b_loader.load_ni(b_row, ni, next_step)

                accum, weights_next = mfma.call_pipe_am(fixed_a_load, weights, accum, sa_cur, sb_cur, next_b)
                # FP8 packs use TWO buffer_load_dwordx4 instructions each.
                # All next-B/scale loads follow the four A DMA instructions
                # (sched_barrier above); preserve those newer VMEM operations
                # while waiting for A before the WG barrier. The default-off
                # experiment is restricted to the ISA-audited split-LDS shape.
                wait_lds_barrier(NUM_ACC_N * _PACK * (2 if fp8_b_waitcnt else 1) + NUM_B_SCALE)
                return list(accum) + [v for row in weights_next for v in row] + sb_next + sa_next

            for pair, values in range(0, K_ITERS - 2, 2, init=init):
                first = fixed_step(values, fx.Int32(pair), 0)
                second = fixed_step(first, fx.Int32(pair) + fx.Int32(1), 1)
                values = yield second
            # The final prefetch is even -> odd; the existing final step consumes it.
            init = fixed_step(values, fx.Int32(K_ITERS - 2), 0)
        loop_steps = const_expr(0 if unroll_a_pingpong else K_ITERS - 1)
        for sp_i, state in range(0, loop_steps, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state[:N_ACC]]
            b_prev = [
                [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)
            ]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = sp + fx.Int32(1)
            if const_expr(async_a_copy):
                sb = [
                    fx.Int32(
                        state[B_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_B_SCALE
                    )
                ]
                sa = [
                    fx.Int32(
                        state[SB_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_A_SCALE
                    )
                ]
            else:
                sb = b_scale.load_step(b_row, sp)
                sa = a_scale.load_step(
                    a_scale_lds,
                    sp,
                )

            cached_a = None
            if const_expr(M_REPEAT == 4):
                # Previous K-end barrier made current ping/pong slot ready.
                # Keep its reads before DMA into the other slot.
                cached_a = [[a_s2r.load_operand(a_buf, mi, ks, cur_off)
                             for ks in range_constexpr(_PACK)]
                            for mi in range_constexpr(M_REPEAT)]
                rocdl.sched_barrier(0)

            def a_load(mi, ks, _base=cur_off):
                if const_expr(M_REPEAT == 4):
                    return cached_a[mi][ks]
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
                sb_next = b_scale.load_step(
                    b_row,
                    spn,
                )
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )

            def load_next(ni, _kn=spn):
                return b_loader.load_ni(b_row, ni, _kn)

            def load_next_ks(ni, ks, _kn=spn):
                return b_loader._load_pack(b_row, ni, _kn, ks)

            acc, b_next = mfma.call_pipe_retire_ks(
                a_load, b_prev, acc, sa, sb, load_next_ks)
            if const_expr(async_a_copy):
                # Retire current A-scale uses before reading the successor.
                rocdl.sched_barrier(0)
                sa_next = a_scale.load_step(a_scale_lds, spn)
                wait_lds_barrier(
                    NUM_ACC_N * _PACK
                    + NUM_B_SCALE
                )
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            yv = list(acc) + [h for ni_list in b_next for h in ni_list]
            if const_expr(async_a_copy):
                yv += sb_next
                yv += sa_next
            state = yield yv
        acc = [Vec(r) for r in state[:N_ACC]]
        b_prev = [
            [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
            for ni in range(NUM_ACC_N)
        ]
        final_off = (last & fx.Int32(1)) * fx.Int32(a_lds_i32)

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf.pong if split_a_lds else a_buf, mi, ks,
                                     fx.Int32(0) if split_a_lds else _base)

        if const_expr(async_a_copy):
            sb = [
                fx.Int32(
                    state[B_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_B_SCALE
                )
            ]
            sa = [
                fx.Int32(
                    state[SB_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_A_SCALE
                )
            ]
        else:
            sb = b_scale.load_step(b_row, last)
            sa = a_scale.load_step(
                a_scale_lds,
                last,
            )
        acc = mfma.call_pipe_am_final(
            final_a_load,
            b_prev,
            acc,
            sa,
            sb,
        )
    else:
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base, a_valid_rows)
        wait_lds_barrier(0 if async_a_copy else 63)
        init = [mfma.zero_value for _ in range(N_ACC)]
        if const_expr(unroll_a_pingpong):
            # Single-stage B reduces live operand state; A retains its two
            # independent LDS slots and compile-time phase addressing.
            assert async_a_copy and mfma_amajor and K_ITERS % 2 == 0

            @flyc.jit
            def streamed_step(values, step: fx.Int32, phase, has_next):
                accum = [Vec(v) for v in values]
                b = b_loader.load_step(b_row, step)
                sa = a_scale.load_step(a_scale_lds, step)
                sb = b_scale.load_step(b_row, step)
                if const_expr(has_next):
                    rocdl.sched_barrier(0)
                    a_gather.prefetch_to_lds(
                        (step + fx.Int32(1)) * fx.Int32(A_K_STEP_BYTES),
                        (a_buf.pong if phase == 0 else a_buf.ping) if split_a_lds else a_buf,
                        fx.Int32(0 if split_a_lds else (1 - phase) * a_lds_i32))
                    rocdl.sched_barrier(0)

                def a_load(mi, ks):
                    return a_s2r.load_operand(
                        (a_buf.ping if phase == 0 else a_buf.pong) if split_a_lds else a_buf,
                        mi, ks, fx.Int32(0 if split_a_lds else phase * a_lds_i32))

                accum = mfma.call(a_load, b, accum, sa, sb)
                wait_lds_barrier(0)
                return list(accum)

            for pair, values in range(0, K_ITERS - 2, 2, init=init):
                first = streamed_step(values, fx.Int32(pair), 0, True)
                second = streamed_step(first, fx.Int32(pair) + fx.Int32(1), 1, True)
                values = yield second
            penultimate = streamed_step(values, fx.Int32(K_ITERS - 2), 0, True)
            acc = streamed_step(penultimate, fx.Int32(K_ITERS - 1), 1, False)
        else:
            for sp_i, state in range(0, K_ITERS, 1, init=init):
                sp = fx.Int32(sp_i)
                acc = [Vec(a) for a in state]
                cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
                nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
                spn = (sp + fx.Int32(1) < last).select(
                    sp + fx.Int32(1),
                    last,
                )

                def a_load(mi, ks, _base=cur_off):
                    return a_s2r.load_operand(a_buf, mi, ks, _base)

                b = b_loader.load_step(b_row, sp)
                sa = a_scale.load_step(a_scale_lds, sp)
                sb = b_scale.load_step(b_row, sp)
                if const_expr(async_a_copy):
                    rocdl.sched_barrier(0)
                    a_gather.prefetch_to_lds(
                        spn * fx.Int32(A_K_STEP_BYTES),
                        a_buf,
                        nxt_off,
                    )
                    rocdl.sched_barrier(0)
                else:
                    a_regs = a_gather.load_regs(
                        spn * fx.Int32(A_K_STEP_BYTES)
                    )
                acc = mfma.call(
                    a_load,
                    b,
                    acc,
                    sa,
                    sb,
                )
                if const_expr(async_a_copy):
                    wait_lds_barrier(0)
                else:
                    a_gather.store(a_buf, a_regs, nxt_off)
                    wait_lds_barrier()
                state = yield list(acc)
            acc = [Vec(r) for r in state]
    if const_expr(return_acc_only):
        # Preserve the physical SBM accumulator layout at the common epilogue.
        # Short-path groups beyond M_REPEAT are fully padding and stay zero.
        return list(acc) + [mfma.zero_value for _ in
                            range_constexpr((epi._m_repeat - M_REPEAT) * NUM_ACC_N)]
    else:
        # The epilogue aliases A_buf after all waves finish their A reads.
        wait_lds_barrier()
        epi.store(acc, m_tile, tile_row_base, n_tile_base)


@flyc.jit
def do_tile_short_or_full(m_tile, n_tile_base, expert, sched, a_gather, a_s2r,
    b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds, a_lds_i32,
    K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc, unroll_a_pingpong=False,
    split_a_lds=False, fp8_b_waitcnt=False, scalar_tile_row_base=False,
    tvr_rsrc=None, tvr_row_bytes=0, wait_payload=None, work=None, mfma_short=None, a_gather_short=None):
    # All waves in this CTA own the same M tile and take the same arm. Each
    # arm has a complete K loop with fixed M and the original B double buffer.
    valid_rows = fx.Int32(rocdl.readfirstlane(
        T.i32, _buffer_load(tvr_rsrc, m_tile, fx.Int32).ir_value()))
    # FlyDSL carries pre-existing values across a runtime scf.if.
    acc = [mfma.zero_value for _ in range_constexpr(M_REPEAT * NUM_ACC_N)]
    if valid_rows > fx.Int32(M_REPEAT * 8):
        acc = do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r,
            b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds,
            a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES,
            pipe_weights, mfma_amajor, async_a_copy, trb_rsrc,
            unroll_a_pingpong, split_a_lds, fp8_b_waitcnt, scalar_tile_row_base,
            tvr_rsrc=tvr_rsrc, tvr_row_bytes=tvr_row_bytes,
            wait_payload=wait_payload, work=work, return_acc_only=True)
    else:
        acc = do_tile(m_tile, n_tile_base, expert, sched, a_gather_short, a_s2r,
            b_loader, b_scale, a_scale, mfma_short, epi, a_buf, a_scale_lds,
            a_lds_i32, K_ITERS, M_REPEAT // 2, NUM_ACC_N, A_K_STEP_BYTES,
            pipe_weights, mfma_amajor, async_a_copy, trb_rsrc,
            unroll_a_pingpong, split_a_lds, fp8_b_waitcnt, scalar_tile_row_base,
            tvr_rsrc=tvr_rsrc, tvr_row_bytes=tvr_row_bytes,
            wait_payload=wait_payload, work=work, return_acc_only=True)
    wait_lds_barrier()
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


# fmt: off
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    tvr_rsrc, sx_addr=None,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource, band_m=1, swiglu_limit=0.0, swiglu_alpha=1.702, swiglu_beta=1.0, packed_a_scale=False, unroll_a_pingpong=False, split_a_lds=False, fp8_b_waitcnt=False, prefetch_a_operand=False, scalar_tile_row_base=False, bf16_intermediate=None, a_tile_bytes=None):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner."""
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_gather = ATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy, tile_bytes=a_tile_bytes)
    # fmt: on
    a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
    b_loader = BWeightLoader(
        w_rsrc=w_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )
    b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
    a_scale = AScaleLoader(
        packed_lds=packed_a_scale,
        scale_rsrc=sx_rsrc,
        scale_addr=sx_addr,
        m_repeat=m_repeat,
        model_dim=model_dim,
        sort_block_m=sort_block_m,
        total_threads=total_threads,
    )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n, prefetch_a_operand=prefetch_a_operand)
    # Fixed M32/M64 entry paths for the qualified SBM64/N256 workload.
    assert (sort_block_m, tile_n, m_repeat, num_acc_n) == (64, 256, 4, 2)
    assert tvr_rsrc is not None and pipe_weights and mfma_amajor and async_a_copy
    assert not unroll_a_pingpong and not prefetch_a_operand
    mfma_short = MfmaScaleGU(m_repeat=2, num_acc_n=num_acc_n)
    # Retain the physical SBM64 ping/pong offsets and shared M64 epilogue.
    a_gather_short = ATileLoader(row_bytes=model_dim, sort_block_m=32,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy, tile_bytes=a_tile_bytes)
    # fmt: off
    epi = SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
        inter_dim=inter_dim, m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
        num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta, always_valid=True,
        out_tensor=out_tensor if use_tile_resource else None, bf16_intermediate=bf16_intermediate,
        tvr_rsrc=tvr_rsrc)
    # fmt: on

    # Work-index order. band_m=1 is upstream: n_tile is the FAST axis, so the
    # concurrently resident blocks share one m_tile -- the A slab -- and between
    # them sweep the expert's entire w1 slab. Weight traffic is
    # pairs*2*I*H/sort_block_m against activation's pairs*2*I*H/tile_n, i.e. 2x
    # larger at sort_block_m=128 / tile_n=256, so that order gives the cache the
    # SMALLER operand and re-reads w1 once per m_tile (M_r/1024 times at
    # sort_block_m=128 -- an amplification that grows linearly with batch).
    #
    # band_m=G makes m_tile the fast axis WITHIN a band of G consecutive
    # m_tiles, so resident blocks instead share a tile_n x model_dim B slab and
    # stream A. It stays banded rather than fully transposed for two reasons:
    # m_tiles map to experts only through a runtime lookup (a global transpose
    # would pair each n_tile with m_tiles from different experts, and B is
    # per-expert, so there would be no reuse at all), and stage1's per-tile
    # payload wait is m_tile-indexed -- marching the whole grid across every
    # m_tile before advancing n_tile would force the dispatch to complete before
    # GEMM1 could start, serialising the overlap the fusion exists for. Keep G
    # at the dispatch's own chunk granularity (payload_chunk_rows/sort_block_m)
    # so the wait granularity and the reuse granularity are the same object.
    #
    # Reordering is bit-exact: every tile writes disjoint output through
    # epi.store and nothing accumulates across tiles.
    band = const_expr(int(band_m))
    band_work = const_expr(band * n_tiles)

    def _decode(flat):
        if const_expr(band == 1):
            m_tile = flat // fx.Int32(n_tiles)
            n_tile = flat - m_tile * fx.Int32(n_tiles)
            return m_tile, n_tile
        band_i = flat // fx.Int32(band_work)
        rem = flat - band_i * fx.Int32(band_work)
        n_tile = rem // fx.Int32(band)
        m_tile = band_i * fx.Int32(band) + (rem - n_tile * fx.Int32(band))
        return m_tile, n_tile

    def m_tile_of_flat(flat):
        m_tile, _n = _decode(flat)
        return m_tile

    def expert_of_flat(flat):
        m_tile, _n = _decode(flat)
        return sched.expert_of(m_tile)

    def do_scheduled_tile(flat, wait_payload=None):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        do_tile_short_or_full(m_tile, n_tile_base, expert, sched, a_gather,
            a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
            a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
            trb_rsrc, unroll_a_pingpong, split_a_lds, fp8_b_waitcnt, scalar_tile_row_base,
            tvr_rsrc=tvr_rsrc, tvr_row_bytes=model_dim,
            wait_payload=wait_payload, work=flat, mfma_short=mfma_short, a_gather_short=a_gather_short)
        # fmt: on

    return expert_of_flat, m_tile_of_flat, do_scheduled_tile


# fmt: off
@functools.cache
def compile_gemm1(
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, async_a_copy: bool = False,
    use_tile_resource: bool = True, waves_per_eu_hint: int = 2, b_cache_modifier: int = 0,
    swiglu_limit: float = 7.0, swiglu_alpha: float = 1.702, swiglu_beta: float = 1.0,
):
    # fmt: on
    """Compile standalone group GEMM1 from the fused Stage1 compute body."""
    num_waves = int(num_waves)
    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_rsrc = _make_buffer(
                out, fx.Int16, max_size=False, num_records_bytes=num_valid * fx.Int32(inter_dim)
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = _make_buffer(
            out_scale,
            fx.Int8,
            max_size=False,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + fx.Int32(8192),
        )
        wave_id = fx.thread_idx.x // 64

        _, _, run_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc,
            sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc, trb_rsrc=trb_rsrc,
            # The standalone group GEMM1 has no per-tile row-count table: its
            # caller passes a dense tile_row_base and nothing else. No bound here.
            tvr_rsrc=None,
            expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, async_a_copy=async_a_copy,
            use_tile_resource=use_tile_resource, swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        )
        total_work = (num_valid // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            run_tile(flat)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def gemm1_kernel(
    out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, stream, *,
    model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 4,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True, waves_per_eu_hint: int = 2,
    num_cu: int = 256, b_cache_modifier: int = 0, swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702, swiglu_beta: float = 1.0,
):
    # fmt: on
    """Run standalone MegaMoEM3 group GEMM1 and return ``(out, out_scale)``."""
    num_valid = int(num_valid)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0:
        return out, out_scale
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = min(total_work, int(num_cu) * int(grid_mult))
    launch = compile_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, expert_offset=expert_offset,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_cache_modifier=b_cache_modifier,
        swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
    )
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8), tile_row_base, expert_ids, out_scale,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )
    return out, out_scale
