# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_group,
    emit_dispatch_payload,
    emit_dispatch_plan,
)
from .gemm1 import _LdsF32View, _SplitABuffer, build_fused_gemm1
from .gemm_util import _buffer_load, _buffer_store, _make_buffer, _make_buffer_from_addr
from .mega_moe_config import (
    SHARED_FUSED_MTPR_LARGE,
    SHARED_FUSED_MTPR_SMALL,
    SHARED_FUSED_S1_GEOMETRY,
)

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32


def ceildiv(a, b):
    return (a + b - 1) // b


def _use_direct_fixed_slot(
    enabled, npes, experts_per_rank, max_tokens_per_rank, cap, tile_m
):
    if not enabled or tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return npes == 8 and experts_per_rank == 48 and cap == required_cap


def _validate_dispatch_capacity(
    batch_size,
    npes,
    experts_per_rank,
    topk,
    tile_m,
    row_bytes,
    output_row_bytes,
    use_tile_resource,
):
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    if not use_tile_resource and max_rows * row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError(
            "MegaMoE v2 stage1 payload exceeds the 32-bit buffer-resource ABI"
        )
    if (
        not use_tile_resource
        and max_rows * output_row_bytes >= _BUFFER_OFFSET_ABI_BYTES
    ):
        raise ValueError(
            "MegaMoE v2 stage1 output exceeds the 32-bit buffer-resource ABI"
        )


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    work_shards: int | None = None, external_grouping: bool | None = None,
    external_counting: bool | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    payload_tile_publish_early: bool = False,
    skip_launch_barrier: bool = False,
    padding_uniform_srcmap: bool = False,
    count_uniform_matrix: bool = False,
    row_base_prefetch: bool = False,
    prefetch_b_before_a: bool = False,
    joint_work_flags: bool = False,
    preplanned: bool = False,
    band_m: int = 1, swiglu_limit: float = 7.0, swiglu_alpha: float = 1.702, swiglu_beta: float = 1.0,
    packed_a_scale: bool = False,
    unroll_a_pingpong: bool = False,
    split_a_lds: bool = False,
    fp8_b_waitcnt: bool = False,
    scalar_tile_row_base: bool = False,
    prefetch_a_operand: bool = False,
    xcd_schedule: bool = False, xcd_home: bool = True, shared_xcd_home: bool = True,
    shared_row_stride: int | None = None,
    schedule_audit: bool = False, reset_stage2_queue: bool = False, shared_l13: bool = False, shared_xcd: bool = False,
    shared_packed_heads: bool = False,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage1 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    # GEMM1's A step is one tile_k of FP8, so tile_k is fixed for every shape and
    # schedule. Checked once here; the geometry checks below no longer repeat it.
    assert tile_k == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert grid_mult in GRID_MULT_VALUES, "grid_mult out of range"
    grid_epoch_slot = GRID_MULT_VALUES.index(grid_mult)
    dispatch_blocks = int(num_dispatch_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    if payload_chunk_rows:
        assert not fixed_slot_dispatch and payload_chunk_rows % sort_block_m == 0
    assert not payload_tile_publish_early or payload_tile_ready
    assert not payload_tile_ready or payload_chunk_rows > 0
    assert not skip_launch_barrier or (not fixed_slot_dispatch and payload_chunk_rows == 0), (
        "Skipping launch arrival requires compact COUNT_DONE and all-destination producers"
    )
    assert not prefetch_b_before_a or (pipe_weights and async_a_copy and not fixed_slot_dispatch), (
        "B prefetch before payload readiness requires compact dispatch and the asynchronous weight pipeline"
    )
    BAND_M = int(band_m)
    assert BAND_M >= 1, "band_m is a count of m_tiles per reuse band"
    # The tile-ready schedule uses GEMM's band decoder. Small XCD bands
    # instead map tickets to canonical m*N+n indices and retain expert waits.
    assert BAND_M == 1 or payload_tile_ready or xcd_schedule, "finite small-XCD bands retain per-expert waits"
    assert not shared_xcd or shared_l13
    if shared_l13:
        assert preplanned and xcd_schedule and prefetch_b_before_a and not schedule_audit
        # Both regimes and their token sets live in mega_moe_config so Stage2 and
        # MegaMoEM3 cannot drift from them.
        if int(fuse_mtpr) in SHARED_FUSED_MTPR_SMALL:
            assert shared_xcd and not payload_tile_ready and not use_tile_resource
            want = SHARED_FUSED_S1_GEOMETRY["small"]
            got = dict(tile_n=tile_n, num_waves=num_waves, band_m=band_m)
            assert got == want, f"fused shared L13 small regime wants {want}, got {got}"
            # A 32-row tile is measured at EP8 b16, where each expert holds only a
            # handful of rows and a 64-row tile is mostly padding. Every
            # sort_block_m user below scales with it, including the shared queue,
            # whose tile count is mtpr // sort_block_m -- so the row table in
            # MegaMoEM3 must stride by the same value, which the assert above now
            # checks directly instead of assuming a 64-row stride.
            assert sort_block_m in (32, 64)
            # The real invariant is that MegaMoEM3's shared row table and the tile
            # count below index the same blocks. Both now take a ceiling, so a batch
            # the sort block does not divide is served by a short last tile rather
            # than rejected. The stride still has to match: the table's stride is
            # chosen by MegaMoEM3, and (mtpr 128, sort_block_m 32) would give four
            # tiles here while the table holds two 64-row entries, which reads past
            # it. So the stride comes in and is checked against what this kernel
            # counts with.
            assert shared_row_stride is None or int(shared_row_stride) == sort_block_m, (
                f'shared row table strides by {shared_row_stride}, Stage1 divides by {sort_block_m}')
            # Eight rows is the granularity the short-tail bounds are written for:
            # Stage2's shared panel bound and this kernel's row table both address
            # whole tokens, and the combine grid search needs mtpr % 8 == 0 to find
            # an exact warp partition per token.
            assert int(fuse_mtpr) % 8 == 0, (
                f'{fuse_mtpr} local tokens must be a multiple of eight')
        else:
            assert int(fuse_mtpr) in SHARED_FUSED_MTPR_LARGE, (
                f"fused shared L13 has no validated regime for {fuse_mtpr} local tokens")
            assert payload_tile_ready and use_tile_resource
            want = SHARED_FUSED_S1_GEOMETRY["large"]
            got = dict(sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, band_m=band_m)
            assert got == want, f"fused shared L13 large regime wants {want}, got {got}"
    planner_blocks = 1
    # Keep the fused grid on an exact CU multiple instead of appending control/producer CTAs as a tail.
    grid_x = num_cu * grid_mult - planner_blocks - dispatch_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = planner_blocks + dispatch_blocks + grid_x
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2
    A_K_STEP_BYTES = tile_k
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = 4 if work_shards is None and int(fuse_mtpr) >= 8192 else 8
    if work_shards is not None:
        WORK_SHARDS = int(work_shards)
    assert WORK_SHARDS in (1, 2, 4, 8)
    # Compile-time scheduling choice; both paths share dispatch and GEMM.
    small_xcd = xcd_schedule and not payload_tile_ready
    if xcd_schedule:
        assert WORK_SHARDS == 8
        if small_xcd:
            assert (tile_n, N_TILES) == (512, 12)
            assert sort_block_m in (32, 64)
            # Deliberately NOT the fused-shared token set: this is the unfused
            # small-XCD path, which EP4 also drives at 512 and 1024 with a 64-row
            # sort block, and which no one has validated at 2048 and above.
            # 96 is here because the schedule itself imposes nothing on the token
            # count -- it maps tickets to canonical m*N_TILES+n indices and the m
            # tile count comes from num_valid at run time -- so this list records
            # which sizes have been measured, not what the decoder can do.
            assert int(fuse_mtpr) % 8 == 0 and int(fuse_mtpr) <= 1024
            assert not fixed_slot_dispatch
        else:
            assert N_TILES % 8 == 0 and payload_tile_ready and BAND_M > 1
    if schedule_audit:
        assert small_xcd or (sort_block_m == 128 and tile_n == 256)

    shared_packed = bool(shared_packed_heads)
    if shared_packed:
        # Independent shared/routed counters in the low/high 16 bits of each
        # existing i32 head, reset by the preceding preplan kernel every forward.
        # Keep the legacy all-shared-queues then all-routed-queues claim order.
        if not (preplanned and shared_l13 and shared_xcd and small_xcd
                and not joint_work_flags and int(fuse_npes) == 8
                and (model_dim, inter_dim) == (6144, 3072)
                and int(fuse_mtpr) in SHARED_FUSED_MTPR_SMALL
                and xcd_home and shared_xcd_home
                and num_cu * grid_mult == 256):
            raise ValueError(
                f"shared_packed_heads requires the preplanned fused-shared small EP8 "
                f"geometry, got preplanned={preplanned} shared_l13={shared_l13} "
                f"shared_xcd={shared_xcd} small_xcd={small_xcd} "
                f"joint_work_flags={joint_work_flags} npes={fuse_npes} "
                f"dims=({model_dim},{inter_dim}) mtpr={fuse_mtpr} "
                f"homes=({xcd_home},{shared_xcd_home}) grid={num_cu * grid_mult}")
        max_shared_claims = 2 * ceildiv(int(fuse_mtpr), sort_block_m) + num_cu * grid_mult
        max_routed_m = ceildiv(int(fuse_npes) * int(fuse_mtpr) * int(fuse_topk), sort_block_m) + experts_per_rank
        max_routed_claims = ceildiv(max_routed_m, BAND_M) * (2 * BAND_M) + num_cu * grid_mult
        # Include every CTA's failed exit. No low-field carry, high-field wrap,
        # or sign bit: concurrent adds to either field cannot change the other.
        assert max_shared_claims < (1 << 16)
        assert max_routed_claims < (1 << 15)

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    if prefetch_a_operand:
        assert split_a_lds and unroll_a_pingpong and mfma_amajor
    if fp8_b_waitcnt:
        assert split_a_lds and pipe_weights, "FP8 wait-count experiment requires pipelined B"
    if split_a_lds:
        assert unroll_a_pingpong and async_a_copy and mfma_amajor
        # Independent A ping/pong and split CShuffle addressing scale with M.
        # Keep the audited N/K/wave geometry while testing smaller row tiles.
        assert sort_block_m in (32, 64, 128)
        assert (tile_n, num_waves) == (256, 8)
        assert K_ITERS % 2 == 0 and lds_pool_bytes == 2 * a_lds_size

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    if external_grouping is None:
        external_grouping = fz_mtpr >= 2048 and fz_npes == 8 and fz_epr == 48
    if external_counting is None:
        external_counting = external_grouping and fz_mtpr >= 8192
    assert not external_counting or external_grouping
    assert not preplanned or (not fixed_slot_dispatch and grid_mult == 1)
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    # GEMM1 bounds its A read with the per-tile row count that emit_dispatch_plan
    # records. emit_direct_fixed_slot_finalize writes tile_row_base and nothing else,
    # so that layout cannot supply the bound; say so rather than silently reading a
    # stale one or silently giving up the saving.
    assert not direct_fixed_slot, (
        'the direct fixed-slot layout does not record per-tile row counts, which the '
        'bounded A read requires')
    fz_total_experts = fz_npes * fz_epr
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if direct_fixed_slot and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim, use_tile_resource
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @fx.struct
    class SplitSharedStorage:
        ping: fx.Array[fx.Int8, a_lds_size, 16]
        pong: fx.Array[fx.Int8, a_lds_size, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    swiglu_suffix += (
        f"_a{str(float(swiglu_alpha)).replace('.', 'p')}"
        f"b{str(float(swiglu_beta)).replace('.', 'p')}"
    )
    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_ptr{int(payload_tile_ready)}_bm{BAND_M}"
        f"{swiglu_suffix}"
        + ("_ptp1" if payload_tile_publish_early else "")
        + ("_pas1" if packed_a_scale else "")
        + ("_upp1" if unroll_a_pingpong else "")
        + ("_sal1" if split_a_lds else "")
        + ("_bwc1" if fp8_b_waitcnt else "")
        + ("_aop1" if prefetch_a_operand else "")
        + ("_trbu1" if scalar_tile_row_base else "")
        + ("_xq1" if xcd_schedule else "")
        + (("" if xcd_home else "_hr0") + ("" if shared_xcd_home else "_hs0") if xcd_schedule else "")
        + ("_qa1" if schedule_audit else "")
        + ("_sxcd1" if small_xcd else "")
        + ("_slb1" if skip_launch_barrier else "")
        + ("_pus1" if padding_uniform_srcmap else "")
        + ("_cum1" if count_uniform_matrix else "")
        + ("_rbp1" if row_base_prefetch else "")
        + ("_bpa1" if prefetch_b_before_a else "")
        + ("_jwf1" if joint_work_flags else "")
        + ("_qplan1" if preplanned else "")
        # The A read's per-tile bound changes the code object, so it belongs in the
        # name: without it a broken plumbing path compiles to the identical kernel and
        # passes every gate, which is exactly how this was missed once.
        + "_atb1"
        # Same for the A-scale read's bound, and for the store-side half: dropping
        # padding-row writes also changes the code object.
        + "_asb1"
        + "_svb1"
        + "_scratchfix1"
        + "_tb1_salate1_brni1"
        + ("_s2qr1" if reset_stage2_queue else "")
        + ("_sharedl13" if shared_l13 else "")
        + ("_shxcd1" if shared_xcd else "")
        + ("_shu1" if shared_l13 else "")
        + ("_shna2" if shared_l13 else "")
        + ("_shsmall1" if shared_l13 and small_xcd else "")
        + ("_shpacked1" if shared_packed else "")
        # A batch larger than its sort block that the block does not divide counts
        # shared tiles with a ceiling and carries a short last tile. That changes
        # the code object, so it belongs in the name: the A-read bounds above were
        # once plumbed into a kernel that compiled identically and passed every
        # gate. Batches below one sort block already took the ceiling, and keep
        # the name they have.
        + ("_sct1" if shared_l13 and int(fuse_mtpr) > sort_block_m
           and int(fuse_mtpr) % sort_block_m != 0 else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_stage2_work_head: fx.Int64, addr_shared_l13: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SplitSharedStorage if split_a_lds else SharedStorage).peek()
        if const_expr(split_a_lds):
            a_buf = _SplitABuffer(lds.ping, lds.pong)
            c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.ping.ptr),
                                 fx.recast_iter(fx.Float32, lds.pong.ptr), a_lds_i32)
        else:
            a_buf = lds.pool
            c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        a_scale_lds = lds.A_scale
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_pair_order_ready = _disp_ptr(DispatchSlot.PAIR_ORDER_READY)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        a_payload_chunks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        # All waves must consume the ticket before A LDS is reused.
        fx.barrier()
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)

        if const_expr(not preplanned):
            if compact_owner:
                next_parity_lane = fx.Int32(0)
                launch_epoch_lane = fx.Int32(0)
                if tid == fx.Int32(0):
                    old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                    next_parity_lane = old_parity ^ fx.Int32(1)
                    previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
                    next_expected = previous_expected + fx.Int32(fz_npes)
                    _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                    launch_epoch_lane = (
                        (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
                    )
                next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
                launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
                if const_expr(payload_tile_ready):
                    if tid == fx.Int32(0):
                        comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                        comm_ops.fence_system_release()
                    fx.barrier()
                # Compact COUNT_DONE already waits for every rank this generation.
                # Before a previous S1 can finish, its all-destination producers
                # have seen every peer's PLAN_READY: all old count reads are done.
                # Thus next-generation count writes may precede peer S1 arrival.
                # Keep the local reset/gate below; it protects local queue state.
                if const_expr(not skip_launch_barrier):
                    if tid < fx.Int32(fz_npes):
                        peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                        comm_ops.fence_system_release()
                        launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
                        remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
                        comm_ops.store_i32_system(remote_launch_ready, fx.Int32(fz_rank), launch_epoch)
                        mori_shmem.int32_wait_until_greater_than(
                            a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
                        )
                        comm_ops.fence_system_acquire()
                if tid == fx.Int32(0):
                    if const_expr(reset_stage2_queue):
                        # Only the eight queue heads are live; each is 256 B apart.
                        # The existing wait/release below completes these stores.
                        # S2 runs after this entire kernel on the same stream.
                        s2_head_rsrc = _make_buffer_from_addr(addr_stage2_work_head, fx.Int32)
                        for queue in range_constexpr(8):
                            _buffer_store(s2_head_rsrc, fx.Int32(queue * 64), fx.Int32(0), fx.Int32)
                    work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                    for shard in range_constexpr(8):
                        _buffer_store(work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32)
                    _buffer_store(_make_buffer_from_addr(a_work_tail, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
                    if const_expr(external_grouping or direct_fixed_slot):
                        group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                        for destination in range_constexpr(fz_npes if direct_fixed_slot else 1):
                            _buffer_store(group_done_rsrc, fx.Int32(destination), fx.Int32(0), fx.Int32)
                fx.barrier()
                if tid == fx.Int32(0):
                    fx.rocdl.s_waitcnt(0)
                    comm_ops.fence_agent_release()
                    _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                    fx.rocdl.s_waitcnt(0)
                    comm_ops.fence_agent_release()
                    comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
                fx.rocdl.s_waitcnt(0)
                fx.barrier()
            else:
                if tid == fx.Int32(0):
                    mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
                    comm_ops.fence_agent_acquire()
                fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)

        if const_expr(not preplanned):
            if compact_owner:  # noqa: SIM102 - keep the device and compile-time branches separate.
                if const_expr(not direct_fixed_slot):
                    emit_dispatch_plan(
                        num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                        fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=payload_parity,
                        expected=payload_expected, external_grouping=external_grouping,
                        external_counting=external_counting,
                        padding_uniform_srcmap=padding_uniform_srcmap,
                        count_uniform_matrix=count_uniform_matrix,
                        row_base_prefetch=row_base_prefetch,
                        dispatch_blocks=dispatch_blocks, payload_chunk_rows=payload_chunk_rows,
                        payload_tile_ready=payload_tile_ready,
                    )

        if compact_producer:
            if const_expr(direct_fixed_slot):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                )
            else:
                if const_expr(external_grouping and not preplanned):
                    emit_dispatch_group(
                        num_waves=NUM_WAVES, fz_k=fz_k, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        external_counting=external_counting, adaptive_grouping=payload_tile_ready,
                    )
                else:
                    if const_expr(not preplanned):
                        if tid == fx.Int32(0):
                            mori_shmem.int32_wait_until_equals(
                                a_pair_order_ready + fx.Int64(payload_parity) * fx.Int64(4), payload_expected)
                            comm_ops.fence_agent_acquire()
                        fx.barrier()
                producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                chunks_per_destination = fx.Int32(1)
                if const_expr(payload_chunk_rows > 0):
                    chunks_per_destination = fx.Int32(
                        (fz_mtpr + payload_chunk_rows - 1) // payload_chunk_rows
                    )
                payload_active = fx.Int32(0) == fx.Int32(0)
                if const_expr(payload_tile_ready and dispatch_blocks > 32):
                    producer_destination = producer_slot % fx.Int32(fz_npes)
                    producer_round = producer_slot // fx.Int32(fz_npes)
                    producers_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_blocks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    chunks_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_chunks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    payload_active = producer_round < producers_per_destination
                if payload_active:
                    emit_dispatch_payload(
                        num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                        fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                        fz_safe_end_i32=fz_safe_end_i32, fz_scale_n_i32=fz_scale_n_i32,
                        fz_enable_scales=fz_enable_scales, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                        addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        producers_per_destination=producers_per_destination, payload_chunk_rows=payload_chunk_rows,
                        chunks_per_destination=chunks_per_destination, payload_tile_ready=payload_tile_ready,
                        payload_tile_publish_early=payload_tile_publish_early,
                    )
        if const_expr(direct_fixed_slot):
            if compact_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, n_tiles=N_TILES, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                )
        else:
            payload_table = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_PAYLOAD_READY)), fx.Int64)
            addr_payload_ready = _buffer_load(
                _make_buffer_from_addr(payload_table, fx.Int64), fx.Int32(fz_rank), fx.Int64
            )
            addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
            addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim)
            out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        def _build_runner(x_tensor, w_buf, sw_buf, sx_buf, out_buf, os_buf, trb_buf, expert_buf, out_tensor, shared_flag=None, tvr_buf=None, sx_addr=None):
            return build_fused_gemm1(
                x_tensor=x_tensor, w_rsrc=w_buf,
                sw_rsrc=sw_buf, sx_rsrc=sx_buf, sx_addr=sx_addr, out_rsrc=out_buf, os_rsrc=os_buf,
                trb_rsrc=trb_buf, tvr_rsrc=tvr_buf, expert_rsrc=expert_buf, out_tensor=out_tensor,
                a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
                model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
                tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
                m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
                total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
                n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
                swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
                async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
                band_m=1 if small_xcd else BAND_M,
                packed_a_scale=packed_a_scale,
                unroll_a_pingpong=unroll_a_pingpong,
                split_a_lds=split_a_lds,
                fp8_b_waitcnt=fp8_b_waitcnt,
                prefetch_a_operand=prefetch_a_operand,
                scalar_tile_row_base=scalar_tile_row_base, bf16_intermediate=shared_flag,
                a_tile_bytes=(shared_flag.select(fx.Int32(fuse_mtpr * model_dim), fx.Int32(sort_block_m * model_dim))
                    if shared_flag is not None and fuse_mtpr < sort_block_m else None),
                swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
            )

        # One int32 per sort block: the rows that carry a token. GEMM1 bounds its A
        # buffer with it so the padding rows cost no bandwidth.
        a_tvr = _disp_ptr(DispatchSlot.TILE_VALID_ROWS)
        tvr_rsrc = _make_buffer_from_addr(a_tvr, fx.Int32)
        if const_expr(not shared_l13):
            expert_of_flat, _m_tile_of_flat, _do_scheduled_tile = _build_runner(
                x, w_rsrc, sw_rsrc, sx_rsrc, out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out,
                tvr_buf=tvr_rsrc, sx_addr=fx.Int64(fx.ptrtoint(fx.get_iter(scale_x))))
        else:
            # A batch the sort block divides keeps the plain division it has always
            # compiled; anything else -- a batch smaller than one block, or one with
            # a short last tile -- takes the ceiling, which is what MegaMoEM3's row
            # table (mega_moe_m3.py) and shared_tasks_per_queue below already use.
            # Counting one tile short here does not lose work quietly: is_shared is
            # `uniform_work < shared_work`, so the missing tile's tasks would be run
            # as routed ones against the routed tables.
            if const_expr(int(fuse_mtpr) % sort_block_m != 0):
                shared_m_tiles = (i32_cur_tok + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
            else:
                shared_m_tiles = i32_cur_tok // fx.Int32(sort_block_m)
            shared_work = shared_m_tiles * fx.Int32(N_TILES)

            def _m_tile_of_flat(flat):
                if const_expr(small_xcd):
                    return flat // fx.Int32(N_TILES)
                return (flat // fx.Int32(BAND_M * N_TILES)) * fx.Int32(BAND_M) + flat % fx.Int32(BAND_M)

            def expert_of_flat(flat):
                return _buffer_load(expert_rsrc, _m_tile_of_flat(flat), fx.Int32) - fx.Int32(fz_rank * fz_epr)

            shared_table = _make_buffer_from_addr(addr_shared_l13, fx.Int64)
            shared_count_addr = _buffer_load(shared_table, fx.Int32(6), fx.Int64)
            # Every CTA exhausts every shared queue once before routed work.
            # Each head therefore advances by its valid tasks + launch_grid_x
            # per invocation, even when the physical XCD placement is uneven.
            # This gives graph-safe epochs without a reset kernel or barrier.
            shared_tasks_per_queue = ((fuse_mtpr + sort_block_m - 1) // sort_block_m) * N_TILES // (8 if shared_xcd else 1)
            shared_count_period = fx.Int64(shared_tasks_per_queue + launch_grid_x)

            def _uniform_addr(addr):
                # The pointer table and selected task are identical across a
                # wave. Preserve that fact when building buffer resources so
                # LLVM does not emit a descriptor waterfall for every K load.
                return fx.Int64(fx.rocdl.readfirstlane(T.i64, addr.ir_value()))

            def _selected_addr(flag, slot, routed_tensor):
                return _uniform_addr(flag.select(
                    _buffer_load(shared_table, fx.Int32(slot), fx.Int64),
                    fx.Int64(fx.ptrtoint(fx.get_iter(routed_tensor)))))

            def _byte_tensor(addr):
                ptr = fx.inttoptr(fx.PointerType.get(fx.Float8E4M3FN.ir_type, fx.AddressSpace.Global, 16), addr)
                return fx.Tensor(fx.make_view(ptr, fx.make_layout(1, 1)))

        if const_expr(not preplanned):
            if tid == fx.Int32(0):
                local_plan_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64)
                ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                mori_shmem.int32_wait_until_equals(
                    local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected)
                comm_ops.fence_agent_acquire()
            fx.barrier()

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        routed_m_tiles = num_m_tiles
        if const_expr(shared_l13):
            num_m_tiles = num_m_tiles + shared_m_tiles
        if const_expr(BAND_M == 1 or small_xcd):
            total_work = num_m_tiles * fx.Int32(N_TILES)
        else:
            # Round the index space up to whole bands so _decode stays a
            # bijection on it. The tail tickets decode to m_tile >= num_m_tiles
            # and are skipped in the work loop below.
            total_work = (
                ceildiv(num_m_tiles, fx.Int32(BAND_M)) * fx.Int32(BAND_M * N_TILES)
            )

        def _wait_routed_payload(flat):
            if const_expr(payload_tile_ready):
                tile_index = _m_tile_of_flat(flat)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                mori_shmem.int32_wait_until_equals(
                    addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4), expected_tiles
                )
            else:
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

        def _wait_tile_payload(flat):
            if const_expr(shared_l13):
                if flat >= shared_work:
                    _wait_routed_payload(flat - shared_work)
            else:
                _wait_routed_payload(flat)

        # Control CTAs join the work pool after dispatch.
        def _wait_payload_after_b(work):
            if tid == fx.Int32(0):
                _wait_tile_payload(work)
            fx.barrier()
            comm_ops.fence_system_acquire()

        consumer_active = fx.Int32(1) == fx.Int32(1)
        # Visit every shared queue before claiming routed work. This preserves
        # shared ticket issuance priority, without a GEMM completion barrier.
        shared_active = fx.Boolean(True)
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(
            work_scratch, fx.make_layout(2 if joint_work_flags else 1, 1)
        )
        work_shard = ticket & fx.Int32(WORK_SHARDS - 1)
        if const_expr(xcd_schedule or schedule_audit):
            physical_xcd = fx.Int32(llvm.inline_asm(
                T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
                "=s", has_side_effects=True))
        if const_expr(xcd_schedule):
            # XCD is a locality hint only. Every CTA eventually visits all
            # queues, so sparse placement/migration cannot leave work undone.
            # The seed picks which queue a CTA starts on; routed and shared can
            # be seeded independently without changing the rotation itself.
            home_queue = (physical_xcd if xcd_home else ticket) & fx.Int32(7)
            if const_expr(shared_xcd_home != xcd_home):
                shared_home_queue = (physical_xcd if shared_xcd_home else ticket) & fx.Int32(7)
        queue_attempt = fx.Int32(0)
        flags = fx.Int32(0)
        def _decode_xcd(local_work, work_shard):
            local_band = local_work // fx.Int32(BAND_M * (N_TILES // 8))
            local_rem = local_work % fx.Int32(BAND_M * (N_TILES // 8))
            n_tile = (local_rem // fx.Int32(BAND_M)) * fx.Int32(8) + work_shard
            work = local_band * fx.Int32(BAND_M * N_TILES) + n_tile * fx.Int32(BAND_M) + local_rem % fx.Int32(BAND_M)
            return work

        def _claim_routed(work_shard):
            local_work = fx.Int32(
                comm_ops.atomic_add_agent(
                    a_work_head + fx.Int64(work_shard) * fx.Int64(64),
                    fx.Int32((1 << 16) if shared_packed else 1)
                )
            )
            if const_expr(shared_packed):
                local_work = local_work >> fx.Int32(16)
            if const_expr(small_xcd):
                # Keep home=(expert*12+n)%8 unchanged, but alternate
                # the two local N slots after each finite M band.
                # Return canonical m*N+n indices to the unchanged GEMM.
                m_index = (local_work // fx.Int32(2 * BAND_M)) * fx.Int32(BAND_M) + local_work % fx.Int32(BAND_M)
                safe_m = (m_index < routed_m_tiles).select(m_index, fx.Int32(0))
                e_index = expert_of_flat(safe_m * fx.Int32(N_TILES))
                n_first = (work_shard + e_index * fx.Int32(4)) & fx.Int32(7)
                n_index = n_first + ((local_work // fx.Int32(BAND_M)) % fx.Int32(2)) * fx.Int32(8)
                mapped_work = m_index * fx.Int32(N_TILES) + n_index
                routed_work_limit = routed_m_tiles * fx.Int32(N_TILES)
                valid_or_skip = ((n_index < fx.Int32(N_TILES)) & (m_index < routed_m_tiles)).select(mapped_work, routed_work_limit)
                ticket_count = ceildiv(routed_m_tiles, fx.Int32(BAND_M)) * fx.Int32(2 * BAND_M)
                work = (local_work < ticket_count).select(valid_or_skip, routed_work_limit + fx.Int32(1))
            elif const_expr(xcd_schedule):
                # Each XCD owns three N panels in the H6144/I3072 case.
                # Within a ready M band, adjacent claims reuse the B panel.
                work = _decode_xcd(local_work, work_shard)
            else:
                work = work_shard + local_work * fx.Int32(WORK_SHARDS)
            return work

        while consumer_active:
            if const_expr(xcd_schedule):
                work_shard = (home_queue + queue_attempt) & fx.Int32(7)
                if const_expr(shared_xcd_home != xcd_home):
                    shared_work_shard = (shared_home_queue + queue_attempt) & fx.Int32(7)
                else:
                    shared_work_shard = work_shard
            else:
                shared_work_shard = work_shard
            if tid == fx.Int32(0):
                work = fx.Int32(0)
                if const_expr(shared_l13):
                    shared_claim = shared_work
                    if shared_active:
                        if const_expr(shared_xcd):
                            if const_expr(small_xcd):
                                # Actual-batch M tiles, twelve N panels: queues 0..3
                                # own two panels, queues 4..7 own one. Every
                                # CTA exhausts each queue exactly once.
                                shared_small_m = (fuse_mtpr + sort_block_m - 1) // sort_block_m
                                shared_limit = (shared_work_shard < fx.Int32(4)).select(fx.Int32(2 * shared_small_m), fx.Int32(shared_small_m))
                                if const_expr(shared_packed):
                                    shared_local = fx.Int32(comm_ops.atomic_add_agent(
                                        a_work_head + fx.Int64(shared_work_shard) * fx.Int64(64), fx.Int32(1))) & fx.Int32(0xffff)
                                else:
                                    shared_period = fx.Int64(shared_limit) + fx.Int64(launch_grid_x)
                                    shared_local = fx.Int32(fx.Int64(comm_ops.atomic_add_agent(
                                        shared_count_addr + fx.Int64(shared_work_shard) * fx.Int64(64), fx.Int64(1))) % shared_period)
                                shared_m = shared_local % fx.Int32(shared_small_m)
                                shared_n = shared_work_shard + (shared_local // fx.Int32(shared_small_m)) * fx.Int32(8)
                                shared_claim = (shared_local < shared_limit).select(
                                    shared_m * fx.Int32(N_TILES) + shared_n, total_work + fx.Int32(1))
                            else:
                                shared_local = fx.Int32(fx.Int64(comm_ops.atomic_add_agent(
                                    shared_count_addr + fx.Int64(shared_work_shard) * fx.Int64(64), fx.Int64(1))) % shared_count_period)
                                shared_claim = (shared_local < fx.Int32(shared_tasks_per_queue)).select(
                                    _decode_xcd(shared_local, shared_work_shard), total_work + fx.Int32(1))
                        else:
                            shared_claim = fx.Int32(fx.Int64(comm_ops.atomic_add_agent(shared_count_addr, fx.Int64(1))) % shared_count_period)
                    if const_expr(shared_xcd):
                        if shared_active:
                            work = shared_claim
                        else:
                            work = _claim_routed(work_shard) + shared_work
                    else:
                        if shared_claim < shared_work:
                            work = shared_claim
                        else:
                            work = _claim_routed(work_shard) + shared_work
                else:
                    work = _claim_routed(work_shard)
                if const_expr(joint_work_flags):
                    # Publish the ticket and its skip/continue flags together.
                    # Keep both as i32 so no ticket bits are lost by packing.
                    publish_in_range = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                    if const_expr(small_xcd):
                        publish_keep_drawing = (work <= total_work).select(fx.Int32(1), fx.Int32(0))
                        publish_flags = publish_keep_drawing | (publish_in_range << fx.Int32(1))
                    elif const_expr(BAND_M == 1):
                        publish_flags = publish_in_range | (publish_in_range << fx.Int32(1))
                    else:
                        publish_valid = (_m_tile_of_flat(work) < num_m_tiles).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                        publish_flags = publish_in_range | ((publish_in_range * publish_valid) << fx.Int32(1))
                    if (publish_flags & fx.Int32(2)) != fx.Int32(0):
                        if const_expr(not direct_fixed_slot and not prefetch_b_before_a):
                            _wait_tile_payload(work)
                    fx.ptr_store(Vec.from_elements([work, publish_flags], fx.Int32), work_scratch)
                else:
                    fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            fx.barrier()
            if const_expr(joint_work_flags):
                work_and_flags = Vec(work_scratch_view.load())
                work = work_and_flags[0]
                flags = work_and_flags[1]
                # All waves must load both words before A LDS or the next
                # ticket reuses them, including skipped and exhausted tickets.
                fx.barrier()
            else:
                work = Vec(work_scratch_view.load())[0]
                # All waves must consume the ticket before the leader overwrites
                # this LDS word with flags. Otherwise a late wave can use 3 as work.
                fx.barrier()
                if tid == fx.Int32(0):
                    # bit0 = keep drawing tickets, bit1 = this ticket is a real tile.
                    # They differ only for a padded band's tail: those tickets must be
                    # SKIPPED rather than terminate the loop, because the ticket
                    # counter is monotonic -- retiring on one would drop every tile
                    # behind it.
                    in_range = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                    if const_expr(small_xcd):
                        keep_drawing = (work <= total_work).select(fx.Int32(1), fx.Int32(0))
                        flags = keep_drawing | (in_range << fx.Int32(1))
                    elif const_expr(BAND_M == 1):
                        flags = in_range | (in_range << fx.Int32(1))
                    else:
                        valid = (_m_tile_of_flat(work) < num_m_tiles).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                        flags = in_range | ((in_range * valid) << fx.Int32(1))
                    if (flags & fx.Int32(2)) != fx.Int32(0):  # noqa: SIM102 - keep the device and compile-time branches separate.
                        if const_expr(not direct_fixed_slot and not prefetch_b_before_a):
                            _wait_tile_payload(work)
                    fx.ptr_store(Vec.from_elements([flags], fx.Int32), work_scratch)
                fx.barrier()
                flags = Vec(work_scratch_view.load())[0]
                # The word aliases GEMM A LDS and the next iteration's ticket.
                # Finish every wave's flags load before either reuses that storage.
                fx.barrier()
            shared_active = shared_active if const_expr(shared_xcd) else (work < shared_work if const_expr(shared_l13) else fx.Boolean(False))
            if (flags & fx.Int32(2)) != fx.Int32(0):
                if const_expr(not direct_fixed_slot and not prefetch_b_before_a):
                    comm_ops.fence_system_acquire()
                if const_expr(schedule_audit):
                    if tid == fx.Int32(0):
                        audit_m = _m_tile_of_flat(work)
                        audit_n = work % fx.Int32(N_TILES) if const_expr(small_xcd) else (work % fx.Int32(BAND_M * N_TILES)) // fx.Int32(BAND_M)
                        audit_index = (audit_m * fx.Int32(N_TILES) + audit_n) * fx.Int32(3)
                        audit_addr = _disp_ptr(DispatchSlot.SCHEDULE_AUDIT)
                        comm_ops.atomic_add_agent(audit_addr + fx.Int64(audit_index) * fx.Int64(4), fx.Int32(1))
                        audit_rsrc = _make_buffer_from_addr(audit_addr, fx.Int32)
                        _buffer_store(audit_rsrc, audit_index + fx.Int32(1), physical_xcd + fx.Int32(1), fx.Int32)
                        _buffer_store(audit_rsrc, audit_index + fx.Int32(2), work_shard + fx.Int32(1), fx.Int32)
                if const_expr(shared_l13):
                    # Uniform descriptor selection feeds one GEMM/MFMA body for
                    # both task kinds. Shared and routed have separate queues
                    # but share the XCD-local band decoder.
                    # The scheduler broadcasts one task to the entire CTA via
                    # LDS, which the compiler otherwise treats as lane-varying.
                    uniform_work = fx.Int32(fx.rocdl.readfirstlane(T.i32, work.ir_value()))
                    is_shared = uniform_work < shared_work
                    selected_x = _byte_tensor(_uniform_addr(is_shared.select(addr_in_tok, fx.Int64(fx.ptrtoint(fx.get_iter(x))))))
                    selected_out = _byte_tensor(_selected_addr(is_shared, 2, out))
                    selected_w = _make_buffer_from_addr(_selected_addr(is_shared, 0, w), fx.Int32, 4)
                    selected_sw = _make_buffer_from_addr(_selected_addr(is_shared, 1, scale_w), fx.Int32)
                    selected_sx_addr = _uniform_addr(is_shared.select(addr_in_sc, fx.Int64(fx.ptrtoint(fx.get_iter(scale_x)))))
                    selected_sx = _make_buffer_from_addr(selected_sx_addr, fx.Int32, 4,
                        num_records_bytes=(is_shared.select(fx.Int32(fuse_mtpr * (model_dim // 32)), fx.Int32(0x7fffffff))
                            if fuse_mtpr < sort_block_m else None))
                    selected_trb = _make_buffer_from_addr(_selected_addr(is_shared, 4, sorted_token_ids), fx.Int32)
                    selected_expert = _make_buffer_from_addr(_selected_addr(is_shared, 5, expert_ids), fx.Int32)
                    selected_os = _make_buffer_from_addr(_selected_addr(is_shared, 3, out_scale), fx.Int8,
                        num_records_bytes=is_shared.select(i32_cur_tok * fx.Int32(scale_cols) + fx.Int32(8192), os_nbytes))
                    if const_expr(use_tile_resource):
                        selected_out_rsrc = None
                    else:
                        selected_out_rsrc = _make_buffer_from_addr(_selected_addr(is_shared, 2, out), fx.Int16,
                            num_records_bytes=is_shared.select(i32_cur_tok, tokens) * fx.Int32(inter_dim))
                    # Slot 7 of the shared pointer table mirrors the routed
                    # tile_valid_rows, so the same bound serves either task kind.
                    selected_tvr = _make_buffer_from_addr(_uniform_addr(is_shared.select(
                        _buffer_load(shared_table, fx.Int32(7), fx.Int64), a_tvr)), fx.Int32)
                    _, _, run_selected = _build_runner(selected_x, selected_w, selected_sw, selected_sx,
                        selected_out_rsrc, selected_os, selected_trb, selected_expert, selected_out, is_shared,
                        tvr_buf=selected_tvr, sx_addr=selected_sx_addr)
                    tile_work = is_shared.select(uniform_work, uniform_work - shared_work)

                    def _wait_selected(ignored):
                        if tid == fx.Int32(0):
                            _wait_tile_payload(work)
                        fx.barrier()
                        # Shared A/scales come from the preceding quant kernel
                        # on this stream, with no concurrent payload writers.
                        # Routed tiles still acquire the remote producer writes.
                        if uniform_work >= shared_work:
                            comm_ops.fence_system_acquire()

                    run_selected(tile_work, wait_payload=_wait_selected)
                elif const_expr(prefetch_b_before_a):
                    _do_scheduled_tile(work, wait_payload=_wait_payload_after_b)
                else:
                    _do_scheduled_tile(work)
            if const_expr(xcd_schedule):
                next_queue_attempt = queue_attempt + ((flags & fx.Int32(1)) == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
                # Finish visiting all shared queues before returning to our home
                # routed queue. No wait for other CTAs to finish shared GEMMs.
                queue_attempt = (shared_active & (next_queue_attempt == fx.Int32(8))).select(
                    fx.Int32(0), next_queue_attempt) if const_expr(shared_xcd) else next_queue_attempt
                consumer_active = queue_attempt < fx.Int32(8)
            else:
                consumer_active = (flags & fx.Int32(1)) != fx.Int32(0)
            shared_active = (shared_active & (next_queue_attempt < fx.Int32(8))) if const_expr(shared_xcd) else shared_active

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_stage2_work_head: fx.Int64, addr_shared_l13: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            addr_stage2_work_head, addr_shared_l13,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None, external_grouping=None, external_counting=None,
    shared_row_stride=None,
    skip_launch_barrier=False,
    padding_uniform_srcmap=False,
    count_uniform_matrix=False,
    row_base_prefetch=False,
    prefetch_b_before_a=False,
    joint_work_flags=False,
    preplanned=False,
    payload_chunk_rows=0, payload_tile_ready=False, payload_tile_publish_early=False, band_m=1, swiglu_limit=0.0,
    swiglu_alpha=1.702, swiglu_beta=1.0, packed_a_scale=False, unroll_a_pingpong=False, split_a_lds=False, fp8_b_waitcnt=False, prefetch_a_operand=False, scalar_tile_row_base=False, xcd_schedule=False, xcd_home=True, shared_xcd_home=True, schedule_audit=False, stage2_work_head=0, shared_l13=0, shared_xcd=False, shared_packed_heads=0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        xcd_home=xcd_home, shared_xcd_home=shared_xcd_home,
        shared_row_stride=shared_row_stride,
        b_nt=b_nt, work_shards=work_shards, external_grouping=external_grouping,
        external_counting=external_counting, payload_chunk_rows=payload_chunk_rows,
        skip_launch_barrier=skip_launch_barrier,
        padding_uniform_srcmap=padding_uniform_srcmap,
        count_uniform_matrix=count_uniform_matrix,
        row_base_prefetch=row_base_prefetch,
        prefetch_b_before_a=prefetch_b_before_a,
        joint_work_flags=joint_work_flags,
        preplanned=preplanned,
        payload_tile_ready=payload_tile_ready, band_m=band_m,
        payload_tile_publish_early=payload_tile_publish_early,
        packed_a_scale=packed_a_scale,
        unroll_a_pingpong=unroll_a_pingpong,
        split_a_lds=split_a_lds,
        fp8_b_waitcnt=fp8_b_waitcnt,
        prefetch_a_operand=prefetch_a_operand,
        scalar_tile_row_base=scalar_tile_row_base,
        xcd_schedule=xcd_schedule, schedule_audit=schedule_audit,
        reset_stage2_queue=bool(stage2_work_head), shared_l13=bool(shared_l13), shared_xcd=bool(shared_xcd),
        shared_packed_heads=bool(shared_packed_heads),
        swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, fx.Int64(stage2_work_head), fx.Int64(shared_l13), stream,
    )
# fmt: on
