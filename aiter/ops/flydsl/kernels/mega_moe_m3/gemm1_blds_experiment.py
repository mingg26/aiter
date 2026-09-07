"""Standalone experiment only: M256 N128 K256, eight M-waves sharing B.

No dispatch/sort integration and no change to production defaults. B's packed
global format is unchanged. Double-buffered, XOR-swizzled B aliases CShuffle;
full A scales retain their existing LDS layout (64 + 48 KiB at K6144).
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View
from .gemm_util import (AS2RLoader, AScaleLoader, BWeightLoader, BScaleLoader,
                       MfmaScaleGU, SiluQuantEpilogue, _make_buffer,
                       _buffer_load, wait_lds_barrier)


class _View:
    def __init__(self, ptr):
        self.ptr = ptr


def _store_b(buf, packs, offset):
    lane = fx.thread_idx.x % 64
    row = (fx.thread_idx.x // 64) * fx.Int32(16) + lane % fx.Int32(16)
    kcol = (lane // fx.Int32(16)) * fx.Int32(4)
    ptr = fx.recast_iter(fx.Int32, buf.ptr)
    for ks in range_constexpr(2):
        for half in range_constexpr(2):
            col = (kcol + fx.Int32(ks * 32 + half * 16)) ^ ((row & fx.Int32(15)) << fx.Int32(2))
            dst = fx.make_view(fx.add_offset(ptr, offset + row * fx.Int32(64) + col), fx.make_layout(4, 1))
            frag = fx.make_rmem_tensor(4, fx.Int32)
            frag.store(Vec.from_elements([packs[ks][half * 4 + j] for j in range_constexpr(4)], fx.Int32))
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), frag, dst)


def _load_a(rsrc, row_base, step, mi, ks, model_dim):
    lane = fx.thread_idx.x % 64
    row = row_base + (fx.thread_idx.x // 64) * fx.Int32(32) + fx.Int32(mi * 16) + lane % fx.Int32(16)
    byte = row * fx.Int32(model_dim) + step * fx.Int32(256) + fx.Int32(ks * 128) + (lane // fx.Int32(16)) * fx.Int32(16)
    v0 = _buffer_load(rsrc, byte // fx.Int32(16), fx.Int32, 4)
    v1 = _buffer_load(rsrc, (byte + fx.Int32(64)) // fx.Int32(16), fx.Int32, 4)
    return Vec(v0).shuffle(Vec(v1), list(range(8)))


@functools.cache
def compile_blds(*, model_dim, inter_dim, prefetch_b=False):
    assert model_dim % 256 == 0 and inter_dim % 64 == 0
    ntiles = inter_dim * 2 // 128
    steps = model_dim // 256

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, 65536, 16]
        scales: fx.Array[fx.Int8, 256 * (model_dim // 32), 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_blds(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, sx: fx.Tensor,
               sw: fx.Tensor, rows: fx.Tensor, experts: fx.Tensor,
               os: fx.Tensor, num_valid: fx.Int32, grid: fx.Int32):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ar = _make_buffer(x, fx.Int32, 4)
        br = _make_buffer(w, fx.Int32, 4)
        sr = _make_buffer(sx, fx.Int32, 4)
        bsr = _make_buffer(sw, fx.Int32)
        rr = _make_buffer(rows, fx.Int32)
        er = _make_buffer(experts, fx.Int32)
        osr = _make_buffer(os, fx.Int8, max_size=False,
                           num_records_bytes=num_valid * fx.Int32(inter_dim // 32))
        wave = fx.thread_idx.x // 64
        bl = BWeightLoader(w_rsrc=br, num_acc_n=1, model_dim=model_dim)
        bs = BScaleLoader(scale_rsrc=bsr, num_acc_n=8, model_dim=model_dim)
        asc = AScaleLoader(scale_rsrc=sr, m_repeat=2, model_dim=model_dim,
                           sort_block_m=256, total_threads=512)
        wave_scales = _View(fx.add_offset(lds.scales.ptr, wave * fx.Int32(32 * (model_dim // 32))))
        blds = AS2RLoader(k_step_bytes=256, swizzle=True)
        mfma = MfmaScaleGU(m_repeat=2, num_acc_n=8)
        epi = SiluQuantEpilogue(out_rsrc=None, out_scale_rsrc=osr, sorted_rsrc=rr,
            tokens=0, inter_dim=inter_dim, m_repeat=2, num_acc_n=8,
            sort_block_m=256, tile_n=128, num_waves=8,
            lds_out=_LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr)),
            swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0,
            always_valid=True, out_tensor=out, waves_along_m=True)
        total = (num_valid // fx.Int32(256)) * fx.Int32(ntiles)
        @flyc.jit
        def run_tile(flat):
            mt = flat // fx.Int32(ntiles)
            nt = (flat - mt * fx.Int32(ntiles)) * fx.Int32(128)
            row = _buffer_load(rr, mt, fx.Int32)
            expert = _buffer_load(er, mt, fx.Int32)
            brow = expert * fx.Int32(inter_dim * 2) + nt
            own_brow = brow + wave * fx.Int32(16)
            _store_b(lds.pool, bl.load_ni(own_brow, 0, fx.Int32(0)), fx.Int32(0))
            asc.stage(lds.scales, row)
            wait_lds_barrier()
            init = [mfma.zero_value for _ in range(16)]
            for si, state in range(0, steps, 1, init=init):
                step = fx.Int32(si)
                acc = [Vec(v) for v in state]
                cur = (step & fx.Int32(1)) * fx.Int32(8192)
                nxt = ((step + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(8192)
                next_step = (step + fx.Int32(1) < fx.Int32(steps)).select(step + fx.Int32(1), step)
                bp = bl.load_ni(own_brow, 0, next_step)
                av = [[_load_a(ar, row, step, mi, ks, model_dim) for ks in range_constexpr(2)] for mi in range_constexpr(2)]
                sa = asc.load_step(wave_scales, step)
                sb = bs.load_step(brow, step)
                for ks in range_constexpr(2):
                    if const_expr(prefetch_b):
                        bvals = [blds.load_operand(lds.pool, ni, ks, cur) for ni in range_constexpr(8)]
                    for ni in range_constexpr(8):
                        bv = bvals[ni] if const_expr(prefetch_b) else blds.load_operand(lds.pool, ni, ks, cur)
                        for mi in range_constexpr(2):
                            idx = mi * 8 + ni
                            acc[idx] = mfma._mfma(av[mi][ks], bv, acc[idx], sa[0], sb[ni // 2], ks, mi, ni % 2)
                _store_b(lds.pool, bp, nxt)
                wait_lds_barrier()
                state = yield acc
            epi.store([Vec(v) for v in state], mt, row, nt)

        for flat in range(fx.block_idx.x, total, grid):
            run_tile(flat)

    @flyc.jit
    def launch(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, sx: fx.Tensor,
               sw: fx.Tensor, rows: fx.Tensor, experts: fx.Tensor,
               os: fx.Tensor, num_valid: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel_blds(out, x, w, sx, sw, rows, experts, os, num_valid, grid,
               value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"}
               ).launch(grid=(fx.Int64(grid), 1, 1), block=(512, 1, 1), stream=stream)
    return launch


def gemm1_blds(out, x, w, sx, sw, rows, experts, os, num_valid, stream,
               *, model_dim, inter_dim, grid_mult=1, prefetch_b=False):
    assert num_valid > 0 and num_valid % 256 == 0
    grid = min(num_valid // 256 * (inter_dim * 2 // 128), 256 * grid_mult)
    _run_compiled(compile_blds(model_dim=model_dim, inter_dim=inter_dim, prefetch_b=prefetch_b),
                  out, x, w.view(torch.uint8), sx, sw.view(torch.uint8), rows,
                  experts, os, fx.Int32(num_valid), fx.Int32(grid), stream)
    return out, os
