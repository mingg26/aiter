# SPDX-License-Identifier: Apache-2.0
"""BF16 staging and completion-counted, deterministic rank-local reduction."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load


def slot_count(mask):
    return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1)


@flyc.jit
def pack_local_reduce_metadata(packed, route_masks, row_valid, *,
                               recv_cap, topk, npes, log2_max_tok):
    """Validate the original 8-bit slot before repurposing its high bits.

    Zero membership is the only invalid-row encoding consumed by the local
    scatter. In particular, slot sentinels must never become legal 2-bit slots.
    """
    token = packed & fx.Int32(0x00FFFFFF)
    original_slot = (packed >> 24) & fx.Int32(255)
    dest_pe = token >> log2_max_tok
    valid = row_valid & (token < recv_cap) & (original_slot < topk) & (dest_pe < npes)
    r_masks = buffer_ops.create_buffer_resource_from_addr(
        route_masks, num_records_bytes=recv_cap * 4)
    mask_index = valid.select(token, fx.Int32(recv_cap))
    loaded_mask = buffer_ops.buffer_load(r_masks, mask_index, vec_width=1,
                                         dtype=fx.Int32, cache_modifier=0x12)
    mask = valid.select(loaded_mask & 15, fx.Int32(0))
    return (packed & fx.Int32(0x03FFFFFF)) | (mask << 26)


@flyc.jit
def scatter_local_reduce(lds_acc_base, n_block_idx, wave, lane, staging, counters, *,
                         N_OUT, BM, BN, npes, topk, rank, log2_max_tok,
                         mask_max_tok, recv_cap, comb_inp_nbytes,
                         lds_packed_off, lds_peer_off, same_xcd=False):
    """Input LDS contains weighted BF16, including the eventual reducer's value.

    Each wave stages its rows, waits for ALL participating lanes' stores, then
    publishes one completion per row. Distinct lanes batch the row atomics.
    A winner never spins: its returned count proves every other producer has
    published. The default device-scope protocol covers cross-XCD access.
    same_xcd requires the caller to pin every producer of a (token, stripe)
    to one physical XCD: cached stores then publish to the shared L2, and NT
    loads bypass stale CU cache entries without invalidating that L2.
    Each workspace permits one in-flight invocation and starts with zero counters.
    """
    stripes = N_OUT // BN
    staging_bytes = recv_cap * topk * N_OUT * 2
    rstage = buffer_ops.create_buffer_resource_from_addr(staging, num_records_bytes=staging_bytes)
    active = lane < fx.Int32(BN // 8)
    col = active.select(lane * fx.Int32(8), fx.Int32(0))

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        p = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_packed_off) + row * 4, T.i32, align=4))))
        token = p & fx.Int32(0x00FFFFFF)
        slot = (p >> 24) & 3
        mask = (p >> 26) & 15
        valid = (token < recv_cap) & ((token >> log2_max_tok) < npes) & (slot_count(mask) > 1)
        pk = fx.Vector(lds_vec_load(lds_acc_base, (row * BN + col) * 2,
                       fx.Vector.make_type(8, fx.BFloat16), fx.BFloat16, align=16))
        offset = ((token * topk + slot) * N_OUT + n_block_idx * BN + col) * 2
        offset = (valid & active).select(offset, fx.Int32(staging_bytes))
        buffer_ops.buffer_store(pk.bitcast(fx.Int32).ir_value(), rstage, offset,
                               offset_is_bytes=True, cache_modifier=0 if same_xcd else 0x12)

    # Wait before another lane of this wave publishes the row. The default
    # device-scope stores use coherent cache bypass. The pinned variant uses
    # completion at the shared XCD L2 and does not publish to other XCDs.
    if const_expr(same_xcd):
        # gfx950: vmcnt=0, expcnt=7, lgkmcnt=15 (other counters unconstrained).
        rocdl.s_waitcnt(0xF70)
    else:
        rocdl.s_waitcnt(0)
    # For the pinned path, the staging lanes and publishing lane for each row
    # belong to the same wave. vmcnt(0) completes that wave's stores; no other
    # wave's LDS or ticket is consumed here. Keep the device-scope path intact.
    if const_expr(not same_xcd):
        fx.barrier()
    row_lane = (lane < BM // 4).select(lane, fx.Int32(0))
    row = wave + row_lane * 4
    p = fx.ptr_load(lds_typed_ptr(fx.Int32(lds_packed_off) + row * 4, T.i32, align=4))
    token = p & fx.Int32(0x00FFFFFF)
    mask = (p >> 26) & 15
    count = slot_count(mask)
    ticket = fx.Int32(-1)
    if (lane < BM // 4) & (token < recv_cap) & ((token >> log2_max_tok) < npes) & (count > 1):
        ticket = fx.Int32(comm_ops.atomic_add_agent(
            counters + fx.Int64(token * stripes + n_block_idx) * 4, fx.Int32(1)))
    # Complete the batched return values and relay the publication to the
    # cooperating row-load lanes before they acquire/read other CTAs' data.
    if const_expr(same_xcd):
        rocdl.s_waitcnt(0xF70)
    else:
        rocdl.s_waitcnt(0)
    # readlane below relays the completed atomic result within the same wave.
    if const_expr(not same_xcd):
        fx.barrier()
    if const_expr(not same_xcd):
        comm_ops.fence_agent_acquire()

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        p = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_packed_off) + row * 4, T.i32, align=4))))
        token = p & fx.Int32(0x00FFFFFF)
        slot = (p >> 24) & 3
        mask = (p >> 26) & 15
        count = slot_count(mask)
        done = fx.Int32(rocdl.readlane(T.i32, ticket, row_iter))
        dest_pe = token >> log2_max_tok
        valid = (token < recv_cap) & (dest_pe < npes)
        emit = valid & ((count == 1) | ((count > 1) & (done == count - 1)))
        if emit:
            own = fx.Vector(lds_vec_load(lds_acc_base, (row * BN + col) * 2,
                            fx.Vector.make_type(8, fx.BFloat16), fx.BFloat16, align=16))
            result = own.to(fx.Float32)
            if count > 1:
                result = fx.Vector.filled(8, 0.0, fx.Float32)
                for k_slot in range_constexpr(topk):
                    other = (mask & (1 << k_slot) != 0) & (slot != k_slot) & active
                    offset = ((token * topk + k_slot) * N_OUT + n_block_idx * BN + col) * 2
                    offset = other.select(offset, fx.Int32(staging_bytes))
                    # NT bypasses CU cache but can hit shared L2 (CDNA4 table 49).
                    # Bare buffer_inv is a NOP on gfx950, so do not replace this
                    # with a cached load plus a bare invalidate.
                    raw = buffer_ops.buffer_load(rstage, offset // 4, vec_width=4, dtype=fx.Int32,
                                                  cache_modifier=2)
                    vals = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                    own32 = own.to(fx.Float32)
                    vals = fx.Vector.from_elements([
                        (slot == k_slot).select(own32[i], vals[i]) for i in range_constexpr(8)
                    ], fx.Float32)
                    result = result + vals
                if lane == 0:
                    fx.ptr_store(fx.Int32(0), global_typed_ptr(
                        counters + fx.Int64(token * stripes + n_block_idx) * 4, T.i32))
            dest_pe_safe = emit.select(dest_pe, fx.Int32(0))
            dest_lid = token & mask_max_tok
            peer = fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_peer_off) + dest_pe_safe * 8, T.i64, align=8))
            peer = rocdl.readfirstlane(T.i64, peer.ir_value())
            rdst = buffer_ops.create_buffer_resource_from_addr(peer, num_records_bytes=comb_inp_nbytes)
            offset = ((dest_lid * npes + rank) * N_OUT + n_block_idx * BN + col) * 2
            offset = active.select(offset, fx.Int32(comb_inp_nbytes))
            buffer_ops.buffer_store(result.to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                                   rdst, offset, offset_is_bytes=True, cache_modifier=2)
