# SPDX-License-Identifier: MIT
"""Compact planning in a dedicated CTA of the preceding quantization kernel."""
import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from .. import communication_ops_utils as comm_ops
from .dispatch import DispatchSlot, emit_dispatch_plan, emit_dispatch_group
from .gemm_util import _make_buffer_from_addr, _buffer_load, _buffer_store

@flyc.jit
def prepare_dispatch(addr_disp, addr_parity, addr_expected, addr_stage2_work_head,
                     addr_in_idx, i32_cur_tok, *, num_waves, fz_npes, fz_epr,
                     fz_k, fz_mtpr, fz_rank, fz_tile_m, dispatch_blocks,
                     reset_stage2_queue, skip_launch_barrier, padding_uniform_srcmap,
                     count_uniform_matrix, row_base_prefetch, external_grouping=False,
                     payload_chunk_rows=0, payload_tile_ready=False, num_cu=256):
    tid = fx.thread_idx.x
    disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
    parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
    expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)
    a_launch_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.LAUNCH_READY)), fx.Int64)
    p_launch_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_LAUNCH_READY)), fx.Int64)
    a_work_head = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.WORK_HEAD)), fx.Int64)
    a_work_tail = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.WORK_TAIL)), fx.Int64)
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
            rows_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PAYLOAD_READY_ROWS)), fx.Int64)
            comm_ops.store_i32_system(rows_addr, fx.Int32(0), fx.Int32(fz_tile_m))
            comm_ops.fence_system_release()
        fx.barrier()
    # Keep the existing cross-rank generation handshake before reusing count metadata.
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
        if const_expr(external_grouping):
            group_done_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.GROUP_DONE)), fx.Int64)
            _buffer_store(_make_buffer_from_addr(group_done_addr, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
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
    fx.barrier()
    if tid == fx.Int32(0):
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_agent_release()
        _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_agent_release()
        if const_expr(external_grouping):
            entry_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.ENTRY_COUNT)), fx.Int64)
            gate_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.EPOCH_GATE)), fx.Int64)
            # grid_mult=1: previous S1 is complete and this counter is stable
            # throughout quant. Match the epoch its next S1 will claim.
            entries = _buffer_load(_make_buffer_from_addr(entry_addr, fx.Int64), fx.Int32(0), fx.Int64)
            gate_epoch = fx.Int32(entries // fx.Int64(num_cu) + fx.Int64(1))
            comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
    fx.barrier()
    comm_ops.fence_agent_acquire()
    parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=1)
    expected = _buffer_load(expected_rsrc, parity, fx.Int32, cache_modifier=1)
    emit_dispatch_plan(
        num_waves=num_waves, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k,
        fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_tile_m=fz_tile_m,
        fz_total_experts=fz_npes*fz_epr, addr_disp=addr_disp,
        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=parity,
        expected=expected, external_grouping=external_grouping, external_counting=False,
        histogram_precomputed=True, padding_uniform_srcmap=padding_uniform_srcmap,
        count_uniform_matrix=count_uniform_matrix, row_base_prefetch=row_base_prefetch,
        dispatch_blocks=dispatch_blocks, payload_chunk_rows=payload_chunk_rows, payload_tile_ready=payload_tile_ready)
    fx.rocdl.s_waitcnt(0)
    comm_ops.fence_system_release()


@flyc.jit
def group_dispatch_in_quant(addr_disp, addr_parity, addr_expected, addr_in_idx,
                            tokens, slot, *, num_waves, num_cu, npes, epr,
                            dispatch_blocks, payload_tile_ready):
    tid = fx.thread_idx.x
    disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
    entry_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.ENTRY_COUNT)), fx.Int64)
    gate_addr = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.EPOCH_GATE)), fx.Int64)
    entries = _buffer_load(_make_buffer_from_addr(entry_addr, fx.Int64), fx.Int32(0), fx.Int64)
    gate_epoch = fx.Int32(entries // fx.Int64(num_cu) + fx.Int64(1))
    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
        comm_ops.fence_agent_acquire()
    fx.barrier()
    parity = _buffer_load(_make_buffer_from_addr(addr_parity, fx.Int32), fx.Int32(0), fx.Int32, cache_modifier=1)
    expected = _buffer_load(_make_buffer_from_addr(addr_expected, fx.Int32), parity, fx.Int32, cache_modifier=1)
    emit_dispatch_group(num_waves=num_waves, fz_k=4, fz_total_experts=npes*epr,
        addr_disp=addr_disp, i32_cur_tok=tokens, addr_in_idx=addr_in_idx,
        dispatch_blocks=dispatch_blocks, producer_slot=slot, parity=parity,
        expected=expected, external_counting=False, adaptive_grouping=payload_tile_ready)
    fx.rocdl.s_waitcnt(0)
    comm_ops.fence_agent_release()
