# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

from dataclasses import replace
import os

import flydsl.expr as fx
import mori.shmem as ms
import torch

from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from .mega_moe_config import (
    FIXED_SLOT_MAX_MTPR,
    _STAGE1_OVERRIDE_ENV,
    MegaMoEConfig,
    Stage1Config,
    Stage2Config,
    select_mega_moe_config,
)
from .quant import per_1x32_mx_quant

__all__ = ["MegaMoEM3"]


class MegaMoEM3:
    """Fused dispatch, GEMM1, GEMM2, and combine with one in-flight launch per instance."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, mega_scheme: str = "fixedslot", swiglu_limit: float = 7.0,
        swiglu_alpha: float = 1.702, swiglu_beta: float = 1.0, local_reduce: bool = False,
        local_reduce_xcd_local: bool = False, shared_w13=None, shared_w13_scale=None, shared_xcd_schedule=True, shared_w2=None, shared_w2_scale=None):
    # fmt: on
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        self.local_reduce = bool(local_reduce)
        self.local_reduce_xcd_local = bool(local_reduce_xcd_local)
        if self.local_reduce_xcd_local and not self.local_reduce:
            raise ValueError("XCD-local staging requires local_reduce=True")
        if self.local_reduce:
            if world_size not in (4, 8) or topk != 4:
                raise ValueError("BF16 local reduction supports EP4 or EP8 with topk4")
            if max_tok_per_rank * topk * model_dim * 2 >= (1 << 31):
                raise ValueError("local reduction per-source staging or top-k partial buffer exceeds the buffer offset ABI")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.quant = "a8w8"
        self.swiglu_limit = float(swiglu_limit)
        # SwiGLU-OAI constants; MiniMax-M3 uses alpha=1.702, beta=1.0. Note the
        # gate is clamped on the UPPER side only, the up projection on both.
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")
        self.dev = torch.device("cuda", rank)
        self.max_recv = self.world_size * self.mtpr
        compact = self.mtpr > FIXED_SLOT_MAX_MTPR or (shared_w13 is not None and self.mtpr in (16, 32, 64, 128))
        capacity_tile_m = 128 if compact else 32
        self._s1_fixed_slot = not compact
        self._s1_scale_dim = self.model_dim // 32
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=capacity_tile_m,
            gm_scheme=mega_scheme, gm_compact=compact, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self._build_fused_stage1(w1, w1_scale)
        self._build_fused_stage2()
        self._shared_l13 = None
        self._shared_xcd_schedule = bool(shared_xcd_schedule)
        if (shared_w13 is None) != (shared_w13_scale is None):
            raise ValueError("shared_w13 and shared_w13_scale must be provided together")
        if shared_w13 is not None:
            if self.mtpr not in (16, 32, 64, 128, 256, 8192):
                raise ValueError("shared L13 supports full 16, 32, 64, 128, 256 or 8192 local tokens")
            if (self.model_dim, self.inter_dim, self.world_size, self.topk) != (6144, 3072, 8, 4):
                raise ValueError("shared L13 requires EP8/top4/H6144/I3072")
            self._shared_w13 = shared_w13.contiguous().view(torch.uint8)
            self._shared_w13_scale = shared_w13_scale.contiguous().view(torch.uint8)
            self._shared_a2 = torch.empty((self.mtpr, self.inter_dim), device=self.dev, dtype=torch.float8_e4m3fn)
            self._shared_a2_scale = torch.empty(self.mtpr * (self.inter_dim // 32) + 8192, device=self.dev, dtype=torch.uint8)
            shared_tile_m = 64 if self.mtpr <= 256 else 128
            self._shared_rows = torch.arange((self.mtpr + shared_tile_m - 1) // shared_tile_m, device=self.dev, dtype=torch.int32) * shared_tile_m
            self._shared_experts = torch.full_like(self._shared_rows, self.rank * self.epr)
            # Separate cache lines for the eight shared XCD queue heads.
            self._shared_task_count = torch.zeros(8 * 8, device=self.dev, dtype=torch.int64)
            self._shared_l13 = torch.tensor([t.data_ptr() for t in (
                self._shared_w13, self._shared_w13_scale, self._shared_a2,
                self._shared_a2_scale, self._shared_rows, self._shared_experts, self._shared_task_count)], device=self.dev, dtype=torch.int64)

        self._shared_l2 = None
        if (shared_w2 is None) != (shared_w2_scale is None):
            raise ValueError("shared_w2 and shared_w2_scale must be provided together")
        if shared_w2 is not None:
            if self._shared_l13 is None or not self._shared_xcd_schedule:
                raise ValueError("shared L2 requires shared L13 and XCD scheduling")
            if (self.model_dim, self.inter_dim) != (6144, 3072) or self.mtpr not in (16, 32, 64, 128, 256, 8192):
                raise ValueError("shared L2 supports H6144/I3072 with 16, 32, 64, 128, 256 or 8192 local tokens")
            self._shared_w2 = shared_w2.contiguous().view(torch.uint8)
            self._shared_w2_scale = shared_w2_scale.contiguous().view(torch.uint8)
            if self._shared_w2.numel() != self.model_dim * self.inter_dim:
                raise ValueError("shared W2 must contain one full MXFP8 expert")
            if self._shared_w2_scale.numel() != self.model_dim * (self.inter_dim // 32):
                raise ValueError("shared W2 scale size does not match one full expert")
            self._shared_out = torch.empty((self.mtpr, self.model_dim), dtype=torch.bfloat16, device=self.dev)
            self._shared_l2_heads = torch.zeros(8 * 8, dtype=torch.int64, device=self.dev)
            self._shared_l2 = torch.tensor([t.data_ptr() for t in (
                self._shared_a2, self._shared_a2_scale, self._shared_w2, self._shared_w2_scale,
                self._shared_experts, self._shared_out, self._shared_l2_heads)], dtype=torch.int64, device=self.dev)


    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import run_mega_moe_stage1

        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._s1_mega = run_mega_moe_stage1

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._build_v2_disp_table()

    def _allocate_dispatch_workspace(self, op, metadata_blocks):
        total_experts = self.world_size * self.epr
        workspace = {
            "local_hist": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "local_cursor": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "pair_base": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=self.dev),
            "epoch_gate": torch.zeros(10, dtype=torch.int32, device=self.dev),
            "pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "work_head": torch.zeros(8 * 16, dtype=torch.int32, device=self.dev),
            "work_tail": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "expert_tile_end": torch.empty(self.epr, dtype=torch.int32, device=self.dev),
            "max_expert_tiles": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_chunk_done": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "tile_expected": torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev),
            "active_payload_blocks": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_blocks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "payload_chunks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "group_done": torch.zeros(1, dtype=torch.int32, device=self.dev),
        }
        workspace["bigcnt"] = op._sym((self.world_size * self.epr,), torch.int32)
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        workspace["launch_ready"] = op._sym((self.world_size,), torch.int32)
        workspace["tile_ready"] = op._sym((metadata_blocks,), torch.int32)
        workspace["payload_ready_rows"] = op._sym((1,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        workspace["p2p_launch_ready"] = op._p2p_table(workspace["launch_ready"])
        workspace["p2p_tile_ready"] = op._p2p_table(workspace["tile_ready"])
        workspace["p2p_payload_ready_rows"] = op._p2p_table(workspace["payload_ready_rows"])
        self._s1_dispatch_workspace = workspace

    def _build_v2_disp_table(self):
        op = self._s1_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        table[DispatchSlot.PAIR_BASE] = workspace["pair_base"].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.SRCMAP] = op.srcmap_em.data_ptr()
        table[DispatchSlot.LOCAL_HIST] = workspace["local_hist"].data_ptr()
        table[DispatchSlot.COUNT_MATRIX] = workspace["bigcnt"].data_ptr()
        table[DispatchSlot.P2P_COUNT_MATRIX] = workspace["p2p_bigcnt"].data_ptr()
        table[DispatchSlot.COUNT_DONE] = workspace["count_done"].data_ptr()
        table[DispatchSlot.P2P_COUNT_DONE] = workspace["p2p_count_done"].data_ptr()
        table[DispatchSlot.TASK_ROW_BASE] = workspace["my_base"].data_ptr()
        table[DispatchSlot.LOCAL_CURSOR] = workspace["local_cursor"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY] = workspace["p2p_payload_ready"].data_ptr()
        table[DispatchSlot.PAIR_ORDER] = workspace["pair_order"].data_ptr()
        table[DispatchSlot.P2P_TASK_ROW_BASE] = workspace["p2p_my_base"].data_ptr()
        table[DispatchSlot.P2P_PLAN_READY] = workspace["p2p_plan_ready"].data_ptr()
        table[DispatchSlot.PLAN_READY] = workspace["plan_ready"].data_ptr()
        table[DispatchSlot.PAIR_READY] = workspace["pair_ready"].data_ptr()
        table[DispatchSlot.ENTRY_COUNT] = workspace["entry_count"].data_ptr()
        table[DispatchSlot.EPOCH_GATE] = workspace["epoch_gate"].data_ptr()
        table[DispatchSlot.PAIR_ORDER_READY] = workspace["pair_order_ready"].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        if "schedule_audit" in workspace:
            table[DispatchSlot.SCHEDULE_AUDIT] = workspace["schedule_audit"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace["expert_tile_end"].data_ptr()
        table[DispatchSlot.GROUP_DONE] = workspace["group_done"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[DispatchSlot.LAUNCH_READY] = workspace["launch_ready"].data_ptr()
        table[DispatchSlot.P2P_LAUNCH_READY] = workspace["p2p_launch_ready"].data_ptr()
        table[DispatchSlot.MAX_EXPERT_TILES] = workspace["max_expert_tiles"].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNK_DONE] = workspace["payload_chunk_done"].data_ptr()
        table[DispatchSlot.TILE_READY] = workspace["tile_ready"].data_ptr()
        table[DispatchSlot.P2P_TILE_READY] = workspace["p2p_tile_ready"].data_ptr()
        table[DispatchSlot.TILE_EXPECTED] = workspace["tile_expected"].data_ptr()
        table[DispatchSlot.ACTIVE_PAYLOAD_BLOCKS] = workspace["active_payload_blocks"].data_ptr()
        table[DispatchSlot.PAYLOAD_READY_ROWS] = workspace["payload_ready_rows"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY_ROWS] = workspace["p2p_payload_ready_rows"].data_ptr()
        table[DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION] = workspace[
            "payload_blocks_per_destination"
        ].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION] = workspace[
            "payload_chunks_per_destination"
        ].data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        config = select_mega_moe_config(
            tokens,
            self.mtpr,
            experts_per_rank=self.epr,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
        )
        # Shared small batches reuse the measured T256 geometry and preplan4.
        # This is a validated inherited configuration, not a per-size tuning claim.
        small_shared = (self._shared_l13 is not None
            and (self.world_size, self.epr, self.model_dim, self.inter_dim, self.topk)
            == (8, 16, 6144, 3072, 4)
            and tokens in (16, 32, 64, 128) and self.mtpr == tokens)
        # Measured EP8 M3 T256 configuration: P1, P2a, P2b and early B prefetch.
        # Retain the launch barrier; removing it with P1 showed no extra gain.
        if (((self.world_size, self.epr, self.model_dim, self.inter_dim, self.topk,
              tokens, self.mtpr) == (8, 16, 6144, 3072, 4, 256, 256) or small_shared)
                and not self.local_reduce and not self.local_reduce_xcd_local
                and config.p2p_quant == "none"
                and not any(os.environ.get(name) for name in _STAGE1_OVERRIDE_ENV.values())):
            config = MegaMoEConfig(
                stage1=Stage1Config(
                    sort_block_m=64, tile_n=512, tile_k=256, num_waves=8,
                    grid_mult=1, num_dispatch_cu=32, mfma_amajor=True,
                    async_a_copy=True, use_tile_resource=False, b_nt=0,
                    xcd_schedule=True, band_m=4, padding_uniform_srcmap=True,
                    count_uniform_matrix=True, row_base_prefetch=True,
                    prefetch_b_before_a=True,
                    preplan_waves=4 if small_shared else 0,
                    skip_launch_barrier=False),
                stage2=Stage2Config(
                    block_m=32, block_n=128, block_k=256, persist=True,
                    persist_cu=128, use_nt=False, queue_grid_mult=5,
                    xcd_schedule=True, band_m=8, shared_schedule="early2"),
                p2p_quant="none")
        # Best measured EP4 M3 8192-token stage configurations. S1 retains
        # M128 and double-stage B; its two-CTA alternatives regress. S2 uses
        # M64/N128/K128 with 1200 queued CTAs to feed its four-CTA capacity.
        # Keep this promotion scoped to the measured operating point.
        # Explicit tuning overrides bypass the measured default preset.
        if ((self.world_size, self.epr, self.model_dim, self.inter_dim, self.topk,
             tokens, self.mtpr) == (4, 32, 6144, 3072, 4, 8192, 8192)
                and (config.stage1.sort_block_m, config.stage2.block_m,
                     config.stage2.block_n, config.stage2.block_k,
                     config.stage2.persist, config.stage2.persist_cu)
                == (128, 64, 256, 256, True, 240)
                and config.p2p_quant == "none"
                and not any(os.environ.get(name) for name in _STAGE1_OVERRIDE_ENV.values())):
            config = replace(
                config,
                stage1=replace(config.stage1, num_waves=8, grid_mult=1,
                    num_dispatch_cu=96, pipe_weights=True, waves_per_eu_hint=2,
                    packed_a_scale=True, unroll_a_pingpong=True, split_a_lds=True,
                    xcd_schedule=True, work_shards=8, band_m=8,
                    fp8_b_waitcnt=False, prefetch_a_operand=False,
                    scalar_tile_row_base=False, schedule_audit=False),
                stage2=replace(config.stage2, block_m=64, block_n=128,
                    block_k=128, band_m=8, queue_grid_mult=5, xcd_schedule=True))
        # Small-token EP4 measurements favor the existing wide tile/B pipe
        # with fewer control CTAs and one CU-sized grid. Smaller tiles and
        # the two-CTA variant lose despite eliminating register spills.
        # Finite M bands rotate the 12 N panels across eight XCD queues;
        # GEMM retains canonical indices and the per-expert readiness wait.
        if ((self.world_size, self.epr, self.model_dim, self.inter_dim, self.topk)
                == (4, 32, 6144, 3072, 4)
                and tokens in (256, 512, 1024) and self.mtpr == tokens
                and (config.stage1.sort_block_m, config.stage1.tile_n,
                     config.stage1.tile_k, config.stage1.num_waves)
                == (64, 512, 256, 8)
                and config.p2p_quant == "none"
                and not any(os.environ.get(name) for name in _STAGE1_OVERRIDE_ENV.values())):
            config = replace(config, stage1=replace(
                config.stage1, grid_mult=1, num_dispatch_cu=32,
                work_shards=8, xcd_schedule=True, b_nt=3 if tokens == 256 else 0,
                band_m={256: 2, 512: 4, 1024: 1}[tokens]))
        # The measured local-reduce optimum at the 1024-token M3 point uses
        # the same M*N tile area as the generic preset, but swaps M32/N256 for
        # M64/N128 and halves K. It raises the resource-limited occupancy from
        # two to three CTAs/CU and changed the complete path from a regression
        # to a small win. Keep it local-reduce-only; the clean path's measured
        # optimum remains the generic M32/N256/K256 preset.
        if (self.local_reduce_xcd_local
                and (self.world_size, self.epr, self.model_dim, self.inter_dim,
                     self.topk, tokens, self.mtpr)
                == (4, 32, 6144, 3072, 4, 1024, 1024)
                and (config.stage1.sort_block_m, config.stage2.block_m,
                     config.stage2.block_n, config.stage2.block_k,
                     config.stage2.persist, config.stage2.persist_cu)
                == (64, 32, 256, 256, True, 240)
                and config.p2p_quant == "none"
                and not any(os.environ.get(name) for name in _STAGE1_OVERRIDE_ENV.values())):
            config = replace(config, stage2=replace(
                config.stage2, block_m=64, block_n=128, block_k=128,
                band_m=8, xcd_schedule=True, queue_grid_mult=5))
        if self.local_reduce_xcd_local and not (
                config.stage2.xcd_schedule and config.stage2.band_m > 1):
            # Cached staging requires every contributor to an N stripe to use
            # one physical XCD. Keep the selected GEMM tiles and S1 unchanged;
            # the queue reset is part of the production Stage2 path.
            if not config.stage2.persist or (self.model_dim // config.stage2.block_n) % 8:
                raise ValueError("XCD-local staging requires persistent S2 and eight equal N queues")
            # A queue grid counts total CTAs; the old persistent grid counted
            # persist_cu CTAs per N stripe. A multiplier of one underfilled
            # the device (4k: 240 CTAs, versus capacity for 512). Use the
            # measured multiplier five when converting to XCD queues.
            config = replace(config, stage2=replace(
                config.stage2, band_m=8, xcd_schedule=True, queue_grid_mult=5))
        self._active_config = config
        return config

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None, config: Stage1Config | None = None,
                          *, stage2_work_head=0):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        if cur_tok > self.mtpr:
            raise ValueError(f"run_tokens={cur_tok} > max_tok_per_rank={self.mtpr}")
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if tuple(x.shape) != (cur_tok, self.model_dim):
            raise ValueError(f"x must have shape ({cur_tok}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (cur_tok, self.topk):
            raise ValueError(f"wts must have shape ({cur_tok}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        if config is None:
            config = self._select_config(cur_tok).stage1
        if self._shared_l13 is not None and cur_tok != self.mtpr:
            raise ValueError("shared L13 requires the full configured local batch")
        op = self._s1_op
        # fmt: off
        self._s1_mega(
            self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim, rank=self.rank,
            experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            fixed_slot_dispatch=self._s1_fixed_slot, num_cu=self._s1_num_cu,
            sort_block_m=config.sort_block_m, tile_n=config.tile_n, tile_k=config.tile_k,
            num_waves=config.num_waves, grid_mult=config.grid_mult, pipe_weights=config.pipe_weights,
            mfma_amajor=config.mfma_amajor, swizzle_a=config.swizzle_a,
            async_a_copy=config.async_a_copy, num_dispatch_cu=config.num_dispatch_cu,
            use_tile_resource=config.use_tile_resource,
            waves_per_eu_hint=config.waves_per_eu_hint, b_nt=config.b_nt,
            work_shards=config.work_shards, external_grouping=config.external_grouping,
            skip_launch_barrier=config.skip_launch_barrier,
            padding_uniform_srcmap=config.padding_uniform_srcmap,
            count_uniform_matrix=config.count_uniform_matrix,
            row_base_prefetch=config.row_base_prefetch,
            prefetch_b_before_a=config.prefetch_b_before_a,
            joint_work_flags=config.joint_work_flags,
            preplanned=config.preplan_waves > 0,
            external_counting=config.external_counting, payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready, band_m=config.band_m,
            payload_tile_publish_early=config.payload_tile_publish_early,
            packed_a_scale=config.packed_a_scale,
            unroll_a_pingpong=config.unroll_a_pingpong,
            split_a_lds=config.split_a_lds,
            fp8_b_waitcnt=config.fp8_b_waitcnt,
            prefetch_a_operand=config.prefetch_a_operand,
            scalar_tile_row_base=config.scalar_tile_row_base,
            xcd_schedule=config.xcd_schedule, schedule_audit=config.schedule_audit,
            swiglu_limit=self.swiglu_limit, swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta, stage2_work_head=stage2_work_head,
            shared_l13=0 if self._shared_l13 is None else self._shared_l13.data_ptr(),
            shared_xcd=self._shared_l13 is not None and self._shared_xcd_schedule)
        # fmt: on
        self._s2_topk_ids = topk_ids
        self._s1_active_tile_m = config.sort_block_m
        return self._s1_active_tile_m

    def route(self, scores, weights, topk_ids):
        """Experimental top4 selected-softmax*2 route; pair once with forward on this stream."""
        from .routing import route_top4
        cfg = self._select_config(int(scores.shape[0])).stage1
        if cfg.preplan_waves and (getattr(self, "_preplan_route_pending", False)
                                 or getattr(self, "_preplan_quant_inputs", ())):
            raise ValueError("consume the preceding preplanned route/quant before routing again")
        histogram = self._s1_dispatch_workspace["local_hist"] if cfg.preplan_waves else None
        result = route_top4(scores, weights, topk_ids, histogram)
        if cfg.preplan_waves:
            self._preplan_route_ids = topk_ids
            self._preplan_route_weights = weights
            self._preplan_stream = torch.cuda.current_stream().cuda_stream
            self._preplan_route_pending = True
        return result

    def quantize(self, x_bf16, topk_ids=None, *, stream=None):
        cfg = self._select_config(int(x_bf16.shape[0]))
        if cfg.stage1.preplan_waves:
            s1 = cfg.stage1
            if (s1.preplan_waves not in (2, 4, 8) or s1.grid_mult != 1
                    or self._s1_fixed_slot or self.topk != 4
                    or self.world_size != 8 or self.epr != 16):
                raise ValueError("preplanning supports compact EP8 top4 with grid_mult=1")
            stream_ptr = stream if stream is not None else torch.cuda.current_stream().cuda_stream
            if (not getattr(self, "_preplan_route_pending", False)
                    or topk_ids is not getattr(self, "_preplan_route_ids", None)
                    or stream_ptr != self._preplan_stream):
                raise ValueError("preplanning requires route(scores, weights, topk_ids) first on the same stream")
            from .quant import preplanned_quant
            result = preplanned_quant(x_bf16, topk_ids, config=s1, op=self,
                reset_stage2_queue=cfg.stage2.band_m > 1, stream=stream)
            self._preplan_quant_inputs = (*result, topk_ids)
            if self.local_reduce:
                self._route_quant_inputs = (*result, topk_ids)
            self._preplan_route_pending = False
            return result
        if self.local_reduce:
            if topk_ids is None:
                raise ValueError("local reduction quantize requires current topk_ids")
            result = per_1x32_mx_quant(
                x_bf16, stream=stream, topk_ids=topk_ids,
                route_peers=self._route_peers, rank=self.rank,
                npes=self.world_size, mtpr=self.mtpr, epr=self.epr)
            self._route_quant_inputs = (*result, topk_ids)
            return result
        return per_1x32_mx_quant(x_bf16, stream=stream)

    def _run_joint(self, x, scales, wts, topk_ids, run_tokens, stream, slice_output):
        config = self._select_config(run_tokens)
        if config.stage1.preplan_waves:
            prepared = getattr(self, "_preplan_quant_inputs", ())
            stream_ptr = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
            if (len(prepared) != 3 or any(a is not b for a,b in zip(prepared,(x,scales,topk_ids)))
                    or wts is not self._preplan_route_weights or stream_ptr != self._preplan_stream):
                raise ValueError("preplanned S1 requires its matching route and quant on the same stream")
            self._preplan_quant_inputs = ()
        reset_queue = config.stage2.band_m > 1
        self._run_fused_stage1(
            x, wts, scales, topk_ids, stream=stream, config=config.stage1,
            stage2_work_head=self._g2_work_head.data_ptr() if reset_queue else 0)
        # Same-stream S1 completion orders the reset before any S2 queue claims.
        return self._run_stage2(run_tokens, stream, slice_output, config,
                                work_head_reset=reset_queue)

    def _run_stage2(self, run_tokens, stream, slice_output, config: MegaMoEConfig,
                    *, work_head_reset=False):
        ret = self._run_fused_stage2(run_tokens, config, stream, work_head_reset=work_head_reset)
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        x_q, scales = self.quantize(
            x_bf16, topk_ids, stream=stream.cuda_stream if stream is not None else None)
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        """Local reduction requires quantize(x, topk_ids) first on all ranks.

        Its fused route output must precede this call on the same stream.
        External quantization without that output is unsupported.
        """
        if self.local_reduce:
            prepared = getattr(self, "_route_quant_inputs", ())
            if len(prepared) != 3 or any(a.data_ptr() != b.data_ptr()
                    for a, b in zip(prepared, (x_q, scales, topk_ids))):
                raise ValueError("call quantize(x, topk_ids) before local-reduce forward_prequant")
        if self._select_config(int(x_q.shape[0])).stage1.preplan_waves:
            prepared = getattr(self, "_preplan_quant_inputs", ())
            if len(prepared) != 3 or any(a is not b for a, b in zip(prepared, (x_q, scales, topk_ids))):
                raise ValueError("preplanned forward_prequant requires matching quantize on the same stream")
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self):
        from .mega_moe_stage2 import run_mega_moe_stage2

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        self._g2_run = run_mega_moe_stage2
        self._g2_work_head = torch.zeros(8 * 64, dtype=torch.int32, device=dev)
        self._g2_schedule_audit = None
        self._g2_staging = None
        self._g2_reduce_counters = None
        if self.local_reduce:
            self._route_masks = self._s1_op._sym((self.max_recv,), torch.int32)
            ms.shmem_barrier_all()
            self._route_peers = self._s1_op._p2p_table(self._route_masks)
            self._g2_staging = torch.empty(
                (self.max_recv, self.topk, self.model_dim), dtype=torch.bfloat16, device=dev)
            # BN >= 64; every successful invocation resets all counters it uses.
            self._g2_reduce_counters = torch.zeros(
                self.max_recv * (self.model_dim // 64), dtype=torch.int32, device=dev)
        self._g2_invariants_by_quant = {}
        for p2p_quant in ("none", "fp8_blockwise_1x32"):
            p2p_row_nbytes = (
                int(comb_cfg.hidden_dim) + int(comb_cfg.hidden_dim) // 32
                if p2p_quant == "fp8_blockwise_1x32"
                else int(comb_cfg.hidden_dim) * 2
            )
            self._g2_invariants_by_quant[p2p_quant] = {
                "model_dim": int(comb_cfg.hidden_dim), "inter_dim": int(self.inter_dim),
                "experts": int(comb_cfg.num_experts_per_rank), "topk": int(k), "rank": int(comb_cfg.rank),
                "npes": int(comb_cfg.world_size), "max_tok": int(comb_cfg.max_num_inp_token_per_rank),
                "recv_cap": int(self.max_recv),
                "comb_inp_nbytes": int(comb_cfg.max_num_inp_token_per_rank) * int(k) * p2p_row_nbytes,
                "HIDDEN_MAX": int(comb_cfg.hidden_dim), "INTER_MAX": int(self.inter_dim), "cu_num": int(cu_num),
                "p2p_quant_type": p2p_quant, "fixed_slot_dispatch": bool(self._s1_fixed_slot),
            }
        self._g2_combine_placeholder = torch.empty(
            1, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev
        )

    def _run_fused_stage2(self, run_tokens, config: MegaMoEConfig, stream=None,
                          *, work_head_reset=False):
        comb_op = self.comb_op
        op = self._s1_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        stage2 = config.stage2
        if stage2.band_m > 1:
            with torch.cuda.stream(stream):
                # Standalone S2 calls have no preceding S1 to reset their queues.
                if not work_head_reset:
                    self._g2_work_head.zero_()
                if stage2.schedule_audit:
                    assert self._g2_schedule_audit is not None
                    self._g2_schedule_audit.zero_()
        p2p_quant = config.p2p_quant
        if self.local_reduce and (p2p_quant != "none" or stage2.block_n < 64):
            raise ValueError("local reduction requires BF16 transport and BN >= 64")
        invariants = self._g2_invariants_by_quant[p2p_quant]
        # fmt: off
        self._g2_run(
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp, self._s1_nvm,
            self._g2v2_inter, self._g2v2_hidden, s_fx, BM=stage2.block_m,
            SBM=config.stage1.sort_block_m, BN=stage2.block_n, BK=stage2.block_k,
            use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_spart=stage2.spatial_partition,
            persist=stage2.persist, persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided, skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds, xcd_schedule=stage2.xcd_schedule,
            band_m=stage2.band_m, schedule_audit=stage2.schedule_audit, queue_grid_mult=stage2.queue_grid_mult,
            shared_schedule=stage2.shared_schedule,
            work_head=self._g2_work_head.data_ptr(), local_reduce=self.local_reduce,
            local_reduce_xcd_local=self.local_reduce_xcd_local,
            staging_ptr=self._g2_staging.data_ptr() if self.local_reduce else 0,
            counters_ptr=self._g2_reduce_counters.data_ptr() if self.local_reduce else 0,
            route_masks_ptr=self._route_masks.data_ptr() if self.local_reduce else 0,
            audit_ptr=self._g2_schedule_audit.data_ptr() if stage2.schedule_audit else 0,
            shared_l2=self._shared_l2.data_ptr() if self._shared_l2 is not None else 0, **invariants)
        # fmt: on
        self._g2_active_block_m = stage2.block_m
        prefetch_local = self._shared_l2 is not None and not self.local_reduce and run_tokens in (16, 32, 64, 128, 256)
        return comb_op.combine_no_stage1(
            self._shared_out if self._shared_l2 is not None else self._g2_combine_placeholder,
            None, None, cur_tok=run_tokens, enable_weights=False,
            shared_input=self._shared_l2 is not None,
            stage2_p2p_quant=p2p_quant,
            stage2_topk_ids=self._s2_topk_ids if self.local_reduce or prefetch_local else None,
            prefetch_local=prefetch_local,
        )
