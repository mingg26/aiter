# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent v4_pro MegaMoEV2 accuracy and performance test."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

NETWORKS = {
    "v4_pro": {
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
    # MiniMax-M3. 128 experts keeps experts-per-rank within the fixed-slot cap
    # at EP4 (32) and EP8 (16), unlike v4_pro which needs 8 ranks to get under it.
    "m3": {
        "model_dim": 6144,
        "inter_dim": 3072,
        "experts": 128,
        "topk": 4,
        "swiglu_limit": 7.0,
        # SwiGLU-OAI: gate is clamped on the UPPER side only, up on both.
        "swiglu_alpha": 1.702,
        "swiglu_beta": 1.0,
    },
}


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _cleanup():
    try:
        ms.shmem_finalize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _reduce_float(value, device, op):
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=op)
    return float(result.item())


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _make_inputs(tokens, model_dim, experts, topk, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _default_b(quant):
    return "fp8" if quant == "a8w8" else "fp4"


def _quantize_weights(model_dim, inter_dim, local_experts, rank, seed, device,
                      g2_b_dtype="fp4", g1_b_dtype="fp4", weight_scale=1.0):
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    # At weight_scale=1 the GEMM1 output has std ~8.9, so nearly every gate is
    # pinned at the +-7 clamp -- and in saturation silu_1.702(7)=7.000 vs
    # silu_1(7)=6.994, so alpha is effectively untested. Shrink to exercise it.
    w1.mul_(model_dim**-0.25 * weight_scale)
    if g1_b_dtype != "fp4":
        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

        if g1_b_dtype == "fp8eq":
            # Same equivalence probe as g2: FP4 -> dequant -> FP8 encodes
            # bit-identical numbers, so a correct FP8 B path must reproduce the
            # FP4 result exactly instead of hiding under the ~5e-2 floor.
            w1_q4, w1_scale4 = quantize(w1, quant_dtype=dtypes.fp4x2)
            vals = fp4_utils.mxfp4_to_f32(w1_q4).view(
                local_experts * 2 * inter_dim, model_dim
            )
            sc = fp4_utils.e8m0_to_f32(w1_scale4).view(
                local_experts * 2 * inter_dim, model_dim // 32
            )
            src = (vals * sc.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
            del w1_q4, w1_scale4, vals, sc
        else:
            src = w1.view(local_experts * 2 * inter_dim, model_dim)
        w1_q, w1_scale = per_1x32_mx_quant(src, quant_mode="fp8")
        w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim)
        del src
    else:
        w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_scale_ref = w1_scale.view(local_experts, 2 * inter_dim, model_dim // 32)
    del w1
    w1_kernel = shuffle_weight_a16w4(
        w1_q.view(torch.uint8) if g1_b_dtype != "fp4" else w1_q, 16, True
    ).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous()

    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    if g2_b_dtype == "fp8eq":
        # Equivalence probe for the FP8 B path: quantize to FP4, dequantize, and
        # re-quantize to FP8. Every e2m1 value (0, .5, 1, 1.5, 2, 3, 4, 6 x scale)
        # is exactly representable in e4m3 and the block amax is unchanged, so the
        # FP8 payload encodes bit-identical numbers to the FP4 one. A correct FP8
        # B path must then reproduce the FP4 run's relL2 exactly; a wrong stride,
        # cell pairing or MFMA operand width cannot hide under the activation
        # quantization floor.
        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

        w2_q4, w2_scale4 = quantize(w2, quant_dtype=dtypes.fp4x2)
        vals = fp4_utils.mxfp4_to_f32(w2_q4).view(local_experts * model_dim, inter_dim)
        sc = fp4_utils.e8m0_to_f32(w2_scale4).view(
            local_experts * model_dim, inter_dim // 32
        )
        deq = (vals * sc.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
        w2_q, w2_scale = per_1x32_mx_quant(deq, quant_mode="fp8")
        w2_q = w2_q.view(local_experts, model_dim, inter_dim)
        del w2_q4, w2_scale4, vals, sc, deq
    elif g2_b_dtype == "fp8":
        # FP8 B: one byte per element instead of a packed nibble pair, so the
        # row is twice as wide. shuffle_weight_a16w4 is a pure byte permutation
        # (the fp4 assumption lives only in the bytes-per-row), so it is reused
        # as-is -- the kernel side compensates via the doubled i32 row stride.
        # aiter.get_torch_quant(per_1x32) is fp4-only; mega_moe's own kernel-side
        # quantizer is the right contract for the FP8 payload (E8M0 round-up).
        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

        w2_q, w2_scale = per_1x32_mx_quant(
            w2.view(local_experts * model_dim, inter_dim), quant_mode="fp8"
        )
        w2_q = w2_q.view(local_experts, model_dim, inter_dim)
    else:
        w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
        w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_scale_ref = w2_scale.view(local_experts, model_dim, inter_dim // 32)
    del w2
    w2_kernel = shuffle_weight_a16w4(
        w2_q.view(torch.uint8) if g2_b_dtype.startswith("fp8") else w2_q, 16, False
    ).contiguous()
    w2_scale_kernel = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    return (
        w1_kernel,
        w1_scale_kernel,
        w2_kernel,
        w2_scale_kernel,
        w1_q,
        w1_scale_ref,
        w2_q,
        w2_scale_ref,
    )


def _dequant_expert(weight, scale, rows, cols, dtype="fp4"):
    if dtype == "fp8":
        values = weight.view(torch.float8_e4m3fn).to(torch.float32).view(rows, cols)
    else:
        values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


@torch.no_grad()
def _reference(
    x,
    route_weights,
    ids,
    ref_weights,
    rank,
    world,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
    g2_b_dtype="fp4",
    g1_b_dtype="fp4",
    swiglu_alpha=1.0,
    swiglu_beta=0.0,
):
    x_all, weights_all, ids_all = (
        _all_gather(x),
        _all_gather(route_weights),
        _all_gather(ids),
    )
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world
    expert_start = rank * local_experts
    expert_end = expert_start + local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_end)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        local_id = expert - expert_start
        w1 = _dequant_expert(
            w1_q[local_id], w1_scale[local_id], 2 * inter_dim, model_dim,
            dtype=g1_b_dtype,
        )
        w2 = _dequant_expert(
            w2_q[local_id], w2_scale[local_id], model_dim, inter_dim, dtype=g2_b_dtype
        )
        inp = x_all[rows].float()
        gate = (inp @ w1[:inter_dim].T).clamp(max=swiglu_limit)
        up = (inp @ w1[inter_dim:].T).clamp(-swiglu_limit, swiglu_limit)
        # alpha=1 / beta=0 reduces to F.silu(gate) * up (DeepSeek-V4).
        hidden = gate * torch.sigmoid(swiglu_alpha * gate) * (up + swiglu_beta)
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
        del w1, w2, inp, hidden, out
    dist.all_reduce(partial)
    start = rank * x.shape[0]
    return partial[start : start + x.shape[0]]


def _time_graph(fn, device, iters):
    _barrier()
    fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean_ms = _reduce_float(local_ms, device, dist.ReduceOp.SUM) / dist.get_world_size()
    max_ms = _reduce_float(local_ms, device, dist.ReduceOp.MAX)
    return mean_ms, max_ms


def _run_size(moe, x, weights, ids, ref_weights, args, rank, world, device):
    tokens = x.shape[0]
    output = moe(x, weights, ids)[:tokens]
    _barrier()
    rel_l2 = -1.0
    if tokens <= args.accuracy_max_bs:
        reference = _reference(
            x,
            weights,
            ids,
            ref_weights,
            rank,
            world,
            moe.model_dim,
            moe.inter_dim,
            moe.experts,
            moe.swiglu_limit,
            g2_b_dtype=getattr(moe, "g2_b_dtype", "fp4"),
            g1_b_dtype=getattr(moe, "g1_b_dtype", "fp4"),
            swiglu_alpha=(
                args.ref_swiglu_alpha
                if args.ref_swiglu_alpha is not None
                else getattr(moe, "swiglu_alpha", 1.0)
            ),
            swiglu_beta=(
                args.ref_swiglu_beta
                if args.ref_swiglu_beta is not None
                else getattr(moe, "swiglu_beta", 0.0)
            ),
        )
        rel_l2 = float(
            torch.linalg.vector_norm(output.float() - reference)
            / torch.linalg.vector_norm(reference)
        )
        rel_l2 = _reduce_float(rel_l2, device, dist.ReduceOp.MAX)
        if rel_l2 >= args.rtol:
            raise AssertionError(f"bs={tokens} relL2={rel_l2:.6f} exceeds {args.rtol}")

    x_q, scale = moe.quantize(x)
    state = {}

    def stage1():
        moe._run_fused_stage1(x_q, weights, scale, ids)

    def stage2():
        state["output"] = moe._run_stage2(tokens, None, True, moe._active_config)

    def end_to_end():
        state["output"] = moe(x, weights, ids)

    stage1_ms = _time_graph(stage1, device, args.iters)
    stage1()
    _barrier()
    stage2_ms = _time_graph(stage2, device, args.iters)
    e2e_ms = _time_graph(end_to_end, device, args.iters)
    sbm = int(moe._s1_active_tile_m)
    gemm2_bm = int(moe._g2_active_block_m)
    p2p_quant = moe._active_config.p2p_quant
    if rank == 0:
        print(
            f"[MEGA-V2] bs={tokens} relL2={rel_l2:.6f} "
            f"path={'fixed' if moe._s1_fixed_slot else 'compact'} "
            f"p2p_quant={p2p_quant} SBM={sbm} G2_BM={gemm2_bm} "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms "
            f"e2e={e2e_ms[0]:.4f}/{e2e_ms[1]:.4f}ms mean/max",
            flush=True,
        )


def _run_burst(moe, x, weights, ids, depth, rank):
    moe(x, weights, ids)
    _barrier()
    for _ in range(depth):
        moe(x, weights, ids)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[BURST] completed={depth}/{depth}", flush=True)


def _install_config_policy(moe, config_tokens, unify_fields):
    if not config_tokens:
        return
    reference = moe._select_config(config_tokens)
    original_select = moe._select_config
    fields = [field for field in unify_fields.split(",") if field]

    if not fields:

        def select_config(_tokens):
            moe._active_config = reference
            return reference

    else:

        def select_config(tokens):
            local = original_select(tokens)
            stage1_updates = {}
            stage2_updates = {}
            p2p_quant = local.p2p_quant
            for field in fields:
                if field.startswith("stage1."):
                    name = field.removeprefix("stage1.")
                    stage1_updates[name] = getattr(reference.stage1, name)
                elif field.startswith("stage2."):
                    name = field.removeprefix("stage2.")
                    stage2_updates[name] = getattr(reference.stage2, name)
                elif field == "p2p_quant":
                    p2p_quant = reference.p2p_quant
                else:
                    raise ValueError(f"invalid config field {field!r}")
            config = replace(
                local,
                stage1=replace(local.stage1, **stage1_updates),
                stage2=replace(local.stage2, **stage2_updates),
                p2p_quant=p2p_quant,
            )
            moe._active_config = config
            return config

    moe._select_config = select_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="v4_pro")
    parser.add_argument("--bs-list", default="128")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    # Negative controls: override the SwiGLU constants on the REFERENCE side
    # only. If the kernel really honours alpha/beta, mismatching them here must
    # blow relL2 up; matching them is otherwise just self-consistency.
    parser.add_argument("--ref-swiglu-alpha", type=float, default=None)
    parser.add_argument("--ref-swiglu-beta", type=float, default=None)
    parser.add_argument(
        "--quant",
        choices=("a8w4", "a8w8"),
        default="a8w4",
        help="a8w4 = DeepSeek-V4, a8w8 = MiniMax-M3; sets the default B dtype "
        "for both GEMMs (--g1-b-dtype/--g2-b-dtype still override)",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="scale w1 so the SwiGLU input lands in a chosen regime; <1 moves "
        "off the clamp and makes alpha observable",
    )
    parser.add_argument(
        "--g2-b-dtype",
        choices=("fp4", "fp8", "fp8eq"),
        default=None,
        help="GEMM2 B-operand dtype; fp8 is the MiniMax-M3 Phase-1 path",
    )
    parser.add_argument(
        "--g1-b-dtype",
        choices=("fp4", "fp8", "fp8eq"),
        default=None,
        help="GEMM1 B-operand dtype; fp8 completes the MiniMax-M3 a8w8 path",
    )
    parser.add_argument("--accuracy-max-bs", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--max-tok-per-rank", type=int)
    parser.add_argument("--rank-tokens", default="")
    parser.add_argument("--config-tokens", type=int, default=0)
    parser.add_argument("--unify-fields", default="")
    parser.add_argument("--burst-depth", type=int, default=0)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.bs_list.split(",")]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("--bs-list must contain positive integers")
    rank_tokens = [int(value) for value in args.rank_tokens.split(",") if value]

    rank, world, device = _setup_dist()
    try:
        network = NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError(
                f"experts={network['experts']} must be divisible by world={world}"
            )
        if args.max_tok_per_rank is not None and args.max_tok_per_rank < max(
            batch_sizes
        ):
            raise ValueError("--max-tok-per-rank must cover the largest batch size")
        if args.config_tokens and args.max_tok_per_rank is None:
            raise ValueError("--config-tokens requires --max-tok-per-rank")
        if args.config_tokens < 0 or (
            args.max_tok_per_rank is not None
            and args.config_tokens > args.max_tok_per_rank
        ):
            raise ValueError(
                "--config-tokens must be between zero and max-tok-per-rank"
            )
        local_experts = network["experts"] // world
        packed = _quantize_weights(
            network["model_dim"],
            network["inter_dim"],
            local_experts,
            rank,
            args.seed,
            device,
            g2_b_dtype=args.g2_b_dtype or _default_b(args.quant),
            g1_b_dtype=args.g1_b_dtype or _default_b(args.quant),
            weight_scale=args.weight_scale,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        if rank_tokens and len(rank_tokens) != world:
            raise ValueError("--rank-tokens must contain one value per rank")
        max_bs = max(batch_sizes + rank_tokens)
        x, weights, ids = _make_inputs(
            max_bs,
            network["model_dim"],
            network["experts"],
            network["topk"],
            rank,
            args.seed,
            device,
        )
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        for batch_size in batch_sizes:
            local_batch_size = rank_tokens[rank] if rank_tokens else batch_size
            max_tok_per_rank = args.max_tok_per_rank or max(
                16, _next_power_of_two(batch_size)
            )
            moe = MegaMoEV2(
                rank=rank,
                world_size=world,
                quant=args.quant,
                g2_b_dtype=(
                    None if args.g2_b_dtype is None
                    else ("fp8" if args.g2_b_dtype.startswith("fp8") else "fp4")
                ),
                g1_b_dtype=(
                    None if args.g1_b_dtype is None
                    else ("fp8" if args.g1_b_dtype.startswith("fp8") else "fp4")
                ),
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=max_tok_per_rank,
                **network,
            )
            _install_config_policy(moe, args.config_tokens, args.unify_fields)
            if rank_tokens:
                selected = moe._select_config(local_batch_size)
                configs = [None] * world
                dist.all_gather_object(
                    configs,
                    (
                        rank,
                        local_batch_size,
                        selected.stage1,
                        selected.stage2,
                        selected.p2p_quant,
                    ),
                )
                if rank == 0:
                    for config in configs:
                        print(f"[CONFIG] {config}", flush=True)
            local_x = x[:local_batch_size].contiguous()
            local_weights = weights[:local_batch_size].contiguous()
            local_ids = ids[:local_batch_size].contiguous()
            if args.burst_depth:
                _run_burst(
                    moe, local_x, local_weights, local_ids, args.burst_depth, rank
                )
            else:
                _run_size(
                    moe,
                    local_x,
                    local_weights,
                    local_ids,
                    ref_weights,
                    args,
                    rank,
                    world,
                    device,
                )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
