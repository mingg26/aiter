# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 MegaMoEM3 accuracy test.

Two modes, because they check different things:

``--mode equiv`` is the decisive structural check. It runs the untouched
upstream ``MegaMoEV2`` on FP4 weights and ``MegaMoEM3`` on FP8 weights derived
from those same FP4 values (FP4 -> dequantize -> FP8). Every e2m1 value
(0, .5, 1, 1.5, 2, 3, 4, 6 x scale) is exactly representable in e4m3 and the
block amax is unchanged, so both payloads encode bit-identical numbers. With
the activation constants matched to upstream's (alpha=1, beta=0), the two
implementations must produce the same output -- a wrong stride, cell pairing or
MFMA operand width cannot survive that, whereas it could easily hide under the
~5e-2 activation-quantization floor of a reference comparison.

``--mode m3`` runs the real M3 configuration (FP8 weights, SwiGLU-OAI
alpha=1.702 / beta=1.0) against a torch reference, with reference-only
overrides so the activation constants can be shown to matter.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori.shmem as ms
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe_m3 import MegaMoEM3
from aiter.ops.flydsl.kernels.mega_moe_m3.quant import per_1x32_mx_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

# MiniMax-M3. 128 experts keeps experts-per-rank inside the fixed-slot cap at
# EP4 (32) and EP8 (16).
M3 = {
    "model_dim": 6144,
    "inter_dim": 3072,
    "experts": 128,
    "topk": 4,
    "swiglu_limit": 7.0,
}
M3_ALPHA, M3_BETA = 1.702, 1.0


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


def _quantize_weights(local_experts, rank, seed, device, weight_scale, equiv):
    """Return (fp4 payloads for upstream, fp8 payloads for M3, dequantized refs).

    When ``equiv`` the FP8 payload is re-quantized from the dequantized FP4
    values so both encode identical numbers; otherwise it is quantized straight
    from the master weights.
    """
    model_dim, inter_dim = M3["model_dim"], M3["inter_dim"]
    gen = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    out = {}

    for name, shape, rows, cols, gate_up, scale_mul in (
        ("w1", (local_experts, 2 * inter_dim, model_dim), 2 * inter_dim, model_dim,
         True, model_dim**-0.25 * weight_scale),
        ("w2", (local_experts, model_dim, inter_dim), model_dim, inter_dim,
         False, inter_dim**-0.25),
    ):
        w = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
        w.mul_(scale_mul)

        q4, s4 = quantize(w, quant_dtype=dtypes.fp4x2)
        q4 = q4.view(local_experts, rows, cols // 2)

        if equiv:
            vals = fp4_utils.mxfp4_to_f32(q4).view(local_experts * rows, cols)
            sc = fp4_utils.e8m0_to_f32(s4).view(local_experts * rows, cols // 32)
            src = (vals * sc.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
            del vals, sc
        else:
            src = w.view(local_experts * rows, cols)
        q8, s8 = per_1x32_mx_quant(src)
        q8 = q8.view(local_experts, rows, cols)
        del w, src

        out[name] = {
            "fp4_kernel": shuffle_weight_a16w4(q4, 16, gate_up).contiguous(),
            "fp4_scale_kernel": shuffle_scale_a16w4(s4, local_experts, gate_up).contiguous(),
            "fp8_kernel": shuffle_weight_a16w4(q8.view(torch.uint8), 16, gate_up).contiguous(),
            "fp8_scale_kernel": shuffle_scale_a16w4(s8, local_experts, gate_up).contiguous(),
            "fp8_ref": q8,
            "scale_ref": s8.view(local_experts, rows, cols // 32),
        }
    torch.cuda.empty_cache()
    return out


def _all_gather(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def _dequant_fp8(weight, scale, rows, cols):
    values = weight.view(torch.float8_e4m3fn).to(torch.float32).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


@torch.no_grad()
def _reference(x, route_weights, ids, packed, rank, world, alpha, beta):
    model_dim, inter_dim = M3["model_dim"], M3["inter_dim"]
    limit = M3["swiglu_limit"]
    x_all, weights_all, ids_all = (
        _all_gather(x), _all_gather(route_weights), _all_gather(ids)
    )
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    local_experts = M3["experts"] // world
    expert_start = rank * local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_start + local_experts)]
    for expert in active.tolist():
        pos = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = pos[:, 0], pos[:, 1]
        e = expert - expert_start
        w1 = _dequant_fp8(packed["w1"]["fp8_ref"][e], packed["w1"]["scale_ref"][e],
                          2 * inter_dim, model_dim)
        w2 = _dequant_fp8(packed["w2"]["fp8_ref"][e], packed["w2"]["scale_ref"][e],
                          model_dim, inter_dim)
        inp = x_all[rows].float()
        # SwiGLU-OAI: gate clamped on the UPPER side only, up on both.
        gate = (inp @ w1[:inter_dim].T).clamp(max=limit)
        up = (inp @ w1[inter_dim:].T).clamp(-limit, limit)
        hidden = gate * torch.sigmoid(alpha * gate) * (up + beta)
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


def _rel_l2(got, ref):
    return float(
        torch.linalg.vector_norm(got.float() - ref)
        / torch.linalg.vector_norm(ref.float())
    )


def _build_m3(packed, world, rank, mtpr, alpha, beta):
    return MegaMoEM3(
        rank=rank, world_size=world,
        w1=packed["w1"]["fp8_kernel"], w1_scale=packed["w1"]["fp8_scale_kernel"],
        w2=packed["w2"]["fp8_kernel"], w2_scale=packed["w2"]["fp8_scale_kernel"],
        max_tok_per_rank=mtpr, swiglu_alpha=alpha, swiglu_beta=beta, **M3,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("equiv", "m3"), default="equiv")
    parser.add_argument("--bs-list", default="8,32,128")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tok-per-rank", type=int)
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="scale w1; at 1.0 the GEMM1 output std is ~8.9 so "
                             "nearly every gate sits at the clamp and alpha is "
                             "unobservable")
    parser.add_argument("--ref-swiglu-alpha", type=float, default=None)
    parser.add_argument("--ref-swiglu-beta", type=float, default=None)
    args = parser.parse_args()
    batch_sizes = [int(v) for v in args.bs_list.split(",")]

    rank, world, device = _setup_dist()
    try:
        if M3["experts"] % world:
            raise ValueError(f"experts must be divisible by world={world}")
        local_experts = M3["experts"] // world
        equiv = args.mode == "equiv"
        alpha, beta = (1.0, 0.0) if equiv else (M3_ALPHA, M3_BETA)

        packed = _quantize_weights(
            local_experts, rank, args.seed, device, args.weight_scale, equiv
        )
        mtpr = args.max_tok_per_rank or _next_power_of_two(max(batch_sizes))

        m3 = _build_m3(packed, world, rank, mtpr, alpha, beta)
        upstream = None
        if equiv:
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            upstream = MegaMoEV2(
                rank=rank, world_size=world, quant="a8w4",
                w1=packed["w1"]["fp4_kernel"], w1_scale=packed["w1"]["fp4_scale_kernel"],
                w2=packed["w2"]["fp4_kernel"], w2_scale=packed["w2"]["fp4_scale_kernel"],
                max_tok_per_rank=mtpr, **M3,
            )

        for bs in batch_sizes:
            x, wts, ids = _make_inputs(
                bs, M3["model_dim"], M3["experts"], M3["topk"], rank, args.seed, device
            )
            # Clone: the returned tensor can alias the instance's combine
            # output buffer, and the upstream run below would overwrite it.
            got = m3(x, wts, ids)[:bs].clone()
            _barrier()
            if equiv:
                ref_bf16 = upstream(x, wts, ids)[:bs].clone()
                ref = ref_bf16.float()
                same = torch.equal(got.float(), ref)
                # Both return bf16. Identical inputs through two MFMA operand
                # widths can still differ in fp32 accumulation order, which
                # shows up as a sparse 1-ULP spread. Measure each disagreement
                # against that element's OWN bf16 ULP -- comparing raw bit
                # patterns is meaningless near zero, where two tiny values with
                # different exponents are far apart in bits but not in value.
                g32, r32 = got.float(), ref
                diff = (g32 - r32).abs()
                # bf16 keeps 7 explicit mantissa bits.
                exp = torch.floor(torch.log2(r32.abs().clamp(min=1e-30)))
                ulp_local = torch.exp2(exp - 7)
                in_ulps = diff / ulp_local
                nz = int((diff > 0).sum())
                over1 = int((in_ulps > 1.001).sum())
                over2 = int((in_ulps > 2.001).sum())
                label = (
                    f"vs upstream MegaMoEV2 (a8w4, identical numbers) "
                    f"| differing {nz}/{diff.numel()} ({100.0 * nz / diff.numel():.2f}%) "
                    f">1ulp={over1} >2ulp={over2} "
                    f"max_ulps={float(in_ulps.max()):.2f} "
                    f"max_abs={float(diff.max()):.3e} of |ref|max={float(r32.abs().max()):.1f}"
                )
                # Decisive: if both implementations sit the same distance from
                # the torch reference, the delta between them is rounding, not
                # one of them being wrong.
                truth = _reference(x, wts, ids, packed, rank, world, alpha, beta)
                label += (
                    f" | vs torch ref: M3={_rel_l2(got, truth):.6f} "
                    f"upstream={_rel_l2(ref_bf16, truth):.6f}"
                )
            else:
                ref = _reference(
                    x, wts, ids, packed, rank, world,
                    args.ref_swiglu_alpha if args.ref_swiglu_alpha is not None else alpha,
                    args.ref_swiglu_beta if args.ref_swiglu_beta is not None else beta,
                )
                same = None
                label = "vs torch reference"
            rel = _rel_l2(got, ref)
            ms_mean = _time_graph(lambda: m3(x, wts, ids), device, args.iters)[0]
            extra = "" if same is None else f" bitwise={same}"
            if rank == 0:
                print(f"[MEGA-M3] bs={bs} relL2={rel:.6f}{extra} {label} "
                      f"e2e={ms_mean:.4f}ms", flush=True)
    finally:
        _cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
