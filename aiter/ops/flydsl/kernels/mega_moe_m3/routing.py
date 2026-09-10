# SPDX-License-Identifier: MIT
"""Experimental top4 selected-softmax router with optional dispatch histogram."""
import torch
import triton as tr
import triton.language as tl
from aiter.ops.triton._triton_kernels.moe.moe_routing.topk import streaming_topk, fpval_to_key, key_to_fpval

@tr.jit
def _route_top4(X, W, I, H, M: tl.constexpr, E: tl.constexpr, WRITE_HIST: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    mask = rows[:, None] < M
    values, ids = streaming_topk(X, E, E, rows, mask, E, 4, 4, E)
    packed = (fpval_to_key(values.to(tl.uint32, bitcast=True)).to(tl.uint64) << 16) | ids
    packed = tl.sort(packed, dim=1, descending=True)
    ids = (packed & 65535).to(tl.int32)
    values = key_to_fpval((packed >> 16).to(tl.uint32)).to(tl.float32, bitcast=True)
    weights = tl.softmax(values.to(tl.float32), dim=1, keep_dims=True) * 2.0
    offsets = rows[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(W + offsets, weights, mask)
    tl.store(I + offsets, ids.to(tl.int32), mask)
    if WRITE_HIST:
        tl.atomic_add(H + ids.to(tl.int32), 1, mask=mask, sem="relaxed")

def route_top4(scores, weights, ids, histogram=None):
    if (scores.dtype != torch.float32 or not scores.is_contiguous()
            or scores.ndim != 2 or scores.shape[1] != 128
            or weights.shape != (scores.shape[0], 4) or weights.dtype != torch.float32
            or ids.shape != weights.shape or ids.dtype != torch.int32
            or not weights.is_contiguous() or not ids.is_contiguous()):
        raise ValueError("experimental router requires contiguous fp32 [M,128], fp32 weights and int32 top4 IDs")
    _route_top4[(tr.cdiv(scores.shape[0], 4),)](scores, weights, ids,
        histogram if histogram is not None else ids, scores.shape[0], 128,
        histogram is not None, num_warps=4)
    return weights, ids
