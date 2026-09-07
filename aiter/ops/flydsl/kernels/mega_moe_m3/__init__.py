# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MiniMax-M3 fused MoE operator (a8w8: FP8 E4M3 activations and weights,
UE8M0 per-1x32 scales, SwiGLU-OAI alpha=1.702/beta=1.0), with lazy imports.

Forked from ``mega_moe`` (ROCm/aiter#4439), which stays untouched for
DeepSeek-V4. This copy is FP8-only: the packed-FP4 weight paths and the
a8w4 quant mode are removed rather than switched on a flag.
"""

import importlib

_LAZY = {
    "MegaMoEConfig": "mega_moe_config",
    "MegaMoEM3": "mega_moe_m3",
    "Stage1Config": "mega_moe_config",
    "Stage2Config": "mega_moe_config",
    "compile_gemm1": "gemm1",
    "gemm1_kernel": "gemm1",
    "select_mega_moe_config": "mega_moe_config",
}

__all__ = list(_LAZY)


def __getattr__(name):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
