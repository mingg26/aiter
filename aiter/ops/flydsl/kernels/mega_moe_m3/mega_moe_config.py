# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEM3 configuration rules for MI355X."""

import os
from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import cache

TOKEN_BUCKETS = (
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
P2P_FP8_MIN_MTPR = 1024

# --- Validated shapes for the fused shared-expert path -----------------------
# One source of truth for the shape sets that MegaMoEM3, Stage1 and Stage2 all
# have to agree on. They used to be six separate literals whose agreement was
# coincidental; anything that widens one of these must widen the others with it.
#
# The split is Stage1's regime boundary, not an arbitrary size list: <=256 runs
# the finite small-XCD band schedule with a 512-wide N tile, >=512 runs the
# tile-ready 8192 schedule with a 128-row sort block. Rows per expert is
# npes*T*topk/experts, i.e. exactly 128 at T=512 and a multiple of 128 above it,
# which is what lets the 128-row sort block and the 128-row shared row table in
# MegaMoEM3 divide evenly at every large size.
# Every multiple of eight up to 256. Stage1 counts shared tiles with a ceiling
# and bounds the short last one, so the local batch no longer has to be a
# multiple of the sort block; see the tile count in mega_moe_stage1 and the
# _sct1 tag that marks the kernels which take that path.
SHARED_FUSED_MTPR_SMALL = tuple(range(8, 257, 8))

# Stage1's sort block per small-regime size, and the stride of the shared row
# table MegaMoEM3 builds. One table because Stage1 divides the local batch by the
# sort block while the row table is built from the stride: if the two drift, the
# shared tile counts disagree and the L13 epoch accounting is silently wrong.
#
# 96 has to take 32. Stage1 only uses a ceiling for the shared tile count while
# the whole local batch is smaller than one sort block; above that it divides, so
# the batch must be a multiple of the sort block. 96 % 64 != 0 would drop the
# third tile, while 96 = 3 * 32 is exact and needs no partial-tile handling.
def small_sort_block_m(mtpr: int) -> int:
    """Stage1's sort block for a small-regime batch, and the stride of the shared
    row table MegaMoEM3 builds from it.

    This is the generic fallback rule. The validated EP8 fused-shared rollout
    separately selects SBM128 for the exact batches in SBM128_PATHS, using the
    same scope guard for the selector and shared-row-table allocation.

    The block has to be large enough to hold a whole expert's rows, not merely
    large enough to be efficient. Stage1 pads each expert up to a multiple of the
    block, so its tile count is sum_e ceil(rows_e / block), and rows_e is random:
    its mean is tokens/4 at EP8 top4 with 128 experts, with a standard deviation
    near its square root. When the mean sits within a couple of deviations of the
    block, whether an expert needs a second tile is close to a coin flip, and the
    per-rank tile count stops being the same on every rank. Each tile makes a
    full pass over that expert's 37.7 MB of GEMM1 weights, so the tile count is
    what Stage1 costs, and the layer runs at the slowest rank.

    Measured at 104 tokens, where the mean is 26 rows against a 32-row block: the
    eight ranks took 17, 19, 18, 19, 19, 17, 17 and 17 tiles, Stage1 ran 155 us on
    the four low ranks and 191 us on the four high ones, and the complete path was
    313.2 us. A 64-row block holds every expert in one tile, gives all eight ranks
    16 tiles, flattens Stage1 to 152-159 us and takes the path to 290.0 us -- and
    it does that while computing nearly twice the padded rows, because the padded
    rows carry no memory traffic (their reads and writes are bounded) and Stage1
    is nowhere near the matrix pipe's limit.

    So 64 from 104 tokens up, where the mean is 26 rows or more, and 32 below,
    where a 64-row tile would be three-quarters empty and the padding stops being
    free (32 is worth 1.7% at 16 tokens and 6.8% at 32). The rule reproduces every
    previously measured entry: 32 at 16 and 32, 64 at 64, 32 at 96, 64 at 128 and
    256. At 200 the same boundary reappears one level up -- the mean is 50 rows
    and the ranks take 16, 17, 16, 16, 18, 16, 16, 16 tiles, Stage1 spreads 19%
    and the path costs 20 us more than at 192 -- so a 128-row block is the next
    thing to measure there.
    """
    return 64 if (mtpr % 64 == 0 or mtpr > 96) else 32


SHARED_SMALL_SORT_BLOCK_M = {t: small_sort_block_m(t) for t in SHARED_FUSED_MTPR_SMALL}
SHARED_FUSED_MTPR_LARGE = (512, 1024, 2048, 4096, 8192)
SHARED_FUSED_MTPR = SHARED_FUSED_MTPR_SMALL + SHARED_FUSED_MTPR_LARGE

# Stage1 geometry each regime was measured with. tile_k is not listed: GEMM1
# pins it to 256 globally (A_K_STEP_BYTES), so repeating it per regime only
# duplicated that assert. sort_block_m is free within the small regime (32 or
# 64, guarded there against the shared row table stride) and fixed in the large.
SHARED_FUSED_S1_GEOMETRY = {
    "small": dict(tile_n=512, num_waves=8, band_m=4),
    "small_sbm128": dict(tile_n=256, num_waves=8, band_m=4),
    "large": dict(sort_block_m=128, tile_n=256, num_waves=8, band_m=4),
}

# Stage2 shapes validated with fused shared L2, as max_tok -> (BN, BK, BM, SBM)
# tuples. This is a ledger of what has actually been run through the numeric,
# bit-exact-replay and epoch gates, not a derivable constraint: BN divides the
# shared L2 queue period (model_dim // BN), which the host's epoch bookkeeping in
# MegaMoEM3 and the kernel compute separately, so an unvalidated BN does not fail
# loudly -- it drifts the epoch. Structural limits are separate and live in
# compile_mega_moe_stage2 (BN % 64, 256 % BK, BK % 128), except one that shows up
# here as an absence: BM=64 with BN=256 exceeds the 64 KB workgroup LDS.
#
# SBM follows Stage1's sort_block_m, so the SBM column mirrors what Stage1 may
# emit at that size: 32 wherever SHARED_SMALL_SORT_BLOCK_M picks it (measured at
# EP8 b32, where it is worth 6.8% on its own), 64 through 256, and 128 for the
# large regime. A 32-row block no longer implies a single shared tile: at 96 local
# tokens it is three. The (64, 64) BM/SBM pair at
# max_tok=64 is that size's measured best: a 64-row S2 tile under a 64-row sort
# block gives each B panel exactly one consumer, which is what makes the
# non-temporal B load profitable there.
#
# BN=256 halves the number of times Stage2 re-reads its A2 panel and is the
# measured optimum at 16, 32 and 128 (0.6-1.0% on the complete path, direction
# reproduced over 6+2+2 paired runs). It loses at 256 (-2.5%) and at 512 it is
# only reachable with BM=32, whose own penalty is larger, so neither size lists
# it: production emits only what this table allows, and sweeps that want to
# explore further use a kernel-source variant.
SHARED_L2_S2_SHAPES = {
    16: ((128, 256, 32, 64), (128, 256, 32, 32), (256, 256, 32, 32)),
    32: ((128, 256, 32, 64), (128, 256, 32, 32), (256, 256, 32, 32)),
    64: ((128, 256, 32, 64), (128, 256, 64, 64)),
    # 96 mirrors 16/32: SBM follows Stage1's 32-row sort block. Both BN values are
    # listed and BN=256 is now the selector default, measured: 273.566 us against
    # 282.025 at BN=128, two interleaved pairs with no overlap.
    96: ((128, 256, 32, 32), (256, 256, 32, 32)),
    128: ((128, 256, 32, 64), (256, 256, 32, 64)),
    256: ((128, 256, 32, 64),),
    512: ((128, 128, 64, 128),),
    1024: ((128, 128, 64, 128),),
    2048: ((128, 128, 64, 128),),
    4096: ((128, 128, 64, 128),),
    8192: ((128, 128, 64, 128),),
}

# The sizes between the measured ones take the same tile their neighbours were
# measured with: BM=32 under a BK=256 skeleton, SBM following Stage1's sort
# block, and both BN values listed so the selector may emit either. max_tok
# enters the shared queue only through ceil(max_tok / BM), and band_m=8 rounds
# that up to the same eight m blocks for every batch at or below 256, so these
# sizes share the epoch period of the 96, 128 and 256 entries already validated
# here -- unlike a new BN, which would change the period itself.
for _t in SHARED_FUSED_MTPR_SMALL:
    if _t not in SHARED_L2_S2_SHAPES:
        _sbm = SHARED_SMALL_SORT_BLOCK_M[_t]
        SHARED_L2_S2_SHAPES[_t] = ((128, 256, 32, _sbm), (256, 256, 32, _sbm))
del _t, _sbm

# Measured b72 M16/M32 N256 default; other SBM32 batches stay generic.
SBM32_PATHS = {72: "m16_m32_n256"}

# Measured M32/M48/M64 N256 defaults.
SBM64_PATHS = {t: "m32_m48_m64" for t in
               (64, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192)}

# Validated rollout sizes only. The generic shared-row table stays unchanged;
# MegaMoEM3 opts into SBM128 only for the measured EP8 fused-shared workload.
SBM128_PATHS = {
    200: "m16", 208: "m16", 216: "m16",
    224: "tiered96", 232: "tiered96", 240: "tiered96",
    248: "tiered96", 256: "tiered96",
}
for _t in SBM128_PATHS:
    SHARED_L2_S2_SHAPES[_t] = tuple(dict.fromkeys((
        *SHARED_L2_S2_SHAPES[_t],
        (128 if _t == 256 else 256, 256, 32, 128),
    )))
del _t


def shared_l2_s2_shape_ok(max_tok: int, BM: int, BN: int, BK: int, SBM: int) -> bool:
    """Is this Stage2 tile validated for the fused shared L2 path at max_tok?"""
    return (BN, BK, BM, SBM) in SHARED_L2_S2_SHAPES.get(int(max_tok), ())

FIXED_SLOT_MAX_MTPR = 255
MAX_MTPR_CLASS = 32768
REFERENCE_EXPERTS_PER_RANK = 48
EXPERT_CONFIG_GRANULARITY = 64


@dataclass(frozen=True, slots=True)
class Stage1Config:
    sort_block_m: int
    tile_n: int
    num_waves: int
    grid_mult: int
    num_dispatch_cu: int
    mfma_amajor: bool
    async_a_copy: bool
    use_tile_resource: bool
    b_nt: int
    waves_per_eu_hint: int = 2
    tile_k: int = 256
    pipe_weights: bool = True
    swizzle_a: bool = True
    work_shards: int = 8
    external_grouping: bool = False
    external_counting: bool = False
    skip_launch_barrier: bool = False
    padding_uniform_srcmap: bool = False
    count_uniform_matrix: bool = False
    row_base_prefetch: bool = False
    prefetch_b_before_a: bool = False
    joint_work_flags: bool = False
    preplan_waves: int = 0
    payload_chunk_rows: int = 0
    payload_tile_ready: bool = False
    payload_tile_publish_early: bool = False
    packed_a_scale: bool = False
    unroll_a_pingpong: bool = False
    split_a_lds: bool = False
    fp8_b_waitcnt: bool = False
    scalar_tile_row_base: bool = False
    prefetch_a_operand: bool = False
    xcd_schedule: bool = False
    schedule_audit: bool = False
    # M tiles per reuse band. Small XCD schedules map bands to canonical
    # GEMM indices; tile-ready schedules use build_fused_gemm1._decode.
    band_m: int = 1
    # Seed of the per-CTA home queue under xcd_schedule: the physical XCD (the
    # default, giving B-panel affinity) or the CTA index. Seed only -- every CTA
    # still visits all eight queues exactly once, so tickets and epochs are
    # unchanged and the default compiles to the same code object as before.
    xcd_home: bool = True
    shared_xcd_home: bool = True
    # Independent shared/routed 16-bit fields; preserve shared-first issuance.
    shared_packed_heads: bool = False
    # Isolated measured S1 implementations; never inferred from the bucket.
    sbm128_path: str = "generic"
    sbm64_path: str = "generic"
    sbm32_path: str = "generic"

    def __post_init__(self):
        if self.sbm32_path not in ("generic", "m16_m32_n256"):
            raise ValueError(f"Unknown SBM32 path {self.sbm32_path!r}")
        if self.sbm32_path != "generic" and (
            (self.sort_block_m, self.tile_n) != (32, 256)
            or self.sbm64_path != "generic" or self.sbm128_path != "generic"
        ):
            raise ValueError("Specialized SBM32 requires SBM32/N256 and generic SBM64/SBM128")
        if self.sbm64_path not in ("generic", "full", "m16_dma", "m16", "m32_m48_m64"):
            raise ValueError(f"Unknown SBM64 path {self.sbm64_path!r}")
        if self.sbm64_path != "generic" and (
            (self.sort_block_m, self.tile_n) != (64, 256 if self.sbm64_path == "m32_m48_m64" else 512)
            or self.sbm128_path != "generic"
        ):
            raise ValueError("Specialized SBM64 paths require SBM64, the path-specific N, and generic SBM128")
        if self.sbm128_path not in ("generic", "m16", "tiered96"):
            raise ValueError(f"Unknown SBM128 path {self.sbm128_path!r}")
        if self.sbm128_path != "generic" and (self.sort_block_m, self.tile_n) != (128, 256):
            raise ValueError("Specialized SBM128 paths require SBM128/N256")


SHARED_L2_SCHEDULES = ("tail", "early2", "jointtail")


@dataclass(frozen=True, slots=True)
class Stage2Config:
    block_m: int
    block_n: int
    persist: bool
    persist_cu: int
    use_nt: bool
    persist_strided: bool = False
    skew_cu: int = 0
    block_k: int = 256
    b_hoist: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = 402
    bf16_lds: bool = False
    queue_grid_mult: int = 1
    xcd_schedule: bool = False
    band_m: int = 1
    schedule_audit: bool = False
    # Compile-time shared L2 task order; ignored when shared L2 is disabled.
    # jointtail shares one routed-first ticket space per N-panel queue.
    shared_schedule: str = "tail"
    # Seed of the per-CTA home queue under xcd_schedule: the physical XCD (the
    # default, giving B-panel affinity) or the CTA index. Seed only -- every CTA
    # still visits all eight queues exactly once, so tickets and epochs are
    # unchanged and the default compiles to the same code object as before.
    xcd_home: bool = True
    shared_xcd_home: bool = True

    def __post_init__(self):
        if self.shared_schedule not in SHARED_L2_SCHEDULES:
            raise ValueError(f"unsupported shared_schedule={self.shared_schedule!r}")


@dataclass(frozen=True, slots=True)
class MegaMoEConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    p2p_quant: str

    def __post_init__(self):
        sbm = self.stage1.sort_block_m
        bm = self.stage2.block_m
        if bm > sbm or sbm % bm:
            raise ValueError(
                f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}"
            )
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")


def nearest_token_bucket(tokens: int) -> int:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    index = bisect_left(TOKEN_BUCKETS, tokens)
    if index == 0:
        return TOKEN_BUCKETS[0]
    if index == len(TOKEN_BUCKETS):
        return TOKEN_BUCKETS[-1]
    lower, upper = TOKEN_BUCKETS[index - 1], TOKEN_BUCKETS[index]
    return upper if upper - tokens <= tokens - lower else lower


def mtpr_config_class(mtpr: int) -> int:
    return mtpr if mtpr <= P2P_FP8_MIN_MTPR else MAX_MTPR_CLASS


def expert_config_class(experts_per_rank: int) -> int:
    return (
        (experts_per_rank + EXPERT_CONFIG_GRANULARITY - 1)
        // EXPERT_CONFIG_GRANULARITY
        * EXPERT_CONFIG_GRANULARITY
    )


def _scale_dispatch_cu(dispatch_cu: int, experts_per_rank: int) -> int:
    expert_waves = (experts_per_rank + 63) // 64
    return min(224, dispatch_cu * expert_waves)


def _fixed_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 64
    if bucket <= 8:
        return 128
    if bucket <= 16:
        return 96
    if bucket <= 32:
        return 128
    return min(224, 16 * (bucket.bit_length() + 7))


def _compact_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 16:
        return 64
    if bucket <= 32:
        return 128
    if bucket <= 64:
        return 192
    return 128


def _large_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 32:
        return 64
    if bucket <= 64:
        return 160
    if bucket <= 128:
        return 192
    if bucket <= 256:
        return 160
    if bucket == 8192:
        return 96
    if bucket >= 16384:
        return 32
    return 64


def _select_fixed_stage1(bucket: int, experts_per_rank: int) -> Stage1Config:
    grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
    return Stage1Config(
        sort_block_m=32,
        tile_n=256 if bucket <= 8 else 128,
        num_waves=4,
        grid_mult=grid_mult,
        num_dispatch_cu=_scale_dispatch_cu(
            _fixed_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=False,
        async_a_copy=False,
        use_tile_resource=bucket <= 16,
        b_nt=0 if bucket == 1 else 3,
        waves_per_eu_hint=1 if bucket == 16 else 2,
    )


def _select_bounded_stage1(
    bucket: int, mtpr: int, experts_per_rank: int, inter_dim: int
) -> Stage1Config:
    if bucket <= 4:
        sort_block_m, tile_n, num_waves = 32, 256, 4
        grid_mult, mfma_amajor, async_a_copy = 1, False, False
    elif bucket <= 128:
        sort_block_m = 32
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        grid_mult, mfma_amajor, async_a_copy = 1, True, True
    elif bucket <= 1024:
        sort_block_m = 64
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        grid_mult, mfma_amajor, async_a_copy = (1 if bucket == 256 else 2), True, True
    else:
        raise ValueError(f"bounded MTPR does not support token bucket {bucket}")

    dispatch_cu = (
        _compact_dispatch_cu(bucket) if bucket <= 128 else 160 if bucket == 256 else 128
    )
    tile_resource = bucket == 256
    b_nt = 0 if bucket == 1 or bucket >= 1024 else 3
    if mtpr > bucket:
        if bucket == 32:
            dispatch_cu = 64
        elif bucket == 64:
            dispatch_cu = 160
        elif bucket == 128:
            dispatch_cu = 192
        elif bucket == 512:
            grid_mult, dispatch_cu, tile_resource, b_nt = 1, 64, True, 0
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=grid_mult,
        num_dispatch_cu=_scale_dispatch_cu(dispatch_cu, experts_per_rank),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=tile_resource,
        b_nt=b_nt,
    )


# gfx950 LDS budget for one GEMM1 workgroup. Achieved bandwidth measured on
# MI350X at DEP4/M3 shapes, total traffic / stage1 time, M_r=8192:
#
#     LDS 152KB -> 3.19 TB/s      LDS 88KB -> 5.37 TB/s
#     LDS 120KB -> 4.43 TB/s      LDS 44KB -> 6.16-6.57 TB/s
#
# Bigger tiles move fewer bytes but resident workgroups per CU drop with LDS,
# and past ~88KB the bandwidth loss outruns the traffic saving. 96KB sits above
# the measured-good point and below the first measured-bad one.
_STAGE1_LDS_BUDGET = 96 * 1024


def _stage1_lds_bytes(sort_block_m: int, tile_n: int, model_dim: int) -> int:
    """Mirror of the LDS struct in mega_moe_stage1.compile_mega_moe_stage1."""
    a_lds = sort_block_m * 256  # tile_k=256 is fixed for GEMM1
    pool = max(2 * a_lds, sort_block_m * (tile_n // 2) * 4)
    a_scale = sort_block_m * (model_dim // 32)
    return pool + a_scale


def _pick_tile_n(sort_block_m: int, inter_dim: int, model_dim: int) -> int:
    """Largest tile_n that keeps num_waves=8 legal and LDS inside the budget.

    num_waves=8 is worth ~8% over 4 (measured 5.37 vs 4.97 TB/s at equal LDS and
    traffic), and it requires NUM_ACC_N = tile_n/8/16 to be even, i.e. tile_n a
    multiple of 256. tile_n must also divide 2*inter_dim. Among the candidates
    that survive, larger is better -- activation traffic is pairs*H^2/tile_n --
    until the LDS budget bites.
    """
    n = 2 * inter_dim
    candidates = [t for t in range(256, n + 1, 256) if n % t == 0]
    fits = [
        t
        for t in candidates
        if _stage1_lds_bytes(sort_block_m, t, model_dim) <= _STAGE1_LDS_BUDGET
    ]
    return max(fits) if fits else min(candidates)


# The dispatch's payload chunk, in rows; must be a multiple of sort_block_m.
# It is also the natural GEMM1 weight-reuse band (band_m =
# PAYLOAD_CHUNK_ROWS // sort_block_m), but banding measured FLAT -- see
# build_fused_gemm1._decode and handoff_megamoe.md S7.4 -- so band_m stays 1
# and the reorder survives only as the M3_MEGAMOE_BAND_M hook.
PAYLOAD_CHUNK_ROWS = 384


def _select_large_stage1(
    bucket: int, experts_per_rank: int, inter_dim: int, model_dim: int
) -> Stage1Config:
    band_m = 1
    if bucket <= 4:
        sort_block_m, tile_n, num_waves = 32, 256, 4
        mfma_amajor, async_a_copy = False, False
    elif bucket <= 128:
        sort_block_m = 32
        tile_n, num_waves = _pick_tile_n(sort_block_m, inter_dim, model_dim), 8
        mfma_amajor, async_a_copy = True, True
    else:
        # sort_block_m sets weight traffic (pairs*H^2/sort_block_m) and must
        # divide both fuse_cap and payload_chunk_rows, leaving {32, 64, 128}.
        # 128 measured best at every prefill size once stage2 is re-selected
        # consistently -- M_r=8192: S1 2701us at 128 against 3678us at 64.
        sort_block_m = 128
        tile_n, num_waves = _pick_tile_n(sort_block_m, inter_dim, model_dim), 8
        mfma_amajor, async_a_copy = True, True

    work_shards = 1 if bucket <= 32 else 4
    if bucket == 2048:
        work_shards = 8
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=1,
        num_dispatch_cu=_scale_dispatch_cu(
            _large_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=True,
        b_nt=3 if 1 < bucket <= 256 else 0,
        work_shards=work_shards,
        external_grouping=bucket == 4 or bucket >= 256,
        external_counting=bucket >= 256,
        payload_chunk_rows=PAYLOAD_CHUNK_ROWS,
        payload_tile_ready=True,
        band_m=band_m,
    )


def _select_bounded_stage2(
    bucket: int, fixed_slot: bool, mtpr: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if not fixed_slot and mtpr > bucket:
        return Stage2Config(
            block_m=64 if sort_block_m == 128 else 32,
            block_n=128 if bucket == 256 and sort_block_m == 64 else 256,
            persist=True,
            persist_cu=240,
            use_nt=bucket <= 128,
            persist_strided=512 <= bucket <= 2048,
        )
    block_n = (
        256
        if bucket in (1, 4, 64) or bucket >= 1024 or not fixed_slot and bucket < 128
        else 128
    )
    if model_dim < 4096:
        block_n = 128
    persist = bucket >= 128
    return Stage2Config(
        block_m=64 if bucket >= 4096 else 32,
        block_n=block_n,
        persist=persist,
        persist_cu=128 if bucket == 256 else 240 if persist else 0,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
    )


def _select_large_stage2(
    bucket: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if bucket == 1024:
        persist_cu = 224
    elif bucket == 2048:
        persist_cu = 256
    elif bucket == 16384:
        persist_cu = 192
    else:
        persist_cu = 240
    block_n = 128 if bucket == 256 or model_dim < 4096 else 256
    return Stage2Config(
        block_m=64 if sort_block_m == 128 else 32,
        block_n=block_n,
        persist=True,
        persist_cu=persist_cu,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
        skew_cu=96 if bucket >= 512 else 0,
    )


@cache
def _select_bucket_config(
    bucket: int, mtpr_class: int, experts_per_rank: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    if mtpr_class == MAX_MTPR_CLASS:
        stage1 = _select_large_stage1(bucket, experts_per_rank, inter_dim, model_dim)
        stage2 = _select_large_stage2(bucket, stage1.sort_block_m, model_dim)
        # p2p_quant governs the STAGE2 COMBINE payload only -- the dispatch is
        # unconditionally fp8 (`mega_moe_stage1.py`: fz_nbytes = model_dim).
        # Upstream sends the large-mtpr class to "fp8_blockwise_1x32", which
        # halves the combine row (N_OUT + N_OUT//32 vs N_OUT*2) but quantizes
        # each expert's contribution BEFORE the cross-rank reduce. The M3
        # baseline reduce-scatters bf16, so fp8 here makes the A/B arms
        # numerically unequal: it is the whole reason megamoe's relL2 jumps
        # 0.0027 -> 0.0266 at mtpr >= 2048 while the baseline stays at 0.0033.
        # Dispatch-fp8 costs nothing against the baseline (aiter's fused_moe
        # quantizes activations to fp8 for GEMM1 anyway); combine-fp8 is a real
        # accuracy loss the baseline does not pay. Keep the combine in bf16 to
        # match, and use M3_MEGAMOE_P2P_QUANT=fp8_blockwise_1x32 to price it.
        return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")

    fixed_slot = mtpr_class <= FIXED_SLOT_MAX_MTPR
    if fixed_slot:
        stage1 = _select_fixed_stage1(bucket, experts_per_rank)
    else:
        stage1 = _select_bounded_stage1(bucket, mtpr_class, experts_per_rank, inter_dim)
    stage2 = _select_bounded_stage2(
        bucket, fixed_slot, mtpr_class, stage1.sort_block_m, model_dim
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    fixed_slot: bool | None = None,
) -> MegaMoEConfig:
    # mtpr_config_class thresholds and nearest_token_bucket bisects, so nothing
    # below this point depends on mtpr being a power of two.
    if mtpr <= 0:
        raise ValueError(f"mtpr={mtpr} must be positive")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    bucket = nearest_token_bucket(tokens)
    mtpr_class = mtpr_config_class(mtpr)
    # Whether the caller runs the fixed-slot dispatch layout is the caller's own
    # decision (MegaMoEM3 takes the compact one for every fused-shared batch, at
    # any mtpr). Inferring it from mtpr alone was right only while every batch at
    # or below 255 was fixed-slot: it rejects 192 and 200, whose nearest bucket is
    # 256, before the fused-shared path can claim them.
    is_fixed_slot = (mtpr_class <= FIXED_SLOT_MAX_MTPR) if fixed_slot is None else fixed_slot
    if is_fixed_slot and bucket > 128:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    if is_fixed_slot and experts_per_rank > 64:
        raise ValueError("fixed-slot supports at most 64 experts per rank")
    config = _select_bucket_config(
        bucket, mtpr_class, expert_config_class(experts_per_rank), model_dim, inter_dim
    )
    return _apply_stage1_overrides(config, bucket, mtpr_class, model_dim)


# The stage1 tables above (notably `_large_dispatch_cu`) are hand-tuned magic
# numbers measured on DeepSeek-V4 v4_pro: 48 experts/rank, model_dim 7168,
# topk 6, EP8. M3 is 32 experts/rank, 6144, topk 4 = world 4, so its dispatch
# moves markedly less traffic for the same token count. These env hooks exist
# to re-derive those numbers for M3 without editing the tables.
_STAGE1_OVERRIDE_ENV = {
    "num_dispatch_cu": "M3_MEGAMOE_DISPATCH_CU",
    "grid_mult": "M3_MEGAMOE_GRID_MULT",
    "work_shards": "M3_MEGAMOE_WORK_SHARDS",
    # The GEMM1 tile shape. Measured at M_r=8192, MegaMoE's fused GEMM1 is 62%
    # slower than aiter's standalone mfma_moe1 even with every synchronisation
    # cost removed, and aiter runs t128x128x256 against this path's 128x512x256.
    "tile_n": "M3_MEGAMOE_TILE_N",
    "sort_block_m": "M3_MEGAMOE_SORT_BLOCK_M",
    "num_waves": "M3_MEGAMOE_NUM_WAVES",
    # GEMM1 work-index order: m_tiles per reuse band (1 = upstream).
    "band_m": "M3_MEGAMOE_BAND_M",
}


_P2P_QUANT_ENV = "M3_MEGAMOE_P2P_QUANT"


def _apply_stage1_overrides(
    config: MegaMoEConfig, bucket: int, mtpr_class: int, model_dim: int
) -> MegaMoEConfig:
    p2p = os.environ.get(_P2P_QUANT_ENV)
    if p2p:
        # Both invariant dicts are pre-built in MegaMoEM3._build_fused_stage2
        # and mori sizes the combine buffer for bf16 (the wider row), so either
        # value is legal at any bucket. This is a real compile-cache key
        # (kernel_name carries p2p_quant_type), so runs may share a JIT dir.
        config = replace(config, p2p_quant=p2p)
    changes = {}
    for field, env in _STAGE1_OVERRIDE_ENV.items():
        raw = os.environ.get(env)
        if raw:
            changes[field] = int(raw)
    if not changes:
        return config
    stage1 = replace(config.stage1, **changes)
    stage2 = config.stage2
    if "sort_block_m" in changes:
        # stage2 is derived from stage1.sort_block_m -- re-select it, or the two
        # stages end up mismatched (the runtime asserts block_m must divide
        # sort_block_m). Overriding stage1 alone silently produced pairings the
        # selector would never emit.
        if mtpr_class == MAX_MTPR_CLASS:
            stage2 = _select_large_stage2(bucket, stage1.sort_block_m, model_dim)
        else:
            stage2 = _select_bounded_stage2(
                bucket,
                mtpr_class <= FIXED_SLOT_MAX_MTPR,
                mtpr_class,
                stage1.sort_block_m,
                model_dim,
            )
    return replace(config, stage1=stage1, stage2=stage2)
