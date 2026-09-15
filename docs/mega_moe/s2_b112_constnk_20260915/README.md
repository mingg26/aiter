# b112 S2 BM32 N/K address specialization

The retained b112 BM32 kernel used runtime N/K address arithmetic and 14 SGPR lane spills. Specializing N to 6144 and K address strides to 3072 removes those spills while retaining the runtime K-loop bound, the same GEMM work, and the same P2P layout.

## Scope and implementation

Only the retained b112 EP8 H6144/I3072 SBM64 shared-L2 path enables this change: BM32/N256/K256, persistent cu128/qg5, band8, XCD jointtail scheduling, no local reduction, no schedule audit or K/N padding. The existing SBM64 rollout also requires 16 local experts/top4. The stride specialization additionally requires INTER_MAX == inter_dim. The host checks runtime H/I against 6144/3072 before launching.

Queue N panels become 24; A DMA and GEMM K address strides become constants. The GEMM helper receives fixed N and an optional fixed_k_stride flag, default False. Its K-loop trip count stays runtime. The new tag is `_b112bm32cnk2`.

S1, BM/BN/BK, four waves per CTA, 640 launched CTAs, epilogue, NT OFF, qdr OFF, and whole-empty-tile skipping ON are unchanged. This does not enable M16 or change any other batch's selector.

## Complete-forward measurements

| Initial compilation and graph capture | Baseline us | Candidate us | Paired time reduction | 95% interval for reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|
| Candidate then baseline | 261.0542 | 260.4563 | 0.2288% | 0.0618%–0.4315% | 10/12 |
| Baseline then candidate | 261.4487 | 260.8335 | 0.2354% | 0.0770%–0.3935% | 10/12 |

Both runs use the existing clock-instability waiver: strict_measurement_passed is false. They support retaining a small positive quick-screening result, not a strict stability claim. Do not combine samples across runs or subtract the null effect. Baseline self-pair ratios are 0.998779 and 0.998989, with intervals crossing 1 in both runs.

## Resources and validation

| Resource | Baseline | Candidate |
|---|---:|---:|
| VGPR | 231 | 231 |
| SGPR | 106 | 102 |
| SGPR lane spills | 14 | 0 |
| VGPR spills / private scratch | 0 / 0 | 0 / 0 |
| LDS bytes | 33088 | 33088 |
| HIP maximum resident CTAs / waves per CU | 2 / 8 | 2 / 8 |

Resource counts and HIP limits match on all eight ranks. These are resource residency limits, not measured achieved occupancy.

Each successful run passes 34 input/routing probes, graph512, bit-exact baseline/candidate and eager/graph comparisons, independent math, queue terminal-head checks, actual timed-graph identity, and all-eight-rank CPU/GPU S1/S2 ISA checks. The independent verifier reconstructs every timing sample from raw GPU-event chunks and rank maxima. CPU b120 and b192 rank0 ISA are byte-identical outside the new scope. The selector is byte-identical. The production files are copied byte-for-byte from the tested candidate. All 12 CPU configuration regression tests and both repositories' diff checks pass.

## Evidence and rejected attempts

- Candidate and first successful run: `/mnt/shared/homes/ming/msa/.scratch/b112_s2_bm32_constnk_20260915_v2`.
- Successful reverse run: `/mnt/shared/homes/ming/msa/.scratch/b112_s2_bm32_constnk_reverse_20260915_v2`.
- Each successful result is under `measure/gpu_v1/b112/result.json`, with `independent_review.json` beside it. Consolidated evidence is `review/final_measurements.json` in the first directory.
- Actual K3 design, implementation, reverse-harness, reference-fix, and final result reviews are saved in the corresponding review directories.
- Initial constant-K-trip version, `b112_s2_bm32_constnk_20260915_v1`, compiled to VGPR260 and was rejected before GPU. The retained version specializes address strides while preserving the runtime K loop.
- Reverse v1 passed the 34 probes but stopped before timing because the harness reconstructed nonexistent local CPU-ISA paths. Reverse v2 reads the frozen absolute per-rank references; all 24 reference paths and hashes were checked independently. No failed-run timing samples exist or are included.

Root redo/report/handoff documents and the historical configuration ledger are unchanged.
