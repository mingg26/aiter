# Retained SBM64 defaults — 2026-09-15

Default S1 uses physical SBM64 with M32/M48/M64 compute branches, N256/K256, for exactly **64, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192** tokens per rank. Scope: EP8, H6144/I3072, 128 experts, top4, fused shared L13/L2 with XCD, full configured local batch, no local reduction and no explicit Stage1 environment overrides. b72 remains SBM32/N512.

The measured kernel reads current A before the next A DMA, retires B by K128 half and loads A scales late. M48 uses ceiling-divided scale groups and bounds the last 16-row A copy. The shared physical M64 epilogue skips empty rows48–63 and rows32–47 in both SwiGLU/LDS production and quantization, with unconditional barriers. The rejected M48 scale-trim and b72 SBM64 experiment are excluded.

S2 retains each measured configuration: b64 BM64/N128 with NT; others BM32/N256 without NT. Empty-half skip is ON except b64 (BM==SBM) and b152. b120/136/144 enable it only in the validated rollout scope.

## Full-forward comparison against each previous fastest

| Batch | BA latency reduction | AB latency reduction | AB previous fastest → candidate (µs) |
|---:|---:|---:|---:|
| 64 | 4.260% | 4.970% | 265.256 → 252.073 |
| 104 | 4.895% | 4.775% | 271.975 → 258.988 |
| 112 | 4.587% | 3.914% | 272.313 → 261.657 |
| 120 | 4.470% | 4.603% | 277.977 → 265.182 |
| 128 | 4.537% | 4.207% | 278.001 → 266.306 |
| 136 | 4.689% | 3.890% | 285.297 → 274.200 |
| 144 | 4.374% | 4.701% | 291.158 → 277.469 |
| 152 | 3.786% | 3.583% | 295.397 → 284.814 |
| 160 | 4.619% | 4.016% | 298.672 → 286.676 |
| 168 | 3.694% | 4.217% | 303.601 → 290.797 |
| 176 | 3.783% | 3.163% | 301.745 → 292.201 |
| 184 | 2.783% | 2.623% | 306.570 → 298.528 |
| 192 | 2.098% | 2.424% | 309.476 → 301.974 |

26 runs / 312 paired samples favor the retained candidate; every within-run 95% ratio CI is below one. These are paired quick-screening results, not strict stability certification or a guarantee for every routing distribution. [Measurements](measurements.json) include per-run ratios, CIs, null controls, paired critical times, configuration and resource records; raw result paths/hashes identify the immutable original evidence.

b136 uses the original ticket-fixed M64/N512 baseline from before this entire SBM64 round. Its mid16-versus-last16 incremental comparisons are excluded from the cumulative table. The mid16 guard was retained by user decision; its incremental stable speedup remains unresolved. b64 uses the older faster ticket-fixed M64 rather than the slower clean e84 default. b160/168/176/184 controls use their previous M16 MFMA plus shortened-A-DMA loop; b192 uses its previous M16 MFMA loop with full A DMA.

The b72 trial was rejected: previous fastest SBM32/N512 versus SBM64 gave +0.411% / +0.048% latency; the latter CI includes parity. No b72 candidate kernel, quant rename, row-table extension, or S2 whitelist extension is included. Raw rejected evidence remains under `.scratch/b72_sbm64_vs_best_20260915_v2`.

## Integration validation

All 104 native-import CPU S1 code objects match the measured candidate ISA byte for byte. b64 uses 200 VGPR, all other retained batches 202; all use 106 SGPR, zero spill/scratch and 45056 B LDS. Source migration changes only relative imports; the measured sweep already contains the 13-batch assertion list. [CPU ISA identity](native_cpu_isa.json).

The actual selector passes 5371 comparisons, changing only the 13 intended full-batch configs (tile_n and sbm64_path); unmeasured shapes, partial batches, b72 and explicit environment overrides retain prior behavior. Nine CPU regression tests pass, including S2 skip boundaries and SBM128 defaults. [Selector audit](selector_audit.json).

Native default GPU validation and final K3 review are recorded separately in `native_gpu_validation.json` and `final_audit.json`. This is integration/correctness validation with no selector, launcher or quantization monkeypatches; it makes no new performance claim. Kernel source files are frozen throughout GPU validation.

Old specialized full/M16 and the older experimental two-branch sources remain as historical/manual entries; the 13 selected defaults use only `sbm64_m32_m48_m64`. Frozen experiments are never edited. Root report/handoff/redo and the live configuration ledger may advance under user authorization; reruns must create fresh source manifests instead of bypassing old hash gates.

Final validation scope, explicitly requested by the user: all **104 native-import S1 ISA objects match**. Supplementary native GPU runs passed for **b64/104/112/120** (16 probes, graph512, all-eight-rank S1/S2/quant ISA each). The other nine batches were not rerun on GPU. Their prior two-direction measurements, CPU ISA identity and selector audit remain the evidence. A b128 launch failed at TCPStore port53128 before workers/kernel execution and was not retried. This is not a claim of a new 13-batch native GPU sweep.
