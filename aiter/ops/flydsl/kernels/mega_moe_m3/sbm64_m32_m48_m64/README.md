# Retained SBM64 M32/M48/M64 defaults

The measured entry is now selected for b64/104/112/120/128/136/144/152/160/168/176/184/192 under the validated EP8 fused-shared scope. b72 stays SBM32. See [the rollout report](../../../../../../docs/mega_moe/sbm64_20260915/README.md) for the 26-run comparison, scope, S2 choices and integration validation.

The implementation retains both last16 and middle16 producer/consumer guards. The middle16 incremental benefit remains unresolved; its user-approved retention is distinct from the cumulative gains against the pre-round fastest sources. The rejected scale trimming and b72 candidate are not present.

## Historical b136 development record

The following sections describe their original experimental status and comparisons, before the 13-batch default rollout above.

### Initial M48 entry

This private experimental entry adds M48 to the M32/M64 N256 candidate.
The production selector remains unchanged. The older two-branch checkpoint
is retained in `sbm64_m32_m64`.

Physical tiles, ping/pong LDS, queues, and the shared epilogue stay at M64.
Valid rows <=32 use M32, 33-48 use M48, and 49-64 use M64. All three paths
cache current A before next-A DMA, retire B by K128 half, and load A scales
late. M48 uses ceiling division for its two logical scale groups, copies
48 rows with the SBM128 wave-uniform tail, and pads two accumulator vectors
with zero for the common M64 epilogue. Empty rows48-63 and rows32-47 skip both
SwiGLU/LDS production and quantization under matching >48 and >32 guards;
barriers remain unconditional. No ni retirement is added.

Scope: b136, EP8, H6144/I3072, top4, 128 experts, shared L13/L2,
physical SBM64, N256, K256, eight waves and the frozen b136 scheduling setup.
The entry is `sbm64_m32_m48_m64.mega_moe_stage1.run_mega_moe_stage1` using
existing fused S1 launch arguments; this is only the S1 entry within the
complete route/quant/S1/S2/combine operator.

All eight ranks compile at 202 VGPR / 106 SGPR, zero spill/scratch,
45056 bytes LDS. M32/M64 instruction and wait sequences match the previous
candidate after register identifiers are erased; register allocation changed.
The middle16 candidate passes 16 correctness probes including 32/33/48/49
rows, graph512, independent math, source identity, and all-eight GPU/offline
ISA equality in both capture orders. K3 reviewed the design, implementation,
direct comparisons and diagnostic controls. After import
path migration, rank6 ISA is byte-identical to the GPU-tested source.

The following historical M48-addition comparisons used the original
fastest ticket-fixed single M64/N512; the latest epilogue comparisons below
use the retained M48 candidate directly.
Adjacent comparisons use the preceding M32/M64 botharbd candidate.
Each comparison uses 12 balanced pairs; reverse suffixes denote actual graph
construction order. Values are full-forward microseconds.

| Run | A us | B us | Faster |
| --- | ---: | ---: | ---: |
| m48_ba | 286.201 | 275.307 | 3.806% |
| m48_ab | 286.244 | 275.529 | 3.743% |
| adjacent_ba | 280.329 | 276.188 | 1.477% |
| adjacent_ab | 280.751 | 276.395 | 1.552% |

These are quick-screen results, not strict stability certification or a
production promotion. Null ratios and confidence intervals are preserved in
`validation.json`; no null subtraction or separate-run mean subtraction is used.

## Last16 quantization experiment

Rejected: skipping quantization of empty rows 48–63 did not pass the two-direction incremental retention rule.
The producer, barriers and all three steady K-loop instruction texts were preserved.
All four correctness and GPU/offline ISA checks passed at 202 VGPR / 106 SGPR, zero spill.
Primary comparison is previous M48 (adjacent_ba/ab). Original M64/N512 runs
are retained only as cumulative historical reference.

| Run | A us | B us | B/A | Null B/A |
| --- | ---: | ---: | ---: | ---: |
| qtail_ba | 286.642 | 275.435 | 0.960903017 | 0.997354196 |
| qtail_ab | 286.252 | 275.116 | 0.961098000 | 0.998471586 |
| adjacent_ba | 276.031 | 275.585 | 0.998385768 | 1.000301311 |
| adjacent_ab | 275.766 | 275.439 | 0.998812411 | 0.997864474 |

Full ratios and confidence intervals are in `validation.json`. Quick screen only;
no null subtraction or production default promotion.

## Last16 producer and consumer epilogue guards

Retained: skip empty rows48-63 before SwiGLU and LDS writes, and skip
the same rows during quantization. The optional high group is produced first.
Both comparisons below directly use the previous fastest M48 candidate.
No kernel defaults are promoted. All8 ranks remain at202VGPR/106SGPR,
zero V/S spill,zero scratch,LDS45056; physical register assignment changes
while steady-loop opcode/constants/waits match after register IDs are erased.

| Run | A us | B us | B/A | Null B/A |
|---|---:|---:|---:|---:|
| epi_ba | 275.255 | 273.898 | 0.995071762 | 0.999858997 |
| epi_ab | 275.973 | 274.804 | 0.995764664 | 0.999718887 |

Both16-probe/graph512/all8 GPU-ISA/source checks pass. These are quick
screening results; confidence intervals and self-null diagnostics are in
`validation.json`. No null subtraction or causal branch-cost claim.

## Additional middle16 producer and consumer guards

Retained by user decision after correctness and diagnostic reviews.
The original two-direction performance criterion was not met, so a stable
speedup is not claimed. The last16 checkpoint remains the performance control.
Both comparisons directly use the previous fastest last16-guard candidate
(63091d668). All8 ranks remain202VGPR/106SGPR,zero V/S spill,zero scratch,
LDS45056; register allocation changes while steady-loop opcode/constants/
waits match after register IDs are erased.

| Run | A us | B us | B/A | Null B/A |
|---|---:|---:|---:|---:|
| mid_ba | 275.307 | 274.913 | 0.998571555 | 1.000285655 |
| mid_ab | 274.296 | 273.972 | 0.998815769 | 1.001477624 |

Both16-probe/graph512/all8 GPU-ISA/source checks pass. These are quick
screening results; confidence intervals and self-null diagnostics are in
`validation.json`. No null subtraction or causal branch-cost claim.

The follow-up no-skip control changes only two ISA compare immediates
(all 8 ranks). At equal work, the new structure changes forward latency by
+0.079% / -0.023%; enabling the skip gives nominal gains of 0.166% / 0.046%.
All four confidence intervals include zero change. No substantial structure
penalty has been demonstrated, and the small gain is not yet resolved.
The middle16 implementation is retained in this experimental entry.
See `validation.json` for the explicit retention basis and uncertainty.
