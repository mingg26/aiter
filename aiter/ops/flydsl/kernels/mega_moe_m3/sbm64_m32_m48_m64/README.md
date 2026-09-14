# Experimental b136 SBM64 S1 with M48

This private experimental entry adds M48 to the M32/M64 N256 candidate.
The production selector remains unchanged. The older two-branch checkpoint
is retained in `sbm64_m32_m64`.

Physical tiles, ping/pong LDS, queues, and the shared epilogue stay at M64.
Valid rows <=32 use M32, 33-48 use M48, and 49-64 use M64. All three paths
cache current A before next-A DMA, retire B by K128 half, and load A scales
late. M48 uses ceiling division for its two logical scale groups, copies
48 rows with the SBM128 wave-uniform tail, and pads two accumulator vectors
with zero for the common M64 epilogue. Empty rows48-63 now skip both
SwiGLU/LDS production and quantization under matching guards; barriers
remain unconditional. No ni retirement is added.

Scope: b136, EP8, H6144/I3072, top4, 128 experts, shared L13/L2,
physical SBM64, N256, K256, eight waves and the frozen b136 scheduling setup.
The entry is `sbm64_m32_m48_m64.mega_moe_stage1.run_mega_moe_stage1` using
existing fused S1 launch arguments; this is only the S1 entry within the
complete route/quant/S1/S2/combine operator.

All eight ranks compile at 202 VGPR / 106 SGPR, zero spill/scratch,
45056 bytes LDS. M32/M64 instruction and wait sequences match the previous
candidate after register identifiers are erased; register allocation changed.
All four runs pass 16 correctness probes including 33/48/49 rows, graph512,
independent math, source identity, and all-eight GPU/offline ISA equality.
K3 reviewed the design, implementation and all four results. After import
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
