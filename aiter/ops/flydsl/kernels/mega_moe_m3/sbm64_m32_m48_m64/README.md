# Experimental b136 SBM64 S1 with M48

This private experimental entry adds M48 to the M32/M64 N256 candidate.
The production selector remains unchanged. The older two-branch checkpoint
is retained in `sbm64_m32_m64`.

Physical tiles, ping/pong LDS, queues, and the shared epilogue stay at M64.
Valid rows <=32 use M32, 33-48 use M48, and 49-64 use M64. All three paths
cache current A before next-A DMA, retire B by K128 half, and load A scales
late. M48 uses ceiling division for its two logical scale groups, copies
48 rows with the SBM128 wave-uniform tail, and pads two accumulator vectors
with zero for the original M64 epilogue. No ni retirement is added.

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

Main comparisons use the original fastest ticket-fixed single M64/N512.
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
