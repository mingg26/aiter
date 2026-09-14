# Experimental b136 SBM64 S1

This module preserves the b136 N256 M32/M64 branch work. It is an explicit
experimental S1 entry, and is not selected by the production defaults.

The physical M tile remains 64. Tiles with up to 32 valid rows use M32 compute
and shortened A DMA; the remaining tiles use M64. N256 uses 24 N panels and
three local slots per XCD queue. The original M64 epilogue and ticket fix are
retained. Scope: EP8, 128 experts, H6144/I3072, top4, 136 tokens per rank,
fused shared L13/L2, and the frozen b136 scheduling settings.

The low-level entry is:

```python
from aiter.ops.flydsl.kernels.mega_moe_m3.sbm64_m32_m64.mega_moe_stage1 import (
    run_mega_moe_stage1,
)
```

Use the existing fused S1 launch arguments with `sort_block_m=64`, `tile_n=256`,
`tile_k=256`, and eight waves. This is not a replacement for the complete
route/quant/S1/S2/combine forward or a new default configuration.

M64 current A is read before next-A DMA. Both branches use K-half B retirement
and late A scales. M64 ni-group retirement was tested separately and rejected:
it was slower by 0.280% and 0.277% versus this M64 schedule.

The M64-current-A candidate beat the original fastest ticket-fixed single
M64/N512 baseline by 1.622% and 1.850% in two direct, balanced full-forward
comparisons (12/12 faster pairs each). These are quick-screen results, not
strict stability certification. That earlier whole-kernel change also altered
M32 wait placement, so it is not a pure M64 runtime attribution.

Current checkpoint also moves M32 current-A reads before DMA. Its source
changes are only two constexpr gates and a new kernel tag. All eight offline
ISAs pass at 202 VGPR / 106 SGPR, zero spill/scratch; M32 has eight A reads
before DMA and the intervening vmcnt(0) is removed. All eight M64 steady loops
match the previous checkpoint. Two complete-forward comparisons against the original M64 baseline passed:
287.067 -> 280.563 us (2.266% faster) and 287.025 -> 280.136 us
(2.400% faster), each 12/12 faster pairs. The direct incremental comparisons
against the previous A-first candidate were 0.311% and 0.165% faster, with
12/12 and 8/12 faster pairs respectively. The reverse-run same-graph null
ratio was 0.998572, so the small incremental effect needs caution: these are
quick-screen results, not a strict stability or production promotion claim.
No null subtraction or subtraction of separate-run means is used.


`validation.json` records source identity, per-run ratios and confidence
intervals. Generated ISA, traces, virtual environments, and build caches are
not source files. The package's import paths differ from the frozen experiments;
its non-import AST is checked against those exact candidate sources.
