# b192 S2 M64 with M32/M64 entry branches

Only the existing full-batch EP8 fused-shared b192 scope (H6144/I3072/E128/top4, no local reduction or explicit Stage1 tuning) selects BM64/N256/K256, eight waves, grid256. S1 remains SBM64 `m32_m48_m64`. Other batch selectors retain their configurations.

Each tile chooses one complete M32 or M64 K loop at entry. A payload descriptors exclude padding; M32 reads one A-scale chunk and computes32 rows. Both routed and shared epilogues guard the upper32 producer and consumer together, with CTA barriers outside the guard. The physical LDS layout remains M64. B retains its two-stage prefetch. b192 shared is three full M64 tiles per rank. In the measured routing, 129 routed tiles choose M32 three times (valid rows32/30/3) and M64 126 times.

## Paired complete-forward measurements

| Comparison | Capture order | A mean, us | B mean, us | Change in time | Faster pairs |
|---|---|---:|---:|---:|---:|
| M64 versus production | candidate → baseline | 302.206 | 298.750 | -1.144% | 12/12 |
| M64 versus production | baseline → candidate | 302.198 | 300.323 | -0.620% | 12/12 |
| Late B versus retained early B | candidate → baseline | 300.252 | 301.646 | +0.464% | 0/12 |

Retain early-B M64; reject late-B. The two construction orders both favor M64, with different effect sizes (about0.6–1.1%). These are quick paired full-forward measurements under the existing clock-instability waiver, not strict stability certification or isolated S2 latency. The runs are reported separately; no null subtraction or cross-run averaging. See [raw paired evidence](measurements.json).

## Resources and validation

- S2 VGPR231→201; SGPR106; SGPR lane spills14→0; VGPR spills0 and private scratch0 in both arms. LDS33,088→66,112bytes.
- HIP resource limit: two four-wave CTAs/CU→one eight-wave CTA/CU; eight resident waves/CU supported in both. This is a resource limit, not measured achieved occupancy.
- Each completed run passed34 route/input probes, graph512, exact output comparison, independent math, full-rank queue terminal checks, source and actual dispatch identity, eight-rank CPU/GPU ISA identity and independent pair arithmetic verification.
- Final cleanup preserves all eight measured candidate ISA hashes. 5,306 selector comparisons change exactly one full-batch default (b192); offscope b136 S2 ISA is identical. Both repositories pass12 existing CPU regression tests with updated b192 expectations. See [integration audit](final_audit.json).
- Initial run stopped before timing because an inherited b200-only S1 `_lp` name check rejected the correct b192 S1. The replacement checks the exact precompiled b192 S1 name and retains eight-rank ISA checks. Original failure remains under `.scratch/b192_s2_m64_branch32_64_20260915_v1`; it contributes no performance sample.
- K3 (`inferact-kimi-k3`, session`session_ebbbe6de-6172-4d40-b346-b3d01ac8313b`) reviewed the dual-entry implementation and the late-B experiment. Logs remain in the experiment review directories. Static wait instructions do not establish actual stall duration; no pipeline cause is inferred from them.

Final source cleanup was checked by ISA identity and selector tests; no separate unpatched production-process GPU rerun was performed. Root redo/report/handoff documents and the historical tuning ledger were not changed.
