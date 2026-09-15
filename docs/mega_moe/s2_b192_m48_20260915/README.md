# b192 S2 M48 entry

Add one complete M48 K loop for33–48 valid rows to the retained physical M64/N256/K256 eight-wave b192 S2. M32 and M64 entries remain. S1, selector geometry, A DMA, both routed/shared epilogues and P2P are unchanged. The measured routing has3/67/59 tiles in M32/M48/M64; this reduces theoretical total S2 MFMA work11.056%, including shared, without implying the same time reduction.

The direct third branch caused4 SGPR lane spills. Branch-order and accumulator-only alternatives still had3–4. The retained implementation specializes the exact b192 queue N-panel count to6144/256=24 and checks runtime dimensions at the host entry. This keeps201 VGPR,106 SGPR, zero SGPR/VGPR spills, zero private scratch and66,112bytes LDS, matching the production resource counts. HIP supports one8-wave CTA/CU. No epilogue tail48 optimization is included.

## Complete-forward paired measurements

| Comparison | Graph capture order | A, us | B, us | Time change | Faster pairs |
|---|---|---:|---:|---:|---:|
| M48 + constant N versus current production | candidate → baseline | 300.559 | 298.114 | -0.813% | 12/12 |
| M48 versus M32/M64 with same constant N | candidate → baseline | 301.327 | 299.756 | -0.522% | 11/12 |
| M48 + constant N versus production, reverse capture | baseline → candidate | 299.096 | 298.629 | -0.156% | 8/12 |

The same-N control isolates adding M48. The production comparison includes the supporting N constant specialization. Do not subtract percentages across runs to estimate N-only performance. These are quick paired measurements with the existing clock-instability waiver; strict stability qualification is false. They measure complete forward, not isolated S2 latency.

All three runs pass34 route/input probes, graph512, bit-exact arm outputs, independent math, full-rank queue terminal checks, source/dispatch identity and eight-rank CPU/GPU ISA checks. Each run has an independent raw-pair arithmetic verification. K3 reviewed the M48 scale chunks/single rowgroup, zero-padding, scope and N specialization; logs are in the experiment review folders. [Measurements](measurements.json) retain per-pair and per-rank times plus original-result hashes. [Audit](final_audit.json) records resources and the CPU-only rejected variants.

Only the measured stage2 source is copied into production. Root redo/report/handoff and the historical tuning ledger are unchanged. No separate unpatched production-process GPU rerun is claimed.
