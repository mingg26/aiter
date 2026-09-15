# Retained S2 defaults, 2026-09-15

EP8 shared H6144/I3072/top4, full batches 120–184 step 8 now select the previously measured M64/N256/K256, qg2 candidates. Existing b192 M32/M48/M64 fixed-N path is reused. S1 and other defaults remain unchanged.

Validation: 12 CPU regression tests in each tree; 37 default configurations; 1312 off-scope/override checks; 88 CPU S2 ISAs (new nine shapes, b192 and b112, all eight ranks) exactly match historical GPU-verified references after kernel-symbol normalization. Zero spills/private storage in these S2 artifacts. K3 reviewed the rollout and required these identity checks. No new GPU run for this rollout.

Historical gains are single diagnostic capture-order observations, 0.235%–2.067%; they do not prove stable/global optimality. Each measured candidate passed 34 correctness/routing probes and graph replay. The b128 gain is particularly small. The user's retained-default policy accepts measured improvements; the kernel algorithms are unchanged here.

See default_audit.json, all_rank_isa.json and prior_measurements.json for exact configurations, hashes, resources and measurement provenance. Workspace restart instructions: md_ming/redo.md.
