"""Entry fee for a producer-consumer combine: device-scope partial-O traffic.

Both live designs for overlapping attn and combine -- a persistent combine
fed by a ready queue, and the cheaper last-arrival variant where the 16th attn
workgroup to finish a query does that query's combine inline -- need one thing
the current pipeline does not: a partial written by a workgroup on one XCD must
be visible to a reader on another, WITHOUT a kernel boundary in between.  L2 is
per-XCD on this part (section 8.1), so that means device-scope traffic:

    store cache_modifier=".wt" -> buffer_store ... sc0 sc1
    load  cache_modifier=".cv" -> buffer_load  ... sc0 sc1
    (verified against the ISA in nt_isa_probe.py, gfx950)

That is the entry fee, and it is paid on the whole 1074 MB of partial-O whether
or not the overlap ever materialises.  The prediction worth falsifying: it costs
about what nt cost, since both force partial-O past the per-XCD L2 -- and nt cost
attn +129 us at 16x1024.  If it does, the ideal prize for the producer-consumer
design (36 + 642.7 + 15.6 = 694 us against today's 863) shrinks to roughly what
plain request-chunking already offers, and neither design is worth writing.

".cs" (nt) is included as a calibration arm so the comparison is in-process.
Correctness is checked but these are cache-policy hints only: every arm must be
bit-identical.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_bench as pb
import kv_outer_dense as dn
import kv_outer_v3 as v3
import kv_outer_v4 as v4
import kv_outer_v6 as v6
from torch.profiler import profile, ProfilerActivity

REPS = 12
ROUNDS = 5
CHUNK_REPLAYS = 200
ARMS = [("base", "", ""), ("wt", ".wt", ""), ("wt+cv", ".wt", ".cv"),
        ("cs (nt)", ".cs", "")]


def dev(fn, pat):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total / REPS for e in prof.key_averages()
               if pat in e.key and e.self_device_time_total > 0)


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    return g


def time_replays(g, replays):
    tot = 0.0; done = 0
    while done < replays:
        c = min(CHUNK_REPLAYS, replays - done)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(c):
            g.replay()
        b.record(); b.synchronize()
        tot += a.elapsed_time(b); done += c
    return tot * 1000.0 / replays


for scheme in ["1x8192", "16x1024"]:
    reqs, tokens = (int(x) for x in scheme.split("x"))
    pc = pb.build_prefill_case(reqs, tokens, 90000, "random", pool=0)
    case = pc["case"]; msb = case["block_table"].shape[1]
    n = pc["query"].shape[0]; topk = pc["topk"].shape[-1]
    FP8MAX = float(torch.finfo(case["k_cache"].dtype).max)
    ws = dn.KVOuterWorkspace(n, topk, reqs, msb, pc["query"].device)
    bt = case["block_table"].view(-1); sel_flat = pc["topk"].reshape(-1)
    cfg = v3.v3p_config(4, 4)
    qrows = n * 16
    qf8 = torch.empty(qrows * 128, dtype=case["k_cache"].dtype, device="cuda")
    qs = torch.empty(qrows, dtype=torch.float32, device="cuda")
    v6.quant_q[((qrows + 63) // 64,)](pc["query"], qf8, qs, qrows, FP8MAX,
                                      HD=128, RPP=64, num_warps=4)
    dn.build_kv_index_dense(pc["topk"], pc["req_id"], ws, pc["abs_pos"])
    torch.cuda.synchronize()

    outs, fns = {}, {}
    for tag, st, ld in ARMS:
        o = torch.empty_like(pc["query"])
        outs[tag] = o

        def attn(st=st):
            v6.kv_outer_kernel_v6[(ws.nslot,)](
                qf8, qs, case["k_cache"], case["v_cache"], bt, ws.off,
                ws.slot_q, ws.slot_rank, ws.diag, pc["abs_pos"], ws.po, ws.pm,
                ws.pl, pb.SM_SCALE, case["scale"], case["scale"], msb,
                TOPK=topk, num_warps=4, FP8_MAX=FP8MAX, PV_FP8=True,
                MASK_FAST=True, QPREQ=True, PO_CM=st, **cfg)

        def comb(ld=ld, o=o):
            v4.combine_v5[(n,)](ws.po, ws.pm, ws.pl, sel_flat, o, TOPK=topk,
                                num_warps=4, PO_CM=ld)

        def pipe(attn=attn, comb=comb):
            dn.build_kv_index_dense(pc["topk"], pc["req_id"], ws, pc["abs_pos"])
            attn(); comb()
        fns[tag] = (attn, comb, pipe)

    tags = [t for t, _s, _l in ARMS]
    gs = {t: capture(fns[t][2]) for t in tags}
    for _ in range(200):
        for t in tags:
            gs[t].replay()
    torch.cuda.synchronize()
    a_us = {t: 1e9 for t in tags}
    c_us = {t: 1e9 for t in tags}
    rep = {t: 1e9 for t in tags}
    for rnd in range(ROUNDS):
        for t in (tags if rnd % 2 == 0 else tags[::-1]):
            at, cb, _p = fns[t]
            a_us[t] = min(a_us[t], dev(at, "kv_outer_kernel_v6"))
            c_us[t] = min(c_us[t], dev(lambda at=at, cb=cb: (at(), cb()),
                                       "combine_v5"))
            rep[t] = min(rep[t], time_replays(gs[t], 300))
    for t in tags:
        fns[t][2]()
    torch.cuda.synchronize()

    print(f"\n===== {scheme}  n={n}  po={ws.po.numel()*2/2**20:.0f} MB =====")
    print(f"{'arm':>9} {'attn':>8} {'d attn':>8} {'comb|attn':>10} "
          f"{'replay':>8} {'vs base':>8} | bitexact")
    for t in tags:
        bx = torch.equal(outs[t], outs["base"])
        print(f"{t:>9} {a_us[t]:>8.1f} {a_us[t]-a_us['base']:>+8.1f} "
              f"{c_us[t]:>10.1f} {rep[t]:>8.1f} "
              f"{rep['base']/rep[t]:>7.3f}x | {bx}")
    print(f"\n  producer-consumer ideal = index + attn + one request's combine")
    idx = 36.0 if scheme == "16x1024" else 25.0
    tail = c_us["base"] / (reqs if reqs > 1 else 8)
    for t in ("base", "wt", "wt+cv"):
        ideal = idx + a_us[t] + tail
        print(f"    with {t:>6} stores: {ideal:>7.1f} us   "
              f"vs today's {rep['base']:.1f}  -> {rep['base']/ideal:.3f}x")
    del pc, ws, outs, fns, gs; torch.cuda.empty_cache()
