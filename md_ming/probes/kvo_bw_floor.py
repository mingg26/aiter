"""What bandwidth does this card actually reach on trivial kernels?

Every bandwidth claim so far leaned on one read kernel whose shape was copied
from the combine (one contiguous 64 KB run per workgroup, serial inner loop).
That is not a standard bandwidth probe. Measure the textbook ones -- flat read,
copy, triad -- across dtype, block size and warp count, at a MALL-resident size
and well beyond it, and find the real ceiling before trusting any "we are at
89% of peak" statement.
"""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.profiler import profile, ProfilerActivity

REPS = 30
def dev(fn, pat):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS): fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total/REPS for e in prof.key_averages()
               if pat in e.key and e.self_device_time_total > 0)

@triton.jit
def k_read(src, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + i, mask=i < n, other=0)
    tl.store(out + tl.program_id(0), tl.sum(x.to(tl.float32), axis=0))

@triton.jit
def k_copy(src, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, mask=m, other=0), mask=m)

@triton.jit
def k_triad(a, b, c, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(c + i, tl.load(a + i, mask=m, other=0)
             + tl.load(b + i, mask=m, other=0), mask=m)

for dtype, name in [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    esz = torch.tensor([], dtype=dtype).element_size()
    for mb in [128, 2048]:
        n = (mb << 20) // esz
        a = torch.randn(n, dtype=dtype, device="cuda")
        b = torch.randn(n, dtype=dtype, device="cuda")
        c = torch.empty(n, dtype=dtype, device="cuda")
        print(f"\n=== {name}, {mb} MB "
              f"({'MALL-resident' if mb <= 256 else 'beyond MALL'}) ===")
        print(f"{'BLOCK':>7} {'warps':>6} {'grid':>8} | "
              f"{'read TB/s':>10} {'copy TB/s':>10} {'triad TB/s':>11}")
        for BLOCK in [1024, 2048, 4096, 8192]:
            for nw in [4, 8]:
                g = ((n + BLOCK - 1)//BLOCK,)
                sm = torch.empty(g[0], dtype=torch.float32, device="cuda")
                try:
                    tr = min(dev(lambda: k_read[g](a, sm, n, BLOCK=BLOCK,
                                                   num_warps=nw), "k_read")
                             for _ in range(2))
                    tc = min(dev(lambda: k_copy[g](a, c, n, BLOCK=BLOCK,
                                                   num_warps=nw), "k_copy")
                             for _ in range(2))
                    tt = min(dev(lambda: k_triad[g](a, b, c, n, BLOCK=BLOCK,
                                                    num_warps=nw), "k_triad")
                             for _ in range(2))
                except Exception as e:
                    print(f"{BLOCK:>7} {nw:>6} {g[0]:>8}  FAILED {str(e)[:40]}")
                    del sm; continue
                nb = n * esz
                print(f"{BLOCK:>7} {nw:>6} {g[0]:>8} | "
                      f"{nb/(tr*1e-6)/1e12:>10.2f} "
                      f"{2*nb/(tc*1e-6)/1e12:>10.2f} "
                      f"{3*nb/(tt*1e-6)/1e12:>11.2f}")
                del sm
        del a, b, c; torch.cuda.empty_cache()
