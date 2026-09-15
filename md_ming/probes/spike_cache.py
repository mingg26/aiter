import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3


@gluon.jit
def k(p, q, N: gl.constexpr):
    lin: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    off = gl.arange(0, N, layout=lin)
    v = cdna3.buffer_load(ptr=p, offsets=off)
    cdna3.buffer_store(stored_value=v, ptr=q, offsets=off)
    cdna3.buffer_store(stored_value=v + 1, ptr=q, offsets=off, cache=".cg")
    cdna3.buffer_store(stored_value=v + 3, ptr=q, offsets=off, cache=".wt")
    cdna3.buffer_store(stored_value=v + 4, ptr=q, offsets=off, cache=".cs")
    w = cdna3.buffer_load(ptr=q, offsets=off, cache=".cv")
    cdna3.buffer_store(stored_value=w, ptr=q, offsets=off, cache=".wt")


a = torch.zeros(64, dtype=torch.int32, device="cuda")
b = torch.zeros(64, dtype=torch.int32, device="cuda")
c = k[(1,)](a, b, N=64, num_warps=4)
torch.cuda.synchronize()
for line in c.asm["amdgcn"].splitlines():
    t = line.strip()
    if t.startswith(("buffer_store", "buffer_load")):
        print("  ", t.split("//")[0].strip())
