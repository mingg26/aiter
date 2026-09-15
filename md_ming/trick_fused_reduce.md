# Split 归约正确性协议终稿（gfx950）

> 范围：split-K / split-KV / staging 归约的发布-订阅正确性协议、gfx950 内存语义、
> graph replay 免清零手法。MSA decode fused-reduce、MoE split-K epilogue、local_reduce
> 三条线共用同一套机制。
> 有效性基准：aiter mega_moe_m3 最终 commit `f7a96515`；MSA decode 线
> `opt/msa-decode-xcd` / 六 PR 栈（2026-09 上旬）。
> 来源：`trick_fused_split.md`、`HANDOFF.md`、`handoff_decode_msa_pr.md`、
> `plan_m32_membership.md`、`tool_root_cause.md`、`handoff_kv_outer.md`。

## 0. 发布链（必须完整，少一环静默错）

没有跨 workgroup 屏障（那东西不存在）。协议是「每个生产者自己保证写完，再用原子
计数把 S 份保证攒起来」：

```
wave       : s_waitcnt vmcnt(0)     "本 wave 的写落地了"     ← per-wave 指令
   ↓ s_barrier 组合（必须在 wait 之后）
workgroup  : "所有 wave 的写都落地了"（partial 由 4 个 wave 分写）
   ↓ 单 lane 发 atomic 到达计数（counter == S）
S 个 split : "S 份 partial 都落地了"                        ← 不是会合，是推断
   ↓
最后到达者从 L2 读并归约，写出最终结果，复位计数器
```

- **`s_waitcnt` 是 per-wave 的**，`s_barrier` 是 workgroup 级的。**barrier 放在 wait
  之前组合的是空的**（修过的 bug：`ddc5041019`）。
- **失效形态不是 NaN，是读到上一个 step 留在 scratch 里的 stale partial**——数值
  完全合理，所以重放相同输入的投毒测试抓不住（两次读到同样的数）。
- **L2 不是同步层级，是介质。** 被淘汰写回 HBM 也读得到；同 XCD 只有一块 L2，不存在
  不一致。「保证还在 L2」和正确性无关。
- **留给 AMD 的开放假设**：`vmcnt(0)` 归零的 store 已在本 XCD 的 L2 里，同 XCD 的
  其他 CU 能读到。未获 AMD 正式确认；要 formal sign-off 就去问这一句。

## 1. 同 XCD 前提（fused reduce 的成立条件）

- XCD = `linear mod 8`（mod 2/4 不成立）；是旋转 `[6 7 0 1 2 3 4 5]` 不是恒等，
  **代码只能依赖同余**。
- `grid.x` 是 8 的倍数 ⇒ 一个 sequence 的所有 split **100% 同 XCD**；不是 8 的倍数
  ⇒ 0%。**padding 是硬要求**，padded workgroup 必须在任何 tensor/semaphore 访问前退出。
- XCD 数运行时查 `hipDeviceAttributeNumberOfXccs`，padding 取 `lcm(查询值, 8)`。
- 同 XCD 成立时，partial 从不离开读它的那块 L2，**不需要任何缓存维护指令**——这是
  fused reduce 免费的全部原因。

## 2. gfx950/CDNA4 内存语义（定位内存类问题必备）

| 事实 | 含义 |
|---|---|
| agent-scope release lowering = `buffer_wbl2 sc1` + `s_waitcnt vmcnt(0)` | **实测 42.8 µs**（每个 workgroup 刷整个 XCD L2，还会挤掉别人在读的 tile）——所以发布侧绝不用 agent fence |
| `CPOL_COHERENT = 0x11`（sc0\|sc1） | store write-through 到 L2 一致性点，不污染本地 cache；split-K partial 用它，waitcnt(0) 等完即完成 release |
| 跨 XCD staging store 用 `cache_modifier=0x12`（nt sc1，LLVM CPol: SC1=16, NT=2） | device scope（SC1=1）走 coherent bypass；wave-scope store（SC1=SC0=0）**被 waitcnt 等完也不升级**到跨 XCD/L2 可见 |
| `buffer_inv` 裸指令在 gfx950 是 **NOP** | 「cached load + 裸 inv」≠ NT load，是错的；读侧用 NT load（cache_modifier=2，绕过 CU cache 但可命中共享 XCD L2）或同一 CPOL_COHERENT load |
| `buffer_inv sc0 sc1` | system acquire：使 CU cache 与 L2 非一致性行失效，**不是清空所有 L2** |
| `s_waitcnt 0xF70`（vmcnt=0, expcnt=7, lgkmcnt=15） | 只覆盖本 wave；跨 wave 可见性要另行论证 |
| `tl.atomic_add` 默认 `sem="acq_rel"` | 改 `relaxed` 后 16x1024 count kernel 23.7→9.7 µs（**2.44x**）。device-scope 原子对争抢不敏感、对顺序语义极敏感 |
| atomic scope 总规则 | GPU 内一律 agent scope monotonic（`atomic_add_agent`，`communication_ops_utils.py:235-237`）；跨 rank（MoRI shmem）才 system scope |

**Triton cache modifier → AMD 位映射实测表**（`probes/spike_cache.py`，已随本目录迁移；
改 cache 策略前用它把 modifier 编出来再看 ISA 落成什么位——名字和位不是一对一）：

| Triton | AMD 位 |
|---|---|
| store 默认 / `.cg` | 无位 |
| store `.wt` | `sc0 sc1` |
| store `.cs` | `sc0 nt` |
| load `.cv` | `sc0 sc1` |
| load `.cg` | `sc0 nt`（Triton 把 `.cg` 降成 `nt`） |
| store `.cv` | 不接受 |

**别用 `.cg` 给读侧「加固」**：它被降成 `nt`（非临时驱逐提示，不是 bypass），而归约
要读 `max_logits` 三遍、`exp_sums` 两遍，方向是反的。

## 3. 不用 fence 时怎么把 waitcnt 穿进 atomic 的排序

atomic 本身用 relaxed 就够，但要防它被提到 wait 之前：用
`tl.inline_asm_elementwise("s_waitcnt vmcnt(0)\n\tv_mov_b32 $0, $1", "=v,v", [token])`
发那一条指令，**把 atomic 的地址数据依赖到穿过 asm 的 `token` 上**，硬件/编译器都
无法挪动（`pa_decode_gluon.py:213-231` `_release_partials`）。

## 4. Graph replay 免清零三法

host 侧 buffer 只分配一次（`lru_cache` + `torch.zeros` 一次），之后绝不在 host 清零：

| 方法 | 用在哪 | 要点 |
|---|---|---|
| winner 用 `atomic_add(-split_k)` **减回** | GEMM split-K（`splitk_epilogue.py:153-158`） | **不要 store 0**：下一 launch 的 early increment 可能被抢先到达，plain store 0 会把它抹掉 |
| 单调 64-bit ticket，`// grid_x` 取 generation | mega_moe stage1 generation/parity 计数器（`mega_moe_stage1.py:383-464`） | 计数器跨 replay 单调增、永不清零；host 与 kernel 各算一遍周期，不一致会**静默 epoch 漂移**（无崩溃无报错） |
| 每 slot 每次 launch **无条件覆写**（含不活跃 partition：存 `-inf` max / `0.0` exp_sum / 零输出） | 两 kernel 方案的 MSA decode（`pa_decode_gluon.py:1596-1612`） | reduce 侧 `-inf` 权重自然为 0；scratch 按 shape 缓存从不清零，靠 stream 序保证 |

对照：attention fused-reduce 里 winner 用 plain store 0 复位是**安全的**
（`pa_decode_gluon.py:1783-1788`）——一个 sequence 的 slot 在本 kernel 结束前不会被
下一次 launch 碰到（单流串行）；GEMM 场景下一 launch 的 early increment 可能抢先，
必须 atomic 减。**「何时能 store 0、何时必须 atomic 减」的现成对照就在这两处。**

## 5. 数值契约

partial 输出 **BF16**，归约累加 **FP32**，最终输出 BF16/query dtype。fused 与独立
reduce 保持同一契约，融合不降模型精度；winner 分支内的全局 softmax 刻意与独立
reduce kernel 的 FP32 运算顺序一致（`pa_decode_gluon.py:1698` 注释）。

原子浮点累加代替归约**出局**：产品要求 determinism，atomic 求和顺序不确定。
（整数 atomic 只用来数到达，partial 由 winner 一个人按固定 `part_idx` 顺序归约，
确定性没问题。）

**`split_k > 1` 路径明确禁止融合 bias/activation**，代码直接拦
（`aiter/ops/flydsl/kernels/preshuffle_gemm.py:170-171`）：

```python
    if split_k > 1 and _has_epilogue:
        raise ValueError("split_k > 1 does not support fused bias or activation")
```

原因是非线性 epilogue 必须作用在**已归约**的结果上，而 split-K 的每个 partial 都只是
部分和。正确落点是归约之后的独立 kernel（现成样例：`silu_and_mul_fq.py`，读已归约的
`tmp_out` 做 gate 激活 + 量化）。想在 split-K GEMM 里省一趟 epilogue kernel 的念头到
这里就该停——这不是 guard 太严，是数学上不成立。

## 6. 仓库零 `buffer_wbl2` 原则与代码位置

**这个仓库没有任何一处用 `buffer_wbl2`**。发布侧核心手法两种：

- **(a) store 自带 sc0|sc1 + waitcnt**：store 的 cache policy 即 release，
  `s_waitcnt(0)` + `gpu.barrier()` 后单 lane 发 atomic。
  代码：`aiter/ops/flydsl/kernels/splitk_epilogue.py`（`:15-18` CPOL_COHERENT 定义与
  注释、`:93-104` 到达计数、`:123` 读侧同一 policy、`:153-158` 减回复位）、
  `aiter/ops/flydsl/kernels/preshuffle_gemm.py:781-786`（store→waitcnt→barrier→
  epilogue 的调用点）。
- **(b) 普通 store + inline-asm waitcnt 穿数据依赖 + relaxed atomic**：
  `vllm/third_party/aiter/pa_decode_gluon.py:213-231`（`_release_partials`）。
  前提是同 XCD pin（见 §1）。

跨 XCD staging：`mega_moe_m3/local_reduce.py` `scatter_local_reduce`（`:39-183`）——
staging store 跨 XCD 时 `cache_modifier=0x12`，读侧 NT load + `fence_agent_acquire()`；
其注释（`:153-155`）：「`buffer_inv` 在 gfx950 是 NOP，别用 cached load + invalidate
替换 NT load」。两 kernel 边界方案（kernel 边界即 fence）：`mla_reduce.py`、
`mxfp8_gemm_kernel.py`。

ISA 审计顺序（正确性只能靠在生成 ISA 上核，不能靠投毒测试）：
**staging stores → vmcnt(0) → publication atomic → atomic 结果完成 → winner loads**。
只数 barrier 个数或看源码顺序不够。投毒测试要带阳性对照，且灵敏度是部分的——
它能抓「算错」，抓不住「有竞态」。

⚠ megamoe S2 local-reduce 的诚实记录：staging store 升 sc1 + 去 release 的变体
（`b8192-sc1-v3`）曾**全部门禁通过一轮**——400-replay 正是它通过的门禁之一。随后在
`warm-graph-trace-v1` 里出现失败，而且**失败的是那一轮里未改动的生产 kernel**
（collision changed replay 29、rank1 exact oracle mismatch；该轮没有产出 warmed
计时/trace）。**原因未定位，SC1-without-release 不被接受为可靠正确**。

两条可复用教训：①「先通过后失败」以后来的失败为准，不能引用先前那轮的通过；
②**间歇性正确性失败不一定落在你改的那个 kernel 上**——失败出现在对照臂/未改动路径，
不构成"我的改动没问题"的证据，反而说明整条发布链的竞态窗口还开着。

## 7. 反面终论：跨 XCD `.wt/.cv` 入场费 > 收益

跨 workgroup、无 kernel 边界地读别人的 partial，在 L2 每 XCD 私有的机器上必须走
device scope（store `.wt`→`sc0 sc1`，load `.cv`→`sc0 sc1`）。实测（16x1024 prefill
attn/combine 重叠，`probes/kvo_scope_ab.py`，已随本目录迁移）：

```
16x1024          attn    Δattn   replay   vs base
  base           587.0     +0.0    874.4   1.000x
  wt             882.9   +296.0   1169.7   0.748x
  wt+cv          882.7   +295.7   1285.1   0.680x
  cs (nt) 校准   594.1     +7.1    876.3   0.998x
```

**+296 µs 的入场费比整个 prize（236 µs，理想 1.37x 上限）还大，且在任何重叠发生
之前就要付。** last-arrival 变体内存序要求一模一样，同一道门槛。attn/combine 重叠
这条线**关闭**——不是门槛高，是理想值本身为负。

同一教训的 decode 版（`plan_m32_membership.md` §C.3，M32 s2 @ union 16）：
`acq_rel` 原子 42.8 µs（每 workgroup 一条 `buffer_wbl2`）、`.wt/.cv`+relaxed
31.4 µs（wbl2 归零但 partial 被迫穿透内存）、**什么都不加（靠同 XCD）28.8 µs**。
最优就是「靠同 XCD L2，什么都不加」。
