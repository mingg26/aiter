# MSA prefill 线终态：Q-outer + XCD 分组 + fp8

> 范围：MiniMax-M3 稀疏注意力 **prefill** 方向的最终结论（kernel 级 + E2E 移植）。
> 有效性基准：分支 `opt/msa-prefill-xcd`（树 `vllm-msa-xcd`）五 commit，2026-09-05 定稿；
> E2E 移植基于六 PR 集成 commit `190f9137bd`，2026-09-06 测完。
> 来源：`handoff_kv_outer.md`、`handoff_decode_msa_pr.md`。

## 0. 方向终论：从 KV-outer 翻转到 Q-outer

2026-09-05 结论翻转。**旧结论「KV-outer 赢 1.06/1.02/1.19x」已推翻**，它建立在两个
不成立的前提上：

1. `topk_mode="random"` 是合成的 worst case：相邻 query 的 topk 重叠只有 **2.3%**
   （`16²/704`，可解析验证）。真实 indexer 的注意力是局部的。
2. 对照基线树 `vllm-msa-base @ e33de4a7af` **缺 fp8 compute commit `b6b4b61713`**，
   而 KV-outer 从第一天就是 fp8。旧表里 KV-outer 的 attn 优势有一大块只是这个。

**最终形态 = Q-outer（生产基线）+ XCD 分组 + fp8 compute**，五个 commit，全部逐位相同。

胜负完全由真实 indexer 的相邻 topk 重叠决定：KV-outer 在 local/random 两种分布下
几乎不动（923 vs 926 µs，它的 KV 已经只读一次，局部性没东西可给它），变的全是
Q-outer。判据阈值：**相邻重叠 >~10% 走 Q-outer + 分组，<~6% 留 KV-outer**。

> **未决（最高优先级）**：真实 indexer 的相邻 topk 重叠**从未量过**。这一个数决定
> 整条方向，是数据测量不是 kernel 实验。

## 1. 五 commit 阶梯（分支 `opt/msa-prefill-xcd`）

按落地顺序，全部逐位相同（kernel 测试每次 140 passed / 13 skipped / 10 failed，
10 个失败在干净树上一致）：

| commit | 内容 | 收益 | 要点 |
|---|---|---|---|
| `790bf5f59d` | 相邻 prefill token 放同一个 XCD | 1.09–1.64x | 纯 host 54 行，不碰 kernel；`_SPARSE_PA_XCD_PREFILL` 开关 + `_adjacent_xcd_run()` |
| `903ad19dad` | ragged 批也分组（padding 预算） | 多请求 1.51–1.71x（原来精确为 0） | 修 `790` 的洞：`num_seqs` 除不尽 `lcm(xcd_count,8)` 时旧版分组整个关掉 |
| `8eebc47430` | `waves_per_eu=4`，门在 `ONE_SHOT` | 1.05–1.08x | prefill vgpr 130→124 零 spill；不加门 decode splits 2/4 要 spill 16、亏 6–21% |
| `84bbe9a666` | 删 prefill 主循环的死 causal mask | 1.03–1.06x | guard 用已有 constexpr；random 分布下精确为 0 |
| `f9528e3f75` | 每个 KV 地址的不变项只求一次 | 1.03–1.05x | 机制是左结合使不变项之间的求和折不拢，不是不变项没提出去 |

### 1.1 累计效果（16x1024，ctx 90000，全流水线 attn + 表构建，同进程同数据）

```
                              attn    other    total    vs baseline
  local band=64（合成 25% 相邻重叠）
    baseline（bf16，无分组）   974.4     20.9    995.3      1.000x
    Q-outer 终态              573.4     20.4    593.8      1.676x
    KV-outer 原型             642.8    280.4    923.2      1.078x   -> Q 是它的 1.55x

  random（旧 KV-outer 数字用的分布）
    baseline                 1007.5     22.9   1030.5      1.000x
    Q-outer 终态              952.4     22.1    974.5      1.057x
    KV-outer 原型             637.6    288.3    925.9      1.113x   -> KV 反赢 1.05x
```

**1.676x 全部押在「真实重叠 >~10%」这个未量的数上。**

### 1.2 三条引用时必须带上的口径警告

1. grouped 臂是 gather 张量测的（64 MB query），置换开销没算；生产实现应在 kernel 里
   swizzle `program_id`（decode 线 `39a63fdda8` 已这么做，零数据搬运）。
2. fp8 vs bf16 的 relerr 是 **5.3e-2 – 9.2e-2**。拆 `sparse_pa.py` 里的 bf16 pin 是
   精度/产品决定，不是工程问题。该 pin 使 `790bf5f59d` 的收益从 1.36x 降到 1.23x。
3. 只在 `context=90000` 下测过，局部性模型是合成的。

## 2. XCD 分组机制：判据是 KV 足迹 vs 聚合 L2

**`790bf5f59d` 的 commit message 把机制写错了**（说是漂移/stride-8 邻居共享窗口），
纯数据测出来不是。机制（`xcd_gain_model.py` 证实）：

- **KV 足迹 = 请求数 × (context/128) × 32 KB**。1 个请求 ctx≈90k = 703 块 × 32 KB
  = **22 MB**。
- **8 个 XCD 的 L2 合计 ~32 MB（每 XCD 4 MB）**。足迹低于它时整个工作集本来就驻留，
  分组没东西可赢（1x2048/1x8191/1x12285 只有 1.01–1.09x）。
- 足迹远超它时收益大：16x1024（350 MB，10.9×L2）1.643x；9x1111（197 MB，6.2×）
  1.712x。
- **变量是足迹不是 token 数**：同样 16384 token，4x4096（88 MB）只有 1.19x。
- **变量也不是请求数**：同一个单请求 ctx 90k→200k，足迹 22→49 MB，收益从
  **0.968x 变 1.118x** —— 长 context 单请求 prefill 也受益。

机制证据（local band=64，16x1024）：分组后 L2 命中 24.8%→93.3%，DRAM 实读
6283.7→501.3 MB（6.34 TB/s 顶到带宽墙 → 0.61 TB/s）。

**ragged 分组（`903ad19dad`）**：launcher 把 grid.x 补齐到 `lcm(xcd_count,8)*run`，
run 越短 padding 粒度越细，而分组收益在 run 远小于最大值时已饱和 —— 「用 run 长度
换 padding」几乎免费。在 padding 预算 **1/8** 内取最长 run：

```
   shape     num_seqs   run    pad     before → after    gain
  16x1024      16384   2048   0.0%     909.2 → 553.4    1.643x
  15x1023      15345   1024   6.8%     851.2 → 543.5    1.566x   <- 原来是 1.000x
   9x1111       9999    256   2.4%     543.0 → 317.1    1.712x   <- 原来是 1.000x
  13x1265      16445    256  12.1%     851.8 → 565.7    1.506x   <- 原来是 1.000x
```

**预算是必须的**：9x1111 pad 到 22.9% 掉到 1.454x；13x1265 pad 99.3% 掉到 1.435x。
padding 超限收益崩塌。1/8 在四个 shape 上选中三个实测最优、第四个差 0.7%。

## 3. fp8 规律：收益跟「离内存下限多远」

分组后 0.61 TB/s，离带宽墙 10 倍，kernel 变 VALU-bound，fp8 才开始值钱：

```
16x1024 attn µs        bf16      fp8    fp8 gain
  random  natural    1019.6    979.3    1.041x   <- 访存受限，省下的发射被藏住
  local   natural     983.1    839.5    1.171x
  local   grouped     798.8    610.2    1.309x   <- 92% 兑现
```

发射预算（PMC，16x1024 grouped）：CVT 省 117.2 µs + 其余 VALU 33.9 + MFMA 29.8
= **预测 181.1 µs，实测 167.4 µs，92% 兑现**。机制：gfx950 的 fp8 MFMA 每条吃
两倍 K（MFMA 指令数 −50%），CVT 指令 −92%（71.7M→5.6M）。

与 decode 同一条规律（b32/b64/b128 = 2.69×/1.45×/1.23× 离下限 →
+26.8%/+15.9%/+1.6%）。**访存受限时 fp8 收益趋零，别在 random 分布上期待它。**

## 4. 其余三个 commit 的要点

- **`8eebc47430`（waves_per_eu=4）**：4 waves/SIMD 要 ≤128 VGPR，prefill
  specialization 是 130——只差两个。后端自己挤到 124 零 spill。门必须开在
  `ONE_SHOT`（= `num_splits<=1`）：decode splits 2/4 的 specialization 要 141 寄存器，
  挤到 128 得 spill 16，亏 6–21%；加门后回到 1.000–1.023x。5/8 waves  spill
  45/101，0.52–0.68x / 0.19–0.28x，**4 是唯一可用目标**。
- **`84bbe9a666`（删死 causal mask）**：`is_causal = query_length > 1`，prefill 传 1
  所以 `IS_CAUSAL=False`；块表已把因果性编码进去（已注意区间是连续前缀，唯一部分块
  在最后），vLLM 循环又剥出了最后一个 partition —— 前 7 个 partition 每轮白算
  24 条指令（cps=256 主循环少 81 条）。收益 1.03–1.06x，**发射→时间转化率只有
  ~23%**（不是 fp8 那次的 92%）。guard = 已有 constexpr（`ONE_SHOT and not
  IS_CAUSAL and SLIDING_WINDOW <= 0 and CONTEXT_PARTITION_SIZE_PER_BLOCK == 1`）；
  **splits>1 时同删会错到 relerr 8.3e-2–1.9e-1**，guard 不是装饰。
- **`f9528e3f75`（地址不变项折叠）**：诊断先于改动——28 条地址加法对 20 条 load，
  5 条不喂任何 load。不变项早就提出去了，**没提出去的是不变项之间的求和**：
  源码 `ids*stride + t2 + t3 + t4` 左结合把变化项排最前，不变对永远折不到一起。
  改后每地址只剩一条加法，且省寄存器（124→122 prefill，141→137 decode split）。
  decode 在 chooser 实际会选的档全部 ≥1.000；强制 b256 splits=4 稳定退 2.2%，但
  chooser 在 b256 选 s2，且关掉融合 reduce 后同点是 1.000x——**那不是这个改动**。

## 5. prefill 移植 E2E（`handoff_decode_msa_pr.md`）

手工 port（**不 cherry-pick**）`790bf5f59d`/`903ad19dad`/`f9528e3f75` 三个 commit
的行为到 `vllm-msa-prefill` 树，分支 `opt/msa-prefill-grouped-fp8`，基于六 PR 集成
commit `190f9137bd`。正确性：grouped+FP8 vs ungrouped 逐位相等（XCD sibling groups
4–8、16）；padding-budget selector 26/26；address-folding 16/16。

**E2E（natural-client A-B-A，TP4，60K–120K ctx，89.926% prefix hit，并发 8，1 输出
token）**：

| 指标 | 值 |
|---|---|
| uncached-prefill 吞吐 | 26,775.5 → 27,073.2 tok/s，**+1.112%**（保守口径，含一个晚段低样本） |
| candidate 中位 | +1.726%；前三紧密集平均 +1.736% |
| p50 TTFT | −1.883% |
| baseline pre/post 漂移 | −0.151% |
| GPU profile MSA device 时间 | 997.73 → 734.69 ms/rank，**−26.36%**（该 profile 早于 padding-budget selector） |
| MSA / client wall | 7.10% → 5.32% |

- address folding 单独证为 **E2E 中性**（+0.034% ~ +0.099%，p50 TTFT −0.160%）；
  晚段低点在 address folding 之前就存在，`f9528e3f75` 没有引入它。
- 环境变量 **`M3_SPARSE_PA_XCD_PAD_BUDGET`** 覆盖 padding 预算，**默认 12.5%**。
- 口径：这是 **90% prefix-hit 异步自然文本增量/cached-prefill** 结果，不能引为
  固定形状或 0%-hit 冷 prefill 的数字。

## 6. 已关闭方向（一行级，防重做）

- **KV 加深流水**：八种做法全负——跨回边 phi home 每轮 32 条寄存器拷贝消不掉；
  手工展开 vgpr 130→511（1 wave/SIMD，慢 2.5×）；`loop_unroll_factor` 是空操作；
  内联汇编 elementwise 够不着寄存器分配器。唯一未否入口是内层循环整体下沉手写 AMDGCN。
- **cps**：256 已最优；512 使 vgpr 189 越过 170 的 3-wave 悬崖（2 waves/SIMD，亏 11.6%）。
- **combine（KV-outer 侧）**：已结案，84% 是纯 po 搬运，同形状访存尺跑不赢它；
  写回争用 ~33 µs 是 DRAM credit 背压（nt 能消但原地转移给 attn）。
- **分块 / 双流 attn+combine 重叠**：全负；双流形态 B 的跨 XCD `.wt/.cv` 入场费
  +296 µs > 收益 236 µs，理想值为负（见 `trick_fused_reduce.md`）。
- **fp8 partial-O / atomic 浮点累加消灭 combine**：产品/确定性原因出局，别再提。
