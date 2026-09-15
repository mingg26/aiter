# M32/M64 成员分派（membership-partitioned）：稀疏 PA decode 的分组重做与终局

范围：MiniMax-M3 稀疏注意力 decode 的 M32/M64 分组线——旧并集方案为何错、成员分派
设计、实测终局档位表、softmax 成本结论，并并入 plan_manual_overlap 的 K 预取与
stage3/4/5 终局。
有效性基准：2026-09-03/04（`plan_m32_membership.md` 第三/四轮记录、`plan_manual_overlap.md`
重启检查点）。**整条线随后被 XCD swizzle 全面超过**（`HANDOFF.md` 第 3 节：
swizzle 拿全 4 路去重且无 tile 浪费，输出逐位不变），分支 `opt/msa-decode-m64`
不要再动；本文是它的终局档案与机制结论。
来源文件：`plan_m32_membership.md`、`plan_manual_overlap.md`、`report_claude.md`（§8）。

背景形状：TP4 rank-local 16Q/1KV/head_dim 128，topk 16×128=2048 KV/query，KV FP8。

## 1. 旧并集方案为什么错

旧 M32 =「取并集 U，铺一个 M32 tile」。三个症状是**同一个根因**：

> tile 的行集合（并集里的所有 token）≠ 块的成员集合（真正选了这块的 token）。

1. **语义错**：kernel 只有位置因果 mask，没有「这块是不是我选的」一层 → token 会
   attend 只有兄弟选中的块。加 membership mask 治不了：load 和 MFMA 在 mask 之前
   已经花掉了。
2. **`ext` 因果裁剪无条件生效**（独立 bug）：`ext + qk_column_offsets < context_length`
   只在「self block 被选中且排在末尾」时才对；真实 top-k 下 self block 被选中概率
   ≈1%（200K ctx 里 16/1563），此时裁剪打在任意更老块上丢真实 key（隔离实测 max
   relerr 7.0e-2/7.3e-2）。合成「最新 16 块」top-k 恒选中最新块，所以这个缺陷在此前
   所有测量里从未显形。
3. **性能崩**：零重叠下实测 0.64x（M32）/0.40x（M64）——分组是拿计算换访存，
   `访存倍率 = |U|/(g×topk)`、`计算倍率 = |U|/topk`；零相关下 M16/M32/M64 实际搬运
   量几乎相同（268.4/267.3/264.4 MB），分组一个字节没省却付了全部多出来的 tile 计算。

## 2. 成员分派设计

正确不变量：**一个 tile 里的每一行，必须都是这个块的真实选择者。**

一个请求的 g 个投机 token 选块集合 B_0..B_{g-1}（|B_i|=topk=16），并集中每块 b 有
成员集 S_b。g=2（相邻配对 (t0,t1)、(t2,t3)）三趟：

| 趟 | 块集 | tile | 行 |
|---|---|---|---|
| I | B0 ∩ B1 | M32 | t0, t1 |
| P0 | B0 \ B1 | M16 | t0 |
| P1 | B1 \ B0 | M16 | t1 |

设 o = |B0∩B1|：

```text
块读次数 = |U| = 32 − o          （最优，等于 KV-outer 的访存下限）
MFMA 行数 = 2o + (16−o) + (16−o) = 32   （恒等于 M16，一分不多）
```

| 方案 | 块读 | tile 行数 | 数学 | o=0 时 |
|---|---|---|---|---|
| M16 | 32 | 32 | ✓ | 基准 |
| 并集 M32（旧） | 32−o | 2(32−o) | ✗ | 0.64x |
| **成员分派 M32** | **32−o** | **32** | **✓** | **= M16（无断崖）** |

单调：o=0 退化为今天的 M16，o=16 访存减半。收益公式（g=4、topk=16、b128）：
**增益 ≈ 64/|U|**。

正确性靠结构而非 mask：

- 并集**升序排列**让每个 token 自己的块天然是它选中的最后一个 → §1 的 ext 裁剪 bug
  结构上不存在。
- `max_logits` 初值必须 `-3.0e38`（大于掩码值 `-3.4e38`；用 `-inf` 会把空转行
  `exp2(0)=1` 当概率）。
- `causal_mask` 里带着 split 区间上下界（`>= sequence_start_idx` 且
  `< sequence_end_idx`），覆盖它会错 39%（`8226652b43` 修回）。

落地（`opt/msa-decode-m64`，6 个 commit）：并集表达从独立 builder 两张表演进到
**每 token 4 个 int32 位图**（`810a16c7a8`）→ 并集表进**寄存器**（`797b58bf61`，
BlockedLayout 让 64 条目在 4 warp 各存一份，选基页是纯 wave 内 6 步 DPP 归约）→
整个 merge 搬进 **attention kernel 的 prologue**（`dee5b6dec5`，表不进内存）。
验收工具：`merge_check.py --sort-ref`（参考路径喂同样升序块号后才能要求逐位一致）。

## 3. 终局档位表

各档最优 split，union 25（`shared_blocks 12`）、q4、ctx 200064、fp8、含 reduce，
kernel trace 计时（第四轮 prologue 压缩后）：

| batch | split | M16 (µs) | M32 (µs) | M32 vs M16 |
|---:|---|---:|---:|---:|
| 16 | s8 | 11.121 † | 15.680 | **0.71x** |
| 32 | s8 | 16.440 | 18.040 | **0.91x** |
| 64 | s4 | 20.840 | 22.600 | **0.92x** |
| 128 | s2 | 38.961 | 31.001 | **1.26x** |
| 256 | s1 | 77.681 | 53.641 | **1.45x** |

† b16 的 M16 臂逐 slice spread 2.3%、跨 5 轮在 11.12–12.04 之间抖；引用时带区间
（0.71x–0.76x）。

**b≥128 赢（1.20–1.46x），b≤64 输 0.71–0.92x；交叉点在 b64–b128。**

## 4. 为什么小 batch 输：不是 prologue，是 tile 浪费

prologue 是每 workgroup 固定成本（第四轮压缩掉 2.2–2.5 µs：四个 all-pairs 矩阵、
七次 axis=1 归约 → layout 修正/删矩阵 3/每 token 标量化/提前 gather/去两次转置），
压掉后各档涨 0.09–0.11 个身位，但 b≤64 仍输。真约束是 **M32 的实际带宽只有 M16 的
55%**（tile 浪费 2×）：

| batch | M16 搬运 MB | M16 TB/s | M32 搬运 MB | M32 TB/s | M32 有效带宽 |
|---:|---:|---:|---:|---:|---:|
| 16 | 33.5 | 4.82 | 19.9 | 1.74 | 2.92 |
| 32 | 67.1 | 5.48 | 39.8 | 2.85 | 4.79 |
| 64 | 134.1 | 6.44 | 79.7 | 3.53 | 5.93 |
| 128 | 268.2 | 6.88 | 159.4 | 5.14 | 8.65 |
| 256 | 536.5 | 6.91 | 318.8 | 5.94 | 10.00 |

读法：M16 在 b128 撞到 ~6.9 TB/s 天花板；M32 只搬 59% 字节，等效带宽可超物理上限，
交叉点正是实测翻转位置。b64 对照：并集省掉 41% 访存（134.1→79.7 MB）但 M32 只跑到
3.53 TB/s（M16 6.44），时间 22.6 vs 20.8 µs——就算 prologue 免费也输。
**M32 在 b128 实际带宽 5.14，离 6.88 还差 25%**；推到贴墙是 23.2 µs、对 M16 1.68x
（实测 1.26x），剩下的肉全在那里。

### 占用约束

两条独立门槛联立：

- **占用**：`wg/CU ≥ 2` 是硬要求（见 §5）；M32 的 `num_groups = batch×2`，256 CU
  ⇒ **`batch × splits ≥ 256`**。
- **融合 reduce**：只在 splits ∈ {2,4}（s8 时 1/8 workgroup 各扛 8 份 partial，
  亏 1.8–2.3 µs）。

⇒ 只有 b≥128 能同时拿到满占用和免费归约（b128 靠 s2 融合，b256 靠 s1 无需归约）；
b≤64 必须 splits≥4，b16 连 s8 也只有 1.0 wg/CU。

## 5. softmax 成本结论

探针：整段 softmax 拆掉（P=S）vs 只去 exp2，其余逐字不变。b128/q4（attn µs）：

| 路径 | 达到带宽 | base | P=S | softmax 代价 | 只去 exp2 |
|---|---:|---:|---:|---:|---:|
| M16 s1 c256 | 6.68 TB/s | 40.161 | 39.441 | 1.8% | — |
| M32 s2 c128 | 5.97 | 22.480 | 21.440 | 4.6% | **0.0%** |
| M64 s4 c128 | 3.21 | 20.880 | 15.521 | **25.7%** | — |

（shared=12 同向：M16 2.0%、M32 4.1%、M64 29.8%；逐 token 路径上整段 softmax 也只值
b64 −4.5% / b128 −2.8%。）

两条结论：

1. **贵的不是 exp2，是跨 wave 的 max/sum 归约和它的 barrier**——M32 上只去 exp2 是
   0.0%。机制：`qk_mfma_layout`/`pv_mfma_layout` 都是 `warps_per_cta=[1,4]`（上游
   aiter 原始选择），4 个 wave 切 KV 列、**一行的 max 横跨 4 个 warp**；一轮 9 个
   `s_barrier`，P 又必须过 LDS 喂 PV MFMA。M32 一轮 457 条指令中 215 条连续指令
   （47%）零 MFMA——**整个 CTA 要么在 MFMA 相要么在 softmax 相，同 CTA 内不可能
   互相掩盖**，唯一掩盖来源是同 CU 的第二个 CTA，这就是 **`wg/CU ≥ 2` 是硬门槛**、
   M64 不开 split（0.5 CTA/CU）必崩的原因。
2. **代价暴露程度 = 离自己内存墙多远**：M16 贴墙全藏住，M64 在 3.21 TB/s 全暴露。

否定结论（勿重做）：

- `instr_shape=[32,32,16]`：MFMA 条数减半但 `convert_layout` 落到 LDS，实测
  −18%~−26%；在重写 `define_layout` 的 `register_bases` 前这条轴关闭（且 kernel
  本就不是 MFMA 发射受限）。
- `warps_per_cta=[4,1]`（softmax 变 wave 内）：−25%——K 每轮要广播给 4 个 warp，
  换边一样贵。

### 边际成本锚点（直接引用，别再推）

| 单位 | 代价 |
|---|---:|
| 一个 32×32 all-pairs 矩阵 + 归约 | 0.20 µs |
| 一次 `merge_row`↔`merge_col` 转置 | 0.12 µs |
| 空 kernel dispatch 地板 | 4.08 µs（独立 reduce 总 4.28 µs 里归约本体只有 0.2 µs） |

## 6. plan_manual_overlap 并入：K 预取与手工重叠终局

### K 预取 mask（阶段 0，已落地）

主 kernel 的 K 预取 `key_tensor2 = gl.load(...)`（`pa_decode_gluon.py:2058`）**无掩码、
每轮无条件发射**：末位 split 读 page 0、非末位读邻居 split 的合法页。浪费 = 每 CTA
固定 1 个 tile 的 K 流量 = 1/T：

| 配置 | T | K 浪费 | K+V |
|---|---:|---:|---:|
| s1 | 8 | +12.5% | **+6%** |
| s2 | 4 | +25% | **+12.5%** |
| s4 | 2 | +50% | **+25%** |
| s8 | 1 | +100% | **+50%** |

旁证：DRAM 实读 80.4 MB ≈ 逻辑 ×1.20（b32q4/s4），与 +50% K（×1.25）吻合——L2 没
兜住浪费，DRAM 真付了钱。修法：谓词 `has_next = partition_idx + STEP < end` +
masked load。**改后必须重扫 split chooser 断点表**（高 split 相对变快）。此项即
decode 线的「削掉死的 K 预取」commit `0539a1304e`（M16 上 +0.4%）。

### stage3/4/5 否定终局：墙在 HBM

手工 MFMA/VALU 重叠（把 softmax 滑进相邻 tile 的 MFMA 排空时间）按决策门关闭。
唯一有效 shape：b128/q4、16 个 KV page 完全独立随机、CPS=128、split=1：

| 配置 | 中位数 (µs) | 相对 stage2 |
|---|---:|---:|
| stage2 | 47.759 | 1.000x |
| stage3 | 53.501 | 1.120x |
| stage4 | 56.597 | 1.185x |
| stage5 | 61.528 | 1.288x |

K 的 4 条 `ds_read_b128` 与 QK MFMA 的软件流水交错已实现（`SQ_WAIT_INST_LDS`
−20.3%），所以损失**不是** LDS round trip；ATT 显示 stage3 最大残余是 K 消费处的
`s_waitcnt vmcnt(1)`——**残余瓶颈是 K 的 HBM 到达依赖**。加 stage 没有赢，墙在 HBM。

同一线其他否定（勿重做）：

- `s_setprio` 调优先级 ±2%（极性选错 −6%），不改任何决策。
- `sched_group_barrier`：所有可达 Triton（3.7.1/3.8.0/pytorch-triton-rocm/main）都没
  这个符号，`pa_decode_gluon` 里 14 处调用全是 no-op。
- 把 future K 的 `global_load_lds` 提到 softmax/P 前：58.9 µs——CDNA4 把同 wave 的
  普通 LDS 与 direct-to-LDS 排序，反而把 K HBM latency 放上 P critical path。
- 8 个 scalar page-table load 去重物理页：VMEM 473k→596k，54.5 µs，已删。
- 用 constexpr 开关做 early/late 同进程 A/B 无效——constexpr 本身改变 lowering 和
  指令调度。
