# MegaMoE 机制级结论（S1/S2 kernel 行为规律）

> 范围：MiniMax-M3 MegaMoE（EP8 / MI350X gfx950 / H6144/I3072/E128/top4）S1/S2 kernel 的可复用机制结论：sort-block 规则、S2 tile 几何规则（BM == SBM）、NT 单向判据、task 阈值与成本模型、split-K 否决、发布链 ordering、VGPR/占用结构、PMC 事实、FlyDSL 坑、S1 微优化清单。
> 有效性基准：aiter-ming-amd-m3-megamoe @ f7a96515（2026-09-15）。文件内代码行号基于当时生产树，重查时以函数名/符号为准。
> 来源：`handoff_megamoe.md`、`option_megamoe.md`、`summary_megamoe.md`、commit f7a96515 内 `sbm64_m32_m48_m64/README.md`。

## 1. sort block 必须装得下一个 expert

**不是"块越小 padding 越少越好"——反了。** S1 把每个 expert 的路由行补齐到 `sort_block_m`(sbm) 的整数倍，tile 数 = `Σ_e ceil(rows_e / sbm)`。`rows_e` 是随机量（EP8/top4/128 experts 下均值 T/4，σ≈√均值）；当均值落在块大小的 1–2σ 内，"这个 expert 要不要第二个 tile"成了掷硬币，**各 rank tile 数不再相同**；每个 tile 要完整走一遍该 expert 的 37.7 MB GEMM1 权重，tile 数即 S1 成本，整层由最慢 rank 决定。

T=104（均值 26 行 vs 32 行块）实测：

| | 每 rank tile 数 | S1 每 rank µs | 整层 |
|---|---|---|---|
| sbm=32 | 17,19,18,19,19,17,17,17 | 155 **193 193 191 190** 157 161 156（跨度 24%） | 313.2 µs |
| sbm=64 | **16 ×8** | 155 159 154 157 155 157 152 157（跨度 4%） | **290.0 µs** |

**sbm=64 多算近一倍 padding 行反而快 23 µs**：padding 行读写已被界掉、无 HBM 流量，只花 MFMA 周期，而 S1 远非矩阵管受限（b256 实测 MFMA 占用 18.7%）。

规则落在 `mega_moe_config.small_sort_block_m`：

```
64 if (mtpr % 64 == 0 or mtpr > 96) else 32
```

逐点复现既有 6 个条目（16/32→32、64→64、96→32、128/256→64）。E[padding] ≈ sbm/2 每 expert，与 batch 无关。

## 1.5 S2 的 tile 收益要同时看计算量和调度

旧 BN128/W4、固定 M64 实验只说明当时几何的取舍，**BM==SBM 不是普遍的收益必要条件，也不保证更快**。当前120–192已采用 BN256/W8、256 CTA，并在 tile 入口选 M32/M48/M64 完整 K-loop；b112 同样满足 BM==SBM 的候选仍比 BM32 慢0.531%，保留 BM32。

b112 的48行在 BM32下是32+16；M16可以少算部分MFMA，却不同比例降低权重读取、队列调度和整层耗时。33/183非空子块可用M16，理论总MFMA行数省9.016%，不是可直接兑现的9% S2加速。shared/routed隔离都能受益，但合并工作不可相加。

最近诊断：512 CTA把no-tile开销降约20%，完整S2无稳定改善；epilogue M16在exact16隔离省0.681%，完整S2仅0.041%、CI跨零。因此都不进默认。数据和范围见 [report_s2.md](report_s2.md)。

## 2. NT（non-temporal）开关判据：R_S1 / R_S2

B 权重的 cache 策略（S1 `b_nt`、S2 `use_nt`）不用试，直接算。定义 R_B = 一份可复用 B panel 被几个输出 tile 读：

```
R_S1 = ceil( (8*T*topk/E) / sort_block_m )    # 每 expert 行数 / S1 M tile 高度
R_S2 = sort_block_m / block_m                 # 与 T 无关（S1 补齐使 S2 看到的行数恒 = sbm）
```

完整判据是"拿 B 的复用换 A 的驻留"：开 NT ⟺ `(R_B-1)·B_total·1[S_B≤C_L2] < (R_A-1)·A_total`；本模型 B/A ≈ 100–200 倍，塌缩成 **开 NT ⟺ R_B == 1**。

**但这个判据是单向的，别写成处方**——它可靠的那一半是排除：

```
R_B == 2  →  一定不要开 NT（稳定亏 4.4–10.6%，可直接跳过不试）
R_B == 1  →  才有可能赚，但收益不保证 —— 必须实测，且按"一个点不成立"的口径至少 4 对交错
```

方向预测（亏/不亏）本轮全对，含一次预先预测成功：b64 把 S2 `block_m` 32→64 使 R_S2 2→1，
`use_nt` 由亏 5.3% 变赚 +0.74%（同 batch 同 SBM 只动 BM）。**但"R_B==1 就该开"是误读**：
R=1 的四个点里只有两个赚，另两个是平的。

| 档 | R_S2 | `use_nt` 实测 |
|---|---:|---|
| b16 (SBM32/BM32) | 1 | 持平（噪声内） |
| b32 (SBM32/BM32) | 1 | **+0.8%**（3 次确认） |
| b96 (SBM32/BM32) | 1 | **持平**（4 对交错：NT 慢 0.35%，符号 2 正 2 负，组内跨度是均值差的 3 倍） |
| b32 (SBM64/BM32) | 2 | **−4.4%**（同 batch 只动 SBM，结论翻转） |
| b64/b128/b256 (SBM64/BM32) | 2 | −5.3% / −8.0% / −10.6% |
| b64 (SBM64/BM64) | 1 | **+0.74%**（预先预测成功） |

**所以：用它省掉 R_B==2 那些档的实验预算，不要用它去改 selector 默认。** 按"R=1 就开"
批量改 selector 会白花 GPU 时间。

**幅度不由 R 决定，机理未定位**：R=1 时收益随 batch 衰减到零（b32 +0.8%→b64 +0.74%→b128/256 ≈0）；R=2 时损失随 batch 增大。按 `(R_B-1)·B_total/8.94TB/s` 估 R=2 代价 33.8 µs，实测 17.7/25.1/40.1 µs（b64/128/256），同量级但小 batch 偏低。**R 只用来定开关，不用来预测收益。**

ISA 佐证：`b_nt=2/3` 时 S1 44 条 buffer_load 中 32 条带 nt；`b_nt=1` 一条没有。复用只是启发式（同 XCD 相邻 ticket、无顺序保证），不是硬保证。

## 3. S1 task 阈值模型

```
S1 ≈ (N × t) / 256 + 一个尾段的时长
```

- N = 总 task 数 = `12 × (routed_tiles + shared_tiles)`，t = 单 task 时长。
- claim 是 atomic 工作池不是整轮同步（`mega_moe_stage1.py:734-762`），尾段代价是**一个 task** 不是一整轮。
- **池恒 = `num_cu × grid_mult` = 256**：planner 和 dispatch producer CTA 发完 task 就流入消费循环（"Control CTAs join the work pool after dispatch"），**改 `num_dispatch_cu` 改不了池大小**。

T=200 rank4 长尾实例：一般卡 240 task < 256 够用；rank4 因两个 expert 超行多 24 个 routed tile → N=264 溢出 8，S1 = 199.4 µs vs 其他卡 167–175，整层被它拖住。ladder T=184–248 证实：spread 走出一条抛物线 3.5% → 4.3% → **19.3% → 22.3% → 16.5% → 18.2% → 25.3%** → 9.7% → 8.0%——**少数卡越界时 spread 最大（越界段约 16.5–25.3%），全部越界后反而回落**；T=192 是该段加速比峰值。判读要点：spread 突然变大是"部分 rank 越界"的指纹，不是噪声变大；它回落也不代表问题解决了。

## 4. task 成本结构与 split-K 恒亏

**关键机制**：合并 4 个 task 只慢 **2.2×** 而非 4×；切分 1 个 task 成 2 片，每片接近 **1×** 而非 0.5×——同一常数从两个方向各测到一次：**一个 task 的时间里约一半不随工作量缩放**（A 的 LDS staging、流水预热、epilogue、barrier）。task 既不能合并也不能切分。

split-K 恒亏（T=200 实测分解，均为单变量对照）：

| 项 | 代价 |
|---|---:|
| ticket 空间翻倍 | ~0 |
| K 切分 + partial 往返 | +24.6 µs |
| agent-scope fence（`buffer_wbl2` lowering） | +22 µs |
| 尾段收益 | 11 µs |

单价：切一个 task 1.03 µs、一次 agent-scope fence 0.46 µs（每 task 两个半片 → 0.92）；溢出 8 个 task、tile 粒度下切 12 → **成本 ≈23 µs 对收益 11 µs，恒亏**。对症版 `spk1` 实测 rank4 199.4→213.7 µs（外推 210，吻合）。

**同 XCD 两难（split-K 无法翻盘的结构原因）**：要缩短尾段，split 必须落在最后被领取的 task 上，而那些 task 恰是队列抽干后被其他 XCD 的 CTA 轮转拿走的；挪到 claim 最前能拿同 XCD，但那样就不在尾部。decode 侧能免 fence 是因为 `grid.x` 为 8 的倍数可**造出** 100% 同 XCD，本 kernel 的动态工作池没有等价杠杆。`buffer_wbl2`（agent-scope release 的 lowering）实测 42.8 µs / +22 µs，禁用。

## 5. 发布链 ordering

正确性链（split 归约/dispatch 发布通用）：

```
每 wave s_waitcnt vmcnt(0)  →  s_barrier（必须在 wait 之后）→  单 wave 发 atomic  →  最后到达者归约
```

少一环静默错——失效样子是读到上一个 step 的 stale partial，数值完全合理，不是 NaN。**ordering 的正确性只能靠 ISA 审计，不能靠 replay 门禁**：反复 replay 同一输入读到同样的数，投毒测试探测的是空的。独立实证：去掉 fence 的变体 8 个 rank 在第一个数学检查就全错——跨 XCD 是真实发生的，不是理论风险。

## 6. 1 CTA/CU 是 VGPR 卡住的

gfx950 每 SIMD 512 VGPR，CTA 8 waves = 2 waves/SIMD，2×256=512 占满 → 1 CTA/CU（占用率 2/8=25%）。**全部 11 个生产档都 1 CTA/CU**（这里的"11 档"是生产基线 ladder 16/32/64/96/128/256/512/1024/2048/4096/8192，与 §1.5 说的"13 个 retained SBM64 档 64–192"是两个不同集合，别混）：

| T | sbm | tile_n | VGPR | LDS | CTA/CU |
|---|---|---|---|---|---|
| 16/32/96 | 32 | 512 | 232 | 38 K | 1（LDS 允许 4 个，被 VGPR 卡住） |
| 64/128 | 64 | 512 | 256 | 76 K | 1 |
| 512…8192 | 128 | 256 | 248–250 | 88 K | 1 |

（本表是 ISA 资源实测那一轮的 ladder。**b200–256 后来改走 SBM128 path，物理 sbm=128
不是 64**——资源为八路径 253 VGPR / 三档 256 VGPR，均 LDS 90112 B，同样 1 CTA/CU，
见 `report_sbm128.md`；`small_sort_block_m` 的 fallback 规则对 200/256 仍返回 64，
生产是 `SBM128_PATHS` override 掉的，§1 那 6 个复现条目讲的是 fallback 规则本身。）

这不是回归：`waves_per_eu_hint` 默认 2，经 `rocdl.waves_per_eu` 让编译器放开到 256 VGPR。

VGPR 组成：`M_REPEAT×NUM_ACC_N = sbm·tile_n/(256·nw)` 三种几何都等于 16 → **累加器恒 64 VGPR（只占 25%）**；真正吃掉的是 **B 的流水双缓冲 128 VGPR**（ISA 中 MFMA B 操作数跨 120 个寄存器）；A 操作数 32、scale 4、其余 ~28 地址/swizzle。**用户已明确否决"砍 B 双缓冲"**；若要 2 CTA/CU 必须动 VGPR 预算（双缓冲 128 → 腾第二套累加器），重开需先推翻该否决。

S2 侧关联线索：`BM=64+BN=256` 时累加器正好 64 KB，超 workgroup LDS 上限——这是 b512 最大未解锁线索。

## 7. 两堵墙都不贴（PMC 事实，b256 档）

| kernel | HBM 读 | 时间 | 达成带宽 | 占可达峰值(6.5–6.8 TB/s) |
|---|---:|---:|---:|---|
| S1 | 779.8 MB | 205.0 µs | 3.80 TB/s | 56–58% |
| S2 | 473.4 MB | 110.0 µs | 4.30 TB/s | 63–66% |

- 矩阵管占用：S1 b256 **18.7%**、b2048 37.1%、b4096 39.9%；S2 b256 **13.7%**。派生式 `MFMA_BUSY/(4×SQ_BUSY_CU_CYCLES)`——分母乘 4，直接相除会误报成 75%。
- 瓶颈在依赖链/等待类：CU 有 75–93% 时间有 wave，wave 60% 周期在 WAIT_ANY。
- 固定截距 **S1 46.9 µs / S2 30.7 µs 始终未定位**（`S1 ≈ 46.9 + 字节/6.42TB/s`、`S2 ≈ 30.7 + 字节/8.94TB/s`，残差 ≤0.12/0.66 µs）。
- 已知最大字节浪费：S2 读 A2 重复 48×（453 MB demand vs 理想 9.4 MB）、S1 读 A 重复 24×。
- 读数限制：融合 kernel 的 counter 是聚合值，shared/routed 分不开；"B 只读一次"是按模型分配后的说法不是逐矩阵实测。

## 8. FlyDSL 实现坑（重启后不要重犯）

1. 运行时 `if` **只在 `@flyc.jit` 装饰的函数体内有效**，普通方法里写报 `cannot evaluate dynamic 'Boolean' as Python bool during tracing`。
2. AST rewriter 把 `if` 分支内**被赋值的名字**（含任何 `obj.method(...)` 的接收者）收集成 scf.if carried state → 报 `Cannot extract IR values from ...`。修法：分支体内零赋值、零属性调用，把 epilogue 包一层普通函数。
3. 计数器 `arrived % split_k` 是**有符号取余**，越过 INT32_MAX 后永远选不出归约者，静默错；必须用 `arrived & (split_k-1)`。
4. flag 经 LDS 广播后**必须再加一个无条件 barrier**，否则 epilogue 的 cshuffle 会在慢 wave 读 flag 前覆盖它 → `elected` CTA 内发散 → barrier 死锁。

## 9. S1 kernel 微优化（f7a96515 保留的 sbm64_m32_m48_m64 路径）

来源：commit 内 `sbm64_m32_m48_m64/README.md`。M32/M48/M64 三分支共用：

- **current-A 前读**：三条路径都先缓存当前 A tile，再发 next-A DMA（current-A-read-before-next-A-DMA）。
- **B 按 K128 half retire**。
- **A scales 晚加载**（late A-scale load）。
- M48 特有：两个逻辑 scale group 用 ceiling division；48 行用 SBM128 的 wave-uniform tail 拷贝；为共用 M64 epilogue 补两条全零累加器向量。
- **M64 epilogue 空行跳过**：rows48–63 和 rows32–47 在 `>48`/`>32` guard 下同时跳过 SwiGLU/LDS 生产与量化两处；barrier 保持无条件（last16 producer+consumer guard 已保留，middle16 由用户拍板保留但增量收益未定论）。
- 资源：全 8 rank 202 VGPR（b64 为 200）/ 106 SGPR / 0 spill / 45056 B LDS。
