# S2 本地预归约（local_reduce）终局报告

> 范围：S2 GEMM2/P2P 的 BF16 本地预归约方案（同 rank 多 partial 先本地 staging 归约再发 P2P）——rootcause 账、终局性能判定、设计不变量。
> 有效性基准：截至 2026-09-10 的最终测量；与 f7a96515（SBM64 S1 选择）是相邻工作线，小 batch 退化在终局**未解决**。
> 来源：`plan_local_reduce.md`、`handoff_megamoe.md`、`tool_root_cause.md`。

## 1. 终局判定（以此为准，历史中间结果作废）

| batch（tokens/rank） | 完整 MoE 窗口 | 判定 |
|---|---|---|
| 8192 | 4415.097→4205.668 µs，**−4.744%** | 胜（12/12，全门禁通过） |
| 2048 | 1524.617→1436.611 µs，**−5.772%** | 胜（12/12，正确性/source/dispatch/XCD/noise 全过） |
| 256 / 512 | **+4.622% / +3.682%**（变慢） | **未解决**，最终状态 |
| 1024 | 未测 | 收益阈值未知 |

- 小 batch 退化不是 CTA 不足：256 档 CTA sweep（256/512/768/1024 CTA 分别 +13.404%/+2.501%/+0.441%/+0.523%）完成后保留 640 CTA，问题仍在。最大稳定增量来自 local-reduce epilogue，其次 queue/setup。
- 曾定位并修复的关键 bug：4k 档 `queue_grid_mult=1` 造成 240 CTA 欠并发，qg1→qg5 后 S2 −29.893%、full −12.406%；8192 sweep 失败源自 S1 的 LDS work/flags 复用竞态。
- 路线侧注：256/512 慢的补救方向（先单变量降 staging/publication/winner 路径成本，再优化 queue 取票/准备/reset）从未执行完成。**此方案未进生产默认。**

## 2. Rootcause 数字（EP4、8192 tok/rank，S2 = M64/N128/K128）

| 量 | 值 |
|---|---:|
| S2 整段（queue reset + GEMM2/P2P + BF16 combine） | 2113.7 µs |
| GEMM2/P2P dispatch | 1927.6 µs |
| 跨 xGMI 字节 | **0.302 GB**（9,376,128 × 32 B，占总写 **74.94%**，理论 75%） |
| 达成 fabric 速率 | **156 GB/s** |
| fabric 天花板 | **168.3 GB/s**（稠密）/ 160.4（散射）；NCCL all-gather 独立测 170 GB/s 吻合 |
| 完美重叠下限 max(GEMM 1151, fabric 1765) | **1765 µs** |
| **全部调度类优化上限** | 1928 − 1765 = **163 µs ≈ 8.4%** |

163 µs 的构成：

- **~26% 是 vmcnt 假依赖**：gfx950 无 `vscnt`，load/store 共用 vmcnt，K 循环尾 `vmcnt(0)` 把上一 tile 的 16 条 P2P store 一起排干。放宽到 `vmcnt(7/10/19)` 实测 2113.73→2071.09 µs（2.0%）。
- **~74% 是真争用**：写请求占住 TA/TCP 队列、耗尽 GMI 信用、把 B/A 挤出 L2。探针实证去掉 fabric 后：读延迟 **339.3→213.6 cyc**、写延迟 **2001.0→174.6 cyc**、DRAM 读 5.471→3.307 GB 而读请求数几乎不变（126.27M→126.16M）。wave specialization 之类动不了它。

结论：**只有减少 fabric 字节能动 1765 这个下限**。已排除方向（附上限）：写宽已是原生 64 B → 0；行宽 256→4096 B 粗化 → 0.2%；散射改稠密 → 4.7%（37 µs）；fp8 transport（−50% 字节）因精度被排除（fp8 partial relL2 0.0266 > 门限 0.015）。

## 3. 方案账（topk=4 / EP4 / 专家均匀）

P（某 rank 有该 token ≥1 个 expert) = 1 − (3/4)⁴ = 0.6836。对本 rank 32,768 个 partial：

| | 行数 | 字节 |
|---|---:|---:|
| 今天跨 fabric | 24,576 | 0.302 GB |
| **归约后跨 fabric** | 16,800 | **0.206 GB（−31.6%）** |
| 非最后 partial（唯一过 staging 的） | 10,368 | 0.127 GB |
| staging 新增读写 | +0.255 GB（20,736 行次） | ≈ +79 µs |

若**所有** partial 都过 staging 再读回：+0.806 GB ≈ **+250 µs，把收益抵消干净、方案归零**——所以"最后 CTA 寄存器合并"是硬不变量（见下）。

k=1 token 占 61.7%，每条带第一次检查就是最后一个，直接发走不过 staging。归约粒度必须是 **(token, n_block 条带)** 不是整行（epilogue tile BM×BN=64×128，一个 CTA 只持有 128 列，6144 列整行由 48 个 CTA 分摊）。

## 4. 设计不变量（重开/移植时必须保留）

1. **最后到达的 CTA 用自己寄存器里的累加器合并**，不再走一遍 staging。读 staging 里同条带的 k−1 份（每份 256 B），与自己寄存器的 128 列按**固定 k-slot 序号**相加后发出。否则 +250 µs，方案归零。
2. **确定性靠固定相加顺序**：归约者身份由 atomic 到达顺序决定，但输入集合与 k-slot 相加顺序固定，结果逐位确定。禁浮点原子累加（产品 determinism 要求），计数原子允许。partial BF16、归约 FP32、输出 BF16。
3. **k_local 打进现有 32-bit packed 元数据高位**（行号按旧生产树：`mega_moe_stage2.py:113-116`，f7a9651 上位置已变：现为 `token(24b) | s(8b)`，topk=4 下 s 只用 2 bit，高位放得下 k_local ∈ {1..4}）。不需要新表、不多一次间接寻址。
4. **计数器自清零**：归约者归约完把自己 (token, 条带) 的计数器写回 0（不是 store 0 整个数组；每 launch 恰好一个归约者，graph replay 兼容）。k=1 条带不碰计数器。
5. **缺席槽位 validity mask 不能用 `-1` 哨兵**：改成 (token × rank) 编址后缺席槽是未初始化内存；validity 由对端用自己的 topk_ids 在本地算 mask（路由的纯函数，不需要 fabric 信号）。不处理的后果是 garbage×weight → NaN。
6. **atomic 批量发射再消费**：tile 内 64 行的计数 atomic 全部发出后再统一消费结果，避免逐行"发 atomic→等 L2→分支"串行（模式参考 `flydsl_dispatch_combine_intranode_kernel.py:1052`）。
7. **S2 写地址** `slot = dest_lid*npes + self_rank`。注意 M3 下 npes==topk==4 时 combine buffer 容量不变是**巧合**，EP8/topk6 要重新算。
8. **release/acquire 成本验证**：非最后 CTA 的 staging 写落地是本地 HBM（写延迟 174.6 cyc vs P2P 的 2001），排干便宜 11 倍——但必须在 ISA 里确认，不能假设。

## 5. 未量化/残余风险

- **发送后移（最大风险）**：k≥2 的 token（38.3%）要等本 rank 最后一个 expert 算完，约 38% fabric 流量后移；fabric 已 91.6% 占用，尾部堆积可能吃掉收益。CPU 模型可估（用真实 dispatch 的 expert→m_block 映射 + 168 GB/s 排空速率模拟尾部）。
- 负载不均残余：归约落在 epilogue，epilogue 已与 `s_barrier` 耦合，同 CTA 内 wave 归约量不同会被拉到最慢。
- 路由偏斜方向是利好（同 rank 碰撞概率升、收益变大），但真实路由下 22,400 这个数要重算。

## 6. 一条与 vmcnt 的交叉引用

本方案的 rootcause 中"vmcnt 假依赖 ~26%"与 S2 asyncmark 支线是同一问题的两面；asyncmark 的最终状态见 `trick_vmcnt_asyncmark.md`（小 batch EP8 无净收益已回退；大 batch EP4 曾赢 1.35–3.88%）。
