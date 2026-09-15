# 根因定位纪律（终稿）

> 范围：找到慢之后如何定位根因。计时口径见 `tool_measure.md`。
> 有效性基准：方法论长期有效；§6 的 S2 根因结论各自绑定形状与时代（EP4/8k 与 EP8-shared
> 两代），引用前回源核对。
> 来源：`tool_root_cause.md`、`handoff_megamoe.md`（ATT/PMC 边界节）、`plan_local_reduce.md`。

## 0. 总原则

- 按 **无 profiler 计时 → kernel trace → PMC → ATT** 逐层深入；后层永远不能替代前层。
  profiler 下的 latency 不是正常计时，profile eager 时间不等于 graph 计时。
- 根因的合格标准：**指出时间丢在哪个依赖/资源上，并用一个只改它的正确改动把正常计时打下来**。
  再一轮参数扫描本身不构成根因。
- 一次只验证一项改变；门禁失败的轮次性能数字作废，但失败诊断保留。"先通过后失败"的变体
  以后来的失败为准。

## 1. 各层工具的边界

**kernel trace**：单 dispatch 绝对 device 时间，同一次 traced run 内可比、跨 run 不可比；
跨 rank recorded span 不是 kernel sum；首尾 replay 不代表稳态；VGPR/SGPR 字段不准（以 ISA 与
HIP code object 查询为准）。用途：验证 dispatch 序列、grid/shape 一致、无多余 memcpy。

**PMC**：多 pass 采集扰动甚至串行化；EP/collective 采集时不能让通信 rank 死锁。原始 counter CSV
混入 warmup，必须按正式 trace 的 Dispatch_Id join 取目标 dispatch。credit-stall/latency 类计数
跨硬件实例求和，不能当 wall-time 占比，只有比值可跨 arm 比较。单 rank 采样时其余 rank 照常工作，
profiler 扰动会改变 rank 间到达时序。

**ATT**：只覆盖选中 SE/CU/SIMD 和少量 dispatch（8 wave 量级）。**采样 stall 占比是所采 wave 的
分布，不是 kernel wall 占比**——不能由 22% stall share 预测 22% kernel 收益。跨 GPU 时间戳未校准。
producer 与 consumer CU 的 stall 结构不同（按 admission ticket 区分，不是物理 CU 编号）。

## 2. exposed-bubble 归因法（核心工具）

不按 per-wave stall 总和排名，而按"**CU 上没有任何 resident wave 能 issue**"的时间归因到 PC：

1. 解码后的 wave 按 4-cycle quantum 对齐；per-PC stall 与 decoder 自身聚合逐 PC 对账。
2. **MFMA 修正**：decoder 对每个 opcode（含 MFMA）只报 ~1 quantum issue 时长；用 PMC 的
   `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_VALU_MFMA_F8` 实测每次 MFMA 占用（本项目实测 = 32 cycles），
   把每次 MFMA issue 延长后再判定 bubble。
3. 区分 exposed（没人能跑）与被掩盖的 stall，两者都要有数。
4. **bubble 占比上升不等于变慢**——指令减少后 bubble 比例可能上升而 kernel 更快。
5. 少数 wave 卡在同一 PC ≠ 相位崩塌；先量化"所有 resident wave 同时卡该 PC"的占比。
6. IMMED/category9 含真实 wait/nop；统计时只合并 `s_barrier` continuation，不能一律删除。

### ATT 分析的运维约束（踩过的坑）

**旧 dense 大矩阵分析曾 OOM，不要重跑。** host 侧分析要用带 numpy 的 venv（host 的
`python3` 没有 numpy）；**大数组分析放到计算节点的小 CPU step 上跑，不要在 8 GB 的 control
主机上跑**——这是运维约束，不是分析方法问题，换机器后同样成立。

## 3. 判别纪律

- **判别性实验**：改动应能区分假设。范例：BM=128 把 B 的名义 DRAM 需求减半，结果变慢——
  "DRAM 读量是瓶颈"被否证（实测流量只降 17%，L2 复用本来就在工作）。
- **floor share ≠ marginal cost**：链路跑在 91.6% 实测带宽上限，不等于去掉它能省 91.6%；
  去掉后其他瓶颈接管（实测只省 40%）。引用根因只能说边际成本。
- **多轮失败也是证据**：五轮 pipelining 改动（改 load 发射时机）都只值 0–2%，共同点是都不改变
  共享队列占用时长——指向请求路径/队列而非发射时机。
- 数值无效探针只能当 ceiling（如把 P2P store 重定向到本地 buffer：地址布局/lane mask/字节数
  完全一致、只去跨 xGMI 一跳，输出无效，时间只可引为上限）。**探针自身的假负载要先排除**——
  典型是多个 peer 重定向到同一个 base 造成行碰撞，那样测出来的"上限"是探针自己的产物。
- 静态资源上限（HIP 查询的 CTA/CU）不是实测 occupancy，分开报告。

## 4. gfx950/CDNA4 一致性语义（定位内存类根因必备）

- **device scope = SC1=1**（多 L2 下走 coherent bypass）；wave-scope store（SC1=SC0=0）即使用
  waitcnt 等完也不升级到跨 XCD/L2 可见；`buffer_wbl2 sc1` 是更宽的 L2 写回，不等于单 store 完成。
- **裸 `buffer_inv` 在 gfx950 是 NOP**——"cached load + 裸 inv" 不等于 NT load。
  `buffer_inv sc0 sc1` 使 CU cache 与 L2 非一致性行失效（system acquire），不是清空所有 L2。
- **NT load（cache_modifier=2）绕过 CU cache 但可命中共享 XCD L2**；staging 用 cached store +
  NT load 的前提是所有 producer 钉在同一物理 XCD。
- 原子发布前的 `s_waitcnt`（如 vmcnt=0/expcnt=7/lgkmcnt=15 即 0xF70）只覆盖本 wave 的 store；
  跨 wave 可见性要另行论证。
- 参考：AMD CDNA4 ISA 文档 §9.1.10、tables 49–52；LLVM CPol 编码（SC1=16，NT=2，staging 0x12）。

## 5. 与正确性门禁的联动

- 重放相同输入会掩盖 stale 数据：在已捕获地址上改变值和路由，检查每次 replay 的输出；
  保留失败张量快照。
- 计时臂之间 ISA 逐条一致（除被测改动）；资源字段变化记录为混淆因素。
- 每次 closure 的 route/quant 等前置不计入被测 kernel 的事件窗口。

## 6. S2 已确立的根因结论（按时代分，引用前回源核对）

**EP4 / 8k 时代（旧 S2 架构）：**

- **跨 GPU P2P 写收费两次**：xGMI 写以 ~156GB/s（可达上限的 91.6%）直接占链路，且挤占共享
  TA/TCP 队列把读延迟抬高 ~4 倍、驱逐 L2 中的 A/B；去掉 P2P 后 DRAM 读 -39.5%，但实际只省 40%
  （floor share ≠ marginal cost 的原始出处）。
- **K-loop 尾 `s_waitcnt vmcnt(0)` 是兑现点不是病因**——放宽它只值 2%；后续 asyncmark 支线实测
  再次证实（等待从 waitcnt 搬到 VMEM 发射口，净零）。
- local-reduce 把 staging 变 XCD-local 后，主要改善来源是读服务**延迟**而非读请求数/读字节数
  （377.5→1482.9→439.9 cycles 的对比链）。

**S2-shared（融合 shared 之后，比上面三条新）**：`tail` 融合下 **shared 要到整段 span
的 ~85% 才开始**；routed 侧最大的暴露点是 **K 循环 B 权重的 VMEM wait** 与
**claim/epilogue 串行链**。这条是"shared 融合进 S1/S2 之后该往哪看"的入口。

注意：这些结论绑定各自的形状与架构时代，新形状/新配置下引用前回源核对。
