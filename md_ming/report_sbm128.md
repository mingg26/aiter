# MegaMoE SBM128 定稿报告（b200–b256）

- 范围：EP8 fused-shared 的 b200–b256 默认接入——SBM128 八路径/三档选择、ticket 修复、正式测量结果与不确定性、验证证据。
- 有效性基准：aiter commit **`b13717ca0a0dbe0cf53df353467b8f816aa7c762`**（2026-09-14，"Select validated EP8 SBM128 paths for shared b200-256"）。它是 f7a9651 的祖先；f7a9651 的 selector 审计确认 SBM128 默认逐点不变。
- 来源：工作区 `report_sbm128.md`（2026-09-14/15 节）、`redo.md`、`handoff_megamoe.md`；代码 `aiter/ops/flydsl/kernels/mega_moe_m3/`（行号按 f7a9651）。

2026-09-15 更新：八档 S1 选择不变；b200 已保留 **S1 和 S2 leader drain**。S2 仍为 BM32、整空子块跳过 ON、内部 M16 OFF。后续 M64 仅接入120–192，不扩展到 SBM128。资源/spill 仍按本文 §4，不将新 S2 的零 spill 结论套到 S1。重启见 [redo.md](redo.md)。

## 1. 最终默认选择

| batch | SBM128 路径 | 逻辑计算行档 |
|---|---|---|
| 200, 208, 216 | `m16`（**逐 16 行八路径**） | tile 入口按 `valid_rows` 一次选定 M16/32/48/64/80/96/112/128 固定 K-loop |
| 224, 232, 240, 248, 256 | `tiered96`（**三档**） | 64/96/128 三档计算，含 ticket 修复 |

"八路径/三档"是 kernel 内逻辑计算行数，**物理 SBM 均为 128**（128 行 A ping/pong LDS 与输出布局不变，B 双缓冲保留）。

- 代码入口：`mega_moe_config.py:168-173` 的 `SBM128_PATHS`（200/208/216→`m16`，224–256→`tiered96`）；runner 分发 `mega_moe_m3.py:574-582`。实测 S1 源码在私有包 `sbm128_m16/`、`sbm128_tiered/`——**不要把通用 `gemm1.py` 当成本轮默认实现**。
- scope（`mega_moe_m3.py:107-113`）：EP8、128 总专家（每 rank 16）、H6144/I3072/top4、完整 fused shared L13+L2 且 XCD 调度、`tokens == max_tok_per_rank`、无 local-reduce、无显式 Stage1 env override；精确覆盖 200/208/216/224/232/240/248/256。**不是对区间内任意整数 batch 或全部路由分布的最优性声明**。
- 配套参数：S1 均 N256/K256；b200–248 dispatch **CU48 / B-NT2**，b256 为 **CU32 / B-NT0**；S2 **BM32**，b200–248 N256、**b256 N128**；S2 整块跳空 ON、内部 M16 跳空 OFF（`mega_moe_stage2.py:389-398` 的 `sbm128_rollout` 分支）。

## 1.5 机制说明

- **为什么 SBM128**：同一 expert 收 65 个 token 时 SBM64 要分 2 块、SBM128 只需 1 块；但直接算满 128 行浪费计算，所以 kernel 内按 `valid_rows` 选固定行档——检查的是"补出来的空行"，不是扫描输入数值是否全零。判断放在整段计算之前（tile 入口一次选定），不在内部反复判断。
- **A 搬运随档缩短**：选 64 行就只搬前 64 行 A；省掉的是尾部搬运指令和填零，不等于 HBM 读取量按比例下降。物理 128 行布局不变、尾部补零，后续步骤不用换布局；B 权重保留双缓冲。
- **S2 整块跳空**：SBM128 给 S2 留下 BM32 小块，尾部整块全空的直接跳过；队列计数照常推进，shared 照常执行。
- b200 正式路由实证：8 rank 合计 128 个本地专家落在 M48 路径 55 个、M64 70 个、M80 3 个；静态行工作从旧两档的 `125×64+3×128=8384` 行降到八路径的 `55×48+70×64+3×80=7360` 行，约 **−12.2%**——跳空是真实发生的。
- **b192 不走 SBM128**：早期扫描中 SBM128 在 b192 实测慢（+2.51%，该输入 routed 行块只从 129 减到 128），b192 最终走 SBM64 retained 路径（见 `report_sbm64.md`）。不要重试把 SBM128 套到 b192。

## 2. ticket 竞态修复（本轮唯一从诊断进入默认的同步修复）

- 缺陷：tid0 把 64 位 launch ticket 写到 LDS[0:8]，全 CTA publish barrier 后各 wave 读取；旧代码缺少"所有 wave 已消费完成"的 barrier，leader 可能先把第一个 32 位 work ticket 写回同一 LDS[0:4]。
- 修复：在 `ticket64 = Vec(ticket_view.load())[0]` 之后、任何复用该 LDS 之前增加 **`fx.barrier()`**。源码与实际 ISA 确认读后出现 `s_waitcnt lgkmcnt(0); s_barrier`。
- 效果：在 b200 旧最快两档版上正式配对快 **0.516%**（12/12）。
- 边界：它证明旧代码有 race；此前一次 GPU fault 没有捕获 fault PC/core，**不能**把 fault 因果归到该 race。

## 2.5 保留的流水形态（最终定稿版）

- 八路径保留最后确认的 M48/M64 流水、上半 64 行 epilogue 跳空、M128 的 ks-half B retirement 与延后 A-scale；**M80 保留原 A-major/提前 A-scale**（被否决的 M80 改动未混入）。
- 八路径把 `valid_rows` 在 tile 入口一次性分到八个固定 K-loop，逻辑 A DMA 和 MFMA 随路径缩短；M64 的 current-A/next-A DMA 顺序、提前发 next-B、wait/barrier 放在 16 条 MFMA 后、延后 next A-scale 均已逐 rank 修好。
- 三档（tiered96）即"64/96/128 三档 + ticket 修复"，是各点此前最快的已测形态。
- **S2 整块跳空的覆盖范围是本轮补齐的**：原生产白名单只含 192/200/256 等点，208/216/224/232/240/248 被漏掉；`sbm128_rollout` 分支（`mega_moe_stage2.py:389-391`）把它扩到 200–256 全部八点。跳过空块时队列计数照常推进、shared 照常执行；S2 内部按 16 行跳空保持关闭。

## 3. 正式测量结果

同轮正式扫描：固定构造 A 后 B，6 个预声明协议（k0/k16/k32 × n400/n200 × c200/c100，k 是 settling replay 数不是 GEMM K）、每协议 12 对、48 秒稳定预热、graph gate replay 512。表中为主协议 K32/N400/chunk200（n400=每 sample 400 次 measured replay、c200=按 200 replay 分 chunk）。计时含 TopK、quant/preplan、routed S1/S2、TP1 shared、combine/final sum，不含 router GEMM/residual；指标是各 rank 累加 chunk event 后的 max-rank graph 时间。A = 目标 batch 历史最快源码 + ticket barrier（不应把修复后 A 称为历史源码逐字节重放）。六种协议方向一致。

| batch | A / B（µs） | B/A 相对变化 | 默认 |
|---:|---|---|---|
| 200 | 324.215 / 309.705 | **−4.475%**（CI [−4.586%, −4.376%]） | m16 |
| 208 | 329.766 / 319.115 | **−3.230%**（CI [−3.336%, −3.114%]） | m16 |
| 216 | 331.721 / 325.704 | **−1.814%**（CI [−1.991%, −1.637%]） | m16 |
| 224 第1轮 | 335.673 / 335.225 | −0.133%（CI [−0.312%, +0.047%]） | tiered96 |
| 224 第2轮 | 333.882 / 334.589 | +0.212%（CI [+0.012%, +0.408%]） | tiered96 |
| 232 | 337.844 / 341.905 | +1.202%（CI [+1.007%, +1.397%]） | tiered96 |
| 240 | 340.488 / 346.227 | +1.685%（CI [+1.513%, +1.837%]） | tiered96 |
| 248 | 342.866 / 347.585 | +1.377%（CI [+1.103%, +1.631%]） | tiered96 |
| 256 | 360.058 / 364.087 | +1.119%（CI [+0.991%, +1.248%]） | tiered96 |

读法：八路径在 b200/208/216 明确胜 → 默认 m16；八路径在 b232–256 明确输、b224 两轮方向反转 → 这些点保守保留修复 ticket 后的三档（tiered96）。合并 CI 跨 1 不证明稳定等价；固定构造顺序的系统偏差未被同向构造抵消。

**b256 构造顺序诊断（勿误用原单向数字）**：同一份三档 kernel 的 B/A 先测 +1.224%（CI [+1.105%, +1.347%]），交换两对象构造先后后为 −0.775%（CI [−0.918%, −0.627%]），8 rank 完整流水线 ISA 相同。真实两方案再做反向构造、完整 6 协议×12 对：主协议 B/A **−0.942%**（CI [−1.094%, −0.804%]）；两构造方向等权 log-ratio 分层 bootstrap 为 **+0.083%**（CI [−0.018%, +0.179%]）→ 最终选 `tiered96`。这是每种构造方向仅一个进程组的条件性估计，不能给出跨 allocation/跨时段不变性的置信区间；原扫描的单向 1.12% 胜负不得照搬。

## 4. 资源与 ISA

| 路径 | VGPR | SGPR | VGPR spill | SGPR spill | scratch | LDS |
|---|---:|---:|---:|---:|---:|---:|
| 八路径（m16） | 253 | 106 | 0 | 3 | 0 | 90112 B |
| 三档（tiered96） | 256 | 106 | 7 | 3 | 32 B | 90112 B |

**不要把所有默认写成零 spill**——三档带 VGPR spill7/scratch32B 是既有状态。M80 被拒绝的改动未混入（M80 保留原 A-major/提前 A-scale）。

## 5. 验证证据链

- 8 档原生默认 GPU 验证与 ISA 检查通过；均不 monkeypatch selector、S1 或 quant。每档 23 个数学/bit-exact/路由与 epoch 探针、graph512、12 对接线对照、8 rank 实际 S1/S2 ISA 与选中实测源码一致，5 dispatch 与 source/telemetry 门禁通过。
- CPU：**4736** 个选择器场景、**768** 个范围外 S2 跳空场景、**5** 个 regression tests 通过。两个私有包仅改 import，通用计算源与范围外选择器行为保持原样；shared L13-only、非目标维度、显式调参不进入本轮默认。
- K3 只读审核确认无阻止提交的代码问题，并要求保留 b224/b256 的测量限制声明。Mori 端口占用与 Slurm credential 失败的启动日志单独保存，未计为性能样本。
- 证据归档：`megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json`；配置台账 `bench/dep8_best_configs.json`（其旧 `normal_timing` 字段保留历史含义，不是本轮默认的新测量）。
- 归档含原始完整结果/日志、冻结 harness/source manifest、源码快照、64 份八路径 ISA、私有包/S2 重接线 ISA 和 K3 审核。原始轮次、反向轮次、同 kernel 诊断全部留档，没有挑掉不利样本。

## 5.5 A 臂（对照）来源

正式扫描的 A 不是 clean 默认，而是**目标 batch 的历史最快源码补 ticket barrier**——本轮遵循"所有方案只在已测最快方案基础上做实验"的规则。b208–256 的旧最快记录来自 09-13 扫描（4 秒预热、6 对），本轮的修复后正式对照（48 秒预热、12 对）取代其结论；旧扫描数字仅作历史留档，不得与本轮数字跨轮相减。

## 6. 复现与继承注意

- 旧冻结 harness pin 住合入前 HEAD 与文档哈希，**合入后不能照抄旧 run.sh 重跑**；从目标 batch 当前选中的私有实现建新实验目录与新 source pin，保留全部门禁。
- 后续实验从目标 batch 当前选中的私有实现开始；不要退回更慢的通用默认或历史失败候选做 baseline。
- 被取代备忘（防误引）：早期未做流水修正的八路径版本（scaletail64）曾测得慢 1.429%，已被后续修正后的正式结果取代，旧数字作废。
