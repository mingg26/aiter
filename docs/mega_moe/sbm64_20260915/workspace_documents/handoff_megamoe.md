> **用户最新要求（2026-09-13，后续实验最高优先级）：所有方案都只在目前已测出来最快的方案基础上做实验。**
> 按目标 batch 和测试条件，先核对已有记录中的最快方案、源码和配置；以该方案作为后续修改的起点及性能对照，即使它尚未合入生产。不得用较慢的 clean commit、生产默认配置或旧方案替代当前已测最快方案。测时仍须遵守 `tool_measure.md`。
> **本条覆盖下文历史记录中要求以 clean/default 为基线的指令；历史数字保留其原有对照含义，不代表后续实验基线。**

> **用户明确要求（2026-09-10）：未经用户明确允许，不得新建或更新文档。**
> **实验、测试或调参的授权不包含文档更新授权；不得以归档、同步结果或交接为由，自动修改报告、配置记录或任何 handoff。只有用户明确授权时，才更新其指定范围。本条必须保留在本文件最前面。**

## 当前工作模式：主助手与 K3 协作（2026-09-14）

用户所说的“老办法”默认指以下流程；若用户本轮明确调整顺序，以本轮指令为准。

1. **先定基线，再讨论。** 主助手先读最新报告、handoff、redo 与目标 batch 的实测记录，只从该条件下目前已测最快的源码/配置出发。主助手与 K3 先讨论一个具体改点、预期收益及寄存器/同步风险；已讨论清楚的方案不重复讨论，每轮优先只改一个变量。
2. **主助手实现并先看 ISA。** 主助手负责修改代码、编译、对比实际分支、指令顺序、wait/barrier、VGPR/SGPR、spill、scratch 和 LDS。若新增寄存器压力或溢出已足以否定方案，先停止该候选，不为走流程继续跑 GPU；不要把“压寄存器”变成增加 spill。具体是否值得实测遵从用户当轮要求。
3. **ISA 可接受后，GPU 与 K3 审核并行。** 主助手负责 GPU 实验和监督；K3 独立审核源码 diff、ISA、正确性/同步风险、适用范围与测试方法。无需等 K3 审完才启动 GPU，但 GPU 实验之间串行，避免相互干扰。K3 默认只读，不与主助手同时修改同一份代码。
4. **发现问题就停本轮、修好重来。** K3 提出问题后，主助手结合源码/ISA核实；若问题成立并使当前实验无效，停止对应实验的明确进程，不取消整个 allocation 或无关任务。修复后使用新的实验目录/输出与 source pins，重新检查 ISA、审核并跑 GPU；运行中不修改已冻结的源码、harness 或受保护文档。
5. **结果由证据决定。** 遵守 `tool_measure.md`，先过正确性、graph/epoch、全 rank dispatch、source/ISA 与硬件稳定性门禁，再看同进程 balanced paired A/B。快速试验只作筛选，不冒充正式收益；不跨进程比较绝对均值、不删除不利样本。小收益、轮次反转或同 kernel 也有差异时，先排查测量偏差，必要时补构造顺序反向对照。
6. **主助手汇总，K3 复查结论。** 两者复核是否保留、默认是否真的命中选中方案、是否影响其它 batch。无收益或退化的候选不混入保留版；不确定就明确说明。继续下一个方向前，简短交代本轮结果和下一改点，不盲目换 M 分支。
7. **提交和文档按用户授权。** 用户要求 commit 时提交实际生产代码，不只提交实验快照；用户明确允许更新文档时，再同步报告、handoff、redo、默认台账及证据。保留旧实验原始记录和不确定性，不改写冻结 checkpoint。没有相应授权时，不自动提交或更新文档。

K3 指实际 Kimi CLI 的 `inferact-kimi-k3` 模型，不是把另一个普通子助手命名为 K3。当前可续用的讨论 session 为 `session_054a5e40-816d-4745-b49b-072ee8da21af`；主助手传给 K3 的材料应包含目标、源码路径、diff/ISA、基线与结果路径，并明确本轮是方案讨论还是只读审核。

## 2026-09-15 最新恢复入口：SBM64 13点保留，b72不采用

先读 [report_sbm128.md 最新 SBM64 小节](report_sbm128.md)，以及 [fork 内完整报告](aiter-ming-amd-m3-megamoe/docs/mega_moe/sbm64_20260915/README.md)。本轮用户授权实际代码提交、文档/台账更新、K3最终审核。

- 默认接入精确 batch：**64,104,112,120,128,136,144,152,160,168,176,184,192**；S1物理SBM64/N256/K256，`sbm64_path=m32_m48_m64`，保留M32/M48/M64、current-A前读、K128 B退休、late A-scale与last16/middle16 epilogue producer+consumer guards；ticket修复保持。
- scope：EP8/H6144/I3072/E128/top4、shared L13+L2/XCD、tokens==mtpr、无local-reduce或显式Stage1 env覆盖。b64 S2 BM64/N128 NT ON；其余BM32/N256 NT OFF；S2整空skip除64/152外ON。b72继续原SBM32/N512，候选与scale6裁剪均不合入。已有SBM128不变。
- 性能来源 `.scratch/sbm64_retained_sweep_20260914_v1`：26轮、312/312对快，耗时降低约2.10%–4.97%，quick screening口径；b136用整轮优化前最快M64/N512+ticket，不用last16增量对照。middle16自身小收益仍未解决。
- 用户最后要求不再重跑所有 batch GPU，只需确保 ISA 一致。13点×8rank 共104份原生 S1 ISA已全部匹配；补充原生GPU仅完成 b64/104/112/120（均16探针、graph512及全8rank S1/S2/quant ISA通过）。其余9点未重跑原生GPU，沿用先前两方向实测证据，加本轮ISA身份和选择器验证；不得声称13点全部完成了本轮GPU。b128启动在worker/kernel前因TCPStore端口53128占用失败，按用户要求不重试。
- 原生导入104份CPU ISA一致、5371选择器比较和9项CPU回归通过。GPU接入验证在 `.scratch/sbm64_final_rollout_20260915_v1/native_v2`，已完成四点真实默认16探针/graph512/terminalheads及全8rank S1/S2/quant ISA；最终状态见fork报告目录的 `native_gpu_validation.json` 和 `final_audit.json`。
- fork：`aiter-ming-amd-m3-megamoe:ming-amd-m3-megamoe` → `ming-fork=mingg26/aiter`；同步生产树 `megamoe-s1-1k-prod/aiter:s1-unified-production`。最新提交身份见 `final_audit.json`；不要使用下方历史HEAD作为当前入口。配置台账随此轮更新，历史normal_timing字段不代表新默认。
- K3为实际CLI模型 `inferact-kimi-k3`，续用session `session_ebbbe6de-6172-4d40-b346-b3d01ac8313b`；设计/实现/最后审核日志在本轮review目录。旧“config N256会改变host队列”的猜测已根据源码撤回：tile_n在此scope只传给S1 launch；仍做原生GPU接入核验。
- 原始失败记录保留：b160旧AB首跑Mori bind errno98无计时；b72首次校验遗漏`.kd`后缀无计时；本轮native首次把容器bind路径误当宿主路径，kernel前中止，新native_v2核对容器路径+源码hash。均不影响保留性能样本。
- 本轮更新文档已获授权。旧实验冻结的文档/HEAD pins不改写，复跑须新目录/新pins；不得绕过门禁。job1913/node03 allocation保留，预计2026-09-15 19:06 UTC结束。

## 2026-09-14 最新：b200–256 默认合入（覆盖下方历史状态）

用户本轮明确授权 GPU 验证、commit 和文档更新。实际生产 `aiter` 已提交 **`b13717ca0a0dbe0cf53df353467b8f816aa7c762`**（不是仅提交外层实验快照）。默认按 batch 选择：**b200/208/216 使用 SBM128 逐 16 行八路径；b224/232/240/248/256 使用修复 ticket 的 64/96/128 三档**。这里“八路径/三档”是 kernel 内逻辑计算行数，物理 SBM 均为 128。

范围仅 EP8、128 总专家（每 rank 16）、H6144/I3072/top4、完整 fused shared L13+L2 且 XCD 调度、`tokens == max_tok_per_rank`，精确覆盖 200/208/216/224/232/240/248/256；无 local-reduce、无显式 Stage1 环境调参。其他 batch、形状和通用默认保持不变；这不是对区间内任意整数 batch 或全部路由分布的最优性声明。

代码入口是 `mega_moe_config.py:SBM128_PATHS` 与 `MegaMoEM3` 的受限选择器。实测 S1 源码在私有 `sbm128_m16/`、`sbm128_tiered/`；不要把通用 `gemm1.py` 当成本轮默认实现。S1 均为 N256/K256，b200–248 的 dispatch CU48/B-NT2，b256 为 CU32/B-NT0；S2 BM32、b200–248 N256、b256 N128，整块跳空 ON、内部 M16 OFF。

证据入口：[megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json](megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json)，[配置台账](megamoe-s1-1k-prod/bench/dep8_best_configs.json)。旧 `normal_timing` 等字段保留历史含义，不能当成本轮默认的新测量。

八档原生默认 GPU 验证及 ISA 检查已通过，4736 个选择器场景、768 个范围外 S2 场景和 5 个 CPU regression tests 通过。b224 两轮方向反转，保守保留三档；b256 检出构造顺序相关时间差，已补真实两方案反向构造，详见 report 最新节与 summary 中 position_diagnostic，不能照搬原单向 1.12% 胜负。八路径为 VGPR253/Vspill0/scratch0；三档已有 VGPR256/Vspill7/scratch32 B。M80 被拒绝的改动未保留。

后续从目标 batch 的当前选中私有实现开始。旧冻结 harness 仍 pin 合入前 HEAD/文档，需新目录与新 pins 才能复跑；不可跳过门禁。下方均为历史，尤其“仍未合入”“逐16行仍输”不再描述当前 b200–216 状态。

## 2026-09-14 重启入口：b160–192 的 SBM64 逐 16 行已完成；b200 的 SBM128 逐 16 行仍输

本节优先于下方所有历史。用户在重启前明确授权记录，并要求 4 个 K3 分别梳理全部实验、各 batch 最快方案、SBM128 逐 16 行的性能原因和叙述一致性。完整表、代码差异、证据边界与下一步已写入工作区根目录的 [report_sbm128.md](report_sbm128.md)；机器恢复后先读该报告本节以及 [汇总 JSON](.scratch/sbm128_poll_serial_review_20260914.json)。

### 当前结论

- **b160/168/176/184 当前最快记录**：`.scratch/sbm64_m16_dma_sweep_20260913_v1/s1_only`，S1 SBM64/N512，在一个 K-loop 内按 16 行屏蔽 MFMA，并同步缩短 A DMA；S2 BM32/N256 整块跳空 ON、内部 M16 OFF。正式结果依次快 **0.088%、0.341%、0.446%、0.249%**，候选时间为 **303.591、304.957、305.020、308.903 μs**。证据：`.scratch/sbm64_m16_dma_sweep_20260913_v1/review/sweep_results.json`。
- **b192 当前最强候选**：`.scratch/sbm64_n512_singleloop_20260913_v1/s1_only`，S1 SBM64/N512，在一个 K-loop 内按 16 行屏蔽 MFMA，但 A DMA 仍搬满 64 行；正式同轮 **316.248→312.255 μs，快 1.263%，12/12**。scalar-A-DMA 版另测快 0.643%，但两者未直接 head-to-head，不得用跨轮绝对均值算两候选差距。证据：`.scratch/sbm64_n512_singleloop_review_20260913/analysis.json`。
- **b200 当前最快**：`.scratch/b200_sbm128_entry_ticket_fence_20260914_v1/candidate_200`，S1 SBM128/N256 的 64/128 两档计算和短 A DMA，S2 BM32/N256 整块跳空 ON，另含初始 launch-ticket 读取完成 barrier。正式同轮 **325.097→323.418 μs，快 0.516%，12/12**。这是“此前最快优化版 + fix”，不是 clean commit + fix；旧轮次的 322.141 μs 不能与本轮 323.418 μs 跨进程比较。
- **b208–248 当前最快记录**：`.scratch/sbm128_sweep_s2empty_20260913/kernels_b256`，S1 SBM128/N256 的 64/96/128 三档，S2 BM32/N256 整块跳空 ON；候选时间依次为 b208 **325.552**、b216 **327.910**、b224 **328.606**、b232 **332.744**、b240 **335.387**、b248 **338.553 μs**。b256 使用相同 S1 三档但 S2 为 BM32/N128，候选 **356.192 μs**。这些都是旧 4 秒预热、6 对结果；b256 另有 8 秒/12 对结果，方向相同。三档源码尚未逐 batch 加 ticket barrier 正式复测。
- 生产 `megamoe-s1-1k-prod/aiter` 仍为 clean commit **`05bbc69252444ece0bff54858eae4be2864000a7`**，本次核对工作区干净。上述所有新方案仍在 `.scratch`，没有合入生产。外层仓库有此前已有的 `HANDOFF.md` 修改和若干未跟踪实验文件，本轮没有触碰。

### b200 逐 16 行实验到底做了什么

完整八路径版在 tile 入口读取实际 `valid_rows`，一次选择 M16/32/48/64/80/96/112/128 固定 K-loop；不是在每个 MFMA 前判断 token，也不是针对测试输入写死。逻辑上只搬运/计算所需行，物理 A ping/pong LDS 和输出布局仍保留 128 行，B 双缓冲保留。S2 一直只做 BM32 整块跳空，内部 M16 跳空关闭。

本轮依次测试 entry 八路径、M64 A 预取、ticket barrier、M64 early A/B、current-A 全缓存、阻止 end wait 前移、延后 next A-scale。唯一采用的是在旧 64/128 两档版上加入 ticket barrier；其余逐 16 行候选全部慢。最终 `scaletail64` 相对当前最快版 **325.211→329.858 μs，慢 1.429%，CI [1.291%,1.569%]，0/12**。全表和每一行自己的 baseline 标签见 `.scratch/sbm128_poll_serial_review_20260914.json`。

初始 ticket race 是源码与 ISA 已证实的缺陷：旧代码让所有 wave 从 LDS[0:8] 读 launch ticket 后，没有在 leader 将第一个 work ticket 写回 LDS[0:4] 前做消费完成 barrier。修复已经验证，并在 b200 提速。早先 GPU fault 没有 fault PC/core，不能把 fault 因果归到该 race。

### 为什么完整八路径版还慢

b200 正式路由在 8 rank 合计实际命中 M48 55 个专家、M64 70 个、M80 3 个；八路径确实把静态计算行数从 8384 减到 7360，约少 **12.2%**。回归说明新增实现成本超过这部分收益，不能说跳空没有生效。

已确认的差异：S1 ISA **4015→6503 行**、标量 branch **55→107**、spill **2→11**、private **12→48 B**、scratch 指令 **5→30**。K-loop 内 scratch 为 0，额外 spill 主要在路径外围。M64 的 current-A、next-B、wait/barrier、next-scale 顺序已逐 rank 修好；实际命中的 M48/M80 仍有 odd-16 A DMA 的每-K wave 分支，其它路径也未完成同等级的流水优化。代码体积/I-cache、外围 spill、尾部分支和其它路径调度各占多少尚无运行时证据，不得写成已定因果。

K3 审计中有两项已被独立复核否决：协议名 `k0/k16/k32` 的 K 是 settling replay 次数，不是 GEMM K；b200 为 EP8 全局路由，本地 16 个专家平均约 50 行，不是用单 rank 的 800 routes 除以全局 128 experts 得到 6.25 行。后续只引用 JSON、源码和 ISA 复核后的结论。

### 恢复后按这个顺序继续

1. 重查 Slurm allocation、节点和 GPU 空闲状态；旧记录为 job1841、`do-mi350x-04`，不得假定重启后仍有效。测量仍严格遵守 `tool_measure.md`。
2. 以 b200 当前最快 `candidate_200` 为唯一源码起点，先只增加 **M48** 路径，不带八路径版的其它调度改动。当前输入 55/128 个专家命中它，这是信息量最高、代码增量最小的消融。
3. 若 M48 有收益，再单独增加 **M80**；它只命中 3/128，主要验证 odd-16 尾部搬运。每个版本先审计 ISA 的 spill/private、代码大小、scratch 位置与实际流水，再做 23 探针、graph512、6 协议×12 对正式配对。
4. 若单个路径就使 spill 明显增加，先缩短 runtime `scf.if` 跨分支携带的 acc/epilogue 状态生命周期；目标是接近最快版 spill2/private12B。之后才考虑 PMC/ATT 验证 I-cache 或 spill 延迟。
5. b208–256 分别从该 batch 已测最快的三档源码加入同一 ticket barrier，与原最快同进程配对，得到修复后的新基线；不能从 b200 的 0.516% 外推。
6. b192 从 single-loop 版加入 ticket barrier，再同进程比较是否缩短 A DMA；现有跨轮证据提示 scalar-A-DMA 可能吃掉约一半收益，但尚未形成直接因果对照。

本轮没有修改生产代码，没有遗留 GPU 测试。4 个 K3 的原始分析中存在上述已纠正误读，恢复时不要直接照抄其推测。

# MiniMax-M3 AMD MegaMoE handoff

## 2026-09-13 重启入口：SBM128 报告已整理；当前只追查 SBM64 的 S2 跳空收益

本节优先于下方历史。用户最新指令：“算了，先更新文档吧，我觉得需要重启了”。本轮更新本文件及工作区根目录的 [report_sbm128.md](report_sbm128.md)。停止继续测量，**尚未实现或运行重复 S2 breakdown / PMC**；上一轮 8 个单点实验和 K3 分析均已结束，写入时未发现本轮遗留进程。不要自动重启 S1 扫描。

### 源码与用户要求

- 生产仓库 `megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，HEAD **`05bbc69252444ece0bff54858eae4be2864000a7`**；本次核对 `git status --short` 为空。SBM128 和本轮 SBM64 跳空改动全部仍在 `.scratch`，**未合入、未提交**。
- 性能必须直接对比上述 **clean commit 的正式默认 selector**，不能用旧候选充当 baseline。保留 B 双缓冲。用户当前要求只看 S2；S1 M32/M64 的回退是独立问题。
- 测量前先读 [tool_measure.md](tool_measure.md)。本轮历史数据仅 4 秒预热、6 对，低于规范的至少 8 秒预热与推荐 12 对；CI 只反映轮内配对，不证明跨进程稳定。
- 报告现位于工作区根目录；旧路径 `megamoe-s1-1k-prod/report_sbm128.md` 已由外部移动，不要恢复旧文件或覆盖用户对报告的改写。

### 本轮 S2 数据与已确认事实

在 clean 的 SBM64/N512、S2 BM32/N256 几何下，仅给原有整块跳空白名单新增 **120/136/144/152**。原生产白名单仍是 `104/112/128/160/168/176/184/192/200/256`。

| batch/rank | clean→候选，完整 forward μs | 时延变化 | 快的配对 | 全空 routed BM32 行块 / 256 |
|---:|---:|---:|---:|---:|
| 120 | 291.001→278.473 | −4.304% | 6/6 | 91（35.55%） |
| 136 | 292.038→289.999 | −0.697% | 6/6 | 48（18.75%） |
| 144 | 294.407→293.660 | −0.253% | 4/6 | 36（14.06%） |
| 152 | 297.191→297.417 | +0.076% | 2/6 | 23（8.98%） |

配对时延变化 95% CI：b120 `[-4.686%, -3.983%]`；b136 `[-1.053%, -0.361%]`；b144 `[-0.542%, +0.034%]`；b152 `[-0.211%, +0.319%]`。后两点视为持平，前两点仅为单轮收益。

全部通过 7 类输入/路由、独立数学参考、两臂逐位一致、graph256、队列终值、源码/dispatch/ISA 门禁。未改动的 S1 ISA 逐字节相同；S2 两臂 VGPR231、spill0、scratch0、LDS33088 B。六对 AB/BA 平衡，每 sample 16 次 settling + 200 次连续 graph replay，取最慢 rank；无 profiler 的完整 forward 包含 TopK、routed MoE、TP1 shared MLP 和最终相加，不含 router GEMM / residual。

每 rank 有 16 个本地专家，本次这四个点每个专家都只占一个 SBM64 块，故 S2 拆成 32 个 BM32 行块/rank。完全空行块可由每专家 token 数 `c` 独立算为 `2*ceil(c/64)-ceil(c/32)`；乘 24 才是含 N 方向的 GEMM 任务数。各 rank 0–7 的空行块：

```text
b120: 13 10 14 11 10 12 12 9
b136:  7  4 10  7  6  6  4 4
b144:  4  3  8  7  2  6  4 2
b152:  3  2  6  3  2  4  2 1
```

简单解释：30 token 拆为 `[30,0]`，后块能跳；34 token 拆为 `[32,2]`，后块还有 30 行 padding，但必须执行 BM32。均匀 top4 的全体专家平均 token 数为 b/4（此处 30/34/36/38），更多专家跨过 32 后空块减少。shared 仅有非空尾块，外层 OR 强制执行 shared。clean 已屏蔽空 A 行读取，新增 skip 省剩余 B 读取指令/MFMA/epilogue，**queue claim、遍历、band padding 仍保留**。同专家的有效块还要读 B，不能把空块比例直接换成 HBM 字节比例。

### 未解决问题与下一步

**空块减少已证实，但仍未解释 b136/144 的完整收益为何这么小。** 缓存复用、persistent 队列尾部、rank 临界路径是候选解释，尚无本轮 PMC/逐任务时间证据；不要写成“已证明 HBM 不变”或“空块都藏在等待里”。

单次 graph trace 的 S2 八卡平均：b120 86.674→74.885、b136 90.024→80.075、b144 89.399→84.750、b152 89.934→86.950 μs。每 arm 每 rank 只有一个实例，**不能当稳定分层计时，也不能与无 profiler 完整 forward 做减法分摊**。K3 已独立核对计数、实际 `_ez1` 和资源，但其日志内的局部均值/范围与等待解释不全可靠；本节独立统计和证据界限优先。

恢复后若用户继续：先做 **保留完整 forward 的重复 S2 分层计时**，b120 作参照、重点 b136/144/152；以同进程 balanced pairs 聚合多次 replay 的逐 kernel duration，保留逐 rank 数据、S2 平均与最慢 rank，并另测无 profiler 完整 forward。至少 8 秒预热、推荐 12 对，不把 replay/rank 当独立统计样本；profile 会扰动工作点，必须分开报告。再根据结果决定是否采 HBM/L2 PMC 或任务级 trace。原对话最后只读了现有 harness 并讨论方法，**尚无新 profiling harness**。

另试的 S1-only M32/M64 均未采用：b64 267.613→296.146（+10.66%）、b104 273.397→320.597（+17.27%）、b112 275.302→324.932（+18.03%）、b128 284.117→338.089（+19.00%），均正确但 0/6 更快，spill2→87、scratch12→272 B；用户已明确先不研究这一项。

### 文件、复现与环境

- 简明报告：[report_sbm128.md](report_sbm128.md)。含前一轮 SBM128 b192–256 全表及本轮 SBM64 补充。SBM128 候选 b200 两档、b208–256 三档；b192 慢，b200–256 单轮快，尚未默认启用。
- 本轮汇总：[scan_results.json](.scratch/sbm64_empty_small_20260913_v2/scan_results.json)、[S2 空块/trace 分析](.scratch/sbm64_empty_small_20260913_v2/s2_empty_analysis.json)。原始结果：v1 的 `b120_s2/result.json`、`b136_s2_only/result.json`，v2 的 `b144_s2_only/result.json`、`b152_s2_only/result.json`；同级保留日志/IR/每 arm 每 rank trace。v1/v2 均在 `.scratch/sbm64_empty_small_20260913_*`。
- S2-only 候选：[mega_moe_stage2.py](.scratch/sbm64_empty_small_20260913_v2/s2_only/mega_moe_m3/mega_moe_stage2.py:389)，仅该文件白名单不同于 clean；`s1_only`/`both` 是未采用实验，后续不要误选。
- Harness：[bench.py](.scratch/sbm64_empty_small_20260913_v2/bench.py)、[run.sh](.scratch/sbm64_empty_small_20260913_v2/run.sh)、`make_manifest.py`、`check_clean.py`、`variants.json`。在正式树加载 clean，候选独立加载，两臂共享输入/权重和 quant buffer；保持实际 selector/ISA 门禁。
- **不能原样重跑 v1/v2：此次授权的文档修改会使其 protected-document 哈希过期。** 新建实验目录，更新当前文档保护路径/哈希及 source pins；旧 manifest/result 一律不改。v1 的 `b144_s2_only` 曾因报告外部搬移而在最后文档检查失败，无 result.json，不引用其计时；v2 重跑完整通过，搬移记录在 `document_relocation.json`。
- K3 实际模型 `inferact-kimi-k3`，本轮 session `session_65251268-36e2-4269-96a5-6bf6eb747733`，[分析日志](.scratch/sbm64_empty_small_20260913_v2/k3_s2_analysis.log)。CLI 可用 `kimi --model inferact-kimi-k3 --prompt '...'`；`--prompt` 不能与 `--auto` 同用。
- 写入前 Slurm **1841 / do-mi350x-04** 仍 RUNNING（恢复必须重查）；EP8 用 8 GPU / 64 CPU，沿用 `gpu_gate.sh` 串行门禁和已有 enroot 配置。不要干扰其他 allocation 或 kimi 会话，不自动 scancel。工作区目录 `/mnt/shared/homes/ming/msa`，主机 Python 命令为 `python3`。



## 2026-09-13 S1 packed shared-first 已提交：领取顺序保持，性能仅有单轮证据

本节优先于下方历史。用户明确授权K3审核、commit并更新文档；生产aiter **`05bbc6925`**，父 **`860181bc6`**，本地提交。

- 三文件47行新增、5行删除；融合shared EP8 b8,16,…,256默认启用 `shared_packed_heads`，S2继续jointtail。
- 每队列低16位shared、高16位routed，复用preplan每forward清零的i32 head；先扫完全部shared队列才发routed。
  保证shared有效票据先发，不保证shared计算先完成；领取次数未减。legacy shared64 buffer保留以兼容旧分支。
- SBM64的VGPR spill4→2、scratch20→12 B；全部尺寸驻留上限仍1 CTA/CU。不能把收益全部归因于atomic。
- K3无阻塞发现；生产默认入口32尺寸×8rank、8种探针、bit-exact/独立参考、graph2048、队列与epoch门通过；1280份ISA逐字节等于测量版。
- 单轮完整forward几何平均观测降低 **0.199%**；21快/3慢/8个CI跨零。仅b224/b248通过全部本轮稳定性检查，二者性能CI均跨零；**未证明跨轮稳定收益**。
  用户要求首轮完成即停，未跑第二/三轮；无新全32尺寸S1 breakdown。
- 配置记录已更新当前S1开关，历史性能字段保留。完整表格和证据：`megamoe-s1-1k-prod/bench/checkpoints/dep8_s1_packed_20260913/README.md`。


## 2026-09-13 S2 jointtail 已提交：共享 ticket 队列，小 batch 默认启用

本节优先于下方历史。用户授权32尺寸验证、K3最终复核后commit并更新文档。
生产aiter **`860181bc6`**，父 **`662b5caba`**，本地提交；唯一性能baseline是父版本默认early2。

- 三个生产文件共39行新增、8行删除；只新增jointtail和8,16,…,256的融合shared EP8默认选择。
  保留tail/early2；大batch、非shared、EP4和local reduction不变。S1/GEMM数学/P2P/Combine未修改。
- 每队列`[R_padded | S_padded]`共用现有routed32 head，每CTA只遍历8个合并队列；
  每GPU失败领取10,240→5,120。R先被领取不代表先完成，shared仍可与在途R/P2P重叠。
  b104 CTA驻留仍是2/CU；不能把收益说成新增驻留，也不能把领取次数降幅当成latency降幅。
- 全32尺寸完整路径同对象paired A/B：几何平均降低 **2.11%**；
  分开跑的rocprof S2几何平均降低 **7.15%**。各batch与CI见下方checkpoint。
- 全32×8rank的bit-exact、7种probe、独立数学参考、graph256、实际默认graph、队列终值、QA1及资源门禁通过；
  数学参考最大相对L2 **0.372%**（门限1.5%）。256份joint ISA与先前最快版完全一致。
- K3最终复核已完成；trace全部5kernel/replay记账与source pins通过；无新增lint诊断。
- 当前配置记录已切jointtail、补齐缺少的11个尺寸并明确区分旧测量字段；原记录完整快照保留在checkpoint。没有重新比较其他性能变体。

完整记录：`megamoe-s1-1k-prod/bench/checkpoints/dep8_s2_jointtail_20260913/README.md`。
原始证据：`.scratch/s2_jointtail_precommit_20260913_v1/`。历史manifest不改hash；后续复现应使用对应源码与配置快照。


## 2026-09-12 S2 全空子块跳过已提交：原外层判断保留，资源不增加

本节优先于下方历史。用户明确授权“先看下目前你的代码具体改了啥，然后commit和更新文档”。
生产 aiter 提交 **`662b5caba`**，父 **`5f469c73f`**；本地提交，未 push。
最终只改 `mega_moe_stage2.py`，15行新增、1行删除；此前所有修改均在实验副本。

### 当前生产行为

- 融合 shared L2、关闭 local reduction、`BM < SBM`，且 batch 在
  **104/112/128/160/168/176/184/192/200/256** 时启用；其余尺寸和路径保留原行为。
- 在 `run_unit` 前用现有 `routed_a_tile_bytes` 判断全空 routed 子块；shared 任务始终执行。
  增加 `_ez1` 编译标记。S1、selector、tile 几何、queue claim/遍历、shared epoch 均未改。
- **生产保留原三字段/12字节 `schedule_audit` ABI。** 实验四字段计数器及其他诊断代码没有合入。
- S1 BM64 的 compact 块至少有一行有效 token，不能整块跳过；S2 BM32 拆出全空半块后可跳过。
  例：67行，S1 `[64,3]`，S2 `[32,32,3,0]`，最后一个可跳。
- 不是“所有空转消失”：queue退出 claim、band padding、部分有效 tile 内的 padding 仍存在。

### 验证与结果

提交前最终 b104 同对象、同 buffer 全路径 A/B：**290.334→283.128 µs，快2.48%，12/12**；
95% ratio CI `[0.9727245,0.9771521]`。七种输入/路由、独立数学参考、bit-exact、graph256通过。
八卡正常S2 ISA逐字节等于此前验证版；VGPR231、next_free SGPR100（metadata106）、LDS33088B、
SGPR转存18、VGPR spill0、scratch0，均不增加。

S2专用breakdown保留完整forward，以rocprof统计S2：12组balanced pairs×200次，取各组最慢卡。
它不是无profiler整路径加速比例。

| batch | 全空GEMM任务（八卡合计） | S2生产→跳过 µs | 本轮结论 |
|---|---:|---:|---|
| 104 | 2712 | 92.390→85.862 | 快7.07%；八卡平均快11.77% |
| 160 | 360 | 95.386→95.317 | 快0.07%，CI跨零，持平 |
| 168 | 240 | 97.103→96.534 | 快0.59% |
| 176 | 144 | 96.342→95.956 | 快0.40% |
| 184 | 48 | 97.045→96.759 | 快0.29% |
| 192 | 72 | 97.364→96.879 | 快0.50% |
| 200 首轮 | 72 | 97.019→97.395 | 慢0.39% |
| 200 独立复测 | 72 | 97.434→97.099 | 快0.34% |

**b200两轮方向翻转，不能认定稳定收益或稳定回退。** 组内CI不包含进程间的最慢卡变化。
均匀top4时单expert均值=b/4；可跳过条件是 `n%64` 在1–32，每个空半块对应24个N方向任务。
低行数尾块随b增长减少，超过64行的新尾块随后出现；省计算量不等于最慢卡延迟等比例下降。
五种路由的逐任务QA1诊断均确认全空GEMM实际进入次数0，claim/entered/skipped与独立直方图一致。

### 未采用方案与后续注意

- 内移到selected A bound后判断：b160曾导致64位计算、SGPR转存18→19，资源硬检查拒绝。
- 显式32位bound修复了资源增加，但S2比较在b160比原跳过慢0.51%，b104无优势；未采用。
- 外层显式标量版也未建立性能优势。保留最初外层OR判断，不宣称全局最优。
- 源码两次routed_a_tile_bytes调用被CSE；不能再说“多一次metadata load”。
- 原双对象的b64/b96加速不能归因于跳过任务，因为没有全空子块；最终BM<SBM排除无用判断。
- b128/b256只有既有数值/graph及双对象计时证据，没有同buffer S2 breakdown，不外推b104收益。

完整说明与归档结果：`megamoe-s1-1k-prod/bench/checkpoints/dep8_s2_empty_skip_20260912/README.md`。
最终生产验证：`.scratch/s2_skip_promotion_20260912_v1/`；范围扫描与复测：
`.scratch/s2_skip_160_200_20260912_v1/`；三版breakdown：`.scratch/s2_skip_breakdown_20260912_v1/`。
历史manifest固定的是旧生产父版本，不要为了新提交重写旧hash。


## 2026-09-12 15:55 UTC 重启入口：T=200 rank4 的 27 µs 长尾已定位到机制；七个变体全部实测否决，生产树未改

用户本轮明确授权记录。**生产树 `megamoe-s1-1k-prod/aiter` 一个字节没动**（HEAD 仍是 `5f469c73f`），
本轮全部工作在 `.scratch/dep8_sweep_all_20260912_v1/` 的 kernel 变体副本里。
另新建 `trick_fused_split.md`（用户授权），是从本仓库已有的 MSA decode split-KV 工作里
挖出来的 fence/waitcnt/计数器正确性手法，本轮靠它抓到一个真 bug（见下）。

### 一、被解释的现象：T=200 的 rank4

**4/4 可复现**，每次都是 rank4：S1 = 199.4 / 200.3 / 204.5 / 198.8，其余七卡 167–183，
跨度 17.7–21.8%。T=192 的跨度只有 4.3–5.4% 且最慢的 rank 逐 run 在漂，那是噪声，200 不是。

直接原因：rank4 有两个 expert 的行数 68 和 65 > sort_block_m=64，各要第二个 tile → 18 个 routed tile。

**combine 的 42 µs 是等待重定位，不是 combine 变慢**：rank4 的 combine 全场最快（11.6），
其余七卡 34.9–42.0 都在等它。把 8 张卡的 trace 放到同一时间轴（它们
`baseTimeNanoseconds` 相同，可直接对齐）：S1 起点跨度 **2.0 µs**、S1 终点 rank4 晚 26.4 µs、
S2 跟着晚 27、combine 终点**全场 354.2–355.4（跨度 1.2 µs）**。整轮长度就是 rank4 的长度。

**quant 那一列（19.6–266.6 µs）读不出任何工作量**：`preplan.py:47-58` 在 quant kernel 的
block 0 里做全 8 卡 generation handshake（每个 rank 向所有 peer 写 launch_epoch 再等齐），
保护紧随其后的对称内存元数据复位。所以 quant 时长 = 屏障放行时刻 − 本 rank 进场时刻，
是 **CPU 侧 graph launch skew 的镜像**（route 起点跨度 86–248 µs），逐行对得上。
注意含长自旋的 kernel 其 begin 时间戳不可信（有 rank 的 quant 起点早于自己的 route 起点），
只用终点。

### 二、阈值模型（本轮的主要产出，已被 ladder 证实）

```
S1 ≈ (N × t) / 256  +  一个尾段的时长
```
N = 总 task 数 = `12 × (routed_tiles + shared_tiles)`，t = 单 task。
**claim 是 atomic 工作池不是整轮同步**（`mega_moe_stage1.py:734-762`），所以尾段的代价是
**一个 task**，不是一整轮 —— 我最初用的 `ceil(N/256)×t` 是错的，据此撤回过一次归因。

池恒 = `num_cu × grid_mult` = **256**：`mega_moe_stage1.py:485` 的 `if compact_producer:`
之后没有 early-exit，`consumer_active` 恒真，planner 和 48 个 dispatch producer 发完就流进
消费循环（`:699` 注释 "Control CTAs join the work pool after dispatch"）。**所以 `num_dispatch_cu` 改不了池大小。**

| | routed×12 | shared×12 | N | vs 256 | 实测 S1 |
|---|---|---|---|---|---|
| T=200 一般卡 | 192 | 48 | 240 | 够 | 167–175 |
| T=200 rank1（17 tile） | 204 | 48 | 252 | 够 | 172.2 |
| T=200 **rank4（18 tile）** | 216 | 48 | **264** | **超 8** | **199.4** |
| T=192 rank4（17 tile） | 204 | 36 | 240 | 够 | 169.2 |

T=200 越界要**两件事同时发生**：T 跨过 192 让 shared tile 从 3 跳到 4（+12），rank4 又恰好
两个 expert 超行（+24）。单独任何一个都不够。

**ladder T=184–248 证实**（`ladder_*`，每点全套门禁）：spread 走出抛物线
3.5% → 4.3% → **19.3% → 22.3% → 16.5% → 18.2% → 25.3%** → 9.7% → 8.0%，
少数卡越界时最大、全部越界后回落；S1_min 在 232→240 之间跳 20.7 µs（179.6→200.3）。
**T=192 是这一段的加速比峰值 1.6455**，之后单调下滑到 248 的 1.4421。

### 三、槽位为什么改不了：VGPR

ISA 实测（`ir-rank0/megamoe_stage1_*/21_final_isa.s`）：`next_free_vgpr` **256**、
`group_segment_fixed_size` 77824。gfx950 每 SIMD 512 VGPR，CTA 8 waves = 2 waves/SIMD，
2×256 = 512 占满 → **1 CTA/CU，占用率 2/8 = 25%**。

**全部 11 个生产档都是 1 CTA/CU**（16/32/96 档的 LDS 只有 38 K、允许 4 个，是 VGPR 卡住的）：

| T | nw | sbm | tile_n | gm | VGPR | LDS | 实际 CTA/CU |
|---|---|---|---|---|---|---|---|
| 16/32/96 | 8 | 32 | 512 | 1 | 232 | 38 K | 1 |
| 64/128/200/256 | 8 | 64 | 512 | 1 | 256 | 76 K | 1 |
| 512…8192 | 8 | 128 | 256 | 1 | 248–250 | 88 K | 1 |

这**不是回归，是从 `d1dc744fa`（M3 算子诞生）起的设定**：`waves_per_eu_hint` 默认 2，
经 `"rocdl.waves_per_eu"`（`mega_moe_stage1.py:955`）让编译器放开到 512/2 = 256 VGPR。
`handoff_megamoe.md:556` 那句"2 CTA/CU 正是设计稳态"描述的是意图，不是编译出来的东西。

VGPR 组成（结构推算 + ISA 佐证）：`M_REPEAT×NUM_ACC_N = sbm·tile_n/(256·nw)` 三种几何都等于 16，
所以**累加器恒 64 VGPR，只占 25%**；真正吃掉的是 **B 的流水双缓冲 128**（ISA 里 MFMA 的 B
操作数跨 v132–v251 共 120 个，是 state 里 64 的两倍）、A 操作数 32、scale 4，其余 ~28 是地址/swizzle。
**用户已明确否决"砍掉 B 双缓冲"。**

### 四、七个变体，全部实测否决（T=200，同窗口，生产基线 B=330.47 / 1.5731×）

| 变体 | 整层 B | 加速比 | S1 min→max | 机制 |
|---|---:|---:|---|---|
| `kernels_s128` sbm=128 | **449.81** | 1.1554 | 272.5–285.6 | 不均衡消除（跨度 4.8%）但每卡 +105；LDS 76 K→152 K |
| `kernels_mskip` 跳 padding 的 MFMA | 329.83 | 1.5757 | 166.0–198.8 | **成本是 B 的 HBM 读不是算**；T=96 同样持平（274.12 vs 274.48） |
| `kernels_shm` 合并 shared 的 m_tile | **524.35** | 0.9900 | 372.1–385.4 | 槽位 264→228、跨度 3.6%，但合并把固定开销乘进关键路径 |
| `kernels_tn256` split-N（tile_n 512→256） | 335.96 | 1.5495 | 176.2–200.7 | 尾段减半但 A 重读 12→24 次，L2 没兜住；rank4 纹丝不动 |
| `kernels_splitk` split-K，SPLIT_TILES=2 | **362.35** | 1.4326 | 212.8–226.6 | 见第五节 |
| `kernels_spkt0` 对照：票翻倍但一个不切 | 336.09 | 1.5428 | **166.2**–204.4 | **ticket 空间翻倍是免费的** |
| `kernels_spk1` split-K，仅溢出时切 1 个 tile | 347.78 | 1.4940 | **169.6**–213.7 | 六张无溢出的卡零成本，rank4 199.4→213.7 |

`kernels_mskip` 的改动确实进了 code object（`s_cbranch` 47→59 = 3 个 mi 块 × 2 ksub × 2 调用点，
kernel 名带 `_mskip1`），不是编成了同一个 kernel。

### 五、split-K：正确性全过，但单价结构性地高于收益（本轮最大的一块工作）

实现（`kernels_splitk`，三步，每步可单测）：
1. `gemm1.do_tile` 参数化成绝对 K 窗口 `[k_lo, k_lo+k_steps)`，默认值与原版等价。
   循环变量**相对**（A 的 LDS ping-pong 相位必须从 ping 起），操作数取数**绝对**（`k_lo + sp`），
   两者只在 k_lo 为偶时自洽 —— 已落成断言。
2. split-K epilogue：fp32 累加器按**寄存器序**停放（lane `tid` 连续写自己的 N_ACC 个 float4，
   一个 slot = `split_k × total_threads × N_ACC` 个 float4），两边都不需要知道 MFMA 的输出布局；
   `atomic_add` 选举，当选者把所有片**按 ks 顺序**从内存读回求和（**包括自己刚写的那片**，
   所以结果与谁当选无关，确定性只靠固定顺序，计数器只用来选人）。
3. claim 编码 `work = (m*N_TILES + panel) * SK + slice`，**slice 在低位**（见下）。

**全套门禁通过**：数值、256 次逐位 graph replay、L2/L13 epoch、五个 dispatch 倾斜探针。
不均衡确实消除（S1 跨度 19.3% → 6.5%）。

**但三段成本分解（靠三个单变量对照测出来，不是估的）**：

| | 代价 | 怎么测的 |
|---|---:|---|
| ticket 空间翻倍 | **~0** | `spkt0`（SPLIT_TILES=0）S1min 166.2 vs 生产 167.1 |
| **K 切分 + partial 往返** | **+24.6 µs** | `spknf`(190.8) − `spkt0`(166.2) |
| agent-scope fence | **+22 µs** | `spk`(212.8) − `spknf`(190.8) |
| 尾段收益 | **11 µs**（22→半片） | |

单价：**切一个 task 1.03 µs、一次 fence 0.46 µs**（每 task 两个半片 → 0.92）。
必须切的 task 数 ≥ 8（溢出量），tile 粒度下 12 → 成本 ≈ 23 µs 对收益 11 µs。
对症版 `spk1` 实测 rank4 199.4 → **213.7**（外推预测 210，吻合）。**恒亏，且两项都压不下去。**

**关键机制（本轮第二个主要产出）**：
> **合并** 4 个 task 只慢 **2.2×** 而不是 4×（`shm`）；**切分** 1 个 task 成 2 片，每片却接近
> **1×** 而不是 0.5×（`splitk`）。同一个常数从两个相反方向各测到一次：
> **一个 task 的时间里约一半不随工作量缩放**（A 的 LDS staging、流水预热、epilogue、barrier）。

这一条解释了本轮全部七个失败：**task 既不能合并也不能切分**，而尾段那 22 µs 的上限两个方向都够不着。

### 六、trick_fused_split.md 与它抓到的真 bug

两个 K3 从 `handoff_decode_msa_pr.md` / `handoff_kv_outer.md` / `HANDOFF.md` 和 vLLM fork 里
挖出本仓库已有的 split-KV 正确性手法，写成 `trick_fused_split.md`（449 行，带 file:line 原文摘录）。
本轮直接命中三处：

1. **`buffer_wbl2`**：A1 明写"不用 `buffer_wbl2`（agent-scope release 的 lowering，**实测 42.8 µs**）"。
   我的 `comm_ops.fence_agent_release()` 正是 lower 成 `buffer_wbl2 sc1`（ISA 对比：生产 wbl2=1/inv=2，
   split-K 版 wbl2=2/inv=3），实测 +22 µs。
2. **发布链的顺序**：`s_waitcnt vmcnt(0)`（per-wave）→ `s_barrier`（workgroup 组合）→ 单 wave 发 atomic。
   文档记着一个已修 commit `ddc5041019 修 bug：barrier 必须在 per-wave wait 之后`。本实现顺序正确。
3. **A5 的第 3 条坑**：「反复 replay 同一输入的投毒测试，读到上一次的 partial 也是同样的数，
   探测的是空的」→ **ordering 的正确性只能靠 ISA，不能靠 replay 门禁**。
   本轮对此有一个独立实证：去掉两个 fence 的 `spknf` 变体，8 个 rank 在**第一个**数学检查就全错
   （`independent complete math: [False ×8]`）——**跨 XCD 是真实发生的，不是理论风险**。

**同 XCD 的两难（split-K 无法翻盘的结构性原因）**：要缩短尾段，split 必须落在最后被领取的
task 上；而最后被领取的恰恰是队列抽干后被别的 XCD 的 CTA 轮转拿走的那些
（`mega_moe_stage1.py:718-720`，`home_queue` 按注释只是 locality hint）。把 split 挪到 claim
顺序最前能拿到同 XCD，但那样它就不在尾部，尾段一微秒不减。decode 那边能免 fence 是因为它能用
`grid.x` 是 8 的倍数**造出** 100% 同 XCD（文档强调"padding 是硬要求，不是保险"），这个 kernel
的动态工作池没有等价杠杆。

**踩过并已修的实现坑（重启后不要重犯）**：
- FlyDSL 的运行时 `if` **只在 `@flyc.jit` 装饰的函数体里有效**，普通方法里写会报
  `cannot evaluate dynamic 'Boolean' as Python bool during tracing`。
- AST rewriter 把 `if` 分支里**被赋值的名字**收集成 scf.if carried state
  （`ast_rewriter._collect_assigned_vars`），而且 `visit_Call` 把**任何 `obj.method(...)` 的接收者**
  也算进去（`:205-208`）——所以 `epi.store(...)` 会让 `epi` 成为 carried 并报
  `Cannot extract IR values from SiluQuantEpilogue`。修法：分支体内零赋值、零属性调用，
  把 epilogue 包一层普通函数。
- 计数器用 `arrived % split_k` 判断是**有符号取余**：越过 INT32_MAX 后永远选不出归约者，
  输出从此不写、kernel 不挂、静默错。必须用 `arrived & (split_k-1)`。
- flag 经 LDS 广播后**必须再加一个无条件 barrier**，否则 epilogue 的 cshuffle 会在慢 wave 读 flag
  前覆盖那 4 字节 → `elected` 在 CTA 内发散 → epilogue 的 barrier 死锁。
- 模块常量 `SPLIT_K` 必须按 `shared_l13 and small_xcd` 门住（`SK = const_expr(...)`），否则
  非融合 small-XCD、大 regime（512–8192 生产默认）、BAND_M==1 三条路径全部静默算错。

### 七、harness 新增：`_timing_only`（在实验副本 `bench.py` 里，未进 pin 列表）

override.json 里点名 `"_timing_only": true` 时，把**数值类**门禁
（label 含 `math`/`versus`/`bit exact`/`refreshed`）降级成记录一条 `GATE_SOFT_FAIL` 并继续跑到计时。
**结构性门禁照旧硬失败**（变体文件清单、source pin、selector 默认值、epoch 对齐）——
那些破了的话这次跑测的就不是它自称的 kernel。结果文件带 `passed=False` +
`soft_gate_failures` + `manifest.experiment.timing_only`，不可能被误引为验证过的结果。
本轮靠它拿到"去掉 fence 的性能"这个否则拿不到的数。

### 八、未完成 / 下一步

1. **`spk1` 的 rank1 = 187.3 未解释**：它 17 tile / 252 task < 256，`overflow = -4`，按 gate
   不该被切，六张同类卡都正常。它在生产基线四次重复里本就最飘（172.2/180.8/183.5/171.7）。
   单点，未追。
2. **T≥232 那一段是另一个问题**：S1_min 在 232→240 跳 20.7 µs，是**整体抬高**不是卡间不均衡，
   本轮所有手段都不对症。
3. 若要再碰这条线，唯一没被堵死的是**动 VGPR 预算**（B 双缓冲 128 → 腾给第二套累加器，
   使一个 task 覆盖同 expert 的两个 m_tile，task 数变成 `12×(#experts+shared)` 恒定）——
   用户本轮已否决砍双缓冲，重开需要先推翻那个否决。
4. `dcu=64 @ b512` 的倾斜正确性 bug 仍未修（见 2026-09-11 17:30 节第三小节）。

### 九、证据目录

`.scratch/dep8_sweep_all_20260912_v1/`：
- 变体 `kernels_{s128,mskip,shm,tn256,splitk,spknf,spkt0,spk1}`，`variants.json` 记录每个的 differs
- 点 `s64r_*`（18 档 sweep）、`rep{200,192}_{1,2,3}`（复现）、`ladder_{208..248}`、
  `probe200_s128`、`mskip_200`、`ms96_{p,m}1`、`shm_200`、`tn256_200`、
  `spk_200`、`spknf_200`、`spkt0_200`、`spk1_200`
- `bench.py` 含 `_timing_only` 与按变体参数化的 `l13_period`
- K3 审查记录在 scratchpad（rev1 K 窗口、rev2 partial/fence/死锁、rev3 claim 覆盖性、
  rev4 split-M vs split-K；sk_a/b/c split-K 三视角）
外层 `trick_fused_split.md`。Slurm 1824（do-mi350x-03）本轮一直在用；另有 1826 在 node04，不是本会话的。


## 2026-09-12 DEP e2e 交接补充：node04 已释放

用户明确授权更新。DEP serving 集成的最新状态以
[handoff_dep.md 最新交接](handoff_dep.md) 为准；不覆盖下方独立 kernel 优化线的提交记录。

- 成功 e2e 工作树是 `dep-dep8-mtp`（HEAD `dd4845624a` 加未提交 MoE 接线），不是 `/dep`；
  只动 MoE 相关生产接线，未改 scheduler、attention 或 MTP forward。
- 每卡 16 并发、MTP7、B128：原始 baseline P50 TPOT **10.7530 ms**，当前集成
  MegaMoE **9.4337 ms**；两边 800/800、零抢占。单次相邻无插桩对比，非多轮统计。
  运行版本固定在各 run 的 `manifest.json`，**不表示已同步下方优化线后续所有提交**。
- 当前 Mori heap **4G** 已跑通；T128 target+draft 的对称工作 buffer 合计约
  **2.405 GiB/卡**，其中 draft 约 **0.263 GiB**，不是 draft 需要 40G。
- 新 MegaMoE breakdown 已得到 B128 target graph **43.2921 ms**，其中
  MoE **23.0924 ms**；baseline 历史 breakdown 曾成功，但本轮复测 warmup 异常慢，
  主代理中止，尚未定位根因或完成同口径配对。每卡 25 并发尚未测。
- 用户已要求停止排查；本轮 K3 已停止，**Slurm job 1822 / do-mi350x-04 已 CANCELLED**。
  下文引用该 allocation 的命令仅为历史，不要自动恢复运行。

## 2026-09-12 05:20 UTC 重启入口：任意 8 的倍数 batch 已支持并提交；sort block 要装下一个 expert

用户本轮明确授权提交并更新本文件。生产树 `megamoe-s1-1k-prod/aiter` 分支 `s1-unified-production`，
父提交 `dfad71696`，本轮提交 `5f469c73f`（6 文件 +121/−24）；外层 `megamoe-s1-1k-prod` 另有 `4288737`：`bench/dep8_best_configs.json` 新增 72–200 的 15 条记录。
（外层 `HANDOFF.md` 有未提交改动，不是本会话的，未动。）

### 现在支持什么

**每 rank 的 batch 可以是 8 到 256 之间任意 8 的倍数**（32 个尺寸），不再只有 6 个 2 的幂加 96。
64–200 的 18 个点全部实测通过全套门禁（数值、逐位 graph replay、L2/L13 epoch、dispatch 倾斜探针、
kernel 名字标记）。16/32/64/96/128/256 六个既有尺寸的 selector 默认值与 code object 逐字节不变。

**生产树回归**（用与新生产逐字节相同的变体，四点全部 `passed=True`）：96 274.53 µs/1.4134×、
128 299.26/1.4615×、104 290.37/1.3619×、200 335.23/1.5502×。96 和 128 的
`selector 默认值 == 记录的最佳配置` 门禁是**活的并通过**；补完配置表后 104 的门禁也已转为活的并通过
（291.01 µs/1.3628×）。

### 唯一的行为性改动是一行

`mega_moe_stage1.py` 的 shared tile 计数：batch 不被 sort block 整除时改用 ceil。

```python
if const_expr(int(fuse_mtpr) % sort_block_m != 0):
    shared_m_tiles = (i32_cur_tok + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
else:
    shared_m_tiles = i32_cur_tok // fx.Int32(sort_block_m)   # 现有尺寸走这条，逐字节不变
```

原来的 floor 是那条整除断言存在的**唯一**理由：少算一个 tile 不会报错，而是让
`is_shared = uniform_work < shared_work` 把那 12 个任务误判成 routed，拿 shared 的索引去查 routed 的表。
新增 kernel 名字标记 `_sct1`（只在 `mtpr > sort_block_m 且不整除` 时出现，既有尺寸名字不变）。

其余改动都是白名单/账本，无 codegen 变化：`SHARED_FUSED_MTPR_SMALL` 扩到 8..256、
`SHARED_SMALL_SORT_BLOCK_M` 换成规则函数 `small_sort_block_m`、`SHARED_L2_S2_SHAPES` 按规则生成、
`stage1.py` 的整除断言换成 `% 8`、small_xcd 的 token 白名单、`stage2.py` 的 `max_tok % 16 == 0` 降到 `% 8`、
combine 两处 token 白名单、`select_mega_moe_config` 新增 `fixed_slot` 形参。

### 为什么 `% 16` 可以降到 `% 8`

git 查证：它原本是 `max_tok % (BM * band_m) == 0`（=256），在 `db9ccdfaa` 加 16–128 小尺寸时被
**削弱成圆整的 16**，同一提交已把 shared m_block 数改成 ceil 并加了 `m_block < current_m_blocks` 守卫。
Stage2 内部没有任何 16 行粒度的不变量：队列/ticket 全是 ceil 运算，短尾读由 `_a2s1` 界住，
epilogue 的写由 `num_records = max_tok*N_OUT*2` 界住。

### 短尾路径不是冷代码

**b16 每天都在跑它**：16 % 32 ≠ 0，所以生产的 b16 S2 kernel 名里一直带 `_a2s1`
（实测 `s16/ir-rank0` 确认）。104 只是把垃圾行从 16 行变成 24 行。整条数据通路的界早已齐全：
`_atb1`(A 读) / `_asb1`(A-scale) / `_svb1`(A2 写，shared 走指针表 slot 7 复用 routed 的 tile_valid_rows) /
`_a2b2`(routed A2 读) / `_a2s1`(shared 短尾读) / epilog 的 output num_records。

### 本轮最重要的发现：sort block 必须装得下一个 expert 的行数

**不是「块越小 padding 越少越好」——反了。** Stage1 把每个 expert 补齐到 sort block 的整数倍，
tile 数 = `Σ_e ceil(rows_e / sbm)`，而 `rows_e` 是随机量（EP8/top4/128 experts 下均值 = T/4，
标准差约其平方根）。当均值落在块大小的一两个 σ 之内，「这个 expert 要不要第二个 tile」就成了掷硬币，
**每个 rank 的 tile 数不再相同**；而每个 tile 要完整走一遍该 expert 的 37.7 MB GEMM1 权重，
所以 tile 数就是 Stage1 的成本，整层由最慢的 rank 决定。

T=104（均值 26 行对 32 行块）实测：

| | 每 rank tile 数 | Stage1 每 rank | 整层 B |
|---|---|---|---|
| sbm=32 | 17, 19, 18, 19, 19, 17, 17, 17 | 155 **193 193 191 190** 157 161 156（跨度 24%）| 313.2 µs |
| sbm=64 | **16 ×8** | 155 159 154 157 155 157 152 157（跨度 4%）| **290.0 µs** |

**sbm=64 多算了近一倍的 padding 行还快 23 µs**，因为 padding 行的读写已被界掉、不产生 HBM 流量，
只花 MFMA 周期，而 Stage1 远不是矩阵管受限（b256 实测占用 18.7%）。

规则落在 `mega_moe_config.small_sort_block_m`：`64 if (mtpr % 64 == 0 or mtpr > 96) else 32`，
逐点复现既有 6 个条目（16/32→32、64→64、96→32、128/256→64）。

**排除法**：这个 4 对 4 的分裂不是代码按 rank 分叉。三个独立证据：
(1) 9 次 T=96 的运行、3 个 kernel 变体，慢的永远是 {1,2,3}；
(2) T=112 时变成 rank 6 独自快、T=1024 时变成 rank 2 独自慢——**换 T 就换人**，结构性依赖做不到这点；
(3) 实测的 tile 表与每 rank 的 Stage1 时长逐点单调对应。
路由输入由 `seed + 31*rank` 固定（`shared_inputs.py:104`），所以同 seed 逐 run 复现。

### 实测表（64–200，每档只列最快的那个配置）

| T | A µs | B µs | 加速比 | route | S1 | S2 | combine | sbm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 347.89 | 271.32 | 1.2822× | 5.2 | 159.6 | 81.4 | 18.5 | 64 |
| 72 | 348.39 | 260.11 | 1.3394× | 5.3 | 152.6 | 79.4 | 17.8 | 32 |
| 80 | 364.33 | 262.07 | 1.3902× | 7.2 | 153.4 | 81.2 | 20.5 | 32 |
| 88 | 384.51 | 270.80 | 1.4199× | 6.2 | 158.9 | 78.9 | 20.5 | 32 |
| 96 | 389.23 | 269.71 | 1.4431× | 6.1 | 160.2 | 82.6 | 26.6 | 32 |
| 104 | 396.57 | 289.97 | 1.3676× | 6.0 | 158.5 | 96.4 | 19.1 | 64 |
| 112 | 401.56 | 292.01 | 1.3751× | 7.7 | 159.8 | 97.9 | 17.2 | 64 |
| 120 | 424.97 | 293.88 | 1.4460× | 5.8 | 161.8 | 99.0 | 16.7 | 64 |
| 128 | 437.35 | 296.98 | 1.4726× | 5.6 | 161.1 | 99.4 | 19.8 | 64 |
| 136 | 462.02 | 296.47 | 1.5584× | 6.7 | 163.7 | 98.5 | 19.7 | 64 |
| 144 | 468.72 | 297.57 | 1.5752× | 6.4 | 163.2 | 98.4 | 18.4 | 64 |
| 152 | 478.12 | 301.74 | 1.5846× | 5.3 | 161.6 | 101.7 | 18.4 | 64 |
| 160 | 483.96 | 304.49 | 1.5894× | 6.0 | 166.4 | 101.1 | 24.3 | 64 |
| 168 | 489.57 | 305.93 | 1.6003× | 5.4 | 165.2 | 99.2 | 18.2 | 64 |
| 176 | 490.97 | 305.68 | 1.6062× | 5.4 | 162.0 | 100.6 | 16.0 | 64 |
| 184 | 494.70 | 309.39 | 1.5989× | 6.0 | 166.5 | 102.7 | 19.0 | 64 |
| 192 | 511.32 | 310.74 | 1.6455× | 6.4 | 169.6 | 101.4 | 18.6 | 64 |
| 200 | 519.87 | 330.47 | 1.5731× | 5.2 | **199.4** | 101.8 | **42.0** | 64 |

breakdown 取 8 rank 的 max；**quant 没列**——preplan_waves=4 把跨 rank launch barrier 放在那个 kernel 里，
它的 66–307 µs 是等待不是工作量。**S1 在 104–192 全段只从 158.5 涨到 169.6（+7%）而 T 涨了 85%**，
这一段 Stage1 由「16 个 expert 各一个 64 行 tile」的固定开销主导。

### 未完成 / 下一步

1. **200 是同一个边界在上一级重演**：rows/expert = 50、σ=7，开始有 expert 超过 64 行，实测 tile 数
   16,**17**,16,16,**18**,16,16,16，S1 跨度回到 19%、combine 跳到 42 µs，B 比 192 高 20 µs。
   **200 附近该测 sbm=128**（512 档用的就是 128）。
2. **104/112 仍低于趋势**（1.368/1.375，而 120 已是 1.446），sbm=64 没有完全补平，原因未定位。
3. **BM 的取舍在 96–200 从未测过**：SBM=64 让 BM=64 第一次合法。两个自洽候选互斥——
   `(BM=32, BN=256, use_nt=False)`（现在用的，沿用 128/256 档）与 `(BM=64, BN=128, use_nt=True)`
   （沿用 64 档，R_S2=1 让每个 B2 panel 只有一个消费者）。两者单元总数相同，差别是把流量压在 A2 还是 B2。
4. 8–56 这一段（8/24/40/56）代码已支持但**未测**。
5. K3 顺带发现的既有问题（与本轮无关）：dispatch 尾轮不均——每个 destination 的最后一轮只由 3 个
   producer 承担，slot d+32/d+40 在第三轮空闲，理论上可把尾部从 3τ 降到 ~2.7τ。

### 证据目录

`.scratch/dep8_sweep_all_20260912_v1/`：`points.txt`（11 档生产基线）、`points_64_200.txt` +
`SWEEP_64_200`（18 档新 sweep）、`VERIFY_PROD2`（生产回归）、`kernels_m8`/`kernels_m8s64`/`kernels_prod2` 三个变体、
`rankload.json`（每 rank 实测路由行数与 tile 数，`.scratch/rankload/`）。
Slurm 1824 到 2026-09-12 05:00 后仍在。

## 2026-09-12 02:05 UTC 重启入口：padding 加界 + 96 默认配置已提交；下一步是 A 的重读倍数

用户本轮明确授权提交并更新本文件。生产树 `megamoe-s1-1k-prod/aiter` 分支
`s1-unified-production`，父提交 `6051eef2242b812e6bd0dac7195942333d08bc3c`，本轮两个提交：

| 提交 | 内容 |
|---|---|
| `dd3a0b3a2` | 四个 padding 边界 + 非 2 幂 batch 支持（9 文件 +326/−66） |
| `dfad71696` | 96 的选择器默认 `block_n` 128→256（2 文件） |

外层 `megamoe-s1-1k-prod` 另有 `f9d21a1`：`bench/dep8_best_configs.json` 新增 96 条目。
（外层的 `HANDOFF.md` 有未提交改动，不是本会话的，未动。）

**注意上一节的 DEP 迁移以 `6051eef22` 为源，本轮两个提交都在其之后，迁移时需要重新取源。**

### 每个 batch 都默认最佳配置：现已成立且被门禁保护

`bench.py` 有一道门 `selector default == recorded best + declared override`，对表里有条目的
尺寸每次运行都会校验。本轮之前表里只有 2 的幂，**96 是唯一的盲区**，门禁对它是 SKIP。
补上 96 后该门在 96 上实际执行并通过（`b96_default_v1`，skipped-count 0）。

现已核实：**11 个支持尺寸（16/32/64/96/128/256/512/1024/2048/4096/8192）的选择器默认值
全部等于记录的最佳配置。** 另外复核了 `preplan_waves`：默认的 4 在全部 10 个 2 的幂尺寸上
都更快（2048/4096 的 0.83/1.14 µs 在漂移内，其余明确），此前"8192 符号翻转"的记忆是错的，
那是 histogram XCD 复制那条线的结论。`local_reduce` 保持关闭、`shared_schedule` 保持 early2、
`p2p_quant` 保持 none，本轮都有实测支撑（见下）。

### 这一轮做了什么

Stage1 把每个 expert 的路由行补齐到 `sort_block_m` 的整数倍，padding 严格位于该 expert
行区间的尾部（`dispatch.py` 只在 `[local_row_base+total_count, local_row_base+padded_rows)`
写 padding），所以 per-tile 的 `num_records` 界正好切在最后一个真实行上。随机路由下
`E[padding] ≈ sort_block_m/2` 每 expert，**与 batch 无关**——T≥512 每 rank 恒定约 1000 行，
在 Stage2 的 ×48 n-panel 倍数下约 150 MB。（此前认为"8192 的 padding 只有 3% 所以没东西可省"
是错的：比例掉了，绝对字节没掉。）

四个界，每个带 kernel 名字标记，使"改了但没进代码对象"可被检出（第一次做 A payload 界时
正是编成了与生产逐字节相同的 kernel 且通过了全部门禁）：

| 标记 | 位置 | 内容 |
|---|---|---|
| `_atb1` | Stage1 | A payload 读，界 `valid_rows*model_dim` |
| `_asb1` | Stage1 | A-scale 读，每 tile 在自己的基址上重建描述符 |
| `_svb1` | Stage1 | A2 payload+scale **写**，padding 行重定向出界 |
| `_a2b2` | Stage2 | A2 payload 读，`clamp(valid_rows - row_in_sort_block, 0, BM)` |
| `_a2s1` | Stage2 | shared panel 尾部界；`const_expr` 在 BM 整除 max_tok 时短路，今天所有尺寸都短路，为"8 的倍数"预留 |

Stage2 的界必须 `readfirstlane`：否则 `num_records` 发散、整个 128-bit 描述符 waterfall，
编译器在 direct-to-LDS 的 A2 load 外面生成 **12 个** `v_cmp_eq_u64 / s_and_saveexec / s_xor exec`
循环，代价超过界省下的量。ISA 上可见：24→0、`s_cbranch_execnz` 12→0，而 `v_med3_i32` clamp 保留。

同时并入非 2 幂 batch 支持（`div_max_tok` 取代 shift/mask、`SHARED_SMALL_SORT_BLOCK_M`、
shared row-stride assert），96 因此原生运行而不是 padding 到 128。

### 测量（EP8 MI350X，每点 12 组配对，12/12 更快，数值逐位不变）

| T | 生产 | 本轮 | Δ |
|---|---|---|---|
| 512 | 558.8 | 524.3 | **−34.5 µs (−6.2%)** 两组分布完全不重叠 |
| 1024 | 830.1 | 799.7 | **−30.4 µs (−3.7%)** |
| 2048 | 1289.2 | 1269.6 | **−19.7 µs (−1.5%)** |
| 64 | 277.8 | 271.6 | **−6.2 µs (−2.2%)** |
| 96 / 128 / 256 / 8192 | — | — | 在跨轮漂移内，不成立 |

96 的 `block_n` 另行扫定（`dfad71696`）：交错 128/256/128/256，**BN=128 282.025 µs
（跨度 0.54）vs BN=256 273.566（跨度 5.0），间隔 8.46 µs 不重叠**。96 是取 256 的四个尺寸里
margin 最大的，因为它双重吃亏：Stage2 A2 的重读倍数 `model_dim/block_n` 是 48 对 24，而
96 的 32 行 sort block 对约 24 行/expert 又是小尺寸里 padding 比例最高的。
**但 96 只扫过 `block_n` 一个维度**，其余字段（sort_block_m / block_m / num_dispatch_cu /
b_nt / use_nt）是从 16/32 档继承的，不是在 96 上实测的——96 的 rows/expert 是 24，而 16/32
是 4/8，差 3–6 倍，所以这是假设不是证据。最佳配置表的 96 条目里已写明，这是下一轮最容易挖的地方。

归因已用单项变体测定：**512 的收益全部来自 Stage2 的界**（只带 Stage1 A-payload 界的变体在
512 测 559.1/556.4，与生产无异）；**64 的收益全部来自 Stage1 的界**。两者在不同尺寸生效，
因为省下的字节只有在真会走 HBM 时才值钱——T≤256 时实测收益的隐含带宽是 19–104 TB/s，
对 7 TB/s 的机器物理不可能，说明那些读本来就是缓存命中；8192 上同样有 150 MB，但 kernel
跑 21 tile/CTA、带宽有余量，省下的请求换不成墙钟时间。

### 量纲：跨轮漂移远大于轮内 cv

两个 Stage1/Stage2 ISA **逐字节相同**的 build 在 T=128 测得 295.188 与 296.727，
**漂移 1.54 µs / 0.52%**；T=512 生产四点极差 7.6 µs。轮内 cv（T≤256 时 0.1–0.2%）严重低估噪声。
**T≤256 小于约 1.5 µs、T≥512 小于约 8 µs 的效应，单点对单点不成立。** 本轮据此撤回过一个
"b128 −1.6 µs"的结论。

### 还没做的（按字节排，都不是 num_records 能解决的）

1. **Stage2 读 A2 重复 48×** —— 453 MB demand，理想 9.4 MB，per-XCD L2 地板 8×=75 MB。
   倍数 = `model_dim/BN`；BN 翻倍即砍半，但 `BM=64` 只能配 `BN=128`（`BM=64,BN=256` 的
   累加器正好 64 KB 超 LDS，见 `mega_moe_stage2.py` 注释），必须同时把 BM 降到 32。
   B2 流量不会因此翻倍：`band_m=16` 覆盖 16 个 m_block，而一个 expert 在 512 下只有 6 个。
   **挡路物**：`SHARED_L2_S2_SHAPES[512]` 只有 `(BN=128,BK=128,BM=64,SBM=128)` 一项。
   估算收益 −16 ± 4 µs。
2. **Stage1 读 A 重复 24×** —— 453 MB demand，地板 151 MB。`_decode_xcd` 的 band 顺序在两次
   复用之间冲掉 3–5 MB。`tile_n` 256→512 可砍半但 LDS 预算不够。

两项各 453 MB，而整层"有用"流量（W1 604 + W2 302 + ~90 MB 激活）约 0.9 GB，同一量级。

### P2P：已接近最小，三项均已定论，不需再查

T=512 每 rank 每层 **33.82 MB**（dispatch payload 12.58 + scales 0.39 + wts/srcmap 0.016 +
scatter 20.83 + 握手 ~2 KB）。Combine 读 partial 的 25.2 MB 是**本地读**，对端 push 进
自己的对称内存，不过 xGMI。

1. **scatter 重复**：`local_reduce` 在 ≥512 一直开着（`bench_large.py` 默认 True，kernel 名字
   带 `_lr1_xl1`），2048 条 route 已压到 1695 个不同 (token,rank) 对，省了 4.34 MB。**已在用。**
2. **fp8 partial**（可省 7.85 MB / 23%）：`mega_moe_config.py` 记录它在跨 rank 归约前量化，
   relL2 从 0.0027 跳到 **0.0266**，门限 0.015。并与 `local_reduce`、`shared_l2` 融合、调优预设、
   harness 逐位门四重互斥。**测过并被否决，不要重开。**
3. **dispatch payload 重复**（可省 2.24 MB / 6.6%）：去重必然把 GEMM1 的 A 读变成逐行 gather，
   会摧毁本轮加的 per-tile `num_records` 界；接收端填充变体付 4.5 MB 本地 HBM 省 2.24 MB xGMI；
   顺序重排不可能（行按 expert-major 分组，每 tile 一个 expert，而同 token 的重复必然跨 expert）。
   量级仅占每层 906 MB 权重流量的 0.25%。**不做。**

判不了、需要 profile 的：dispatch 的发送是否在关键路径上，还是被 tile-ready 流水藏在 GEMM1 底下。

### local_reduce 在小 batch：本轮首次实测，是净亏

此前"只在大 batch 有收益"的说法在代码里找不到测量支撑；实际是小 batch 从未在 LR 打开的前提下
验证过，被五层挡住：`mega_moe_m3.py` 调优预设的 `not local_reduce`、xcd_local 要 persistent S2、
`mega_moe_stage2.py` 的 early2 几何白名单 `small_ok` 要 `not local_reduce`、`bench.py` 的逐位
校验器只描述 LR 关闭的 partial 布局、以及 rank 归一化正则漏了 `_route_r`。后两层
`bench_large.py` 早已正确，是 `bench.py` 没跟上。

用 `shared_schedule='tail'` 两臂一致（绕开 early2 白名单，不放宽任何断言）测得：

| T | LR off | LR 单开 | LR + xcd_local |
|---|---|---|---|
| 64 | 267.750 | +25.82 | bucket=64 无 persist，开不了 |
| 128 | 294.146 | +16.18 | **+2.34** |
| 256 | 389.953 | +45.70 | **+8.74** |

**xcd_local 是关键**（跨 XCD staging 占了大部分损失），但即使全开仍是亏。结论成立：
小 batch 不要开 local_reduce。`bench.py` 已补 `_local_reduce` / `_local_reduce_xcd` 两个独立
override（默认 False，既有结果含义不变）与 LR 感知的校验器。

副产品已验并否定：曾观察到 `tail` 在 b64 比 `early2` 快 3.86 µs（单点对单点），交错复现 3 对 3 后
不成立——early2 均值 272.416 跨度 1.78，tail 均值 270.702 跨度 **5.54**，完全包住前者，A/B 比范围
也重叠。那 3.86 µs 是漂移。**反向结论成立且有用：early2 的可复现性比 tail 好一个数量级**
（A/B 比跨度 0.0014 vs 0.0239，17 倍），小 batch 预设选 early2 是有依据的，即使均值无差别。
这条也是"单点对单点不能下结论"的又一个实例。

### TODO

- 支持所有 8 的倍数（不只是 2 的幂）。`_a2s1` 已为此预留；`.scratch/k3_lr96/` 有一份给
  local_reduce 补非 2 幂支持的改动（keyword-only 参数、条件标记 `_lrnp1`、构造函数审计已做），
  **未编译验证**，需确认 pow2 尺寸下 ISA 逐字节不变。鉴于 local_reduce 在小 batch 是亏的，
  它的价值是移除假约束而非解锁收益。
- 上面"还没做的"两项 A 重读。
- 一次 PMC（TCC miss on the A2 buffer）可以分离 512 上字节模型低估 37% 的原因：边际带宽更低
  vs 每 tile 访存延迟暴露在关键路径（1.9 tile/CTA 太浅）。


## 2026-09-12 00:28 UTC 重启入口：`6051eef22` 正在迁入 DEP；正确性已过，迁移后 latency 未测

本节优先于下方 MegaMoE 调优历史。用户本轮明确授权更新 handoff 后重启。迁移目标位于
`/mnt/shared/homes/ming/msa/dep`，分支 `codex/m3-ep-experiments`，父提交
`d17a65bce797516af508c5c7f370be1f28bd9ff7`，尚未 commit/stage；来源固定为本仓库提交
`6051eef2242b812e6bd0dac7195942333d08bc3c`。完整迁移状态、文件白名单、测试结果和恢复步骤
以 outer `handoff_dep.md` 顶部 2026-09-12 节为准。

迁入 DEP 的准确流水是 5 个主计算 kernel：TopK（融合 local histogram/sync 状态）→
MXFP8 quant+preplan → Stage1 → Stage2 → Combine；没有独立 sync kernel，但适配层最后仍有
`out.copy_` device node。新 MoE 路径已去掉 AITER Python 依赖，固定 FlyDSL 0.3.2 +
Mori exact ABI，并 fail-closed 到 gfx950 单节点 EP8/TP1/DP8/T16/H6144/I3072/E128/top4。

cleanup 后已通过 8 卡 lifecycle/全 kernel compile、eager 数学和输入切换后的 16 次 graph
replay；最差 eager rel-L2 `0.0033197198063`、max-abs `0.046875`，graph 对 eager 8 rank
均为零差异。CPU 测试为 41 + 29 passed；Ruff/format/diff/source patch verifier 全过，K3
终审未发现剩余 P0/P1。

**不要引用本文件下方约 251.8 us / 1.05x 的 b16 历史数字作为 DEP 迁移后的结果。**
当前新 DEP backend 尚未正式测时。重启后的第一项是做同口径 EP8/T16 balanced paired
benchmark：TopK+routed+TP1 shared+final sum，同进程同输入权重，8 秒 warmup，12 组 AB/BA，
K16+N200，8-rank max-local-mean；现有 DEP standalone harness 默认 T96，只能复用计时方法，
必须切到 T16 并加入 `rocm_mega_moe` candidate arm。Profiler 必须与 latency 分开。

## 2026-09-11 17:30 UTC 重启入口：512–8192 已进生产 selector 并提交；S1 超额流量定位在 A 侧复用

本节优先于下方历史。用户本轮明确授权 commit、更新最佳配置与更新文档。已提交两个：
`aiter 196a4399d`（kernel + selector）、`megamoe-s1-1k-prod 4dd0c02`（`bench/dep8_best_configs.json`）。
本节与 `tool_measure.md` 的 ATT 段是本轮唯一的文档改动；`plan_megamoe.md` / `summary_megamoe.md` /
`option_megamoe.md` 未改。顶部文档授权规则继续保留。

### 一、每档默认值现状（**每一档都已是实测最优**）

| batch | 加速比 | n | 默认=最优？ | 缺口 |
|---|---:|--:|---|---|
| 16 | **1.0502×** | 3 对 | 是（本轮） | BN=256 已采用，见一之三；三档里最弱的一条 |
| 32 | **1.1619×** | 2 对 | 是（本轮） | BN=256 已采用 |
| 64 | 1.2547× | 2 | 是（已测范围） | BN=256 在此档要把 BM 退回 32（LDS），实测 `b64_bm32_bn256` 更慢 |
| 128 | **1.4740×** | 2 对 | 是（本轮） | BN=256 已采用 |
| 256 | 1.470–1.522 | 1+2 | 是 | BN=256 在此档输 −2.52%；见下方区间说明 |
| **512** | **1.4807×** | 3 | 是（本轮） | 原 1.4596× |
| **1024** | **1.5777×** | 2 | 是（本轮） | 原 1.5658× |
| 2048 | 1.8483× | 4 | 是（已测范围） | dcu 32/48/64 全在 0.62% 门槛内 |
| 4096 | 1.9595× | 2 | 是（已测范围） | 除 dcu 外几乎未调 |
| 8192 | 1.9872× | 1 | 是（本轮起） | 此前 EP8 的 8192 掉到通用启发式 |

**b256 必须给区间**：JSON 记 1.5218×（A=591.39/B=388.62，单次），今天 sweep 的两个基线点是
1.4702×（A=577.7/B=392.9），§2 文字里是 1.488×。差异主要在 standalone 臂（591.4 vs 577.7，差 2.4%），
即 §6 记的"b256 上 A 臂跨 run 漂 1.31%"。要给单值必须在同一 session 重测基线，本轮没做。

2048/4096 仍只扫过 dcu/qgm/bn256 三类旋钮，"已测范围内最优"的范围很窄。

### 一之二、guard 清理（`aiter 166e78b1c`）

MegaMoEM3 / Stage1 / Stage2 三处必须互相一致的形状集合，原本是**六份独立字面量 + 同一张
Stage2 表的两种写法**，放宽其中一份而忘掉另一份就是静默算错。现在集中在 `mega_moe_config`：
`SHARED_FUSED_MTPR_SMALL/_LARGE`（两个 regime 及其并集，含"rows/expert = npes·T·topk/experts、
T=512 恰为 128、更大是 128 的整数倍"这个划分理由）、`SHARED_FUSED_S1_GEOMETRY`（按字段名比较，
失败信息直接说哪个字段错，不再是 tuple != tuple）、`SHARED_L2_S2_SHAPES`
（`max_tok -> ((BN,BK), (BM,SBM) 对)`，同时取代 13 元组白名单**和** early2 的乘积式 `allowed`）。

**按"被蕴含"删掉的**：`tile_k` 原本重复在 4 个 tuple assert 里，而 GEMM1 全局把它钉死
（A step 就是一个 tile_k 的 FP8），现在提前成一条 assert；`TILE_K_BYTES % 128 == 0` 由它蕴含。
**刻意保留的**：`M_REPEAT % 2 == 0` 与 `NUM_ACC_N % 2 == 0` 是"手工传入 sort_block_m / tile_n
且不受任何白名单约束"时的唯一防线；stage1 的 small_xcd `fuse_mtpr in (16..1024)` **不是**融合 shared
的集合（它服务未融合路径，EP4 在 512/1024 上也走它，2048+ 从未验证），已加注释说明。

**可接受集合逐点不变，用枚举证明**：3072 个 `(max_tok,BM,BN,BK,SBM)` 形状、24192 种 early2 组合
（含 `npes/band_m/cu_num/queue_grid_mult/local_reduce/xcd_local` 变化）与被取代的字面量逐点一致，
0 处不同。GPU 冒烟三点全过（12/12）：b16 `sort_block_m=32` 走 SBM=32 行（编译出 `t32x512x256/_sbm32_`）、
b256 走 SBM=64 行、**b512 用生产 kernel 副本**（不是实验 variant）走 SBM=128 大 regime，
558.47 µs 对同配置三点均值 557.62 µs。

`kernels_large` 至此基本退役：它的 m3/stage1 放宽已进生产，只剩 stage2 的 BN/BK 实验自由度还有用；
512–4096 的常规运行应直接用 `kernels_prod_check`（= 生产）。

### 一之三、BN=256 的采用与两处门禁的含义（`aiter 6051eef22` / `megamoe-s1-1k-prod 06622db`）

BN=256 让 Stage2 对 A2 面板的重读次数减半。它在上一轮就测到了，但**只能借 `kernels_s2geo` 变体跑**，
因为生产有两处门禁拦它：stage2 的 `(max_tok,BM,BN,BK,SBM)` 账本和 early2 几何门禁。

**这两处门禁在保护什么**（值得记住，不是形式主义）：**BN 决定 shared L2 的队列周期
`model_dim // BN`，而这个周期由 host 的 epoch 记账（`_shared_l2_heads`）和 kernel 各算一遍**。
用一个未验证的 BN，失败方式不是崩溃或报错，而是 **epoch 静默漂移**——正好是 harness 里
`L2 epoch aligned` / `L13 epoch aligned` 两条门禁在抓的东西。所以账本是"验证过的几何清单"，
不是可推导的约束；放宽它只有在同时跑过 epoch 门禁的前提下才安全。另有一条结构性限制以"缺席"
的形式体现在账本里：**BM=64 配 BN=256 超 64 KB workgroup LDS**（这也是 b512 上 BN=256 只能配
BM=32、从而净亏的原因）。账本现在是 `mega_moe_config.SHARED_L2_S2_SHAPES`，只多了三行
`(16,32,256,256,32)`、`(32,32,256,256,32)`、`(128,32,256,256,64)`，用 3072 个形状枚举确认
其余接受/拒绝逐点不变。

**生产路径复测结果**（每点全套数值/位精确/epoch 门禁 + "selector 默认值等于记录值"的 GPU 门禁）：

| 档 | 各对 Δ | 对间离散 | 加速比 |
|---|---|---|---|
| b32 | −0.45%, −0.39% | 0.06 pp | 1.1522 → **1.1619×** |
| b128 | −0.48%, −0.40% | 0.08 pp | 1.4673 → **1.4740×** |
| b16 | +0.18%, −1.37%, −0.57% | **1.55 pp** | 1.0454 → **1.0502×** |

**上一轮报的 +0.94%/+0.90%/+1.00% 要按生产路径的量级修正为约一半**（0.42–0.59%），那三个数来自
另一个子集 + Welch 检验。**b16 是三档里最弱的**：run 内 cv 只有 0.07–0.14%，但三对之间跨 1.55 pp，
说明它的噪声几乎全在 run 之间，今天 3 对单独看 sem 0.45pp、不显著；保留它的依据是两个 session
合计 9 对（今天 3 对 −0.59% + 归档 6 对 −0.86%）加上与另两档相同的机理，JSON 里已如实标注。

### 二、selector 改动与验证方式（**验证的是生产默认值本身，不是 harness 传进去的配置**）

新增 EP8 512–8192 分支：逐字段显式给出 8192 冻结几何，按档字段只有两个——
`payload_chunk_rows=min(T,2048)` 和 `num_dispatch_cu = 32 if tokens in (512,1024) else 96`。
打开这些档还需要三处枚举放宽（shared L13/L2 的 mtpr 表、stage1 大 batch regime assert、
stage2 的 `(max_tok,BM,BN,BK,SBM)` 白名单），依据是 rows/expert = `8T·topk/experts`，T=512 时正好 128、
更大时是 128 的整数倍，所以 128 行 sort block 与 128 行 shared tile 在每一档都整除。

验证手段是 `pmc_large.py` 新增的 `PMC_USE_SELECTOR=1`：它读 `_select_config` 的输出，用**在 harness 内
独立构造的期望值**（8192 JSON 条目 + 两条按档规则）逐字段门禁，然后**跑这个配置**过全套数值/位精确
graph replay/epoch/dispatch 门禁。512、1024、8192 三档已验（编译出的 kernel 名分别带
`_dcu32_/_pc512_`、`_dcu32_/_pc1024_`、`_dcu96_/_pc2048_`）；2048/4096 走同一分支、发出的值与 8192 相同，
但没有单独验证点。

### 三、dcu 曲线与一个**未修的正确性 bug**

b512 上 `num_dispatch_cu` 有明确极小值：16 (+10.71%)、24 (−0.63%)、**32 (−1.70%)**、48 (−0.24%)、
64（**正确性失败**）、96（继承值）。收益随 batch 衰减：b512 −1.70%、b1024 −0.93%/−0.96%、b2048 −0.37%（噪声）、
8k 噪声——dispatch CTA 数只在 dispatch 工作量相对 GEMM 还不小时有意义。

**`dcu=64` @ b512 在 `all_local` 倾斜探针下，rank 7 连续 256 次 graph replay 后输出与 eager 不逐位一致**
（`graph256 versus current-input eager`，前四个探针 initial/negative_large/zero/changed_routes 全过）。
`dcu=64` 时 `producers_per_destination = 64/8 = 8` 恰等于 npes，而 `payload_chunk_rows=512` 在 mtpr=512 下
只有 1 个 chunk——8 个 producer 抢 1 个 chunk。与本文件"已修复的错误"节记的 8k dispatch 倾斜 bug 同族。
dcu 32/48 七个探针全过。证据 `.scratch/dep8_s2_bnbk_20260911_v1/b512_dcu64/run.log`。**未修，未定位到行。**

### 四、本轮主线：S1 的超额流量全在 A 侧，而且是**复用**问题不是体积问题

13 个 traffic 点 + 2 个 cache 组 + sqwait/sqmix，全部单 kernel counter，逐点 spread 已记录。
**两个 B（routed B1 622.9 MB、融合 shared L13 B 38.9 MB）各自只从 HBM 读一次**；超出"权重各读一次"
模型的部分全是 A：A tile 被**每个 (m_tile, N_panel) 任务重取**（小 batch 12 个 panel、大 batch 24 个），
且**按 tile 高度补齐**（b32 每 expert 8 行真实数据按 64 行读）。

A 占 S1 读流量：b32 7.0% → b256 15.1% → b512 37.9% → b1024 48.4% → b2048 61.4% → b4096 77.8%。
三条独立证据：b64→b128 只涨 3.6 MB（shared B 若按 m_tile 重读应涨 38.9）；`sort_block_m` 64→32 让残差
恰好减半（50.2→25.8 MB，写侧同步减半）；b256/b2048 的 cache 组与 traffic 组互证到 0.4%/0.7%
（miss×128 vs EA 字节），L2 请求总量与任务模型差 0.7%/3.8%。

**大 batch 上 A 与 B 的划分是模型归因，不是测量**：counter 只给界限（b2048 上 A ∈ [110,1053] MB，
即读流量的 6%–61%）。取上端的理由是小 batch 上 B 被钉在 1.00–1.08×，且只有这个记账能让 A 的 miss 率
在整条 ladder（5.7× tile 数、两种几何、12 vs 24 panel、2.35× CTA 数）上稳定在 0.62–0.75。ATT 想直接
分开，四次尝试全部解不出 wave 数据（见 `tool_measure.md` ATT 段新增说明）。

**字节确实换时间，但只在大 batch**：用 `b_nt=2` 做已知字节增量的探针（S1 流量 +8.2%/+12.1%），
整层实测 **+3.32%（b2048）/+3.24%（b4096）**，各两到三个基线、12 对配对、跨 run 跨度 0.3%。
这也第一次实测了 EP8 512–4096 的 `b_nt`（结论：`b_nt=0` 正确，填掉 8k 上靠 NT 判据预测跳过的空档；
但那个判据只算 B panel 的复用次数，没算 B 把 A 挤出 L2 的代价，机理解释不完整）。

**两堵墙都不贴**：带宽 39–70% 可达峰值；矩阵管占用 **S1 b256 18.7%、b2048 37.1%、b4096 39.9%、
S2 b256 13.7%**（`MFMA_BUSY/(4×BUSY_CU_CYCLES)`——`SQ_VALU_MFMA_BUSY_CYCLES` 恒等于
`SQ_INSTS_MFMA×32` 且按 SIMD 累加，而 `SQ_BUSY_CU_CYCLES` 按 CU，**本轮中途一度按 MFMA/BUSY 直接相除
报成 75%，是错的，已撤回**）。CU 有 75–93% 的时间有 wave，wave 有 60% 的周期在 WAIT_ANY。

**没有配置层面的出路**：融合 shared 的两个 regime 都把 `(sort_block_m,tile_n,tile_k,num_waves,band_m)`
硬 assert 死；`num_dispatch_cu` 把并发 CTA 数改 2.35 倍**流量一动不动（±0.3%）**，说明 A 的 miss 是结构性的、
与共调度无关；`b_nt` 反向。要省只能改 GEMM 内层（一个 CTA 吃掉该队列的 2–3 个 N panel、A 的 k-slice
在 CTA 内复用），代价是 accumulator 翻倍到三倍、VGPR 128→192+，会从 2 CTA/CU 掉到 1（这个 kernel 正好
卡在 128 VGPR 的 2-CTA 边界上）。

### 五、本轮否决清单（都有数）

`sort_block_m=64` @ b512：**请求量 −29%（padding 归零，rows/expert=128 正好两个满 tile）但 A 的 DRAM
流量 +7.7%**（半大的 tile 连原本 0.69 的 L2 复用也丢了，升到 1.03，即每次请求都 miss），整层 +0.32%/+0.45%
（时间相邻配对）。variant 连 assert 归档在 `.scratch/dep8_s2_bnbk_20260911_v1/kernels_sbm64/`，
注释里写明已否决。**这恰好反向印证第四节：只减体积不改复用结构，一分钱省不下来。**

b512 其余：`BM=32` +7.85%、`BM=32+BN=256` +3.62%（BN=256 本身在此档值约 −3.9%，但被迫搭配 BM=32；
**想要的 BM=64+BN=256 被 64 KB workgroup LDS 挡住**——这是 b512 剩下最大的一条线索）、`BK=256` +1.95%、
`qgm8` +2.21%、`qgm2` +6.12%、`qgm3` −0.32%（两次、跨度 0.101%，低于 1.28% 门槛，且与 dcu32 叠加后
完全消失：555.23 vs 555.22）、`pc2048` +0.56%。

`_local_reduce=false` @ b512 **无法测**：`bench_large.py` 的 `check_state` 断言
`op._g2_reduce_counters.any()`，该张量在关掉 local reduce 时是 `None`。要测需复制一份 harness
（`bench_large.py` 在 pin 列表里，不能改）。EP4 老表在 512 档是关的，值得一测。

### 六、未完成（按价值排序）

1. **BN=256 的三档收益兑现**（+0.9~1.0%）：需放开 stage2 BN 白名单 + early2 几何门禁，并复验一点。
2. **`dcu=64` @ b512 的倾斜正确性 bug** 定位与修复。
3. **S2 降 LDS 以解锁 `BM=64+BN=256`**（b512 上分解出约 4%，可能同样适用 1k/2k/4k）。
4. **A 侧复用的 GEMM 内层改动**（大 batch，b2048 弹性 0.86、b4096 0.56，对应整层 12–14% 的量级）。
5. `_local_reduce` 开关在 512/1024 上的实测（需 harness 副本）。
6. 2048/4096 的其余旋钮（除 dcu/qgm/bn256 外基本空白）。
7. 大 batch 上 A/B 的直接分离（ATT 解码不通，需另找手段）。

证据目录 `.scratch/dep8_s2_bnbk_20260911_v1/`：`analyze_s1_traffic_all.py` + `S1_TRAFFIC_TABLE_ALL.txt`
出全部 13 个流量点与 occupancy/wait；`summarize_all.py` 按 batch 分组、按候选臂排序出所有计时点；
`SWEEP_B512*`/`SWEEP_DCU1K`/`SWEEP_SBM64`/`PMC_*` 是逐点进度；`variants.json` 五个变体的 `differs`
声明已与新生产对齐（`kernels_prod_check` 逐字等于生产）。Slurm 1820 到 2026-09-11 23:03:42 UTC 到期，
本轮 16:43–15:50 期间它**对新 step 失效过约 7 分钟**（`Unable to create step: Error generating job credential`，
新作业可正常建 step，munge 与时钟均正常），driver 内置重试第 6 次后自行恢复，未污染任何结果。


## 2026-09-11 13:45 UTC 重启入口：S2 block_n=256 三档取胜、b256 清单扫空、512–4096 已启用、S1/S2 都不贴带宽墙

本节优先于下方历史。用户授权“更新文档 要重启对话了”。本轮只更新本交接与 `tool_measure.md`（新增 §7.5）；
`plan_megamoe.md` / `summary_megamoe.md` / `option_megamoe.md` 未改。顶部文档授权规则继续保留。

### 一、生产改动（**已改未提交**，重启后第一件事是决定去留）

`aiter/ops/flydsl/kernels/mega_moe_m3/mega_moe_m3.py` 的 `_select_config` 改成**按 token 数发各档实测最佳**，
替代原来五档共用一份 b256 配置。`git status` 只有这一个文件 ` M`，父提交 `bab7de481`。

```
sort_block_m = 32 if tokens in (16,32) else 64      num_dispatch_cu = 32 if tokens==256 else 48
b_nt         = 0 if tokens==256 else 2               s2.block_m = 64 if tokens==64 else 32
s2.use_nt    = tokens in (32,64)                     preplan_waves = 4 if fused_shared else 0
```

顺带修掉一个真 bug：旧分支对 256 档发 `preplan_waves=0`，而 `mega_moe_stage1.py:133` 是
`if shared_l13: assert preplanned` ——**融合 shared 的 b256 走旧默认分支会直接 assert 失败**，
实测 1.488× 用的是 4。现在按“是否融合 shared”决定，未融合的 256 路径行为逐字未变。
bench 里加了每点都跑的 GPU 门禁：`_select_config(tokens)` 必须逐字段等于 `dep8_best_configs.json` 该档记录值。

**注意：下面 BN=256 的三档胜利尚未写进 `dep8_best_configs.json`，也未进 selector。**

### 二、S2 `block_n` 128→256：b16/b32/b128 赢，b256 输（本轮主结果）

判据用候选臂 B（见第五节），Welch t 检验：

| 档 | ΔB | p | n | 裁定 | 新 speedup |
|---|---:|---:|---:|---|---:|
| b16 | +0.94% | 0.0073 | 7/6 | **赢** | 1.0562× |
| b32 | +0.90% | <0.0001 | 5/5 | **赢** | 1.1593× |
| b128 | +1.00% | 0.022 | 5/5 | **赢** | 1.4735× |
| b256 | −2.52% | — | 3/3 | **输**（反向完全分离）| 1.488× 不变 |

三档赢的幅度高度一致（0.90~1.00%）。机理是 A2 重读次数 48→24。**b256 为什么输，至今没有解释**——
我先后给过“grid 超订”和“B 面板变大掉 L2”两个解释，**都被自己的判据打掉**（qgm4 把驻留补满后差距不变；
R_S2 在 b128/b256 相同）。b128 与 b256 的 S2 几何完全一致（1024 补齐行、SBM=64、BM=32），只差 shared tile 行数。

可行域已算清，只有 BN=256 一个新点合法：BN=64 的 `mni_base` 里 `wave*(BN//64//2)=0`（四 wave 共用一份 B scale，
需改 kernel）；BN=512 被 `band_m>1` 的 `(6144//BN)%8==0` 拒；BK=512 的 `tilesPerScaleChunk=256//BK=0` 除零。
**BK=128 四档全输**（b32 −2.28%、b64 −1.82%、b128 −1.83%、b256 −4.80%）。

### 三、b256：整份标准清单扫空

按其他档同一份清单扫完（`dcu` 40/48/56/64、`b_nt=3`、`preplan_waves` 2/8、`pas`、`upp`、`trbu`、`jwf`、
四个 xcd home 全关），基线 n=4、B 跨度 0.77%：

- **没有任何单旋钮有收益。** 最好的 `jwf` +0.18%、`dcu40` +0.10%，都在基线跨度内。
- **`dcu48` 在 b256 明确亏 1.23%**——它在其他四档全胜。dcu 在 b256 单调向下（48 −1.2%、56 −0.5%、64 −1.8%），已在最优点。
- **XCD 全关亏 2.36%**（b32 上是 0.94%），是所有档里 XCD 播种收益最大的一点。
- `upp` −2.62%、`dcu64` −1.83% 确认输。
- `num_dispatch_cu` 必须被 8 整除（`assert dispatch_blocks % fuse_npes == 0`），36/44 不存在。

唯一有信号的是 **S2 `queue_grid_mult`**：768 +0.48%（p=0.026, n=5）、**896 +0.79%（n=4）**、1024 +0.19%（不显著）。
非单调，**机理不明**（我的“填满 768 驻留槽”解释预测饱和在 768，实测 896 更好）。**未采用、未进 JSON。**
另：`b_stages=1`（新增的单级 B 流水开关）把 VGPR 147→96、capacity 3→5 CTA/CU，但 grid 640 亏 2.07%、
grid 896 亏 0.56%、只有 grid 1024 赢 0.71%（n=1），且叠加 BN=256 是灾难（b256 −14.9%、b128 −11.3%、b32 −4.9%）。
**用户指出 occupancy 低本不是问题**——2 CTA/CU 正是设计稳态（一个做 MFMA、一个读全局内存），
`b_stages` 的数据恰好支持这一点。**我用“capacity 3→2”去解释 BN=256 失败的那套说法应当撤回，它从无直接证据。**

### 四、8k 与 b2048：调度类旋钮全部无效；512–4096 已启用

**8k**（9 个有效点，基线 n=2 跨度 0.36%）：`queue_grid_mult` 720/960/1440 与 `num_dispatch_cu` 64/128
**全部落在 ±0.33%、门槛 0.4% 以下**。移植 harness 复现记录值（1.98536× vs 1.98718×，差 0.09%）。
NT 判据算出 8k 的 `R_S1=16`、`R_S2=2`，预测 `b_nt=0`/`use_nt=False` 已最优——省掉 4 个点，判据在这里兑现了。
`external_counting=True` 在 8k 被 preplan 绕过，是死字段（`option_megamoe.md:156` 已记）。

**b2048**（基线 n=3 跨度 0.36%）：`grid 720` −0.38%、`grid 960` +0.19%、`dcu64` +0.20%，全在噪声内。

**512/1024/2048/4096 的启用**：实验变体 `kernels_large` 只放宽 token 枚举，不动任何真实不变量——
`stage1:134` 的大 batch regime（`== 8192` → 五个 2 的幂）、`m3:95` 的 shared **L13** mtpr、`m3:118` 的 shared **L2** mtpr、
`stage2` 的 `(max_tok, BM, SBM)` 白名单。依据：rows/expert = `8T·4/128`，T=512 时正好 128、更大时是 128 的整数倍，
**SBM=128 在五档都整除**。`stage1:175` 的 `fuse_mtpr in (...,1024)` 在 `small_xcd` 分支不适用。

已验证：**b2048 1.85168×（A=2385.7 B=1288.4）、b4096 1.95773×（A=4612.9 B=2356.3）**，12/12 faster，
数值、graph replay 位精确、L2/L13 epoch、dispatch 审计全过。b512/b1024 **未拿到数字**（被我自己的两次误操作撞掉，
不是启用问题；b1024 当时已过全部门禁、正在出计时）。大 batch 路径向 2× 饱和：2048→4096→8192 = 1.852→1.958→1.985。

### 五、PMC：S1 和 S2 都**不**贴带宽墙（本轮最重要的测量）

| kernel | HBM 读 | 时间 | 达成带宽 | 占可达峰值(6.5~6.8 TB/s) | 模型 |
|---|---:|---:|---:|---:|---|
| S1 | 779.8 MB | 205.0 µs | **3.80 TB/s** | 56~58% | B1 读一次 736 MB → 比值 1.06 ✅ |
| S2 | 473.4 MB | 110.0 µs | **4.30 TB/s** | 63~66% | B2 读一次 509.6 MB → 比值 0.93 ✅ |

S2 的两组 counter 独立互证：`TCC_MISS × 128 B = 475 MB` vs traffic 组 473.4 MB，**差 0.4%**，
隐含每 miss 127.4 B = cache line，字节换算得到校验。L2 命中率 62.7%，L2 层面总请求 1275 MB。
**`R_S1=1`、`R_S2` 的复用都与模型相容**，权重只从 HBM 读一次。算术强度 S2 = 94.8 FLOP/byte vs
机器平衡点 ~625，**离算力墙 6.6 倍**。

**两个必须随数据一起读的限制（用户指出）**：

1. **shared 与 routed 分不开。** regex 抓的是一个融合 kernel，counter 是整个 dispatch 的聚合。
   S2 的 473.4 MB 含 routed B2(302)+shared L2 B(18.9)+两者的 A 重读；S1 的 779.8 MB 含 routed B1(604)+shared L13 B(37.7)。
   **“B2 只读一次 / B1 只读一次”是把一个聚合数按模型分配后的说法，不是逐矩阵实测**——
   S1 超模型 6%、S2 低于模型 7%，这部分差额完全可能全在 shared 侧。要分开需要 shared 不融合的对照，或逐 buffer counter（PMC 给不了）。
2. S2 那次 run 最后在 `graph path reported on one stream` 门禁上失败（rocprofv3 改变 stream 上报），
   **只有 CSV，没有 result.json**；counter 可信（硬件计数、三次一致在 2% 内），但该 run 的时间数字不得用于配置排序。

**结论：两头都不靠。** 带宽用不到三分之二、算力差 6.6 倍——这解释了为什么本轮所有“减字节”和“加 CTA”
的旋钮都只在 ±1% 量级动。瓶颈在第三类（依赖链/barrier/访存并发度），而**这一类今天没有任何直接证据**。
`sqwait` / `sqmix` 两组（`SQ_WAVE_CYCLES`、`SQ_WAIT_ANY`、`SQ_WAIT_INST_ANY`、`SQ_INSTS_MFMA/VMEM`）
已加进 `pmc_rank.sh` 并写好清单 `pmc_list_wait.txt`，**未跑**。那是区分这三类的唯一直接手段。

### 六、方法学：判据统计量取决于哪条臂在漂（务必保留）

**排序候选配置前必须先分别算两条臂的跨 run cv。** 配对能消 run 内共模漂移，消不掉参照臂的 run 间独立漂移，
后者会被除进比值里放大。

| batch | A(standalone) | B(candidate) | B/A 配对比 |
|---|---|---|---|
| b256 | 跨度 1.31% | **0.77%** | **1.85%（最差）** |
| b2048 / 8k | 0.03% | 0.13~0.36% | 可用 |
| b16（旧记录）| 0.46% | 0.93% | — |

**关系随 batch 反转。** b16 上 B 更吵，所以 `tool_measure.md` 原来那条“必须用同 run 配对、不能比裸 B”是对的；
b256 上 A 漂 1.3% 而 B 只有 0.3~0.8%，**配对比是三者中最差的**。代价是实打实的：b256 上 `dcu48` 按配对比读是
−0.15%（噪声），按候选臂 B 读是 **−1.23%（明确亏）**；`dcu40` 按配对比 +0.76%（假胜、我追了两轮），按 B 是 +0.10%（无效）。
**做法**：报“相对 standalone 的加速比”用配对比（那是它唯一有效的用途），**排序候选用跨 run 跨度更小的那条臂，并说明用了哪条、为什么**。

`tool_measure.md` **新增 §7.5 无人值守运行的监督与失败可见性**，四条规则 + 三条配套做法，
每条对应本轮一次真实事故（详见该节）。最重要的一条：**会话只在“注册过的后台任务退出”时被唤醒**，
`setsid nohup <driver> &` 写在前台调用里什么都没注册——本轮因此让 GPU 空转 5 小时 20 分无人知晓。

### 七、本轮的操作事故（都已写进 §7.5，重启后不要重犯）

1. **运行中改被 pin 的文件**：改 `variants.json` 撞掉正在跑的 8k 点（最后一关 `final source pins` 失败）。
2. **运行中改正在执行的 driver 脚本**：bash 按字节偏移续读，报假语法错，**杀掉一个已过全部门禁、正在出计时的 b1024**。
3. **两次用 `pkill -f` 杀到自己的 shell**（`[p]attern` 括号技巧防不住，自己的命令行里有匹配文本）。只按 PID 杀。
4. **补丁锚点只断言“存在”不断言“唯一”**：`xcd_home/shared_xcd_home` 在 Stage1Config 和 Stage2Config 各有一份，
   `replace(...,1)` 把 `b_stages` 加到了 Stage1Config。
5. **移植 harness 没核对它读的路径**：`bench_large.py` 继承了 `PREVIOUS / 'protected_documents.json'`（昨天的旧 pin，
   4 份文档早已变化），连续三次失败才查出来。
6. **两个 sweep 并发上 GPU**（`rm -rf` 报 “Directory not empty” 时没停下来查）。已弃用那两个点。
7. **漏了第三处 token 枚举**（`m3:95` 的 L13），只读了 115–122 行就下结论；应全文件 grep 所有 token 列表。

### 八、未完成（重启后的起点，按价值排序）

1. **`sqwait` / `sqmix` 两组 PMC**（清单已备、未跑）——唯一能定位“第三类瓶颈”的直接证据。
2. **shared vs routed 的流量分离**——需要 shared 不融合的对照 run，否则第五节的逐矩阵归因都是模型分配。
3. **BN=256 三档胜利写进 `dep8_best_configs.json` 与 selector**；selector 改动**尚未提交**。
4. **b512 / b1024** 补跑（`sweep_redo_large.txt` 已备）；**b2048/b4096 的 `bn256` 点**（各省约 50% S2 流量，
   是这两档唯一有理由赢的旋钮，`b2048_bn256` 跑到一半被停）。
5. **b256 的 `queue_grid_mult` 896**（+0.79%, n=4）要不要采用——需放宽生产白名单，且机理不明。
6. **`external_grouping` True→False**（分组 CTA 在 quant kernel 还是 S1 内）：EP8 从未扫过，
   `pending_patches.py` 已把它加进两个 harness 的 ALLOWED 并配了 `_eg{0,1}` 名字门禁，**未应用**。
   预计 8k 上被摊薄、小 batch 才有戏。真正的“只在 S1”做不到（`shared_l13` 强制 `preplanned`）。
7. **S1 那 46.9 µs / S2 30.7 µs 的截距**仍未识别（长期未决）。

证据目录 `.scratch/dep8_s2_bnbk_20260911_v1/`：`summarize_all.py` 出全表、`PIPELINE` 是带时间戳的阶段日志、
`SWEEP_*`/`SMOKE2_*`/`PMC_*` 是逐点进度、`variants.json` 声明每个 kernel 变体允许与生产不同的文件
（`kernels_prod_check` differs==[]、`kernels_s2geo` 只差 stage2、`kernels_s2b1` 差 4 个、`kernels_large` 差 3 个）。
失败目录一律保留并加后缀说明原因（`*_DOCPIN_RACE`、`*_STALE_DOCPIN_PATH`、`*_CONCURRENT_SWEEP_DISCARDED`、
`*_DOCPIN_RACE_SELF`、`*_STOPPED`）。Slurm 1820 到 2026-09-11 23:03:42 UTC 到期。重启先读本节与 `tool_measure.md`。


## 2026-09-11 03:00 UTC 重启入口：b32/b64/b128 调优完成，NT 判据已推导并预测成功

本节优先于下方历史。用户授权“commit代码 更新最佳配置 更新md文档 记录这个理论 我要重启对话了”，
因此本轮更新本交接、`option_megamoe.md`、`bench/dep8_best_configs.json` 及 outer `HANDOFF.md` 副本。
顶部文档授权规则继续保留。**上一节（四个上游开关）来自另一个并发会话，未被本轮改动。**

### 四档实测最佳（统一口径：EP8、H6144/I3072、128 experts、top4、TP1 shared；
窗口 TopK + routed MoE + shared MLP + 最终和；同进程同输入、8 秒 warmup、12 组 balanced AB/BA、
K16 settling、N200 measured、无 profiler、8rank 各自 HIP-event 均值取最大）

| 档 | 原（b256 冻结配置） | 最佳 | 相对 b256 冻结配置的 delta | 重复次数 |
|---|---:|---:|---|---:|
| b16 | 0.942× | 1.055× | S1 `sort_block_m` 32、`b_nt` 2、`num_dispatch_cu` 48 | 1（上一轮 1.0492×，本轮复现 1.0550×）|
| b32 | 1.025× | **1.14536×** | 同上 **+ S2 `use_nt=True`** | 3 |
| b64 | 1.167× | **1.25470×** | S1 `b_nt` 2、`dcu` 48 **+ S2 `block_m` 64 + `use_nt=True`** | 2 |
| b128 | 1.400× | **1.47110×** | S1 `b_nt` 2、`dcu` 48 | 2 |
| b256 | 1.488× | 1.488× | **无改动**（三个旋钮全部无效，见下）|

完整参数与全部证据入口写在 `bench/dep8_best_configs.json` 的 `entries."32"/"64"/"128"`。
每条都带两条限制：**不是 selector 默认**（生产小 batch selector 仍原样发 `entries."256".config`，要用必须显式传）；
**只在该档测过**，其他档不得继承。

### 本轮的核心成果：NT（non-temporal）开关的定量判据

**这是本轮最可复用的结论。** B 权重的 cache 策略（S1 `b_nt`、S2 `use_nt`）不需要试，可以直接算。

定义 **R_B = 一份可复用的 B panel 被几个输出 tile 读**：

```
R_S1 = ceil( (8*T*topk/E) / sort_block_m )    # 每 expert 的行数 / S1 M tile 高度
R_S2 = sort_block_m / block_m                 # 注意：与 T 无关！
```

**R_S2 与 T 无关**，因为 S1 把每个 expert 的行数补齐到 `sort_block_m` 的整数倍，
所以 S2 看到的每 expert 行数恒等于 `sort_block_m`，而不是真实行数 `T/4`。
**我第一次就是在这里算错**（用了真实行数），得出 b64/b128 的 R_S2 也是 1，与实测矛盾。

判据来自"拿 B 的复用换 A 的驻留"：

```
开 NT  ⟺  (R_B - 1) * B_total * 1[S_B <= C_L2]  <  (R_A - 1) * A_total
```

代入本模型：`B_total/A_total ≈ 100~200`（S1 的 B 604 MB vs A 3.1~6.3 MB；S2 的 B2 302 MB vs A2 1.6 MB），
右边永远赢不了，所以判据塌缩成 **开 NT ⟺ R_B == 1**：

```
b_nt   = 2 if R_S1 == 1 else 0
use_nt = True if R_S2 == 1 else False
```

**符号预测全部 7 个点正确，并成功做出一次预先预测**：b64 把 S2 `block_m` 由 32 提到 64
使 R_S2 由 2 变 1，`use_nt` 随即由亏 5.3% 变成赚 0.74%（同一 batch、同一 SBM，只动 BM）。

| 档 | R_S2 | `use_nt` 实测 |
|---|---:|---|
| b16 (SBM32/BM32) | 1 | 持平（收益被噪声淹没）|
| b32 (SBM32/BM32) | 1 | **+0.8%**，3 次确认 |
| b32 (SBM64/BM32) | 2 | **−4.4%** ← 决定性验证：同 batch 只动 SBM，结论翻转 |
| b64/b128/b256 (SBM64/BM32) | 2 | −5.3% / −8.0% / −10.6% |
| b64 (SBM64/BM64) | 1 | **+0.74%** ← 预先预测成功 |

**幅度不由 R 决定，机理仍未定位**：R=1 时收益随 batch 衰减到零（b32 +0.8% → b64 +0.74% → b128 0 → b256 0）；
R=2 时损失随 batch 增大（−5.3% → −8.0% → −10.6%）。按 `(R_B-1)*B_total/8.94TB/s` 算 R=2 的代价是 33.8 µs，
实测 b64/b128/b256 分别为 17.7/25.1/40.1 µs —— 同量级，小 batch 偏低说明复用本来只抓住了一部分。

**复用不是被保证的，只是启发式。** S2 同一队列连续 `band_m` 个 ticket 共用同一 `n_block`
（`m_block = band*band_m + rem%band_m`，`n_block = (rem//band_m)*8 + queue`），且队列按物理 XCD 归属，
所以共用同一 B2 的两个 C tile 确实拿到相邻 ticket、同一 XCD。但相邻 ticket 落在不同 CTA 上，
**没有 barrier、没有顺序保证**，两次读的时间间隔无界；队列取空后 `(home+attempt)%8` 会偷别的 XCD 的队列。
S1 在小 batch 上更彻底：`small_xcd` 解码给出连续 `m_index`，而小 batch 每 expert 恰好 1 个 M tile，
所以"连续 m_index"= 连续的不同 expert = 完全不同的 B，**band 在这里一块 B 都没复用到**（与 R_S1=1 自洽）。

### 已提交的生产改动

inner `megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`。改 4 个文件：

1. **S2 SBM 白名单**加 `(32,32,128,256,32)`（b32 的 SBM32）与 `(64,64,128,256,64)`（b64 的 BM64）。
   early2 geometry 相应放宽：`small_sbm = 32 if max_tok in (16,32) else 64`，并对 `max_tok==64` 增加 BM=64 元组。
   **只加被实测采用的元组**；b128/b256 的 BM=64 虽然也通过了全部数值门禁但未被采用，没有进白名单。
2. **新增四个 home-seed 开关**：`Stage1Config.xcd_home` / `shared_xcd_home`、`Stage2Config.xcd_home` / `shared_xcd_home`，
   默认全 `True`（= 物理 XCD 播种）。语义是"只换 home queue 的种子"，队列数 8、band decoder、ticket 记账、
   epoch 协议全不变——每个 CTA 无论如何都遍历全部 8 个队列，只改起点。
   **默认时不追加任何名字后缀，已实测 codegen 中性**（`b32_sw_baseline` 的 S1/S2 kernel 名与改造前逐字相同），
   关掉才出现 `_hr0`/`_hs0`。S1 为此新增了第二条轮转 `shared_work_shard`，让 shared L13 与 routed 可以分别播种。

**注意不能用 `xcd_schedule=False` 当这个开关**：S1 里 `small_xcd = xcd_schedule and not payload_tile_ready`，
一旦关掉会掉进 `work = work_shard + local_work*WORK_SHARDS` 那条完全不同的工作分解分支，
且 129 行 `assert BAND_M == 1 or payload_tile_ready or xcd_schedule` 直接拦下 band_m=4。那是未适配的分支。

### XCD 亲和性：实测该保持全开

四个 home seed 全关 vs 全开：**0.88956 vs 0.88129，全关慢 0.94%**，CI 不重叠（`b32_sw_all0` vs `b32_sw_baseline`）。
我曾预测 b32 上 XCD 应该零效应（四条路径 B 复用因子都是 1，A 足迹 3.1 MB 对 4 MB per-XCD L2 怎么放都装得下），
**这个预测错了**。B 侧几何分析没错，错在把"B 没有复用"等同于"XCD 没有价值"。
剩下的解释是队列头 atomic 的 XCD 局部性（S1 约 1656 次、S2 约 10240 次 claim），**但这是排除法，没有直接证据**。

### 被否决的候选（本轮新增，都有证据，不要重试）

- **S2 `block_m=64` 在 b128/b256 无效**：b128 两次 0.67653/0.68385 与 BM=32 的 0.67907/0.68045 完全重叠
  （我曾基于单次 0.67653 报过"赚 0.5%"，被重复跑推翻）；b256 的 0.67029 落在 frozen 两次 0.67201/0.66965 的跨度内。
- **b256 三个旋钮全部无效**：`b_nt=2` 0.67172、`block_m=64` 0.67029、三者叠加 0.67740，均在 frozen 跨度内。
  注意这证伪了我"b256 的 R_S1=1 所以 b_nt=2 该赢"的推论——**R 只预测符号，不预测幅度**。
- `scalar_tile_row_base`(0.88239)、`joint_work_flags`(0.88334)、`unroll_a_pingpong`(0.88062)：均在 b32 基线区间
  (0.88129~0.88455) 内。那条"打在 46.9 µs 固定截距上"的思路没有兑现，**该截距仍未定位**。
- `packed_a_scale`：单独 0.87948/0.87445 看似略好，但叠在 `use_nt` 上是 0.87170，落在 `use_nt` 三次
  (0.87102~0.87440) 的区间内，**不叠加收益，判为噪声**。
- `preplan_waves`=2 (0.88051，噪声内) / 8 (0.89333，更差)；`b_nt`=3 (0.89057)；`dcu`=40/56 (0.89080/0.88903)；
  `work_shards`=4/16（**guard 拒绝**：`assert WORK_SHARDS == 8`，8 个 work shard 就是 8 个 XCD 队列）。

### 尚未扫过的维度（下一轮的起点）

- **S2 `block_n`（现 128）、`block_k`（现 256）在 EP8 上完全没扫过。** 动它们要放宽同样两道白名单。
- S1 `tile_n` 只在 b16 扫过一个点（256 被否），b32/b64/b128 没扫过。
- **S1 `tile_k` 不可调**：`assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"`。
- S2 `block_m` 只试过 16（编译过但 routed 数值错 rel 0.951，需改 kernel）、32、64。
- 约束：`SBM % BM != 0` 直接 raise；S2 `block_m` 必须整除 S1 `sort_block_m`。

### baseline 漂移：新证据指向 standalone 侧

b256 frozen 本轮两次 **A=578.620/579.962、B=388.836/388.369**。文档旧值 A=591.387/B=388.617。
**B 逐字吻合，A 差 2.1%** —— 即旧表的 1.52177× 今天复现不出来（实测 1.488~1.493×），
**原因完全在 standalone 侧，候选侧纹丝不动**。这是 handoff 长期未决的 575.863→591.387 漂移，今天又回到 579 附近。

### 方法论：本轮踩到并已修复的坑

1. **跨 run 漂移按档不同**：同 code object 重复跑，b32 0.37%、b64 1.6%、b128 0.20%。
   **b64 的 1.6% 意味着那里 0.5% 以下的排序不可信**；每档的最佳都必须有重复跑，不能单点定论。
2. **`srun` 会吃掉驱动脚本的 stdin**，把 sweep 列表的剩余行一起读走 → 每批只跑第一行。已加 `< /dev/null`。
3. **NFS 属性缓存**会让刚写完的 `result.json` 在 `srun` 退出瞬间 stat 不到 → 误判为失败。
   已改为先 `ls` 破缓存并同时要求 `run.log` 里有 `RESULT` 行。
4. **rank 归一化正则写死了 `_plan_w4`**，`preplan_waves=2/8` 时 8 个 rank 名字不同被误拦。已改 `_plan_w\d+`。
5. **`pkill -f run_sweep.sh` 会匹配到自己的 shell**，把发起命令的 bash 一起杀掉，后续命令静默不执行。
6. **Slurm 偶发 `Error generating job credential`**，持续约 4 分钟后自愈；重试即可，不要改配置。
7. 并发会话会改 `handoff_megamoe.md` 等受保护文档，导致运行中的 document pin 门禁在**最后一关**失败。
   那不是测量问题；重新 pin 后重跑即可，**不要因此把文档移出保护集**。

### 证据目录

全部在 `.scratch/dep8_b32_tune_20260911_v1/`：`summarize.py` 打印全表，`stages.py` 出分段，
`variants.json` 登记每个 kernel 变体允许与生产不同的文件，`run_sweep.sh` + `sweep*.txt` 是驱动与批次，
`SWEEP_PROGRESS*` 是进度。失败目录一律保留并加后缀说明原因
（`*_CREDFAIL`、`*_DOCPIN_RACE`、`*_MISSING_KWARG`、`*_HARNESS_REGEX_BUG`、`*_TOKENS_ASSERT`、`*_EARLY2_GUARD_BLOCKED`）。
`kernels_prod_check/` 是生产源码的逐字副本（门禁 `differs == []`），用于在生产源码上复验四档最佳配置。

重启先读本节与 `tool_measure.md`，核对 inner/outer HEAD、git status、Slurm 与 GPU 占用。

## 2026-09-11 用户确认：迁移时保留四个上游开关

用户授权“好你记录一下”，本次仅记录以下来源与清理决定，并同步修正 `plan_clean_megamoe.md` 中对应的删除建议。

- `pipe_weights`、`swizzle_a`、`b_hoist`、`ascale_prefetch` **均来自原版 DSV4 MegaMoEV2，不是本轮 M3 实验新增**，上游默认值均为 `True`。
- Git 证据：上游引入提交 `97d0c6e4c`（`megamoe (#4439)`）；在迁移基线 `24a62b1c1` 的
  `aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py` 中，S1 的前两项位于 46–47 行，S2 的后两项位于 65–66 行。
  M3 移植提交 `d1dc744fa` 继承了这些开关。
- 含义：`pipe_weights` 流水预取下一 K 块 B 权重；`swizzle_a` 调整 A 的 LDS 布局以减少 bank conflict；
  `b_hoist` 将 S2 下一块 B 的预取前移到 barrier / A 读取之前；`ascale_prefetch` 预取 S2 下一块 A 的 scale。
- **本次 PR 迁移先保留这四个配置字段、透传链及实现，优先清理本轮新增且已废弃的实验参数，减少无关改动。**
  当前已验证的 b16/b256/b8192 最佳配置均开启这四项，但这不证明所有支持配置都可以删除关闭分支。
  本决定覆盖清理计划此前将这四项按“恒 True”直接删除的建议；其他清理项仍需按各自证据判断。

## 2026-09-11 00:27 UTC 重启入口：b16 已调到快于 standalone，S1 三参数已提交

本节优先于下方历史。用户先后授权“把目前的修改commit 把最佳配置记录好”与“也更新文档 我要重启对话”，
因此本轮更新本交接、`summary_megamoe.md`、`plan_megamoe.md` 及 outer `HANDOFF.md` 副本；顶部文档授权规则继续保留。

### 当前提交与预约

- inner `/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，
  提交 **`6b99bac6a`** `Allow a 32-row Stage1 sort block for small EP8 shared batches`，父提交 `db9ccdfaa`。
  **只放宽 3 处 guard，逐行核对过没有任何 codegen 变化**：Stage1 两处接受 `sort_block_m ∈ {32,64}`，
  Stage2 的 SBM 白名单与 early2 geometry 接受 SBM=32。此前能编译的配置都编译成同一个 code object。
  新增的一条 assert 把安全前提写进代码：`mega_moe_m3.py` 中 64 跨步的 shared row table，
  只在本地 batch 装得下**单个** shared tile 时成立（b16/b32 满足，b64 起不满足）。
- outer 提交 **`68f9357`** `Record the measured EP8 b16 best configuration`，父 `3e1a473`。均本地提交，未 push。
- **预约换成 1820**（旧的 1816 已按用户要求取消）：`do-mi350x-03`，8×MI350X、192 CPU、独占，
  2026-09-10 23:03:42 起，**到期 2026-09-11 23:03:42 UTC**。脚本 `.scratch/reserve_mi350x03_20260910.sbatch`。
  重启先查实时 Slurm/GPU；GPU 实验串行，用 `srun` 不用 SSH，不取消他人 job。集群 4 节点全满，只有 node03 是我们的。

### b16 调参结果：0.942× → 1.0492×

口径统一为：EP8、tokens/rank=16、H6144/I3072、128 experts、top4、TP1 shared；
窗口 TopK + routed MoE + shared MLP + 最终和；同进程同输入、8 秒 warmup、12 组 balanced AB/BA、
K16 settling、N200 measured、无 profiler、8rank 各自 HIP-event 均值取最大。

| 配置 | Standalone µs | MegaMoE µs | speedup | 更快 | 95%CI (B/A) |
|---|---:|---:|---:|---:|---|
| 生产配置（文档旧表口径复现） | 263.073764 | 279.260319 | 0.94204× | 0/12 | [1.06032, 1.06263] |
| + S1 `sort_block_m` 64→32 | 263.046797 | 258.683268 | 1.01686× | 12/12 | [0.98121, 0.98509] |
| + `b_nt` 0→2 | 263.860073 | 254.982629 | 1.03481× | 12/12 | [0.96451, 0.96781] |
| **+ `num_dispatch_cu` 32→48（最佳）** | 264.182781 | **251.793405** | **1.04920×** | 12/12 | [0.95098, 0.95478] |

最佳配置完整参数与全部消融/否决证据写在 **`megamoe-s1-1k-prod/bench/dep8_best_configs.json` 的 `entries."16"`**。
它是 `entries."256".config` **加且仅加这三项**（写入前用断言校验过 delta）；S2 的 SBM 跟随 S1 变 32，S2 其余不动。
最佳档 profiled 单次 replay 分段：S1 141.82、S2 75.47、Combine 16.51、TopK 4.85、quant 142.62（quant 是等待，见下）。

**两条必须随配置一起读的限制（已写进 JSON 条目）**：
1. **不是 selector 默认**。生产小 batch selector 仍原样发 `entries."256".config`，要用必须显式传。
   没有改默认是因为那会同时改掉 b32/b64/b128 的行为。
2. **只在 b16 测过**。b32/b64/b128 走同一条小 batch 代码路径但没有各自的配对测量，不能继承这三项。

### 先前 b16 文档值的复现（仍成立）

`.scratch/dep8_b16_production_retime_20260910_v2/b16/result.json`：在**生产源码 db9ccdfaa** 上复现旧表的 b16 行。
旧表值（实验副本）263.633/278.933/0.94515×，本次 263.074/279.260/0.94204×。方向完全复现，
但幅度差约 0.35 个百分点、**两个 95%CI 不重叠**——paired CI 只约束 run 内配对，不覆盖 run 间偏移。
容器内 aiter 加载自 `/usr/local/lib/python3.12/dist-packages/...`，其 sha256 与 pin 的 workspace 源码逐字一致（按内容而非路径核对）。

### 已测出的定量模型与被证伪的假设

- **S1 在 b16 只认字节**：四种几何拟合 `S1 ≈ 46.9 µs + 字节 / 6.42 TB/s`，最大残差 0.12 µs。
  字节 = routed A(16×N_TILES×BM×K) + B(16 expert 权重 604 MB) + shared L13。
  **BM64/N12 与 BM32/N24 字节完全相同（679.5 MB），S1 时间 152.59 vs 152.53**——尽管前者 1 CTA/CU、192 tile，
  后者 **2 CTA/CU、384 tile、VGPR 仅 126**。据此：
  - **“更多 tile 填满 CU 会更快”被证伪**；**“提高 occupancy 有用”被证伪**。不要再走这两条路。
- **S2 同构**：`S2 ≈ 30.7 µs + 字节 / 8.94 TB/s`，最大残差 0.66 µs。8.94 TB/s 已超 HBM 规格，
  说明按“A2 每 N tile 重读一次”算的流量有相当部分命中 L2；**该拟合是描述性的，不是机理**。
- **quant+preplan 的大数字是等待不是成本**：rank 间 17.2–125.8 µs，最快 rank 17.2 µs
  与 b256 正式 breakdown 的 quant 均值 18.45 µs 基本相同。别把它当优化目标。
- **`b_nt` 的机理经 ISA 确认**：`b_nt=2/3` 使 Stage1 的 44 条 `buffer_load_dwordx4` 中 **32 条带 `nt`**（即 B 权重加载），
  `b_nt=0` 一条都没有，`b_nt=1` 也没有（所以 b_nt=1 无收益）。604 MB 权重只读一次，不应占 L2。
  注意这个 0 是从 b256 表继承来的，而代码对 `mtpr<=512` 的默认本来就是 3。
- **`num_dispatch_cu` 两个方向都敏感**：8 → S1 +43 µs、整体 0.870×；32 基线；48/64 更好（48 最佳）。

### standalone b16 breakdown 与上限

`.scratch/dep8_b16_production_retime_20260910_v2/b16/result.json.arm0.rank*.trace.json`，profiled 单次 graph replay：
kernel-sum 286.7 µs（elapsed 263.07，profiled≠elapsed）。
**通信 73.97 µs / 25.8%**（all-gather 45.27 + reduce-scatter 28.70）｜routed GEMM 149.91 / 52.3%｜
shared expert 23.47 / 8.2%｜quant+sort+misc 34.91 / 12.2%｜TopK 4.39。

b16 的 all-gather 每卡只发 **0.197 MB**，所以那 74 µs **是延迟/rendezvous，不是带宽**。
“通信全免”的上限约 **195 µs / 1.35×**；但其中一部分是 skew 等待，**只会搬家不会消失**（我们的 quant 就是它的新家）。
当前 251.79 µs 距该上限约 57 µs，主体是 S1/S2 那两个与字节无关的截距（约 46.9 + 30.7 ≈ 78 µs）。

### 未解决问题与建议的下一步

- **那 46.9 µs（S1）/ 30.7 µs（S2）的截距尚未识别**。它只是“字节解释不掉的残差”，不是已定位的开销项。
  已排除：shared L13 权重读取（在斜率里）、tile 形状/tile 数/占用。已知强相关：dispatch CTA 数量。
  Combine 只搬 786 KB 却要 15–16 µs、TopK 4.4 µs，同样几乎全是固定成本。
- **建议的决定性探针（未跑）**：用同一份最佳配置跑 **b32**。b32 每 expert 8 行、仍是 1 个 32 行 M tile，
  **字节数与 b16 完全相同**，但 payload 与跨 rank 传输翻倍。S1 上升 → 截距是 payload/等待，值得继续攻；
  持平 → 是 launch/ramp 类硬性固定成本，b16 这个规模到头。需把 bench 的 `tokens==16` 断言放宽到 16/32。
  注意 b32 同样满足“单个 shared tile”前提（ceil(32/32)=1），新 assert 不会拦。
- b256 baseline 575.863→591.387（+2.696%）的跨 run 偏移**仍未定位**，与本轮无关但仍在。

### 被否决的候选（都有证据，不要重试）

- `tile_n` 512→256：更慢（单独 288.32 µs/0.913×，叠加 BM32 266.06/0.987×）。原因是 A 被重读的次数翻倍、字节从 641.7→755.0 MB。
  为跑通它我在实验副本里推广了两处硬编码（shared panel 切分、routed claim 的 N 槽数与 home 步长），
  **这些推广没有进生产**，因为 N_TILES=24 用不上了。若将来要用，代码在
  `.scratch/dep8_b16_s1_tilen256_20260910_v1/kernels/`，且已用 CPU 枚举验证过 N_TILES=12 与现状逐 tile 等价、24 完整覆盖。
- `num_dispatch_cu=8`（S1 +43 µs）、`waves_per_eu_hint=4`（S1 385 µs）、`b_nt=1`（ISA 无 nt 位）。
- `skip_launch_barrier`、`waves_per_eu_hint=1`：落在跨 run 噪声内，判为无效。
- `grid_mult=2`：被 preplan 拒绝（`preplanning supports compact EP8 top4 with grid_mult=1`）。
- **S2 `block_m=16`**：编译通过但数值错。诊断 `.scratch/dep8_b16_s2_sweep_20260911_v1/s2bm16_diag/run.log`：
  **shared L2 正确（rel 0.0014），routed 错误（rel 0.951）**；`total_m_blocks` 已是 BM 通用的，
  问题在 routed epilog/scatter 数据通路，**需要改 kernel 而不是放宽 guard**。模型上限仅约 4.5 µs，投入产出比低。
- **S1 `sort_block_m=16` 不可行**：`M_REPEAT = sort_block_m//16` 必须为偶数（MFMA 累加器成对布局），
  16 会得到 1 直接 assert 失败。32 是当前 GEMM 结构下的下界。

### 方法论：本轮踩到并已修复的三个坑（务必保留）

1. **override 经环境变量会被 enroot 白名单静默吞掉**。`run_mi355x_enroot.sh` 需要逐个 `--env` 转发。
   后果：`dcu8/dcu16/bnt3/slb1` 四轮实际跑的都是同一个 baseline，我据此得出的“dcu 无影响”结论一度是错的。
   **已改为通过文件传递**（`OUT/override.json`），并**新增编译产物门禁**：S1/S2 kernel 名字必须带上
   每个 override 对应的后缀（`_dcu48_`、`_bnt2_`、`_t32x`、`_qg5_` 等），对不上直接 fail。
   这道门禁与传参方式无关，是这类静默失效的通用防线。四个无效目录保留为 `*_NO_OVERRIDE_APPLIED`。
2. **打补丁必须先断言锚点存在**。S2 BM16 的白名单补丁因元组顺序不符而静默没生效（用了 `str.replace` 没配 `assert old in s`），
   浪费一次 GPU 运行。现在每处补丁都先断言、打完再 grep 复核。
3. **harness 因控制机内存不足杀后台任务时，SIGTERM 会传到 srun 并取消 GPU step**（1820.9 就是这样没的）。
   控制机只有 7.9 GB 且与其他 agent 会话共享，可用常在 3 GB 上下。
   **长实验一律用 `setsid nohup` 脱离进程树**，进度写文件（如 `SWEEP_PROGRESS`），回来读文件即可。
4. **跨 run 差异 <1% 不可分辨**：同一配置连跑 5 次，B 落在 258.68–261.09 µs（0.93%），A 落在 263.05–264.26（0.46%）。
   排序配置必须用**同 run 的配对 speedup**，不能比裸 B。

### 证据目录

- 生产复现：`.scratch/dep8_b16_production_retime_20260910_v2/`（v1 因我写错的路径门禁失败，无有效数据）
- BM32：`.scratch/dep8_b16_s1_bm32_20260910_v1/b16_v4/`（v1/v2/v3 分别被 S2 白名单、我的过时门禁、Slurm credential 拦下）
- S1 参数扫描：`.scratch/dep8_b16_s1_sweep_20260910_v1/`（`bnt2_v2`/`bnt2_rep`/`bnt3_v2`/`bnt3_rep2`/`dcu48`/`dcu64`/`bnt2_dcu48` 有效）
- tile_n=256：`.scratch/dep8_b16_s1_tilen256_20260910_v1/b16_v3/`、组合 `.scratch/dep8_b16_s1_bm32_tilen256_20260910_v1/b16/`
- S2 BM16 诊断：`.scratch/dep8_b16_s2_sweep_20260911_v1/s2bm16_diag/`
- 每个有效目录都有 `result.json`、8rank trace、`ir-rank0..7` 的 `21_final_isa.s`、`source_manifest.json`、`protected_documents.json`

重启先读本节与 `tool_measure.md`，核对 inner/outer HEAD、git status、Slurm 1820 与 GPU 占用。
本轮所有 GPU 工作已结束，无待轮询任务。**不要自动恢复 combine-front、旧扫描或作图；用户偏好只导出 trace。**

## 2026-09-10 22:03 UTC 小 batch shared 支持已提交，测时与交接更新

本节优先于下方历史。用户明确授权“代码commit一下，更新文档”。inner `megamoe-s1-1k-prod/aiter`
分支 `s1-unified-production`，当前提交 **`db9ccdfaa851d87dbf562d9082744d067191285b`**，父提交 `7e433ab4e7053fb3d901b7976762e1fb4da23437`，本地提交，未 push。
此前 b16/32/64/128 的 shape-support 实验副本现已合入生产源码；六档 batch 使用同一份源码的编译期特化。
旧实验、测量 manifest 和 checkpoint 保持原样。本节区分原正常测时与本次提交验证。

### 当前配置与调用约束

- 新增 EP8/H6144/I3072/E128/top4、TP1 shared 的完整本地 batch **16/32/64/128** 支持，实际输入和 `mtpr` 都取该档大小，未补输入到256。
  shared L13 融入 S1、shared L2 融入 S2，最终 Combine 保留 FP32 求和；完整路径5个 kernel。
- 小 batch 在完整本地 batch、无 local_reduce、P2P BF16（`p2p_quant="none"`）、无 Stage1 环境覆盖的条件下，
  shared selector 默认采用此前实测的完整 b256 参数：compact routed、S1 M64/N512/K256、**preplan4**、
  S2 M32/N128/K256/SBM64、640 CTA、band8、**early2**；routed/shared XCD 均开启，Combine shared/本地 routed 预读开启。
  同一 stream 先调用 **`op.route(scores, topk_weights, topk_ids)`，再调用 `op(x, topk_weights, topk_ids)`**，以满足 preplan4 的状态准备。
  shared 仍要求 `cur_tok == mtpr`；其他 batch/形状或自定义参数不能据此视为已验证。
- 小 batch 的记录为 **`megamoe-s1-1k-prod/bench/dep8_b256_config_scaling.json`**，完整参数位于 `.config`。
  这是沿用 b256 参数的 scaling 测试，**未做逐 batch 最佳配置搜索**；b16 standalone 更快，未添加自动 fallback。
- 原 routed-only selector 与既有 b256/b8192 调优默认保持不变。原最佳观测表
  **`bench/dep8_best_configs.json`** 仍只记录 b256 early2、b8192 tail；这两档重现性能仍使用该表完整冻结参数，
  不能把裸 b256 selector 的 preplan0 当作本次测时配置的 preplan4。
  b8192 使用1200 CTA、local_reduce+XCD-local，Combine 不启用本地/shared 预读。
- `Stage2Config.shared_schedule` 仍是已有的 `tail`/`early2` 编译期开关，本次没有新增调优开关。
  <=256 的 early2 仍用 `((block_id//8)%8)<2`：160/640 CTA shared-first，其余 routed-first，两类各遍历8队列。
  **不保证同一 CU 驻留混合，也不是逐 tile 1:4 交错。** b8192 early2 的原快速测时慢0.5524%，所以仍用 tail。

### 正常完整路径测时（原测量，本次未重新计时）

| Tokens/rank | Standalone (µs) | MegaMoE (µs) | Speedup | Latency reduction | Standalone CV | MegaMoE CV | Faster pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 263.633170 | 278.932633 | 0.94515× | -5.8036% | 0.24477% | 0.07680% | 0/12 |
| 32 | 291.750827 | 286.118733 | 1.01968× | 1.9304% | 0.16094% | 0.12447% | 12/12 |
| 64 | 347.672803 | 298.017828 | 1.16662× | 14.2822% | 0.09850% | 0.15813% | 12/12 |
| 128 | 437.512811 | 313.337557 | 1.39630× | 28.3821% | 0.04416% | 0.11671% | 12/12 |
| 256 | 591.386833 | 388.617347 | 1.52177× | 34.2870% | 0.20013% | 0.10161% | 12/12 |
| 8192 | 8999.817912 | 4528.935496 | 1.98719× | 49.6776% | 0.10401% | 0.23992% | 12/12 |

统一口径：EP8，每rank实际 token 数，TopK+routed+TP1 shared+最终和，不含 router GEMM/residual。
各 batch 同进程、相同输入/权重，warmup8s、12组 balanced AB/BA，K16，<=256 N200、b8192 N20；
无 profiler 的 HIP-event，报告8rank max-local-mean，非全局 wall-clock makespan。
四档小 batch 使用同一份 b256 冻结参数；b256/b8192 两行引用此前各自正常运行。
b16 慢 **5.8036%**（95%CI 慢5.6287%–5.9411%，0/12更快）；b32快1.9304%，b64快14.2822%，b128快28.3821%。
完整置信区间和12轮 variation 见 scaling JSON。

有效原测量入口：

- b16/b32：`.scratch/dep8_b16_b32_b256_config_20260910_v1/b16/`、`b32/` 的 `result.json`、`isa_audit.json`。
- b64/b128：`.scratch/dep8_b64_b128_b256_config_20260910_v3/b64/`、`b128/` 的 `result.json`、`isa_audit.json`。
- b256：`.scratch/dep8_b256_early2_vs_standalone_20260910_v5/paired/result.json`；
  b8192：`.scratch/dep8_b8192_current_vs_standalone_20260910_v1/paired/result.json`。
- 四档合表：`.scratch/dep8_b16_b32_b256_config_20260910_v1/comparison.json`。
  原正常测时采用隔离实验源码，本提交与各档所选 GPU code object 全8rank逐字一致，见下方验证；不能表述为新提交重新测时。

b64 首轮 v1 在正常计时前被 baseline graph/eager 的0.1%阈值拒绝，诊断显示 standalone 自身 eager/eager 也可达约0.14%，
graph/eager 同量级，独立完整数学参考误差约0.48%。诊断见
`.scratch/dep8_b64_b128_b256_config_20260910_diag_v2/b64/failure_diagnostic.json`。
正式 v3 与后续 b16/b32 对 baseline 每个探针采用1%独立数学参考及 graph/eager 检查；candidate 仍要求1.5%数学参考和 graph/eager 逐位一致。
v1 与诊断 v2 不属于有效性能结果，记录原样保留；不能把 baseline 的 BF16 atomic 累加顺序差异当成 candidate 错误或优化收益。

### 部分 tile 与队列边界

- shared S1 用 `ceil(T/64)` 个 M tile；只对 T16/32 的部分 tile 使用运行时 ceil，完整 tile 保留既有计算，避免改变 b64及以上的编译结果。
  shared A 和输入 scale descriptor 按实际有效行数设界，routed A 仍是完整 tile；shared 输出界限按实际容量。
- S1 十二个 N panel：queue0..3 各分两个，queue4..7 各分一个。每次调用 shared head epoch 分别为
  `256 + 2*ceil(T/64)`、`256 + ceil(T/64)`：T16/32/64是258/257，T128是260/258，T256是264/260。
- shared S2 用 `ceil(T/32)` 个 M tile，队列票据按 band8 补齐，未覆盖有效行的 tile 跳过。
  <=256 每个 shared queue 仍为48个票据+640次耗尽，**epoch688**；T16/32/64/128/256 每队列实际有效 tile 数为6/6/12/24/48。
  T16 的 shared A descriptor 同时约束初始预读和 GEMM 循环加载；既有最终 BF16 输出界限保留。
- Combine 保留128 CTA×8waves。T16/32 每token最多16warp，分别256/512个有效warp；其余warp的路由读有界，
  本地/远端 descriptor 长度为0，shared读和最终输出均有界。>=64 的原编译路径保留。
- 本次没有恢复此前 async-mark/K0/最后两 iter peel 实验，没有改变已接受的 GEMM 流水或 FP32 Combine 求和语义。

### 提交验证与可恢复证据

最终有效验证目录 **`.scratch/dep8_small_batch_production_20260910_v2/`**：

- 16/32/64/128/256/8192 六档均在8rank通过 tail/early2 两种调度、7组输入/路由探针
  （initial、negative_large、zero、changed_routes、all_local、all_remote、restored），每探针每调度256次连续 graph replay。
  每个探针重算独立数学参考；最终/shared 输出 poison 后逐位一致，持久 epoch/counter、FP32 Combine oracle 均通过。
- 每档全部8rank所选 quant/S1/S2/Combine **code object 与对应原正常计时逐字一致**；5-kernel dispatch 一致，无额外 memcpy/memset。
  小 batch 的 early2 测试实际使用生产 selector，并与冻结 b256 config 完整相等；另有32组原 routed selector 行为对比通过。
- source/document pin 从开始到结束通过。HIP 资源上限均3 CTA/CU；<=256 S2 147VGPR、24896B LDS，b8192 163VGPR、33344B LDS，
  private memory=0、VGPR spill=0。存在记录在 resource JSON 的 SGPR spill，不能写成所有寄存器均无 spill；资源上限并非实测 achieved occupancy。
- 初次 promotion v1 的 b64 功能检查通过但 S1 code object 不同，定位为将运行时 ceil 无条件用于完整 tile。
  已改成仅部分 tile 分支并在独立 v2 中重验全部六档；v1 不用作当前提交等价证据。
- 版本化 checkpoint：**`megamoe-s1-1k-prod/bench/checkpoints/dep8_small_shared_production_20260910/`**，
  收录四档原正常测时结果/ISA审计、测时驱动与相对 `7e433ab4e` 的实验源码补丁、最终六档验证/审计/脚本/原始 pin、
  baseline诊断、配置表、三份文档快照及提交信息。大型 IR/trace 保留原 `.scratch` 路径。
  历史 manifest 保留测量当时的 hash；文档在验证结束后才更新，未来重跑需生成新目录和新 pin，不能修改旧记录来匹配新文档。

本任务已完成，无待自动续跑实验。用户偏好仍为只导出trace、不画图；本次没有测 breakdown、没有重做性能配对。
后续未决问题仍包括 baseline 575→591 µs 跨run偏移的归因，以及更好的 CTA mixed schedule；本次不宣称已定位这些根因。

## 2026-09-10 21:30 UTC 配置更新：shared_schedule 显式开关与 EP8 最佳配置表

本节优先于下方历史。用户授权“把它也加进去，然后commit”，将 early2 纳入配置及配置记录；
本次仅更新相关交接与配置记录。文档授权规则仍保留在文件最前面。
inner `megamoe-s1-1k-prod/aiter` 提交 **`7e433ab4e7053fb3d901b7976762e1fb4da23437`**，
父提交 `e327e7eafabe4f339c9c1e8cb164179af1a8998a`，分支 `s1-unified-production`，本地提交，未 push。

- 新字段 **`Stage2Config.shared_schedule = "tail" | "early2"`** 已贯通调用参数与编译缓存键。
  这是同一份源码的 compile-time 选择，无新增 GPU runtime 分支。
  dataclass 通用默认 `tail`；EP8 b256 生产 preset 显式设 `early2`；b8192 保持 `tail`。
  shared L2 未开启时此字段不影响调度；非法字符串报错。
- 新权威 EP8 配置表：**`megamoe-s1-1k-prod/bench/dep8_best_configs.json`**，含两档完整配置、
  local_reduce/XCD-local/shared fusion/Combine 预读状态，以及正常计时与调度选择的证据入口。
  这是当前已验证的最佳观测配置，不代表全局最优。旧 `s1_best_configs.json`、`s2_best_configs.json`、
  `full_best_configs.json` 是既有历史表，不用 EP8 数据覆盖它们。
- **只有调度字段被显式化，其他裸 selector 调优默认未改。** 重现 388.617/4528.935 µs 的完整配置须读新表
  `entries["256"/"8192"].config`，分别构造 `Stage1Config`、`Stage2Config`、`MegaMoEConfig`，
  并按同一 entry 的 metadata 设置 local_reduce/shared 权重等。
  老 JSON 不含 `shared_schedule` 时按通用默认 `tail`；自定义 b256 配置若要 early2，须补字段。
  已有 cfg 可用 `replace(cfg, stage2=replace(cfg.stage2, shared_schedule="early2"))` 切换。

| EP8 tokens/rank | 记录的 shared_schedule | S2 CTA | Routed/shared XCD | Local reduce | Combine 本地/shared 预读 |
|---|---|---:|---|---|---|
| 256 | early2 | 640 | 均开启 | 关闭 | 开启 |
| 8192 | tail | 1200 | 均开启 | 开启，XCD-local | 关闭 |

early2 仅接受已验证的 shared L2 几何：b256 `(BM,BN,BK,SBM,band,cu,qg)=(32,128,256,64,8,128,5)`、
无 local_reduce；或 b8192 `(64,128,128,128,16,240,5)`、local_reduce+XCD-local。
b256 early2 调度本身未变：160/640 CTA 先 shared，其他先 routed，每类各遍历 8 个队列。
b8192 的可选 early2 用同一 block-ID 规则选 304/1200 CTA 先 shared，其余 896 先 routed；
**每 CTA routed 恰访问一次 home 队列，shared 恰遍历 8 队列**，共 9 次访问，保留 XCD-local 约束和 epoch。
两档都不保证同一 CU 驻留混合，也不是逐 tile 1:4 交错。

### b8192 early2 的快速正常计时结论

此前按用户要求完成，未新增 breakdown：同 op/workspace，12 组 balanced AB/BA、warmup 8s、K16/N20，
TopK+routed+TP1 shared+最终和的正常 HIP-event 窗口，8rank max-local-mean。
tail **4523.471324 µs**，early2 **4548.475583 µs**；early2 慢 **0.5524%**，
95%CI 慢 **0.2899%–0.8124%**，仅 1/12 pair 更快。因此 b8192 最佳配置仍选 tail。
证据 `.scratch/dep8_b8192_early2_20260910_v1/paired/result.json` 与 `isa_audit.json`，均通过。
不要用此次 tail 的 4523.471 与此前 4528.935 跨 run 差值宣称代码优化。

### 显式开关验证与提交证据

- `.scratch/dep8_shared_schedule_config_20260910_v1/b256/`、`b8192/` 的 `result.json`、`isa_audit.json` 均通过。
  每档 8 卡同时验证 tail/early2；独立完整数学参考、7 组输入/路由探针，每探针每调度 256 次 graph replay，
  切换后最终/shared 输出逐位一致，持久 epoch/counter、5-kernel dispatch、source/document pin 检查通过。
- 两档所选配置（b256 early2、b8192 tail）在全部 8rank 的量化/S1/S2/Combine **code object 与此前正常计时逐字一致**。
  本次为配置接口与二进制等价验证，未重新进行整层性能测试；下节既有正常计时仍是原来的测量，不伪装为新提交重测。
- HIP 静态资源上限两档两调度均 3 CTA/CU，无 private memory/VGPR spill。
  b256 两者147 VGPR/LDS24896B，SGPR 保存至 VGPR lane 为 tail9/early2 11；
  b8192 两者163 VGPR/LDS33344B，对应29/30。静态上限不是实际 achieved occupancy。
- 新 checkpoint：`megamoe-s1-1k-prod/bench/checkpoints/dep8_shared_schedule_config_20260910/`，
  保存配置、三份交接快照、验证脚本和小型结果/ISA审计。原测量 manifests 与旧 checkpoint 保持原样。
  本次测试已结束才更新文档；测试时 source/document pin 是不可变的历史快照。
  用户偏好仍是 **只导出 trace，不画图；本轮不测 breakdown**。

### 其他已证实有收益的开关（已在当前冻结配置中使用）

| 选项 | 历史同轮对照证据 | 当前状态 |
|---|---|---|
| S1 `preplan_waves=4` | b256 无 shared 路径 377.312→371.694 µs，降 1.489%，3/3 pair | 两档配置表均为4；裸默认仍0 |
| S1 `payload_tile_publish_early=True` | b8k biased routing、固定 chunk512：4907.060→4761.495 µs，降 2.966%，3/3 pair | b8k 开、b256 关 |
| early publish 配合 `payload_chunk_rows=2048` | b8k biased routing：4904.601→4657.655 µs，降 5.035%，3/3 pair；含两项改变 | b8k 已用此组合 |
| Combine `prefetch_local` | b256 S2+L2+Combine、同为 FP32 sum：159.903→158.567 µs，降 0.8355%，12/12 pair | b256 自动开启；不是 Stage2Config 字段，b8k 关闭 |

这些是不同阶段和路由场景的历史单项/组合证据，**不能相加，也不代表在当前 388/4529 µs 上还能重复获得收益**。
preplan 证据 `.scratch/dep8_b256_preplan_20260910/w4_v1/result.json`；
early publish 证据 `megamoe-s1-1k-prod/bench/checkpoints/dep8_preplan_tile_publish_20260910/summary.json`；
Combine 证据 `.scratch/dep8_b256_combine_prefetch_20260910_v2/paired/result.json`。
本轮 asyncmark/K0/最后两 iter peel 未保留，没有可靠净收益证据支持把它们新增为最佳配置开关。

## 2026-09-10 21:00 UTC 后重启入口：early2 已提交，b256/b8k 完整测时与 8 卡 trace 已完成

本节优先于下方历史。用户最新授权：“代码commit一下，更新文档，我要重启对话了”。
本次更新本交接、`summary_megamoe.md`、`plan_megamoe.md` 与 outer `HANDOFF.md`；顶部文档授权规则继续保留。
用户对 breakdown 的最新偏好是 **只导出 trace，不画图**。本轮没有待续跑的 GPU 实验。
**用户再次确认的关键区别：b256 是 early2，b8k 仍是 tail；这次没有给 b8k 启用 early2。**
版本化 checkpoint：`megamoe-s1-1k-prod/bench/checkpoints/dep8_early2_full_20260910/`，
保存三份文档、冻结 `configs.json`、两档正常测时/ISA审计/trace摘要；大型trace仍在下文 `.scratch` 原目录。

### 当前提交、源码与配置边界

- inner：`/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`。
  新提交 **`e327e7eafabe4f339c9c1e8cb164179af1a8998a`**：`Use early shared CTA scheduling for EP8 b256 S2`，
  父提交 `cab83c22a0c1682d6a9d2e4ec25ea7606ba00fdd`。提交后 inner worktree 干净；本地提交，未 push。
  仅提交 `mega_moe_stage2.py` 的既有 early2 改动，未改配置 selector。
- b256 与 b8192 两轮有效测量的 **19 个生产源码文件哈希全部一致**，提交时再次核对仍一致。
  测量 manifest 记录的是提交前 `cab83c22a + dirty stage2`，与新提交源码逐字一致，不要误判成不同版本。
  Stage2 SHA256：`6765090f4179cde6914ef5e71a30874e4be809997302a54ac0b82b8659b990f3`。
  `gemm2.py` SHA256：`eb82721179374929ca9965ce1c65baa010bb4afd6f294d8df3c5e7aa986f8689`。
  **本轮试验的 asyncmark、拆 K0、最后两迭代 peel 已回退，不在当前代码中。**
- 两档使用同一份源码，但配置与特化分支不同。完整冻结配置位于
  `.scratch/dep8_b256_worktree_early2_20260910_v1/configs.json`，也保存在各有效测量的 `manifest.config`。
  **下面性能是冻结调优配置，不是裸 `_select_config` 默认性能。** b256 额外设 `preplan_waves=4`（裸默认 0）。
  b8k 额外使用 S1 work_shards8、padding_uniform_srcmap/count_uniform_matrix/row_base_prefetch/prefetch_b_before_a、
  preplan4、payload_chunk_rows2048、early tile publish、packed A scale、unroll A pingpong、split A LDS、XCD/band4；
  S2 N128/K128/band16。对应裸默认差异也保存在两轮 `manifest.default_selector_config` 中。

| 当前 EP8 分支 | b256 | b8192 |
|---|---|---|
| S1 M/N/K | 64/512/256，dcu32 | 128/256/256，dcu96 |
| S2 M/N/K，SBM | 32/128/256，64 | 64/128/128，128 |
| S2 persistent CTA | 128×5=640 | 240×5=1200 |
| Shared L13 | 融入 S1 | 融入 S1 |
| Shared L2 | 融入 S2，early2 | 融入 S2，tail |
| Routed / shared XCD | 两者均开启 | 两者均开启 |
| Local reduce | 关闭 | 开启，XCD-local |
| Combine | `ep_combine_intranode_pf_local_shared_f32` | `ep_combine_intranode_shared_f32` |
| 完整路径 kernel 数 | 5 | 5 |

### early2 的实际调度，以及没有保证的事情

- 生效 guard：shared L2 开启、无 local_reduce，且
  `(npes,max_tok,BM,BN,BK,SBM,band_m,cu_num,queue_grid_mult)==(8,256,32,128,256,64,8,128,5)`。
- `shared_first=((block_id//8)%8)<2`：160 CTA 先 shared 后 routed，480 CTA 先 routed 后 shared。
  每 CTA 先遍历第一类的 8 个队列，再遍历第二类的 8 个队列；当前队列取到失败 claim 后才换下一个队列。
  **不是每 tile 按 1:4 交错，也不是保证同一 CU/SM 内混合。** block-ID 排列只是一种分布启发式。
- 先由 CTA/phase 选择 routed 或 shared，再对该类独立 XCD head 做 atomicAdd；ticket 只解码该类 tile 坐标。
  不是共用一个全局 job-id 再决定任务类型。每 CTA 仍对每个 shared 队列恰有一次失败 claim，
  b256 每 head 每调用递增 **688=48 valid+640 exhausted**，graph epoch 协议未变。
- b8k 不进 early2：routed 只处理 home XCD 队列，然后 shared 遍历 8 队列；L2 epoch 1968，L13 epoch 448。
- 当前 b256 S2：147 VGPR、106 SGPR、LDS24896B、private0/VGPR spill0，SGPR 保存至 VGPR lane 共11。
  HIP 查询静态上限3 CTA/CU，不等于实测 occupancy。b8k 为163 VGPR、LDS33344B、private0、SGPR lane29。
- 当前 Combine 语义仍为 `BF16(sum_FP32(routed)+FP32(shared))`；shared HBM 输出本身仍为 BF16。
  b256 先预读 shared 与本 GPU routed，等待 peer flags/system acquire，再读 remote；b8k 无该预读特化。

### 最新完整路径正常测时（有效结果）

EP8、tokens/rank=256 或8192、H6144/I3072、128 experts、top4、TP1 shared。
窗口为 **TopK + routed MoE + shared MLP + 最终求和**，不含 router GEMM、residual。
同进程组、同输入权重、8秒 warmup、12组 balanced AB/BA；每 sample 取8 rank 各自 HIP-event mean 的最大值。
K16 settling 不计入窗口；b256 每 sample 200 measured replays，b8k 20；不能称为独立单次延迟尾分布。

| Batch | Standalone mean µs | Current mean µs | Paired speedup | Latency reduction | Standalone CV | Current CV |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 591.386833 | 388.617347 | 1.52177× | 34.2870% | 0.20013% | 0.10161% |
| 8192 | 8999.817912 | 4528.935496 | 1.98719× | 49.6776% | 0.10401% | 0.23992% |

- 两档均12/12更快；降时95%CI分别34.2325%–34.3425%、49.5885%–49.7544%。
- b256：`.scratch/dep8_b256_early2_vs_standalone_20260910_v5/paired/result.json`，同目录 `isa_audit.json`。
  **v1–v4 均未通过最终 harness dispatch gate，其时间作废，只引用 v5。**
- b8k：`.scratch/dep8_b8192_current_vs_standalone_20260910_v1/paired/result.json`，同目录 `isa_audit.json`。
- 两档均通过独立完整数学参考、initial/negative_large/zero/changed_routes/all_local/all_remote/restored、
  各探针256次 graph replay、持久 counter/epoch、source/document pin、GPU独占与全8rank dispatch gate。
  Standalone BF16 atomic reduction 使用事先容差，不能强求逐位一致；candidate 对相应 oracle 检查。
  b8k all-rank S2 code object 与此前有效 regression 逐字一致。
- baseline 为 native MXFP8 shared（`VLLM_ROCM_MXFP8_PTPC_AITER=0`），b256 每 forward18 kernels，b8k19；
  b8k 的 intermediate quant/sort 是2节点。候选均5 kernels，不漏 shared/final sum。
- 核对 launch 顺序须使用 eager trace 的 GPU correlation → host hipLaunch 顺序，graph 核对 multiset；
  不要按 GPU start timestamp 排序断言跨 CU 的先后，否则会重复旧 harness 误判。

### 全部 8 卡 breakdown trace（已完成，仅导出即可）

- b256 根目录：`.scratch/dep8_b256_early2_full_breakdown_20260910_v1/exports/`。
  `standalone_8gpu.trace.json`、`early2_8gpu.trace.json`。
- b8k 根目录：`.scratch/dep8_b8192_current_full_breakdown_20260910_v1/exports/`。
  `standalone_8gpu.trace.json`、`current_b8k_8gpu.trace.json`；只含 trace 的包为 `traces_all8gpu.zip`。
- 每份合并 trace 含 rank0..7；另有逐卡 trace、`*_8gpu_pair0_replay10.trace.json`、
  `stage_breakdown_all8ranks.csv`、`forward_spans_all8ranks.csv`、`dispatches_all8ranks.csv.gz`、`breakdown.json`。
  原始 rocprof 数据在各根目录的 `capture/rank0..7`。无需再生成图片。
- 12组、K16/N20，每臂每 rank240 measured+192 settling forwards。b256 每 rank baseline7776/current2160 kernels，
  b8k8208/2160；全部8rank dispatch、shape、ISA与源文件核验通过，无漏卡。
- 当前 profiler 阶段均值（8rank×12pair，µs）：b256 TopK4.924、quant/preplan18.452、S1+L13 215.537、
  S2+L2 117.506、Combine26.140；b8k分别41.705、80.387、2144.853、1806.886、442.073。
  **Profiler 时间只作诊断，不替代上表；kernel sum 不等于 elapsed span；跨 GPU 时钟未独立校准。**

### 重启时须保留的结论与未决项

- 用户问旧575 vs466为何成为591 vs388：旧 `.scratch/dep8_current_full_20260910_v2/b256/result.json`
  实际为575.863215/466.535568，**当时 b256 shared L13 和 L2 都独立执行**，并非仅 L2 separate。
  旧候选11 kernels；当前5 kernels。相同配置但实现已含 L13融合、L2融合、early2、Combine优化与移除独立 final add。
  历史独立配对支持 L13融合省59.372µs、L2融合省13.636µs、Combine改动省1.816µs；
  L2对照是提取出的独立 GEMM，非旧 native linear，窗口也不同，不能相加成77.918µs的精确分解。
- **Baseline 575.863→591.387 的+15.524µs/+2.696%尚未找到根因。** 已核对18项baseline源码哈希、
  全8rank18-kernel序列、两 routed GEMM 的ISA均一致，输入权重/主机/软件一致。
  旧100replays、新200，但新100/K16敏感性对照仍591.716µs，不能归因于N翻倍；采样时钟接近。
  不要把它直接叫随机噪声或降频，也不要把两侧差距扩大93.442µs全部计为候选优化收益。
- 历史153µs是 **S2+L2+Combine局部窗口**，不是完整层时间，且跨进程不稳定；
  `.scratch/dep8_b256_early2_reproduce_20260910_v1/comparison.json`：一次复现153，另一次157.82。
- 最终选回原 GEMM early2。`.scratch/dep8_b256_worktree_early2_async_20260910_v1/paired/result.json`：
  tail+async158.394、early2原GEMM156.257、early2+async159.764µs，async版本慢2.244%（0/12胜），未保留。
- 当前 early2 XCD开/只关routed/两者关：154.199/155.080/154.981µs，保留两者开。
  来源 `.scratch/dep8_b256_early2_xcd_off_20260910_v1/paired/result.json`；其他旧tail+async的关闭结果不可混用。
- 比例/队列旋转试验 `.scratch/dep8_b256_early_schedule_ratio_20260910_v1/paired/result.json`：
  early2/sparse/rotated=155.224/157.020/156.988µs，后两者均0/12胜，未采纳；对应 PMC 已结束。
  **仍无保证同 CU 混合的实现。** 不要将 block-ID 轮转描述成硬件驻留保证。
- VMEM诊断入口 `.scratch/dep8_b256_vmem_issue_rootcause_20260910_v1/result.json`：
  async/K0实验实际ISA已放宽目标稳态wait；K10必要的B11/scale等待、K11最小wait的核对通过，
  但正常窗口无净收益，VMEM issue stall增加。PMC不支持简单归因LFIFO/RFIFO或DRAM credit耗尽。
  **这些诊断针对实验版，不代表当前已回退的原 GEMM 的 vmcnt 已改变。**
  其他诊断入口：`.scratch/dep8_b256_overlap_cost_split_20260910_v1/result.json`、
  `.scratch/dep8_b256_p2p_isolation_20260910_v1/paired/result.json`；local替代P2P为诊断探针，不能当正确实现性能。
- 本次授权文档更新与commit会改变文档hash/HEAD，旧manifest保留原样；续测要生成新的source/document保护快照，
  不要覆写旧结果或删除保护gate。当前19个生产源码哈希仍与两档有效测量一致。
- 21:07 UTC查看 allocation1816仍在 `do-mi350x-03`，8 MI350X，预计23:30:19 UTC到期；
  重启须查实时Slurm/GPU状态。GPU任务串行用srun，不用SSH，不取消allocation或无关历史steps。
  本轮所有 GPU 计时/trace/PMC 已完成；不要自动恢复 combine-front、旧扫描或作图。
- 复现实验 wrapper：`megamoe-s1-1k-prod/bench/experiments/dep8_preplan_tile_publish_20260910/uniform/run.sh`，
  `M3_JOB=<有效allocation>`、`LOCAL_REDUCE_BENCH_ENTRY=<对应bench.py>`、`LOCAL_REDUCE_OUT=<新目录>`，
  `M3_ENROOT_CONFIG=/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/bench/experiments/dep8_standalone_shared_20260910/enroot.conf`，
  `VLLM_ROCM_MXFP8_PTPC_AITER=0`。从现有bench重新准备pins后运行，不能原样复用旧doc快照。

## 2026-09-10 18:00 UTC 重启入口：b256 shared 已融合，Combine FP32 + 本地预读已提交

本节优先于下方历史。用户本轮明确要求“在等结果的同时更新文档，后面要重启了”，因此授权更新本交接、
`summary_megamoe.md`、`plan_megamoe.md` 及 `megamoe-s1-1k-prod/HANDOFF.md` 副本；顶部文档授权规则继续保留。

### 当前代码与执行状态

- inner：`/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`。
  HEAD **`cab83c22a0c1682d6a9d2e4ec25ea7606ba00fdd`**，提交标题
  `Enable b256 shared L2 and prefetch local Combine inputs`。父提交 **`c4c093eb1`**
  `Support shared L13 fusion in EP8 256-token S1`，再前面是 `999ace3bbe9c86d1602ae87fb25856e9460b309d`。
  两次代码提交均已完成，未 push。以下 CTA/提前调度试验仅在 `.scratch`，没有改生产默认或新增生产提交。
- 当前支持 EP8、tokens/rank=256 或8192、H6144/I3072、128 experts、top4，完整 TP1 shared。
  b256：S1 M64/N512/K256/W8，dcu32、grid_mult1、work_shards8、band4、XCD；
  S2 M32/N128/K256、SBM64、4waves/256threads，persist_cu128 × qg5 = **640 CTA**，band8、XCD、无 local_reduce。
  完整冻结配置见下述每个有效实验目录的 `configs.json`。
- 完整 shared 路径共 **5 kernels**：TopK；input quant + preplan；S1 + shared L13；
  S2 + shared L2；Combine。shared 不走独立通信 expert。
- GPU仍是 allocation **1816**、`do-mi350x-03`、8卡 MI350X/gfx950，记录到期23:30:19 UTC。
  重启先查实时 Slurm/GPU 状态，GPU实验串行，用 `srun`，不用SSH；不要取消allocation或无关历史steps。
  旧 combine-front 后台仍已停止，不恢复。本轮全部GPU实验已结束，新Combine三项对照已通过；没有本任务活跃GPU step。
  本轮文档保存在工作区，生产代码HEAD仍为上述cab提交，没有新增生产提交。

### 已接受的 Combine 语义与顺序

- 用户明确指定：**`BF16(Σ_FP32 routed + FP32(shared))`**。删除的是 routed 求和后、加 shared 前的中间 BF16 舍入。
  routed P2P partial 和独立 shared HBM 输出仍是 BF16；没有把 S2 的共享输出改成 FP32，也没有删除 shared HBM 中间张量。
- b256 shared 开启且无 local_reduce 时启用 `ep_combine_intranode_pf_local_shared_f32`：
  先按既有协议发布本rank完成，再预读 **shared 和本GPU负责的 routed slots**，随后等待peer flags/system acquire，
  再读远端来源slots，按slot0..3做FP32和，shared最后加入，只在最终输出转BF16。
- 本地slot由topk expert-id和rank确定。预同步对远端slot用零长度descriptor屏蔽；同步后用互补descriptor避免重复读本地slot。
  不能把全部 routed 读都提前到系统同步之前。CTA内同步及原有release/acquire链均保留。
- b256预读按128 CTA × 8 waves、每token4waves分配，每lane12 packed BF16 words/数组。
  全8rank ISA审计确认同步前60条静态dword loads（48条masked local + 12条shared），同步后48条masked remote。
  Combine VGPR94/SGPR50/LDS64B/private0，VGPR/SGPR spill均0。
- b8192也采用最终FP32求和语义，但没有启用b256专用预读；kernel是 `ep_combine_intranode_shared_f32`。
  不传shared时继续 `ep_combine_intranode`。cache schema为 `v12-combine-fp32-local-prefetch`。
- `cab83c22a` 改4文件：`flydsl_dispatch_combine_intranode_kernel.py`、对应op、`mega_moe_m3.py`、`mega_moe_stage2.py`。
  新旧输出relL2约0.0024来自用户要求的舍入变化；不可要求与旧两次舍入结果bitexact。

### S1 / Combine 已完成的配对验证

所有下述性能均为无profiler、同进程同输入、8rank lockstep、HIP external event + graph，取每轮max-rank mean。
S1为6个balanced paired samples；下述S2/Combine实验为12个，每sample200个窗口，≥8秒轮转warmup。
不能把不同run绝对时间相减作为单项收益。

- b256 S1-only：`.scratch/dep8_b256_s1_fused_20260910_v1/paired/result.json`：
  无shared S1 **221.631 µs**；旧S1 + 独立shared L13路径 **282.977 µs**；融合L13 S1 **223.605 µs**。
  融合相对无shared只加 **1.974 µs**，相对独立路径快20.981%；窗口不含S2/L2。
  b256 shared L13为48tiles，每head周期 `[264]*4+[260]*4`。
- Combine改动配对：`.scratch/dep8_b256_combine_prefetch_20260910_v2/paired/result.json`，窗口 **S2+L2+Combine**：
  旧两次舍入 **160.383 µs**；只改最终FP32和 **159.903 µs**；再加本地/shared预读 **158.567 µs**。
  最终/旧快 **1.132%**（12/12；95%CI快0.956%–1.291%），预读相对只改FP32额外快0.836%。
  `paired/isa_audit.json` 是8rank指令顺序/资源核对。
- 上述目录 `b8192_regression/result.json` 已通过8192数值、256graph、负大输入、零、换路由、全local、全remote、恢复输入、
  shared-disabled检查；这是正确性回归，没有重新测b8k性能。

### 用户最新要求：用新 Combine 重测三项（已完成，18:03 UTC）

有效结果 `.scratch/dep8_b256_s2_new_combine_20260910_v2/paired/result.json`，`passed=true`、进程退出0：
A=S2+Combine；B=S2+separate L2+Combine；C=默认tail S2-merge-L2+Combine。
三项共用当前S1、同一输入和640CTA；B/C用当前FP32+预读Combine，A用当前shared-disabled Combine。
separate L2指**从相同S2 GEMM提取出的独立launch**，不是native AITER linear。

| b256，新Combine，同轮配对 | S2至Combine窗口 µs |
|---|---:|
| S2 + Combine | **143.437** |
| S2 + separate L2 + Combine | **172.463** |
| S2-merge-L2 + Combine（默认tail） | **158.828** |

merge相对separate节省 **13.636 µs / 7.906%**，12/12更快，paired bootstrap 95%CI快 **7.769%–8.067%**。
merge比无L2多 **15.391 µs**，separate多29.027µs；这是完整窗口差，不是单独测出的P2P隐藏时间。
三项跨pair CV分别0.378%/0.324%/0.225%，spread均<1.15%。
initial、negative_large、zero、changed_routes、all_local、all_remote、restored的数值通过；
B/C完整输出逐位一致，三项routed FP32和一致，最终输出各自满足对应oracle。所有输入切换256graph、L13/L2 epoch、
source/doc pins、独占8GPU、全部8rank dispatch审计通过。A/C各5kernel，B各6kernel。
每rank trace为同目录 `result.json.arm{0,1,2}.rank{0..7}.trace.json`。

v1数值、graph与计时完成后，在最后dispatch审计失败：脚本要求`_qg5_`，但无shared符号以`_qg5.kd`结束。
已修为同时接受下划线/符号结尾，kernel未变。**v1整轮时间作废，不引用。** v2保留全部检查并重跑。

仅供历史对照，Combine优化前有效旧表：`.scratch/dep8_b256_s2_combine_20260910_v1/paired/result.json`：
**143.577 / 173.102 / 160.292 µs**；merge/separate快7.400%，merge比无L2多16.714µs。
不要把旧A/B与新C拼成一张对照表。

### S2 调度、CTA/资源与提前调度结论

- **默认没有固定的L2专用CTA池。** 同一个persistent CTA先遍历全部8个routed队列，再遍历8个shared队列；
  快CTA做shared时，慢CTA可能仍做routed。没有全grid等待，但也没有从前段开始交错shared。
  b8192 local_reduce_xcd_local不同：routed只处理home，shared遍历全部8队列。
- b256 shared共 **384 tiles**，每队列48个valid任务；每个CTA对每shared队列恰好一个失败claim。
  默认每head周期 **688=48+640**，8个64B间距int64 head。默认失败claims总数8×640=5120。
  调度变化必须保持周期/graph replay正确，不能只改任务先后而漏掉尾部claim规则。
- 增CTA实验 `.scratch/dep8_b256_s2_more_cta_20260910_v1/paired/result.json`（新Combine）：
  640/960/1280 CTA分别 **163.072 / 162.652 / 169.439 µs**。
  960仅快0.258%（9/12），1280慢3.905%（12/12）；未改默认。
  三档S2资源相同：**VGPR147、SGPR106、LDS24896B、VGPR spill0、private0、SGPR lane spill9**。
  9个标量保存到同一个VGPR的不同lane，不走scratch；不能据此断言所有资源/通信瓶颈已排除。
- HIP对实际code object查询（全8rank）确认默认及下面两种提前版本均为 **最多3 CTA/CU =12 waves/CU**。
  每卡256 CU、640CTA平均总量2.5CTA/CU；3是资源允许的驻留上限，不是counter实测achieved occupancy。
- 提前调度 `.scratch/dep8_b256_s2_early_shared_20260910_v2/paired/result.json`：

  | b256调度，640CTA | S2+L2+Combine µs | 相对默认 |
  |---|---:|---|
  | 默认routed-first/tail | 159.708 | 基线 |
  | 1/8（80）CTA shared-first | 162.000 | 慢1.435%，12/12 |
  | 1/4（160）CTA shared-first | 159.533 | 快0.109%，95%CI跨0，无明确收益 |

  提前CTA按`((block_id//8)%8)<1或2`分组，避免低3位把early集中在单一XCD；先shared后routed，其余顺序相反。
  每CTA两类8队列各遍历一次，shared周期仍688。VGPR/SGPR/LDS/驻留不变，SGPR lane spills为11、private仍0。
  全local/remote等7组数值探针、256graph、source/docs、独占8GPU、全rank5kernel审计通过。**未采纳提前调度。**
  v1在编译前因occupancy辅助脚本的MLIR反斜杠解析问题主动停止；v2修复后有效，勿恢复v1。
- 提前调度任务插桩：`.scratch/dep8_b256_s2_early_overlap_20260910_v1/capture/result.json`，3样本×8rank×3方案。
  默认shared首任务约在记录区间 **73.47%** 开始，routed结束时平均仅 **25.81%** shared完成；
  1/8/1/4前置版本shared从约0%开始，分别 **99.66% / 100%** shared在routed结束前完成。
  1/4全部384tiles均提前完成，但routed结束点后移，正常计时没有净收益；这是观察，不等于证明具体HBM/P2P竞争根因。
  不再简单归因“shared没有与routed重叠”；实际P2P传输/远端可见窗口仍没有单独测出。
  插桩时钟为每rank `s_memrealtime`，不跨GPU对齐；epilogue结束不代表remote write已可见，记录区间不含最终空队列探测。

### trace、图和重启入口

- 提前调度8rank图：`.scratch/dep8_b256_s2_early_overlap_20260910_v1/capture/early_overlap_rank0-7.png`（另有SVG）。
  图每个panel独立归一化，只看阶段关系，性能看正常HIP配对结果。
- 内部tile记录：同目录 `tasks_arm{0,1,2}_rank{0..7}_sample{0,1,2}.npz`，
  `overlap_summary.json`；分析脚本在上一层 `analyze.py`。
- 每个正常/诊断实验均有全部8rank kernel trace：`result.json.arm{0,1,2}.rank{0..7}.trace.json`。
  当前新Combine三项对照也要求A/C完整5kernel、B完整6kernel审计。
- 早先b256整层breakdown：`.scratch/dep8_b256_breakdown_shared_20260910_v1/capture/rank{0..7}/`，
  每rank `trace_kernel_trace.csv` 和 `trace_results.json`；图在该实验 `plots/`。
  这是当时b256仍外置shared的历史版本，不代表当前已融合L13/L2/新Combine。
- 早先overall比较 `.scratch/dep8_current_full_20260910_v2/`：b256 baseline575.863/Mega466.536µs（当时外置shared），
  b8192 baseline9000.458/Mega4518.906µs（当时shared已融合）。不能称当前cab提交的最新整体性能。
- 运行模板：`M3_JOB=1816 LOCAL_REDUCE_OUT=<全新目录> LOCAL_REDUCE_BENCH_ENTRY=<实验bench.py> VLLM_ROCM_MXFP8_PTPC_AITER=0`
  加 `M3_ENROOT_CONFIG=/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/bench/experiments/dep8_standalone_shared_20260910/enroot.conf`，
  调用 `bash megamoe-s1-1k-prod/bench/experiments/dep8_preplan_tile_publish_20260910/uniform/run.sh --tokens 256`。
  脚本默认timeout1800，torchrun8rank，64CPU，fresh-output/source/graph/dispatch/GPU独占门禁。
- 本次授权文档更新会使旧实验的文档hash失配，属于预期。**不覆写旧实验源码/结果/manifest**。
  后续新实验应审查本次授权变更后生成新的文档保护快照；不要删除保护gate。生产源码pins必须继续严格通过。
- 重启先读本节、`tool_measure.md`，查HEAD/status/Slurm和本轮最终result。下一步若继续分析，问题是
  “shared提前后为何routed变长、如何避免资源竞争或额外队列开销”，尚无被证明的单一根因；不要自动恢复旧front或批量扫描。

## 2026-09-10 历史重启入口：保留 shared L2 在 S2 尾部，combine-front 已停止

用户最后决定：“算了，还是切回 L2 在 S2里的版本”，随后明确授权 commit 和更新文档。
**当前采用 S1 shared L13 + S2-tail shared L2 + 原 combine 加回 shared；不继续 combine-front。**
本节优先于以下全部历史记录，旧节“shared L2 尚未接入”已经过时；不要自动续跑实验。

### 代码和执行状态

- inner repo：`/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`。
  当前提交 **`999ace3bbe9c86d1602ae87fb25856e9460b309d`**，父提交 `209cc3e6a480a2da320b9ebfe9c61f866e23d818`。
  提交仅4个生产Python文件：S2 kernel、MegaMoEM3入口、combine kernel、combine op。
- outer repo `megamoe-s1-1k-prod` 保存本 handoff、实验脚本及
  `bench/checkpoints/dep8_s2_shared_20260910/`；其提交号用 `git log -1` 查看。均为本地提交，未 push。
- combine-front 从未接入主路径，用户决定保留 S2-tail 后无需撤销生产源码；已核对4文件哈希与成功测试一致。
- 后台 worker 2779026 于 2026-09-10 16:38:48 UTC 停止，已确认无本任务活跃 worker/GPU step。
  原目录 `.scratch/dep8_s2_shared_20260910/front_fix_background_20260910/` 保留；
  `worker_status.json` 是 `stopped_by_user`，`return_to_s2_tail.json` 记录核对结果。
  不要启动该目录 runner.py，也不要把没有最终 result.json 当成全部实验失败：paired_v1..v4 各自结果已落盘。
- allocation1816、do-mi350x-03、8GPU，记录到期 2026-09-10 23:30:19 UTC。
  重启先查实时状态；不要取消该allocation或无关旧steps，不用SSH绕过Slurm。GPU实验仍须串行。

### 保留实现与数值语义

- 固定 EP8、8192 tokens/rank、H6144/I3072、128 experts、top4、BF16 P2P，shared TP1、系数1。
  S1配置沿用上一节记录；S2 M64/N128/K128、4waves/256threads、persist_cu240、qg5，共1200CTA，XCD band16。
  完整配置见本checkpoint `config.json`，来源 `early2048`；不是给任意形状启用 shared。
- `MegaMoEM3` 接受 `shared_w2/shared_w2_scale`，要求同时传入 shared L13 权重及XCD scheduling。
  `_shared_l2` slots0..6：A2、A2scale、W2、W2scale、expert-id、BF16 shared_out、shared queue heads。
- S2 kernel 后缀 `_sharedl2_tail_xcd1`。CTA先耗尽其routed工作，再进入shared队列；
  XCD-local routed只处理home，shared可遍历/偷取全部8队列。没有等待所有CTA完成routed的全grid barrier。
  shared每rank6144tiles，独立8个64B间距head，每head每次调用增长1968=768valid+1200exhausted；
  用周期取模支持graph重放，不能随意改变失败claim次数。shared无P2P publication/reduction tickets，保留CTA内LDS barrier。
- S1产出的shared激活是FP8/E8M0。S2 GEMM内部MFMA累加是FP32，shared epilogue转BF16，
  经LDS重排后16B向量store写本地 `_shared_out`。**当前保留版确实有BF16 shared HBM中间输出。**
- `combine_no_stage1(shared_input=True)` 保留原routed归约到BF16的舍入，再与BF16 shared相加并转BF16。
  这与废弃front的“shared FP32 accumulator直接按rank0..7加routed partial、最后只转一次BF16”不同；
  两者初始输出relL2差约0.002872是有意的舍入差异。不要把front的FP32 oracle当成tail的bitexact要求。
- shared权重不传时保持关闭。未改生产selector默认几何，也未采纳early-start扫描或front优化。

### 已有验证与正常计时：必须区分口径和对照

所有路径以下均为无profiler正常计时；原始文件位于 `.scratch/dep8_s2_shared_20260910/`。
本次提交前再次检查 `git diff --check`、4文件Python语法、成功run源码哈希；没有重复运行已通过的GPU测试。

1. `paired_v1/result.json`：tail和独立L2在8rank逐位一致；initial、negative_large、zero、changed_routes、restored，
   shared独立reference、128graph replay与queue检查通过。
   S2-only A=routed 1643.681µs，B=routed+独立L2 1879.474µs，C=tail 1873.017µs；C/B快0.344%。
   此轮完整forward B4505.014µs→C4534.121µs，tail慢0.646%。不宣称L2已完全隐藏或tail快于独立L2整层。
2. `combine_front_paired_v1/result.json`：同次12对，**S2+全部L2/combine窗口** tail2287.261µs→front2395.177µs，
   front慢4.718%，95%CI慢4.430%–5.007%；独立完整forward tail4553.203µs→front4655.639µs，慢2.250%。
   tail/独立BF16结果一致，front通过它自己的FP32 oracle和128graph验证；source/dispatch audit通过。
3. 后台 `front_fix_background_20260910/paired_v1..v4/result.json` 只比较未修复front A与候选front B，**不是与tail比**。
   v1 shift+row offset窗口慢0.887%；v2 packed dword+lane shuffle窗口无显著变化；
   v3仅 `eid//16` 改shift：窗口快0.487%，CI快0.323%–0.660%，12/12；完整forward无显著变化。
   v4 packed streaming在停止前已完成：窗口快0.508%但CI跨0，完整forward快0.699%，CI快0.481%–0.895%。
   v4未完成额外敏感性复测/ATT，未采纳。v5 pointer preload未测。四个已完成变体均通过各自全部正确性gate。
   后台harness用event-free graph，不能拿其绝对时间和前一harness跨run相减，也不能链乘speedup来声称胜过tail。
4. earlier-start扫描 `early_scan_v2` 有约1.1% S2收益，重叠仍有限；未合入。
   用户已指出P2P通信与load会竞争，不要默认恢复该优化路线。

### 根因/ATT：保存证据，后续不要误读

- tail本身诊断见 `rootcause_s2.json`：shared大多开始得晚，重叠窗口窄；共享routed寄存器预算，
  tail3CTA/CU、独立L2 4CTA/CU，资源影响没有被单独消融。
- **以下仅针对未修复front，不能归到保留的tail：**
  `front_att_se16_cu4_v1` 与 `front_att_se0_cu4_v1` 各20waves完全stitched，
  各有96tile-waves；`decoder_reconciliation.json` 核对5874PC的hit/stall一致。
  rank/slotdecode+地址计算占已解码issue约38.48%/40.26%；这不是kernel wall占比。
- front每lane/tile的256条2B partial load真实存在；4个ID源码标量load已被编译成dwordx4。
  rrecv是本地接收buffer，absent-rank OOB尝试不产生对应实际数据读取；不要说4亿次远端DRAM transaction。
  旧专用combine真正归约是4B读取，16B是跳过的copy阶段，不能声称8倍指令差。
- front所有驻留wave同时位于首次ready段的时间占CUspan约32.31%/6.53%；两份是独立capture。
  first post-acquire context pointer wait（PC11548）首次约68.8k/66.0k cycles，后续约710/681cycles。
  cache invalidation/acquire burst是未消融的假设，不是已经证实根因。
- front VGPR128、SGPR106、68 SGPR lane spills、private0、LDS33344B、4CTA/CU；
  SGPR spill写v127 lanes，无scratch/HBM spill，不是资源溢出，暴露stall占比极小。
- 解析用 `analyze_front_att.py`；旧dense `analyze_s2_bubble.py` 曾OOM，不要重跑大矩阵。
  host分析Python `/tmp/s2-att-analysis-venv/bin/python`；host python3无numpy，GPU/container Python环境另行处理。
  PMC取 `selected_pmc.json`；原counter CSV混入warmup，必须按formal trace Dispatch_Id join。
  profile eager时间不等于正常graph计时；局部CUspan/wavecycle不能外推全卡或跨GPU拼时钟。

### 重启入口与文件保护

先读本节和workspace `tool_measure.md`，核对inner HEAD、git status、Slurm/GPU，再决定用户新的任务。
用户当前只要求保存版本和交接，**没有授权重启后自动继续优化或恢复后台**。
实验脚本已版本化于 `bench/experiments/dep8_s2_shared_20260910/`；
`prepare.py`/`bench.py` 是tail与独立L2控制，`prepare_combine_front.py`/`bench_combine_front.py`为已放弃front实验。
已有raw control在 `.scratch/dep8_s2_shared_20260910/control/`，checkpoint也保留恢复副本。
旧源码pins、历史结果、旧checkpoints不可覆写。外层checkpoint保存压缩结果、配置、诊断、hash清单和handoff。

本次文档修改已获用户明确授权。旧 `protected_documents.json` 对本次被授权更新的文档会失配，属于预期；
后续若重新跑测试，必须先审查授权变更并生成新运行所用的保护基线，**不要删除文档保护gate或覆写旧证据来绕过**。
旧S1 `prepare_restart.py` 固定上一版本commit/hash，不能直接作为当前S2版本的启动器。

## 2026-09-10 历史交接：S1 shared L13（已由上方 S2-tail 入口取代）

本轮用户明确授权“先commit一版，然后更新文档，我要重启了”。后续文档修改仍需明确授权。
当前代码：`/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，
提交 **`209cc3e6a480a2da320b9ebfe9c61f866e23d818`**；父提交 `d50c44f3e77ba327cc81a3676e632ee82a9e7cd5` 是完全没有 shared 的版本。
外层脚本/交接提交用 `git -C megamoe-s1-1k-prod log -1` 查看；均为本地提交，未 push。
本节优先于下面历史记录；此前重做前的 full-fusion/S2 实验不可当作当前实现。

### 当前目标、范围和下一步

- 用户目标：8k 下 shared 开销能被隐藏，S1 时间达到完全无 shared 的水平。**目标尚未达到。**
- vLLM 命名：L13 是 gate/up projection，位于 S1；L2 是 down projection，属于 S2。
  **当前只融合 S1 shared L13 + SwiGLU + MXFP8 输出；shared L2/S2、shared 最终加回输出尚未接入。**
  不要把传入 shared 权重后的普通完整 forward 当作已经实现完整 shared expert。
- 当前保留 XCD shared 队列、descriptor wave-uniform 修复、仅 shared 省掉 system acquire；dcu 保持 96。
- 下一重点：**routed 和 shared 共用的首批 A/scale staging**。优先考虑分段预取、上游产出 packed scale，
  再考虑减小 scale LDS 占用以容纳更多 CTA。以上新 staging 改法尚未实现或测量。
  每 tile A=768 KiB、scale=24 KiB；现有证据指向装载/同步/重排等待链，不能断言 scale 耗尽 HBM 带宽。
- 下一次如做 ATT，应先为当前 `_shu1_shna2` 版本建立新 profile/reference audit；现存 profiler wrappers
  主要对应省 acquire **之前**的 normal reference，直接运行其旧 audit 会发生合理的 ISA/source mismatch。
  每次只验证一项改变，继续用同进程 S1 HIP-event 配对计时和 canonical 数值校验；不要扩大参数扫描。

### 最新有效结果（S1 only，同进程、同输入）

原始结果：`/mnt/shared/homes/ming/msa/.scratch/dep8_s1_shared_a_stage_20260910/acquire_pair_v2/result.json`。

| arm | S1 µs |
|---|---:|
| 无 shared | 2020.535 |
| shared，保留 system acquire | 2374.912 |
| shared，仅 shared 省 system acquire，当前代码 | **2309.867** |

省 acquire 节省 **65.045 µs / 2.739%**，3/3配对更快；当前相对无 shared 仍多 **289.332 µs / 14.320%**。
三 arm 的跨轮 spread 分别0.096%、0.251%、0.631%。`acquire_pair_v1`曾把 uniform_work 也传入 routed wait，
得到约48µs收益；最终v2恢复传入原始work，保持routed等待参数一致，只对shared跳过fence，采用v2结果。

更早的独立配对实验（不要把不同运行的 arm 拼在一起计算 speedup）：

| 实验 | 无 shared µs | 修改前 µs | 修改后 µs | 结论 |
|---|---:|---:|---:|---|
| descriptor uniformity | 2014.082 | 2544.317 | 2368.126 | 省176.190µs，6.925%；当时仍保留shared acquire |
| global→XCD shared queue | 2012.189 | 2550.913 | 2548.966 | 时间基本不变；PMC显示局部性确实改善 |

DCU 实验（uniformity已修，**shared acquire还未去掉**）：

| dcu | 实际payload CTA | 无 shared µs | 有 shared µs | shared相对开销 |
|---|---:|---:|---:|---:|
| 96 | 32 | 2019.896 | 2387.649 | 18.207% |
| 32 | 32 | 2019.859 | 2383.792 | 18.018% |
| 16 | 16 | 2373.504 | 2444.392 | 2.987% |

96→32收益0.16%，在噪声内；16让带shared绝对时间变慢2.38%，比例接近3%主要因为无shared基线被拖慢。
总grid始终256CTA。`dcu=96`仅32CTA真正搬payload，其余名义producer加入计算；32→16才减少搬运并发。

### 冻结输入、几何和验证

- EP8、8192 tokens/rank、H6144/I3072、128 experts（每rank16）、top4；seed42，uniform Gaussian router
  bias0/noise1，真实top4 selected-softmax×2；shared权重seed+104729、每rank完整复制、TP1、系数1。
- 完整配置仍取 `bench/experiments/dep8_preplan_tile_publish_20260910/uniform/variants.json` 的 `early2048`。
  S1 M128/N256/K256、8 waves/512threads、grid_mult1/256CTA、dcu96、work_shards8、XCD band4、wpe2、b_nt0；
  packed_a_scale/unroll_a_pingpong/split_a_lds、pipeweights/asyncA/mfma_amajor/prefetch_B_before_A均开；
  preplan_waves4、chunk2048、tile_ready/earlypublish开。S2未改，实验不运行S2。
- 每次closure依次route→quant/preplan→S1；event仅包S1，route/quant不计入S1事件。
  2次settling + 20个external事件样本组成graph；8秒预热；ABC/BCA/CAB，每arm每轮100样本；max-rank mean。
- 当前所有rank的routed量化值及scale经source-token/topk-slot canonical排序后逐字节相等；shared
  与acquire对照逐字节相等。BF16→SwiGLU→BF16→MXFP8 reference最大relL2=0.001287304。
  128次graph重放、负大输入、零输入、换路由、恢复输入、queue epoch检查全部通过。
- raw routed行顺序受原子调度影响，**不能直接按物理行比较**；按srcmap编码排序。MXFP8 scale需unswizzle。
- 无shared及旧shared对照的所有8rank ISA与旧reference指令逐条一致。最终源码与有效运行source pin一致。

### 资源、调度和必须保留的细节

| 版本 | VGPR | SGPR | LDS | spill/scratch | 驻留 |
|---|---:|---:|---:|---:|---|
| 无shared | 242 | 71 | 90112 B | 0 | 1 CTA/CU，8 waves/CU，25% |
| 旧XCD shared，未修uniformity | 252 | 87 | 同上 | 0 | 同上 |
| 当前shared | 238 | 105 | 同上 | 0 | 同一静态资源约束 |

实际HIP code object residency在uniformity实验中查询过全部24份二进制；最终acquire版本静态资源完全相同。
MI350X/gfx950：256CUs、32waves/CU、160KiB LDS/CU、8XCD。两CTA需要176KiB LDS，超过160KiB。
不要使用rocprof kernel trace里不准确的VGPR/SGPR字段代替实际ISA/HIP查询。

- shared采用独立8个XCD-homed head、64B间距，和routed复用band4 decoder及stealing结构。
  每CTA在完成自己的producer工作后先走shared队列，再走routed；不等其他CTA全体完成shared。
  8192下shared1536tile（64M×24N），rank0 routed6336tile（264M×24N）。
- `_shared_l13` int64指针表slots0..6：W13、W13scale、A2、A2scale、M-row-base、expert-id、queue-heads。
  `_shared_task_count[::8]`为8个head，每head每调用增长448=192valid+256exhausted；global旧队列period1792。
  epoch循环无需reset kernel；不能任意更改exhaust计数规则。工作总grid是256，不随dcu减少。
- shared与routed共用一个GEMM体、相同MNK。LDS广播task和选中地址显式readfirstlane保持wave-uniform，
  防止每次buffer load出现descriptor waterfall。当前kernel名末尾`_sharedl13_shxcd1_shu1_shna2`。
- `_byte_tensor`必须用`fx.Float8E4M3FN`，不能换成Int8。shared output scale resource界限必须保持
  `8192*96+8192`；epilogue非writer使用1GiB sentinel，若给无界4GiB descriptor会写越界。
- shared先把gate/up结果round到BF16，SwiGLU后再round到BF16；routed保持既有FP32 intermediate。
- shared A/scale来自同stream前一quant，无并发payload writer，因此跳过system acquire，仍保留CTA barrier；
  routed的payload wait、system acquire完全保留。不能把shared的优化直接应用于routed。
- 本实验guard限制full8192、固定S1 geometry，不能假设任意token数/selector可用；共享权重默认未传即关闭。
  `shared_xcd_schedule=True`仅在shared开启时生效；全局队列flag False仅作为历史控制。

### ATT / PMC 结论及边界

根因工具为Claude先前的exposed-bubble方法，S1版`analyze_s1_bubble.py`。MFMA延长32cycles来自实测
`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_VALU_MFMA_F8=32`。IMMED/category9含真实wait/nop，
统计指令只合并`s_barrier` continuation，不能一律删category9。

- 修uniformity前后：每tile/wave K-body实际指令 **6829→3049**，readfirstlane **1264→88**，
  branches **305→11**，MFMA仍768；额外294个execnz waterfall分支消失（旧分支全not-taken）。
- 一处SE16/CU4 consumer样本：无shared / 修前 / 修后routed K-body约81.9k / 91.8k / 68.8k cycles；
  routed tile间隔29.2k / 47.3k / 52.1k cycles。bubble约26.40% / 20.17% / 29.06%。
  指令减少后bubble占比可能上升，同时kernel更快；不要把bubble比例直接当墙钟收益。
- **上述ATT/PMC都是省shared acquire之前的版本**。最后一次分析已按shared/routed拆开旧trace：
  SE16/CU4为7shared+25routed，A DMA stall每tile/wave分别9.48k/7.31k cycles，scale wait8.59k/6.84k；
  两类都有等待。聚合热点不是shared独有。PC覆盖重叠，不能把各项stall相加当kernel时间，也不能外推整卡。
- PMC（每arm单次rank0采样，其余7rank同时工作）：uniformity前后L2 hit73.62%→74.93%，
  DRAM读5.138→4.997GB，TCP读/returning-atomic latency330.0→396.6cycles，MFMA util31.25%→36.24%。
  更早global→XCD DRAM读5.735→5.138GB，约-10.4%，说明XCD局部性有效；不能称4倍流量改善。
  shared加入后MFMA工作量+24.24%；GMI写183590400B保持一致。以上均是整个S1的流量，不是隔离的shared B流量。
- `buffer_inv sc0 sc1`使CU缓存及L2非一致性缓存行失效，属于system acquire；不是清空所有L2。
  依据：[AMD CDNA4 ISA表52](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf#page=94)。
  profile eager duration不等于无profiler的graph计时，不混用。
- producer/consumer取决于CTA admission ticket，不能只按物理CU编号判断。旧CU4 producer样本未见shared拖慢dispatch。

### 归档与重启命令

版本化归档：`megamoe-s1-1k-prod/bench/checkpoints/dep8_s1_shared_20260910/`，含完整压缩result、
诊断JSON、最终rank0 ISA、两份control源码、哈希清单及本handoff。原始大体积ATT/IR继续保留在workspace `.scratch`：
`dep8_s1_shared_rootcause_20260910`、`dep8_s1_shared_uniform_20260910`、`dep8_s1_shared_dcu_20260910`、
`dep8_s1_shared_a_stage_20260910`。最后有效运行是后者`acquire_pair_v2`。

脚本在 `bench/experiments/dep8_kernel_shared_20260910/`：
`run.sh`默认已改为最新`s1_shared_acquire_pair.py`；`s1_uniform_pair.py`/`s1_dcu_pair.py`供历史对照。
历史full-fusion的`perf.py`、`profile_shared.py`、`reorder_pair.py`等仍可能在未跟踪目录里，**不要用作当前入口**。
`bench/standalone_m3.py`是AMD原MiniMaxM3MLP.forward+TP1 native MXFP8 adapter，未使用PTPC；standalone已实现完整shared MLP，
其历史整段MoE结果与本次S1-only结果不可直接比较。enroot仅覆盖AMD activation/native MXFP8两个文件。

重启先检查Slurm/GPU。交接时所有benchmark、ATT和CPU分析均已退出，无待轮询任务。
预约1816，node`do-mi350x-03`，8GPU，截止2026-09-10 23:30:19 UTC；保留预约，不取消他人job或旧step。
如果Slurm credential故障复发，用户已要求轮询至恢复；不要用SSH绕过，约30秒轮询并汇报。

从`/mnt/shared/homes/ming/msa`执行以下命令。OUT必须全新；预约已结束则换有效job。

```bash
squeue -j 1816 -o '%i %T %N %L'
S1_RESTART_OUT=/mnt/shared/homes/ming/msa/.scratch/dep8_s1_after_restart_v1
python3 megamoe-s1-1k-prod/bench/experiments/dep8_kernel_shared_20260910/prepare_restart.py --out "$S1_RESTART_OUT"
M3_JOB=1816 LOCAL_REDUCE_OUT="$S1_RESTART_OUT" \
  bash megamoe-s1-1k-prod/bench/experiments/dep8_kernel_shared_20260910/run.sh --tokens 8192
```

`prepare_restart.py`核对提交和runtime source hash，必要时从归档恢复缺失的旧control文件，生成新source pin；
不会覆盖已有control或结果。不再依赖control主机`/tmp/pin_shared_run.py`。
它是当前checkpoint复现工具：后续修改代码做实验应另建对应源码pin，不能覆盖旧结果或关闭source gate。
在enroot内执行的脚本必须放共享文件系统，不能只放control主机`/tmp`。cache off、rank独立IR dump保留。
CPU bubble分析用node上的小CPU step，避免在仅约8GB内存的control主机运行大数组分析；GPU实验串行。

## 历史记录：2026-09-10 topk/quant 预规划与 payload 按 tile 提前发布

用户本次明确授权“commit 一版，更新文档，要重启了”。本节优先于下方历史记录；后续文档修改仍需用户授权。
代码位于 `megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，提交 **`d50c44f3e77ba327cc81a3676e632ee82a9e7cd5`**。
外层交接资料提交可用 `git -C megamoe-s1-1k-prod log -1` 查询。两仓库分别提交，未 push。

### 当前结果与比较口径

EP8、每卡8192 tokens、H6144/I3072、128 experts、top4；同进程、同输入，**两边都包含 topk**。
Standalone 指 all-gather + standalone MoE + reduce-scatter，不是旧版 MegaMoE。

| 路由输入 | Standalone µs | 新 MegaMoE µs | 同轮加速比 |
|---|---:|---:|---:|
| 均匀 Gaussian，bias=0，最新恢复状态 | 7999.564 | **3845.306** | **2.080345×** |
| 8热点、每卡一个、bias=+1.0 | 7999.659 | **4634.449** | **1.726130×** |

强偏输入热点占路由31.3927%，expert行数1411–10391，单source→expert最大1381行；卡间总负载仍接近。
因此 standalone 在这些输入下约8000µs，不代表对热点集中到少数卡的情况也无感。

另一次同进程强偏实现对照：原chunk512约4905µs；early512=4761.495µs；原chunk2048=4962.784µs；
early2048=4657.655µs。early2048相对其配对的原chunk512快 **5.0349%**，3/3获胜。
这组4658µs与standalone对照组4634µs来自不同运行；不要拼接成一个paired sample。

均匀与偏斜对照只改router bias，实际topk由输入logits计算，selected-softmax权重乘2；不是手工分配top4。
均匀输入的logits哈希和expert行数与之前均匀分布实验一致。默认下一次复现先用bias=0。

### 提交包含什么，哪些默认仍关闭

- `routing.py`：可选top4直方图；`MegaMoEM3.route(scores, weights, ids)` 为预规划入口。
- `preplan.py` / `quant.py`：topk完成直方图，quant中的planner CTA做epoch/reset、count交换、接收布局；
  8k外部grouping也移入quant，并保留LR route mask写入。S1消费已有计划，不重复做这些工作。
- 跨rank LAUNCH_READY代际握手仍保留；remote PLAN_READY和payload/tile-ready依赖没有删除。
  host guard要求route→quant/forward使用相同tensor与stream，不能重复累计未消费的直方图。
- `payload_tile_publish_early=True`：chunk仍负责producer任务划分；每搬完一个目标M-tile与本chunk的交集，
  所有wave完成store并同步，随后system-release并增加该tile的ready计数。跨source/chunk的预期计数保持原算法。
  M取实际配置/接收端发布值，不硬编码64或128。当前8k的M=128。
- `preplan_waves`默认0，`payload_tile_publish_early`默认false；**生产selector没有自动切到本次实验配置**。
  已有 `joint_work_flags` 实验开关一并保存，默认false；它不是本轮收益来源，不推荐自动开启。
  P1/P2a/P2b/early-B沿用此前已启用的实验配置，`skip_launch_barrier=false`。

### 推荐的已测8k配置

完整字段以 `bench/checkpoints/dep8_preplan_tile_publish_20260910/selected_8k_config.json` 为准。

- S1：M128/N256/K256，8 waves，grid_mult1，dispatch96，work_shards8，XCD band4，resource=true，b_nt0，wpe2。
- external_grouping/counting=true，`preplan_waves=4`，`payload_chunk_rows=2048`，
  `payload_tile_ready=true`，**`payload_tile_publish_early=true`**；packed A/unroll/split LDS开启。
- S2：M64/N128/K128，persist240，skew96，queue_grid_mult5，XCD band16。
- local_reduce=true、local_reduce_xcd_local=true，p2p_quant=none。
- 开启preplan后必须在同stream先调用 `op.route(scores, weights, ids)`，再 `op(x, weights, ids)`；
  若走forward_prequant则先完成对应quantize，不能只改配置而沿用未配对的预量化输入。

此前256的预规划选择为4 waves：同轮旧MegaMoE377.311→371.694µs，降低1.489%；
轻偏+0.25 standalone482.849→365.780µs，1.320×。这些数据早于8k支持扩展与early-publication，
不能当作最终提交版本新测的256结果。256保留M64/N512/K256、dispatch32、LR关闭、chunk0、tile-ready关闭；
若接下来回到256，先复核这套完整配置，不套用8k的chunk2048。

### 对之前解释的纠正与剩余问题

`_configure_payload_geometry` 用的是 **max(ceil(max_source_count/chunk_rows), 4)**，不是上限4。
活跃payload CTA数为min(该值, dispatch_blocks/npes)。之前“循环槽位32→16”是错误解释：
轻偏+0.25下384与512均有64个槽位/组合、4个活跃CTA；全八卡有效搬运任务从1087降至1024。
384→512的4272.712→3999.976µs实测有效，不能归因于循环次数减半。
热点第二段确实排在别的expert第一段之后；本次在大chunk内部提前发布ready改善了等待。
尚无本次early-publication的ATT分段，不能精确声称273µs或5.03%全部来自某一种stall。

P1/P2a/P2b/early-B已在上一提交82aa85b3e；量化前移的256 ATT在
`.scratch/dep8_b256_preplan_20260910/att_w4_v2`，但没有同窗口新版/旧版ATT配对，不能据此报ATT加速比。
下一步如继续优化：在当前输入和配置上逐项验证同步开销、任务次序或更强卡间偏斜；不要未经请求开大范围扫描。

### 证据、验证与重启命令

归档：`bench/checkpoints/dep8_preplan_tile_publish_20260910/`；脚本：
`bench/experiments/dep8_preplan_tile_publish_20260910/{screen,standalone,uniform}/`。
原始结果、逐rank trace和ISA仍在 `.scratch/dep8_b8192_tile_publish_20260910/`；三份有效result原字节gzip归档。

- 有效运行：`strong_v2`、`standalone_strong_v2`、`standalone_uniform_v1`。
- 8秒预热、K16前置、N100计时、3轮AB/BA快速对照、max-rank HIP event；不是12轮正式最优性证明。
- 数值reference、源/配置/dispatch检查、graph128与输入扰动、LR mask/counter、tile expected/ready和srcmap通过。
  实现对照还做了16次交替输入、逐卡20ms延迟测试。关闭新功能时S1指令与先前512版本一致。
- 新旧S1均242 VGPR、90112B LDS、无spill；新SGPR71，旧69。standalone与实现对照中的候选ISA一致。
- `strong_v1`是编译作用域错误；`standalone_strong_v1`是旧配置断言拦截，均无有效性能数据。
- 旧protected_documents哈希是实验当时的保护基线。本次用户授权更新后，归档的新audit使用
  `document_hashes.json`；不要改写历史audit或关闭source gate。后续再授权改文档时另建新的保护基线。

重启后先查1816与GPU占用；本次全部GPU测试已退出，没有待轮询的benchmark。
1816/node03，8GPU、192CPU，结束时间2026-09-10 23:30:19 UTC；保留预约，不取消其他人的job或旧step。
从workspace根目录复现均匀输入（OUT必须全新；预约失效则先换有效job）：

```bash
M3_JOB=1816 LOCAL_REDUCE_OUT=/mnt/shared/homes/ming/msa/.scratch/dep8_uniform_after_restart_v1 \
  bash megamoe-s1-1k-prod/bench/experiments/dep8_preplan_tile_publish_20260910/uniform/run.sh \
  --tokens 8192 --candidate early2048 --router-hot-bias 0 --router-noise-std 1
```

强偏复现改用 `standalone/run.sh`，加 `--router-hot-experts 8 --router-hot-bias 1 --router-hot-placement spread`。
用同目录 `audit.py <输出目录>` 审计，保留完整源码/配置/输入核对；运行路径仍依赖本workspace已有wrapper、container与scratch参考证据。
不要用不含topk的历史3856µs与当前完整路径直接作归因比较。


## 2026-09-10 重启入口：下一步解决 EP8-256 加速比不及预期

本次实验/配置/计时审计已提交：outer `1cf23f6aeef245cf69fbea8a594b2da26fc38aa6`；inner aiter仍为
`b5746d8b6731e46170ccf52e56e9b73af4780578`，源码clean。未push。
当前任务停在可恢复点；本次只保存与交接，没有启动新GPU测量。
后续优先256，8k band扫描保持暂停。以下旧章节是历史记录，以本节及最新结果为准。

### 恢复时先读

1. workspace `/mnt/shared/homes/ming/msa/tool_measure.md`：计时窗口与trace方法规范。
2. `bench/checkpoints/dep8_b256_s1_quick_20260910/REPORT.md`：5个候选、每点3组的结果。
3. `bench/checkpoints/dep8_b256_s1_quick_20260910/best_s1_candidate.json`：冻结S2条件下最快候选的完整配置。
4. `bench/ep8_best_configs.json` 的256项 `s1_fixed_s2_quick_screen`：统一更新位置。
   主完整配置仍为EP4继承种子；不能将它与条件初筛记录混为同一个配置。

### 当前数据与用户约束

- EP8、每rank256 tokens、总128 experts、topk4、rank0–7，每token最多4份partial/copy。
- **固定S2，先调S1；每点3组paired A/B，只看完整MoE总时间，不采新breakdown、不自动跑12组。**
- S1最快候选为完整EP4-512 S1：M64/N512/K256、dcu32、band4、XCD on、resource off、b_nt0。
  固定S2：M32/N128/K256、persist128、qg5、band8、skew0、XCD on、LR与XCD-local均true。
- 本轮旧MegaMoE对照429.979µs，候选397.503µs，paired耗时降低7.553%，3/3更快。
  这证明当前候选集的条件改进，尚不构成完整配置全局最优。
- **Standalone最近独立测得494.985µs**，来自方法审计b256_v2的K16协议；同协议旧MegaMoE440.022µs。
  本轮S1扫描未包含standalone。不能把494.985/397.503当成同轮已验证的新加速比。
  历史490.034/454.264（1.078742×）也属于另一轮、另一计时窗口。

### 下一步执行顺序与待验证假设

1. 从已归档完整最快候选出发，用相同输入、K16前置、100 measured replay及max-rank HIP event，
   补standalone与候选同进程3组paired A/B，确认当前真实加速比；继续保留旧MegaMoE作S1对照。
   先复用已有正确性、graph、source/dispatch与隔离门禁，再计时，不能仅换数字或跳过门禁。
2. 在固定S2/LR条件下，围绕合法S1配置小范围逐项比较dispatch CU、band及调度相关选项；
   参数必须有EP4表、已测点或源码约束依据，不随意设置。仅dcu160→32已快4.17%，
   支持先排查dispatch预留和consumer计算资源分配，但不能由总时间单独证明通信未掩盖。
3. N256和BM32此前未成功测量：small-XCD guard只支持M64/N512/K256，quick_v1在编译时失败，
   quick_v2才有有效结果。若探索几何变化，先分析合法调度路径，不能删guard硬跑。
4. EP4→EP8每expert平均行数翻倍只提供计算侧候选，不能整表机械平移。
   老256实际各rank tile数为264/288/252/264/288/276/276/300，不是简单192；
   consumer CTA还受dcu影响。S1比standalone GEMM1长本身不能证明overlap失败，S1包含dispatch与等待。
   rank2长Combine可能是在等其他rank（其S1更短），不要仅凭单段长度认定其为瓶颈。
5. 只有后续明确转向S2/LR时才解除本轮冻结条件；当前不要混入S2或LR开关变化。

### 复现、图与资源

- 实验：`bench/experiments/dep8_b256_sweep_20260910/`；有效raw：
  `/mnt/shared/homes/ming/msa/.scratch/dep8_b256_sweep_20260910/quick_v2/`。
- `variants.json`的来源哈希对应实验前的EP8表；后来只追加结果元数据也会改变live表哈希。
  重启后新实验应显式生成新spec/source pins，或使用归档的`initial_ep8_best_configs.json`复现；
  不要关闭hash gate，也不要直接覆盖旧实验/输出。
- 新256图与rank0–7 trace：`bench/checkpoints/dep8_measurement_audit_20260909/gpu_validation/trace_b256_v1/`，
  图`timeline_arms_b256.png`；CSV `plot_data/dispatches_rank0-7.csv.gz`。
  按settling与region区分前置/计时区间，不能混合K0/K8/K16；不使用首replay代表稳态。
  用户/Claude添加的图保持原样，未并入本次实验提交。
- 计时审计与256初筛的SHA256清单已逐文件核对，Python/JSON/shell语法检查完成。
  原始trace、日志、CSV和已固定哈希的历史副本按原字节保存（含工具产生的空白）。
- 预约1816：do-mi350x-03，192 CPU、8 GPU，独占，结束时间2026-09-10 23:30:19 UTC。
  本次确认allocation仍RUNNING并保留。重启后先查预约和实际GPU进程；Slurm可能保留旧step记录，
  不应仅凭step条目判定有测量运行。不要取消parent job，也不要动Claude的1814/node04。

## 2026-09-10 最新：EP8-256只扫S1，S2固定，每点3组配对

用户明确要求快速扫、每点3组A/B、不跑12组、不采新breakdown，并固定S2。
已完成5个有效候选，全部正确性/graph128/source/dispatch/隔离/噪声门禁通过。
完整MoE max-rank HIP-event总时间：EP4-512 S1 397.503µs（配对旧429.979，快7.553%）；
EP4-256 S1 400.308µs（快6.921%）；只改dcu32/64/96分别412.522/414.330/422.425µs。
各点3/3更快。K16前置+100 measured replay，8秒全候选轮转预热；仅快速初筛。
S2固定M32/N128/K256，persist128、qg5、band8、XCD+local-reduce及XCD-local。
初筛最快S1：M64/N512/K256，dcu32，band4，XCD on，resource off，b_nt0（完整抄EP4-512 S1）。
N256实际触发small-XCD只支持M64/N512/K256的编译guard；BM32也受同一guard限制，
两个几何候选延后，未修改production guard。quick_v1无计时结果；quick_v2为有效运行。
报告与原始结果：`bench/checkpoints/dep8_b256_s1_quick_20260910/`。
脚本：`bench/experiments/dep8_b256_sweep_20260910/`；raw `.scratch/dep8_b256_sweep_20260910/quick_v2/`。
`bench/ep8_best_configs.json`的256 entry新增`s1_fixed_s2_quick_screen`记录；
主完整配置未替换，因为其原S2/LR与本轮冻结条件不同，不能据此宣称全配置更优。
全部GPU工作结束，保留allocation1816。8k band sweep未继续。


## 2026-09-10 完成：DEP8 计时与 breakdown 方法审计

旧计时max-rank/paired算术与旧完整CSV重新解析通过。但GPU窗口敏感性补测表明：
K0相对K16的B/A比值，256 +0.0477%（95% CI -0.0884%至+0.1742%），
8192 +0.8975%（+0.5342%至+1.2762%）。两档均未建立预设±0.1%计时等价。
K是前置、不计时的replay次数；没有修改GEMM维度或S2 band。
新的K0/8/16三marker trace全部八rank通过dispatch/shape/queue/no-memcpy检查；
K8与K16的kernel-sum都通过±0.5%等价，逐kernel±1%并非全部建立。
旧图591/543µs是跨rank recorded span，不能标kernel sum；首replay不能充当稳态代表。
完整报告：`bench/checkpoints/dep8_measurement_audit_20260909/REPORT.md`。
规范：workspace `tool_measure.md`；审计目录有相同快照。
raw与新绘图CSV归档在审计目录 `gpu_validation/`，保留rank0–7、前置和测量区间。
复现脚本：`bench/experiments/dep8_method_sensitivity_20260909/`，全部GPU工作已正常结束。
Slurm credential曾失败后恢复；一次MORI端口TIME_WAIT失败重试成功，无失败样本混入。
原始scratch：`.scratch/dep8_method_sensitivity_20260909/`；生产inner仍为b5746d8且clean。
快速band扫描继续暂停：8/12/24/32仅初筛，4/48/64无有效结果，未更新最佳表。
保留独占allocation1816/node03/192CPU/8GPU，到2026-09-10 23:30:19 UTC。

## 2026-09-09 当前 allocation 与 EP8 快速 band 扫描

EP8 参数表已提交：outer `7a32cb8`，完整复制 EP4 参数，尚未更新为 EP8 最优。
用户新申请的独占 allocation：1816，do-mi350x-03，192 CPU、8 GPU，24小时，
2026-09-09 23:30:19 UTC 开始，2026-09-10 23:30:19 UTC 到期。保持预约，不随测量结束释放。
旧 allocation1814/node04留给原有任务；本轮不再与 Claude 共享 GPU。
用户要求仅快速初筛一次：每点8秒预热、2组AB/BA、20 replay、graph40，
以新表8k全参数（S1 band4 / S2 M64N128K128 band16）为对照，只改S2 band。
候选32/8/24/12/4/48/64，包含band16对照共8个值；未请求正式复测，不自动推广最优。
入口 `bench/experiments/dep8_s2_band_sweep_20260909/quick.py` / `quick_run.sh`；
串行 driver `sweep_quick.py`；raw `.scratch/dep8_s2_band_sweep_20260909/quick_band*_v1/`。
当前S2公式不要求band为8倍数：n_block=8*j+queue；要求N_tiles整除8以及XCD-local下band>1。
完整理论与121组CPU映射枚举在该experiment目录的 `THEORY.md`、`mapping_check.json`。

## 2026-09-09 最新：EP8 配置改从 EP4 最佳配置表完整继承

用户要求先完整复制 EP4 参数，再继续 EP8 调参。
新文件：`megamoe-s1-1k-prod/bench/ep8_best_configs.json`（在仓库内为 `bench/ep8_best_configs.json`）。
这是后续 EP8 调参基准和最佳已测配置的统一记录位置。六档 256/512/1024/2048/4096/8192
均逐字段复制当前 `bench/full_best_configs.json` 的完整 S1/S2、实现选择、p2p_quant 和 local-reduce 开关，
并与 `s1_best_configs.json`、`s2_best_configs.json` 交叉核对。原始来源与 SHA256 已记录。
初始状态全部为 `inherited_from_ep4_pending_ep8_validation`，EP8 性能为 null；不能宣称已是 EP8 最优。

8k：S1 M128/N256/K256、band4、work_shards8、XCD on、packed_a_scale/unroll_a_pingpong/split_a_lds on；
S2 M64/N128/K128、band16、persist240、skew96、queue_grid_mult5、XCD on；两个 local-reduce 开关均 true。
256/512 的 local-reduce 开关也按 EP4 原样复制为 false；没有统一强行改成 true。
后续 EP8 实验必须显式使用这个文件的完整配置及开关，不能再落回生产 selector 的 band8 默认值。
更新各档时记录 EP8 同条件测量证据；此前 DEP8 256/8k 归档仍对应旧默认配置，不是此新表的性能。
原 band8-vs32 实验已在 GPU 空闲门禁处停止，无正式测量；其脚本已标注 superseded。
当前只完成配置复制，未启动新的 GPU 测试、未更改生产 selector。

## 2026-09-09 提交：EP8 local reduce 与八 rank 测量归档

源码已提交到 inner aiter `s1-unified-production`：`b5746d8b6731e46170ccf52e56e9b73af4780578`。
支持 EP8/topk4，每 token 最多 4 份 partial/copy；源码与通过门禁的 256/8192 测量哈希一致。
Outer checkpoint 已提交：`981c92dca02000bf3a631e8e1ee4cea7d224d1c4`。
本 outer checkpoint 提交包含 DEP8 实验脚本、baseline/local-reduce 结果、rank 0–7 全量 trace、
绘图 CSV/PNG/SVG、完整压缩包与 handoff。下面“6 个源码修改未提交”是采集时的历史状态。
测量报告：`bench/checkpoints/dep8_breakdown_20260909/REPORT.md`。
提交对应关系：`bench/checkpoints/dep8_revision_20260909.json`；未执行 push。

## 2026-09-09 最新：DEP8 256/8192 breakdown，rank 0–7 全部保留

按用户最终要求保存八个 rank 的 trace。两档均开启 local_reduce 和 XCD-local，EP8/topk4，
每 token 仍最多 4 份 partial/copy。使用同一份已验证的 EP8 生产补丁，未新增生产源码修改。
无 profiler 全路径：256 baseline 490.033933 → MegaMoE 454.264381 µs，paired 1.078742×；
8192 baseline 8086.866061 → MegaMoE 4318.792280 µs，paired 1.872501×；均 12/12 更快。
256 使用新的 LR-on A/B；8192 复用已完成的有效 v3 unprofiled，trace 逐项匹配 source/config/routes。
两档 rocprof 均 60 秒预热、12 balanced pairs、20 replays/slice，全八 rank 数学、graph400、
BF16 oracle、poison 槽、counter、dispatch 顺序/grid、无额外 memcpy 和噪声门禁通过。
MegaMoE trace 四段 Quant/S1/S2/Combine 的八 rank 均值（µs）：
256 = 7.188970 / 281.901892 / 109.712625 / 45.898270；
8192 = 26.754 / 2557.400 / 1428.639 / 293.368。
这些是 profiled kernel duration，不能把其和当作无 profiler elapsed critical path。

完整报告与绘图数据：`megamoe-s1-1k-prod/bench/checkpoints/dep8_breakdown_20260909/REPORT.md`。
该目录的 `b256` / `b8192` 均保留 rank0–7 原始 rocprof CSV/JSON（gzip 无损）、
全部 dispatch 整数时间戳 CSV、stage sample/summary CSV、每 rank Chrome trace、PNG/SVG 时间线。
根目录合并两档 stage CSV，`experiment/` 为复现解析与绘图脚本，`production_sources/`、
`production.patch`、`isa/`（在各档子目录）保存来源；完整编译 IR 留在对应 `.scratch` raw run。
所有测量进程正常退出，8 GPU 利用率和 VRAM 占用均为 0，仅零占用 gpuagent 常驻。
Inner 仍有 6 个 EP8 源码修改未提交。本节覆盖之前只保留四 rank 及旧 no-LR 256 的当前测量口径。

## 2026-09-09 最新：local-reduce支持EP8，8k全路径验证完成

用户要求local-reduce扩展EP8，并明确topk仍为4、每token最多4份copy。
`megamoe-s1-1k-prod/aiter`在`9a588ac8370b8bdb65feb9d991155fa59ccb1ea4`上有6个未提交源码修改。
EP8仍用4个topk partial槽：同rank归并后写该rank首个原topk槽，combine按rank顺序只读参与rank。
3GiB staging按source rank重定位为384MiB窗口，避免signed32偏移溢出；EP4原路径保留。
入口/quant/S2/combine均支持EP4或EP8/topk4；local_reduce默认仍opt-in。

DEP8每卡8192 tokens(全局65536)，local_reduce及XCD-local均开启：
standalone 8086.866061 → MegaMoE 4318.792280 µs，
paired加速1.872501×，耗时降低46.5955%，12/12更快，B/A CI[0.532734462,0.535545892]。
同进程60秒预热、12 balanced pairs、20 replay/sample、max-rank统计，全部门禁通过。
7组输入/路由切换(含rank4–7、单rank4专家、双rank乱序槽)、graph400、poison槽检查、
逐位rank-grouped BF16 oracle、全source route mask和counter reset全部通过。
另24个production helper测试通过：device-scope、EP4回归、EP8 compact槽、>2GiB地址。
原8k no-local-reduce run按用户新方向停止，未产生有效正式计时；local v1/v2编译失败，v3有效。
实际baseline12kernel，MegaMoE4kernel，S1 128/256/256 band1 dispatch96 ws4；
S2 64/256/256 band8 XCD-local qg5，非EP8调参最优结论；Seed42 synthetic非生产router。
报告 `megamoe-s1-1k-prod/bench/checkpoints/dep8_local_reduce_b8192_20260909/REPORT.md`；
入口 `bench/experiments/dep8_local_reduce_20260909/`，raw `.scratch/dep8_local_reduce_b8192_20260909_v3/`。
GPU进程已正常退出。下面关于内层clean/EP4-only local-reduce的旧状态已被本节覆盖。

## 2026-09-09 最新追加：DEP8 每卡256，latest production vs standalone

已读 tool_measure.md，完成8卡同进程A/B：每卡256 tokens、全局2048，16 experts/rank。
工作区 `megamoe-s1-1k-prod`；inner仍为 `9a588ac8370b8bdb65feb9d991155fa59ccb1ea4`，无生产源码修改。
使用最新生产 **EP8默认selector**，EP4专用preset不触发；不是EP8最优调参结果。
S1 M64/N512/K256、band1、dispatch160、8queues、XCD off；S2 M32/N128/K256、persist128，local-reduce off。
Baseline为无额外输出复制的正常AITER standalone + vLLM all-gather/reduce-scatter。
完整MoE baseline 505.649288 → MegaMoE 429.451372 µs，
paired耗时降低 15.0694%，加速 1.177432×，12/12更快；
B/A CI [0.848721040, 0.849859012]。60秒预热、12 balanced pairs、299replays/sample、max-rank统计。
八rank数学、输入/路由/hot-expert切换、graph400、epoch/source/dispatch/单stream/独占/噪声门禁全部通过。
实际baseline11kernel、MegaMoE4kernel；spread A0.5507%、B0.6575%。
Seed42 synthetic，未验证真实router。GPU实验成功退出。
报告 `megamoe-s1-1k-prod/bench/checkpoints/dep8_baseline_b256_20260909/REPORT.md`；
入口 `bench/experiments/dep8_baseline_20260909/`，raw `.scratch/dep8_baseline_b256_20260909_v1/`。
本轮用户已授权用全部8卡；旧的分卡并发约定不适用于本轮DEP8实验。

## 2026-09-09 最新：production S1 统一完成，256/512/1k 全部验证

活动工作区 `megamoe-s1-1k-prod`，outer base `57194f7` / inner base `4b6c4f3ef`。
内层已提交 `9a588ac8370b8bdb65feb9d991155fa59ccb1ea4`，branch `s1-unified-production`。
三个改动文件：`mega_moe_stage1.py` 合入原 small-band expert 轮换队列映射；
`mega_moe_m3.py` 接上 exact EP4 256/512/1024 S1 preset；`mega_moe_config.py` 更新注释。
small S1 用 M64/N512/K256、XCD、band 2/4/1、work stealing；共享 GEMM 主体。
先前固定 N-padding 调度已被替代。六档当前配置表均指向同一 production 路径和哈希，冻结文件仅用于历史对照。
S2 默认未随此次改动；测量时两臂固定相同的已记录 S2/local-reduce 配置。
外层代码/配置/证据提交：`8ea2b847da64c03575a94c00ec2a4b0f666934cf`，
branch `s1-unified-production`。内外两层工作区均 clean。

用户允许分卡并发：本 agent GPU 0–3，另一 agent GPU 4–7。
按 `tool_measure.md` 同进程、同输入、60秒 warmup、12组 balanced A/B，比较 max-rank 相对时间。
三档都已测完，进程全部退出。完整 MoE 结果：

| tokens/rank | Old frozen µs | Production µs | Paired change | 95% paired CI | Production faster |
|---:|---:|---:|---:|---|---:|
| 256 | 497.768131 | 498.902804 | +0.2280% | [+0.1175%, +0.3357%] | 2/12 |
| 512 | 636.696048 | 637.212228 | +0.0811% | [-0.0212%, +0.1920%] | 6/12 |
| 1024 | 849.391472 | 849.000225 | -0.0461% | [-0.2063%, +0.1161%] | 7/12 |

三档四 rank 的 S1 ISA 指令及资源用量逐项完全一致；数学、逐位、调度、graph400、
路由/input/hot-expert 切换、source/dispatch/单stream/噪声门禁全部通过。
效率基本一致，没有新增 S1 指令开销；不宣称墙钟时间严格相等。

完整归档 `megamoe-s1-1k-prod/bench/checkpoints/s1_unified_small_20260909/REPORT.md`；
raw `.scratch/s1_unified_small_20260909_b{256,512,1024}_v2/`。
当前入口 `bench/experiments/s1_unified_small_20260909/`，空闲检查仅前四卡。
256 v1 按当时全节点独占规则排除；有效数据全用 v2。

上轮固定 N-padding 的 band2/4 均比 band1 慢，归档在
`megamoe-s1-1k-prod/bench/checkpoints/s1_1k_npad_band_20260909/REPORT.md`。
下文 N-padding 和旧 checkpoint 是历史记录，不覆盖本节最新状态。

## 2026-09-09 新方向：1k 统一 production S1，N 调度 padding

用户要求回到 clean commit，保留1k M64/N512/K256，不再维护独立 small-band S1，
将真实12个 N tile 的调度空间补到16。已完成实现和完整同进程对照。
独立工作区 `megamoe-s1-1k-prod`：outer base `57194f7`，inner base `4b6c4f3ef`；
内层有未提交 padding 改动，见 `aiter/aiter/ops/flydsl/kernels/mega_moe_m3/mega_moe_stage1.py`。
只padding调度索引，不扩tensor；dummy N slot跳过并继续取队列。统一production支持
band1/per-expert readiness，保留8队列work stealing。没有新增S1副本。
1k其余S1/S2参数不变。历史最佳848.239228→production padding861.808683µs，
慢13.569455µs/1.5998%，0/12更快，ratioCI[1.015145459,1.016841778]。
数学/逐位/输入路由切换/graph400/poison/source/dispatch/单stream/噪声门禁全通过。
GPU schedule audit确认每个有效tile恰执行一次，观测到work stealing；audit不纳入计时。
报告与完整归档：`megamoe-s1-1k-prod/bench/checkpoints/s1_1k_npad_20260909/REPORT.md`。
入口：`megamoe-s1-1k-prod/bench/experiments/s1_1k_npad_20260909/`；
raw：`.scratch/s1_1k_npad_20260909_v1/`。本轮GPU实验已退出，selector/最佳配置表未推广。
此前N256方案及ATT保留在同工作区的 `s1_1k_production_xcd_20260909` 目录；
用户切换方向后停止剩余counter流程，后续不自动恢复8k combine任务。


更新：2026-09-09。本文件只保留当前可用状态、有效证据和未解决问题。
路径未特别说明时相对 workspace `/mnt/shared/homes/ming/msa`。
版本化副本位于 `megamoe-local-reduce/HANDOFF.md`；workspace 根目录
`handoff_megamoe.md` 是重启入口。旧轮次的报告不再决定当前状态。

## 当前 checkpoint

- 原 checkpoint 已完成；之后按用户要求完成 8k breakdown，并继续 combine 就绪消费实验。
- 内层 `megamoe-local-reduce/aiter`，branch `s2-local-reduce`：
  **`4b6c4f3ef787cf05a2937402b37d91cd44850872`**。
  在已有 local-reduce / work_scratch 修复上，新增 adaptive dispatch 发布等待修复。
- 外层 `megamoe-local-reduce`，branch `s2-local-reduce`：包含本文件的提交。
  保存配置表、冻结实验实现、测量脚本/结果和文档。outer 忽略 `aiter/`，必须分别恢复。
- 六档 standalone、8k breakdown、ready-queue grid sweep 和老版 combine grid 对照均已完成；后续实验状态见下方当前任务，恢复时检查实际进程。
- 生产 selector 没有推广全部实验最优参数。不能把默认 selector 当作当前配置表。

主文档：`megamoe-local-reduce/bench/CURRENT_MOE_RESULTS.md`。
原始证据仍在 `.scratch/`；关键脚本、JSON、原始 paired samples 已版本化到
`megamoe-local-reduce/bench/checkpoints/{standalone_best_20260909,tuning_20260909}/`。

## 2026-09-09 combine 实验与当前用户约束

用户明确要求 **一个 stream，S2 后启动 combiner，取消入口全 rank 完成等待，
从 mapping 取已就绪 tile**。不要恢复双 stream 路线。先跑通并测完整路径，
容量压缩、消费后回收等优化暂缓；清零可融合到 quant/S1。

此前实验已保存在 `bench/checkpoints/combine_band_publish_20260909/`：

| candidate | 完整路径相对同进程当前最佳 |
|---|---:|
| v1 acqrel band publication + 原 combine | +20.19% |
| v2 轻量 band publication + 原 combine | +0.17%，约 +7.18 µs |
| 双 stream bucket scalar consumer | +116.24% |
| 双 stream balanced scalar consumer | +41.53% |
| 双 stream balanced vector consumer | +22.27% |

全部正式比较通过逐位一致、路由/输入切换、poison、连续400次graph等门禁。
最后 vector 版本有4096个正向提前完成观测；不是“从未实际提前消费”。
另有1352次每 replay 改输入/路由的 stress，检查点输出全部逐位一致。
没有 PMC/ATT 证据，不能把消费端慢的比例归因到某一类 stall。
实验性删除 S2 claim-loop barrier 导致 fault，未纳入有效性能结论。

当前实现目录：`bench/experiments/combine_single_stream_queue_20260909/`。
S2仍用轻量 flag；combiner 中每 CTA 一个 scout wave、三个 worker wave；
scout 聚合完整输出 tile 依赖，atomicAdd 预留队列槽并发布 group ID；
worker 跳过未发布槽，唯一消费。quant 融合本地 mapping/tail 清零。
保留末尾跨 rank 完成同步保护复用，所有 GPU launch 在单 stream。
初版容量8×512个int64槽，不在此阶段调容量。详细实现见该目录 README。
单 stream 32 CTA：4213.09 → 5092.51 µs，+20.87%；96 CTA：4230.47 → 4451.37 µs，+5.22%。
均12对0对更快，全部正确性/graph/单stream/噪声门禁通过，均有提前输出观测。
结果和源码快照已归档 `single_stream_queue_b{32,96}/`；报告 `SINGLE_STREAM_QUEUE.md`。
用户指出 CU/grid 问题：四卡实机均256 CU。原 combine 实际128 CTA×8 wave；
新96 CTA×4 wave只有288个reduction wave（另96 scout）。HIP精确二进制查询
资源上限8 CTA/CU=32 wave/CU，新96 CTA的全卡wave容量上限仅4.6875%。
队列256 CTA：4216.64 → 4390.43 µs，+4.12%；512 CTA：4231.53 → 4468.92 µs，+5.61%。
同样全部门禁通过、0/12更快，原始数据和源码已归档 `single_stream_queue_b{256,512}/`。
用户继续指出老版128 grid来源：是上游默认，EP4无geometry调参表；此前只调S1/S2遗漏combine grid。
老版全grid barrier的wrapper保守上限是256 CU，并不要求停128。老版128→256对照已完成：
4227.264 → 4202.333 µs，快0.590%（24.93 µs），12/12更快，CI ratio [0.993019,0.995319]，全部门禁通过。
源码 `bench/experiments/combine_bulk_grid_20260909/`，归档 `bulk_grid_b256/`；后续比较应纳入
`combine_block_num=256, combine_warp_num_per_block=8`。生产配置和历史冻结结果尚未改写。
队列实验没有更快版本，不要推广。恢复时检查实际进程。
生产 kernel、原最佳配置未修改，未推广 candidate。


## 2026-09-09 mapping 排查、修复与当前任务（进行中）

最新用户认为队列仍比此前最好慢很多，要求继续优化；当前目标是追回相对bulk256的差距。
当前最新已验证队列实现为 `bench/experiments/combine_queue_skip_done_20260909/`：
预取+monotonic清零+worker寄存器位图跳过已消费槽位，256CTA×4wave，1scout+3worker。
跳过槽位独立A/B：4376.112→4353.282µs，快0.5217%/22.83µs，12/12更快，
ratioCI[0.993957,0.995736]，全部门禁通过；71VGPR、69SGPR、零scratch。
归档 `bench/checkpoints/combine_band_publish_20260909/queue_skip_done/` 完整。

预取版新ATT(CU0/CU4)已完成：payload采样stall44.12%/58.99%，mapping21.43%/16.93%。
两次mapping检查336/330，均消费9groups；百分比不是walltime。
归档 `bench/checkpoints/combine_queue_prefetch_overhead_20260909/`。
直接诊断counts：空检查3,867,271，其中已消费槽位896,658(23.19%)。
另加started marker的稀疏scan：77,785次，24.15%看到其他worker所属已发布但尚未标记开始的slot；
99.50%的当前live-empty slot还未被tail预留。诊断本身扰动时序，不推导可偷取时间比例。
归档 `bench/checkpoints/combine_queue_mapping_20260909/{counts,scan}/`，报告已写。

同进程bulk256 vs最新skip-done队列已完成：4192.945→4367.752µs，慢174.807µs/4.1688%，0/12更快。
全部门禁通过，归档 `bench/checkpoints/combine_band_publish_20260909/queue_skip_done_vs_bulk256/`。
代码 `bench/experiments/combine_queue_best_compare_20260909/`，
输出 `.scratch/combine_queue_best_compare_20260909_v1/`；当前可引用上述直接对照差距。
融合分桶已完成：4386.206→4324.717µs，快61.489µs/1.4021%，12/12更快，全部门禁通过。
归档 `bench/checkpoints/combine_band_publish_20260909/queue_fused_bucket/`；原始 `.scratch/combine_queue_fused_bucket_20260909_v1/`。
`bench/experiments/combine_queue_fused_bucket_20260909/`。
A=skip-done队列，B=quant读取IDs时atomic append到bucket order，取消独立bucketkernel。
live counts一次初始化，已有finalizer每轮清32项；验证histogram在CPU按当前路由算，避免计时内额外copy。
同stream保证下一轮quant前清零完成；不能在quant自身一边清一边atomic写。
B consumer仍同一归约算法，但bucket内token顺序可变；保留FP32 source顺序。
B计时路径5kernel vsA6，reset全部纳入；新gate检查livecounts清零、token守恒。
全套数学/逐位/graph400/epoch/dispatch gates已通过，生产kernel未改。
4wave vs8wave已完成（保持256CTA、1scout，workers3→7）：4354.051→4350.210µs，
ratio0.999118，CI[0.997758,1.000451]跨1，7/12更快；无可靠收益，保持4wave。
完整归档 `bench/checkpoints/combine_band_publish_20260909/queue_workers8/`。
代码 `bench/experiments/combine_queue_workers8_20260909/`，
输出 `.scratch/combine_queue_workers8_20260909_v1/`，GPU实验已结束。
双方都是融合分桶5kernel路径；B audit_stats按8wave分配，worker索引/launch512均调整。
prefix候选已完成：4364.218→4347.302µs，快16.916µs/0.3877%，11/12更快，
CI[0.994966,0.997327]，全部门禁通过；归档 `bench/checkpoints/combine_band_publish_20260909/queue_prefix_deps/`。
原始 `.scratch/combine_queue_prefix_deps_20260909_v1/`，代码同名实验目录。
双方用融合分桶4wave队列；B把scout逐group读token/topk/计算expert区间/wave_or，改成
按bucket从每source的expert_end构造完整prefix bitmask。key=max(localIDs)保证依赖包含于prefix。
仍逐bit检查全部必要band，不假设S2按prefix顺序完成；可能多等一些，性能须实测。
该改动目的是减少scout启动/描述符构建开销，没有新增kernel。
当前正在跑最新prefix融合队列vsbulk256直接对照：
代码 `bench/experiments/combine_queue_prefix_vs_bulk_20260909/`，
输出 `.scratch/combine_queue_prefix_vs_bulk_20260909_v1/`，exec session24989。
另外动态head领取已写，尚未运行：`bench/experiments/combine_queue_dynamic_claim_20260909/`。
A=已测prefix静态分槽；B=每个worker空闲时对tails后8个int32 head做atomicAdd取下一slot，
等待该slot发布后消费并再次领取。没有retiredmask；每slot只有一个消费者。
quant的tails清零8→16，新增32B counters、不加kernel；B head最终应=G+workers_perq。
B遇到预留未发布hole会等该slot，不能声称没有head阻塞；需全套门禁和同进程计时。

最新用户问kernel数量：已答当前融合队列5个(quant含分桶/reset、S1、S2、consumer、finalizer)，
新增小kernel只剩finalizer；老bulk4个同步在combine内。继续原优化任务，不以该问题取消任务。

## 2026-09-09 partial 下一 stripe 预取

用户选第1项，已完成独立A/B。双方均256 CTA×4 wave，1 scout+3 worker；
A为已测monotonic-clear队列，B只增加下一stripe partial预取。
整层4401.568→4347.480 µs，快1.229%（54.09 µs），12/12更快，
ratio CI [0.986409,0.988962]，所有正确性/graph/单stream/噪声门禁通过。
ISA确认跨stripe预取，VGPR58→71，SGPR62→67，无scratch，LDS512B。
精确HSACO的HIP资源上限8→7 CTA/CU；不是实测occupancy，实际grid仍256CTA。
没有与bulk256重新同进程对照，不宣称超过bulk。没有新ATT归因。
实现 `bench/experiments/combine_queue_prefetch_20260909/`；完整归档
`bench/checkpoints/combine_band_publish_20260909/queue_prefetch1/REPORT.md`。
有效原始数据 `.scratch/combine_queue_prefetch_20260909_v3/`；v1为Slurm临时凭据
错误，v2为DSL变量名编译错误，日志/失败源码保留，无有效计时。
生产kernel不变；本轮GPU实验已结束。后续队列优化可用此版本为对照。

## 2026-09-09 队列开销定位

用户要求继续定位队列版开销。已完成 queue256 的一组 PMC、CU0/SE0 与CU4/SE0
两组ATT，均在完整单stream EP4路径第11次调用采集；每组12次输出/epoch/队列检查通过。
每组ATT完整重建4个wave。主要采样stall：partial load/use 52.08%–61.52%，
mapping load/use 18.82%–27.45%。**这是所采wave的stall分布，不是kernel总耗时占比。**
两个CU分别360/364次mapping检查，仅9/6个消费任务；空检查97.5%/98.35%。
空检查包含数据未就绪等待，尚未证明有可被其它worker领取的闲置ready任务。
独立bucket每次trace约77 µs：32个单wave CTA各扫完整8192路由。

精确对照只把消费后mapping清零从system-release改为coherent monotonic atomic store，
保留post-store wait、入队release、S2 flag及finalizer。完整路径4404.864→4374.927 µs，
快0.680%（29.94 µs），12/12更快，CI ratio [0.992244,0.994210]；全部正确性、
输入/路由切换、poison、400 replay、实际dispatch/单stream/噪声门禁通过。
对应 `bench/experiments/combine_queue_clear_ablation_20260909/`；原始对照是queue256，
不能把这个结果当作对bulk256的直接比较。生产代码未修改。

报告和原始PMC/ATT/源码：`bench/checkpoints/combine_queue_overhead_20260909/REPORT.md`。
clear对照归档：`bench/checkpoints/combine_band_publish_20260909/queue_clear_mono/`。
后续优先：payload读取延迟隐藏；减少空槽流量并测静态归属是否造成额外等待；
改善bucket。带宽饱和与最优worker/scout比例仍未被这些证据确定。无残留GPU实验。

## 已修复的错误

**8k dispatch 错误已定位并修复。** grouping 使用 producer-slot 连续前缀，payload CTA
按目的 rank 交错分配；倾斜时前缀外 producer 会跳过 PAIR_ORDER_READY 等待，消费
未完成的路由表，导致重复/丢失路由。PLAN_READY 不能代替路由表发布。
修复把 wait+acquire 移出 group_active 条件，使所有 producer 都等待。
位置：内层 `aiter/ops/flydsl/kernels/mega_moe_m3/dispatch.py:emit_dispatch_group`。
固定源码 SHA256：`23d0c624f153ee605ab963abbc1293fcf9a2b90eab73048da60a66df344150ee`。

验证：8k/4k完整正确性与graph门禁、8k三类路由共192次conservation检查通过；
旧源码负对照仍失败。入口 `bench/check_skew_dispatch_ep4.py`，证据
`.scratch/b8192-bugfix-v1/diagnosis.json`。不要再将其描述为未定位、BF16舍入或S2 band2错误。
此前 S1 work_scratch 读后 barrier 修复继续保留；两者是不同的同步问题。

**本次 standalone baseline 没有两次多余输出复制。** 冻结入口直接使用 fused_moe 和
reduce_scatterv 返回值；trace已验证。旧 `.scratch/s2-local-reduce/standalone-total-2k-8k-v1/`
及旧通用 `bench_ab_dep4.make_baseline` 仍含多余copy，不能用于当前baseline性能结论。
本次入口只从旧helper导入输入/setup/MegaMoE构造，不调用它的make_baseline。

## 当前配置：已测阶段最佳，完整组合已验证

适用：MI350X/gfx950，EP4，H6144/I3072，128 experts/top4，
MXFP8权重/激活、BF16输出，精确 `tokens_per_rank=MTPR`。

完整参数及证据：`megamoe-local-reduce/bench/{s1_best_configs,s2_best_configs,full_best_configs}.json`。
逐batch baseline身份和实际kernel：`bench/standalone_baseline.json`及
`bench/STANDALONE_BASELINE.md`（相对outer）。Baseline ID为`standalone_ep4_copy_free_20260909`，
使用正常AITER selector的heuristic fallback；不是clean MegaMoE，也未宣称穷举调优过baseline。
JSON记录每档完整GEMM符号、tile、selector日志、四rank ISA/source哈希、原始结果和候选配置引用。

| tokens/rank | S1 M/N/K | S1 band | S2 M/N/K | S2 band | S2 local reduce |
|---:|---|---:|---|---:|---|
| 256 | 64/512/256 | 2 | 64/128/256 | 1 | off |
| 512 | 64/512/256 | 4 | 32/128/256 | 1 | off |
| 1024 | 64/512/256 | 1 | 64/128/128 | 2 | on, XCD-local |
| 2048 | 128/256/256 | 2 | 64/256/256 | 4 | on, XCD-local |
| 4096 | 128/256/256 | 4 | 64/256/256 | 4 | on, XCD-local |
| 8192 | 128/256/256 | 4 | 64/128/128 | 16 | on, XCD-local |

256/512/1k 必须加载 `bench/s1_best_kernels/stage1_small_band.py`，
使用 JSON 中 import_as、module_globals 和 sha256；`STRICT_HOME=False`。
该实现建立有限 M band 和 home=(expert×12+n)%8，可偷任务。
仅给生产 M64 设置 xcd_schedule/band_m 不等价。
旧 stage1_b512.py / stage1_b1024.py 已不选用，只保留历史复现。
2k+使用生产S1；8k保留packed/unroll/split、dispatch96。
256/512不启用local-reduce；1k+启用两个local-reduce开关及S2 qg5。

## 最新完整 MoE 时间

同进程、同输入/权重、CUDA Graph，含全部必要通信、quant、reset及combine；
不含router、shared expert、residual。Seed42 synthetic routing，非生产回放。
每档60秒预热、12 balanced pairs，HIP events取四rank最大值。
时间是sample均值；加速比来自paired log-ratio。

| tokens/rank | standalone µs | 当前 S1+S2 µs | 加速比 |
|---:|---:|---:|---:|
| 256 | 589.58 | 487.68 | 1.209× |
| 512 | 810.43 | 626.04 | 1.295× |
| 1024 | 1185.94 | 833.24 | 1.423× |
| 2048 | 1996.25 | 1394.49 | 1.432× |
| 4096 | 3904.62 | 2320.68 | 1.683× |
| 8192 | 7701.42 | 4170.65 | 1.847× |

六档均12/12更快，最大sample spread1.703%；全rank数学、graph/input-switch、
连续400replay、source/dispatch检查通过。计时不带profiler，trace单独采集。
版本化结果：`bench/checkpoints/standalone_best_20260909/measured/b*/result.json`；
完整日志/IR/trace：`.scratch/standalone-best-sweep-v1/measured/b*/`。

### 8k 最近调参只带来小幅整层收益

4216.826µs是历史8/8 local-reduce整层时间，不是standalone。
4216.826→4170.648跨轮约1.10%，对照实现和replay数不同，不能全算作优化收益。
真正同进程且双方都修复dispatch的8/8→4/16对照：
**4205.126→4192.377µs，快0.303%，95%CI快0.242%–0.368%，12/12。**
证据：`.scratch/b8192-full-best-after-fix-v1/run1/result.json`，版本化副本在
`bench/checkpoints/tuning_20260909/b8192_full_after_fix.json`。
S2 band16独测比8仅快约1.21µs（0.068%）。不能把独测改善直接相加成整层收益。

**“各阶段已测最快”不等于“联合全局最优”。** 8k还缺S1/S2 band=4/8、8/16两个
交叉组合的整层验证。用户最后质疑的正是这个边界；不要再把1.847×说成最近band调参收益。

## 新增：8k 最佳配置 breakdown（2026-09-09）

本轮只测量，未调参。保存于 `bench/checkpoints/breakdown_8192_20260909/`：
`REPORT.md`、可复用 `standalone_breakdown.json`、分rank原始样本和压缩trace。
无profiler整层复测：standalone7715.16µs，MegaMoE4164.11µs，paired1.8528×。
rocprof 单dispatch duration 的四rank平均：MegaMoE quant26.67 / S1 2257.78 /
S2 1633.88 / combine234.13µs；baseline通信kernel合计4009.11µs（52.13%，含等待）。
Kernel耗时之和不当作elapsed critical path。所有原始重叠timestamp保留，按dispatch ID排序；
最初nonoverlap门禁过严，不是数值或dispatch错误。大IR/完整日志另留
`.scratch/b8192-breakdown-{20260909-v1,rocprof-v2}/`；后者为最终CSV采集。

## 调度共识与待解决问题

- S1/S2统一含义和验证口径，参数各自选；不强求相同队列解码或相同band。
- 既有S2普通队列已按N条带归属，同机实测bx%8→物理XCD稳定置换；显式home的
  matched control近乎持平，不能推出局部性无用。历史8k S1在band8/shards8/N_tiles24时
  普通队列打散同B的M；不能推广到band1或小S1的N_tiles12。映射观察不是跨机器保证。
- S1允许偷任务；S2 XCD-local partial staging禁止跨XCD偷任务，不可直接合并此策略。
- band_m是M tile分组宽度，不改变tile M；增大不必然更快，也不能一概规定偶数或2的幂。
  合法范围依实现；已测过band3。理论用于缩小候选，不能代替实测。
- 8k联合交叉组合未测；小batch/其他形状也未穷举联合空间。S2暂按已测配置冻结，
  不宣称无需再调。后续如继续优化，先检查整层组合，避免只看阶段单测。
- 不用历史未校准跨GPU绝对时间戳分摊等待；只有同rank duration和正常max-rank计时可用。
- 真实router、真实checkpoint及其他拓扑未验证；ready-tile consumer实验已回退，不自动恢复。

## 恢复和复现

先读workspace `tool_measure.md`。读取Slurm和OS实际状态，不续跑旧失败driver或覆盖旧目录。
最后核对allocation1814仍RUNNING，节点do-mi350x-04；计划到期2026-09-10 16:02:59 UTC，
这是写入时的状态，恢复须重新确认。EP4用GPU0–3、32CPU，所有8GPU空闲门禁，串行运行。
已知gpuagent PID2053148为0 VRAM/0 CU监控进程，不要因其存在而杀进程。

从workspace运行单点完整对照（替换job和新目录）：

```bash
M3_JOB=<valid-job> \
LOCAL_REDUCE_OUT=/mnt/shared/homes/ming/msa/.scratch/<fresh-directory> \
LOCAL_REDUCE_BENCH_ENTRY=/mnt/shared/homes/ming/msa/megamoe-local-reduce/bench/checkpoints/standalone_best_20260909/bench.py \
bash megamoe-local-reduce/bench/run_local_reduce.sh --tokens 8192
```

入口已包含正确配置加载、数值/graph门禁、60秒预热、12对计时及dispatch审计。
冻结driver含历史job与输出路径，不要不改参数就重启。
Wrapper使用独立overlay/MoRI/FlyDSL，端口29852、900秒超时；不要与其他EP4任务重叠。
共享文件系统结果可能延迟可见，进程退出后先重读，不要直接重复成功测试。

接受规则：MegaMoE纯排程/graph输出逐位一致；standalone原子归约按历史独立数值门槛，
详见当前结果文档。source或dispatch不符、正确性失败、spread>5%均不能引用性能。
保留原始样本和失败诊断；不把replay/rank当独立统计样本，不混eager/graph或traced时间。
