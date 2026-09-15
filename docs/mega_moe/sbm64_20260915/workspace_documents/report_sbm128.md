> **用户最新要求（2026-09-13，后续实验最高优先级）：所有方案都只在目前已测出来最快的方案基础上做实验。**
> 按目标 batch 和测试条件，先核对已有记录中的最快方案、源码和配置；以该方案作为后续修改的起点及性能对照，即使它尚未合入生产。不得用较慢的 clean commit、生产默认配置或旧方案替代当前已测最快方案。测时仍须遵守 `tool_measure.md`。
> **本条覆盖下文历史记录中要求以 clean/default 为基线的指令；历史数字保留其原有对照含义，不代表后续实验基线。**

## 当前工作模式：主助手与 K3 协作（2026-09-14）

用户所说的“老办法”默认指以下流程；若用户本轮明确调整顺序，以本轮指令为准。

1. **先定基线，再讨论。** 主助手先读最新报告、handoff、redo 与目标 batch 的实测记录，只从该条件下目前已测最快的源码/配置出发。主助手与 K3 先讨论一个具体改点、预期收益及寄存器/同步风险；已讨论清楚的方案不重复讨论，每轮优先只改一个变量。
2. **主助手实现并先看 ISA。** 主助手负责修改代码、编译、对比实际分支、指令顺序、wait/barrier、VGPR/SGPR、spill、scratch 和 LDS。若新增寄存器压力或溢出已足以否定方案，先停止该候选，不为走流程继续跑 GPU；不要把“压寄存器”变成增加 spill。具体是否值得实测遵从用户当轮要求。
3. **ISA 可接受后，GPU 与 K3 审核并行。** 主助手负责 GPU 实验和监督；K3 独立审核源码 diff、ISA、正确性/同步风险、适用范围与测试方法。无需等 K3 审完才启动 GPU，但 GPU 实验之间串行，避免相互干扰。K3 默认只读，不与主助手同时修改同一份代码。
4. **发现问题就停本轮、修好重来。** K3 提出问题后，主助手结合源码/ISA核实；若问题成立并使当前实验无效，停止对应实验的明确进程，不取消整个 allocation 或无关任务。修复后使用新的实验目录/输出与 source pins，重新检查 ISA、审核并跑 GPU；运行中不修改已冻结的源码、harness 或受保护文档。
5. **结果由证据决定。** 遵守 `tool_measure.md`，先过正确性、graph/epoch、全 rank dispatch、source/ISA 与硬件稳定性门禁，再看同进程 balanced paired A/B。快速试验只作筛选，不冒充正式收益；不跨进程比较绝对均值、不删除不利样本。小收益、轮次反转或同 kernel 也有差异时，先排查测量偏差，必要时补构造顺序反向对照。
6. **主助手汇总，K3 复查结论。** 两者复核是否保留、默认是否真的命中选中方案、是否影响其它 batch。无收益或退化的候选不混入保留版；不确定就明确说明。继续下一个方向前，简短交代本轮结果和下一改点，不盲目换 M 分支。
7. **每次保留就直接 commit。** 用户已授权每次决定保留后直接提交实际生产代码，不只提交实验快照，不追加提交前检查或再次确认。候选选择阶段的 ISA/GPU 验证照常进行；若寄存器、spill、scratch 等超过当前基线预算，立即停止并等待用户。报告、handoff、默认台账等结果文档仍按用户单独授权更新；保留旧实验原始记录和不确定性，不改写冻结 checkpoint。

K3 指实际 Kimi CLI 的 `inferact-kimi-k3` 模型，不是把另一个普通子助手命名为 K3。当前可续用的讨论 session 为 `session_054a5e40-816d-4745-b49b-072ee8da21af`；主助手传给 K3 的材料应包含目标、源码路径、diff/ISA、基线与结果路径，并明确本轮是方案讨论还是只读审核。

## 2026-09-15 最新：保留 SBM64 M32/M48/M64，b72 不采用

本节覆盖下方历史中“SBM64 仅 b136 实验入口”“仍用 N512 默认”的状态。用户明确授权只保留前面 SBM64 优化、提交实际代码、更新文档并与 K3 最终审核。

**保留范围：b64/104/112/120/128/136/144/152/160/168/176/184/192。** 物理 SBM64，S1 N256/K256，按 valid_rows <=32、33–48、49–64 分别走 M32/M48/M64。限制为 EP8、H6144/I3072、128 experts/top4、fused shared L13+L2/XCD、完整配置 batch、无 local-reduce、无显式 Stage1 环境调参。SBM32 的 b72/80/88/96 和已有 SBM128 默认不属于本轮接入范围。

代码入口：`mega_moe_config.py:SBM64_PATHS` → 默认 selector 的 `sbm64_path=m32_m48_m64, tile_n=256` → `sbm64_m32_m48_m64.mega_moe_stage1`。保留 current A 前读、K128 半组 B retirement、late A-scale，M48 正确向上取整 scale 分组及尾16行 A DMA，以及共用 M64 epilogue 的 last16/middle16 producer+consumer 跳空；barrier 不放进条件分支。**M48 冗余 scale 裁剪、b72 改 SBM64 均丢弃。** middle16 按用户决定保留，其相对 last16 的小幅增量收益仍未明确，不能把累计收益全归给该项。

S2 沿用每点最快对照：b64 为 BM64/N128、NT ON；其余 BM32/N256、NT OFF。整空半块 skip 仅 b64（BM==SBM）与 b152 关闭，其余开启；b120/136/144 的新增启用受 rollout scope 保护。b72 保持 SBM32/N512，不带候选 quant 改名、shared 行表扩展或 S2 skip。

**完整 forward 同轮比较（不是默认/standalone 对照）：** 每点两种真实 graph capture 顺序，各12平衡配对；下表百分比为耗时降低，绝对时间只列 AB 同轮。全部26轮/312对候选更快，每轮 ratio 95% CI 均低于1；仍是 quick screening，没有严格稳定性或任意路由分布保证。

| batch | BA 耗时降低 | AB 耗时降低 | AB 原最快 → 保留版（µs） |
|---:|---:|---:|---:|
| 64 | 4.260% | 4.970% | 265.256 → 252.073 |
| 104 | 4.895% | 4.775% | 271.975 → 258.988 |
| 112 | 4.587% | 3.914% | 272.313 → 261.657 |
| 120 | 4.470% | 4.603% | 277.977 → 265.182 |
| 128 | 4.537% | 4.207% | 278.001 → 266.306 |
| 136 | 4.689% | 3.890% | 285.297 → 274.200 |
| 144 | 4.374% | 4.701% | 291.158 → 277.469 |
| 152 | 3.786% | 3.583% | 295.397 → 284.814 |
| 160 | 4.619% | 4.016% | 298.672 → 286.676 |
| 168 | 3.694% | 4.217% | 303.601 → 290.797 |
| 176 | 3.783% | 3.163% | 301.745 → 292.201 |
| 184 | 2.783% | 2.623% | 306.570 → 298.528 |
| 192 | 2.098% | 2.424% | 309.476 → 301.974 |

b136 对照是**整轮 SBM64 优化前、修好 ticket 的原 M64/N512 最快版**，不是 last16；后者仅用于增量诊断。b64 对照是旧的较快 M64+ticket，不是较慢的 e84 clean；b160/168/176/184 对照为原先 M16 MFMA+缩短A-DMA，b192 为 M16 MFMA+全A-DMA。b144 原 S2-on 胜幅曾不显著，本轮两臂仍一致保留该几何。

b72 放弃：其原最快 SBM32/N512 对比 SBM64 两轮 **250.143→251.171 µs（慢0.411%）**、**250.636→250.755 µs（慢0.048%，CI跨1）**。这份输入的128个 routed expert tile 均为7–31行，没有可减少的 M32 MFMA 行数；不将其外推为通用 batch 阈值。

**证据与接入验证：** [13点冻结结果](.scratch/sbm64_retained_sweep_20260914_v1/README.md)、[repo内报告/精简配对数据](aiter-ming-amd-m3-megamoe/docs/mega_moe/sbm64_20260915/README.md)。全104份原生导入 CPU S1 ISA 与实测候选逐字节相同；b64 200VGPR，其余202，均106SGPR、零spill/scratch、LDS45056B；5371选择器比较和9项CPU回归通过。实际默认 GPU 的16探针/graph512/S1与S2终值/全8rank S1+S2+quant ISA验证在 `.scratch/sbm64_final_rollout_20260915_v1/native_v2`，最终完成状态及 K3 审核见 repo 内 `native_gpu_validation.json` / `final_audit.json`。本轮不新增性能数字，接入验证不依赖 selector/launcher/quant monkeypatch。

**最终验证范围调整：** 用户最后要求不再重跑所有 batch GPU，只需确保 ISA 一致。13点×8rank 共104份原生 S1 ISA已全部匹配；补充原生GPU仅完成 b64/104/112/120（均16探针、graph512及全8rank S1/S2/quant ISA通过）。其余9点未重跑原生GPU，沿用先前两方向实测证据，加本轮ISA身份和选择器验证；不得声称13点全部完成了本轮GPU。b128启动在worker/kernel前因TCPStore端口53128占用失败，按用户要求不重试。

**代码与恢复：** 主分支为 ming fork 的 `ming-amd-m3-megamoe`（`mingg26/aiter`），同步生产树为 `megamoe-s1-1k-prod/aiter:s1-unified-production`；提交身份以本轮 `final_audit.json` 为准。原始实验/source manifests/ISA不改写。此次文档与配置台账更新有用户明确授权；旧 frozen harness pin 了旧文档/HEAD，下次重跑须新建快照和source pins，不能忽略校验。node03/job1913为本轮既有24h allocation，脚本不得取消整个 allocation。

## sbm32

### 2026-09-14 重启状态：代码已提交，M16 分支未保留

实际生产仓库为 `megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，当前 HEAD **`e84ac1b388b709126d40b0e2f79207dc4269ca97`**，工作区干净。最近保留的三步均已直接 commit：

| 提交 | 保留内容 | 本节相关范围 |
|---|---|---|
| `40c3eeedc` | generic S1 的 launch-ticket 消费完成 barrier；next A-scale 延后到当前 MFMA 完成后读取 | b64/72/80/88/96 都命中；ticket 是正确性修复 |
| `5571deb9a` | 每个 N 组的旧 B 全部消费后，才加载该组 next B（`call_pipe_retire_b`） | `M_REPEAT==2 && NUM_ACC_N==4 && mfma_amajor && async_a_copy`；本次 b72/80/88/96 命中，b64 不命中 |
| `e84ac1b38` | 先读当前 A，再发下一步 A DMA（`read_a_before_dma` / `_arbd1`） | 本次五点只有 b96 命中 |

这里记录的是 **S1 实现**；性能表测的是只替换 S1 的完整 forward。b200–256 已合入的 SBM128 私有路径不受本轮 generic `gemm1.py` 改动影响。最新五点补测没有修改代码，没有新 commit，也**尚未回退 b64 的退化**。

### SBM、M16/M32 与实际分支分别指什么

- **SBM32** 是排序/dispatch、A ping/pong LDS 和输出布局采用的物理 32 行 tile；不等于每个 tile 都有 32 个有效 token。
- **M16/M32** 在本节指一个 tile 内逻辑计算 16/32 行的路径，避免与 S2 配置的 `block_m`（BM32/BM64）混淆。
- **当前保留版没有按 `valid_rows` 选择 M16/M32 的两条 K-loop。** SBM32 固定 `M_REPEAT=2`，完整计算两组 16 行；即使 `valid_rows<=16`，也不跳过上半 16 行 MFMA。`tile_valid_rows` 仍用于 A/scale 的边界及输出 mask；“越界不读真实数据/不写有效输出”不等于“MFMA 已跳算”。
- shared 与 routed 是任务来源和地址选择；都进入同一个当前 tile 计算器，不是 M16 与 M32 两个算法分支。b72/b80 的 shared 尾 tile 也没有 M16 专用 K-loop。

本次默认配置的三种编译情况如下（不是根据有效行数在 tile 内动态三选一）：

| Batch | 物理 S1 / 逻辑 M 路径 | 稳态 K-loop 的流水 | S2 默认 |
|---|---|---|---|
| b64 | SBM64/N512/K256；只有完整 M64，`M_REPEAT=4` | 原 A-major pipeline + late A-scale；B retirement 的 mr2 条件不成立；arbd 关闭 | BM64/N128，NT 开 |
| b72/80/88 | SBM32/N512/K256；只有完整 M32，`M_REPEAT=2` | 先发 next A DMA，再读 current A；按 N 组用完旧 B 后发 next B；最后读 next A-scale | BM32/N256，NT 关 |
| b96 | SBM32/N512/K256；只有完整 M32，`M_REPEAT=2` | **先把 current A 读出，再发 next A DMA**；随后沿用同样的按 N 组 B retirement 和 late A-scale | BM32/N256，NT 关 |

五点均为 EP8、H6144/I3072/E128/top4、8 waves、dispatch CU48、B-NT2、work shards8、band4、preplan4、shared packed counters；本轮输入 `tokens==max_tok_per_rank`。S2 使用既有 jointtail，未修改。

arbd 的源码 gate 在 `mega_moe_stage1.py`：H6144/I3072、EP8/epr16/top4、**`fuse_mtpr==96`**、SBM32/N512/8 waves、preplanned/shared L13/pipe weights/A-major/async A；排除 tile-resource、unroll pingpong、split LDS、prefetch-A-operand、packed A-scale。它不是“所有 SBM32 都自动开启”，也不要把 `mtpr==96` 的 gate 误写成按运行时 `valid_rows` 分支。

源码入口：[generic S1 launcher](megamoe-s1-1k-prod/aiter/aiter/ops/flydsl/kernels/mega_moe_m3/mega_moe_stage1.py)、[K-loop](megamoe-s1-1k-prod/aiter/aiter/ops/flydsl/kernels/mega_moe_m3/gemm1.py)、[MFMA/B retirement](megamoe-s1-1k-prod/aiter/aiter/ops/flydsl/kernels/mega_moe_m3/gemm_util.py)。kernel 后缀 `_tb1_salate1_brni1` 是 generic 版本标记，**仅有 `_brni1` 字样不能证明 mr4 等形状实际执行了 B retirement**；要核对编译条件和 ISA。`_arbd1` 则按上述 gate 追加。

### 当前 M32 流水具体做什么

模型 K=6144，按 K256 分为24步：23步稳态循环加1步 final 收尾。每个 K256 step 分两个 K128 子段，四个 N 组、两组 M16；每 wave 合计 16 条 MFMA。`call_pipe_retire_b` 对每个 N 组先完成 `2 ks × 2 mi = 4` 条 MFMA，再加载该组下一 K step 的 B，并用调度 barrier 限定编译器重排，缩短旧 B 与 next B 的同时存活区间。最后一个 K step 不再预取下一组，用独立的 final-step 路径收尾；这不是新增 M16 分支。

b96 的 arbd 把 current A 的四个操作数（实际 ISA 共八条 LDS read）放到 next A DMA 之前。当前 A 已由上一步尾部 wait/barrier 保证就绪，下一步 DMA 写另一个 ping/pong slot；A 地址、计算量与布局没有变化。收益是调度调整的实测结果，不能仅凭时序图断言已经定位了全部硬件原因。

next A-scale 仍在当前 MFMA 之后加载。保留的 SBM32 尾等待为原 `wait_lds_barrier(NUM_ACC_N * _PACK + NUM_B_SCALE)`，本形状参数值为 **10**；改到 16 的试验没有保留。实验是在 FlyDSL 源码中改变显式 wait 参数并由 LLVM 编译，再审实际 ISA，未直接改二进制。packed A-scale 方向只讨论过，尚未实施。

### M16 方向确实试过，但都没有进入当前生产

| 方案 | 分支位置与计算内容 | 结果/状态 |
|---|---|---|
| 单 K-loop M16 跳算 | 仍是一条 M32 K-loop、M32 累加器和物理布局；A DMA 指令形状和 A LDS 读取仍按 M32，tile 入口读取有效行数，循环中按 N 组条件跳过上半 M16 的 MFMA；没有复制完整 M16/M32 两条 loop | 184 VGPR，零 spill/scratch；正向约快 0.38%，另一轮约快 0.08%、CI 跨零，未保留。旧“反向”只反转初始 capture，不能算最终计时 graph 的反向验证 |
| 入口 M16/M32 两条固定 K-loop | tile 入口 `valid_rows<=16` 走 M16，否则 M32；M16 使用 `m_repeat=1`、16 行 A DMA/MFMA/epilogue，输出仍按物理 SBM32 stride；判断不放在每条 MFMA 前 | **245 VGPR、SGPR spill7、scratch0**；用户明确允许超预算候选上 GPU 后，正确接线的 v2 实测慢 **2.023%**，0/12，未保留 |
| 共用 epilogue 的双路径变体 | 尝试减少两条逻辑 M 路径的 epilogue/合流负担 | 242 VGPR，仅离线编译，未保留；不得填写 GPU 收益 |
| 只跳 epilogue 上半 M16 | K-loop 仍完整 M32，仅在输出处理上跳空 | 实测慢约 0.714%，未保留 |
| M16 单循环 wait-once | 试图减少重复等待，仍基于未保留的 M16 单循环 | 实测慢约 0.094%、CI 跨零，未保留 |

入口双路径的有效证据是 `.scratch/b96_s1_m16m32_gpu_20260914_v2/measure/gpu_v1/b96/result.json`；**v1 的“245 VGPR 候选测时”实际上误跑了 184 VGPR wait-once，全部撤销该候选归因**。单循环证据在 `.scratch/b96_s1_m16_singleloop_20260914_v1/`。两类 harness bug 的范围与修复见 [redo 最新重启入口](redo.md)。

b96/seed42 的 routed 路由统计：136 个物理 M32 tile 中，17 个只有 1–16 行，即 **12.5% tile 可以少算半块**；理论 routed MFMA 减少 6.25%，计入 shared 后总 S1 MFMA 减少 5.3125%。这不是 S1 时间或完整 forward 的可直接兑现加速比例，也不能拿“平均每专家 3 行”解释它：EP8 总 routed rows=3072、128 专家，平均为 24 行。

另两个调度方向：next A DMA 延后到首个 N 组之后的 `_adni01` 与 arbd 直接对测 B/A=1.000807，CI [0.999721,1.001962]，没有额外收益，保留 arbd。tail wait10→16 的正向快约 0.823%，旧反向约慢 0.046%；修复真正反向 capture 后的 job1881 严格复测未通过 null-after、敏感性和正式窗口硬件稳定性检查，因此不保留，也不把其诊断慢 0.098% 称为稳定退化。

### b64–b96 与之前保留最快版的补测（job1886）

2026-09-14，node03、EP8、同一进程组内同 operator/输入/固定 quant buffer 配对；每点单独进程组，12 对平衡 AB/BA，每 sample400 replay、chunk200、settling64，实际稳定预热约 21–29 秒。前后各 6 对同 baseline graph 自比较；不扣除 null。计时 graph 真正按 candidate→baseline 捕获并复用，已修复旧重捕获问题。

b64/72/80/88 的 A 为 `b13717ca0 + 仅 ticket barrier`，不是含已知 race 的裸旧版；b96 的 A 为先读 A 之前已保留的最快 `5571deb9a`。B 均为实际当前生产 `e84ac1b38`。所以 b96 是 arbd 的增量，其余四点是本轮 generic 优化的累计补测，不能把百分比直接相加。

| Batch | SBM | 基线 μs | 当前 μs | 当前耗时降低 | 95% CI（降低） | 更快对数 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 266.469 | 270.611 | **-1.555%** | [-1.718%, -1.373%] | 0/12 |
| 72 | 32 | 255.411 | 251.916 | +1.368% | [+1.281%, +1.457%] | 12/12 |
| 80 | 32 | 257.310 | 253.408 | +1.517% | [+1.430%, +1.611%] | 12/12 |
| 88 | 32 | 261.471 | 257.730 | +1.431% | [+1.275%, +1.594%] | 12/12 |
| 96 | 32 | 261.272 | 257.891 | +1.294% | [+1.127%, +1.470%] | 12/12 |

10 种输入/路由探针、graph512、默认配置、5 dispatch、8 rank 源码/ISA/资源和独立配对算术复核全部通过，K3 复核方向一致。**这是用户授权的快速筛查，未跑严格敏感性/null 等价资格判断，也无跨轮稳定性声明。** 时间为完整 forward，不是孤立 S1 时间。

资源：b72/80/88 基线232→当前184 VGPR；b96 两臂184；这些 SBM32 均零 spill/scratch、LDS38912 B。b64 两臂256 VGPR，但 VGPR spill2→0、scratch12→0 B、LDS77824 B，依然慢 1.56%；因此资源改善不自动等于速度改善。五点资源驻留上限均为1 CTA/CU，没有资源超预算。

**重启后的基线选择：** b72/80/88/96 继续从当前 e84 保留实现出发；b64 当前生产不是本轮较快者，如要优化/隔离退化，先用本次冻结的 `b64/baseline`（含 ticket）作较快参照，不能因为 e84 是 clean HEAD 就拿它替代最快基线。b64 尚未回退；下一步是否缩小 late-A-scale 适用范围，留待用户决定。本轮之外、更小的 SBM32 batch 未做这次补测，不外推收益。

完整证据：[五点报告](.scratch/b64_96_s1_current_vs_best_20260914_v1/README.md)、[汇总/顺序/null/资源](.scratch/b64_96_s1_current_vs_best_20260914_v1/review/summary.json)、[最终决定](.scratch/b64_96_s1_current_vs_best_20260914_v1/review/decision.json)。调度 job1883 在任何样本产生前启动失败，后由直接 srun 的 job1886 完成全五点；记录见同目录 `review/scheduler_reassignment.json`。生产没有新修改，也没有运行中的 GPU 作业。

## 2026-09-14 最新：b200–256 默认合入（覆盖下方历史状态）

用户本轮明确授权 GPU 验证、commit 和文档更新。实际生产 `aiter` 已提交 **`b13717ca0a0dbe0cf53df353467b8f816aa7c762`**（不是仅提交外层实验快照）。默认按 batch 选择：**b200/208/216 使用 SBM128 逐 16 行八路径；b224/232/240/248/256 使用修复 ticket 的 64/96/128 三档**。这里“八路径/三档”是 kernel 内逻辑计算行数，物理 SBM 均为 128。

范围仅 EP8、128 总专家（每 rank 16）、H6144/I3072/top4、完整 fused shared L13+L2 且 XCD 调度、`tokens == max_tok_per_rank`，精确覆盖 200/208/216/224/232/240/248/256；无 local-reduce、无显式 Stage1 环境调参。其他 batch、形状和通用默认保持不变；这不是对区间内任意整数 batch 或全部路由分布的最优性声明。

代码入口是 `mega_moe_config.py:SBM128_PATHS` 与 `MegaMoEM3` 的受限选择器。实测 S1 源码在私有 `sbm128_m16/`、`sbm128_tiered/`；不要把通用 `gemm1.py` 当成本轮默认实现。S1 均为 N256/K256，b200–248 的 dispatch CU48/B-NT2，b256 为 CU32/B-NT0；S2 BM32、b200–248 N256、b256 N128，整块跳空 ON、内部 M16 OFF。

证据入口：[megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json](megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json)，[配置台账](megamoe-s1-1k-prod/bench/dep8_best_configs.json)。旧 `normal_timing` 等字段保留历史含义，不能当成本轮默认的新测量。

### 同轮正式结果与不确定性

原扫描采用固定构造 A 后 B，6 个预声明协议、每协议 12 对。表为主协议 K32/N400/chunk200；计时含 TopK、quant/preplan、routed S1/S2、TP1 shared 和 combine/final sum，不含 router GEMM/residual。指标是各 rank 累加 chunk event 后的 max-rank graph 时间，不是孤立 S1 时间或跨 GPU 同步 makespan。A 为目标 batch 的历史最快源码补 ticket barrier；不应把修复后 A 称为历史源码逐字节重放。

| batch | 修复后旧方案 A / 八路径 B（μs） | B/A 相对变化 | 默认 |
|---:|---|---|---|
| 200 | 324.215 / 309.705 | -4.475%（95% CI [-4.586%, -4.376%]） | m16 |
| 208 | 329.766 / 319.115 | -3.230%（95% CI [-3.336%, -3.114%]） | m16 |
| 216 | 331.721 / 325.704 | -1.814%（95% CI [-1.991%, -1.637%]） | m16 |
| 224 第1轮 | 335.673 / 335.225 | -0.133%（95% CI [-0.312%, +0.047%]） | tiered96 |
| 224 第2轮 | 333.882 / 334.589 | +0.212%（95% CI [+0.012%, +0.408%]） | tiered96 |
| 232 | 337.844 / 341.905 | +1.202%（95% CI [+1.007%, +1.397%]） | tiered96 |
| 240 | 340.488 / 346.227 | +1.685%（95% CI [+1.513%, +1.837%]） | tiered96 |
| 248 | 342.866 / 347.585 | +1.377%（95% CI [+1.103%, +1.631%]） | tiered96 |
| 256 | 360.058 / 364.087 | +1.119%（95% CI [+0.991%, +1.248%]） | tiered96 |


b224 两轮方向反转，轮次差异显著；保守保留三档。合并 CI 跨 1 不证明稳定等价，固定构造顺序的系统偏差也未被两次同向构造抵消。

b256 原生入口接线验证中，**同一份三档 kernel** 的 B/A 先为 +1.224%（95% CI [+1.105%, +1.347%]），交换两对象构造先后后为 -0.775%（95% CI [-0.918%, -0.627%]）；八 rank 的完整流水线 ISA 相同。因此不能仅凭原扫描的 1.12% 宣称三档胜出，也不能把这些同 kernel 时间差当作生产接线优化收益。构造顺序相关效应存在，但地址/L2/XCD 机制并未证实，不能用该差值扣减真实两方案结果。

真实三档/八路径再做反向构造、完整 6 协议×12 对：主协议 B/A -0.942%（95% CI [-1.094%, -0.804%]）。两构造方向等权 log-ratio、分层 pair bootstrap 的结果为 +0.083%（95% CI [-0.018%, +0.179%]），最终选择 `tiered96`。这是每种构造方向仅一个进程组的条件性估计，不能给出跨分配/跨时段不变性的置信区间；若八路径收益未建立，则保守保留修复后三档。原始轮次、反向轮次、同 kernel 诊断全部留档，没有挑掉不利样本。

### 代码、ISA 与正确性

- 保留八路径最后确认的 M48/M64 流水、上半 64 行 epilogue 跳空、M128 的 ks-half B retirement 与延后 A-scale。**M80 保留原 A-major/提前 A-scale，拒绝的 M80 实验未混入。**
- 八路径 8 batch×8 rank 的 S1 ISA 检查通过：VGPR253、SGPR106、VGPR spill0、SGPR spill3、scratch0、LDS90112 B。三档既有资源为 VGPR256、SGPR106、VGPR spill7、SGPR spill3、scratch32 B、LDS90112 B；不要把所有默认写成零 spill。
- 8 档原生默认候选均不 monkeypatch 选择器、S1 或 quant；每档 23 个数学/bit-exact/路由与 epoch 探针、graph512、12 对接线对照、8 rank 实际 S1/S2 ISA 与选中实测源码一致，5 dispatch 与 source/telemetry 门禁通过。
- CPU 4736 个选择器场景、768 个范围外 S2 跳空场景、5 个 regression tests 通过。两个私有包仅改 import，通用计算源与范围外选择器行为保持原样；shared L13-only、非目标维度、显式调参不进入本轮默认。
- K3 只读审核确认无阻止提交的代码问题，并要求保留 b224/b256 的测量限制。Mori 端口占用和 Slurm credential 失败的启动日志单独保存，未计为性能样本。

### 后续复现

归档含原始完整结果/日志、冻结 harness/source manifest、源码快照、64 份八路径 ISA、私有包/S2 重接线 ISA 和 K3 审核。旧 harness 钉住合入前 HEAD 与文档哈希，**合入后不能直接照抄旧 run.sh 重跑**；应从目标 batch 当前选中私有实现建立新实验目录和新 source pin，保留全部门禁。后续不要退回更慢的通用默认或历史失败八路径做 baseline。b160–192 的历史最快实验未在本轮合入，相关历史结论保留。

## 2026-09-14 重启记录：当前最快边界、逐 16 行实验与未决原因

本节是当前结论，优先于下文 2026-09-13 的历史报告。生产 `aiter` 仍停在 clean commit `05bbc69252444ece0bff54858eae4be2864000a7`，工作区干净；下面的新方案均只在 `.scratch`，没有合入生产。所有新实验必须从目标 batch **已测最快源码**构造，并在同一进程、同一输入下直接配对；不同实验进程的绝对均值不能相减或排序。

### 目前各 batch 的最快记录

统一形状是 EP8、每 rank 给定 batch、128 experts、top4、H6144/I3072、seed42；时间窗口为 TopK + quant/preplan + routed S1/S2 + TP1 shared + Combine/final sum，不含 router GEMM 与 residual。表中时间只能和同一行注明的同轮对照一起解释。

| batch/rank | 当前最快的实验源码与配置 | 同轮时间与相对结果 | 证据强度 |
|---:|---|---|---|
| 160 | `.scratch/sbm64_m16_dma_sweep_20260913_v1/s1_only`；S1 SBM64/N512，按 16/32/48/64 行固定路径缩短 MFMA 与 A DMA；S2 BM32/N256 整块跳空 ON、内部 M16 OFF | 303.857→303.591 μs，快 0.088%，7/12 | 新规范正式测量；效应很小，部分协议的等价性 CI 未收进 ±0.2% |
| 168 | 同上 | 306.000→304.957 μs，快 0.341%，12/12 | 新规范正式测量 |
| 176 | 同上 | 306.385→305.020 μs，快 0.446%，12/12 | 新规范正式测量 |
| 184 | 同上 | 309.673→308.903 μs，快 0.249%，10/12 | 新规范正式测量 |
| 192 | `.scratch/sbm64_n512_singleloop_20260913_v1/s1_only`；S1 SBM64/N512，在一个 K-loop 内按 16 行屏蔽 MFMA，A DMA 仍搬满 64 行；S2 整块跳空 ON | 316.248→312.255 μs，快 1.263%，12/12 | 新规范正式测量；这是当前最强候选，但尚未与下述 scalar-A-DMA 版直接 head-to-head |
| 200 | `.scratch/b200_sbm128_entry_ticket_fence_20260914_v1/candidate_200`；S1 SBM128/N256，64/128 两档计算和短 A DMA；S2 BM32/N256 整块跳空 ON、内部 M16 OFF；含初始 ticket 读取完成 barrier | 325.097→323.418 μs，相对此前最快版再快 0.516%，12/12 | 新规范正式测量；不要把 323.418 与旧进程的 322.141 直接比较 |
| 208 | `.scratch/sbm128_sweep_s2empty_20260913/kernels_b256`；S1 SBM128/N256，64/96/128 三档；S2 BM32/N256 整块跳空 ON | 331.970→325.552 μs，快 1.93%，6/6 | 旧协议：4 秒预热、6 对；尚未补 ticket 修复的逐点正式复测 |
| 216 | 同上 | 336.160→327.910 μs，快 2.45%，6/6 | 同上 |
| 224 | 同上 | 344.917→328.606 μs，快 4.73%，6/6 | 同上 |
| 232 | 同上 | 354.333→332.744 μs，快 6.09%，6/6 | 同上 |
| 240 | 同上 | 360.767→335.387 μs，快 7.03%，6/6 | 同上 |
| 248 | 同上 | 384.598→338.553 μs，快 11.97%，6/6 | 同上；收益包含给该 batch 补 S2 整块跳空 |
| 256 | 同上；S2 为 BM32/N128，其余配置同三档方案 | 378.622→356.192 μs，快 5.92%，6/6 | 另一次 8 秒/12 对测得快 4.45%，方向复现但幅度不可跨轮相减 |

b160–192 的 scalar-A-DMA 正式汇总见 [SBM64 逐 16 行结果](.scratch/sbm64_m16_dma_sweep_20260913_v1/review/sweep_results.json)；b192 更快的 single-loop 证据见 [single-loop 审核](.scratch/sbm64_n512_singleloop_review_20260913/analysis.json)。b208–256 的旧轮次汇总见 [SBM128 扫描](.scratch/sbm128_sweep_s2empty_20260913/scan_results.json)。b200 的最新实验全表见 [逐 16 行与 ticket 汇总](.scratch/sbm128_poll_serial_review_20260914.json)。

### 实际做过什么

1. **SBM64 的 scalar-A-DMA 逐 16 行版在 b160–192 有小幅收益。** 它保留 S1 SBM64/N512 和 B 双缓冲，在一个 K-loop 内按实际 `valid_rows` 屏蔽 16 行 MFMA 组，并让 A DMA 也按 16 行缩短；对 16/48 行，最后 16 行 A DMA 仍有每个 K step 的 wave-uniform 尾部分支。候选从各 batch 当时最快的 SBM64/S2-on 源码构造，完成 6 种预声明协议、每种 12 对、48–56 秒稳定预热、15 个正确性探针和 graph512。b160–192 依次快 0.09%、0.34%、0.45%、0.25%、0.64%。

2. **b192 另有一个更快的 single-loop 版。** `.scratch/sbm64_n512_singleloop_20260913_v1/s1_only` 同样在一个 K-loop 内按 16 行屏蔽 MFMA，但不缩短 64 行 A DMA；它相对同轮最快 SBM64 基线快 1.263%，明显大于 scalar-A-DMA 版的 0.643%。两候选没有直接同进程 head-to-head，因此不能用 312.255 与 312.596 两个跨轮均值精确算差；后续 b192 实验应先以 single-loop 为起点，并把 scalar-A-DMA 作为同进程对照消融。

3. **SBM128 在 b200 的旧最快版是 64/128 两档。** 它保留物理 128 行 A ping/pong LDS 和输出布局；选择 64 行时只搬运、计算逻辑上需要的前 64 行，并保留 B 双缓冲。S2 使用 BM32/N256，整块跳空始终开启，内部 M16 跳空关闭。

4. **修复了一个真实的初始 ticket 同步缺陷。** tid0 把 64 位 launch ticket 写到 LDS[0:8]，全 CTA publish barrier 后各 wave 读取；旧代码缺少“所有 wave 已消费完成”的 barrier，leader 可能先把第一个 32 位 work ticket 写回同一 LDS[0:4]。修复是在 `ticket64 = Vec(ticket_view.load())[0]` 后、任何复用该 LDS 前增加 `fx.barrier()`。源码和实际 ISA 已确认读后出现 `s_waitcnt lgkmcnt(0); s_barrier`。在 b200 旧最快两档版上，该修复正式配对快 0.516%。它证明旧代码有 race；此前 GPU fault 没有捕获 fault PC/core，因此**不能**声称 fault 已被该 race 因果解释。

5. **SBM128 完整逐 16 行版未采用。** 当前实现把 `valid_rows` 在 tile 入口一次性分到 M16/32/48/64/80/96/112/128 八个固定 K-loop，逻辑 A DMA 和 MFMA 随路径缩短，物理 128 行 LDS 布局、A ping/pong 和 B 双缓冲不变。最终版本还修正了 M64 的 current-A/next-A DMA 顺序、提前发出 next-B、把 wait/barrier 放到 16 条 MFMA 后，并延后 next A-scale 读取。8 rank ISA 门禁和全部正确性门禁通过，但相对“64/128 两档 + ticket 修复”仍慢 **1.429%**，95% CI 为 **[1.291%, 1.569%]**，0/12 更快。

### 为什么 SBM128 逐 16 行仍不够快

已经确认的是“节省的行工作存在，但实现新增成本更大”，还没有完成因果拆分。b200 正式输入在 8 rank 合计的 128 个本地专家计数落在：M48 路径 55 个、M64 路径 70 个、M80 路径 3 个；旧两档需要处理 `125×64 + 3×128 = 8384` 行，新八档处理 `55×48 + 70×64 + 3×80 = 7360` 行，静态行工作减少约 **12.2%**，所以不能说跳空没有真正发生。

已见的结构性代价：S1 ISA 从 4015 行增到 6503 行，标量 branch 从 55 增到 107；VGPR 上限均为 256，但 spill 从 2 增到 11、private segment 从 12 B 增到 48 B，ISA 的 scratch load/store 指令从 5 增到 30。K-loop 审计中 scratch 为 0，因此不能描述成“每轮 K 都 spill”；额外 scratch 主要在路径前后与外围控制区。代码体积/I-cache、spill 延迟分别贡献多少，尚无 PMC/ATT 证据。

此外，最近的流水修正只完整针对实际 M64 路径。M48/M80 是本次输入中实际命中的新增路径：它们的 odd-16 A DMA 尾部仍在每个 K step 执行 wave-uniform 分支；其它 M 路径的 A 读取、next-B 发出和 LDS wait 排布也没有得到与 M64 相同程度的优化。这些是明确可检查的差异，但目前不能断言其中任何一项单独造成 1.43%。6 种协议中的 `k0/k16/k32` 表示不计时的 settling replay 数，不是 GEMM K 长度，不能据此推断回归是否位于 K-loop。

### 重启后的最小实验顺序

1. 从 b200 当前最快的 `candidate_200` 复制新实验，保留 ticket barrier、64/128 两档、S2 跳空和 B 双缓冲；只新增 **M48** 固定路径，不带本轮 M64 专用重排。它覆盖本次 55/128 个专家，直接判断少算 16 行能否抵过单路径成本。
2. 若 M48 有收益，再单独加入 **M80**；它本轮只命中 3/128，主要用于验证 odd-16 尾部实现，不应先带入其它五条未命中路径。
3. 每一步先看实际 ISA：VGPR/spill/private、代码大小、scratch 位置、M48/M80 的 A DMA 与 wait/MFMA 顺序；随后严格按 `tool_measure.md` 做同进程 23 探针、graph512、6 协议×12 对正式测量。
4. 若只加 M48 仍使 spill/代码显著膨胀，先改入口分支的返回值/acc 生命周期以消除跨 `scf.if` 保活，再测；只有消融仍无法解释时才采 I-cache/HBM/L2 PMC 或 ATT。
5. b208–256 的下一步不是直接沿用未修复旧源码继续调参：分别从该 batch 已测最快三档源码加同一个 ticket barrier，先与原最快同进程配对，建立各 batch 的修复后基线，再进行其它实验。
6. b192 若继续，先从 single-loop 版加入 ticket barrier，并在同一进程直接比较“满 64 行 A DMA”与“按 16 行缩短 A DMA”；现有两轮已经提示缩短 A DMA 可能损失约一半收益，但跨轮结果不能当成因果证明。

4 个 K3 分别审计了历史、batch 最快表、逐 16 行代码/ISA 和叙述边界。其输出只作为线索；本节已独立纠正两类误读：`k0/k16/k32` 不是 GEMM K，b200 每个本地专家平均约 50 routed 行而不是 6.25 行。结论以上述 JSON、源码和实际 ISA 为准。

**SBM128 优化记录 — 2026-09-13**

这次有效的方案是：**把同一个专家收到的 token 按 128 行分块，但尽量少算补出来的空行；S2 遇到整块空数据就直接跳过。** 在本次均匀随机路由测试中，b200–b256 都比 clean commit 快，b192 反而慢，因此不建议在 b192 启用这套方案。这里的 b200 指每张卡（一个 rank）有 200 个输入 token。

代码目前在实验副本中，尚未合入生产仓库。对照基线始终是未修改的 clean commit `05bbc69252444ece0bff54858eae4be2864000a7`。

每个 token 按路由选择的专家称为 routed expert；全部输入都经过的共享专家称为 shared expert。SBM64 是把同一个专家收到的 token 每 64 个分一块，不足的部分补空位；SBM128 则每 128 个分一块。例如，一个专家收到 65 个 token，SBM64 要处理两块，SBM128 只需一块。但如果直接把 128 行全部计算，也会浪费很多计算，所以扩大分块必须配合下面的改动。

1. **S1 用大块装数据，用较短的计算处理不足一块的数据。** S1 是专家的第一层矩阵计算。分块从 64 改为 128，同时把输出方向的块宽从 512 改为 256，避免每个任务需要保存的计算结果翻倍。shared 的分块方式和任务分配也同步调整。

   每块开始前，根据有效行数选择一段固定大小的计算。b200 使用 64/128 行两档；b208–b256 使用 64/96/128 行三档。例如三档版本遇到 70 行，只按 96 行计算，不再按 128 行计算。判断放在整段计算之前，不在内部反复判断。这里检查的是“补出来的空行”，不是扫描输入数值是否全为零。

2. **S1 的输入 A 搬运也跟着缩短。** 选择 64 行计算，就只搬运前 64 行 A；选择 96 行，就只搬运前 96 行。省掉的是后面无用部分的搬运指令和填零工作，并不等于显存（HBM）读取量按比例下降。结果仍保留 128 行分块的布局，未计算的尾部补零，因此后续步骤不用换布局。B 权重继续保留双缓冲，即计算当前一批时预取下一批。

3. **补齐 S2 的整块跳空范围。** S2 是专家的第二层矩阵计算，在本次扫描中一直是每块 32 行，没有改成 128 行。

   一个专家有 50 行数据时，SBM128 会给 S2 留下四块：`[32 行、18 行、0 行、0 行]`。后两块应直接跳过。原代码只对部分 batch 开启这项功能，包含 192、200、256，却漏了本次扫描的 208、216、224、232、240、248。这次补齐了这六个点；跳过空块时，队列计数照常推进，shared 的计算也照常执行。S2 内部按 16 行跳过的功能保持关闭。

   这项修复有直接数据支持：b208 的候选端到端时延从**修复前的 353.98 降到修复后的 325.55 μs**；这两个数都是候选结果，clean 的对照值见下表。单次 trace 中，S1 的 8 rank 平均时间约为 182.5→183.3 μs，S2 则为 **123.1→92.8 μs**。trace 用来定位变化，不与下面的稳定计时混算。

下面是最终方案与 clean 的完整对比。**负数表示更快。** b192 这一行用于确定不启用的边界；它和 b200 一样使用 64/128 行两档，其余点使用三档。

| 每 rank 的 batch | clean（μs） | SBM128 方案（μs） | 时延变化 |
|---:|---:|---:|---:|
| 192 | 311.69 | 319.51 | +2.51% |
| 200 | 332.83 | 322.14 | −3.21% |
| 208 | 331.97 | 325.55 | −1.93% |
| 216 | 336.16 | 327.91 | −2.45% |
| 224 | 344.92 | 328.61 | −4.73% |
| 232 | 354.33 | 332.74 | −6.09% |
| 240 | 360.77 | 335.39 | −7.03% |
| 248 | 384.60 | 338.55 | −11.97% |
| 256 | 378.62 | 356.19 | −5.92% |

208–248 的收益包含补齐 S2 整块跳空，不能全部算作 SBM128 本身的收益。每个点都沿用 clean 在该 batch 的 S2 参数：每块都算 32 行；输出方向的块宽在 b256 为 128 列，其余点为 256 列。因此 b256 与 b248 也不是只差 8 个 token。

b256 上一轮 12 对测试的收益为 4.45%，本轮为 5.92%。两轮实验树的源码并不完全相同（`mega_moe_config.py` 差 6 行、`mega_moe_stage2.py` 差 2 行），但差异只影响其它尺寸：b256 本来就在原白名单内，两轮为它选出的配置一致。这个 1.5 个百分点的差异是跨轮波动，不能当作新增优化收益。

**为什么 b192 不用？** 在这批路由中，8 个 rank 的 routed 行块合计只从 129 减到 128，省得很少。从工作量看，64 行的计算分支也只是回到 clean 原本的计算量，而输出块变窄带来了更多任务和重复读取 A 的开销。最终端到端实测变慢。S1 的 shared 行块在 b192 从 3 块减到 2 块，在 b200 则从 4 块减到 2 块；所以 b192→b200 的边界也包含 shared 多出一块的影响，不能只看 routed。

**真实流量变化时怎么理解？** 同一个本地 batch 下，shared 处理全部输入，工作量固定；routed 的工作量则取决于各专家实际收到多少 token。如果一个专家收到 200 个 token，SBM64 要分 4 块，SBM128 只需 2 块，通常更有机会受益，尤其当它所在的 rank 正在拖慢整体时。但 SBM128 不会把这个专家的工作分给其他 rank，也省不掉有效 token 的计算。实际效果还取决于其他专家的负载、缓存和计算开销。因此 b200–b256 是这次均匀随机路由下测到的范围，不是所有部署流量的固定阈值；偏斜路由的性能尚未在本轮扫描中测量。小 batch 的 SBM128 适用范围仍未测；后续对 SBM64 的跳空检查见文末补充。

测试使用 8 卡 EP8、128 个专家、top4，模型维度 6144、中间维度 3072，路由种子为 42。计时包含选专家、routed MoE、TP1 shared MLP 和结果相加，不包含 router 矩阵乘和残差。每个 batch 在同一进程内共享输入和权重，交替测试 clean 与候选共 6 对，每个 sample 先跑 16 次不计时的 settling replay，再测 200 次 graph replay，取 8 个 rank 中最慢的时间。b200–b256 每点都是 6 对全胜；全部点通过 7 类输入与路由检查、256 次连续 graph replay，候选与 clean 输出逐位一致。K3 分别审核了 S1 两版代码和 S2 跳空范围扩展，未发现阻塞性正确性问题。

**本轮相对 `tool_measure.md` 的两处偏差，需要在引用这些数字时一并说明：**预热为 4 秒，低于规范要求的至少 8 秒（规范的理由是让频率、功耗和温度进入稳定区间）；配对数为 6，低于规范推荐的 12，也低于 b256 上一轮所用的 12。两项都会放大轮内估计的不确定性；本轮各点 6/6 全胜、CI 不跨零、候选臂 cv 在 0.08–0.14%，但在补足预热与配对数之前，这些收益应按“单轮观测”引用，而不是稳定加速。

逐 16 行反复判断的 S1/S2 跳过方案没有带来稳定收益，未采用。

可复查的文件：

- [本轮扫描数据](.scratch/sbm128_sweep_s2empty_20260913/scan_results.json)：包含每点原始结果路径、置信区间和各 rank 的 tile 数。
- [b208 修复前](.scratch/sbm128_sweep_20260913/b208/result.json)、[b208 修复后](.scratch/sbm128_sweep_s2empty_20260913/b208/result.json)、[b256 上一轮 12 对测试](.scratch/sbm128_tile_fastpath_vsprod_20260913_v1/b256_clean_vs_fast/result.json)。
- [S1 两档代码](.scratch/sbm128_sweep_s2empty_20260913/kernels_b200/mega_moe_m3/gemm1.py)、[S1 三档代码](.scratch/sbm128_sweep_s2empty_20260913/kernels_b256/mega_moe_m3/gemm1.py)、[S2 整块跳空代码](.scratch/sbm128_sweep_s2empty_20260913/kernels_b256/mega_moe_m3/mega_moe_stage2.py)。这些是实验代码，不是已经启用的生产默认配置。
- [b208 逐 rank trace（修复后）](.scratch/sbm128_sweep_s2empty_20260913/b208/)、[b208 逐 rank trace（修复前）](.scratch/sbm128_sweep_20260913/b208/)：上文 S1 182.5→183.3、S2 123.1→92.8 μs 的来源，每 arm 每 rank 一份。
- [K3 的 S1 审核](.scratch/sbm128_sweep_20260913/k3_review.log)、[K3 的 S2 审核](.scratch/sbm128_sweep_s2empty_20260913/k3_s2_review.log)、[本报告的 K3 复核](.scratch/sbm128_sweep_s2empty_20260913/k3_report_review.log)。报告已采纳复核提出的两处澄清。

**补充：SBM64 的 S2 整块跳空 — 2026-09-13 重启前**

这轮保持 clean 的 S1/S2 参数，只在实验副本里给 S2 的跳空白名单补上 **b120、136、144、152**。S1 仍是 SBM64/N512，S2 仍是 BM32/N256；没有改成 SBM128。生产仓库仍在 clean commit `05bbc6925`，这四个点尚未默认启用。

**b120 有明显的单轮收益，b136 小幅受益；b144、152 暂时只能说基本持平。** 空块按 8 个 rank 合计的 routed BM32 行块计数，分母都是 256；每个行块另有 24 个 N 方向任务，占比不变。

| batch/rank | clean（μs） | 仅补 S2 跳空（μs） | 时延变化 | 更快的配对 | 全空行块 / 256 |
|---:|---:|---:|---:|---:|---:|
| 120 | 291.00 | 278.47 | −4.30% | 6/6 | 91（35.55%） |
| 136 | 292.04 | 290.00 | −0.70% | 6/6 | 48（18.75%） |
| 144 | 294.41 | 293.66 | −0.25% | 4/6 | 36（14.06%） |
| 152 | 297.19 | 297.42 | +0.08% | 2/6 | 23（8.98%） |

以上仍是完整 forward 的同进程 clean 对照，不是 S2 独测。四点全部通过 7 类输入/路由、独立数学参考、逐位一致、256 次连续 graph replay 与源码/实际 kernel 检查；S1 编译代码逐字节不变。S2 两臂都是 VGPR231、零 spill、scratch0、LDS33088 B。仍只用了 **4 秒预热、6 对计时**，与上文相同，不能视为跨轮稳定结果；b144、152 的配对置信区间包含零收益。

**空块确实存在，只是随 batch 增大而减少。** 专家收到 30 个 token 时，SBM64 拆给 S2 的两块是 `[30, 0]`，后者可跳；收到 34 个时变成 `[32, 2]`，第二块虽然有 30 行空着，却必须计算整个 BM32。均匀 top4 路由下全体专家的平均 token 数为 `batch/4`，b120、136、144、152 分别是 30、34、36、38，正好跨过 32 这条线。shared 的末块可能不满，但仍有真实 token，这项判断不跳过 shared。

各 rank（0–7）的全空 BM32 行块数如下，每 rank 的分母都是 32：

| batch | rank 0–7 的全空行块数 |
|---:|---|
| 120 | 13、10、14、11、10、12、12、9 |
| 136 | 7、4、10、7、6、6、4、4 |
| 144 | 4、3、8、7、2、6、4、2 |
| 152 | 3、2、6、3、2、4、2、1 |

**为何省掉这些块，完整耗时却没有等比例下降，目前尚未解释清楚。** 代码可确认：clean 已按有效行数屏蔽 A 的空行读取；新增判断省掉空块的 B 读取指令、矩阵计算和收尾操作，但保留队列领取与遍历。同一专家的有效块还要读取权重，少执行多少块不能直接换算为少读多少 HBM。缓存复用、调度尾部和 rank 等待是待验证方向；没有本轮 HBM/L2 计数器或逐任务时间证据，不能断言它们是主因。

已有单次 trace 中，S2 的 8 rank 平均耗时为 b120 **86.67→74.88**、b136 **90.02→80.08**、b144 **89.40→84.75**、b152 **89.93→86.95 μs**。每 arm 每 rank 只有一个 S2 实例；这些数与稳定计时不是同一窗口，不能用来解释完整 forward 的差额，也不能当成重复分层计时。**计划中的重复 S2 breakdown 尚未实现、尚未运行；用户决定先更新文档后重启。**

另试的 SBM64 S1 M32/M64 分支在 b64/104/112/128 全部变慢约 10.7%–19.0%，VGPR spill 从 2 增到 87，未采用；后续按用户要求只查 S2。

证据与恢复入口：

- [本轮完整结果汇总](.scratch/sbm64_empty_small_20260913_v2/scan_results.json)、[S2 空块与逐 rank trace 统计](.scratch/sbm64_empty_small_20260913_v2/s2_empty_analysis.json)。
- [b120 原始结果](.scratch/sbm64_empty_small_20260913_v1/b120_s2/result.json)、[b136 原始结果](.scratch/sbm64_empty_small_20260913_v1/b136_s2_only/result.json)、[b144 原始结果](.scratch/sbm64_empty_small_20260913_v2/b144_s2_only/result.json)、[b152 原始结果](.scratch/sbm64_empty_small_20260913_v2/b152_s2_only/result.json)。
- [S2 实验代码](.scratch/sbm64_empty_small_20260913_v2/s2_only/mega_moe_m3/mega_moe_stage2.py)、[K3 独立分析日志](.scratch/sbm64_empty_small_20260913_v2/k3_s2_analysis.log)、[重启 handoff](handoff_megamoe.md)。K3 核对了空块计数、实际 `_ez1` 分支及资源；日志中的缓存/等待解释仍是推测，结论以本文注明的证据范围为准。
