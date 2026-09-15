# MegaMoE SBM64 定稿报告（b64–b192，13 档 retained）

- 范围：EP8 fused-shared 小 batch 的 SBM64 Stage1 最终接入——13 个 batch 选定 `m32_m48_m64` N256 路径、b72 否决、S2 配套选择、性能与验证证据链。
- S1 历史性能基准：fork commit **`f7a96515aa8e3b7c1e41506c37d49b78651ce8ad`**（2026-09-15），"Enable retained SBM64 branches for measured EP8 batches"。
- 来源：commit 内 `docs/mega_moe/sbm64_20260915/{README.md,final_audit.json,selector_audit.json,k3_final_review.txt,native_cpu_isa.json,native_gpu_validation.json,measurements.json}`；工作区 `report_sbm128.md` / `redo.md` / `handoff_megamoe.md` 的 2026-09-15 节（与 commit 归档版正文一致）。

当前 S2 和重启命令见 [report_s2.md](report_s2.md)、[redo.md](redo.md)。下文性能表仍为 S1 历史配对，不与后续收益相加。

## 1. 最终默认选择

**接入 batch（tokens/rank，精确匹配）**：`64, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192` —— 共 13 点。

为什么是 SBM64（机制，源码注释 `mega_moe_config.py` 的 `small_sort_block_m`）：sort block 必须**装得下一个 expert 的全部行**。S1 把每 expert 补齐到 block 整数倍，tile 数 = Σ_e ceil(rows_e/block)；EP8 top4 下 rows_e 均值 = tokens/4。当均值落在 block 的 1–2σ 内，"是否要第二个 tile"接近掷硬币，各 rank tile 数不再一致，而整层按最慢 rank 计费。实测（T=104）：sbm32 各 rank 17/19/18/19/19/17/17/17 tile、S1 跨度 155–193 µs、整层 313.2 µs；sbm64 恒 16 tile、S1 152–159 µs、整层 290.0 µs——多算近一倍 padding 行反而快 23 µs，因为 padding 行无访存流量（读写都有界）。通用 fallback 规则：`64 if (mtpr%64==0 or mtpr>96) else 32`；13 个 retained 点在此之上再换成实测的 `m32_m48_m64` N256 实现。

- 代码入口：`mega_moe_config.py:163` 的 `SBM64_PATHS = {t: "m32_m48_m64" for t in (...)}`；selector 分支 `mega_moe_m3.py:386-387`；kernel 模块 `aiter/ops/flydsl/kernels/mega_moe_m3/sbm64_m32_m48_m64/`。
- scope guard（`mega_moe_m3.py:115-122`）：shared_w13+w2 提供且 `shared_xcd_schedule` 开、`(world_size, epr, H, I, topk)==(8,16,6144,3072,4)`、`mtpr ∈ SBM64_PATHS`、无 local-reduce/xcd_local、无任何 `M3_MEGAMOE_*` Stage1 env override。其余 batch/形状/显式调参行为不变。
- S1：物理 **SBM64 / N256 / K256**，按 tile 入口 `valid_rows` 选 M32（≤32）/ M48（33–48）/ M64（49–64）固定 K-loop。

**当前 S2 选择**：

| batch | S2 | NT / 跳空 |
|---|---|---|
| 64 | BM64/N128/K256，W4、640 CTA | NT ON；BM==SBM |
| 104/112 | BM32/N256/K256，W4、640 CTA | NT OFF；整空子块跳过；112 constant N/K |
| 120–192 步8 | BM64/N256/K256，W8、256 CTA，M32/M48/M64，fixed N24 | NT OFF；无 BM32 子块；上半32 epilogue 跳过 |

**b72 当前为 SBM32/N256/M16-M32**，并保留 S1/S2 leader drain。早期改 SBM64 两轮慢0.411%/0.048%，所以不走 SBM64；随后已优化 SBM32，不能继续写为 N512。M48 冗余 scale 裁剪（scale6）仍未采用。

## 2. 保留的 kernel 特性（sbm64_m32_m48_m64）

- current-A 前读：先读当前 A 操作数，再发下一步 A DMA。
- K128 半组 B retirement：每个 N 组的旧 B 全部消费后才加载该组 next B。
- late A-scale：next A-scale 延后到当前 MFMA 完成后读取；M48 用向上取整的 scale 分组、末 16 行 A copy 有界。
- 共用物理 M64 epilogue 的 **last16 / middle16** producer+consumer guards：空 rows48–63 与 rows32–47 在 SwiGLU/LDS 生产与量化两处都跳过，barrier 无条件（不放进条件分支）。
- launch-ticket 修复保持（读后 `fx.barrier()`，见 `report_sbm128.md` §ticket）。
- middle16 是用户决定保留；其相对 last16 的增量收益**未解决**（no-skip 对照 +0.079%/−0.023%，CI 全跨 0），不能把 13 点累计收益归给该项。

资源：b64 为 200 VGPR，其余 12 点 202 VGPR；全部 106 SGPR、零 spill、零 scratch、LDS 45056 B。

## 3. 性能（同轮完整 forward 配对，非默认/standalone 对照）

每点两种真实 graph capture 顺序各 12 平衡配对，共 **26 轮 / 312 对全部候选更快**，每轮 ratio 95% CI 均低于 1。口径是 quick screening，非严格稳定性认证，也不保证任意路由分布。

| batch | BA 降 | AB 降 | AB 原最快 → 保留版（µs） |
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

对照基线选择（都是"该 batch 此前最快"，不是 clean 默认）：b136 用整轮 SBM64 优化前、修好 ticket 的原 M64/N512 最快版（不用 last16 增量对照）；b64 用旧的较快 M64+ticket（不是较慢的 clean e84 默认）；b160/168/176/184 对照为 M16 MFMA+缩短 A-DMA；b192 为 M16 MFMA+全 A-DMA。b144 原 S2-on 胜幅曾不显著，本轮两臂一致保留该几何。

## 4. 验证证据链

| 项 | 结果 |
|---|---|
| 原生导入 CPU S1 ISA | **104/104**（13 点 × 8 rank）与实测候选逐字节相同；三方 sha256 比对（记录 == 当前文件 == 历史 sweep 候选） |
| selector 审计 | **5371** 个比较通过；唯一差异是 13 个目标 batch 的 `stage1.tile_n=256` 与 `stage1.sbm64_path=m32_m48_m64`；未测形状、partial batch、b72、显式 env override 全部不变 |
| CPU 回归 | **9 项**通过（`op_tests/test_mega_moe_m3_config.py`，含 S2 skip 边界与 SBM128 默认不变） |
| 原生 GPU 复验 | **仅 b64/104/112/120 四点**通过（各 16 探针、graph512、全 8 rank S1/S2/quant ISA，共 96 个 GPU ISA 对象） |
| 其余 9 点（128/136/144/152/160/168/176/184/192） | **未重跑原生 GPU**——用户明确要求 ISA 一致即可后停止。其证据是：先前两方向实测 + 本轮 104 份 ISA 身份 + 5371 选择器验证。**不得声称 13 点全部完成了本轮 GPU 验证** |

启动失败实录（不影响保留样本）：b128 的 native_v2 启动在 worker/kernel 之前因 TCPStore 端口 53128 占用失败，按用户要求未重试；b64 的 native v1 因容器 bind 路径误当宿主路径在 kernel 前中止，native_v2 已纠正。

**接入验证的性质**：integration/correctness 验证，不含 selector/launcher/quant monkeypatch，kernel 源码全程冻结，不新增性能主张。源码迁移只改相对 import；measurements.json 保留 26 轮的逐轮 ratio/CI、null 对照、配对 critical 时间、配置与资源记录，原始结果路径+hash 指向不可改写的原始证据。

K3（`inferact-kimi-k3`）最终复核 PASS：独立重算 104 份 ISA hash 链、四点 GPU 记录、core diff 范围（仅 `mega_moe_config.py` / `mega_moe_m3.py` / `mega_moe_stage2.py` / `op_tests` + `sbm64_m32_m48_m64/` 新模块）、生产树与 fork 逐字节一致。其文稿中"b64/104/104/112/120"的 b104 重复是笔误，实际四点为 b64/104/112/120；早先"config N256 会改变 host 队列"的猜测已根据源码撤回（tile_n 在此 scope 只传给 S1 launch）。

## 4.5 b136 开发链的增量对照（模块内 README/validation.json 留档）

`m32_m48_m64` 路径是在 b136 上逐增量开发的（每步 12 平衡配对、quick screen、null 对照留档，"reverse"后缀表示真实 graph 构造顺序）：

| 步骤 | 对照 | 结果 |
|---|---|---|
| 加入 M48 路径 | vs 原最快 ticket-fixed 单 M64/N512 | 快 3.806% / 3.743%（ba/ab） |
| 相邻对照 | vs 前一个 M32/M64 botharbd 候选 | 快 1.477% / 1.552% |
| last16 quantization 跳空 | — | **否决**：未过双向增量保留规则（producer/barrier/K-loop 指令文本保留，guard 未进） |
| last16 producer+consumer epilogue guards | vs 前一 M48 候选 | B/A 0.9951 / 0.9958（保留） |
| middle16 producer+consumer guards | vs 前一 last16 候选 | B/A 0.9986 / 0.9988，双向性能判据未达，**用户决定保留**；no-skip 对照 +0.079%/−0.023%，CI 全跨 0——小收益未解决 |

整条链全部保持 202 VGPR / 106 SGPR / 零 spill / 零 scratch / LDS 45056 B（b136 口径），16 探针、graph512、全 8 rank GPU/offline ISA 一致。

## 5. commit 内容、身份与复现

**commit 改动**（`git show f7a9651 --stat`）：`mega_moe_config.py`（`SBM64_PATHS` 从旧的 7 点实验入口改为 13 点全部 `m32_m48_m64`）、`mega_moe_m3.py`（新增 `sbm64_m32_m48_m64` 分支 import 与 tile_n=256）、`mega_moe_stage2.py`（`sbm64_rollout` assert 扩到 13 点、empty-half skip 对 120/136/144 由 rollout 触发）、`op_tests/test_mega_moe_m3_config.py`、新模块 `sbm64_m32_m48_m64/`（README/`__init__`/stage1/validation.json）与 `docs/mega_moe/sbm64_20260915/` 全套审计文档。该模块从"experimental b136"升级为 retained 默认。

- fork：`mingg26/aiter` 分支 `ming-amd-m3-megamoe`，commit **`f7a96515aa8e3b7c1e41506c37d49b78651ce8ad`**；可 push 到该 fork，不推 ROCm upstream。
- 同步生产树：`megamoe-s1-1k-prod/aiter` 分支 `s1-unified-production`，commit **`6dfd36cef6cbde7c32b1355e48f265a7fbd49e0d`**。
- 提交身份另录于 `megamoe-s1-1k-prod/bench/checkpoints/sbm64_retained_20260915/published.json`；报告与审计在 commit 内 `docs/mega_moe/sbm64_20260915/`。
- 历史/手动入口保留但不作默认：`sbm64_full`、`sbm64_m16_dma`、`sbm64_m16`（旧实验模块）。13 点默认只用 `sbm64_m32_m48_m64`。
- 旧冻结 harness pin 的是合入前 HEAD/文档哈希；复跑必须新建实验目录与新 source pins，不得绕过门禁或改写旧 manifest。

## 6. 机制备注（S2 整空半块 skip）

均匀 top4 路由下全体专家平均 token 数为 `batch/4`：b120/136/144/152 分别为 30/34/36/38，正好跨过 BM32 线。专家收 30 token 时 SBM64 拆给 S2 的两块是 `[30,0]`（后块可跳）；34 token 变 `[32,2]`（后块有 30 行 padding 仍须算整 BM32）。skip 省掉空块的 B 读取/MFMA/epilogue，但 queue claim、遍历、band padding 保留；同专家有效块仍要读 B，空块比例不能直接换算成 HBM 字节比例。"省块≠最慢 rank 等比例省时"（b136/144 收益小于空块比例）尚无 PMC 级解释。
