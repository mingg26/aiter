# md_ming — MiniMax-M3 AMD (MI350X/gfx950) 优化知识库

本目录保存当前结论与必要历史证据。**新对话先读 [redo.md](redo.md)**，再读 [summary_megamoe.md](summary_megamoe.md) 和 [report_s2.md](report_s2.md)。2026-09-15 收尾：b120–184 已测 M64 候选补入默认；b112 的最新诊断未采用。

历史性能表保留测量当时的 commit；当前双仓库版本、完整默认账本和复现命令统一见 redo。`.scratch` 实例只在复现入口保留，旧根目录 handoff 无需重读。

## 仓库归档与恢复

整个目录随 `mingg26/aiter` 的 `ming-amd-m3-megamoe` 分支保存于仓库根目录 `md_ming/`。文中的工作区相对路径以 `/mnt/shared/homes/ming/msa/md_ming` 为基准；恢复时将整个目录复制到工作区同级位置。GPU复现依赖仍按 `redo.md`，不会随文档上传。

## 入口

- 做 MoE 工作：先读 `summary_megamoe.md`，再按需要深入 report/option/trick。
- 做 MSA attention：decode 读 `report_msa_decode.md`，prefill 读 `report_msa_prefill.md`。
- 做任何测量前：必读 `tool_measure.md`（kernel 级）或 `tool_e2e.md`（serving 级）。

## MegaMoE（aiter mega_moe_m3，EP8 MoE kernel）

| 文件 | 内容 |
|---|---|
| `redo.md` | 重启入口、当前版本、默认配置审计、GPU 复现命令与源码锁定 |
| `report_s2.md` | 最新 S2 默认补齐、九点完整 forward 表、b112 诊断终态 |
| `summary_megamoe.md` | 入口总览：5-kernel 流水线、逐档加速比轨迹、S1/S2 成本模型、读数四规则、部署事实、未决问题清单 |
| `report_sbm64.md` | S1 历史定稿 + 当前 S2 选择：13 batch 选 `m32_m48_m64`/N256、b72 保持 SBM32、S2 选择、26 轮 312 对性能表、验证证据链（104 ISA / 5371 selector / 9 CPU / 仅 4 批 GPU 复验） |
| `report_sbm128.md` | b200–256 SBM128 八路径/三档（`b13717ca`）、ticket 修复、资源数 |
| `option_megamoe.md` | 53 项配置开关权威表（S1 40 + S2 19 + p2p 1，按 f7a9651 实数），含最新默认补丁 |
| `trick_megamoe.md` | 机制级结论：sort-block 规则、S2 计算量与调度的取舍、NT **单向**判据（只用来排除，不用来预测收益）、task 阈值模型、split-K 恒亏、VGPR/LDS 结构、FlyDSL 坑 |
| `report_s2_local_reduce.md` | S2 本地预归约终局（8k/2k 胜、小 batch 净亏未解）、8 条设计不变量 |
| `trick_vmcnt_asyncmark.md` | 已回退技术档案：asyncmark 配方、两轮实测无净收益、重开条件 |
| `plan_megamoe_open.md` | 未决/待办：TODO_k3 十二项、PR 清理计划、未解决问题清单 |

## MSA 稀疏注意力

| 文件 | 内容 |
|---|---|
| `report_msa_decode.md` | decode 线终态：六 PR 栈（#446–#451）、split chooser、A-B-A 终表、e2e 终验、BRANCH_SWEEP 两表、union 探针 |
| `report_msa_prefill.md` | prefill 线终态：Q-outer+XCD 方向翻转、五 commit 阶梯（累计 1.676x）、fp8 规律、E2E +1.112% |
| `report_m32_membership.md` | M32/M64 成员分派（整条线已被 XCD swizzle 超过，分支勿再动）、交叉点 b64–b128 |
| `trick_xcd_swizzle.md` | XCD 硬件事实（mod 8 旋转、padding 硬要求）、swizzle 数字、DEP 拍平序坑 |

## 跨线机制与平台

| 文件 | 内容 |
|---|---|
| `trick_fused_reduce.md` | split 归约正确性协议（vmcnt(0)→barrier→atomic 发布链）、gfx950 内存语义、graph replay 免清零三法 |
| `trick_mi350x.md` | 平台尺子：7.3 TB/s 峰值、MALL 1.15x、边际成本锚点表、驻留公式 |

## 测量方法论与 e2e

| 文件 | 内容 |
|---|---|
| `tool_measure.md` | kernel 级测量规范：paired A/B、rocprofv3/PMC/ATT、判废规则（含严格协议 ±0.2% null 等价带与 sclk/power 门限）、两条数值地板（3 µs / null 0.2%）、多 rank collective 循环禁令、无人值守四规则、测量事故档案 |
| `tool_e2e.md` | serving e2e 测量规范：TPS 公式、admission barrier、warmup 污染、三大测量毒药 |
| `tool_root_cause.md` | 根因定位纪律：exposed-bubble 归因、判别性实验、floor share ≠ marginal cost |
| `report_dep8_e2e.md` | DEP8+MTP7 集成终态：负载口径、TPOT 9.4337 vs 10.7530ms（−12.27%）、Mori heap、MXFP8 NaN 修复、高并发 segfault |

## probes/ —— 默认审计、复现与测量工具

默认审计与复现准备脚本见 [redo.md](redo.md)。另有三个带宽脚本（`kvo_bw_floor.py` / `spike_cache.py` / `kvo_scope_ab.py`）原在
`.scratch/diag/`，但它们被 trick 文件指定为**权威尺子**，不带过去那些"以 XX 为准"的说法
就落空了，所以单独抽出来随本目录迁移。只依赖 torch + triton(+gluon)，详见
`probes/README.md`。**换机器后第一件事是跑 `kvo_bw_floor.py` 重新定标带宽尺**——
`trick_mi350x.md` 的带宽表是本机实测值，不是规格值。

## 源文件对照（迁移时不需要带的）

5 份 handoff（handoff_megamoe / handoff_kv_outer / handoff_decode_msa_pr / handoff_dep /
HANDOFF.md）与 redo.md、BRANCH_SWEEP.md、report_codex.md 等已全部抽取精华到上表，
原件不迁移。其中 handoff_megamoe / redo / report_sbm128 三份的正文已随 f7a9651 归档进
aiter 仓库，任何机器 clone 该 commit 即可得。
