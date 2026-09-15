# MegaMoE 未决事项与待办计划

> 范围：MegaMoE 尚未执行的清理/重构待办（TODO_k3 十二项）、PR 级代码清理计划（plan_clean_megamoe）、未解决的问题清单。
> 有效性基准：TODO_k3 记录于 2026-09-11（基于工作树 `bab7de481` + 未提交修改）；plan_clean_megamoe 基于分支 `s1-unified-production` HEAD `7e433ab4e`。两者均**早于** f7a96515（2026-09-15 冻结），执行前必须逐项对照当前生产树核对是否已被吸收 [待复核]。
> 来源：`TODO_k3.md`、`plan_clean_megamoe.md`、`handoff_megamoe.md`、`summary_megamoe.md`。

## 0. 本次停点（2026-09-15）

先读 [redo.md](redo.md) 与 [report_s2.md](report_s2.md)。b120–184 已补入默认；b112 保留 BM32 constant N/K。b112 M16/shared-setup、shared/routed隔离、512 CTA、epilogue M16 均已完成，未采用，别重新当作未测方向。

下一候选只有讨论、未动手：**b112 在640 CTA下试 leader empty drain**，与K3先审方案、一次一个实验/形状。用户当前只要求收尾文档，没有自动启动该实验。

以下清理计划为旧档案，逐项复核后再用。

## 1. TODO_k3：S1 实验开关清理清单（12 项，均未执行）

分类：**6 项计划删、2 项待测、2 项复查、2 项重构**。"计划删除"≠ 已证明所有 batch 无收益。

### 1.1 计划删除（D1–D6）

| 项 | 开关 | 内容与依据 |
|---|---|---|
| D1 | `skip_launch_barrier` | 删实验开关及分支，保留 False 路径的同步语义。小 batch 无稳定额外收益。**不能把"删开关"实现成"删 barrier"**，也不能据此认定 preplan 已替代全部握手 |
| D2 | `scalar_tile_row_base` | 删 `readfirstlane` 处理 tile 行地址的实验分支及配置/传参/变体名后缀。b32 结果约 0.88239 无稳定收益（只覆盖已测配置） |
| D3 | `joint_work_flags` | 删 work ticket 与标志联合发布的实验分支，连带检查 `_jwf1`、LDS 视图宽度及读写端配套。b32 约 0.88334 无稳定收益；删后保留原 ticket/标志同步关系 |
| D4 | `unroll_a_pingpong` | 清理固定两拍展开路径。删的是额外展开实现，**基础 A 双缓冲保留**。b32 结论不能外推大 batch |
| D5 | `packed_a_scale` | 清理 A scale 预打包路径（`_pas1`、`AScaleLoader._stage_packed` 及配套布局）。b32 单开曾略好，叠 `use_nt` 后无稳定额外收益；不是"所有配置都无效"的证据 |
| D6 | `split_a_lds` | 检查独立 ping/pong LDS 对象、访问偏移及 epilogue 内存复用。小 batch 无独立否决结论；**大 batch 正在使用，须先验证替代实现再删** |

### 1.2 依赖链与先决条件（执行顺序的硬约束）

- **D4/D5/D6 在 b8192 冻结配置及当前工作树 EP8 b512/b1024/b2048/b4096 配置中均为 `True`**——删前必须做大 batch 消融，复测范围不能只写 b8k，还要查 EP4 等其他调用路径。
- `split_a_lds` 依赖 `unroll_a_pingpong` / `async_a_copy` / `mfma_amajor` 及合法几何；T1/T2 的待测路径又依赖 split/unroll。**顺序：先完成 T1/T2 和大 batch 消融，再删它们依赖的实现。** packed scale 可独立消融；split/unroll 按合法组合比较。
- 每项记录：正确性、生成代码资源占用、完整执行路径性能；不能仅凭 b32 数字决定全局删留。结论区分"无稳定收益 / 明确退化 / 组合不支持 / 尚未测量"。

### 1.3 待测（T1/T2）

- **T1 `fp8_b_waitcnt`**：调整等待 A DMA 时允许保留的 B/scale VMEM 未完成计数；当前关闭 ≠ 已充分测过。依赖 `split_a_lds` + `pipe_weights`，继承 split 对 unroll/async/MFMA 几何的依赖。固定其他条件比较开/关，检查实际 wait 指令、正确性及完整路径性能。
- **T2 `prefetch_a_operand`**：提前把下一份 A operand 从 LDS 读入寄存器；要求 `split_a_lds`、`unroll_a_pingpong`、`mfma_amajor`。先独立比较，再视结果与 T1 组合；关注寄存器增加是否抵消等待减少。

### 1.4 复查（R1/R2）

- **R1 `preplan_waves`**（保留）：虽存在 `Stage1Config`，调参应归 **A 类（TopK/quant/preplan）**——它决定融合 quant/planner CTA 的线程数并影响 quant 的 grid，不只是规划 CTA 宽度。当前生产用 4；b32 的 2/4/8 测量不足以定所有 batch。复查时保持整条 route → quant → S1 状态一致，比较完整路径。
- **R2 `use_tile_resource`**（保留）：控制 GEMM1 epilogue 的 tile 局部 buffer resource，**不是 cache 开关**。更正默认值表述：**EP8 b16–b256 shared 路径为 False 且 guard 禁止打开；b512–b8192 要求打开**。历史 EP4-512 最快候选用 off 可作复查线索；若要放宽 guard 属实现扩展，不能把 guard 拒绝记为性能失败。

### 1.5 重构（F1/F2）

- **F1 `grid_mult` 自动推导**：去掉手动调节入口，按 kernel 资源占用/occupancy 和设备信息推导 persistent grid。区分"发射 CTA 总数"与"每 CU 可同时驻留数"；persistent 不要求 CTA 数 = CU 数。同时处理 planner/dispatch/GEMM CTA 分配与 preplan 约束，先保持当前有效配置行为。不要把某形状的 ~88 KiB LDS 占用写成所有 batch 的常量，不要混入 S2 的 `queue_grid_mult`。
- **F2 `work_shards` 随硬件 XCD 数推导**：当前**硬性要求 `WORK_SHARDS == 8`**，硬件查询是待实现目标。需可靠的设备拓扑来源，并检查队列分配、ticket 映射、XCD 编号解码和 N-panel 分配中的所有 8/位掩码假设。**不能只删 assert 就声称支持其他 XCD 数**；非 XCD 路径处理也需明确。

### 1.6 明确保留（不在删除范围）

`payload_tile_ready`、`payload_tile_publish_early`、`pipe_weights`、`swizzle_a`、`b_hoist`、`ascale_prefetch`（后四项来自上游 MegaMoEV2 `97d0c6e4c`，默认 True）。

## 2. plan_clean_megamoe：PR 清理计划

> 目标：`megamoe-s1-1k-prod/aiter`（分支 `s1-unified-production`，基线 HEAD `7e433ab4e`）清成 PR。**PR 边界 = 现有 20 commits / 17 文件 / +7427-10，不扩大**：mega_moe_m3 包 14 文件 + combine 两文件（+151/+41，对存量调用方零影响）+ test_mega_moe_m3.py（+344）。行号均基于 `7e433ab4e` [待复核：与 f7a96515 的关系]。

### 2.1 直接删除项（无争议：默认关/恒值/无调用方，const_expr 裁剪下 ISA 不变）

- 文件级：`gemm1_blds_experiment.py` 全文件（无 import、无许可头）；连带 `gemm_util.py` 的 `SiluQuantEpilogue.waves_along_m`。
- config 字段及整条透传链：`skip_launch_barrier`、`joint_work_flags`、`fp8_b_waitcnt`/`prefetch_a_operand`/`scalar_tile_row_base`（从未开启）、S1/S2 `schedule_audit`（半完成调试口，自启即崩）连带 `DispatchSlot.SCHEDULE_AUDIT`、`tile_k`（assert 锁死 256，改常量）、`spatial_partition`（恒 402）、`bf16_lds`（恒 False；local_reduce 的 BF16 LDS 路径独立保留）。
- S2 死链：`has_pad`/`i32_kpad`/`i32_npad`（恒 0）、BM16/BM128 死链（收紧白名单为 {32,64}）、非 FP8 A 死分支。
- 死属性/恒真构造参数、kernel 名后缀卫生（`_shu1`/`_shna2`/`_shsmall1`/`_scratchfix1`，只影响 JIT cache key）。
- **2026-09-11 用户修订**：`pipe_weights`/`swizzle_a`/`b_hoist`/`ascale_prefetch` 四个上游开关**撤回删除建议，本次 PR 保留**。

### 2.2 需拍板项（决策项）

1. **env 调参钩子全套**（7 个 `M3_MEGAMOE_*` + `_P2P_QUANT_ENV`）：使命已完成，建议整套删，连同 m3.py 的 env 旁路检查。
2. **fp8 P2P quant 链**（~200 行跨 4 文件）：config 注释已宣判它把 relL2 从 0.0027 拉到 0.0266，生产恒 none；但它曾是 fabric 定价实验臂，**删前确认定价实验已终结**。
3. **`skew_cu`/`persist_strided`**：上游继承，bucket 512–2048 非队列路径仍可达但 M3 从未实测；`band_m>1` 时 `skew_cu` 被静默忽略。建议保留 + `assert not (skew_cu and band_m>1)` fail-loud。
4. **`enable_std_moe`/`zero_copy`**（mori-parity 残留，面大）：建议独立 PR，本 PR 不动。
5. **`direct_fixed_slot`**（~190 行）：硬编码 npes==8/epr==48（DSV4 几何），M3 恒 False 不可达。PR 声明 M3-only 可删。
6. **`shared_xcd_schedule`（恒 True）与 `mega_scheme`（恒 "fixedslot"）**：冗余构造参数，建议删除或内部推导。

### 2.3 验证策略（行为保持的证明）

1. 清理前 pin ISA 基线：生产档位（EP4 256/512/1024/8192 + EP8 256/8192）全部 kernel 的 code object 哈希（含资源字段）。
2. 清理后重出 ISA，与基线**逐字节对比**（kernel 名后缀会变，按内容哈希比）。
3. `op_tests/multigpu_tests/test_mega_moe_m3.py` 通过；CI 口径 `ruff check` + `black --check`；pre-commit hook 本地跑。
4. 至少一轮 paired A/B 计时门禁证明无性能回退。

旧 selector 缺 preset 的问题已修复。当前再补齐 b120–184 M64；使用新默认账本逐字段审计，勿用旧 best-config 文件作为当前全档门禁。

## 3. 未解决问题清单（长期有效，均未定位）

| 问题 | 现状 |
|---|---|
| **S1/S2 固定截距** | `S1 ≈ 46.9 µs + 字节/6.42TB/s`、`S2 ≈ 30.7 µs + 字节/8.94TB/s`（残差 ≤0.12/0.66 µs），截距来源未识别。已排除：shared L13 权重读取（在斜率里）、tile 形状/tile 数/占用；已知强相关：dispatch CTA 数量 |
| **A 侧重读** | S2 读 A2 重复 **48×**（453 MB demand vs 理想 9.4 MB，per-XCD L2 地板 8×=75 MB）；S1 读 A 重复 **24×**（453 MB demand，地板 151 MB）。两项各 453 MB，与整层"有用"流量（~0.9 GB）同量级，是已知最大字节浪费。b512 想要的 `BM=64+BN=256` 被 64 KB workgroup LDS 上限挡住（累加器正好 64 KB） |
| **`dcu=64 @ b512` 正确性 bug** | 未修。`all_local` 倾斜探针下 rank 7 连续 256 次 graph replay 后与 eager 不逐位一致；`dcu=64` 时 `producers_per_destination = 8 = npes`，8 producer 抢 1 chunk，与 8k dispatch 倾斜 bug 同族 |
| **middle16 增量收益未定论** | f7a96515 中 last16 guard 已保留、middle16 由用户拍板保留，但无跳过对照显示 +0.079%/−0.023%（CI 含零），稳定收益未证明 |
| **b104/b112 低于趋势** | 加速比 1.368/1.375（b120 已 1.446），sbm=64 没完全补平，原因未定位 |
| **local_reduce 小 batch 退化** | 256/512 慢 +4.6%/+3.7% 未解决，方案未进生产；详见 `report_s2_local_reduce.md` |
| **`route_masks` 单缓冲隐患** | 未修：quant 远端写 peer 掩码的发布顺序只保护下一轮不保护本轮 S2；修法：双缓冲或迁到 S1 rendezvous 之后 |
| **S1 asyncmark 移植半成品** | `kernels_s1async/` 是中途状态、未经验证勿用；`gemm1.py:309` 手工 waitcnt 值 10→18 的隐患未修；S1 vmcnt 审计显示值层面近最优，唯一系统点是 ISA:1455 的 alias 保守 vmcnt(2)（上限 5–20 µs） |
