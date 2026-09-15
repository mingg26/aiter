# MegaMoE M3 开关与 batch 配置清单

- 范围：`mega_moe_m3` 生产实现的全部配置开关（Stage1 / Stage2 / 顶层 p2p_quant）、逐 batch 取值、依赖约束与 EP4 默认。
- 当前默认以 [summary_megamoe.md §5](summary_megamoe.md#5-当前默认配置2026-09-15-收尾) 和 [redo.md](redo.md) 为准；下面开关主表保留历史三档对照。
- 来源：`option_megamoe.md`（47KB 原版，本文为其 f7a9651 重基线版）、`mega_moe_config.py` / `mega_moe_m3.py`（f7a9651 实读核对）。

## 0. 阅读口径

主表完整覆盖 `Stage1Config` 35 项、`Stage2Config` 17 项、顶层 `p2p_quant` 1 项，共 **53 项**；不仅列 bool，也列影响性能的整数和枚举。**f7a9651 的实际字段数更多**：Stage1Config 40（+`xcd_home`/`shared_xcd_home` 见 §0.6，+`shared_packed_heads`/`sbm128_path`/`sbm64_path` 见 §1）、Stage2Config 19（+`xcd_home`/`shared_xcd_home`），顶层不变。

- **b16 / b256 / b8192 是每个 rank 的 token 数**，不是 EP 组总 token 数。主表对应 EP8、H6144、I3072、128 experts、top4、TP1 shared，且 `tokens == max_tok_per_rank`。
- 主表数值来自 `bench/dep8_best_configs.json` 的 `entries.16/256/8192`，是已验证、已记录的最佳观测配置，**不是全局最优证明**；当前默认更新见 §1，不直接用旧 JSON 判定最新配置。
- `开` = `True`，`关` = `False`；`†` = 字段保留该值，但当前路径不执行对应分支。无 `†` 不表示已有逐项性能消融。
- `默认` 列仅指 dataclass 字段默认值；`必填` 表示由 selector/调用者提供。底层 compile 函数可能另有默认值，不能混用。
- `上游` = 原版 DSV4 MegaMoEV2 已有该字段；`M3` = M3 分支新增。字段继承不代表实现从未修改。

**这些参数并非只影响 S1/S2 两个 kernel。** `Stage1Config` 是配置的归属位置：`preplan_waves` 同时影响 TopK、输入 quant 和 S1；`local_reduce` 跨 S2 与 combine；shared 融合和 `p2p_quant` 也有跨阶段约束。完整执行链为 `TopK → 输入 quant/preplan → S1 → S2 → combine`。

## 0.5. NT cache 策略：可直接算出「一定不要开」的那半边

**判据是单向的**：`R_B == 2` 的档可以直接算出来、不必试（稳定亏）；`R_B == 1` 只是"可能赚"，
仍须实测。机制与实测点见 `trick_megamoe.md` §2（该节为本判据的权威表述）。
定义 R_B = 一份可复用 B panel 被几个输出 tile 读：

```
R_S1 = ceil( (8·T·topk/E) / sort_block_m )    # 每 expert 行数 / S1 M tile 高度
R_S2 = sort_block_m / block_m                 # 与 T 无关（S1 把每 expert 补齐到 SBM 整数倍）
R_S1 == 1 → b_nt 可以是 2；R_S1 == 2 → b_nt 必须 0
R_S2 == 1 → use_nt 才有可能赚（须实测）；R_S2 == 2 → use_nt 一定关
```

完整形式是"拿 B 复用换 A 驻留"：`开 NT ⟺ (R_B−1)·B_total·1[S_B ≤ C_L2] < (R_A−1)·A_total`；本模型 B_total/A_total ≈ 100~200，右边赢不了，塌缩成 `R_B == 1`。

实测：符号预测 7/7 正确并预先预测成功一次（b64 把 S2 `block_m` 32→64 使 R_S2 2→1，`use_nt` 由亏 5.3% 变赚 0.74%；反向决定性验证：b32 只把 `sort_block_m` 32→64，`use_nt` 由赚 0.8% 变亏 4.4%）。

下表最后一列是 **f7a9651 的实际取值**（`mega_moe_m3.py:423` 为 `use_nt=tokens in (32, 64)`），
不是"R=1 就该开"的推论——注意 b16 同为 R=1 但实际是关（实测持平，没有理由开）：

| 几何 | R_S2 | f7a9651 实际 `use_nt` |
|---|---:|---|
| SBM32/BM32（b16） | 1 | **关**（实测持平） |
| SBM32/BM32（b32） | 1 | 开（实测 +0.8%） |
| SBM64/BM32（b128、b256） | 2 | 关 |
| SBM64/BM64（b64） | 1 | 开（实测 +0.74%） |
| SBM128/BM64（b8192） | 2 | 关 |

**幅度不由 R 决定，机理未定位**：R=1 收益随 batch 衰减到零（b32 +0.8% → b128 0 → b256 0），R=2 损失随 batch 增大（−5.3% → −10.6%）。**用它决定开关，不要用它预测收益。** 复用是启发式：同 B 的 C tile 拿相邻 ticket 但没有 barrier/顺序保证，队列取空后会偷别的 XCD；S1 小 batch 上连续 m_index 对应完全不同的 B。

## 0.6. XCD home seed 开关

`Stage1Config.xcd_home` / `shared_xcd_home`、`Stage2Config.xcd_home` / `shared_xcd_home`，默认全 `True`。语义是**只换 home queue 的种子**（物理 XCD / CTA 序号），队列数 8、band decoder、ticket 记账、epoch 协议全不变；每个 CTA 无论如何遍历全部 8 个队列，只改起点；默认时 codegen 中性（已实测）。

**不能用 `xcd_schedule=False` 代替**：S1 里 `small_xcd = xcd_schedule and not payload_tile_ready`，关掉会掉进另一条工作分解分支，且 `assert BAND_M == 1 or payload_tile_ready or xcd_schedule` 会拦下 band_m=4。实测四个全关比全开慢 0.94%（b32），全开即最优；用途是大 batch 上分别调 routed/shared。

## 0.7. 已否决开关（不要重试；一行清单）

b32 基线区间 0.88129~0.88455、噪声底 0.37% 下：`scalar_tile_row_base`、`joint_work_flags`、`unroll_a_pingpong`、`preplan_waves=2/8` 均在噪声内或更差；`packed_a_scale` 不与 `use_nt` 叠加；`work_shards≠8` 被 guard 拒绝；`split_a_lds` 要 N256 几何、`fp8_b_waitcnt` 传递性被挡；`payload_tile_ready`/`payload_tile_publish_early`/`payload_chunk_rows`/`use_tile_resource` 在 b16~b256 被 shared guard 硬挡（b8192 专属）。另：`sort_block_m=64` @ b512（请求 −29% 但 A 的 DRAM +7.7%，否决）、b512 `BM=32` +7.85%、`BK=256` +1.95%、`qgm2/8` 更差、BK=128 四档全输、`b_stages=1` 叠 BN256 是灾难。更早的大面积否决清单见旧 `summary_megamoe.md` §9（本文件不重复）。

## 0.8. 仍然完全没扫过的维度

- S2 `block_n`（现 128/256）与 `block_k`（现 256/128）在 EP8 大档从未系统扫过；动它们要放宽白名单。
- S1 `tile_n` 只在 b16 扫过一个替代点（256，被否）；**`tile_k` 不可调**（`assert A_K_STEP_BYTES == 256`）。
- S2 `block_m` 只试过 16（routed 数值错 rel 0.951）、32、64；`SBM % BM != 0` 直接 raise。

## 1. 当前默认补丁（2026-09-15）

完整 batch 表集中在 [summary_megamoe.md §5](summary_megamoe.md#5-当前默认配置2026-09-15-收尾)，避免多处漂移。37 点完整字段见 [默认账本](probes/megamoe_defaults.json)，CPU 审计命令见 [redo.md](redo.md)。

1. S1 SBM64 仍为 64、104–192 步8，共13点，N256/M32-M48-M64；SBM128 仍为200–256八点。
2. **b72 已换 SBM32/N256/M16-M32**；早期 SBM64 失败不代表继续用 N512。
3. **S2 120–192：BM64/N256/K256、W8、256 CTA、M32/M48/M64、fixed N24。** b64 继续 BM64/N128/W4；b104/112 继续 BM32/N256/W4，112 有 constant N/K 地址。
4. S2 NT 仅 b32/b64；BN128 仅小档 b64/b256。b64 的旧 W4/LDS 限制不能外推到 W8 big-CTA。
5. S2 M16、epilogue M16、512 CTA 在 b112 未证明整体收益，保持关闭。leader drain 保留范围见总表。

三个新 Stage1 字段：

| 参数 | 默认 | 含义 |
|---|---|---|
| `shared_packed_heads` | 关（融合 shared 时 selector 开） | 队列 head 的低 16 位 shared、高 16 位 routed，复用 preplan 每 forward 清零的 i32 head；先扫 shared 再发 routed |
| `sbm128_path` | `"generic"` | SBM128 专用实现选择（`m16`/`tiered96`）；仅 `SBM128_PATHS` 内且 scope 满足时由 selector 发出，**不从 bucket 推断** |
| `sbm64_path` | `"generic"` | SBM64 专用实现选择（`m32_m48_m64`；`full`/`m16_dma`/`m16` 为历史手动入口） |

## 2. 调参分组与依赖约束

按实际优化对象分六类（配置在 Python 中属于哪个 dataclass 不决定分组）：

| 大类 | 主要调整内容 | 初筛固定边界 |
|---|---|---|
| A. TopK / 输入 quant / 预规划（§3） | preplan wave 数、计数/分组、planner 地址预取 | 固定路由语义、量化格式、S1/S2 计算配置 |
| B. S1 dispatch / GEMM1（§4） | tile、执行规模、A/B 流水、cache 策略、payload 就绪 | 保持 A 类规划模式；M tile 改动的接口联动随候选记录 |
| C. S2 GEMM2 / 调度（§5） | M/N/K tile、persistent/XCD 队列、加载流水 | 固定 S1 输出布局、local-reduce/shared 模式 |
| D. 跨阶段通信 / local-reduce（§6） | local-reduce、XCD-local staging、传输格式 | 初筛保持数值契约；配套合法 S2 调度与 combine 路径 |
| E. Shared 融合 / 调度（§7） | shared 两级融合、XCD 调度、tail/early2/jointtail | shared 数学计算与最终加和始终保留 |
| F. Combine（§8） | 本地结果预取、combine grid/waves | 固定 S2 写回格式、路由权重已乘位置 |

方法：每 batch 先冻结完整基线 B0 → 逐类初筛（只改一类）→ 每类保留 B0 + 1–2 候选 → 先验证强关联组合（A×B、B×C、C×D、B/C×E、D/E×F）→ 逐步合并 → 组合后回看 → 最终同口径确认。收益不相加、不跨 run 比裸延迟。

初筛必须一起处理的依赖：

| 关联项 | 处理方式 |
|---|---|
| `split_a_lds` ↔ `unroll_a_pingpong` / `async_a_copy` / `mfma_amajor` / 合法 N/K/wave | 作为 B 类内部合法组合生成候选 |
| S1 `sort_block_m` ↔ S2 `SBM`/`block_m` ↔ shared row table | M tile 改动必须同步接口；若改 S2 几何，标为 B×C 联合候选 |
| `preplan_waves` ↔ route histogram / quant planner / `grid_mult=1` / shared | shared 实现要求 preplan，不能保持其他条件直接关掉 |
| chunk / tile-ready / early publish ↔ S1 M tile / shared 几何 | early publish 不能脱离 tile-ready，chunk 须整除 S1 M tile |
| `prefetch_b_before_a` ↔ B pipeline / async A / compact | 需保留相应加载前提 |
| local-reduce / XCD-local ↔ S2 persistent / band / XCD ↔ combine | D 类切换可能改变 C/F 类合法配置 |
| `shared_schedule=early2/jointtail` ↔ 指定 S2/shared 几何 | 先在已支持形状内调；改 S2 tile/grid 需重新验证 |
| combine 本地预取 ↔ shared 开 / local-reduce 关 / 小 batch | 只在支持该分支时比较；b8192 local-reduce 路径不可直接打开 |
| `p2p_quant` / `bf16_lds` 等数值表示变化 | 同数值验收标准；FP8 combine 不在已验证主路径，另列精度实验 |

## 3. A 类：TopK / 输入 quant / 预规划（6 项）

字段存放在 `Stage1Config`，实际影响 TopK histogram、输入 quant 中的 planner、S1 消费的路由元数据。

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `preplan_waves` | M3 | 0 | 4 | 4 | 4 | 0 关闭预规划；2/4/8 为融合 quant+planner 的每 CTA wave 数。开启后 TopK 统计 histogram，quant 内完成 dispatch 规划 |
| `external_grouping` | 上游 | 关 | 关 | 关 | 开 | 按 expert 分组的工作分给额外 producer CTA；开 preplan 时这些 CTA 位于 quant kernel 内 |
| `external_counting` | 上游 | 关 | 关 | 关 | 开† | 非预规划路径把路由计数分给 producer CTA。三组最佳均启用 preplan，计数已在 TopK 完成，该分支被绕过 |
| `padding_uniform_srcmap` | M3 | 关 | 开 | 开 | 开 | padding 元数据写入前明确 srcmap 表地址是 wave 一致值，减少重复 descriptor 处理 |
| `count_uniform_matrix` | M3 | 关 | 开 | 开 | 开 | 明确 planner 的 count matrix 地址 wave 一致，减少读计数时的 descriptor waterfall |
| `row_base_prefetch` | M3 | 关 | 开 | 开 | 开 | 提前批量读取各 peer 的 row-base 表指针，缩短地址依赖链 |

实现：`routing.py`、`quant.py`、`preplan.py`、`dispatch.py`（均在 `aiter/ops/flydsl/kernels/mega_moe_m3/`）。

## 4. B 类：S1 dispatch / GEMM1（29 项）

S1 含 payload 通信、GEMM1、SwiGLU 和中间量化。A = 输入 activation，B = expert 权重；S1 M tile 是路由后按 expert 排列的行，不等于本地 batch。

### 4.1 Tile 与执行规模（10 项）

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `sort_block_m` | 上游 | 必填 | 32 | 64 | 128 | S1 每 M tile 行数，也决定排序/padding 粒度；S2 的 SBM 跟随它。f7a9651 起按档选择（SBM64/SBM128 档见 §1） |
| `tile_n` | 上游 | 必填 | 512 | 512 | 256 | GEMM1 N tile 宽度，覆盖 gate/up 投影；影响 A 重读次数和 LDS 用量。f7a9651 起 SBM64/SBM128 档为 256 |
| `tile_k` | 上游 | 256 | 256 | 256 | 256 | 每次 K 循环宽度；当前 S1 实现断言固定 256 |
| `num_waves` | 上游 | 必填 | 8 | 8 | 8 | S1 每 CTA wave 数，每 wave 64 线程 |
| `grid_mult` | 上游 | 必填 | 1 | 1 | 1 | 整个 S1 grid 的 CU 数倍数（含 planner、dispatch、GEMM CTA）；预规划要求 1 |
| `num_dispatch_cu` | 上游 | 必填 | 48 | 32 | 96 | 分给 payload dispatch 的 CTA 数；增加它也减少同 grid 的 GEMM CTA。f7a9651 小档：48（b256 为 32） |
| `waves_per_eu_hint` | 上游 | 2 | 2 | 2 | 2 | 给编译器的 waves-per-EU 资源提示，不是实际 occupancy 保证 |
| `work_shards` | 上游 | 8 | 8 | 8 | 8 | S1 工作队列分片数；当前 XCD 调度要求 8 |
| `xcd_schedule` | M3 | 关 | 开 | 开 | 开 | 按实际 XCD 分配工作，改善相同权重/N 面板的缓存复用 |
| `band_m` | M3 | 1 | 4 | 4 | 4 | 一组内相邻 M tile 数，调整 M/N 工作顺序以复用 B；1 表示不分带 |

### 4.2 A/B 加载、scale 与 LDS（12 项）

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `mfma_amajor` | 上游 | 必填 | 开 | 开 | 开 | 偏向复用 A operand 的 MFMA 组织顺序；不改数学结果 |
| `async_a_copy` | 上游 | 必填 | 开 | 开 | 开 | A 从全局内存异步直接搬进 LDS，与当前块计算重叠 |
| `use_tile_resource` | 上游 | 必填 | 关 | 关 | 开 | S1 epilogue 输出用 tile 局部 buffer resource；不是"开启 tile 缓存"。EP8 b16–b256 shared 路径被 guard 禁止开 |
| `b_nt` | 上游 | 必填 | 2 | 0 | 0 | B 权重 load 的 cache modifier：0 无、1 SC0、2 NT、3 SC0+NT；仅作用于 B 权重加载。判据见 §0.5 |
| `pipe_weights` | 上游 | 开 | 开 | 开 | 开 | 计算当前 K 块时预取下一块 B 权重 |
| `swizzle_a` | 上游 | 开 | 开 | 开 | 开 | A 在 LDS 按行 XOR 布局，减少 bank conflict |
| `packed_a_scale` | M3 | 关 | 关 | 关 | 开 | A scale 写入 LDS 时预拼成 MFMA 所需 i32 格式；不压缩数据大小 |
| `unroll_a_pingpong` | M3 | 关 | 关 | 关 | 开 | A 双缓冲循环按两步展开，固定 ping/pong 相位 |
| `split_a_lds` | M3 | 关 | 关 | 关 | 开 | A 大 LDS 池拆成独立 ping/pong 对象；总 LDS 不变 |
| `fp8_b_waitcnt` | M3 | 关 | 关 | 关 | 关 | 实验性调整等 A DMA 时保留的 B/scale VMEM 未完成数量。当前最佳均不用 |
| `scalar_tile_row_base` | M3 | 关 | 关 | 关 | 关 | 用 readfirstlane 明确 tile 行地址 wave 一致。当前最佳均不用 |
| `prefetch_a_operand` | M3 | 关 | 关 | 关 | 关 | 提前把下一 A operand 从 LDS 读入寄存器。当前最佳均不用 |

### 4.3 Payload 就绪与工作发布（6 项）

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `skip_launch_barrier` | M3 | 关 | 关 | 关 | 关 | 尝试跳过跨 rank launch/代际握手；False 表示握手保留，不代表"不同步" |
| `prefetch_b_before_a` | M3 | 关 | 开 | 开 | 开 | **等远端 A payload 就绪之前**先取首块 B 和 B scale；不同于循环内 `pipe_weights` |
| `joint_work_flags` | M3 | 关 | 关 | 关 | 关 | 工作 ticket 和有效/继续标志一起发布到 LDS 的实验。当前最佳均不用 |
| `payload_chunk_rows` | 上游 | 0 | 0 | 0 | 2048 | payload 生产任务的分块行数；0 表示不拆。不是 GEMM M tile |
| `payload_tile_ready` | 上游 | 关 | 关 | 关 | 开 | 每目的 M tile 维护预期贡献数和完成计数；tile 到齐即可开始 GEMM |
| `payload_tile_publish_early` | M3 | 关 | 关 | 关 | 开 | chunk 内完成某 tile 的部分就发布贡献完成；跨 source/chunk 的 tile 仍须等所有贡献 |

### 4.4 调试（不参与性能排名，1 项）

| 参数 | 默认 | 含义 |
|---|---|---|
| `schedule_audit` | 关 | 记录 S1 tile/XCD 调度用于审计；高层入口未配置完整审计 workspace，shared 路径显式禁止——不能当成随时可开的日志开关 |

实现：`mega_moe_stage1.py`、`gemm1.py`、`gemm_util.py`（通用路径；SBM64/SBM128 档的实测实现在私有包 `sbm64_m32_m48_m64/`、`sbm128_m16/`、`sbm128_tiered/`）。

## 5. C 类：S2 GEMM2 / 调度（17 项）

S2 含 GEMM2、乘路由权重及结果写回。`shared_schedule` 存放在 Stage2Config，按调参对象归 §7。两阶段同名的 `band_m`、`xcd_schedule`、`schedule_audit` 是各自独立字段。

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `block_m` | 上游 | 必填 | 32 | 32 | 64 | S2 每 tile M 行数；必须整除 S1 `sort_block_m` |
| `block_n` | 上游 | 必填 | 128 | 128 | 128 | S2 每 tile 输出列数。**f7a9651 小档：256（仅 b64/b256 为 128）**；BN 决定 shared L2 队列周期 `model_dim//BN`，host epoch 记账与 kernel 各算一遍，用未验证 BN 的失败方式是 epoch 静默漂移 |
| `block_k` | 上游 | 256 | 256 | 256 | 128 | S2 每次循环的 K 宽度 |
| `persist` | 上游 | 必填 | 开 | 开 | 开 | 固定一组 CTA 循环取工作 |
| `persist_cu` | 上游 | 必填 | 128 | 128 | 240 | persistent grid 规模基数；队列模式再乘 `queue_grid_mult` |
| `queue_grid_mult` | M3 | 1 | 5 | 5 | 5 | `band_m>1` 队列模式的 grid 倍数；分别发 640/640/1200 CTA |
| `use_nt` | 上游 | 必填 | 关 | 关 | 关 | S2 B 权重 load 的 NT cache modifier；判据见 §0.5。f7a9651 小档 b32/b64 为开 |
| `b_hoist` | 上游 | 开 | 开 | 开 | 开 | 下一块 B 预取前移到 barrier 和 A 的 LDS 读取之前 |
| `ascale_prefetch` | 上游 | 开 | 开 | 开 | 开 | 预取下一块 A scale 并随循环状态传递 |
| `persist_strided` | 上游 | 关 | 关† | 关† | 关† | 老 persistent 调度的跨步 M 分配；当前走 `band_m>1` 队列分支，不参与 |
| `skew_cu` | 上游 | 0 | 0† | 0† | 96† | 老调度的 expert 偏斜检测；队列分支优先，不能写成"b8k 开启 skew 优化" |
| `spatial_partition` | 上游 | 402 | 402† | 402† | 402† | 老非 persistent 路径的 M/N tile 重排编码（402 = GroupNum 4、M01 2）；persistent 队列不使用 |
| `bf16_lds` | 上游 | 关 | 关 | 关 | 关 | S2 epilogue CShuffle 用 BF16 LDS 表示；与 `local_reduce` 的 BF16 staging 无关 |
| `xcd_schedule` | M3 | 关 | 开 | 开 | 开 | 按实际 XCD 选 home 队列；与 S1 同名但控制不同阶段 |
| `band_m` | M3 | 1 | 8 | 8 | 16 | 大于 1 启用八分片工作队列，按该 M band 组织 N 面板复用 |
| `schedule_audit` | M3 | 关 | 关 | 关 | 关 | S2 调度审计；高层 `_g2_schedule_audit=None`，直接打开会被断言拦下 |
| `shared_schedule` | M3 | tail | early2 | early2 | tail | 见 §7。**f7a9651 起融合 shared 小档默认 `jointtail`** |

实现：`mega_moe_stage2.py`、`gemm2.py`。

## 6. D 类：跨阶段通信 / local-reduce（1 项 + 构造选项）

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `p2p_quant` | 上游；MegaMoEConfig | 必填；M3 selector 为 none | none | none | none | S2→combine 传输格式。none 在此为 BF16；`fp8_blockwise_1x32` 已被否决（relL2 0.0266 > 门限 0.015）。S1 dispatch 仍为 FP8 |
| `local_reduce` | M3；构造参数 | 关 | 关 | 关 | 开 | 同 token 本 GPU 的 expert 贡献先归并再跨 rank 发送。要求 EP4/EP8、top4、BF16 transport。**小 batch 实测净亏，不要开** |
| `local_reduce_xcd_local` | M3；构造参数 | 关 | 关 | 关 | 开 | local-reduce staging 保持同 XCD N 分片内处理；依赖 local-reduce 和 S2 XCD 队列；必须禁止跨 XCD 偷任务 |

## 7. E 类：Shared 融合 / 调度（1 项 + 构造选项）

`shared L13` = shared expert 的 gate/up 第一层，`shared L2` = 第二层线性投影（与 GPU L2 cache 无关）。

| 参数 | 来源 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|---|
| `shared_w13`/`shared_w13_scale`（记录名 `shared_l13_fused`） | M3；可选权重 | None，关闭融合 | 开 | 开 | 开 | 提供 shared gate/up 权重及 scale，shared 第一层与 routed S1 共用 kernel；两者必须同时提供 |
| `shared_w2`/`shared_w2_scale`（记录名 `shared_l2_fused`） | M3；可选权重 | None，关闭融合 | 开 | 开 | 开 | shared 第二层进入 S2；依赖第一层 |
| `shared_xcd_schedule` | M3；构造参数 | 开 | 开 | 开 | 开 | shared 第一层使用 XCD 队列；只有提供 shared W13 才生效 |
| `shared_schedule` | M3 | tail | early2 | early2 | tail | `tail`：CTA 完成 routed 后再处理 shared 第二层；`early2`：每组 8 选 2 个 CTA 优先 shared；`jointtail`：共享 ticket 队列（f7a9651 小档默认）。无 shared L2 时不生效 |

## 8. F 类：Combine

| 参数 | 默认 | b16 | b256 | b8192 | 含义 / 注意 |
|---|---|---|---|---|---|
| `combine_local_prefetch`（实际 `prefetch_local`） | 按条件决定 | 开 | 开 | 关 | combine 等跨 rank 就绪前先加载本 rank 已完成的 routed 贡献。条件：shared L2 开、local-reduce 关、tokens∈{16,32,64,128,256} |

| 内部名称 | b16/b256/b8192 | 来源 / 作用 |
|---|---|---|
| `enable_std_moe` / `zero_copy` | 关 / 关 | 上游通用通信选项，M3 不启用，shared/local-reduce 路径要求关闭 |
| `dispatch_dtype` / `combine_dtype` | FP8 / BF16 | 构造时固定；`scale_dim=192`、`scale_type_size=1`（H6144 的 per-32 E8M0 scale） |
| `quant_type` / `combine_quant_type` | none | 上游 combine 的 direct-cast 开关，归一化为 none；不同于顶层 `p2p_quant` |
| `_ENABLE_COMBINE_NO_STAGE1` / `skip_stage1` | 开 | 跳过 combine 自身第一阶段搬运（S2 已写入 P2P 缓冲）；**不是跳过 MegaMoE GEMM1** |
| `enable_weights`（combine） | 关 | S2 已乘路由权重，combine 不重复乘 |
| `shared_input`（combine） | 开 | M3；combine 在 routed 归并时加本 rank 的 shared 输出 |
| `stage2_topk_ids` | 提供 | M3；前两档为本地预取识别本 rank expert，b8192 为 local-reduce 确定来源 |
| `local_reduce_epr` / `prefetch_local_epr` | 0/16、0/16、16/0 | M3；0 关闭对应 combine 分支；16 是每 rank expert 数 |
| `dispatch_block_num` / `dispatch_warp_num_per_block` | 不调用 | 上游独立 dispatch 几何；MegaMoE dispatch 已融合进 S1，不能替代 `num_dispatch_cu` |
| `combine_block_num` / `combine_warp_num_per_block` / `tuning_table` | 由通信算子解析 | 显式参数优先于 tuning table，再回退默认 128 blocks、8 waves；best-config JSON 未冻结这些字段 |

combine 的 FP32 累加属于当前实现分支，没有单独的 `accum_fp32` 用户开关。

## 9. 固定实验条件：模型语义与调用参数

不为速度随意改变。构造函数位于 `mega_moe_m3.py`。

| 参数 | 默认 / 取值 | 含义 / 注意 |
|---|---|---|
| `mega_scheme` | fixedslot | 底层 group-major 存储 scheme；不要与 `fixed_slot_dispatch` 混淆，三组实际均走 compact dispatch |
| `swiglu_limit` / `swiglu_alpha` / `swiglu_beta` | 7.0 / 1.702 / 1.0 | SwiGLU-OAI：gate 仅上截断、up 双侧截断；0 关闭 clamp。属数值语义 |
| `max_tok_per_rank` | 必填 | 每 rank 容量，必须为正的 2 次幂；影响分配、分桶和 kernel |
| `world_size` / `rank` | 必填 | EP 通信 rank 数 / 当前 rank |
| `model_dim` / `inter_dim` | 6144 / 3072 | hidden 与 expert 中间维度 |
| `experts` / `topk` | 128 / 4 | 全局 routed experts 数 / 每 token 选取数；EP8 每 rank 16 |
| `w1`/`w1_scale`/`w2`/`w2_scale` | 必填 | routed 权重和 scale，须符合当前 shuffle/量化布局 |
| `slice_output` | 开 | 返回张量切到真实 token 行数；不缩减内部容量 |
| `stream` | 当前 stream | preplan 的 route、quant 和 forward 必须匹配同一 stream |
| `forward` / `forward_prequant` | 调用者选择 | 实测调用 forward（含输入量化） |

当前 `self.quant="a8w8"` 是固定值，**没有独立 quant 构造开关**。TopK 的 selected-softmax×2、E128/top4 是 `route()` 的固定语义。完整 `TOKEN_BUCKETS` 为 1/4/8/16/32/64/128/256/512/1024/2048/4096/8192/16384/32768——是最近桶选择表，不是"全部 batch 均已验证"。

## 10. 内部派生开关、别名与调试

跟随 A–F 类实际配置生成，不作为独立搜索变量。取值按三组 EP8 最佳路径记录。

| 内部名称 | b16 | b256 | b8192 | 来源 / 作用 |
|---|---|---|---|---|
| `preplanned` / router `WRITE_HIST` / `histogram_precomputed` | 开 | 开 | 开 | M3；来自 `preplan_waves>0` |
| `adaptive_grouping` | 关 | 关 | 开 | 上游；外部分组存在时跟随 `payload_tile_ready` |
| `reset_stage2_queue` / `work_head_reset` | 开 | 开 | 开 | M3；来自 S2 `band_m>1`；经 quant/S1 前序路径复位队列 |
| `shared_l13` / `shared_xcd` / `shared_l2` | 开 | 开 | 开 | M3；由 shared 权重指针与 `shared_xcd_schedule` 派生 |
| `shared_early2` | 开 | 开 | 关 | M3；`shared_l2 && shared_schedule == early2`（f7a9651 小档为 jointtail 时此值随之变） |
| `small_xcd` | 开 | 开 | 关 | M3；S1 `xcd_schedule && !payload_tile_ready` |
| `same_xcd` | 关 | 关 | 开 | M3；local-reduce 内部对 `local_reduce_xcd_local` 的透传 |
| `fixed_slot_dispatch` | 关 | 关 | 关 | 上游；由容量和 shared 模式推导 |
| `direct_fixed_slot` | 关 | 关 | 关 | 上游；要求 EP8、每 rank 48 experts 的 DSV4 几何，M3 不满足 |
| `has_pad` / `i32_kpad` / `i32_npad` | 关 / 0 / 0 | 同 | 同 | 上游；S2 通用尾部形状掩码，当前形状整除不开启 |
| `is_f8_a` / `is_f8` | 开 | 开 | 开 | M3 固定 FP8 A |
| `single_rg`（BM16 路径） | 关 | 关 | 关 | 上游 GEMM2 helper；当前 BM32/BM64 不用，不能单独开它来支持 BM16 |
| `always_valid` | 开 | 开 | 开 | 上游 GEMM1 epilogue 参数；当前构造固定 True |
| `bf16_intermediate` | shared 开、routed 关 | 同 | 同 | M3；shared 任务对 gate/up 及激活结果做 BF16 往返舍入，保持 shared 数值路径；最终中间输出仍量化 FP8 |
| `g2_bhoist` / `g2_ascale_pf` / `g2_spart` / `g2_bf16_lds` | 见 S2 | 见 S2 | 见 S2 | S2 `b_hoist`/`ascale_prefetch`/`spatial_partition`/`bf16_lds` 的底层参数名 |
| `swizzle` / `async_copy` / `packed_lds` | 开/开/关 | 开/开/关 | 开/开/开 | loader 层对 S1 `swizzle_a`/`async_a_copy`/`packed_a_scale` 的别名 |
| `enable_group_major` / `gm_compact` / `gm_unit_size` | 开/开/128 | 同 | 同 | 上游 combine config；group-major compact 存储粒度 128，不是实际 S1 sort_block_m |

`max_size`、`num_records_bytes`、`a_tile_bytes` 等是 buffer resource 范围/ABI 参数，由具体 tensor 和容量决定，不是可调性能开关。独立实验 `gemm1_blds_experiment.py` 还暴露 `prefetch_b`/`waves_along_m`，三组生产路径均不调用该文件。

## 11. 环境变量入口（8 个，全部为 M3 新增）

主表最佳配置由 harness 注入完整 config，不依赖下表环境变量。

| 环境变量 | 对应字段 |
|---|---|
| `M3_MEGAMOE_DISPATCH_CU` | S1 `num_dispatch_cu` |
| `M3_MEGAMOE_GRID_MULT` | S1 `grid_mult` |
| `M3_MEGAMOE_WORK_SHARDS` | S1 `work_shards` |
| `M3_MEGAMOE_TILE_N` | S1 `tile_n` |
| `M3_MEGAMOE_SORT_BLOCK_M` | S1 `sort_block_m`；同时重新选择 S2 几何 |
| `M3_MEGAMOE_NUM_WAVES` | S1 `num_waves` |
| `M3_MEGAMOE_BAND_M` | S1 `band_m` |
| `M3_MEGAMOE_P2P_QUANT` | 顶层 `p2p_quant`（none / fp8_blockwise_1x32） |

**注意**：任一非空 S1 override 会让 `MegaMoEM3._select_config()` **绕过整组 measured preset**，且同时使 SBM64/SBM128 scope 失效（scope guard 显式检查这些 env）。显式替换 `_select_config` 的 harness 又可以绕过这套 env 解析。复现实验必须核对最终 config 和实际编译产物。`FLYDSL_DUMP_*`、JIT cache 路径、`MORI_SHMEM_HEAP_SIZE` 等是工具链控制，不是 MegaMoE 算法开关。

## 12. EP4 常用 batch 的代码默认值（非 EP8 实测最佳）

条件：H6144/I3072/E128/top4，EP4 每 rank 32 experts，`tokens == mtpr`，无 shared，`local_reduce=False`、`local_reduce_xcd_local=False`，无 env override。f7a9651 的 EP4 分支与 6b99bac6a 逐字相同（已 diff 核对），下表继续有效。

### 12.1 Stage1

| 参数 | b256 | b512 | b1024 | b2048 | b4096 | b8192 |
|---|---|---|---|---|---|---|
| `sort_block_m` | 64 | 64 | 64 | 128 | 128 | 128 |
| `tile_n` | 512 | 512 | 512 | 256 | 256 | 256 |
| `num_waves` | 8 | 8 | 8 | 8 | 8 | 8 |
| `grid_mult` | 1 | 1 | 1 | 1 | 1 | 1 |
| `num_dispatch_cu` | 32 | 32 | 32 | 64 | 64 | 96 |
| `mfma_amajor` / `async_a_copy` | 开 | 开 | 开 | 开 | 开 | 开 |
| `use_tile_resource` | 开 | 关 | 关 | 开 | 开 | 开 |
| `b_nt` | 3 | 0 | 0 | 0 | 0 | 0 |
| `waves_per_eu_hint` / `tile_k` | 2 / 256 | 同 | 同 | 同 | 同 | 同 |
| `pipe_weights` / `swizzle_a` | 开 | 开 | 开 | 开 | 开 | 开 |
| `work_shards` | 8 | 8 | 8 | 4 | 8 | 8 |
| `external_grouping` / `external_counting` | 关 | 关 | 关 | 开 | 开 | 开 |
| `skip_launch_barrier` / `joint_work_flags` | 关 | 关 | 关 | 关 | 关 | 关 |
| `padding_uniform_srcmap` / `count_uniform_matrix` / `row_base_prefetch` | 关 | 关 | 关 | 关 | 关 | 关 |
| `prefetch_b_before_a` | 关 | 关 | 关 | 关 | 关 | 关 |
| `preplan_waves` | 0 | 0 | 0 | 0 | 0 | 0 |
| `payload_chunk_rows` | 0 | 0 | 0 | 384 | 384 | 384 |
| `payload_tile_ready` | 关 | 关 | 关 | 开 | 开 | 开 |
| `payload_tile_publish_early` | 关 | 关 | 关 | 关 | 关 | 关 |
| `packed_a_scale` / `unroll_a_pingpong` / `split_a_lds` | 关 | 关 | 关 | 关 | 关 | 开 |
| `fp8_b_waitcnt` / `scalar_tile_row_base` / `prefetch_a_operand` | 关 | 关 | 关 | 关 | 关 | 关 |
| `xcd_schedule` | 开 | 开 | 开 | 关 | 关 | 开 |
| `schedule_audit` | 关 | 关 | 关 | 关 | 关 | 关 |
| `band_m` | 2 | 4 | 1 | 1 | 1 | 8 |

### 12.2 Stage2

| 参数 | b256 | b512 | b1024 | b2048 | b4096 | b8192 |
|---|---|---|---|---|---|---|
| `block_m` | 32 | 32 | 32 | 64 | 64 | 64 |
| `block_n` | 128 | 128 | 256 | 256 | 256 | 128 |
| `persist` | 开 | 开 | 开 | 开 | 开 | 开 |
| `persist_cu` | 128 | 240 | 240 | 256 | 240 | 240 |
| `use_nt` | 关 | 关 | 关 | 关 | 关 | 关 |
| `persist_strided` | 关 | 开 | 开 | 开 | 关 | 关 |
| `skew_cu` | 0 | 0 | 0 | 96 | 96 | 96 |
| `block_k` | 256 | 256 | 256 | 256 | 256 | 128 |
| `b_hoist` / `ascale_prefetch` | 开 | 开 | 开 | 开 | 开 | 开 |
| `spatial_partition` | 402 | 402 | 402 | 402 | 402 | 402 |
| `bf16_lds` | 关 | 关 | 关 | 关 | 关 | 关 |
| `queue_grid_mult` | 1 | 1 | 1 | 1 | 1 | 5 |
| `xcd_schedule` | 关 | 关 | 关 | 关 | 关 | 开 |
| `band_m` | 1 | 1 | 1 | 1 | 1 | 8 |
| `schedule_audit` | 关 | 关 | 关 | 关 | 关 | 关 |
| `shared_schedule` | tail | tail | tail | tail | tail | tail |

顶层 `p2p_quant` 六档均为 `none`。S2 b8192 的 `skew_cu=96` 被队列模式绕过；六档 `spatial_partition=402` 在 persistent 路径均不使用。b2048/b4096 的非队列 skew 分支仍可达，不能按 EP8 最佳路径删除。

EP4 的 `local_reduce` / `local_reduce_xcd_local` 是构造选择，不由 batch 自动打开。若两项同时打开，b1024 的 S2 默认由 M32/N256/K256 改为 M64/N128/K128，`band_m=8`、`queue_grid_mult=5`、`xcd_schedule=True`；`persist_strided=True` 原值保留但队列模式不用。其他几何按实际配置核对。

## 13. 组合约束与迁移注意

1. **四个上游开关已决定先保留**：`pipe_weights`、`swizzle_a`、`b_hoist`、`ascale_prefetch`；当前配置全 True 不能推出所有支持形状都可删除关闭分支。
2. b8192 专用优化不是不可拆分的整体：packed scale 可独立理解；`split_a_lds` 依赖 `unroll_a_pingpong`、`async_a_copy`、`mfma_amajor`，且要求 N256/K256/W8、偶数 K 循环等几何。
3. `payload_tile_publish_early` 要求 `payload_tile_ready`；tile-ready 要求正的 chunk 行数，chunk 必须被 S1 M tile 整除。
4. `prefetch_b_before_a` 要求 compact dispatch、`pipe_weights`、`async_a_copy`。`preplan_waves>0` 当前要求 EP8/epr16/top4、compact、`grid_mult=1`，并维护匹配的 route→quant→forward 状态。
5. S2 `block_m` 必须整除 S1 `sort_block_m`。源码允许某些 BM16 编译组合不等于数值正确；当前最佳只用 BM32/BM64，不能仅放宽 guard。
6. S2 `band_m>1` 要求 persistent；XCD/audit 调度要求 band>1；`local_reduce_xcd_local` 还要求 local-reduce 和 XCD 队列。`queue_grid_mult` 合法区间 1–8。
7. `local_reduce` 要求 BF16 transport；shared 当前也要求 BF16 transport，并只支持代码明确列出的形状。`shared_schedule=early2` 另有严格的 EP8 几何检查。
8. **没有效果的字段不只是假值**：`external_counting=True` 被 preplan 绕过；S2 `skew_cu=96`、`spatial_partition=402` 被队列模式绕过。清理前按生效路径归一化，不凭 True/False 或非零决定保留/删除。
9. 当前最佳全部关闭的实验项包括 S1 `fp8_b_waitcnt`、`scalar_tile_row_base`、`prefetch_a_operand`、`joint_work_flags`、`skip_launch_barrier` 和两阶段 `schedule_audit`。这是清理候选清单，**不表示已授权删代码，也不证明历史实验从未开启**。
10. 支持列表之外的 batch、`tokens != mtpr`、其他 EP/模型形状必须重新检查 selector 和约束。

## 14. 来源与核对方法

- 主表值：`bench/dep8_best_configs.json`；实现：`aiter/ops/flydsl/kernels/mega_moe_m3/` 下的 config、stage1/2、dispatch/preplan、GEMM、`mega_moe_m3.py` 入口（f7a9651 核对）。
- 上游字段对照：`aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py`（原版 MegaMoEV2）。
- combine：`aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py` / `..._kernel.py`。
- 建表核对：用 AST 枚举 dataclass 字段，逐字段对照 JSON，检查无漏项、无重复、取值一致；EP4 默认只执行无 GPU 依赖的配置选择代码；f7a9651 的字段计数与 EP4 分支已按 §0/§12 重新核对。

更新时先核对源码 HEAD 与 JSON；区分"配置值变了""默认 selector 变了""某组合新完成实测"三件事。
