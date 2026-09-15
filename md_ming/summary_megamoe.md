# MegaMoE（MiniMax-M3 / AMD MI350X）总览

- 范围：aiter `mega_moe_m3`（M3 适配版 MegaMoE）的最终状态、成本模型、读数规则、部署事实与未决问题。
- 当前状态：2026-09-15 收尾，仓库版本与复现见 [redo.md](redo.md)。历史性能表保留原测量基线，旧行号仅作定位线索。
- 来源：`handoff_megamoe.md`、`summary_megamoe.md`（旧版）、`plan_megamoe.md`、`redo.md`、`report_sbm128.md`，及 commit 内 `docs/mega_moe/sbm64_20260915/`。详细定稿见 `report_sbm64.md` / `report_sbm128.md`，开关全表见 `option_megamoe.md`。

## 1. 流水线与适用范围

完整 forward 是 **5 个 kernel**：

```
TopK（融合 local histogram）→ MXFP8 quant+preplan → Stage1（GEMM1+SwiGLU+输出量化）→ Stage2（GEMM2+P2P）→ Combine（FP32 求和）
```

- 计时口径：TopK + routed MoE + TP1 shared + 最终求和，**不含** router GEMM 与 residual。
- quant+preplan 的大读数（17–126 µs）是**跨 rank 等待**（preplan_waves=4 把 launch barrier 放在该 kernel），不是成本。
- 调用方式：同 stream 先 `op.route(...)` 再 `op(...)`；要求 `tokens == max_tok_per_rank`（mtpr）。

**保留路径的 scope（SBM64/SBM128 专用实现只在以下条件全部满足时生效）**：
EP8（world_size=8、每 rank 16 experts）、H6144/I3072、E128/top4、shared L13+L2 融合且 XCD 调度开、`tokens==mtpr`、无 local-reduce、无任何 `M3_MEGAMOE_*` Stage1 env 覆盖。scope guard 在 `mega_moe_m3.py:107-122`（`_sbm128_scope` / `_sbm64_scope`）。

## 1.5 数值与集成契约（架构级事实）

- **量化契约**：kernel 内 `per_1x32_mx_quant`（e8m0 round-up ceil + RNE）是唯一 bitwise 对拍参照。**不要用 `aiter.ops.quant.per_1x32_mx_quant_hip` 当参照**（round 非 ceil、scale 返回 fp32、重建误差差 10 倍，不在生产路径）。
- **a8w8**：GEMM1/GEMM2 的 B 通路为 fp8（每 lane 32 个 fp8 按两个相隔 1024B 的 16B cell 读入并 shuffle 成 i32x8 fragment）；A 通路与两侧 E8M0 scale 通路零改动。
- **SwiGLU-OAI**：α=1.702 / β=1.0 / limit=7.0，gate **只截上界**（双侧 clamp 是方向错误）。
- **scale=2.0** 折进 fp32 topk_weights（2 的幂，bit-exact）；routing 在 kernel 外。
- **shared expert** 作为追加的"第 129 号本地 expert"（尾部槽位），数据全在本地、零跨卡流量；combine 输出为 `BF16(Σ_FP32 routed + FP32(shared))`。
- **依赖**：单节点 intranode（mori shmem）；FlyDSL 钉 0.3.2；mori ≥ 1.2.2（1.0.1 在 shmem 初始化 SIGSEGV）；arch 限定 gfx950。

## 2. 逐档加速比轨迹（vs standalone，EP8 完整路径）

测量时间 2026-09-10/11，standalone = 标准 EP8 路径（all_gatherv → AITER fused MoE → reduce_scatterv 口径的独立实现），同进程 balanced AB/BA 配对、8 rank max-local-mean。**这些数字早于 09-13~15 的 SBM64/SBM128 合入**；合入后又快 2.10%–4.97%（b64–b192，见 `report_sbm64.md`），未重测对 standalone 的终值。

| tokens/rank | speedup | 样本 | 备注 |
|---:|---:|---:|---|
| 16 | 1.0502× | 3 对 | 三参数调优（SBM32+b_nt2+dcu48）+ BN256；三档小 batch 中最弱，跨 run 噪声 1.55 pp |
| 32 | 1.1619× | 2 对 | BN256 已采用 |
| 64 | 1.2547× | 2 | BM64/BN128（BM64+BN256 超 64 KB LDS） |
| 128 | 1.4740× | 2 对 | BN256 已采用 |
| 256 | 1.470–1.522 | 1+2 | standalone 臂跨 run 漂 1.31%，只能给区间 |
| 512 | 1.4807× | 3 | SBM128 大 regime |
| 1024 | 1.5777× | 2 | |
| 2048 | 1.8483× | 4 | 除 dcu/qgm/bn256 外几乎未调 |
| 4096 | 1.9595× | 2 | 同上 |
| 8192 | 1.9872× | 1 | 大 batch 向 2× 饱和（早期一轮记录为 1.985，移植 harness 复现差 0.09%） |

绝对值锚点（同一轮次才可比）：b256 591.387→388.617 µs；b8192 8999.818→4528.935 µs；b16 调优后 251.793 µs。

## 3. S1/S2 成本模型（2026-09-11 建立，仍有效）

```
S1 ≈ 46.9 µs + 字节 / 6.42 TB/s   （残差 ≤ 0.12 µs）
S2 ≈ 30.7 µs + 字节 / 8.94 TB/s   （残差 ≤ 0.66 µs）
```

- **b16 的 S1 只认字节**：BM64/N12 与 BM32/N24 字节相同（679.5 MB）时 S1 152.59 vs 152.53 µs。"更多 tile 填满 CU"和"提高 occupancy"两个方向均被证伪。`b_nt=2` 机理经 ISA 确认（44 条 `buffer_load_dwordx4` 中 32 条带 nt）。
- **两个截距（46.9 / 30.7 µs）未识别**：已排除 shared 权重、tile 形状/数量/占用；与 dispatch CTA 数强相关。Combine 搬 786 KB 却要 15–16 µs，同属固定成本。
- standalone b16 通信占 kernel-sum 25.8%（73.97 µs），但每卡 all-gather 仅 0.197 MB——**是延迟/rendezvous 而非带宽**。"通信全免"上限约 195 µs / 1.35×，b16 当时 251.79 µs 距其约 57 µs，主体就是两个截距（≈78 µs）。
- PMC（b256 档）：S1 3.80 TB/s（56–58% 可达峰值）、S2 4.30 TB/s（63–66%）；矩阵管占用 S1 18.7%、S2 13.7%。**两堵墙都不贴**，瓶颈在依赖链/等待类，无直接证据。

## 4. 读数四规则（引用任何百分比前必读）

1. **百分比不可相加**：每条改动的百分比各有基线和口径，来自不同轮次、不同代码状态。
2. **看清基线**：`vs 当轮默认`（真收益）/ `恢复自身回归`（填自己挖的坑）/ `vs 已取代配置`（今天无此增量）/ `vs 另一基线`（如 vs vLLM 标准路径，不能和项目内数字混用）。
3. **看清口径**：`fused S1` / `S2 path` / `完整 forward` / `standalone 总时` 分母差好几倍。
4. **旧时代百分比不能平移**：分母（如 8k 整层 4.88 ms → 4.2 ms）和 S2 占比都变了。

唯一可靠参照是同轮配对的绝对时间。跨 run 漂移使单点对单点的小效应不成立——**地板按 `tool_measure.md` §2 的两条为准（T≤256 用 3 µs、T≥512 用 8 µs，外加 null 自比较 ≈0.2%）**；本文早先写的 1.5 µs 只由 b128 两个同 ISA build 得出，已被 b96 档同 session 四点 3.42 µs 的跨度推翻（`dep8_best_configs.json` 的 `96` 条目注记），不要再按 1.5 µs 判。配对消不掉参照臂的 run 间漂移（b256 上 standalone 臂漂 1.31% 而候选臂只 0.77%）。

测量口径要点（细节见 `tool_measure.md`）：relL2 门先于一切性能数字（基线对 bf16-intermediate ref 0.0033、megamoe 对 fp32-intermediate ref 0.0026）；同进程 12 组 balanced AB/BA，chunked replay + HIP event，跨 rank 按 MIN 对齐，`critical_path = max(per_rank_ms)`；CI 按 12 对 slice bootstrap，replay×rank 不是独立样本；链式收益不得称作直接实测。

## 5. 当前默认配置（2026-09-15 收尾）

入口：`MegaMoEM3._select_config`。精确范围为 EP8、E128、H6144/I3072、top4、TP1 shared L13+L2、XCD、`tokens == max_tok_per_rank`，无 local-reduce、无显式环境覆盖。batch 均为 **tokens/rank**；本轮说的 b200 是 batch 200，硬件始终 AMD gfx950。

| batch | S1：SBM / N / 路径 | S2：BM / N / K；waves、CTA |
|---|---|---|
| 8–56 步8、80/88/96 | 32 / 512 / generic | 32 / 256 / 256；4、640 |
| 64 | 64 / 256 / m32_m48_m64 | 64 / 128 / 256；4、640 |
| 72 | **32 / 256 / m16_m32_n256** | 32 / 256 / 256；4、640 |
| 104/112 | 64 / 256 / m32_m48_m64 | **32 / 256 / 256；4、640**；112 保留 constant N/K 地址 |
| **120–192 步8** | 64 / 256 / m32_m48_m64 | **64 / 256 / 256；8、256**；M32/M48/M64、fixed N24 |
| 200/208/216 | 128 / 256 / m16 八路径 | 32 / 256 / 256；4、640 |
| 224/232/240/248 | 128 / 256 / tiered96 | 32 / 256 / 256；4、640 |
| 256 | 128 / 256 / tiered96 | 32 / 128 / 256；4、640 |
| 512/1024/2048/4096/8192 | 128 / 256 / 大档既有流水 | 64 / 128 / 128；4、1200 |

- S2 NT 仅 b32/b64 开；小档 `jointtail`，512+ `tail`。S1 小档 K256/W8，dcu48/B-NT2（b256 为 dcu32/B-NT0）。大档 tile-ready/early-publish、packed scale、unroll/split LDS 保留。
- b72、b200 的 S1/S2 均保留 leader drain；120–192 的 S2 big-CTA 路径也有 leader drain。b136 的 S1 leader drain 保留。
- b104/112、b200–256 的 S2 整空 BM32 子块跳过开启；120–192 的 BM==SBM，无 BM32 子块。**b112 的 S2 M16、epilogue M16、512 CTA 均未采用**。
- 已补齐此前只在实验目录里的 b120–184 九点 M64 默认，完整配置账本见 [probes/megamoe_defaults.json](probes/megamoe_defaults.json)，重启与验证见 [redo.md](redo.md)，新结果见 [report_s2.md](report_s2.md)。旧 `bench/dep8_best_configs.json` 是历史快照，不代表最新全档默认。
- “最快”指已比较方案中决定保留的配置。37 点均校验默认路由；部分小档沿用 fallback，大档沿用既有结果，**没有逐点新测或全局最优保证**。单次计时会波动。
- 任一非空 `M3_MEGAMOE_*` S1 override 会绕过整组 measured preset，并关闭专用 scope；复现前由审计脚本拒绝覆盖。不要为复现默认手工设置 TILE_N/SORT_BLOCK_M。

## 5.5 关键 commit 时间线（全部为 f7a9651 的祖先，git 已核实）

| commit | 内容 |
|---|---|
| `6b99bac6a` | 允许小档 `sort_block_m=32`（b16 三件套生效的前提，无 codegen 变化） |
| `196a4399d` | selector 对 16–8192 每档发实测配置 |
| `6051eef22` | S2 `block_n=256` 落到 b16/32/128；guard 集中于 `mega_moe_config` 的 SHARED_* 形状表 |
| `dd3a0b3a2` | 全部 padding A 读写加界；支持非 2 幂 batch |
| `dfad71696` | b96 的 S2 `block_n=256` |
| `662b5caba` | S2 全空 routed 子块跳过（初版白名单） |
| `860181bc6` | S2 jointtail：routed/shared 共享 ticket 队列 |
| `05bbc6925` | S1 packed shared-first 队列计数器 |
| `e84ac1b38` | b96 S1 先读 current A 再发 next DMA（generic 路径最后一次 S1 调度优化） |
| `b13717ca0` | b200–256 SBM128 八路径/三档默认接入 |
| **`f7a96515a`** | **b64–b192 13 档 SBM64 `m32_m48_m64` 默认接入（最终定稿）** |

## 5.6 DEP serving 端到端参照（2026-09-12，集成快照）

- 工作树 `dep-dep8-mtp`（HEAD `dd4845624a` + MoE 接线），每卡 16 并发、MTP7、B128：原始 baseline P50 TPOT **10.7530 ms** → 集成 MegaMoE **9.4337 ms**；两边 800/800、零抢占。**单次相邻无插桩对比，非多轮统计**，且不表示已同步 SBM64/SBM128 等后续提交。
- 资源：Mori heap 4G 已跑通；T128 target+draft 对称工作 buffer 合计约 **2.405 GiB/卡**（其中 draft 约 0.263 GiB）。
- breakdown（B128 target graph）：43.2921 ms，其中 MoE 23.0924 ms。

## 6. 未决问题清单

| 问题 | 现状 |
|---|---|
| S1 46.9 µs / S2 30.7 µs 截距 | 长期未识别；与 dispatch CTA 数强相关；Combine 15–16 µs 固定成本同类 |
| S2 读 A2 重复 48× | 倍数 = `model_dim/BN`；453 MB demand vs 理想 9.4 MB，per-XCD L2 地板 8×=75 MB。BN 翻倍砍半，但 BM64+BN256 超 64 KB workgroup LDS（见下行），须 BM 降 32 |
| S1 读 A 重复 24× | 453 MB demand，地板 151 MB；`_decode_xcd` 的 band 顺序在两次复用间冲掉 3–5 MB；tile_n 256→512 可砍半但 LDS 预算不够 |
| 大档 BM64/N256 | 小档 120–192 已以 W8/256 CTA 解锁；512+ 的 BK128/队列/流水不同，尚未迁移验证 |
| `dcu=64 @ b512` 倾斜正确性 bug | **未修、未定位到行**：all_local 探针下 rank7 连续 256 次 graph replay 后与 eager 不逐位；`producers_per_destination=8==npes` 且只有 1 个 chunk，与 8k dispatch 倾斜 bug 同族。因此 dcu=64 不向 b512 提供（代码注释明写） |
| middle16 增量收益 | f7a9651 按用户决定保留 middle16 producer+consumer guard，但其相对 last16 的小幅增量未解决（no-skip 对照 CI 全跨 0）；不能把累计收益归给该项 |
| `route_masks` 单缓冲隐患 | **未修**（只在 `local_reduce` 开启时分配使用，`mega_moe_m3.py:778`，单次分配、索引无 parity）：quant kernel 顺带远端写 peer 掩码（system\|NT store），发布顺序只保护下一轮、不保护 peer 仍在跑的本轮 S2。修法：双缓冲或把掩码发布迁到 S1 rendezvous 之后 |
| b104/112 低于趋势 | SBM64 时代对 standalone 的加速比 1.368/1.375，低于 b120 的 1.446，原因未定位 |
| b256 上 BN256 为何亏 2.52% | 两个解释（grid 超订 / B 面板掉 L2）均被自己的判据打掉，至今无解释 |

## 7. 一行级否决备忘（防重做）

- fp8 combine（`p2p_quant=fp8_blockwise_1x32`）：值 5.0/5.6/7.6 点 layer 时间但 relL2 0.0027→0.0266 超门限 0.015，否决，只留 env hook（默认 none）。（旧 summary 记录作 0.0026，handoff/plan_clean 作 0.0027，源文档本身分歧，取后者。）
- S2 `block_m=16`：编译过但 routed 数值错（rel 0.951），非 guard 问题。
- S1 `tile_n=256` @ 大档旧几何：A 重读翻倍，更慢（注意：这与 SBM64/SBM128 保留路径的 N256 是不同时代的不同几何，不矛盾）。
- S2 逐 16 行内部跳空（内部 M16）：OFF；SBM128 完整八路径早期未修流水版慢 1.43%（已被修正版取代）。
- split-K / combine 提前消费 / fence 弱化：均实测否决（后者 replay 正确性失败）。
- b256 `queue_grid_mult=896`（+0.79%）：机理不明，未采用。
