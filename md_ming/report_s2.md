# S2 当前结果与停点（2026-09-15）

适用 AMD gfx950、EP8、H6144/I3072、E128/top4、TP1 shared。batch 指 tokens/rank。当前完整选择见 [总表](summary_megamoe.md#5-当前默认配置2026-09-15-收尾)，版本和复现见 [redo](redo.md)。

## 1. 默认已补齐

此前 b120–184 的 M64 胜者只在实验目录，默认仍是 BM32；本次把九点接入默认，与已保留 b192 共用 **BM64/N256/K256、W8、256 CTA、M32/M48/M64 入口、fixed N24**。shared/routed 都走该实现，上半32 epilogue 跳空和 leader drain 保留。只有 selector、scope、kernel tag 扩展，计算与同步循环不改。

| batch | 旧默认完整 forward µs | M64 µs | 降低 |
|---|---:|---:|---:|
| 120 | 265.235 | 264.269 | 0.365% |
| 128 | 266.840 | 266.213 | 0.235% |
| 136 | 272.096 | 270.220 | 0.689% |
| 144 | 277.415 | 274.555 | 1.031% |
| 152 | 284.383 | 278.506 | 2.067% |
| 160 | 285.905 | 280.567 | 1.867% |
| 168 | 290.156 | 285.260 | 1.687% |
| 176 | 292.150 | 287.938 | 1.442% |
| 184 | 298.274 | 293.600 | 1.567% |

每点是历史单个 capture 顺序、12 对 AB/BA 的**时钟诊断**结果，CI 均小于1；不等于通过严格稳定性协议，尤其 b128 的0.235%很小。按“只要不慢就保留”的决定接入，不承诺每轮都更快，也不把收益叠加到旧 standalone 表。

本次验证：双仓库各12个CPU回归；37点默认逐字段检查；1312个范围外/partial/env比较；九个新点+b192+b112共 **88份 S2 ISA** 与历史GPU验证版本一致（仅归一化kernel符号名）。M64为201VGPR/106SGPR、LDS66112B，spill/private均0；b112保留231/102、LDS33088B、spill/private均0。历史九点各有34组数值/routing probe和graph replay。K3审核通过；**此次接线没有重跑GPU**。

证据：[默认接入审计与原始数据索引](../aiter-ming-amd-m3-megamoe/docs/mega_moe/s2_defaults_20260915/README.md)。

## 2. b112 保留 BM32 constant N/K

- M64 已含 M32/M48/M64 和 fixed N24，完整 forward **261.229→262.617µs，慢0.531%**；不是漏了 M48。b104 M64 遇 Slurm credential 失败，没测成，继续原默认；b64没有本轮对比，继续 BM64/N128。
- b112 当前 BM32/N256/K256、W4、640CTA、整空子块跳过，保留 constant N/K **地址**；K-loop 次数仍运行时取值，host 断言限定真实 H/I。
- 48行在 BM32下分成32+16。该输入33/183非空子块可以改M16，总MFMA行数理论少9.016%；权重读取、队列和 epilogue 不会同比下降，不能直接估成9%时间收益。

## 3. 最近诊断：均未进入默认

以下为固定真实S1输出后的 **S2局部时间**，不可与§1完整forward相加。shared/routed隔离均显示M16有收益，未支持“只有routed实现有问题”的判断。

| 实验 | 有效工作结果 | 最终判断 |
|---|---|---|
| M16、shared setup | 完整forward正反顺序收益落在约0.01%–0.36%，反向不稳定 | 不采用 |
| M16 shared/routed隔离 | exact16：shared −2.576%、routed −2.578%、合并 −1.066%；全部S2 −0.404% | 隔离收益不可相加，不推广 |
| 640→512 CTA | 全部S2：M16关 −0.067%，开基本0，CI均跨1；no-tile约 −20% | 只减少失败claim，未兑现整体收益 |
| **epilogue M16** | 全部S2 **77.320→77.289µs，−0.041%，CI跨1**；exact16 **33.345→33.118µs，−0.681%** | 正确，但整体收益不足，不采用 |

最新 epilogue 诊断两臂是同一ISA、GEMM M16均开，只有runtime epilogue阈值0/16不同；247VGPR/106SGPR、零spill/private、8waves/CU。比之前诊断多17VGPR/1SGPR；34组probe、graph512、8rank CPU/GPU ISA、poison/epoch/head、独立算术复核和K3审核通过。它不是生产默认，结果仅属时钟诊断。

源码/结果：[epilogue M16](../.scratch/b112_s2_epi16_20260915_v3/README.md)、[512 CTA](../.scratch/b112_s2_grid512_20260915_v1/README.md)、[shared/routed隔离](../.scratch/b112_s2_m16_paths_20260915_v2/README.md)。

**下一方向只讨论未执行**：b112保持640 CTA，试 leader empty drain，先和K3审，再一次一个实验/形状。当前没有运行中的GPU实验。
