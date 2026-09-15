# MI350X / gfx950 平台尺子

> 范围：MI350X（gfx950/CDNA4）上实测定标的带宽、cache 层级、边际成本锚点与资源
> 上限——任何「离峰值多远」「这个改动值多少」的说法都必须以本文为尺。
> 有效性基准：2026-09-02 ~ 09-05 期间在 MI350X（SPX + NPS1、ROCm 7.2.3）上的实测；
> 带宽尺以 `probes/kvo_bw_floor.py`（flat read/copy/triad 教科书探针，已随本目录迁移）为准。
> 来源：`handoff_kv_outer.md`、`HANDOFF.md`、`report_claude.md`、
> `plan_m32_membership.md`、`OPTIMIZATIONS.md`、`handoff_megamoe.md`、`tool_root_cause.md`。

## 1. 带宽尺

**机器峰值就是 ~7.3 TB/s**（HBM3E 理论 8 TB/s 的 91%，HBM 上典型的 STREAM 效率）：

```
                 read   copy  triad   TB/s
128 MB (MALL 内)  7.07   7.28   6.10
2048 MB (超 MALL) 6.41   6.18   6.07
```

- **MALL 相对 HBM 只有 ~1.15x，没有 2x。** MI300X 时代 HBM 5.3 / MALL ~17 的大差距在
  MI350X 换 HBM3E 后被追平。缓存层给不了 2x，整条曲线都是 HBM 量级。
- 散布访存实测可达 **6.7–6.8 TB/s**（MSA decode kernel，非 flat 探针）。
- 「我们离峰值多远」的**唯一可引用尺子**是 `probes/kvo_bw_floor.py`（换机器后先跑它重新
  定标，上表是本机实测值不是规格值）；别拿逻辑 TB/s 直接当
  DRAM 实测（kernel 逻辑带宽 ×1.2 ≈ DRAM 实测）。
- 读写混合 kernel 的屋顶不是 flat read：combine 那种「瘦 WG × 64KB 连续读 + 4KB 写」
  形状的屋顶实测 5.4–5.5 TB/s，**拿错屋顶会量出不存在的优化空间**。
- MALL cliff 阈值 = **256 MB**：写出流量低于它时根本不去 DRAM（争用税精确为 0），
  超过立刻出现 15–22% 的争用税并饱和（非渐变，是开关）。

## 2. Cache 层级：L2 每 XCD 私有

- **L2 每 XCD 私有（每 XCD 4 MB，8 XCD 合计 ~32 MB），跨 XCD 读很慢。**
  「多个 workgroup 读同一块所以 L2 会吸收掉」的心智模型不成立，除非共享者被钉在
  同一个 XCD 上（decode swizzle / prefill 分组的全部收益来源）。
- 反过来说：每个 block 只被一个 workgroup 读的设计完全没有跨 XCD 暴露。
- L2 命中率本身可以是假象：31% 命中可能只是同一 cache line 的空间合并，不省 DRAM
  流量（以 `TCC_MISS × 128 B` 对账）。

## 3. 边际成本锚点

| 操作 | 代价 | 出处 |
|---|---|---|
| 空 kernel dispatch 地板（同 grid 立即 return） | **4.08 µs** | plan_m32_membership §C.2；四种独立 reduce 改写都停在 4.28–4.68 |
| 一个 32×32 all-pairs 矩阵 + 归约 | **0.20 µs** | `.scratch/diag/m64_mat3x.py`（271 KB vendored 整核变体，未迁移），b16/b128 同值 |
| 一次 `merge_row`↔`merge_col` 转置 | **0.12 µs** | 去掉两个测得 b16 −0.12 / b128 −0.24 |
| 一次 agent-scope fence | **0.46 µs** | handoff_megamoe split-K 定价 |
| agent-scope release（`buffer_wbl2`，每 workgroup 刷整个 XCD L2） | **42.8 µs** | HANDOFF §5；所以发布侧绝不用 agent fence，见 trick_fused_reduce.md |
| device-scope 原子 acq_rel → relaxed | 2.44x（23.7→9.7 µs） | handoff_kv_outer §5.2，count kernel |

两个用法：

- **别拿矩阵面积推代价**——一个 32×32 all-pairs 只有 0.20 µs。
- CUDA graph 只消 per-replay 那一次 5.3–6.8 µs，**不消 per-node dispatch 地板**
  （瘦 kernel 1.3–1.7 µs）。多出来的 node 是真实净损失。

## 4. 计算资源与驻留上限

- **256 CU / 8 XCD**；每 CU **160 KB LDS、131072 VGPR（每 SIMD 512×64）**、2048 threads。
- **驻留上限由 VGPR 决定不是 LDS**：`⌊512/VGPR⌋` **waves/SIMD**。
  例（waves/SIMD）：VGPR ≤128 → 4、130 → 3、171 → 2（**170 是那道悬崖**）、256 → 2。
  **换算成 CTA/CU 之前必须先看 CTA 几个 wave**：4-wave CTA（一 SIMD 摊一个）时
  CTA/CU == waves/SIMD；8-wave CTA 时 CTA/CU == waves/SIMD ÷ 2。所以 256 VGPR 在
  4-wave CTA 下是 2 CTA/CU，只有在 8-wave CTA 下才是 **1 CTA/CU、占用率 2/8 = 25%**
  （MegaMoE S1 的生产档全是后者：8-wave CTA，VGPR 232–256 三档都落在 ⌊512/VGPR⌋=2
  waves/SIMD 上，所以全部 1 CTA/CU——是被 VGPR 档位卡住，不是恰好占满 512）。
  别把某个 kernel 的 CTA/CU 结论直接搬到 wave 数不同的 kernel 上。
- **CDNA 上 MFMA 不是异步指令**（不是 Hopper WGMMA）；MFMA/softmax 重叠靠同 SIMD 多
  wave 共发射，**≥2 CTA/CU 共驻就自动发生**；被 VGPR 压到 1 CTA/CU 时 softmax 完全暴露。
- gfx950 fp8 MFMA 每条吃两倍 K：bf16→fp8 同循环 MFMA 指令数减半，CVT 指令 −92%。
- 不存在固定的 CTA/CU 目标：CTA 多 → 共发射好；每 CTA tile 多 → CTA 内软流水好 +
  免 reduce。平衡点随 shape 移动，chooser 用实测表不用公式。

## 5. XCD 映射（硬件 `XCC_ID` dump 出来的，不是推的）

- XCD = `linear mod 8`（mod 2/4 不成立）；**旋转 `[6 7 0 1 2 3 4 5]` 不是恒等**，
  workgroup 0 在 XCD 6 上。代码只能依赖同余，写死 `mod 8` 映射是错的。
- `grid.x` 是 8 的倍数 ⇒ 同 sequence 所有 workgroup 100% 同 XCD；否则 0%。
- XCD 数运行时查 `hipDeviceAttributeNumberOfXccs`（=10018），padding 取
  `lcm(查询值, 8)`。MI350X/MI355X 都是 8。

## 6. 测量假象与机制判读

- **刷缓存必须只读刷**。用 `zero_()` 做写式刷缓存会得到 **2.27x 的假象**（384 MB 脏行
  回写和被测 kernel 的读抢带宽）；只读刷的真实 warm/cold 差是 1.22x。
- **写回争用是 DRAM 侧 credit 背压，不是 cache 排空**：PMC 钉死——「幽灵写」一个字节
  都没有（combine 窗口 DRAM 写流量就是它自己的 64 MB），TCC miss% 三位有效数字不变
  （排除「po 被挤出 MALL」），变的只有读速率（−15%）和
  `TCC_EA0_RDREQ_DRAM_CREDIT_STALL`（5.7 倍）。写已离开 TCC 但堵在内存控制器下游，
  后续读者的读拿不到 DRAM credit。
- **私有化（replica counter）阈值 ~32 edges/bin**：低于它时私有化反而更慢（地址摊宽
  多拉的 cache line 比省下的争抢贵）。Fireworks 的 `_COUNT_REPLICAS_FLOOR=16` 在这台
  机器上是错的，会把 16x1024 的 count 从 9.4 推到 11.8 µs。
- **device-scope 原子对争抢不敏感、对顺序语义极敏感**（acq_rel→relaxed 即 2.44x）。
  这和 L2 统一的 Blackwell 是两种机器，Fireworks 的 playbook 不能照抄。
- 占用越高越快的直觉在小 batch 上反向：小 batch 是延迟受限（1.93 TB/s）非并行度受限，
  加大 split 反而更慢（ns=32：s4 8.681 µs < s8 9.040 < s16 12.720）。
- rocprof trace / counter CSV 的 `VGPR_Count` 字段不准（漏 AGPR）；资源/驻留结论以
  ISA metadata 与 HIP occupancy 查询为准。
