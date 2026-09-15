# DEP8+MTP7 集成终态报告

> 范围：MegaMoE backend（`rocm_mega_moe`）迁入 vLLM DEP8+MTP7 decode 的端到端终验结果与未修问题。
> 有效性基准：集成树 HEAD `dd4845624a`（分支 `codex/m3-mtp-real-acceptance` 系），
> 测量日期 2026-09-11/12；aiter kernel 侧终态为 commit `f7a96515aa8e3b7c1e41506c37d49b78651ce8ad`。
> 来源：`handoff_dep.md`、`e2e.md`。

## 1. 负载口径

- 拓扑：**TP1 / DP8 / EP8**，8×MI350X 单节点；MTP7（每步 draft 7 token）。
- **合成 acceptance length 5.2**（非真实接受率/质量评测）。
- **每卡 `max-num-seqs=16`**，verify batch **128 = 16 × (7+1)**，`max-num-batched-tokens=128`。
- `FULL_DECODE_ONLY` graph、KV FP8、GPU util 目标 0.93。
- 输入均匀 60K–120K（平均 90K），输出参数 800（range ratio 0.5，非每条固定 800），seed=42，
  正式 800 请求。三次有效结果输入/输出 token 总数逐位一致（72,958,305 / 482,876）。
- **每卡并发 ≠ 整机并发**：vllm-bench 的 C 是整机客户端并发，C=16 摊到 8 卡 ≈ 每卡 2；
  配方里"N 并发"指每卡 `max-num-seqs`。不能把 client C=25 当每卡 25。

## 2. 终验数字（无插桩，同负载相邻运行）

| arm | P50 TPOT | P99 TPOT | 输出 tok/s | 请求/抢占 |
|---|---:|---:|---:|---|
| 原始 AITER baseline | 10.7530 ms | 11.4333 ms | 10282.11 | 800/800，0 抢占 |
| MegaMoE（Mori heap 4G） | **9.4337 ms** | 10.7531 ms | 10911.50 | 800/800，0 抢占 |

P50 TPOT **−12.27%**，吞吐 **+6.12%**。口径警告：这是一对相邻单次运行，**不是多轮配对统计**。

## 3. 单步分解（HIP-event 插桩值）

**MoE 占单步 57%，且含 EP all2all 通信**（12 并发/卡口径：MoE 24.5ms / 单步 42.8ms）。
单步时间对 batch 极不敏感：**16→96 token 同为 ~43ms**（decode 是权重带宽/固定开销主导）；
MoE 是唯一随 batch 大涨的项（16.6→24.5→26.5ms），dense attention 随并发线性涨。

带插桩的 B128 graph span（MegaMoE，仅定位用，不能替代无插桩 TPOT）：总 43.29ms，
其中 MoE 23.09ms（53.3%）、dense attention 5.59ms、未覆盖余量 11.55ms（含稀疏 attn/indexer、
sampling）。对应带插桩 P50 TPOT 10.3526 ms。

baseline 侧带插桩 breakdown 的慢速问题**根因未定位即被叫停**；历史 baseline breakdown
（verify 47.9ms、MoE 26.5ms）口径不同，不能与本轮配对求 speedup。

### 3.1 DEP8 复现基线（MI350X vs MI355X）

同配方 DEP8+MTP7 在 MI350X 上复现 peiyuanz 的 MI355X 结果（运行时代码零改动、12 并发口径、
权重节点本地盘加载）：1.55 QPS/卡点位实测 **1.338 QPS/卡（vs 1.392–1.404，−4.5%）**、
P50 TPOT **10.36ms（vs 9.76ms）**；1.60 QPS/卡跟不上（排空尾长）。约 4100 请求零失败零抢占。
硬件差异（MI350X vs MI355X）是最大变量。KV 池容量：MegaMoE 日志报 **2,612,480 tokens /
186.87 GiB/卡**，baseline 约 259–260 万 tokens/卡（长度估算口径，非运行中实测占用）。

## 4. Mori heap 口径

- 40 GiB 是旧 benchmark 硬编码的 static heap，不是实际需要。
- T128 实测：target 57 层 **2.142 GiB** + draft 7 个 runtime **0.263 GiB**，
  合计 **2,582,331,392 B ≈ 2.40 GiB/卡**（256B 对齐后；只是 Mori 对称工作 buffer，
  不含模型权重与 KV cache）。
- 当前下限已改为 **4 GiB + target/draft 累计 + 64 MiB reserve 检查**；800 请求终验用的就是 4G。
  不保证更大 batch 够用。

## 5. 集成事实

- 5 个主计算 kernel：TopK route（融合 local histogram/状态发布）→ MXFP8 quant + dispatch preplan →
  Stage1 → Stage2 → Combine；无独立 sync kernel。
- 运行依赖：精确 `flydsl==0.3.2`（私有 MLIR ABI，版本检查 fail-closed）+ `amd_mori` + Torch/vLLM
  Triton wrapper；无 aiter import。
- 只接受已验证的 gfx950 单节点 EP8/TP1/DP8/T16/H6144/I3072/E128/top4/single-shared-expert；
  1–16 真实 token 补到 T16，padding route ID=-1 不进 histogram。错误拓扑/版本/shape 提前拒绝。
- 镜像用 pinned sqsh + 精确文件 overlay（baseline 34 文件 / MegaMoE 56 文件），启动校验
  `runtime.sha256`；不是整树 PYTHONPATH。
- 8 卡数值门禁：eager 对 standalone 参考最差 rel-L2 0.00332；CUDA graph 连 replay 16 次与 eager
  逐位一致。

## 6. 已修复：MXFP8 全零 block NaN

CDNA 把 `float32.tiny` flush 成 0 → 全零 block 的 descale=0 → 0/0 量化出 NaN。
修复：全零 block 显式用 **E8M0 unit scale 127**（`vllm/third_party/aiter/mxfp8_gemm.py`，7 行）。
修复后两种宽度的量化 payload 从 `[255]`/非 finite 变 `[0]`/finite，scale `[127]`。

## 7. 未修：MI350X 高并发 segfault

- 现象：高负载 decode 时 8 个 rank 的 worker 同一秒全部段错误（信号杀死）。
- 触发：负载压力组合（并发 × 上下文 × 输出长度）。轻载下 **C=104 过、C=112 崩**；重负载
  （600 输出）64 并发也崩。
- 定性：**`HIP_LAUNCH_BLOCKING=1` 下同负载不崩 → 竞态/异步执行 bug**。非 OOM、非 JIT、非 MTP 缺失。
- 状态：**未修复**。24 并发配方在 MI350X 上不安全；竞态非必现（16/卡曾 800/800 零失败跑完，
  生产不可赌）。

## 8. 已叫停的工作（勿自动恢复）

baseline breakdown 插桩慢速排查被用户叫停、节点已释放：成功的无插桩 baseline 与失败插桩版的
34 个 runtime hash 完全相同，差异只有测时 PYTHONPATH/输出目录/marker；py-spy 只证明"等 GPU 完成"，
不是根因。若恢复：先原样复现旧 hook/旧本地输出路径并记录 warmup 逐步进度，再单变量拆分
replay 外 events、graph 内 stage events、flush 行为。
