# MiniMax-M3 AMD MSA decode 优化终态报告

范围：MI350X/gfx950 上 MiniMax-M3 稀疏注意力 **decode** 线的最终状态——六 PR 栈、
decode E2E 终验、kernel 级 BRANCH_SWEEP、真实 union 探针、两个正确性修复的影响面。
有效性基准：六 PR 栈终验完成于 2026-09-06（`handoff_decode_msa_pr.md`）；kernel 线
`opt/msa-decode-xcd` 定稿于 2026-09-03（`HANDOFF.md`/`OPTIMIZATIONS.md`）；
E2E 终验 2026-09-04（`report_e2e.md`）。
来源文件：`handoff_decode_msa_pr.md`、`HANDOFF.md`、`OPTIMIZATIONS.md`、
`BRANCH_SWEEP.md`、`report_e2e.md`、`report_claude.md`、`report_codex.md`。

对象形状：TP4 rank-local 16 Q head / 1 KV head / head_dim 128，top-k 16×128=2048 KV
token/query，KV cache FP8 E4M3，decode_query_len=4（3 draft + 1）。每行恒定 attend
2048 token，**decode 延迟只是 `num_seqs = batch × query_len` 的函数**，与 context 无关。

## 1. 六 PR 栈（Inferact/vllm-internal，目标分支 `minimax-m3-amd`）

| PR | 远端分支 | 内容 |
|---|---|---|
| #446 | `pr/msa-aiter-mtp-mask` | vendor AITER Gluon paged-decode 到 `vllm/third_party/aiter/` + 修 MTP causal mask（正确性） |
| #447 | `pr/msa-prefill-page-stride` | 修 fused-prefill sparse page stride（正确性，非 decode 优化） |
| #448 | `pr/msa-decode-split-selector` | gfx950 实测 split chooser |
| #449 | `pr/msa-gluon-fp8-compute` | 删死 K 预取 + FP8 直算 FP32 累加（基于 #446） |
| #450 | `pr/msa-xcd-launch-placement` | XCD 数运行时查询 + grid padding + XCD 感知 workgroup 放置（基于 #449） |
| #451 | `pr/msa-gluon-fused-reduce` | 融合 split-KV reduce 进 Gluon attention dispatch（基于 #450） |

依赖关系：#447/#448 独立基于 `minimax-m3-amd`；#449–#451 依次叠在 #446 上。
干净集成树 `vllm-msa-e2e`（branch `integ/msa-final-e2e`）@ `190f9137bd`，六个 commit 顺序：

```
e0f580431f  Vendor AITER Gluon paged decode and fix MTP mask
9311643880  Fix fused prefill sparse page stride
2935ad4a62  Tune sparse decode split selection
67be1c589e  Optimize FP8 Gluon sparse decode
1711275767  Add XCD-aware sparse decode placement
190f9137bd  Fuse Gluon split-KV reduction
```

E2E 对照：baseline `5f9861bd`（更新后的 `minimax-m3-amd` + padded-tail fix），
candidate `0e1ff2f8`（六 PR 栈 + 同一 padded-tail fix）——公共修复两臂同含，不偏袒。

### split chooser 终表（#448）

```text
num_seqs ≤ 112  → 8 splits
num_seqs ≤ 160  → 4 splits
num_seqs ≤ 224  → 2 splits
更大            → 1 split
```

`num_seqs = total_query_tokens × num_kv_heads`（拍平后）；当前 TP4 路径 rank-local
KV head=1。chooser 的 `_MEASURED_SPLITS` key 不含 `decode_query_len`：**若存在非投机的
q1 decode，ns≈256 这一点会选错（慢约 13%）**；q4 生产路径下表是实测最优。

### fused split-KV reduce（#451）

- 只在 splits ∈ {2,4} 融合；s8 不融合仍走独立 reducer，s1 无归约。
- 正确性链：每 wave 写完 partial → `s_waitcnt vmcnt(0)` → `s_barrier`（**必须在 wait
  之后**）→ 原子到达计数 → 最后到达者从 L2 读各 partial 归约并清零信号量。
- 依赖的硬件假设（未获 AMD 正式确认）：同 XCD 内 `vmcnt(0)` 之后的 store 对本 XCD
  其他 CU 可读。
- 数值契约：partial BF16、归约 FP32、输出 BF16/query dtype——融合不降精度。
- fused vs 独立 reduce 最差 relerr ≈ 2.3e-3~2.7e-3，与 BF16-partial 契约一致。

## 2. decode A-B-A 终表（六 PR 栈，终验 2026-09-05）

口径：4×MI350X TP4，MiniMax-M3-MXFP8 + EAGLE3 draft（合成接受率 0.7/0.5/0.4），
真实 36,000-token prefill，无 prefix cache，2,000 输出 token，server 侧 barrier
release-to-drain 计时（`decode TPS = B×(2000−1)/(drain−release)`，prefill 与首 token
不计入）。每臂每 batch 4 个有效同步 wave。

| Batch | A-B-A baseline TPS | 六 PR TPS | 提升 | baseline 漂移 |
|---:|---:|---:|---:|---:|
| 32 | 3,016.7 | 3,044.1 | +0.91% | −0.36% |
| 48 | 3,893.8 | 3,988.7 | **+2.44%** | +0.29% |
| 64 | 4,617.6 | 4,718.1 | **+2.18%** | −0.05% |
| 80 | 4,081.4 | 4,184.8 | +2.53% | −0.35% |
| 96 | 4,380.2 | 4,491.8 | **+2.55%** | −0.08% |
| 112 | 4,651.7 | 4,773.9 | **+2.63%** | −0.48% |
| 128 | 4,943.8 | 5,007.4 | **+1.29%** | −0.14% |

B32–B128 全部 +0.91%~+2.63%，baseline 前后漂移 ≤0.48%；合成接受率逐臂一致，
draft-token 归一化吞吐同向。这是栈级证据，**不能归到单个 PR 头上**。

B64 四 rank profile（每 rank 50 个完整 decode step）：单步 wall 79.057→75.525 ms
（−4.47%）；target 模型 57 层 MSA 2.134→1.525 ms（−28.55%）。

## 3. E2E 终验（report_e2e，2026-09-04，较早的两 wave 版本）

口径：baseline `9093102bc2` vs `merge/msa-decode-xcd` @ `aed1d3bae`；profile 用 B64、
60K–120K 输入、256 输出；target 模型每步 57 次 MSA 调用（draft 的 6 次不计）。

| 指标 | Baseline | 优化后 | 变化 |
|---|---:|---:|---:|
| 57 层 MSA 每步时间 | 1.940 ms | 1.311 ms | **−32.42%** |
| 其中 target attention | 1.611 ms | 1.311 ms | −18.63% |
| 其中 standalone reducer | 0.329 ms | 0 ms | 消除 |
| MSA / decode wall | 2.40% | 1.73% | −0.67 pp |

MSA −32.42% 的大头是融合 reduce 消掉了独立 reducer（0.329 ms/步）。
summed-kernel 占比（2.97%→3.06%）只是诊断量：GPU kernel 时长互相重叠、同步 kernel
含等待（基线 trace 里 `reduce_scatter_cross_device_store` rank0 记 40.17 ms/step 而
rank3 只 4.80），不能当墙钟分解用。

decode TPS（B32–128、24K–48K 输入、2,000 输出，两 wave 均值）：

| Batch | Baseline | 优化后 | 变化 |
|---:|---:|---:|---:|
| 32 | 2,905.6 | 2,947.5 | +1.44% |
| 48 | 3,859.4 | 3,867.6 | +0.21% |
| 64 | 4,494.0 | 4,506.5 | +0.28% |
| 80 | 3,957.9 | 4,012.4 | +1.38% |
| 96 | 4,228.6 | 4,326.1 | **+2.31%** |
| 112 | 4,502.0 | 4,586.1 | **+1.87%** |
| 128 | 4,766.2 | 4,803.9 | +0.79% |

B32–B64 在噪声内；B80–B112 有可测的 1.4%–2.3%；B128 约 +0.8%。
**注意**：此表是更早的两 wave、旧代码树结果，已被第 2 节的四 wave A-B-A 取代为最终
PR 级宣称；保留它是因为 −32.42% 的 MSA profile 分解只有这里有。

## 4. BRANCH_SWEEP：`minimax-m3-amd` vs `opt/msa-decode-xcd`（kernel 级）

两臂同进程交错、各自用自己的 chooser 和 dtype（baseline bf16 + 自己的 chooser；
新分支 fp8 + XCD swizzle + 融合 reduce + 重标 chooser）。`tot` = attention + 该臂
归约 dispatch，µs。`ns = batch×4` 是唯一形状变量。一个 request 的 4 个投机 token
各选 16 块 = **64 次块读**，其中多少块 distinct 决定收益落在两表之间。

64 块读 → **25 distinct**（兄弟重叠重，`shared_blocks 12`）：

| batch | ns | base spl | base tot | new spl | new tot | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 8 | 10.761 | 4 | 8.680 | **1.24x** |
| 16 | 64 | 8 | 12.120 | 4 | 9.200 | **1.32x** |
| 24 | 96 | 6 | 17.960 | 4 | 9.480 | **1.89x** |
| 32 | 128 | 4 | 16.641 | 4 | 9.761 | **1.70x** |
| 40 | 160 | 4 | 20.601 | 4 | 12.760 | **1.61x** |
| 56 | 224 | 3 | 29.120 | 2 | 15.561 | **1.87x** |
| 64 | 256 | 2 | 29.000 | 2 | 15.880 | **1.83x** |
| 88 | 352 | 2 | 38.120 | 2 | 20.800 | **1.83x** |
| 104 | 416 | 2 | 47.321 | 1 | 21.720 | **2.18x** |
| 128 | 512 | 1 | 41.121 | 1 | 22.320 | **1.84x** |
| 176 | 704 | 1 | 55.161 | 1 | 29.320 | **1.88x** |
| 256 | 1024 | 1 | 79.721 | 1 | 44.881 | **1.78x** |

64 块读 → **~63 distinct**（几乎无重叠，下界）：

| batch | ns | base spl | base tot | new spl | new tot | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 8 | 10.680 | 4 | 8.680 | **1.23x** |
| 16 | 64 | 8 | 11.760 | 4 | 9.080 | **1.30x** |
| 24 | 96 | 6 | 18.000 | 4 | 12.120 | **1.49x** |
| 32 | 128 | 4 | 16.640 | 4 | 14.280 | **1.17x** |
| 40 | 160 | 4 | 20.600 | 4 | 17.840 | **1.15x** |
| 56 | 224 | 3 | 29.280 | 2 | 20.840 | **1.40x** |
| 64 | 256 | 2 | 27.081 | 2 | 23.440 | **1.16x** |
| 88 | 352 | 2 | 35.760 | 2 | 30.720 | **1.16x** |
| 104 | 416 | 2 | 44.961 | 1 | 33.401 | **1.35x** |
| 128 | 512 | 1 | 41.521 | 1 | 39.920 | **1.04x** |
| 176 | 704 | 1 | 64.640 | 1 | 58.720 | **1.10x** |
| 256 | 1024 | 1 | 100.162 | 1 | 93.601 | **1.07x** |

全档区间：**25 distinct 下 1.24x–2.18x；63 distinct 下 1.04x–1.49x**。
诚实口径 = 两表之间的**区间**；合成 `shared` 模式的 c 直方图是人造双峰（只有 c=4 和
c=1），且 TB/s 超过 HBM 实测上限 6.7–6.8 的行说明 L2 在供流。per-slice spread>5% 的
行不可信（上表已标剔除规则，区间不受剔除影响）。

## 5. 真实 union 探针（2026-09-05，接替合成假设）

非计时、eager 探针：candidate 代码 + 诊断用 `common/indexer.py` 插桩，B64、真实
36K prompt、q=4 投机验证、256 输出 token、同一个真实 decode barrier；全部 64 请求
进入 decode 后才采样，断言每行 `query_len==4`。

- **116,736 个观测**（1,824 个采样行 × 全部 57 层 MSA，跨 rank）。
- 4 行 top-16 选块并集的 distinct KV 块数：**mean 25.002，p50 25，p90 32，p99 41**，
  观测范围 16–53（理论范围 16–64）。

意义：真实 indexer 的并集均值正好落在第 4 节「25 distinct」那一列上，因此真实收益
大概率靠近上表而非下表；但真实分布有 c=2/3 的中间重叠（合成双峰没有），逐点对应是
近似的。c 直方图（64 次读里 c=4/3/2/1 各多少块）仍是最精确的未决输入。

## 6. 两个正确性修复的影响面

### 6.1 MTP causal mask（aiter 上游 bug，PR #446 vendored 修复）

上游 `pa_decode_gluon` 把允许列窗口按因果延伸 `ext = query_seq_len−1−token_idx`
平移，但**每个 split 实际加载的 tile 范围没平移**：每个 split 边界下方 `ext` 列被加载
它的 split 掩掉、只被没加载它的 split 接纳 → **`ext×(splits−1)` 个 key 从 softmax
静默消失**。

影响面超出 MSA：**任何 `query_length>1` 且 `splits>1` 的调用方**，包括稠密 AITER FA
后端（`rocm_aiter_fa.py` 投机解码时传 `query_length=decode_query_len`）。

误差随 split 数和 qlen 增长（相对 splits=1 的 max relerr）：

```text
upstream   qlen=4:  s2=1.3e-01  s4=4.6e-01  s8=4.3e-01
vendored   qlen=4:  s2=4.6e-03  s4=4.6e-03  s8=4.6e-03   ← 平坦，只剩 bf16 累加顺序差异
```

修法：归属改用原始列轴，因果性作为全局界只施加一次；零额外 tile。判据是
**split 不变性**（单 split 无边界，其他 split 数必须与它一致）。
已知残留：修复**只覆盖非 sliding-window 分支**——`rocm_aiter_fa.py` 会传真实
sliding window，sliding-window 模型 + qlen>1 + 多 split 仍丢 key（新测试传
`sliding_window=-1`，覆盖不到）。vendored 基底是 aiter 0.1.19（md5
`672e779b48c362656afa71b17b11abe6`），升级 aiter 需重放三处 hunk。

### 6.2 多 KV head 调用方崩溃

vendored wrapper 只有一个 launch 点，无条件传 9 个只有单 KV head kernel 声明的参数。
稠密 FA 后端也 import 这个文件，所以**任何 KV head>1 的 ROCm 模型第一次 gluon decode
就崩**，与 MSA 无关。已复现、已修、已验证。关联修复：多 KV head kernel 没有
`GRID_PADDED` 边界检查 → 不给它 padding；融合归约缺 sinks 项时显式报错而不是静默丢
归一化；`xcd_count=0` 显式 override 路径拒绝 `ZeroDivisionError`。
[待复核] 该崩溃修复在 OPTIMIZATIONS.md（2026-09-03）中标记为「未提交」；
handoff_decode_msa_pr（2026-09-06）称 PR 测试已含 1/4-KV-head 分组用例、kernel 与
XCD 分组逻辑支持 4 KV head（sibling group = `decode_query_len × num_kv_heads`），
大概率已并入 #446/#450，迁移后需对 PR 分支核实。

## 7. report_codex 的留存与作废

**其 M32 +23.2% / M64 +8.2% / "bit-exact" 结论已被推翻**（合成 top-k `shared=15`
端点 + 零 KV cache + 硬编码 split=4 弱基线，真实散布选块下分组是 0.64x 回退；见
`report_m32_membership.md`）。只保留它的 AMD↔CUDA 术语对照表：

| AMD/通用术语 | CUDA 对应术语 |
|---|---|
| CU, Compute Unit | SM, Streaming Multiprocessor |
| Workgroup | CTA / thread block |
| Wave64 / wavefront | Warp（AMD 64 lanes，CUDA warp 32 threads） |
| VGPR | Registers per thread |
| LDS | Shared memory |
| MFMA | Tensor Core MMA instruction |
| HBM/TCC | Global memory / L2 路径 |
| CPS, context partition size | Attention `BLOCK_N`，CTA 每轮处理的 KV tokens |
| Split-KV | 多 CTA 分割 KV context，最后 split-K reduction |

## 8. 未决

1. 真实 indexer 的 **c 直方图**（不只是并集宽度）——决定收益在 BRANCH_SWEEP 区间
   内的精确位置。
2. MTP mask 修复未提上游 aiter；sliding-window 分支未修（见 6.1）。
3. chooser 在 q1 / ns≈256 的 13% 误选（仅当存在非投机 q1 decode 才触发）。
4. fused-reduce 的 XCD L2 可见性假设未获 AMD 正式确认。
5. 无新的任务精度（GSM8K/MMLU）评测——现有证据是 kernel 逐位一致 + 真实模型 E2E
   全请求完成。
