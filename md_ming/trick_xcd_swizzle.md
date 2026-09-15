# trick：MI350X XCD 分组与 sibling swizzle

范围：MSA 稀疏注意力 decode 的 XCD 硬件事实、`XCD_SIBLING_GROUP` swizzle 的机制与
数字、DEP（attention 数据并行）拍平顺序踩坑。
有效性基准：2026-09-03/04（`opt/msa-decode-xcd` tip `22715519b2`；六 PR 栈中对应
PR #450 `pr/msa-xcd-launch-placement`）。硬件事实在 MI350X / SPX / ROCm 7.2.3 上
5 次 launch × 3 进程完全一致。
来源文件：`OPTIMIZATIONS.md`、`HANDOFF.md`、`dep.md`、`handoff_decode_msa_pr.md`；
§3 的 PMC counter 名出自 `plan_m32_membership.md`（本目录 `tool_measure.md` §4 有同一套 counter 的用法）。

## 1. XCD 硬件事实（硬件读出来的，不是推的）

探针：`benchmarks/kernels/minimax_m3/xcd_workgroup_map.hip`，用 `XCC_ID` 寄存器 dump
workgroup→XCD 映射。

- **XCD = `linear mod 8`**；比 8 粗的都不是（mod 2 / mod 4 都不成立）。
- **它是旋转不是恒等**：映射是 `[6 7 0 1 2 3 4 5]`，workgroup 0 跑在 XCD 6 上。
  **代码只能依赖同余，不能依赖旋转**——把 `linear mod 8` 写死成恒等会是错的。
- **grid.x 是 8 的倍数 ⇒ 同一 sequence 的所有 split 100% 同 XCD**（256/352/512 全中）；
  **不是 8 的倍数 ⇒ 0%**（252 和 100 一个都不中）。**padding 是硬要求，不是保险。**
- XCD 数**运行时查询**：`hipDeviceGetAttribute(..., hipDeviceAttributeNumberOfXccs
  (=10018), ...)`，init 时查一次缓存进 `_DEVICE_GEOMETRY`，不是每次 kernel 调用都查；
  padding 取 `lcm(查询值, 8)`，所以查询低报到 8 的任何约数也仍然正确。特性 fail-closed：
  仅当设备是 gfx950 且报告 8 XCD 才启用。MI350X/MI355X 都是 8，这条风险已关闭。
- fused split-KV reduce 的正确性正依赖这条几何：`linear = x + z·grid.x`，grid.x 是 8
  的倍数 ⇒ `z·grid.x ≡ 0 mod 8` ⇒ 同 sequence 的 split 从不离开读它的那块 L2。

## 2. 问题与做法

**问题**：逐 token launch 把一个 request 的 4 个投机 token 放成**连续的 grid.x 索引**，
而 workgroup 按 `linear mod 8` 分 XCD → **4 个连续索引 = 4 个不同 XCD、4 个不同
L2**，兄弟 token 共享的块被 HBM 读 4 遍。早期结论「L2 兜不住兄弟重用」（TCC miss
98.3%）不是 L2 太小，是**兄弟从来没在同一个 L2 上**。

**做法**：重映射 grid.x（`XCD_SIBLING_GROUP`）。XCD `e` 的第 k 个 slot 服务 request
`(k/G)*8 + e` 的第 `k%G` 个 token：一个 request 的 token **共享 XCD，且是该 XCD
收到的连续 G 个 workgroup**——同时启动，第二个读共享块的人发现它还在 L2。
grid.x 补齐到 `XCD 数 × token 数` 的倍数，多出的 workgroup 立即返回。
**输出逐位不变**——只改了哪个 workgroup 做哪个 sequence，是纯置换。

排布效果：swizzle 下 0/128 个 request 的 token 跨 XCD；4 个 token 的启动间隔中位数
**0 tick**，而排布不变时是 **84–95 tick**（整个派发窗口 150）。

## 3. 数字

PMC（b128/q4，`TCC_EA0_RDREQ_DRAM_32B_sum × 32`）：

| | HBM 实读 | L2 命中 |
|---|---:|---:|
| 之前（fp8 commit） | 135.5 MB | 0.9–1.7% |
| + swizzle | **53.6 MB** | **60.6%** |

25/64 distinct 预测 39.1% 流量、39/64 = 60.9% 命中；实测 39.6% / 60.6%。
**兄弟重用被吃干了**；命中率没超过预测，也说明命中来自兄弟重用而不是块连续假象。

单项加速（splits=1）：**b64 1.17x，b128 1.72x，b256 1.65x**。
（全档端到端区间见 `report_msa_decode.md` 第 4 节：25 distinct 1.24x–2.18x /
63 distinct 1.04x–1.49x。）

为什么比 merged M32 强：swizzle 拿到**全 4 路**去重（64→25），M32(g=2) 只有成对的
（64→38）；而且**没有 tile 浪费**（M32 一个块平均只有一半行是真成员）。

## 4. DEP（attention 数据并行）拍平顺序坑

头数事实（已核实）：全局 **q_h=64、kv_h=4**；**16 是 group size（64÷4）**，是 kernel
的 MFMA M 维和 partial `[16,128]` 的那个 16，**不是 KV 头数**。TP4 rank-local 16/1，
DEP4/DEP8 每卡持全量 64/4。

TP4 → DEP4/DEP8：kernel 与 kernel 级 harness 不用动，改动集中在模型接入层 3 个点：

1. **拆三处 `kv_h==1` 硬门**：`common/sparse_attention.py:112-120`、
   `amd/sparse_attention_msa.py:44-48`、`sparse_pa.py:431`。
2. **拍平顺序破坏 sibling 语义（唯一静默坑）**：生产路径把 (token, kv_head) 拍平进
   sequence 维——`index_topk.py:546-547` emit `(block_start+pid_q)*NUM_KV_HEADS+pid_h`——
   同 request 的 q 个投机 token 在 grid.x 上**相隔 kv_h=4 位**。
   `xcd_sibling_group=q` 会把**同一 token 的 4 个 kv head 错当 sibling**——它们不共享
   任何 KV 块，L2 去重的目标对象错了，形状上无报错、收益静默消失。
   **修法二选一**：topk emit 换成 (kv_head, token) 序；或 remap 改成 stride 感知
   （sibling = 相隔 kv_h 的 q 个 token）。必须配 `_LAST_LAUNCH` 式断言/置换检查防回归。
3. **indexer 的 `num_idx_heads==1` gate 回落**（`amd/ops/index_topk.py:108,162,198`
   等）：q4 MFMA 特化、`_MAT_ARGMAX_Q4_TUNING`、balanced-score gate 会回落到未调优
   路径。

次要项：scratch/semaphore 按展开后 num_seqs 分配，每卡总量不变；union 探针语义不变
（仍按 (request, kv_head) 的 16q 次块读量 union），但要确认采样覆盖全部 4 个 kv head。
DEP4 与 DEP8 对 kernel 无区别（拍平后 ns 相同或减半），**DEP4 下 chooser 断点不用
重标**。

**禁忌**：别让 kernel 原生吃 kv_h=4——`xcd_capable = num_kv_heads==1 and ps`
（`vllm/third_party/aiter_pa_decode_gluon.py`，在 `vllm-serve-xcd` 树是 `:6186`、在
`vllm-msa-xcd` 树是 `:6256`；**这个文件是 vendored 的，行号随树漂，一律按符号
`grep -n xcd_capable` 定位**）会整体关掉 XCD remap + 融合 reduce + padding，
走另一个不接收这些参数的多头 kernel，swizzle 收益（HBM 135.5→53.6 MB）全没。
保留拍平是唯一保住主线优化的路。

## 5. 相关正确性要点（详见 report_msa_decode.md §6 与 fused-reduce 链）

- fused reduce 只在 splits∈{2,4} 开；正确性链 = per-wave `s_waitcnt vmcnt(0)` →
  `s_barrier`（必须在 wait 之后）→ 原子计数到 S → 最后到达者归约。少一环静默错，
  失效样子是读到上一 decode step 的 partial（数值合理，不是 NaN）。
- 未获 AMD 确认的假设：vmcnt 归零的 store 同 XCD 其他 CU 可读。
- 读侧别用 `.cg`「加固」——Triton 在这里把它降成 `nt`（驱逐提示不是 bypass），
  归约要读 `max_logits` 三遍，方向是反的。
