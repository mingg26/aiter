# trick（已回退档案）：用 LLVM asyncmark 精确放宽 GEMM K 循环的 vmcnt

> 范围：gfx950 上把 A 操作数 global→LDS 拷贝换成 async DMA（`BufferLoadAsyncLDS128b`），用 `rocdl.asyncmark()`/`rocdl.wait_asyncmark(n)` 分组记账，让编译器按组龄推出精确 `s_waitcnt vmcnt(N)` 替代保守 `vmcnt(0)`。
> **状态：两轮实测均无净收益，已回退，不在 f7a96515 生产代码中。本文是纯技术档案，记录方法与重开条件。**
> 来源：`trick_vmcnt_asyncmark.md`（源）、`plan_vmcnt.md`、`side_s2_async.md`（EP8 终局）。

## 1. 终局结论（先看这个）

| 轮次 | 基线 | 结果 |
|---|---|---|
| 09-10 EP8 b256（S2+L2+Combine 窗口） | tail 版 | ratio 1.00096，CI 跨 1，**无净收益** |
| 09-13 EP8 全尺寸 31 档 | jointtail 基线（aiter `860181bc6`） | **几何平均 +0.42%（净亏）**，22/31 档 CI 显著慢、6 档显著快、3 档跨零 → **支线关闭** |
| 09-08/09 EP4 大 batch（S2-only） | — | 赢 1.35%–3.88%，见 §6 对照段 |

负机制：ATT 显示 **VMEM wait 降了但 VMEM issue stall 升了**（exposed/MFMA 49.1→68.3 / 44.7→105.7）——等待从 waitcnt 搬到发射口，净零。一句话（`tool_root_cause.md` 早有此结论）：**K 循环尾的 `vmcnt(0)` 是兑现点不是病因，放宽它只值 ~2%**。

## 2. 机制原理

- 同步 LDS-DMA 的问题：FlyDSL 直发 addrspace(3) 访问、memoperand 不带 `!alias.scope`，LLVM `SIInsertWaitcnts`（`:1075`）把每条 LDS-DMA 落到通用保守记分板 → 任何 LDS 读被逼成 `vmcnt(0)`。
- async 版被 `isNonAsyncLdsDmaWrite`（`SIInsertWaitcnts.cpp:422`）排除出保守记分板；pre-GFX12 上 `shouldUpdateAsyncMark`（`:430`）返回 LOAD_CNT 即 vmcnt，故 gfx950 逃逸路径存在。依赖用"mark 年龄"表达，后端在 `wait.asyncmark` 点回算真实 `vmcnt(M)` 立即数。`MaxAsyncMarks = 16`（`:826`）。
- 边界：硬件是否有独立 async 计数路径本地无据（有界 ISA 原型仍把 async DMA 计入同一 vmcnt FIFO）；可写死的只有"变的是编译器插等待的策略，立即数由后端按组龄推导而非手算"。
- **工具链要点**：ROCm 7.2.3 自带 LLVM **22**（`/opt/rocm/llvm`）**没有** `llvm.amdgcn.asyncmark` 三个 intrinsic；真正做 codegen 的是 **FlyDSL 自带的 `libFlyPythonCAPI.so.24.0git`（LLVM 24）**，它全有。所以必须用 FlyDSL 一等 API，不要走 ROCm LLVM。

FlyDSL 0.3.2 API（`flydsl/expr/rocdl/universal.py:58-77`、`cdna4.py:42-62`）：

- `rocdl.cdna4.BufferLoadAsyncLDS128b()` —— copy atom；与同步版区别是**后端不会在 LDS 数据被读前自动插 vmcnt**，完成度靠 mark 组手动追踪。
- `rocdl.asyncmark()` —— 关闭当前 async 操作组。
- `rocdl.wait_asyncmark(count)` —— 等到"最多剩 count 组未完成"，count 必须编译期常量。

## 3. 配方五要点（以 S2 gemm2 为例，127 行 diff）

不变量：**"比 tile kt 新的组数恒为 1"**——每个 K 迭代无论是否真发 DMA 都无条件关一个组（守卫分支没发就关空组），尾部不需要依赖 trip count 的守卫。

1. A tile 的 DMA atom 换成 async 版。
2. K 循环每步：B 预取 → `wait_asyncmark(1)`（放掉最老组 = 本 tile 的 A，容忍 kt+1 在飞）→ WG barrier → ds_read A → 发 A(kt+2) async DMA → **无条件 `asyncmark()`** → scale → MFMA。
3. prologue 每个预取 tile 后各关一组；mark 放无条件执行的汇合点。
4. epilogue 前 `wait_asyncmark(0)` 显式断 A LDS slot 与 epilogue C staging 的跨 tile WAR（async 组不被编译器自动等待覆盖；稳态下是 no-op）。
5. **B 的 load 不打 mark**——普通 VMEM 会被后端按年龄自动计入算出的 `vmcnt(N)`（这是相对手算的关键优势）。

两个配套重构（缺一不可）：

- **尾部 peel**：稳态循环只跑 `[0, K−kStages)` 定 trip，最后 kStages=2 步移进单独的运行时上界循环（不发新 A DMA、两步关空组维持组龄）。运行时上界是为防常数 trip 展开；配套 launch 断言运行时 K == 编译期固定 K。静态 peel 会让 VGPR 128→152、驻留 4→3；尾部普通循环版 VGPR 回 128、驻留回 4、主体 `vmcnt(2)` 保留。
- **K0 拆分**（GEMM 第一个 K 迭代剥出，与测量方法论的 K0/K8/K16 settling 档无关）：把首迭代 B0 就绪等待隔离出重复体，否则被所有稳态迭代共享。

## 4. ISA 前后对照（b256 S2，K=3072/BK=256，12 步/tile）

- 原版稳态：MFMA 后、19 条 B/scale carry 拷贝前是**单条 `s_waitcnt vmcnt(0)`**——排空包括 2 条在飞 A DMA 的全部 VMEM，B 预取全压在这条等待之后。
- async+K0 版稳态：MFMA 前 `vmcnt(14) lgkmcnt(5)`（本轮新发的都不用等）；MFMA 后 carry 拷贝走**阶梯式等待** `vmcnt(11)→(10)→…→(2)`，每条拷贝只等自己的生产者；循环出口只剩 2 条最新 A DMA 在飞。
- 尾部审计：**K10（倒数第 2 步）的 `vmcnt(0)` 是必要的**（只有 B11/scale 预取在飞，必须排干；实测 stall 646/423 cycles，无法放宽）；K11 无新 VMEM，无放宽空间。

## 5. 两轮实测细节（均已回退）

09-10 轮（b256，S2+L2+Combine 窗口，12 对 balanced AB/BA，8s warmup）：

| 对比 | ratio | 判定 |
|---|---:|---|
| async+K0+peel vs tail | 1.00096 [0.99878, 1.00332] | 跨 1，无净收益 |
| async+peel（无 K0）vs tail | 1.00496 [1.00296, 1.00684] | 显著慢 0.5% |
| async rolled（不动循环）vs tail | 0.99903 | 跨 1，且 ISA 与 baseline 逐字节相同（见陷阱） |
| early2+async vs early2 原 GEMM | 159.764 vs 156.257 µs | 慢 2.244%，0/12 |
| async + B 权重改 NT | — | **+10.83% 回归，否决** |

资源代价：VGPR 147→157（K0 版），SGPR spill 9→13。

09-13 轮（EP8 全尺寸，jointtail 基线 `860181bc6`，12 对平衡 AB/BA ×200 replay）：几何平均 **+0.42% 净亏**，22/31 档 CI 显著慢。机制澄清：

- b128 回归真机制是**"等"不是"撞"**：wave 在步首 `wait_asyncmark` 空等 A 的 HBM→LDS 到货；b128 负载全均匀使错位逐拍复发，b200 不均则抖散。早期 +39.3% MFMA_BUSY 签名测错了代码（连环编辑中间态，ISA 指纹比对证伪）；跨 CTA 相位假说被 cta_skew 实验否决。
- b136 旧基线 −2.49% 的"优势"与 jointtail 的收益**同源**，已被后者吸收——**评估支线必须在新基线上复测**。
- jointtail 基线上 b128 回归消失（−0.54%，CI [−2.08,+1.01]），但全尺寸无正收益，支线关闭。
- 更正：小档 S2 驻留为 **2 CTA/CU**（VGPR231/LDS33088B），不是按 gfx942 规格推的 1。

**rolled 循环陷阱**：初版只打 mark 不动循环，编出的 ISA 与 baseline 逐字节相同——稳态发 DMA、尾两步不发，两条路径在 carry copy 汇合，LLVM 只能用保守 `vmcnt(0)`。不把尾部从稳态循环分离出去，async mark 等于白打。

**卡死记录（根因未证实）**：09-08/09 轮 256/512 两次在通过正确性门禁后于 warmup 卡死（四张 GPU 100% 忙、数分钟无样本）。栈只证明在等 HIP event。"另一会话并发干扰"是未验证假说，不能写成已确认原因。处置：隔离 enroot runtime 目录、独立 `AITER_JIT_DIR`/`FLYDSL_CACHE_DIR`、对 rank0 主动 SIGABRT（先禁 core dump）抓栈。

## 6. 对照：同一技术在 EP4 大 batch 曾有效（c495b60a2）

aiter 侧枝 `c495b60a2`（09-08，开关 `g2_async_a_lds`，最终形态 `_aal1_peel2tl`）在 EP4、S2-only 完整路径上（同进程 paired A/B、60s warmup、12 对）：

| batch | before µs | after µs | 收益 |
|---|---:|---:|---:|
| 256 | 180.45 | 175.56 | 2.71% |
| 512 | 232.17 | 226.60 | 2.40% |
| 1024 | 330.32 | 325.18 | 1.55% |
| 2048 | 613.79 | 591.68 | 3.60% |
| 4096 | 1152.53 | 1107.76 | 3.88% |
| 8192 | 2102.91 | 2074.55 | 1.35% |

8192 档由 LLVM 自动生成 `vmcnt(2)`、VGPR 回 128、驻留回 4 CTA/CU、零 spill。但 `async_a_lds` 始终默认 False、未按 batch 自动选版本，**从未进生产**；这些数字不在任何端到端结果里。主线自 `1cf2322cd` 起无 asyncmark。

**展开类实验全部无效，但"部分展开"和"全展开"是两件不同的事，别混：**

- **部分展开（U=2/4/6/12）没能放松 wait**：U=2/4 + commit 拷贝 → VGPR 133/185，可 vmcnt
  直方图只剩 {0,1,2}，因为**拷贝把 B 串行化了**，放松幅度塌回 1~2；U=6/12 + parity
  role-swap（无拷贝）→ VGPR 224，vmcnt 仍只有 {0,1}。每步加 `sched_barrier(0)` 则 ISA
  完全不变（上一步 mfma 后本来就有一个），纯 no-op。
- **K 循环全展开是另一种死法（代价，不是失效）**：EP4 上"展开 + async"是 **+1.06%**、
  VGPR **128→214**、驻留从 4 CTA/CU 掉到 **2 CTA/CU**；EP8 b128 的移植版是 **126.27 µs
  的灾难**。两处独立测到同一结论。

可复用判读：**判断"wait 有没有真的放松"要看 vmcnt 直方图，不要看 VGPR**——VGPR 涨了
不代表 wait 放松了（U=2/4 就是 VGPR 涨到 133/185 而直方图仍塌在 {0,1,2}）。反过来不成立：
直方图变了也不能据此断定根因，源里明确"不能由等待直方图证明回边处理是根因"
（`plan_vmcnt.md:135`），它只是必要条件。

## 7. 重开条件（满足任一再说）

1. 目标 kernel 的 ATT 显示 waitcnt stall 占比大**且** VMEM issue 侧有余量——本轮两者此消彼长是净零的直接证据，重开前先测这两个数。
2. 循环结构本身变了（K 步数、流水深度、A/B staging 布局），使稳态又被压成单条 `vmcnt(0)`——先编 ISA 确认，是则 §3 配方直接可用。

生产中可参考的 async 样例（非本实验）：`aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py:719-744`（prologue 每组一 mark、主循环 `wait_asyncmark(stages-2)`、drain 段递减）。可复用基建：worktree `aiter-s2-async-8601` 的 `kernels_jt_async/`（四旋钮 `cta_skew`/`drain_vmcnt`/`probe_drain0`/`a_depth`，默认=现状、关闭时 code object 逐字节不变；`probe_drain0=True` 默认是探针不是移植版）与 `bench_v2_async.py`+`s2jtv2.sh` 全尺寸批量测量。

**四旋钮各值多少（旧基线实测，四个都追不回同期的 sync 对照臂 `ctl`——b128 的 ctl 实测
103.61 µs，不是什么理论上限；别重扫）**：
`drain_vmcnt=2`（排 B）捞回约一半，113.8→**111.3 µs**；`a_depth=3`（A 加深一级）再收
约 **1.3 µs**（→112.5）；两者组合最好 **+1.9%**。`cta_skew`（CTA 入口四档相位偏移）
对 b128 **回归零效果**——冲突取决于 CTA *内部*"DMA 完成时刻 − MFMA 窗口"的相对偏移，
整体平移改变不了它，**跨 CTA 相位假说已被实验否决，不要再试**。
