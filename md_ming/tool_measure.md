# kernel 级测量规范（终稿）

> 范围：单 kernel / EP 多 rank kernel 的计时、profiling 与测量纪律。serving 层见 `tool_e2e.md`。
> 有效性基准：aiter mega_moe 终态 commit `f7a96515aa8e3b7c1e41506c37d49b78651ce8ad`（2026-09-15）；
> 方法论本身与 commit 无关，长期有效。
> 来源：`tool_measure.md`、`redo.md`、`handoff_megamoe.md`、`e2e.md`（仅抽通用结论，不含 job 号/实例路径）。

## 0. 一句话协议

**同进程、同输入、balanced paired A/B**：12 对配对样本，chunk ≤200 次 replay，8 秒 time-based warmup，
主结果取 pair log-ratio 的 geometric mean + paired bootstrap 95% CI。正确性 / source / dispatch / graph
任一门禁失败，该轮全部时间数据作废，修好 harness 整轮重跑，不事后删样本。

## 1. 先选对工具

| 问题 | 首选工具 | 注意 |
|---|---|---|
| 两个实现谁更快 | HIP event + CUDA graph paired A/B | 优先同进程；EP collective 必须全 rank 同序 |
| 单 dispatch 绝对 device 时间 | `rocprofv3 --kernel-trace` | profiler 扰动工作点，只在同一 traced run 内比 |
| HBM/cache/wave wait/指令构成 | rocprofv3 PMC（多 pass） | collection 会扰动或串行化 |
| kernel 内指令/源码阶段 stall | ATT thread trace | 只覆盖选中 CU/wave 和少量 dispatch |

逐层深入顺序：**无 profiler 计时 → kernel trace → PMC → ATT**。任何 profiler 下的 latency
都不能替代第一步。

## 2. paired A/B 计时闭包

1. 两臂同进程、共用同一输入 tensor/workspace；先过正确性（纯置换必须 bit-identical），并用
   source hash、kernel 名、VGPR/grid 断言两臂确实不同。
2. 各 capture 一次 CUDA graph；交错运行 ≥8 秒 time-based warmup（按时间不按调用次数）。
3. 预生成固定 seed 的 balanced order，`AB`/`BA` 等量，不看结果改顺序。
4. 12 个 paired samples；每 sample 覆盖约 0.1–0.2s device work（replay 20–400 次），A/B 同 replay 数。
5. 分块入队，chunk ≤200 次 replay，每 chunk 一对 HIP event、末尾同步；不在每次 replay 内同步。
6. 每对算 candidate/base ratio；主结果 = pair log-ratio geometric mean + paired bootstrap 95% CI。

```python
total_ms = 0.0
done = 0
while done < replays:
    chunk = min(200, replays - done)
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(chunk):
        graph.replay()
    end.record(); end.synchronize()
    total_ms += start.elapsed_time(end)
    done += chunk
```

报告：raw samples、两臂 mean/median/CV、pair effect 全部值、正向 pair 数、geomean、bootstrap CI。
replay 不是独立样本。

低噪声原因的反面即判读边界：同进程消编译/状态差，相邻配对消温漂频漂，balanced 消顺序偏差。
**配对消不掉参照臂自身的 run 间漂移**——实测同 code object 跨 run：b32 0.37%、b64 1.6%、b128 0.20%；
b256 上 A 臂漂 1.31%、B 臂 0.77%，配对比最差 1.85%。含义：排序候选用跨 run 跨度更小的臂；
报加速比才用配对比；亚百分比结论必须有重复跑，单点不成立。

**两条数值地板（小于它的效应不成立，直接别声称）：**

1. **跨 run 绝对时间地板**：小 batch（T≤256）按 **3 µs** 用，大 batch（T≥512）按 **8 µs** 用。
   注意 1.5 µs 这个旧值只由 b128 的两个同 ISA build 得出（`handoff_megamoe.md` 的 T=128
   两个逐字节相同 ISA，漂移 1.54 µs），**低估了其他档**。可查的反例在
   `megamoe-s1-1k-prod/bench/dep8_best_configs.json` 的 `96` 条目注记里：
   *"Four further block_n=256 points from the same session sit at **269.614-273.035**"*
   ——同一 session、同配置四个点跨 **3.42 µs**；该条目主点本身也是
   *"273.566 us mean over two interleaved pairs, range [271.048, 276.083]"*。
   推论：**对照必须与候选在同一窗口相邻交错采样，不要复用 20 分钟以上的旧基线点**。
2. **null 自比较地板 ≈ 0.2%**：同一个 baseline graph 在两个标签下 replay 的 null 对照，本身
   就有约 0.2% 的偏差，**而且 12 对的 CI 会排除 1**（实测 26 轮 null 最大偏差 0.238%）。
   所以"配对 CI 不跨 1"在 **0.3% 以下不构成收益证据**。偏差在 AB/BA 两种次序下同号，不是
   pair 内顺序效应。*（机理是推断、未验证：`critical_us` 取 8 rank 最大值，带某个标签的
   那次采样只要有单 rank 掉队就会被抬高，balanced 抵消不掉这种上尾。数据本身只支持
   "存在系统性偏差"这一步。）*
   做法：**每轮都跑 null 自比较，并排列出候选比值与 null 比值，不要拿 null 去扣减**。
   候选效应若不明显大于 null，结论写"未建立收益"。候选/null 的差值只能当诊断量。

正式采样只在 sclk/功耗/温度进入稳定区间后开始（`rocm-smi --showgpuclocks --showpower --showtemp --json`
记录 warmup 前后）。

## 3. EP / 多 rank / persistent kernel

- 实验单位是完整多 rank 进程组。所有 rank 以相同顺序构造 arm、capture、warmup、replay，
  replay/chunk 数完全一致；rank 0 预生成并广播唯一 arm 顺序，不允许各 rank 独立随机化。
- **任何套着 collective 的循环，迭代次数必须是固定值或跨 rank 商定过的值，绝不能由
  wall-clock / rank 本地计时推出来。** §0/§2 的"8 秒 time-based warmup"是**单进程**口径；
  多 rank 下直接照搬（`while time.perf_counter()-t0 < warm_s:`）会让各 rank 跑不同次数的
  collective，进程组失同步、NCCL 挂死（本项目实际踩过：`summary_megamoe.md` 记
  "harness replay 次数分歧致 NCCL 挂死"，修法是 `time_arm` 返回前对 replay count 做
  `all_reduce(MIN)`）。现象长得像通信 bug、不像 harness bug，一次挂死就烧掉一个完整的
  collective watchdog 超时。*（经验推断、非本项目记录：签名通常是 watchdog timeout 且各
  rank 的 "Last enqueued NCCL work" 计数差一。）*
  做法：warmup 和采样都用固定次数；次数确实要自适应时先 `all_reduce(MIN)` 再用；一次性探针
  里把 `init_process_group(timeout=...)` 设短，让错误一分钟内暴露而不是十分钟。
- capture 前后及每个 sample 边界做进程组 barrier；任一 rank 失败整组作废。
- graph-capture gate 逐配置重过：eager/captured 数值一致、replay 深度覆盖最长 sample、无 hang、
  无陈旧 epoch/signal。
- 每 rank 本地 HIP event 计时后汇总，主报 **`max_rank_device_ms = max(per_rank_ms)`**，
  同时存 per-rank/mean/max-mean。**rank 时间绝不能相加**；EP 不能只报 rank 平均。
- CPU barrier（`monitored_barrier`、`torch.cuda.synchronize`）不提供跨 GPU 同时起跑；
  "所有 rank 同序 replay" ≠ "所有 GPU 同时启动"。
- 稳态测量用 K 次不计时 settling replay + N 次 measured；K 必须做敏感性验证（K=0 vs K>0 的
  paired ratio 稳定性），不能看结果挑 K。b8192 已证实窗口敏感（K0 vs K16 的 B/A 差 +0.8975%），
  不得预设启动边界对亚百分比优化无影响。

## 4. rocprofv3 用法

### kernel trace（绝对时间）

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace --kernel-include-regex '<target-regex>' \
  --output-format csv --output-directory <outdir> -- <cmd>
```

- `duration_ns = End_Timestamp - Start_Timestamp`；`Grid_Size_*` 单位是线程不是 workgroup。
- gfx950 上 `LDS_Block_Size=0`/`Accum_VGPR_Count=0` 可能只是字段未填充，不能据此断言不用 LDS/AGPR。
- 固定 N 次 warmup 后用 `--kernel-iteration-range '[11-13]'` 只采 formal dispatch。
- **traced 与 untraced 的绝对值不能混比**（trace 改变设备工作点）。
- 交错 trace 用显式 separator kernel（如 `separator.add_(1)`），不用可能 lower 成 async memset 的
  `zero_()`；parser 硬校验 slice 数、dispatch 数、grid mismatch==0，任何 mismatch 整轮作废。

### PMC（整个 dispatch 的计数）

先 `rocprofv3-avail list --pmc` + `pmc-check` 验证可组合性（counter 名与组合性随 GPU/ROCm 版本变）。
MI350X/gfx950 已验证分组：

```text
sq-wait: SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS
sq-mix:  SQ_BUSY_CU_CYCLES SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_MFMA SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_VALU
sq-ops:  SQ_INSTS_VALU_TRANS_F32 SQ_INSTS_VALU_CVT SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_ADD_F32
tcc:     TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_DRAM_32B_sum TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum
tcp:     TCP_TCC_READ_REQ_sum TCP_TCC_READ_REQ_LATENCY_sum TCP_PENDING_STALL_CYCLES_sum TCP_TCP_TA_DATA_STALL_CYCLES_sum
```

每组一个全新进程、输入完全相同、一次只采一个兼容组；各组按相同 shape 汇总，不假装同一次 dispatch。

派生式（两条是踩坑换来的）：

```text
issue_wait%      = SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES
MFMA_busy%       = SQ_VALU_MFMA_BUSY_CYCLES / (4 * SQ_BUSY_CU_CYCLES)   # 分母必须乘 4
TCC_miss%        = TCC_MISS / (TCC_HIT + TCC_MISS)
HBM_bytes        = TCC_EA0_RDREQ_DRAM_32B_sum * 32
TCP_read_latency = TCP_TCC_READ_REQ_LATENCY_sum / TCP_TCC_READ_REQ_sum
```

- **MFMA_busy% 分母乘 4**：`SQ_VALU_MFMA_BUSY_CYCLES` 按 SIMD 累加（实测恒等于
  `SQ_INSTS_MFMA × 32`），`SQ_BUSY_CU_CYCLES` 按 CU 计；不乘 4 曾把 18.7% 误报成 75%（已撤回）。
- **`SQ_WAIT_ANY` 与 `SQ_WAIT_INST_ANY` 不保证子集关系，不能相减**（b200 S2 实测
  INST_ANY 44.9% > WAIT_ANY 41.3%）；两个数只能分开当信号强度读。
- `SQ_WAIT_INST_LDS` 接近零是强排除证据（等待不在 LDS/ds_read 链上）。
- EP 八卡采法：只有 rank 0 套 rocprofv3，其余裸跑保持通信路径真实；PMC run 的墙钟不得用于排序。
- credit-stall/latency 类计数跨硬件实例求和，不能当 kernel wall-time 占比；只有比值可跨 arm 比。

### ATT（指令时间线）

`--att-library-path` 传 decoder **目录**（定位 `librocprof-trace-decoder.so*`，不照抄版本路径）。
排查顺序（实测定型）：**`.att` 字节数 → `output generation` 耗时 → 换 CU**。空解码不等于空 trace：
四次不同 CU 都拿到 22KB–1.3MB 非空 `.att` 但解码全空，问题在 decoder/镜像组合而非 CU。

两个 harness 级前提，任一不满足 ATT 白跑：

- **解码在进程退出时才做**——被 profile 的 rank 必须干净退出；后续门禁失败触发的 SIGTERM 会把
  rocprofv3 打断在解码中途（`ui_output` 目录存在但空 + `caught signal 15`）。profiler 下无效的
  门禁要降级为 advisory（打日志，不静默跳过）。counter 组不受影响（逐 dispatch 落盘）。
- rocprofv3 接管 dispatch 回调后该 rank 上 torch profiler 看不到任何 kernel；依赖 torch trace 的
  dispatch 审计必须整块跳过。

ATT stall share 只描述被采 wave，不能外推成整个 kernel 的 wall-time 百分比。

## 5. Case / workspace gate

- KV/输入填非零可复现数据；全零输入测不出精度回归（mask bug 要随机值才显形）。
- 不用 `--shared-prefix` 当访存 workload（全 batch KV 常驻 LLC，作废一切访存测量）。
- synthetic top-k/route 分布必须接近真实；无真实 trace 时用独立均匀随机并报告 union width/overlap。
- 背靠背 replay 同一份 KV 会复用 cache；工作集 ≤ LLC 时至少比 hot/cold 并用 TCC/DRAM counter 验流量，
  不按逻辑 token 数推 HBM bytes。
- 带宽分母写清 main kernel 还是 main+reducer；真实 DRAM 流量用 counter 算。
- 共享 workspace 的 builder 可能让后构造的 arm 覆盖前一个：每 arm 运行前重建（计时外）。
- 每个源码替换/开关必须 assert 命中。

## 6. 判废条件（任一即停）

正确性/bit-identical 失败；两臂实际加载了相同代码或 source gate 不匹配；shape/grid/split/dispatch 数
不匹配；formal sample 内发生 JIT；GPU 被共享或 raw slice 出现无法解释的双峰/离群（per-slice spread
>5% 整轮作废并查因，不用中位数"洗掉"污染）；必要 kernel 漏计或 benchmark-only copy/reset/sync 混入；
**traced 和 untraced 数字放进同一绝对时间比较**。

### 6.1 严格协议的额外门限（`strict_measurement_passed`）

默认用 12 对快筛；**收益很小、或两轮方向反转时才升级到严格协议**（用过的规格：48 对、
每 sample 约 104–105 ms、两端各 24 对 null、6 组敏感性协议）。严格协议下任一条不过即
`strict_measurement_passed=False`，该轮只能作诊断：

- **null-after 的 CI 没有完全落在预设 ±0.2% 等价区间**；
- settling 敏感性比较（如 N200 vs N400）未通过等价门；
- **正式计时窗口内硬件波动超限：sclk 门限 2%、power 门限 5%**（实测踩过 sclk 2.81%、
  power 6.68%）。

判废后的纪律：**保留全部样本，不扣 null，不与旧的反向轮次拼平均**；也不能因为 CI 落在 1
的一侧就宣称严格证明了加速或退化。有一轮 B/A=1.000978（慢 0.098%，CI [1.000076, 1.001861]，
17/48 更快）就是这样被判为不可声称的——**CI 不跨 1 但门限没过，等于没测**。

## 7. 无人值守运行四规则（来自一次过夜空转 5h20m 事故）

1. **每个测量点单独汇报，不按批汇报**——系统性故障会让整批以同一原因全灭，按点汇报第一个点就停。
2. **后台任务必须显式注册等待**——`setsid nohup <driver> &` 写在前台调用里什么都没注册，driver
   脱离进程树但成败无人接收；另注册一个等待任务。
3. **单次终点等待抓不到挂死**——监督同时查三件事：完成标记、driver PID 是否还在、进度文件 mtime
   是否超阈值（用过 **12 分钟**）。
4. **零产出不是成功**——`passed: none` 必须显式记为失败并保持有任务挂起。

配套：run 飞行中不得修改被 pin 的文件（重新 pin 只能在点与点之间）；移植 harness 后 grep 一遍
`PREVIOUS`/`FROZEN`/硬编码目录引用（旧 pin 永远不可能通过且本目录查不出来）。

## 8. 测量事故档案（通用教训）

### 8.1 反向测时 bug：graph 重捕获覆盖

症状：harness 把初始 capture 顺序改成 B→A，但随后的"默认入口验证"又重建 candidate-only graph 并
`graph_bank['candidate']=default_graph`，正式计时用的不是最初反向那一对。目录名/自报字段写着
反向，实际计时图没反。

修复协议（已定型）：

- 两臂各 capture 一次，保存 `graphs[0]`/`graphs[1]`，之后**复用而非重捕获**；
- 计时前硬断言 `graph_bank['baseline'] is graphs[0]`、`graph_bank['candidate'] is graphs[1]`；
- 每 rank 记录实际 `timed_graph_capture_events`（顺序、arm、对象 id），进程结束后独立 verifier 核对；
- **平衡 AB/BA 是 replay 次序，不是 graph 创建次序**，二者分别记录。

### 8.2 跑错 harness

症状：run.sh 由字符串替换复制，`exp=$root/...` 没被替换命中，实际跑的是旧候选；旧 harness 自己的
gate 全过，不代表新候选跑过。

修复：harness 路径一律绝对路径；启动参数带 `--expected-harness` / `--expected-candidate-tag`
断言；结果同时核实际加载源码、trace 中的 kernel 名、ISA 哈希。**补丁脚本"替换成功"不等于
"跑的是新代码"，必须用运行时证据闭环。**

### 8.3 运维坑（通用结论）

| 坑 | 结论 |
|---|---|
| enroot env 白名单吞 override | 经环境变量传的 override 会被静默吞掉；改文件传递（override.json）+ kernel 名后缀门禁 |
| `srun` 吃 stdin | 会把驱动脚本 stdin 的剩余行读走，每批只跑第一行；一律 `< /dev/null` |
| `pkill -f <pattern>` 自杀 | 发起命令的 shell 命令行含同字符串会被一起杀；`[p]attern` 括号技巧也防不住。按 PID 杀，或 `[v]llm` 类首字符括号（命令行里不含完整 pattern 时才安全），杀完 `ps` 复核 |
| 控制机内存不足顺进程树杀 GPU 任务 | 长任务 `setsid nohup` 脱离进程树，进度写文件，回来读文件 |
| NFS 属性缓存 | 刚写完的 result 文件在 srun 退出瞬间 stat 不到 → 误判失败；先 `ls` 破缓存再 stat |
| 补丁锚点 | `str.replace` 必须配 `assert old in s`；且断言"唯一"不只"存在"（同名变量在多个 Config 里各一份，只替到一处） |
| 相对输出路径 | enroot 内相对路径写到短生命周期 cwd，容器退出即丢；输出路径一律绝对 |
| `server-ready` 文件 | 是遗留文件不是状态；就绪判据用 `/health` + 日志时间戳 |

## 9. 资源/ISA 先审

改动先看 ISA：实际分支、wait/barrier 顺序、VGPR/SGPR/spill/scratch/LDS。新增寄存器压力或 spill
已足以否定方案时先停，不为了走流程上 GPU。"资源改善 ≠ 速度改善"有实测反例（spill 2→0 仍慢 1.56%）。
rocprof trace 里的 VGPR/SGPR 字段不准，资源/驻留以 ISA 与 HIP code object 查询为准。
