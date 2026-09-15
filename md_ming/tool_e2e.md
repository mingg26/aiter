# serving e2e 测量规范（终稿）

> 范围：vLLM serving 层 decode/prefill 的端到端测量、profile 开窗、环境与权重 staging。
> 有效性基准：MiniMax-M3 AMD（MI350X/gfx950）工作线，2026-09-03 至 09-12；方法论长期有效。
> 来源：`tool_e2e.md`、`e2e.md`、`handoff_decode_msa_pr.md`、`report_claude.md`、`handoff_dep.md`。
> kernel 级测量见 `tool_measure.md`。

## 0. 一句话协议

**先判断 phase：decode 和 prefill 不能共用同一种重复方式。**

- Decode：同一个长 cohort 内先 warmup，再测固定 KV 区间，server 侧 barrier release→drain 计时。
- Prefill：一次 cohort 只有一次目标 prefill step，必须重复完整 cohort（每 arm ≥8 个过 gate 的 cohort）。

两种实验都必须 balanced paired A/B、同一 allocation/节点/GPU 集合；每个 arm 全新 server 进程。
**性能 E2E 不挂 profiler**；profiler 只用于定位，traced latency/TPS 不能当生产数字。

**统计与报告口径（与 `tool_measure.md` 同规格，serving 侧同样适用）**：每对先把 arm 内
formal samples 汇总，再算 candidate/base ratio；最终报告要给出**所有 pair、geometric-mean
ratio、正向 pair 数、pair effect 的 sample variance/标准差、paired bootstrap 95% CI**，
并同时保存 raw duration / mean / median / CV。**arm 内的重复共享同一次 server 初始化，
不是独立的版本复现——独立实验单位是相邻的 server-arm pair。** 另存 GPU 型号与
ROCm/镜像版本，**并在每个 arm 前后记录 clock、power、temperature**；新 harness 一律用
monotonic clock（源口径是"优先使用"，不是无条件）。**跨 rank 的 release skew 只能用同节点
共享的 monotonic clock 或 coordinator 观察到的 ack**；两者都没有时，不要相减时间戳，只校验
generation/step lockstep。

## 1. TPS 公式与计时窗口

```text
decode_duration = drain_time - release_time        # server 侧 barrier
decode_TPS      = B * (N - 1) / decode_duration    # 例:B×1999（输出 2000,release 前已各产 1 token）
cohort_TPOT     = decode_duration / B / (N - 1)
prefill_TPS     = exact_fresh_tokens / (drain_time - release_time)
```

**主分母是逐请求记录的实际剩余 token intervals，不是名义值**（投机 decode 一次接受多个 token，
允许预先规定的小幅 skew，但必须按 DP rank 分组记录每个请求的实际 token 数与 acceptance rate）。
模型加载、graph capture、真实 prefill、首 token 不在 decode TPS 内。

不用 client task 启动/HTTP 完成时间代替模型窗口；不跨 client/server/worker rank 相减未同步的
wall-clock timestamp。

## 2. admission barrier 与 SchedulerOutput 校验

- **`concurrency=B` + `max_num_batched_tokens=B*q` 只是容量上限，不保证 B 个请求进同一个
  scheduler step**。实测并发 4 的运行 TTFT 散布 509–539ms（client 发射只差 1.54ms），结果只能称
  natural staggered-admission，不能称单步 `4×4096/Q=16384`。
- 严格形状必须用 scheduler admission barrier：每 DP rank 本地挡住请求，跨 rank coordinator 等
  所有本地 barrier ready 后统一 release。
- **逐 rank 校验第一份 `SchedulerOutput`**：request count、逐请求 scheduled tokens、local total，
  再验 global total。例：strict 4×4096，TP-only 时单 rank `request_count=4, local_total=16384`；
  TP1×DP4（每 rank 一请求）时每 rank `request_count=1, local_total=4096`。**不能只验全局 total**，
  聚合 cache-miss 数不能证明这些 token 在同一个 model step。

## 3. warmup 污染

- benchmark 内部 warmup 会缓存被测后缀（实测把 hit rate 抬到 96%）：必须 **`num_warmups=0`**
  加单独的不计时同形状 premeasure pass。
- premeasure 同时解决 JIT 混入正式窗口的问题（premeasure 覆盖所有 compiled shape）。
- 投机 decode 的 A/B 两臂 draft/accepted token 与 acceptance rate 必须一致或差异已解释。
- prefill 先明确 workload 类型并照实标注：**cold**（0% prefix hit）、**cached incremental**
  （先缓存 prefix 再测固定 fresh tokens）、**natural client**（允许 tokenize/admission 错峰，
  不能声称固定 scheduler shape）。`uncached_prefill_token_throughput` 在 90% hit 下分子只算
  fresh token 但墙钟含 cached 路径——只能标 incremental/cached，永远不能标 cold/full。
- natural-client 结果只能报自然 serving 口径；strict-shape 结论必须 barrier + SchedulerOutput
  校验另跑。

## 3.5 在 E2E 里测单个 kernel / layer

| 问题 | 工具 | 结果含义 |
|---|---|---|
| 优化是否改善完整 serving | 无 profiler barrier E2E | 生产路径 latency/TPS |
| 目标 kernel 在真实 workload 用了多久 | GPU-only torch trace / 受控 kernel trace | 单 dispatch 或每 model-step device duration |
| kernel 内部在等什么 | 真实 shape 复现到 unit harness + PMC/ATT | 全 dispatch counter 与指令级 stall |

E2E trace 中同名 kernel 跨 layer/step/rank 出现：先按 model-step 边界切片，kernel 名不带 layer ID
时只能报 kernel-family 聚合（要测指定 layer 必须加 `record_function`/ROCTx annotation 或验证过的
固定 call position）。kernel/layer profile 与无 profiler E2E 分别报告，不能从前者推后者。

## 4. CPU / 资源模板

| 配置 | CPU 起点 | 说明 |
|---|---|---|
| kernel-only 单 GPU | 8 CPU | PMC/ATT wrapper 用 16 |
| TP4 / DEP4 serving | **32 CPU**（8/GPU） | 覆盖 4 worker + EngineCore + API server + tokenizer/client |
| DEP8 / EP8 serving | **64 CPU** | 96 只在监控到持续 host 饱和后才申请 |

A/B 两臂必须相同 CPU 数与 affinity；更多 CPU 不会缩短 device 时间。启动前 `scontrol show job`
核对 `NumCPUs`；资源变更视为新实验配置重新测量。

## 5. profiler 只用于定位

- profiler trace 文件数 = 实际启用 profiler 的 GPU worker rank 数 = `PP × TP × PCP × DP`
  （不是 DP×EP）。
- TP/EP collective 路径报 per-rank + mean + max，以 max 为 critical path；**rank 时间绝不能相加**；
  summed kernel duration 不是 wall-time 分解（stream 会重叠）。
- torch profiler 流程：本地 launcher 用 `M3_PROFILE=1`，建议 `M3_PROFILE_IGNORE_FRONTEND=true`、
  `M3_PROFILE_RECORD_SHAPES=false`；模型加载/capture/warmup/premeasure 全在 profiler 关闭时完成；
  barrier ready 后 `/start_profile`；**drain 后才 `stop_profile`**（满载下 Kineto save trace 会把
  server 打崩，exit 139，实测）；`record_shapes` 单独跑，不与 timing trace 混比。
- **profile 窗口必须稳态 + 按 step 归一**：ramp-up 开窗会把爬升采进去；固定墙钟窗口下更快的一边
  跑更多 step，"多 kernel"可能只是"多 step"。按 1-per-step 的 marker kernel 数 step。
- **rocprofv3 attach 三前提**：目标进程带 `ROCP_TOOL_ATTACH=1` 启动（否则无 attach 线程，报
  status 1）；`ptrace_scope` 允许同用户 attach；enroot 不开 PID namespace，容器内进程 host PID
  直接可见。形态：`rocprofv3 --kernel-trace --pid <pid> --attach-children --attach-duration-msec <N>`。
  外部 attach 不会让 server 崩（对比 torch profiler save）。
- torch profiler 在 ROCm 上会扰动执行，单步分解用 HIP-event `sitecustomize.py` 插桩替代。

## 6. 三大测量毒药（serving/benchmark 级）

1. **`--shared-prefix`**：所有请求指向同一批物理块，全 batch KV 常驻 LLC，作废一切访存相关测量；
   曾据此误判"不是 bandwidth-bound"，换 distinct blocks 后结论反转。
2. **零 KV cache**：`make_case` 默认清零 KV，所有 provider 平凡相等，精度回归测不出来；"bit-exact"
   结论在 `--nonzero-kv` 下变 2.55e-2。
3. **单形状调优**：split chooser 在 ns≤128 全对、≥192 系统性错，只测一个形状看不出来。

配套：synthetic `shared_blocks` 是严格双峰分布，不接近真实 indexer；真实结论要报区间不报单点。

## 7. 权重 staging 与环境

- **权重放节点本地盘**：共享 NFS 冷启动 ~44 分钟 vs 本地盘 ~5 分钟（fastsafetensors ~170MB/s）。
  staging 做字节数校验（stat 跟随 symlink，`stat -Lc`），断点续传按字节完整性跳过分片。
- **`--load-format fastsafetensors` ≠ lazy mmap**：后者是普通 safetensors loader 的 mmap 策略，
  fastsafetensors 走自己的路径；两者行为不能互相推断。
- bench 投机 decode 必须开 **`M3_SYNTHETIC_SPEC=1`**：假 KV + 真实拒绝采样 ⇒ acceptance=1.00，
  TPOT 退化成无投机的 ~39.8ms；synthetic 模式固定接受率（如 [0.7,0.5,0.4]）。
- **shuffle KV 三角（新机器第一次起服务必撞，按这个顺序解）**：`ROCM_AITER_FA` +
  `--block-size 128` 要求 **shuffle 开**（否则 backend 只报 `[16, 32]` 两种 block size）；
  而 shuffle 开又要求 native **`zero_kv_blocks`** op，**该 op 只存在于 fullbuild 镜像里**。
  解法：分支自带的 `csrc/rocm/kv_cache_zero.cu` 有 `VLLM_KV_ZERO_STANDALONE` 宏，可以单独
  编成一个 `.so`，再照 overlay 的做法给 `utils.py` 补一个 env 加载入口（分支里原本只有
  `raise`）。**三个条件是连环的，缺一环就退回 block-size 16/32**——recipe 是按 bs128 定的，那种情况下
  测出来的数不能与 bs128 的结论并列（后半句是推论，源文只给了三个条件本身）。
- **`M3_SKIP_MIXED_SPEC_WARMUP=1` 必须开且要显式传进容器**：enroot 只转发白名单变量；混合投机/
  非投机 warmup 批会撞上面那个 shuffle KV 断言（合成批的 q 长不均匀）。真实流量均匀 q 长，
  碰不到该断言。
- PYTHONPATH 指向纯源码树时 `import vllm._C` 静默失败（`get_cuda_view_from_cpu_tensor` 缺失）：
  镜像编译产物（`.so`）要抽进源码树，不是放弃 PYTHONPATH。
- `vllm bench serve` 的 `--random-range-ratio` 这版语义是"越小范围越宽"，1.0 直接报错。
- stock bench random 数据集 detok→重 tokenize 拖垮 TTFT/output-tok/s；**SLA 对比用 TPOT，TPOT
  不受此工件影响**。
- 无投机时该栈 decode ≈ 39.8ms TPOT——投机的收益全靠接受率。

## 8. 进程管理坑（通用）

- `pkill -f "vllm serve"` 会杀掉自己（发起 shell 命令行含同字符串）：用 `pkill -f "[v]llm serve"`。
- 后台 serve 的 srun 进程被中断 = slurm 取消整个 step，server 跟着死；被打断过的 serve 任务先
  `pgrep -f "[v]llm serve"` 验尸。
- 杀 bench 负载要杀进程树：`bash → enroot → vllm bench` 多层，`kill $!` 只杀最外层。
- bench 客户端生成数据集要几十秒，期间 server 空闲；"等固定 N 秒再 profile"会空采——轮询
  `vllm:num_requests_running` 到目标再开窗。
- 多轮 bench 共用结果目录会互相污染；profiler 供载的垃圾负载写到独立目录。

## 9. decode/prefill gate 清单（任一失败整 arm 作废）

Decode：active/ready/completed 数=目标 batch、failed=0；DEP 下逐 DP rank 数与预定 assignment 一致；
同一 global barrier generation、release skew 在门限内；release 时无 in-flight token；分母可逐请求
复算；formal window 无 JIT。

Prefill：每 DP rank 首个 SchedulerOutput 精确匹配（request count / per-request / local total）；
prefix hit counter 与 cold/cached 定义一致（cold 硬校验 0% hit；ready probe 和 warmup 不得命中
formal suffix）；无 JIT；source hash 与 server command 与声明一致。
