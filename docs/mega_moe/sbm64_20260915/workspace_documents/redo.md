## 2026-09-15 最新恢复入口：SBM64 13点保留，b72不采用

先读 [report_sbm128.md 最新 SBM64 小节](report_sbm128.md)，以及 [fork 内完整报告](aiter-ming-amd-m3-megamoe/docs/mega_moe/sbm64_20260915/README.md)。本轮用户授权实际代码提交、文档/台账更新、K3最终审核。

- 默认接入精确 batch：**64,104,112,120,128,136,144,152,160,168,176,184,192**；S1物理SBM64/N256/K256，`sbm64_path=m32_m48_m64`，保留M32/M48/M64、current-A前读、K128 B退休、late A-scale与last16/middle16 epilogue producer+consumer guards；ticket修复保持。
- scope：EP8/H6144/I3072/E128/top4、shared L13+L2/XCD、tokens==mtpr、无local-reduce或显式Stage1 env覆盖。b64 S2 BM64/N128 NT ON；其余BM32/N256 NT OFF；S2整空skip除64/152外ON。b72继续原SBM32/N512，候选与scale6裁剪均不合入。已有SBM128不变。
- 性能来源 `.scratch/sbm64_retained_sweep_20260914_v1`：26轮、312/312对快，耗时降低约2.10%–4.97%，quick screening口径；b136用整轮优化前最快M64/N512+ticket，不用last16增量对照。middle16自身小收益仍未解决。
- 用户最后要求不再重跑所有 batch GPU，只需确保 ISA 一致。13点×8rank 共104份原生 S1 ISA已全部匹配；补充原生GPU仅完成 b64/104/112/120（均16探针、graph512及全8rank S1/S2/quant ISA通过）。其余9点未重跑原生GPU，沿用先前两方向实测证据，加本轮ISA身份和选择器验证；不得声称13点全部完成了本轮GPU。b128启动在worker/kernel前因TCPStore端口53128占用失败，按用户要求不重试。
- 原生导入104份CPU ISA一致、5371选择器比较和9项CPU回归通过。GPU接入验证在 `.scratch/sbm64_final_rollout_20260915_v1/native_v2`，已完成四点真实默认16探针/graph512/terminalheads及全8rank S1/S2/quant ISA；最终状态见fork报告目录的 `native_gpu_validation.json` 和 `final_audit.json`。
- fork：`aiter-ming-amd-m3-megamoe:ming-amd-m3-megamoe` → `ming-fork=mingg26/aiter`；同步生产树 `megamoe-s1-1k-prod/aiter:s1-unified-production`。最新提交身份见 `final_audit.json`；不要使用下方历史HEAD作为当前入口。配置台账随此轮更新，历史normal_timing字段不代表新默认。
- K3为实际CLI模型 `inferact-kimi-k3`，续用session `session_ebbbe6de-6172-4d40-b346-b3d01ac8313b`；设计/实现/最后审核日志在本轮review目录。旧“config N256会改变host队列”的猜测已根据源码撤回：tile_n在此scope只传给S1 launch；仍做原生GPU接入核验。
- 原始失败记录保留：b160旧AB首跑Mori bind errno98无计时；b72首次校验遗漏`.kd`后缀无计时；本轮native首次把容器bind路径误当宿主路径，kernel前中止，新native_v2核对容器路径+源码hash。均不影响保留性能样本。
- 本轮更新文档已获授权。旧实验冻结的文档/HEAD pins不改写，复跑须新目录/新pins；不得绕过门禁。job1913/node03 allocation保留，预计2026-09-15 19:06 UTC结束。

## 2026-09-14 最新重启入口：SBM32 已提交、补测结果与反向测时 bug

**先读 [report_sbm128.md 的 sbm32 小节](report_sbm128.md#sbm32)。** 它覆盖下方旧记录的 generic S1/小 batch 状态；下面 b200/SBM128 复现记录仍是历史，不要把其旧 HEAD/命令当作当前启动条件。

- 生产目录：`/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter`，分支 `s1-unified-production`，HEAD **`e84ac1b388b709126d40b0e2f79207dc4269ca97`**。ticket+late-A-scale=`40c3eeedc`，B retirement=`5571deb9a`，b96 先读 A=`e84ac1b38`，都已 commit；最新五点测量未改代码。当前生产工作区干净。
- 当前 **SBM32 只有完整 M32 K-loop，没有保留 M16/M32 双路径或 M16 跳算**。b72/80/88：late A-scale+B retirement；b96 再开启 arbd；b64 实际为 SBM64/M64，仅有 late A-scale（ticket 两臂共有）。物理 SBM、逻辑 M 行路径与 S2 BM 的区别见 report。
- job1886/node03，五点每点12对快速完整 forward 配对：b64 **慢1.555%（0/12）**；b72/80/88 分别快1.368%/1.517%/1.431%（均12/12）；b96 相对5571快1.294%（12/12）。正确性/实际默认/源码/8 rank ISA/独立重算通过，未做严格稳定性资格判断。
- **b64 退化尚未回退。** 后续如处理它，以 `.scratch/b64_96_s1_current_vs_best_20260914_v1/b64/baseline` 为当前较快参照（b137+ticket）；不要拿较慢的 clean e84 默认替代最快基线。其 spill2→0、scratch12→0 并未带来速度改善。其它四点从当前保留 e84 出发；更小 batch 尚未做本轮补测。
- 所有GPU作业已结束。job1883在脚本执行/采样前反复重排，换直接 srun 的1886后五点完成；无样本混用。详情见五点实验 `review/scheduler_reassignment.json`。
- 本次用户授权更新 `report_sbm128.md` 与 `redo.md`；未更新 `handoff_megamoe.md` 或配置台账。它们关于 generic 小 batch 的旧 HEAD/状态可能过时，以本节与 report 新节为准。

### 必须记住的反向测时 bug：反转了初始 capture，却又单独重建 candidate graph

旧 b96 `measure_reverse/bench.py` 的初始编译和 capture 顺序确实改成 **B→A**，但之后的“真实默认入口 graph 验证”又新建了一个 **candidate-only `default_graph`**，并执行 `graph_bank['candidate']=default_graph`。正式计时最终使用的是 **原 baseline graph + 更晚重捕获的 candidate graph**，不是最初 B→A 的那一对。因此 `initial_compile_and_graph_capture_order=['candidate','baseline']` 或目录名 `measure_reverse` 本身都不能证明最终计时图顺序反转。

**受影响的旧反向记录**（均在 `.scratch/` 下）：

1. `b96_s1_aread_before_dma_20260914_v1/measure_reverse/gpu_v1/b96/result.json`；
2. `b96_s1_m16_singleloop_20260914_v1/measure_reverse/gpu_v1/b96/result.json`；
3. `b96_s1_m16m32_gpu_20260914_v1/measure_reverse/gpu_v1/b96/result.json`（另有下节“跑错 harness”错误，不能用于245候选）；
4. `b96_s1_tailwait16_20260914_v1/measure_reverse/gpu_v1/b96/result.json`。

对于仅受 capture 问题影响的记录，保留原始同进程配对数据及已经核对的源码/ISA身份；**撤回“最终计时 graph 已反向构造”“两种构造顺序都确认收益”等更强结论**，不将这个发现直接当成某一时间差的因果解释。第3项同时跑错候选，候选归因仍然作废。该排查针对列出的 b96 harness，不能没有检查就宣布全部历史 SBM128 反向记录也有同一错误。

完整范围与原记录路径：`.scratch/b96_s1_tailwait16_reverse_strict_20260914_v1/review/historical_reverse_capture_scope_correction.json`。旧原始文件没有重写。

**已经实现的修复：**

- 两臂都使用实际默认 selector；在需要的顺序各 capture 一次，保存 `graphs[0]`/`graphs[1]`。
- 默认入口验证直接复用 `default_graph=graphs[1]`，不得再只重捕获 candidate 并替换正式计时图。
- 计时前硬断言 `graph_bank['baseline'] is graphs[0]`、`graph_bank['candidate'] is graphs[1]`。
- 每 rank 记录实际 `timed_graph_capture_events`（顺序、arm、对象id）、`timed_graph_capture_order`、`timed_graphs_reuse_initial_graphs=True`；进程结束后独立 verifier 核对这些字段、真实kernel名、每rank源码/ISA哈希与资源。仅核 manifest 中自报顺序不够，harness 也要审。
- 两臂继续共用同 operator、同输入、同量化输出/scale指针、同输出buffer；平衡 AB/BA 是 replay 次序，**不是 graph 创建次序**，二者分别记录。

已修复的模板：`.scratch/b96_s1_tailwait16_reverse_strict_20260914_v1/measure_reverse/bench.py`；最新快速版：`.scratch/b64_96_s1_current_vs_best_20260914_v1/b96/measure/bench.py`（其它四点同修复）。最新job1886是一次真正 B→A 的快测，不能因此把旧记录追认成有效反向实验。

### 修复后的尾等待严格反向：结果不支持保留 wait16

`.scratch/b96_s1_tailwait16_reverse_strict_20260914_v1`，job1881，A=当前保留 arbd1/tail10，B=arbd1/tail16。源码、8 rank ISA资源、正确性、真实默认、最终计时 graph 反向身份全部通过。

严格设置48对、每sample约104–105ms、两端各24对null、6组敏感性协议。原始诊断 B/A=**1.000978**（慢0.098%，95% CI [1.000076,1.001861]，17/48更快），但 **`strict_measurement_passed=False`**：

- null-after CI 没有完全落在预设 ±0.2% 等价区间；
- 正 settling 的 N200/N400 敏感性比较未通过等价门；
- 正式计时窗口硬件波动超限：sclk最大约2.81%、power约6.68%，门限为2%/5%。

因此这轮只作诊断，不能宣称稳定加速，也不能因CI在1上方就宣称严格证明了稳定退化。保留全部样本；不扣null、不与旧反向轮次拼平均。最终 **不保留 tail16，生产仍为 e84/tail10**，没有为该候选commit。详情在该实验 `review/decision.json`、`review/hardware_failure_audit.json`。

### 另一个独立错误：run.sh 仍指向旧实验，实际跑的不是声称的候选

此前新脚本由字符串替换复制，但 shell 中 `exp=$root/...` 没被“绝对路径替换”命中，仍执行旧 wait-once 的 harness。旧 harness 的自己的 gate 通过，不代表新候选跑过。

作废候选归因的路径：`.scratch/b96_s1_dma_after_ni0_20260914_v1/measure/gpu_v1`，以及 `.scratch/b96_s1_m16m32_gpu_20260914_v1/measure/gpu_v1`、`measure_reverse/gpu_v1`。实际 candidate 是 **`_lm16sl1_aw1` / 184 VGPR**，不是 `_adni01` 或245 VGPR双路径。原“adni0慢0.30%”“245正向快0.24%/反向慢0.08%”不能再作为那些候选的证据。

修复后每次用绝对 harness 路径，启动参数带 `--expected-harness` 与 `--expected-candidate-tag`，结果同时检查实际加载源码、trace 中 S1 名、八rank资源和GPU ISA相对离线参考的字节哈希。补跑的正确证据：

- `b96_s1_m16m32_gpu_20260914_v2/measure/gpu_v1/b96/result.json`：245 VGPR、SGPR spill7、scratch0，慢2.023%，0/12，未保留；这是用户明确允许超预算上GPU的特例，不能推广该豁免。
- `b96_s1_dma_after_ni0_gpu_20260914_v2/measure/gpu_v1/b96/result.json`：正确 `_adni01`/184 VGPR，相对5571快约1.483%。与arbd的直接比较在 `b96_s1_dma_order_head_to_head_20260914_v1/measure/gpu_v1/b96/result.json`，B/A=1.000807、CI跨1，没有额外收益，保留arbd。

原始错误范围清单：`.scratch/b96_s1_harness_correction_20260914_v1/invalidated_results.json`。里面 `unaffected_checked` 仅表示当时未受“跑错harness”影响，**不表示已排除后来发现的 graph 反向问题**。

### 重启后怎样继续

1. 先核 HEAD=e84、工作区状态，再读 report 的 `sbm32` 节及五点 `review/summary.json`；不要重复把未保留的 M16/尾等待当成生产默认。
2. 用户已恢复允许快速筛查；默认可12对，不自动强制每轮48对/完整敏感性。ISA/GPU正确性、源码身份、实际计时graph身份和全rank锁步仍必须过；小收益/反转再加测。寄存器/spill/scratch超预算立即停等用户；决定保留后直接commit实际代码，不追加提交前检查或确认。
3. 本轮用户已说“先这样”，没有待执行的新优化。packed A-scale 仅讨论、未动手；K128半段B释放方向也没有实施。下一步由用户决定，b64退化是已知未处理项。
4. **更新文档后，旧实验冻结的 protected-document/source hashes 已过时。** 从最快源复制到新目录、重新冻结新pins再运行；不得在旧目录改manifest以“修复”门禁或覆盖旧输出。参考template可以复用代码，不可直接照抄旧run.sh假装原冻结环境未变。
5. 主助手K3 session仍为 `session_054a5e40-816d-4745-b49b-072ee8da21af`；CLI `/home/ming/.kimi-code/bin/kimi --model inferact-kimi-k3 --session <session> --prompt <内容>`，续会话cwd为 `/mnt/shared/homes/ming/msa/.scratch/b200_sbm128_m16_epitail_20260914_v1/candidate_200`。用户独立讨论窗口是tmux `k3-discuss-0914`，与主助手的K3会话无关，不要混用。

## 当前工作模式：主助手与 K3 协作（2026-09-14）

用户所说的“老办法”默认指以下流程；若用户本轮明确调整顺序，以本轮指令为准。

1. **先定基线，再讨论。** 主助手先读最新报告、handoff、redo 与目标 batch 的实测记录，只从该条件下目前已测最快的源码/配置出发。主助手与 K3 先讨论一个具体改点、预期收益及寄存器/同步风险；已讨论清楚的方案不重复讨论，每轮优先只改一个变量。
2. **主助手实现并先看 ISA。** 主助手负责修改代码、编译、对比实际分支、指令顺序、wait/barrier、VGPR/SGPR、spill、scratch 和 LDS。若新增寄存器压力或溢出已足以否定方案，先停止该候选，不为走流程继续跑 GPU；不要把“压寄存器”变成增加 spill。具体是否值得实测遵从用户当轮要求。
3. **ISA 可接受后，GPU 与 K3 审核并行。** 主助手负责 GPU 实验和监督；K3 独立审核源码 diff、ISA、正确性/同步风险、适用范围与测试方法。无需等 K3 审完才启动 GPU，但 GPU 实验之间串行，避免相互干扰。K3 默认只读，不与主助手同时修改同一份代码。
4. **发现问题就停本轮、修好重来。** K3 提出问题后，主助手结合源码/ISA核实；若问题成立并使当前实验无效，停止对应实验的明确进程，不取消整个 allocation 或无关任务。修复后使用新的实验目录/输出与 source pins，重新检查 ISA、审核并跑 GPU；运行中不修改已冻结的源码、harness 或受保护文档。
5. **结果由证据决定。** 遵守 `tool_measure.md`，先过正确性、graph/epoch、全 rank dispatch、source/ISA 与硬件稳定性门禁，再看同进程 balanced paired A/B。快速试验只作筛选，不冒充正式收益；不跨进程比较绝对均值、不删除不利样本。小收益、轮次反转或同 kernel 也有差异时，先排查测量偏差，必要时补构造顺序反向对照。
6. **主助手汇总，K3 复查结论。** 两者复核是否保留、默认是否真的命中选中方案、是否影响其它 batch。无收益或退化的候选不混入保留版；不确定就明确说明。继续下一个方向前，简短交代本轮结果和下一改点，不盲目换 M 分支。
7. **每次保留就直接 commit。** 用户已授权每次决定保留后直接提交实际生产代码，不只提交实验快照，不追加提交前检查或再次确认。候选选择阶段的 ISA/GPU 验证照常进行；若寄存器、spill、scratch 等超过当前基线预算，立即停止并等待用户。报告、handoff、默认台账等结果文档仍按用户单独授权更新；保留旧实验原始记录和不确定性，不改写冻结 checkpoint。

K3 指实际 Kimi CLI 的 `inferact-kimi-k3` 模型，不是把另一个普通子助手命名为 K3。当前可续用的讨论 session 为 `session_054a5e40-816d-4745-b49b-072ee8da21af`；主助手传给 K3 的材料应包含目标、源码路径、diff/ISA、基线与结果路径，并明确本轮是方案讨论还是只读审核。

## 2026-09-14 最新：b200–256 默认合入（覆盖下方历史状态）

用户本轮明确授权 GPU 验证、commit 和文档更新。实际生产 `aiter` 已提交 **`b13717ca0a0dbe0cf53df353467b8f816aa7c762`**（不是仅提交外层实验快照）。默认按 batch 选择：**b200/208/216 使用 SBM128 逐 16 行八路径；b224/232/240/248/256 使用修复 ticket 的 64/96/128 三档**。这里“八路径/三档”是 kernel 内逻辑计算行数，物理 SBM 均为 128。

范围仅 EP8、128 总专家（每 rank 16）、H6144/I3072/top4、完整 fused shared L13+L2 且 XCD 调度、`tokens == max_tok_per_rank`，精确覆盖 200/208/216/224/232/240/248/256；无 local-reduce、无显式 Stage1 环境调参。其他 batch、形状和通用默认保持不变；这不是对区间内任意整数 batch 或全部路由分布的最优性声明。

代码入口是 `mega_moe_config.py:SBM128_PATHS` 与 `MegaMoEM3` 的受限选择器。实测 S1 源码在私有 `sbm128_m16/`、`sbm128_tiered/`；不要把通用 `gemm1.py` 当成本轮默认实现。S1 均为 N256/K256，b200–248 的 dispatch CU48/B-NT2，b256 为 CU32/B-NT0；S2 BM32、b200–248 N256、b256 N128，整块跳空 ON、内部 M16 OFF。

证据入口：[megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json](megamoe-s1-1k-prod/bench/checkpoints/sbm128_b200_256_rollout_20260914/summary.json)，[配置台账](megamoe-s1-1k-prod/bench/dep8_best_configs.json)。旧 `normal_timing` 等字段保留历史含义，不能当成本轮默认的新测量。

八档原生默认 GPU 验证及 ISA 检查已通过，4736 个选择器场景、768 个范围外 S2 场景和 5 个 CPU regression tests 通过。b224 两轮方向反转，保守保留三档；b256 检出构造顺序相关时间差，已补真实两方案反向构造，详见 report 最新节与 summary 中 position_diagnostic，不能照搬原单向 1.12% 胜负。八路径为 VGPR253/Vspill0/scratch0；三档已有 VGPR256/Vspill7/scratch32 B。M80 被拒绝的改动未保留。

后续从目标 batch 的当前选中私有实现开始。旧冻结 harness 仍 pin 合入前 HEAD/文档，需新目录与新 pins 才能复跑；不可跳过门禁。下方均为历史，尤其“仍未合入”“逐16行仍输”不再描述当前 b200–216 状态。

# b200 SBM128 `scaletail64` 复现记录

本文记录 2026-09-14 在 `node04` 上复测 b200 最后一个 16 行跳空候选的过程。测量方法遵循 `tool_measure.md`。

## 1. 复测对象

这里的 b200 配置始终是：

- EP8，每 rank 200 tokens，TopK=4，seed=42；
- hidden=6144，intermediate=3072；
- Stage1 `sort_block_m=128`、`tile_n=256`、`tile_k=256`；
- Stage2 BM32/N256，整块跳空开启，内部 M16 跳空关闭；
- 测量范围为 TopK、完整 routed MoE、TP1 shared MLP 和最终求和，不含 router GEMM 与 residual。

`M48/M64/M80` 等名称只是同一个 SBM128 kernel 内按 `valid_rows` 选择的逻辑计算路径，不代表把物理 `sort_block_m` 改成了 48、64 或 80。

同进程比较的两个 arm：

- A：`best_b200_sbm128_tiered_short_dma`，即当前最快的 SBM128/N256，内部 64/128 两档计算和短 A DMA，并含 launch-ticket 消费完成 barrier；
- B：`b200_sbm128_m16_scaletail64`，物理布局仍为 SBM128/N256，内部使用 M16/32/48/64/80/96/112/128 八条固定路径。

冻结后的复测目录：

```text
/mnt/shared/homes/ming/msa/.scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1
```

其中 `baseline_200/` 和 `candidate_200/` 是两个 arm 的源码，`bench.py` 是 EP8 paired A/B harness，`run.sh` 是执行入口，`source_manifest.json` 保存 source pin。

复测目录中的两个 arm 与原实验逐文件一致；baseline 也与 ticket-fence 最快候选一致：

```bash
cd /mnt/shared/homes/ming/msa

diff -qr \
  .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/baseline_200 \
  .scratch/b200_sbm128_m16_scaletail64_20260914_v1/baseline_200

diff -qr \
  .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/candidate_200 \
  .scratch/b200_sbm128_m16_scaletail64_20260914_v1/candidate_200

diff -qr \
  .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/baseline_200 \
  .scratch/b200_sbm128_entry_ticket_fence_20260914_v1/candidate_200
```

三条命令都应无输出。

## 2. 原始复测命令

原始复测使用 `node04` 上的 Slurm job `1841`：

```bash
cd /mnt/shared/homes/ming/msa
export M3_JOB=1841

bash .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/run.sh \
  b200_scaletail64_repro 200 s1_only
```

原始输出已经存在于：

```text
.scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/b200_scaletail64_repro/
```

`run.sh` 会拒绝覆盖已有输出。再次复测必须使用新的输出名和当前有效的 `node04` allocation：

```bash
cd /mnt/shared/homes/ming/msa
export M3_JOB=<active-node04-job-id>

squeue -j "$M3_JOB" -h -o '%i %N %T'

bash .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/run.sh \
  b200_scaletail64_redo_v2 200 s1_only
```

`squeue` 必须显示该 job 位于 `node04` 且仍在运行。实验需要单节点 8 GPU、64 CPU。不要删除或复用已有证据目录。

## 3. 脚本实际执行的内容

`run.sh` 首先要求生产树：

```text
/mnt/shared/homes/ming/msa/megamoe-s1-1k-prod/aiter
HEAD = 05bbc69252444ece0bff54858eae4be2864000a7
git status --porcelain 为空
```

随后脚本：

1. 新建输出目录，写入 `override.json`：`{"_kernels":"s1_only"}`；
2. 通过 `srun --jobid="$M3_JOB" --overlap -N1 -n1 -c 64` 进入 allocation；
3. 运行 GPU idle gate，要求显存空闲且 GPU busy 不超过 15%；
4. 使用固定 enroot 配置和 1800 秒总超时；
5. 设置 `FLYDSL_RUNTIME_ENABLE_CACHE=0`，重新编译并保存 8 rank IR/ISA；
6. 使用 `torchrun --nproc_per_node=8` 启动完整 EP8 进程组；
7. 在 benchmark 内校验 `source_manifest.json` 中的 158 个 source pin；
8. 在同一进程、同一输入和同一逻辑 workspace 中加载 A/B 两个 arm。

普通 redo 不要运行 `make_manifest.py`。它只用于最初冻结 harness；重新运行会按当前内容生成新 pin，失去验证原证据的意义。

## 4. 正确性与结构门禁

正式计时前必须全部通过：

- 生产树 commit/clean gate；
- frozen harness 和 kernel source hash gate；
- EP8 全 rank lockstep；
- eager 和 captured graph 数值正确性；
- 23 个 graph probe，包括路由变化、全本地、全远端及 16/32/48/64/80/96/112/128 各阈值两侧；
- graph 连续 replay 深度 512；
- 两个 arm 的实际 dispatch 列表，各 5 个 kernel；
- source、kernel 名和资源信息门禁；
- `soft_gate_failures` 必须为空。

原始复测的实际路由分布：

```text
M48: 55 个本地专家
M64: 70 个本地专家
M80:  3 个本地专家
```

本次正式输入不会动态进入 M16/M32/M96/M112/M128 路径。

## 5. Warmup 和 paired timing

本轮使用：

```text
graph gate replays: 512
稳定 warmup:        48 秒
paired samples:     每个协议 12 对
主协议:             k32_n400_c200
```

`k32_n400_c200` 的含义：

- `k32`：每个 measured chunk 前 32 次不计时 settling replay；不是 GEMM K；
- `n400`：每个 sample 聚合 400 次 measured replay；
- `c200`：分为两个不超过 200 replay 的 chunk；
- A/B 使用相同 replay 和 chunk 划分；
- 每个 rank 用本地 HIP event 计时，以 8 rank 最大时间作为 `critical_us`；
- 主结论使用 12 个相邻 A/B pair 的 log-ratio geometric mean 和 paired bootstrap 95% CI。

同时执行六种预先固定的敏感性协议：

```text
k0_n400_c200
k16_n400_c200
k32_n400_c200
k16_n200_c200
k32_n200_c200
k32_n400_c100
```

六种协议方向一致；主协议为 `k32_n400_c200`。

## 6. 原记录与复测结果

| run | baseline A | candidate B | B/A | 95% CI | B 更快 pair |
|---|---:|---:|---:|---:|---:|
| 原记录 | 325.211086 us | 329.857594 us | +1.428717% | +[1.291250%, 1.568889%] | 0/12 |
| node04 复测 | 325.327977 us | 329.347569 us | +1.235534% | +[1.102717%, 1.354023%] | 0/12 |

跨 run 诊断漂移：

```text
baseline:  +0.035943%
candidate: -0.154620%
相对效应: -0.193183 percentage point
```

绝对时间有轻微漂移，但两轮都稳定复现候选慢约 1.2%–1.4%，CI 不跨 0，主协议均为 0/12 更快。

结果文件：

```text
原记录：.scratch/b200_sbm128_m16_scaletail64_20260914_v1/b200_vs_best/result.json
复测：  .scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/b200_scaletail64_repro/result.json
```

## 7. redo 后快速验收

假设新输出名为 `b200_scaletail64_redo_v2`：

```bash
cd /mnt/shared/homes/ming/msa
redo_result=.scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/b200_scaletail64_redo_v2/result.json
redo_log=.scratch/b200_sbm128_m16_scaletail64_repro_20260914_v1/b200_scaletail64_redo_v2/run.log

jq '{
  passed,
  soft_gate_failures,
  warmup_seconds,
  graph_gate_replays,
  paired_samples,
  A_us: .summary.stats.A.mean_us,
  B_us: .summary.stats.B.mean_us,
  candidate_over_base: .summary.candidate_over_base,
  ci95: .summary.ci95,
  faster_pairs: .summary.faster_pairs,
  spread_pass: .summary.spread_pass
}' "$redo_result"

rg -n 'INITIAL_MATH_PASS|GRAPH_PROBE_PASS|DISPATCH_ARM_PASS|ROUTE_HISTOGRAM|S1_RESOURCES|WARMUP_STABILITY|RESULT' "$redo_log"
```

有效结果至少满足：

```text
passed = true
soft_gate_failures = []
graph_gate_replays = 512
paired_samples = 12
spread_pass = true
```

复现判断以 paired ratio、95% CI 和方向一致性为主，不要求绝对微秒值逐位相同。source pin、正确性、dispatch、graph 或 spread gate 任一失败，整轮时间作废。

## 8. 本轮未做的事情

- 未修改或合入生产 kernel；
- 生产树仍是 clean commit `05bbc69252444ece0bff54858eae4be2864000a7`；
- 未用 profiler 时间替代无 profiler paired timing；
- 未覆盖旧实验目录。
