# MegaMoE 重启与复现（2026-09-15）

先读本文 → [当前batch配置](summary_megamoe.md#5-当前默认配置2026-09-15-收尾) → [最新S2结果](report_s2.md)。目录实际叫 **md_ming**。根目录旧 redo/handoff/report 不再作为当前入口。

## 1. 当前状态

| 仓库（相对工作区） | 分支 | 当前 commit |
|---|---|---|
| `aiter-ming-amd-m3-megamoe` | `ming-amd-m3-megamoe` | `3c19734b9` |
| `megamoe-s1-1k-prod/aiter` | `s1-unified-production` | `e11429895` |

两边实现一致。上表是代码基线；文档与主分支的远端归档位置为 `mingg26/aiter` 的 `ming-amd-m3-megamoe`，生产分支保持本地镜像。新增默认：b120–184步8用已测更快的S2 M64/N256/W8/256CTA；b192本来已启用。b112继续BM32 constant N/K；其M16、512CTA、epilogue M16实验均不采用。b72当前S1是 **SBM32/N256/M16-M32**。

“默认最快”的约定：完整scope内，默认直接选已比较方案中保留的最佳配置，无需手工override。37点配置已检查；未逐点新测所有fallback/大档，计时也有波动，不作全局最优或每次最低值保证。本次接线通过88份ISA身份与历史GPU结果验证，没有新跑GPU。

## 2. 先核对默认（CPU，不占GPU）

```bash
cd /mnt/shared/homes/ming/msa
python3 md_ming/probes/audit_megamoe_defaults.py --repo aiter-ming-amd-m3-megamoe
python3 md_ming/probes/audit_megamoe_defaults.py --repo megamoe-s1-1k-prod/aiter
python3 aiter-ming-amd-m3-megamoe/op_tests/test_mega_moe_m3_config.py
python3 megamoe-s1-1k-prod/aiter/op_tests/test_mega_moe_m3_config.py
```

预期：两次37点PASS、两次12 tests OK。审计执行真实selector及scope表达式，逐字段对照 [固定账本](probes/megamoe_defaults.json)。限定 EP8/E128/H6144/I3072/top4、TP1 shared L13+L2、XCD、满batch，无local-reduce。脚本遇非空算法环境覆盖会拒绝；清除它列出的变量后再跑，不要注入TILE_N来“帮助”默认。

默认变更证据在两个仓库的 `docs/mega_moe/s2_defaults_20260915/`；CPU编译复核工具在 `.scratch/s2_defaults_finalize_20260915_v1/review/`。那里 `compile_all.py` 可复核已有88份ISA；如需从零编译，使用新的输出目录及原compile-venv，最多4个CPU编译并行。无需为文档重启重编译。

## 3. 复现最新 b112 epilogue 诊断

保留旧实验不可改。新默认修改了原harness锁定的外部源码，**不能直接重跑旧run.sh，也不能跳过source-pin门禁**。准备脚本只允许审计过的前后源码哈希组合，复制原实验到新目录，更新路径/来源元数据和明确的差异清单，重新锁定；任何额外漂移都会停止。原测量两臂、输入、图和数值/ISA门禁保留。

需要本工作区现有的`.scratch`依赖、overlay、容器镜像及一份有效的8卡gfx950 Slurm allocation；新机器不能只拷md执行。下面`M3_JOB`填当前allocation，不继承过期job号。输出目录每次新建，一次只跑一个实验/形状。

```bash
cd /mnt/shared/homes/ming/msa
replay_dir="$PWD/.scratch/redo_b112_epi16_run1"
python3 md_ming/probes/prepare_megamoe_replay.py \
  --experiment .scratch/b112_s2_epi16_20260915_v3 \
  --out "$replay_dir"
export M3_JOB=你的有效Slurm编号
export S1_OUT="$replay_dir/measure/gpu_run1"
bash "$replay_dir/measure/run.sh"
python3 "$replay_dir/review/verify_result.py" "$S1_OUT/b112/result.json"
```

该实验是S2局部诊断：GEMM M16两臂都开，仅epilogue阈值0/16不同。预期方向：全部工作基本持平（历史77.320→77.289µs），exact16局部约省0.68%。判定看配对CI、null、34组probe、graph512、8rank ISA/资源；**`strict_measurement_passed=false`，不得报告为严格稳定收益**。准备脚本已做CPU预检，重建的入口尚未新跑GPU。

## 4. 复现本次补入默认的完整 forward 数据

同一准备流程支持120–184步8。例：b128（胜幅仅0.235%，证据最弱）：

```bash
cd /mnt/shared/homes/ming/msa
replay_dir="$PWD/.scratch/redo_b128_m64_run1"
python3 md_ming/probes/prepare_megamoe_replay.py \
  --experiment .scratch/b128_s2_m64_backward_20260915_v1 \
  --out "$replay_dir"
export S1_OUT="$replay_dir/measure/gpu_run1"
bash "$replay_dir/measure/run.sh"
python3 "$replay_dir/review/verify_result.py" "$S1_OUT/b128/result.json"
```

仍需§3的有效`M3_JOB`。其他batch更换实验名和结果路径；只有b184用`v2`，其余`v1`。A为历史BM32最快默认，B为已测M64快照；B与当前默认全字段相同、ISA仅kernel符号名不同，**这是历史A/B的复现，不能称为本轮新测原生入口**。输入seed42、12个AB/BA配对、34组probe及原capture顺序保持；不要在同一GPU同时跑多个batch。

## 5. 新对话继续点

- 最新结论与不再重做的方向见 [report_s2.md](report_s2.md)。下一候选仅讨论：b112/640CTA的leader empty drain，尚未实施。
- 继续与真实K3合作：CLI `/home/ming/.kimi-code/bin/kimi --model inferact-kimi-k3`。旧session `session_ebbbe6de-6172-4d40-b346-b3d01ac8313b` 需在 `.scratch/b136_sbm64_m64_schedule_20260914_v1` 下恢复；换目录则建新session。
- 老规矩：一次一个实验、一次一个形状；CPU≤4并发；资源/无spill及K3审核通过后可直接GPU；结果须独立复核。当前没有GPU实验在跑。
