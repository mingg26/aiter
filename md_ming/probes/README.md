# probes —— trick 文件点名的测量工具（随身带，不然那些"尺子"在新机器上不存在）

这三个脚本原在 `.scratch/diag/`（按 README 的取舍原则 `.scratch` 不迁移），但它们被
`trick_mi350x.md` / `trick_fused_reduce.md` 指定为**权威尺子**，所以单独抽出来。
**两个可直接跑**（`kvo_bw_floor.py`、`spike_cache.py`：只依赖 torch + triton/gluon），
**一个是参考实现**（`kvo_scope_ab.py`：它 `import prefill_bench / kv_outer_dense / kv_outer_v3
/ kv_outer_v4 / kv_outer_v6`，这 5 个兄弟模块仍只在 `.scratch/diag/`，**换机器后它跑不起来**，
带过来是留 A/B 的构造方式做参考；结论数字已记在 `trick_fused_reduce.md` §7）。
`kvo_bw_floor.py` 和 `kvo_scope_ab.py` 有 docstring 说明为什么这么测，`spike_cache.py`
是 27 行无注释的最小复现。

| 脚本 | 被谁点名 | 作用 |
|---|---|---|
| `kvo_bw_floor.py` | `trick_mi350x.md` §1（"唯一可引用尺子"） | 教科书式 flat read / copy / triad 带宽探针，跨 dtype/block/warp，在 MALL 内与远超 MALL 两个尺寸上定标。**任何"我们离峰值多远"的说法都必须先跑它**，不要拿某个 kernel 的形状当带宽探针。 |
| `spike_cache.py` | `trick_fused_reduce.md` §2（Triton cache modifier 位映射表） | 编一个最小 gluon kernel，把 `.cg`/`.wt`/`.cv` 等 cache modifier 编出来，再去 ISA 里看实际落成什么位。**Triton 的 modifier 名字和 AMD 的位不是一对一**，改 cache 策略前用它确认。 |
| `kvo_scope_ab.py` | `trick_fused_reduce.md` §7（跨 XCD `.wt/.cv` 入场费） | 量化 device-scope partial 流量的入场费：`.wt` store + `.cv` load 的 sc0sc1 版本 vs 普通版本的 A/B。得出"跨 XCD 生产者-消费者 combine 入场费 > 收益"这个反面终论。 |

注意：`m64_mat3x.py`（`trick_mi350x.md` §3 锚点表里 0.20 µs 那条的出处）是 271 KB 的
vendored 整核变体，不是通用工具，没有迁移；那个锚点值本身已记在 trick 文件里。

## MegaMoE 默认与重现

- `audit_megamoe_defaults.py --repo <aiter仓库>`：CPU执行真实selector，核对`megamoe_defaults.json`的37点固定账本。
- `prepare_megamoe_replay.py --experiment <历史实验> --out <新目录>`：校验全部原source pins，只允许`megamoe_replay_rebase.json`中已审过的源码变更，生成独立冻结重现目录；不启动GPU，不修改旧结果。
- 命令、依赖、证据范围见 [../redo.md](../redo.md)。账本不替代性能测量，准备成功不等于GPU复验通过。
