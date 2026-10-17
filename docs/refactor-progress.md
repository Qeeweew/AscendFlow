# 当前实现、资格与未完成项

2026-10-17。同步生成已替换旧 Path/progress 链路，公共 DSL→IR→ASC 只有一条路径。
**重构尚不能宣布完成：默认 FA 冷编译目标尚未达到。** 下面分别记录实现、离线编译和性能；没有设备验证或算子性能资格。

## 代码与用户工作

接手 HEAD 为 `01ddce954b5bc269dc0d39e65d96bc1702164742`，实验快照为 `1231f4f`。
接手已有的 `tests/models/structured_events.py` 修改、`tests/test_model_reference.py`
和 `analysis/generation_order.py` 原文保存在外层
`runs/generation-redesign/incoming/`。独立 pre-sync 对照已集成；临时 generation_order
原型已由当前规则替换，没有复制进 legacy。实际提交身份以产品 Git 为准。

生成重构的冻结执行证据对应代码提交 `1286a783271ab123a430440e6e306140944df744`，
前两个实施提交为 `bf36729`、`0fc4923`。实际 Python package tree SHA256 为
`eebbf12fe2ad23afb8609e471de46d4793a18b9674efbf4644a077c2dda6a297`；
本页后续文档提交不改变该代码身份。

后续数据表示优化的独立代码身份与执行结果见下节，不借用冻结版的测试或
CANN 资格作为新代码已经验证的证据。

外层没有 Git，未创建根 Git；reference、旧 TileLang 工作区和容器没有修改或重建。
CATLASS、ops-transformer、ops-nn 的实际 HEAD 已再次核对，见 [source-notes.md](source-notes.md)。

## 已实施

- [生成规则及事实归属](generation-rules.md)：逐项输入、建立者、产出、消费者和失效条件。
- 真实 native capture/acquire、有序 unsigned guard、坐标往返和完整 mode-2 quorum。
- 分层 physical-step/phase 准入，拒绝匹配但互相先 Wait 的死锁；父层只消费实际边界。
- 由相邻值/存储 lifetime 构造稳定 cell credit。非稳定地址使用有限使用或明确 queue handback。
- credit 的全部成员进入有限 bank 台账；新 ID 共享检查全体成员，保留配置与最终 drain。
- 两个 AIV 共用一份 Vector body，由 runtime sid 区分地址、私有 UB 和 quorum 贡献。
- 物化不改变 native、地址、循环身份或 issue order；ASC 只打印最终 IR。
- 已删除旧 `event_progress`、`event_paths`、`event_lifetimes`、`progress_*`、
  `structured_event_sharing` 及其 solver 形状测试，无兼容入口或第二编译链。

`generated-lifetimes` 是来源标签，不能授权外部 physical IR，也不替代规则论证、
独立模型或硬件证据。输入与规则前提检查保留，未把必要检查搬进 debug。

## 数据表示优化（冻结版之后）

当前仅实施了有隔离原型收益的两项：单次编译内驻留完全相同的 raw Expr，
以及在 BoundPlacement、LogicalNotification、StructuredNeed 中按需保存 hash。
没有合入尚未测收益的 range/polynomial 索引。原型分析、火焰图和单次对照在
[外层报告](../../runs/compile-cost/head-0a749f4/analysis.md)。

生产实现使用独立 scope，退出或异常即释放；结构相等、原式定义性和每程序
同步检查保持。新增回归覆盖 arena 作用域、异常清理、不同事实环境及非法
`True`/`1.0` 与合法 `1` 的 key 碰撞，以及 hash 后的记录替换。

生产代码的 Python package SHA256 为
`9b8b67e3975e6c4779123f5ace337dc41fe431140405d24ff14d2bd2587c5220`。
默认 FA 的一次冷 source→ASC 为 **8.5678s**，import 0.3597s，进程总计
9.1774s，peak RSS 63652 KiB。对应
[原始冷测量](../../runs/data-structures/cold/fa-default/benchmark.json)
包含完整源码构造、分析、化简、输出和快照；未改变计时边界。
source/同步前后 IR/ASC hash 及完整分析报告与冻结版一致。

该结果是生产实现的单次 probe，不能与原型 8.8814s 混为同一个版本或统计
样本；仍未达到五次 p50≤5s/max≤10s 资格，不执行五次验收。

本次生产实现全套回归为 **1623 passed in 148.68s**，见
[完整日志](../../runs/data-structures/full-tests.log)，SHA256 为
`6c446d58de5975f15310e1aed1a75b97d4fb15c1fdff8ae294c5f4e6239923f6`。
Ruff 与 `git diff --check` 通过。四项新冷进程 probe 的源码、source/同步前后
IR、ASC hash 和完整分析报告均与冻结版相同，见
[冷测量清单](../../runs/data-structures/cold/source-to-asc.json)：

| 算子 | 单次冷 source→ASC | import | 冷进程总耗时 | 独立构建 CANN | 独立构建总计 |
|---|---:|---:|---:|---:|---:|
| Vector scale，B64 | 0.4858s | 0.3500s | 1.0683s | 5.3073s | 6.3270s |
| 默认 GEMM | 0.6379s | 0.3442s | 1.1673s | 5.4754s | 6.6785s |
| 嵌套 CV | 0.9182s | 0.3477s | 1.4676s | 5.6483s | 7.1414s |
| 默认 FA | 8.5678s | 0.3597s | 9.1774s | 8.0131s | 17.3919s |

四项均在新的完整 300 秒进程组期限内取得实际 `.o`、`.aibin`、`.so`，见
[新构建清单](../../runs/data-structures/qualification/source-builds.json)。
冷测量和 CANN 属于不同构建，不直接相加。新构建的 source/同步前后 IR/ASC
hash 与冻结版逐项一致；本次没有设备数值或性能验证。

## 独立回归

实际最终 IR 模型检查队列、controller、二值消费、mode-2 四位 ID/十五贡献容量及字节 RAW/WAR/WAW。
独立 pre-sync 快照提供预期 native、地址和读版本；生产生成器不提供 oracle 的预期边。

默认 FA 模型使用原始默认政策，Sq/Sk 未特化，有限实例为
`(128,1024)`、`(128,2048)`、`(128,3072)`、`(512,2048)`，检查 group 0 和 19，
包含空 owner、两个 query invocation、双 AIV 和多 KV 复用。它不执行浮点运算，
不外推所有输入或设备性能。`invocations` 表示 controller 原始 body 重入，
不隐含 mixed kernel 的跨 actor launch 完成；顺序 launch 由 runtime 排序。

首轮全套为 1566 passed、29 failed、1 error（146.17 秒）。用原 HEAD 独立复现的
相关旧测试为 51 passed、21 failed、1 error（15.47 秒），记录在
`runs/generation-redesign/baseline-failures.log`：主要为旧诊断字符串和两项从未接通的
positive 测试（prefix、循环内 allocation）。现已明确它们的拒绝边界，并修正
一个未对齐 alias fixture，使其实际触发 alias 检查。没有把这些拒绝算成 DSL 支持。

最终全套为 **1617 passed in 158.19s**，包含 child boundary 事实复用回归。
日志为 [qualified-tests.log](../../runs/generation-redesign/qualified-tests.log)，SHA256 为
`dd57821a5a0362acfca8f8917db6ffa2b646e3f7f8c7d0438508886631681b3d`。
Ruff 和 `git diff --check` 通过。汇总身份与结果见
[evidence.json](../../runs/generation-redesign/qualified/evidence.json)；其整体状态为
`incomplete`，功能/离线检查为 `passed`，不能据此宣称耗时资格或设备资格通过。

## 生成重构冻结版的编译证据与计时边界

默认 FA 政策保持 B1/Hq8/Hkv8/D128、BQ=BM128、BK1024、CK128、VR8、GROUPS20、f16。
source→ASC 包含源码读取/构造、全部分析/化简/输出和内部快照；import、进程总耗时、
文件发布和 CANN 分别记录。没有跨编译结果缓存。

最终四项均从公共 Python 源码取得同步前后 IR、可读 ASC、`.o`、`.aibin` 和 `.so`。
构建清单为 [source-builds.json](../../runs/generation-redesign/qualified/qualification/source-builds.json)，
冷进程测量为 [source-to-asc.json](../../runs/generation-redesign/qualified/cold/source-to-asc.json)。
每项构建使用现有进程组硬期限，覆盖 Python 构造至 CANN，总计上限 300 秒。

| 算子 | 单次冷 source→ASC | import | 冷进程总耗时 | 独立构建的 CANN | 独立构建总耗时 |
|---|---:|---:|---:|---:|---:|
| Vector scale，B64 | 0.4948s | 0.3462s | 1.0179s | 5.3526s | 5.8865s |
| 默认 batched GEMM | 0.7015s | 0.3494s | 1.2684s | 5.3354s | 6.0908s |
| examples/cv/nested.py | 1.0455s | 0.3472s | 1.6178s | 5.5951s | 6.6996s |
| 默认 FA | **10.4814s** | 0.3464s | 11.0821s | 8.9355s | 19.9849s |

冷测量与 CANN 构建属于不同进程，表中两类时间不直接相加。两者的 source、
source IR、同步前后 IR 和 ASC hashes 已逐项核对相同；实际库与 CANN 清单一致。
ASC 使用 `dav-2201`、`-O2 -fPIC -std=c++20` 和
`-mllvm -cce-aicore-dcpreload-args=false`，完整实际命令、编译器版本和产物
hashes 保存在每个算子的 `*.compile.json`。

默认 FA 的 [source IR](../../runs/generation-redesign/qualified/qualification/fa-default/source.xdsl)、
[同步前 IR](../../runs/generation-redesign/qualified/qualification/fa-default/before.xdsl)、
[同步后 IR](../../runs/generation-redesign/qualified/qualification/fa-default/after.xdsl)、
[ASC](../../runs/generation-redesign/qualified/qualification/fa-default/attention_target.asc)
和 [CANN 清单](../../runs/generation-redesign/qualified/qualification/fa-default/attention_target.compile.json)
均已保存。ASC SHA256 为
`b8bd67c607f9dacf883fb313e65fc76ff22142c6175acda6806d5bdc6cb05b7e`。
其余算子的对应产物在同级 `vector/`、`gemm/`、`nested-cv/` 目录。
直接审查的 ASC 保留 `__mix__(1,2)`、runtime sub-block ID 和共享 Vector body，
无 `PIPE_ALL`；native 模板数与实际打印数一致。

以上各算子只做一次冷进程 probe。**默认 FA 未达到 p50≤5s、max≤10s 的目标，
没有执行五次耗时验收，也不能从单次数据推出 p50。** 文件发布另计约 1–2ms。
此前 `fa-coefficient-probe` 的 10.6469s、其他阶段的 14.2963s/67.5209s
来自各自不同快照；不能混成当前五次成绩。带 cProfile 的 74.5243s 只作历史诊断。

## 默认 FA 的当前工作量和耗时来源

以下均来自上述最终冷测量，不使用旧 solver 的失败预算或历史 profile 解释当前耗时。

| 当前工作 | 秒 | 具体规模或操作 |
|---|---:|---|
| endpoint domains | 1.9026 | 453 组端点 guard；1162 次坐标往返；2702 次配对蕴含查询 |
| notification binding | 1.2768 | 526 条 retained requirements，绑定并合并实际捕获/获取为 264 条通知（含 fence） |
| queue order | 0.8888 | 7 个 region，累计 662 个位置、354 个保留边界、5113 条边界传递关系 |
| credit lifetimes | 0.8098 | 161 个 cell 分片组成 69 个 physical credit；同 occurrence 分片域检查 |
| dependencies | 1.0797 | 249 个投影访问、90 个 cell、726 条诊断 needs，其中 loop carries 485 条 |
| bounds / admission | 1.2689 | 43 个 view/45 个 domain 合同；89 个 read proofs、90 个初始化覆盖检查 |
| materialize / scalar simplification | 1.3487 | 445 个计划动作及其有序 guard 的 SSA 构造和化简 |
| event banks | 0.2574 | 消费既有生命周期、检查新增 ID 共享，12 个 mode-2 ID（0..11） |
| lowering、构造、快照、输出及其余工作 | 1.6489 | 最终 ASC 445929 字节；无 runtime 迭代展开 |

源码只有 44 个 native 模板、6 个循环，但分析对象仍按物理参与者、cell、
首末/loop-carried 分片和实际 activation 域细分。69 个 credit 的合并发生在域
建立之后，尚未减少前面按通知做的全部域工作。绑定阶段还枚举已有 acquire
候选并做实际次序比较；queue order 逐层枚举相关 Wait/Set，逐保留边界建立
有限可达摘要。父层消费子层边界，不重开子层路径，但摘要本身仍较密：
累计 zero/positive relations 为 5888/2172，summary visits 为 13263。

因此同步生成加分配为 5.1353s（总计约 49%），耗时分散在符号域、次序和
分片处理；不再是旧 Path 搜索触预算。上述计数是操作次数，不代表每次查询
成本相同；当前快照没有函数级 profile，不能据此声称已量出底层算术热点。
后续要降低不必要的分片/端点/边界关系及重复符号变换，须保持原式定义性、
实际配对、全部 reader release 和新增资源约束，不能删检查或只缓存旧查询。

其他三个算子的实测规模如下：

| 算子 | native 模板 | needs / retained | notifications | cell credits | plan actions |
|---|---:|---:|---:|---:|---:|
| Vector B64 | 3 | 12 / 8 | 8 | 8 | 22 |
| GEMM | 6 | 36 / 24 | 20 | 20 | 48 |
| 嵌套 CV | 8 | 60 / 44 | 28 | 28 | 76 |

默认 FA 的 window constructions 和 streaming rings 都为 0；参数化 body 没有
按 runtime 长度展开。其他 streaming 规则有独立回归，不能把该计数推广为所有算子。

## 支持与具体拒绝

支持的规则覆盖双缓冲 GM→UB→Vector→GM、真实最后 reader/WAR、slot/代次复用、
Cube/Fixpipe、参数化嵌套和相邻循环、共享双 AIV CV 以及本页默认 FA 结构。
数学/依赖前提、完整 quorum、phase rank 或有限资源不能建立时具体拒绝。

未知动态 alias/ownership 需要可验证输入合同；外部根的 disjoint pointer 合同保留。
数据相关控制、无范围的 scalar carry、未封闭的多时钟/稀疏 activation、prefix native
地址、循环内 allocation、continuous FA 状态边界仍不宣称支持。没有全局 barrier、
额外搬运、减少 overlap、调大旧 solver 预算或 FA 名称/shape 特判。
其他 FA constexpr 变体没有本次完整资格；未做设备数值验证或性能测量，不能
宣称相对 AscendC/ACLNN 没有性能损失。

## 复现

工作区根运行，使用独立 `.venv`；产物留外层。以下每条实际算子构建都有 300 秒总期限：

```sh
PYTHONPATH=compiler PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m xdsl_npu.cli --input compiler/examples/vector/scale.py --entry scale --parameters '{"B":64}' --output runs/reproduce/vector --shared --timeout 300
PYTHONPATH=compiler PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m xdsl_npu.cli --input compiler/examples/gemm/batched.py --entry batched_matmul --output runs/reproduce/gemm --shared --timeout 300
PYTHONPATH=compiler PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m xdsl_npu.cli --input compiler/examples/cv/nested.py --entry nested_cv --output runs/reproduce/nested-cv --shared --timeout 300
PYTHONPATH=compiler PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --output runs/reproduce/fa --shared --timeout 300
PYTHONPATH=compiler PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.compile --input compiler/examples/flash_attention/attention.py --entry attention_target --mode normal --trials 1 --timeout 300 --output runs/reproduce/fa-cold
```

上面的 `--trials 1` 复现单次诊断，不是五次验收。全套回归在 `compiler/` 下运行：

```sh
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -m pytest -q --tb=short --basetemp=/workspace/xdsl-910b/runs/generation-redesign/qualified-test-tmp -o cache_dir=/workspace/xdsl-910b/runs/generation-redesign/pytest-cache
```

## 历史失败只作溯源

`91eb40d` 的 125.44s、`475f49b` 的 56.87s、Path 接入但共享树之前的 56.69s
均是失败耗时，未产生 ASC/CANN，不能与成功构建作提速倍率。
原证据在 `runs/scalar-cost/`、`runs/factored-progress/`、
`runs/lazy-progress/fa-default/` 和 `runs/lazy-progress/build-source.json`。
旧 `4b2c432` 的 shared L0C、`dbcef70` 的双 AIV 记录不提供当前资格。
