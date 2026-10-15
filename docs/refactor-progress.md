# 当前实现与验证状态

更新：2026-10-15。目录整理保存点为 `1d5c092`，公共查询保存点为 `0ff08c2`。
当前已替换 serial region/call 的同步选择与增量关系查询；
**整个重构仍未完成，新 AST FA 仍未接通**。
下一部分的唯一实施顺序见[重构计划](refactor-plan.md)。

## 已实现与未实现

| 能力 | 当前结论 |
|---|---|
| AST 前端、runtime 循环、explicit stage/slot | 已用于11个局部/父体示例；不按运行长度展开 |
| 局部 MTE/Vector、Scalar、Cube/Fixpipe | 各自已验证子集可生成 ASC 并离线/设备运行 |
| 串行父体 init/多个 child/use | 已支持受限 local actor、检查过的 frame 和共享 local scratch |
| 跨父调用共享 readonly GM | 依据实际 R effects 和 group 范围验证；有1GiB设备证据 |
| 同一调用多 region 写/读同一个 GM frame | stationary 已检查范围支持；保持 frame SSA 身份 |
| fresh frame 跨调用 | 只有已验证不相交的 frame 可省跨调用存储冲突，事件寿命仍检查 |
| 多 region 漂移范围 | 只读的已有子集保留；可写的 translated/drifting frame 仍拒绝 |
| 源码结构入口 | `region_lowering` 已统一准入和 actor 构造 |
| 公共完成端点/条件覆盖查询 | Site 与条件路径已共用；region/call 使用稳定端点索引、增量可达前沿和共同缺口生成器；层次 CV 未接通 |
| 一般 AST 嵌套 pipeline/mixed 层次 | 明确拒绝，常量界也不隐式展开 |
| 多参数 BNSD、runtime task/query 商余关系 | FA 所需完整子集未接通 |
| 跨 region event ID 共享 | 未完成通用组合；不能套用 legacy 支持声明 |
| 新 AST 完整 FA、宏、真实编译缓存 | 未完成 |
| legacy FA/变体和手写 AscendC FA | 有各自历史证据；不计入新 AST 支持 |

动态间接别名、不平衡 ownership、跨 group 交接、全局 phase、varlen/paged/sparse/
persistent 等仍需专门合同。无法证明时明确拒绝，不加任意 barrier 或调用 fallback。

## 当前明确问题

1. 已删除原 `BoundaryProof.covered` 和 `CallProof` 的独立路径搜索，叶及调用
   使用共同 Site/CountDomain/条件路径。`RegionCompletion` 仍是串行控制适配器，
   不能用于重叠 CV；单 count 域尚不能表达完整 query/task/child 时钟。
2. frame snapshot 的跨 region 传递路径已能组合，直接 MTE3→MTE2 通知缩为 N=0。
   region/call 已直接生成未覆盖条件，不再逐候选试删。叶流水原有的 witness 剪枝仍在。
   有界 CountDomain 用于同步条件；一般 bounded child transfer summary 仍明确拒绝。
3. 同一示例的 slot0 最后读取已按奇偶选择：偶数 N 释放在 N−2，奇数在 N−1；
   generated 使用互斥 id2/id3，参考使用一个 id2，不能声称 event packing 完全相同。
4. 核心入口统一不等于同步算法统一；局部设备对照接近也不等于 FA 性能达标。
5. benchmark 仍有重复设备执行流程。本轮只收敛公共启动、产物校验和数值参考，
   没有用大型 benchmark 框架替代同步重构。

## TileLang 对照后的同步与测试重构

核对的参考 HEAD 为 `6ba65d204c937fcb1c43b1a22b32979506595655`，未更新参考仓库。
源码入口为 `insert_sync.cc` 的 `AnalyzeNodeList`、`OptimizeSyncPoints`、
`DepClosure::Saturate`。采用“已有覆盖 → 必要边 → 增量关系”的组织，不搬入
TileLang 的完整闭包、特殊协议或跨时钟投影。

- `sequence_plan.py` 共用 region/call 的缺口生成、条件合并与资源分配。
- `RegionCompletion` 不再按已选同步列表重建节点池；`CompletionPaths` 保留
  同源/同条件的搜索状态，新增边只通知到达其源端点的前沿。
- 最终 verifier 仍从实际 IR 重提取；event 配对/复用、内存关系与独立有限模型
  各自保留，不把数据覆盖当作资源寿命证明。
- [测试](../tests/README.md)分为纯关系、生成、最终 IR 破坏验证，公共有限模型
  移至 `tests/models/`。没有删除原测试函数；多运行边界共用不可变编译产物。

对照基线为 `0ff08c2` 加本轮开始前已有的未提交直接生成器修改，原始 patch
保存在运行目录中。验证记录位于外层 `runs/refactor/tilelang-sync/`：两组定向
检查为 53/93 项通过；
11 份 AST 默认政策下 ASC SHA256 与本轮开始时全部相同，frame 的 dav-2201
CANN 离线编译通过。最终全量回归 **826 passed / 415.36s**，Ruff 与 diff 检查
通过；其中旧 FA 的 2048-group ownership 检查为 131.75s，双向 quorum 检查为
39.21s。日常同步修改使用分层测试入口，不再默认反复运行这些历史全量检查。
本轮没有设备运行或算子调参。

frame 默认 B4096/G32 的五独立进程源码→ASC：改动前 p50 **1.2551s**、
max **1.3551s**；改动后 p50 **1.3003s**、max **1.3166s**。两组 ASC hash
相同，包源码 hash 各自记录，import/CANN 单列。**没有编译性能改善结论，仍未
达到 1s 门槛。** 一次 B16/G1 调用计数对照显示：72 次覆盖查询保持；旧节点池
查询 71 次变为四个分析阶段各建一次索引，但 native 关系实际求值从 11516 次
增至 13525 次。稳定索引扩大了部分查询的可访问作用域；下一步需要在复用前沿
的同时保留源/目标间的作用域裁剪，而非继续添加全局缓存。

## 公共完成关系迁移：保存点 0ff08c2

`completion.py` 统一条件集合、完成端点和带条件路径；`region_completion.py`
集中现有串行域的真实 native 关系，域/调用编号本身不提供顺序。Native 关系
缓存只在同一个分析快照内共用，最终 verifier 从实际 IR 建立新快照。

新增条件交集/路径并集的有限图差分，及新 query QK→旧 query PV 的关系级反例；
后者仅检查公共关系查询不强加 query 串行，不能作为 AST CV 编译证据。
实际 frame 的直接通知只保留在零次 child；正长度由初始 MTE3→V 和相应最后
Vector reader→MTE2 的现有路径覆盖。偶数 N 仍释放 N−2 的 slot0，不等待最后
slot1；没有增加搬运、buffer、barrier 或 per-call drain。

- 完整回归801 passed，397.82s；其中182项相关回归先行通过，Ruff通过。
- 11份 AST 中10份 ASC hash不变；frame 只改变该通知条件及互斥等待排列。
- CANN generated/AscendC offline/shared通过；全卡检查后physical1→logical0，
  18条小例（含舍入）和两条1GiB大例精确正确，输入/guard完好。
- 新大例的非profile ACL stream间隔为755.547/765.463μs；包含host空隙，
  不能替代三批kernel profiling或作为性能改善结论。新版本profile尚待验证。
- 初次统一后冷编译p50为1.608s；按profile共享查询和规范化后一次五进程记录为
  1.373s/max1.393s、RSS39168KiB，**仍超过1s门槛**。各次源码/package hash见原始记录。

证据位于外层 `runs/refactor/unified-sync/`。本检查点没有达到重构第1步的全部
退出条件。当时的试删生成器已由本页新记录替换；编译预算、一般 region 组合
及多级 CV 仍未完成。

## 保存点 3f95139 的 frame 验证

例子为 [frame_snapshot](../examples/state/frame_snapshot.py)：显式写入初始 state
的 GM snapshot，执行有序递推，最后重读 snapshot 并加到 state，再覆盖输出。
这次搬运属于源算法，不是编译器偷偷加入的同步手段。

保留两个 runtime 循环、7个 native 模板；B4096 时每 AIV UB为49152字节。
parent=0 不写；child=0 输出2×Initial；FP32 snapshot 加法在递推后执行。
缺初始化、缺 ready/retirement、部分范围初始化和大常量界均有回归。有限 byte/event
模型覆盖小域，并不证明任意动态程序。

历史记录位于外层 `runs/refactor/frame-regions/`：

| 项目 | 已有结果与限制 |
|---|---|
| 相关 Python 回归 | `final-tests.log` 217 passed；追加范围片段2项通过，非全测试集 |
| 冷源码→ASC | 五次 p50 0.837261s / max0.838811s，RSS38000KiB；import/CANN另计 |
| 证明访问开销 | invocation path visits 63720→24620；优化期间 ASC 不变 |
| CANN 与小设备 | generated/AscendC 编译通过；20条精确结果，包括舍入反例 |
| 固定大例 | B4096/G32/outputs16/tiles128，X=1GiB；2097152个输出精确正确 |
| 三批 profile 的批中位数中位数 | generated754.114μs，AscendC752.824μs，比值1.001714 |
| 工作量 | GM请求1107296256字节；请求速率1468.34GB/s，不是HBM实测带宽 |

设备物理1→逻辑0；warmup5/repeats30；没有新 FA 设备结果。两版本差异在观测
波动范围内，不外推性能无损。旧 manifest 中相对路径要用明确的 `--build-cwd`。

## 本轮目录清理的验证

- 清理前保存点 `3f95139`；未修改 `src/xdsl_npu/` 的算法或 native 契约。
- 原 examples 的65个非缓存文件重新归位；保留16份算法源码，11份当前 AST、
  5份 legacy。新增目录标记和索引帮助定位，不保留旧入口兼容壳。
- 10个 profiling shell 合并成一个 `benchmarks/profile.sh`；ASC 模板集中在
  references，公共数值参考脱离设备 runner，三类 analyzer 共用产物校验。
- 完整 Python 回归 **785 passed，396.68s**；Ruff与diff空白检查通过。
- 11份当前 AST 示例迁移前后生成 ASC 的 SHA256 全部相同。
- 29个模块命令入口的 `--help`、通用 profiling 包装器和全部本地文档链接通过；
  9份独立 ASC 模板逐字节不变，旧 frame profile 重新汇总的 JSON 完全一致。
- 新路径下 frame 的 generated/AscendC，以及独立 FA reference 均完成 CANN
  离线/shared 编译。FA reference 成功仍不代表新 AST FA 可编译。
- wheel 不包含 examples/benchmarks/tests；从隔离目录加载 wheel 内容，编译
  自带 DSL 源码成功，过程中未导入这三个仓库包。
- 新 frame 源码五次冷源码→ASC p50 **0.835902s** / max **0.836449s**，
  RSS37964KiB；import/CANN另计。与旧测量接近，不声称本轮改善了同步复杂度。
- 本轮没有启动设备运行或重新测算子性能；历史设备结果仍对应各自源码/库 hash。

本轮具体命令及测试日志保存在外层 `runs/refactor/cleanup/`。源码变化后的冷
计时记录单独保存，不能用旧 package/source hash 冒充新测量。

## 文档与历史

[设计](design.md)规定目标语义；[DSL](dsl-multistage.md)标注目标片段；本页负责
当前支持；[计划](refactor-plan.md)负责下一步。详细旧算子与设备记录保留在
[文档索引](README.md)，只适用于各自保存点。

之前长篇按时间追加的进度移至[历史记录](history/refactor-progress-20261015.md)，
其中旧“下一步”不再作为当前计划；原始测量和 hash 不重写。
