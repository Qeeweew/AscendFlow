# 当前实现与验证状态

更新：2026-10-15。直接生成迁移以 `61c5d06` / `760278f` 为审查基线。
**叶流水已删除候选路线搜索；层次状态压缩、一般nested CV和完整AST FA尚未完成。**
当前路线见[计划](refactor-plan.md)，算法与复杂度见[设计](design.md)。

## 当前实现

| 能力 | 当前结论 |
|---|---|
| AST runtime循环、explicit stage/slot | 已用于11个局部/串行父体示例，运行长度不展开 |
| 物理访问、ready/WAR/多缓冲与最后读取 | 已有局部验证子集，实际边界见回归 |
| 共同native milestone转移 | `native_order`由叶与serial region/call共用 |
| 叶生成 | `_requirements`直接确定协议；只为选中同步增量加边，已删除候选图、route和试删 |
| 条件同步 | 跳过Set/Wait仍传递输入前缀，Set effect不成为后续native fence |
| region/call与单child invocation | 仍复制首尾窗口并传播条件事实，这是下一主要替换对象 |
| 最终叶检查与cross-counter义务 | 批量传播；实际IR重提取，资源和进展检查保留 |
| 显式诊断 | `explain_loop`可调用`Graph.path`，普通编译不调用 |
| 一般AST nested mixed/CV、跨区域event共享 | 未完成；不能按串行模型推断支持 |
| 多参数BNSD、runtime商余、完整AST FA | 未完成；legacy FA证据不计作新DSL支持 |

未知动态别名、跨group ownership、全局phase、varlen/paged/sparse/persistent
仍需实际合同，未实现时拒绝。局部结果不证明FA性能目标已达成。

## 直接生成的实际证据

- 原生图开始时没有候选同步节点；选中协议后原地扩展翻译域，只有物化时才
  创建xDSL同步指令。没有逐缺口的协议路线搜索或逐候选图重建。
- 正/负关系差分比较增量图和实际指令流；真实Vector反例保持R后的early release，
  不等待后续无关X。新增反例拦截并修复过条件credit Wait跳过时丢失Scalar顺序。
- 11个AST例子及2个局部CV例子均生成最终ASC。同步数没有增加；分段CV省去
  `pv_15_mad`、`pv_18_mad`前两条已覆盖的Cube barrier，没有改变buffer或搬运。
- ASC并非逐字节相同：同一native位置的通知/等待排序及部分event编号变化。
  原生计算、访问和早release保持；尚无新设备性能结果，不能宣称无性能损失。
- scale、hierarchical GEMM、frame、segmented CV四份ASC通过CANN 9.1/dav-2201
  离线编译，源码/hash/编译命令与产物记录在外层runs。

## 编译规模与剩余热点

GEMM叶生成图由166节点/3357翻译状态降为91节点/1617状态。91来自
6×9个native里程碑、7×(3个Set节点+2个Wait节点)、1×2个barrier节点。
这只描述叶生成器，不能代表层次分析的状态已经压缩。

当前GEMM最终调用检查仍有：

| 项目 | 数值 |
|---|---:|
| native / 加同步后的指令模板 | 6 / 21 |
| 每次调用首/尾窗口 | 6 / 7个位置 |
| 相邻调用副本 | 2 |
| 每份窗口有效指令实例 / 图节点 | 176 / 571 |
| 父层图节点 | 1 + 2×571 = 1143 |
| 源完成里程碑 / 检查义务 | 41 / 182 |
| 节点/guard事实记录 / 最大guard数 | 10122 / 23 |

运行时N没有展开，但`指令×首尾窗口×调用副本×里程碑`以及逐事实guard包含
扫描仍放大成本。下一步将child转移提取并组合，不能只继续优化这1143个节点
的队列。实际计数见`runs/refactor/direct-sync/complexity-breakdown.json`。

五个独立冷进程、源码到ASC、import/CANN另计：

| 例子 | 审查基线p50/max | 直接生成p50/max |
|---|---:|---:|
| frame | 0.3842 / 0.4087s | 0.3923 / 0.4158s |
| hierarchical GEMM | 0.7594 / 0.7657s | 0.7516 / 0.7634s |

基线和当前的源码/package/ASC hash分别记录。首组基线frame计时开始时短定向
测试尚在收尾，不以这组细微差异判断性能优劣；没有显著冷编译加速结论。
直接生成测量在新增插入阶段取消检查之前完成；该新增检查不改变生成ASC。

## 回归与复现

本轮扩展同步/周期/层次212项通过（41.45s，条件前缀补充修复之前）。随后6项
增量差分与早release测试通过。稳定代码一次完整回归为841 passed、1 failed，
392.09s；唯一失败是取消测试仍挂在已经退出编译路径的`Graph.path`上。
更新到真实直接生成阶段并保留输入可复用检查后，取消/直接生成14项通过，2.15s。
未为这一测试挂点修正重复运行完整回归。Ruff和diff检查通过。

```bash
.venv/bin/python -m pytest -q compiler/tests/sync/test_direct_synthesis.py compiler/tests/test_budget.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/gemm/hierarchical.py --entry hierarchical_gemm --trials 5 --output runs/refactor/direct-sync/reproduce-gemm
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/state/frame_snapshot.py --entry frame_state_sum --trials 5 --output runs/refactor/direct-sync/reproduce-frame
```

证据集中在外层`runs/refactor/direct-sync/`：`final-examples/`保留前后IR、ASC和
差异，`cann/`保留离线编译，`validation/`保留测试日志，`cold-*`为五冷进程记录。
本段未运行设备或调整算子政策。整个重构目标仍未完成。

## 历史检查点

以下结果只适用于各自源码保存点。当时的“下一步”和未完成类名不覆盖上面的
当前结论；原始产物、版本和限制保留以便复现。

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
增至 13525 次。稳定索引扩大了部分查询的可访问作用域。再次审视后，下一步不止恢复范围
裁剪，而是替换“上层枚举端点对、再调用底层路径搜索”的求解结构。新的
[算法合同](design.md)与[替换顺序](refactor-plan.md)改用真实执行关系上的批量
完成信息传播。本次再审只修改设计和计划，尚未实现该算法，也没有启动新测试。

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
