# 当前重构状态

**重构尚未完成；旧实现与兼容路径继续直接删除。** 真实双runtime循环/双slot已完成条件生命周期、最终IR有限模型与CANN闭环；默认FA的整数规约缺口已修复，当前阻塞是source顺序的物理反依赖与实际排步的关系。算子调优未恢复，本轮没有设备结果，编译耗时未达标。

算法合同见[编译不变量](compilation-invariants.md)，职责见[源码组织](source-layout.md)，剩余步骤见[重构计划](refactor-plan.md)。历史状态由产品Git及外层runs追溯，不作为当前资格。

## 唯一编译链路

```text
DSL/source → 保留控制与地址SSA的native IR
  → 物理访问/准入/ownership → region依赖 → 逻辑通知
  → 精确事件域 → 进展/生命周期/资源分配
  → 实际同步IR → 通用纯整数化简 → ASC → CANN
```

- 被替代的planner、旧物理IR、解码器与兼容入口已删除，不移入tests保存。算法留在examples。两个AIV共用body，以runtime sid区分地址和角色，分析保留各自ownership和贡献。
- 参数域直接使用`EventDomains.cases`，多余数窗口保留实际membership、rank和首末成员。各case使用自己的编号模数，一个通知只预留一次最大宽度。没有全程序case产品或runtime迭代展开。
- 同一`IntegerFacts`持有本次编译的完整有序前提；原始端点的算术定义性早于交集和空域裁剪。父region消费child接口，不把控制退出当异步完成。

## 本轮完成的重构

1. `event_paths.py`统一消费到复用的真实路径。每条路径保留activation及队列顺序条件，`guard_cover.py`证明这些条件覆盖整项义务。内层复用、外层重入和有限ID共享使用同一合同；旧bridge、单steps字段和布尔顺序包装已删除。
2. `EventDomains`以实际`coordinate+stage/lexical/phase`构造条件顺序。生命周期与progress共同消费；progress不再另写一份native顺序推导。child queue cut、stream步长下界仍为各自保守阻塞包络，不提供buffer退休证明。
3. `modular.py`只处理证明中的相关多项式余数；整数三分和余数mask都不成为运行期case或排步。`(p*m+g)%2`整体关联保留。大小比较先求原余数范围内的mask，不能以模等价代替大小关系。
4. 路径搜索逐层判断完整覆盖，覆盖后停止扩展。相同端点仍保留不同条件，部分direct路径不能遮掉其他相位所需的间接路径。候选按需构造；状态、边、offset和证明节点/谓词工作均有预算，失败查询也计数。
5. Attention runner删除旧`cross_query_verified/query_verification`报告合同及测试fixture，只读取当前source构建产物。build状态、manifest及计时文件一致性属于来源检查，不冒充独立同步证明。`Compilation.source_ir`不再可选，source/schedule始终写出。

## 真实双slot闭环

源码为`tests.analysis.test_event_parameter_cases.nested_slot_source(2)`，外层p和内层g均为runtime循环，物理slot为`(p*m+g)%2`。连续RAW流的下一同ID复用需经两条不同slot的WAR通知；m=4时g=0与g=1不能任选一个peer。现已保存完整条件路径覆盖，删除任一peer的回归仍拒绝。

- [公共CLI/CANN构建](../../runs/lease-cover/slots-build/build-result.json)通过：总60.00秒，Python57.38秒、CANN2.41秒；这是功能构建记录，期间有短CPU回归，不是独占性能基准。
- [同步前IR](../../runs/lease-cover/slots-build/before.xdsl)、[最终IR](../../runs/lease-cover/slots-build/after.xdsl)、[ASC](../../runs/lease-cover/slots-build/slots.asc)可直接查看。2个native模板仍只打印2次，12条通知、29条同步指令，未增加搬运或更改buffer/排步。
- 最终IR独立事件/字节模型覆盖n=0..3、m=0..5共24项：包含空父/空子、短循环、奇偶长度和跨父slot复用。n=3,m=5有30个native实例、30次通知消费，读写各7680字节；见[模型记录](../../runs/lease-cover/slots-final-model.log)。这是有限实例，不是任意N证明或设备数值结果。
- 产品模型回归只完整编译一次，24项共享最终产物。逐层覆盖即停、失败工作计数和顺序接口清退后，同次编译的IR/ASC仍与上述CANN产物逐字一致；见[回归及hash记录](../../runs/lease-cover/slots-sync-regression.log)。末次progress改动仅更换顺序证明消费者，相关正负回归另行通过，不改变物化方案。

本轮受影响分析集成261项通过，见[日志](../../runs/lease-cover/integrated-tests.log)。最后progress统一消费者后6模块38项通过（5.29秒）；Attention runner与小型source产物组78项通过（3.39秒）。各组有重叠，不相加称独立总数；未跑全套suite。全局Ruff与diff检查通过。

此前`c1d4465`上的嵌套CV曾通过CANN和n=0/1/2、groups=0/1、双AIV的最终IR模型，见[构建](../../runs/refactor-cases/nested-cv/build-result.json)。这是基线资格，不替代当前完整算子验收。

## 本轮整数表示与fence职责修复

- `indexing.py`现在由一个有界多项式实现同时承担完整索引规约和保留函数的ring规约。typed函数key规范化实参，raw代表按查询选择；`IntegerFacts.normalize/bounded`共用同一身份环境。删除原`$resource_atom_N`伪参数；公共输出仍为Expr。完整unsigned前缀检查继续读取原表达式。
- 声明可整除动态商进入rem/floor/ceil/min/max时，嵌套同义索引现已可判等。独立标量/索引/窗口集成159项通过（3.21秒），见[日志](../../runs/atom-identity/scalar-window-tests.log)。之后补充的跨查询raw代表隔离与共享DAG回归也通过；此前后置前提、数学零掩盖非法运算等负例保留。
- 同流水fence保留选择阶段的capture与全部端点准入，直接按原consumer guard物化一次，不再构造无消费者的binary事件case/window/lease。7个零/短/奇偶循环最终IR模型实例和非法端点负例通过。资源、准入、实际同步与域的集成136项通过，见[日志](../../runs/atom-identity/integrated-tests.log)。全局Ruff与diff检查通过，未运行全套suite。
- 当前嵌套CV再次通过公共CLI/CANN，总24.09秒，见[构建](../../runs/atom-identity/nested-cv/build-result.json)。最终IR/ASC与先前合格产物逐字相同，直接复用n=0/1/2、groups=0/1、双AIV的有限证据，不重跑未改变产物；见[hash/模型记录](../../runs/atom-identity/nested-cv/finite-model.log)。8个native模板各打印一次，AIV native body仍只有一份。这是功能构建，不能称编译性能达标。

## 默认FA的当前拒绝与下一步

修复后默认DSL的公共CLI/CANN构建请求已越过全部事件域，在146.05秒被`structured-event-progress`拒绝，尚未产生成功ASC/CANN产物，见[状态](../../runs/atom-identity/fa-default/build-result.json)。失败耗时不是成功编译性能。

一次60秒上限的只读提取在24.93秒完成source→EventDomains（不运行progress/lease/分配）：44个native、473条依赖、213条通知、138个参数case、245个窗口。记录了原始源码/编译器hash、IR、端点及派发轨迹，见[分段记录](../../runs/atom-identity/progress-gap/checkpoint.json)。这是诊断时间，不能和完整构建各阶段直接相减当精确性能归因。

`Sq=128,Sk=2048,group=0,query=0`下存在明确队列环：

```text
n15 Wait(FIX→M)，QK(kv1,chunk0)前，native前缀38
  → QK M不能完成，n21 Set(M→FIX)不能完成
  → n21 Wait，QK FIX前，前缀43，阻塞FIX队列
  → PV(kv0) FIX及n15 Set位于同FIX队列后面，前缀106
  → n15 Wait无法满足
```

只使用这两对真实通知、两个directed bank的fresh token，省略其余同步/implicit completion的148-native有限模型仍报告死锁；n21两端原始activation各6项均为真。见[证据](../../runs/atom-identity/progress-gap/evidence.md)与[有限队列记录](../../runs/atom-identity/progress-gap/finite-cycle.log)。这排除了本次环只是opaque child cut误报的解释，但不证明FA算法或固定issue_order本身非法。

当前`structured_dependencies`明确以source顺序合成全部RAW/WAR/WAW，随后才绑定dispatch。n15由source回边的PV(kv−1)最后读取→QK(kv)覆盖产生，而实际private L0C的稳态发射为QK(j)→PV(j−1)→QK(j+1)。模型尚未保留足够的值版本/完整kill生命周期事实来判断这类独立临时值能否换序。**下一步重构值依赖与物理存储复用的接线**，保留read→writer、RMW递推和外部可见顺序，再由实际dispatch推导最后读取→下一覆盖；不能直接删除n15（它还保护plane1）、改用dispatch最近writer或修改FA排步。

其他明确边界：

- 同类五slot源码仍在更早的memory frontier拒绝`multiple activation residues`；证明模块支持period5不等于整个五slot源码已支持。
- 条件路径搜索和进展图组合最坏仍会增长，不能声称同步算法整体线性。此前双native小例约一分钟，当前尚无完整FA资格；必须在结构和依赖合同稳定后按阶段与工作量分析，性能问题没有解决。
- event ID及ASC里仍有多余min/max、除一和取模。通用标量pass已接入，但可读性未达标；没有做新的算子性能调优。
- FA变体、设备正确性及ACLNN/profiling/理论硬件性能对比没有当前资格，不能用历史旧路径结果替代。

## 复现

工作区根目录执行：

```bash
# 当前参数路径、删除peer反例与整数覆盖
cd compiler
../.venv/bin/python -m pytest -q tests/analysis/test_event_path_cover.py tests/analysis/test_guard_cover.py tests/analysis/test_conditional_queue_order.py
# 只编译一次，24个最终IR模型实例共用
../.venv/bin/python -m pytest -q tests/sync/test_structured_event_semantics.py -k nested_slots
cd ..
# 默认FA：目前会按上述合同明确拒绝，单次完整构建最多300秒
.venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --compile --timeout 300 --output runs/reproduce/fa-default
```

产品文件仅提交compiler Git，运行证据留外层runs；不推送或发布。
