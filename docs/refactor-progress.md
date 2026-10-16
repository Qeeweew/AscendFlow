# 当前重构状态

**重构尚未完成；旧实现与兼容路径继续直接删除。** 真实双runtime循环/双slot已完成条件生命周期、最终IR有限模型与CANN闭环；默认FA仍有整数等价规约缺口。算子调优未恢复，本轮没有设备结果，编译耗时未达标。

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

## 默认FA的当前拒绝与未解决项

已用未改动的默认DSL执行一次公共CLI/CANN构建请求，14.24秒后在`notification_58`的`event-parameter-cases`阶段明确拒绝，尚未生成成功ASC/CANN产物，见[状态](../../runs/lease-cover/fa-default/build-result.json)和[具体堆栈](../../runs/lease-cover/fa-default/build.log)。拒绝退出耗时不算成功编译性能。

这次未决条件是嵌套余数中的等价坐标：`p-1`与`((p-1)+1)-1`。已缩成[纯标量复现](../../runs/lease-cover/equality-gap.py)及[记录](../../runs/lease-cover/equality-gap.log)：当相乘的runtime系数为声明可整除的动态商（如s/4），polynomial的分数系数阻止quotient实参重建，retained atom又按原始Expr区分，缺少函数实参的同余标识。于是窗口构造错误地转而请求uniform参数划分。下一步在现有整数/atom表示内统一规范化身份，保留原Expr作定义性与区间代表；不新增另一套solver，不借值等价擦除原始unsigned前提，不添加FA特判或盲目重试。

其他明确边界：

- 同类五slot源码仍在更早的memory frontier拒绝`multiple activation residues`；证明模块支持period5不等于整个五slot源码已支持。
- 条件路径搜索和进展图组合最坏仍会增长，不能声称同步算法整体线性。双native小例仍需约一分钟，必须在结构资格稳定后按阶段与工作量分析；不能称性能问题已解决。
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
