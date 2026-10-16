# 产品文档索引

当前入口如下；算子优化已停止，新重构尚处于方案阶段：

| 文档 | 负责回答 |
|---|---|
| [design.md](design.md) | 当前复杂度来源、真实调用链与新的结构化编译算法 |
| [compilation-invariants.md](compilation-invariants.md) | 不变量的唯一建立者、变换保持/失效、正常与深度验证边界 |
| [dsl-multistage.md](dsl-multistage.md) | 如何表达 runtime、局部/CV 多阶段和嵌套流水 |
| [refactor-progress.md](refactor-progress.md) | 实际支持、明确缺陷及已验证证据 |
| [refactor-plan.md](refactor-plan.md) | 下一部分具体修改、替换职责及退出门槛 |

当前专项：[同步与事件已知问题](sync-known-issues.md)。[Vector设备记录](vector-row-performance.md)
和FA性能页保留历史证据，其中的旧“下一步”不覆盖当前停止优化和编译流程重构安排。
旧 task DSL 和旧FA入口已删除；硬件反例迁入内部测试，不继续增加旧算子优化。

[算子示例](../examples/README.md)与[验证工具](../benchmarks/README.md)分别给源码和
复现入口。设计模板不作为可运行示例；历史证据不自动适用于新 AST 路径。

## FA 实现与对照

- [已实现数学、存储与验证范围](attention-target.md)，[DSL 算子源码](../examples/flash_attention/attention.py)。
- [新DSL与AscendC/ACLNN的4K设备基线](attention-performance.md)。
- [独立 AscendC 的合同和测量](attention-reference.md)。
- [legacy FA 的变体与性能](flash-attention.md)。

## 已有能力的详细合同和证据

以下是相应保存点的详细验证记录。若其“仍不支持”或“下一步”与当前状态不同，
以状态页和当前计划为准；保留原始测量，复现路径已随目录清理更新。

| 范围 | 记录 |
|---|---|
| 同步、effects、事件、ownership | [同步反例](synchronization.md)、[事件分配](event-allocation.md)、[混合核](mixed-ownership.md)、[物理扩展](physical-extensions.md) |
| runtime 与局部流水 | [prefix](runtime-prefix.md)、[周期分解](factored-periods.md)、[归一化](ast-normalization.md)、[Cube](ast-cube.md) |
| 归约和 region 组合 | [K归约](runtime-reduction.md)、[层次边界](hierarchy-boundaries.md)、[父子调用](runtime-hierarchy.md)、[sequence](region-sequence.md)、[父体](runtime-sequence-calls.md)、[共享输入](shared-invocation-inputs.md) |
| legacy 算子 | [算子合同](operators.md)、[GEMM](hierarchical-gemm.md)、[GDN](recurrent-gdn.md) |
| 来源与历史 | [源码依据](source-notes.md)、[首轮验证](validation-stage1.md)、[旧算子验证](validation-operators.md)、[旧保存点](checkpoint-before-refactor.md)、[历史进度](history/refactor-progress-20261015.md) |

参考源码和外层报告只作为研究材料，不成为 wheel 或构建依赖。
