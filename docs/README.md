# 编译器文档

当前正在重新设计循环与自动同步算法，算子优化暂停。**实现、测量与未解决问题只在[状态页](refactor-progress.md)登记；设计和计划不能当支持证据。**

| 文档 | 唯一职责 |
|---|---|
| [design.md](design.md) | 统一prologue/loop/epilogue、边界/回边、完成前沿与资源语义 |
| [source-layout.md](source-layout.md) | 包结构、依赖方向、当前模块去留和迁移纪律 |
| [compilation-invariants.md](compilation-invariants.md) | 每项保证的建立者、保持/失效与normal/debug边界 |
| [refactor-plan.md](refactor-plan.md) | M0–M6实施顺序、具体交付和退出门槛 |
| [refactor-progress.md](refactor-progress.md) | 当前代码身份、支持范围、失败指标及证据 |
| [dsl-multistage.md](dsl-multistage.md) | 用户语法、运行时/constexpr、CV多阶段和组合限制 |
| [attention-target.md](attention-target.md) | 当前FA example的数学、存储和资格合同 |
| [validation.md](validation.md) | 分层测试、复杂度、冷编译、CANN及设备性能验收 |
| [source-notes.md](source-notes.md) | 参考HEAD、必须保留的硬件语义和来源边界 |

用户源码在[examples](../examples/README.md)，工具入口在[benchmarks](../benchmarks/README.md)，现有测试入口在[tests](../tests/README.md)。

重写前36份文档和四个README原文保存在[产品仓库外的归档](../../reports/compiler-refactor/20261016-before-structured-loops/README.md)，有HEAD、状态和SHA256清单。历史性能/阶段报告不再堆在产品docs，也不保留指向旧计划的转发页。历史链接和命令按原文保存，不保证在当前代码上可运行。

维护约定：新能力更新对应合同和状态；新阶段只更新计划完成标记，不再追加一份“当前设计”。测量JSON、ASC、日志和历史报告放外层，Git仅提交产品文件。必要的新文档须有独立长期职责，避免重复状态页。
