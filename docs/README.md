# AscendFlow 文档入口

2026-10-17。当前同步生成采用物理 cell credit、精确边界与分层派发规则。先读[实际状态](refactor-progress.md)、[生成规则](generation-rules.md)及[不变量](compilation-invariants.md)，区分实现、编译资格与设备证据。

| 文档 | 职责 |
|---|---|
| [refactor-progress.md](refactor-progress.md) | Git身份、停止点、失败证据及资格边界 |
| [refactor-plan.md](refactor-plan.md) | 生成职责审计、规则/组合重构及旧职责退出顺序 |
| [generation-rules.md](generation-rules.md) | 具体规则、适用前提、归纳理由和事实归属 |
| [design.md](design.md) | 必须保持的DSL、循环、硬件和同步语义目标 |
| [compilation-invariants.md](compilation-invariants.md) | 输入前提、构造保证、保持/失效和独立验证的区别 |
| [source-layout.md](source-layout.md) | 真实调用链、代码入口和替换边界 |
| [dsl-multistage.md](dsl-multistage.md) | 用户语法、runtime/constexpr与CV多stage模型 |
| [attention-target.md](attention-target.md) | 当前FA example的数学与存储合同；资格以状态页为准 |
| [validation.md](validation.md) | 分层验证、工作量、冷编译、CANN及后续设备性能口径 |
| [source-notes.md](source-notes.md) | 参考HEAD与硬件语义来源 |

算法在[examples](../examples/README.md)，基准工具在[benchmarks](../benchmarks/README.md)，语义回归在[tests](../tests/README.md)。设计、计划、测试总数和报告标签都不能代替实际资格。

产品文档只维护上述职责，历史实现由Git追溯，不另建legacy说明或复制旧solver。更早36份文档的外层归档见[历史归档](../../reports/compiler-refactor/20261016-before-structured-loops/README.md)，内容只供溯源，不能覆盖本次交接。运行数据、ASC、日志与临时审查留外层runs，产品Git只包含源码、测试、示例和当前文档。
