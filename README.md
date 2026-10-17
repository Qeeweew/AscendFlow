# xDSL NPU compiler

面向Ascend 910B / dav-2201，使用xDSL 0.73.0 → 物理执行IR → ASC → CANN 9.1。
用户表达计算、物理存储、缓冲和多阶段排步，编译器负责真实异步访问与资源复用同步。不引入MLIR工程或参考仓库构建依赖。

**当前重构未完成，算法开发已停止并准备交接。** 现有normal链路在通知选择之后仍执行progress、lease与ID共享路径分析，尚未建立完整的、可组合的同步生成保证。接手请先读[重构计划](docs/refactor-plan.md)，不要默认继续优化当前后置求解器。源码/测试快照`1231f4f`是未完成实验；FA、CANN与设备资格只以[当前状态](docs/refactor-progress.md)的对应证据为准。

## 文档与代码入口

- [重构计划](docs/refactor-plan.md)：具体阶段、旧职责退出和验收门槛。
- [循环与同步设计](docs/design.md)、[源码包结构](docs/source-layout.md)、[变换不变量](docs/compilation-invariants.md)。
- [当前状态](docs/refactor-progress.md)：代码身份、支持、测量及未解决问题。
- [DSL](docs/dsl-multistage.md)、[算子示例](examples/README.md)、[FA合同](docs/attention-target.md)。
- [测试](tests/README.md)、[编译/设备工具](benchmarks/README.md)、[验收口径](docs/validation.md)。
- [文档索引](docs/README.md)：唯一权威入口与历史归档位置。

## 现有环境中的示例

从 `/workspace/xdsl-910b` 运行，使用已有 `.venv`，产物写到外层runs。功能产物包含 source、同步前后 IR 和 ASC：

```bash
PYTHONPATH=compiler .venv/bin/python -m examples.vector.scale --output runs/quickstart/vector
```

`PYTHONPATH=compiler`使示例和benchmark模块可导入；产品包使用现有环境。示例输出source/同步前后IR、ASC与报告，增加`--shared`可做CANN离线编译，不默认运行设备。

```python
from xdsl_npu import compile
from examples.vector.scale import scale

artifact = compile(scale.specialize(B=64))
print(artifact.asc)
```

`tiles`是runtime参数，B是tile容量。`debug_verify=True`执行最终IR的结构检查，不能替代异步正确性证明；普通编译当前仍包含通知选择后的协议分析。旧`cube.task/vector.task`已删除，不恢复兼容。

## 仓库边界

| 路径 | 职责 |
|---|---|
| src/xdsl_npu | DSL、IR、通用分析/变换、ASC和运行支持 |
| examples | 算法及显式政策，编译器不得导入它们来选择实现 |
| tests | 语义反例、实际IR破坏、有限模型与复杂度回归 |
| benchmarks | 编译计时、手写AscendC/ACLNN、数值与profiling工具 |
| docs | 当前设计、合同、计划、状态和必要用户参考 |

只有compiler使用产品Git。外层环境、reference、历史报告、runs和缓存不纳入产品提交，不创建根Git。文件CLI/benchmark构建入口使用≤300s的进程组硬期限；直接调用Python `compile()`使用进程内协作预算，不能替代上述隔离构建入口。

设备运行前检查所有卡健康、进程、显存和利用率，选择健康空闲物理卡并显式映射逻辑0；不用/tmp锁，不reset或换卡绕过错误。CPU/IR通过不等于CANN或设备资格，历史证据不等于当前工作树已验证。
