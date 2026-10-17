# AscendFlow

面向 Ascend 910B / dav-2201 的多阶段流水编译器，基于 xDSL 0.73.0 构建，输出 ASC，由 CANN 9.1 离线编译。
用户表达计算、物理存储、缓冲和多阶段排步，编译器负责真实异步访问与资源复用同步。不引入MLIR工程或参考仓库构建依赖。

自动同步使用[物理 cell credit 与分层派发规则](docs/generation-rules.md)，生成完整生命周期后交给有限 event 分配器。旧 Path/progress 求解器已删除，公共 DSL→IR→ASC 只有一条链路。当前资格、编译耗时与未解决项见[状态页](docs/refactor-progress.md)；规则接通不等于设备或性能验收。

## 文档与代码入口

- [重构计划](docs/refactor-plan.md)：具体阶段、旧职责退出和验收门槛。
- [循环与同步设计](docs/design.md)、[源码包结构](docs/source-layout.md)、[变换不变量](docs/compilation-invariants.md)。
- [当前状态](docs/refactor-progress.md)：代码身份、支持、测量及未解决问题。
- [DSL](docs/dsl-multistage.md)、[算子示例](examples/README.md)、[FA合同](docs/attention-target.md)。
- [测试](tests/README.md)、[编译/设备工具](benchmarks/README.md)、[验收口径](docs/validation.md)。
- [文档索引](docs/README.md)：唯一权威入口与历史归档位置。

## 开发环境与入口

从 `/workspace/xdsl-910b` 使用独立 `.venv`。首次使用或切换包名后，安装当前产品的开发入口：

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e AscendFlow
.venv/bin/ascendflow --help
.venv/bin/python -m ascendflow --help
```

Python 包名、安装项目名和命令名均为 `ascendflow`。xDSL 是底层 Python IR 框架依赖，不是产品名。两个命令入口调用同一个编译流程。

## 示例

从 `/workspace/xdsl-910b` 运行，使用已有 `.venv`，产物写到外层runs。功能产物包含 source、同步前后 IR 和 ASC：

```bash
PYTHONPATH=AscendFlow .venv/bin/python -m examples.vector.scale --output runs/quickstart/vector
```

`PYTHONPATH=AscendFlow` 使示例和 benchmark 模块可导入；AscendFlow 包来自上述开发安装。示例输出 source/同步前后 IR、ASC 与报告，增加 `--shared` 可做 CANN 离线编译，不默认运行设备。

直接编译 Python DSL 文件：

```bash
.venv/bin/ascendflow --input AscendFlow/examples/vector/scale.py --entry scale --parameters '{"B":64}' --output runs/quickstart/vector-cli --timeout 300
```

```python
from ascendflow import compile
from examples.vector.scale import scale

artifact = compile(scale.specialize(B=64))
print(artifact.asc)
```

`tiles`是runtime参数，B是tile容量。`debug_verify=True`执行最终IR的结构检查，不能替代异步正确性证明；普通编译检查生成规则的输入、配对和排步前提，消费既有 credit；不重跑旧路径证明器。旧`cube.task/vector.task`已删除，不恢复兼容。

## 仓库边界

代码直接放在 `src/` 下，没有额外的同名包目录。`pyproject.toml` 将逻辑 Python
包 `ascendflow` 映射到 `src/`，所以仍使用 `ascendflow.analysis` 等明确的导入名。
开发安装直接指向这份源码。

| 路径 | 职责 |
|---|---|
| src | DSL、IR、通用分析/变换、ASC和运行支持 |
| examples | 算法及显式政策，编译器不得导入它们来选择实现 |
| tests | 语义反例、实际IR破坏、有限模型与复杂度回归 |
| benchmarks | 编译计时、手写AscendC/ACLNN、数值与profiling工具 |
| docs | 当前设计、合同、计划、状态和必要用户参考 |

AscendFlow 通过独立产品 Git 管理。外层环境、reference、历史报告、runs和缓存不纳入产品提交，不创建根Git。文件CLI/benchmark构建入口使用≤300s的进程组硬期限；直接调用Python `compile()`使用进程内协作预算，不能替代上述隔离构建入口。

设备运行前检查所有卡健康、进程、显存和利用率，选择健康空闲物理卡并显式映射逻辑0；不用/tmp锁，不reset或换卡绕过错误。CPU/IR通过不等于CANN或设备资格，历史证据不等于当前工作树已验证。
