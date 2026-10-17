# AscendFlow

面向 Ascend 910B / dav-2201 的多阶段流水编译器，基于 xDSL 0.73.0 构建，输出 ASC，由 CANN 9.1 离线编译。
用户表达计算、物理存储、缓冲和多阶段排步，编译器负责真实异步访问与资源复用同步。不引入MLIR工程或参考仓库构建依赖。

自动同步使用[物理 cell credit 与分层派发规则](docs/generation-rules.md)，生成完整生命周期后交给有限 event 分配器。公共 DSL→IR→ASC 只有一条链路；编译、数值与性能验证的口径见[验证说明](docs/validation.md)。

## 文档与代码入口

- [循环与同步设计](docs/design.md)、[源码包结构](docs/source-layout.md)、[变换不变量](docs/compilation-invariants.md)。
- [DSL](docs/dsl-multistage.md)、[算子示例](examples/README.md)、[FA合同](docs/attention-target.md)。
- [测试](tests/README.md)、[编译/设备工具](benchmarks/README.md)、[验收口径](docs/validation.md)。
- [文档索引](docs/README.md)：设计与使用说明入口。

## 开发环境与入口

以下命令均从本仓库根目录运行，使用已激活的独立 Python 环境。安装开发入口：

```bash
python -m pip install -e .
ascendflow --help
python -m ascendflow --help
```

Python 包名、安装项目名和命令名均为 `ascendflow`。xDSL 是底层 Python IR 框架依赖，不是产品名。两个命令入口调用同一个编译流程。

## 示例

功能产物包含 source、同步前后 IR 和 ASC；下面将产物写入已忽略的 `build/`：

```bash
python -m examples.vector.scale --output build/quickstart/vector
```

示例和 benchmark 模块从仓库根目录导入；AscendFlow 包来自上述开发安装。示例输出 source/同步前后 IR、ASC 与报告，增加 `--shared` 可做 CANN 离线编译，不默认运行设备。

直接编译 Python DSL 文件：

```bash
ascendflow --input examples/vector/scale.py --entry scale --parameters '{"B":64}' --output build/quickstart/vector-cli --timeout 300
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
| docs | 当前设计、合同和使用说明 |

生成产物和缓存不纳入 Git，文档中的源码链接均相对于本仓库。文件CLI/benchmark构建入口使用≤300s的进程组硬期限；直接调用Python `compile()`使用进程内协作预算，不能替代上述隔离构建入口。

设备运行前检查所有卡健康、进程、显存和利用率，选择健康空闲物理卡并显式映射逻辑0；不用/tmp锁，不reset或换卡绕过错误。CPU/IR通过不等于CANN或设备资格，历史证据不等于当前工作树已验证。
