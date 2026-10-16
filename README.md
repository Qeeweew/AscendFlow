# xDSL NPU compiler

面向Ascend 910B / dav-2201，使用xDSL 0.73.0 → 物理执行IR → ASC → CANN 9.1。
用户表达计算、物理存储、缓冲和多阶段排步，编译器负责真实异步访问与资源复用同步。不引入MLIR工程或参考仓库构建依赖。

**编译复杂度尚未达标，正在重新设计。** 默认FA的已提交版本source→ASC冷测p50为53.114s；未提交实验单次为31.022s，最近诊断为37.840s，均不满足≤5s目标。新方案统一循环为prologue/loop/epilogue与边界组合，取消query专项求解路径，并重划包职责。本轮只整理设计与文档，未实施新算法；算子优化暂停。

## 文档与代码入口

- [重构计划](docs/refactor-plan.md)：具体阶段、旧职责退出和验收门槛。
- [循环与同步设计](docs/design.md)、[源码包结构](docs/source-layout.md)、[变换不变量](docs/compilation-invariants.md)。
- [当前状态](docs/refactor-progress.md)：代码身份、支持、测量及未解决问题。
- [DSL](docs/dsl-multistage.md)、[算子示例](examples/README.md)、[FA合同](docs/attention-target.md)。
- [测试](tests/README.md)、[编译/设备工具](benchmarks/README.md)、[验收口径](docs/validation.md)。
- [文档索引](docs/README.md)：唯一权威入口与历史归档位置。

## 现有环境中的示例

从 `/workspace/xdsl-910b` 运行，使用已有 `.venv`，产物写到外层runs。本轮文档整理未运行以下命令：

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

`tiles`是runtime参数，B是tile容量。`debug_verify=True`显式独立检查最终IR；normal应消费变换已建立的事实。旧`cube.task/vector.task`已删除，不恢复兼容。

## 仓库边界

| 路径 | 职责 |
|---|---|
| src/xdsl_npu | DSL、IR、通用分析/变换、ASC和运行支持；目标目录尚待迁移 |
| examples | 算法及显式政策，编译器不得导入它们来选择实现 |
| tests | 语义反例、实际IR破坏、有限模型与复杂度回归 |
| benchmarks | 编译计时、手写AscendC/ACLNN、数值与profiling工具 |
| docs | 当前设计、合同、计划、状态和必要用户参考 |

只有compiler使用产品Git。外层环境、reference、历史报告、runs和缓存不纳入产品提交，不创建根Git。文件CLI/benchmark构建入口使用≤300s的进程组硬期限；直接调用Python `compile()`使用进程内协作预算，不能替代上述隔离构建入口。

设备运行前检查所有卡健康、进程、显存和利用率，选择健康空闲物理卡并显式映射逻辑0；不用/tmp锁，不reset或换卡绕过错误。CPU/IR通过不等于CANN或设备资格，历史证据不等于当前工作树已验证。
