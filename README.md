# xDSL NPU compiler

面向 Ascend 910B / dav-2201 的 Python DSL 与自动同步编译器。
使用 xDSL 0.73.0 → 物理执行 IR → ASC → CANN 9.1，不依赖 MLIR 工程或参考仓库。
用户表达计算、物理存储和多阶段排步，编译器负责异步访问及缓冲复用的同步。

公开 AST DSL 已贯通默认 dense f16 Flash Attention 的 CANN 和设备验证，但编译耗时
和相对 ACLNN 性能仍未达标；其他变体的资格见状态页。**算子优化已停止，当前转为
编译流程重构方案：控制静态分析规模、整理变换不变量、消除重复证明。新流程尚未实现。**

旧 `cube.task/vector.task`、公开 physical builder、Python tracer 与旧 FA 入口
已删除，不保留兼容层。内部 Task IR 和硬件反例继续用于编译器验证。
新 DSL 支持 `n: T.int64(minimum=1, multiple_of=B)` 输入合同；文件构建入口统一
覆盖输入构造、Python、ASC 和 CANN 的300秒总期限。

## 从这里开始

- [示例目录](examples/README.md)：按算子查找当前 AST 源码和入口。
- [DSL 语法](docs/dsl-multistage.md)：运行时循环、Stage、AIC/AIV 和多级流水。
- [设计](docs/design.md)：当前复杂度来源、实际调用链与新编译算法。
- [编译不变量](docs/compilation-invariants.md)：每个变换保持什么、何时失效、哪些证明可复用。
- [当前状态](docs/refactor-progress.md)：实际支持、已知问题和验证结果。
- [重构计划](docs/refactor-plan.md)：分阶段替换职责、复杂度和性能验收、删除旧路径的条件。
- [性能工具](benchmarks/README.md)：AscendC/ACLNN、profiling、硬件指标和编译耗时。
- [测试分层](tests/README.md)：关系算法、同步生成、最终 IR 破坏检查与有限执行模型。
- [文档索引](docs/README.md)：详细合同与历史证据。

## 安装与一个可运行例子

以下命令从 `/workspace/xdsl-910b` 执行，所有产物写入外层 `runs/`。
`PYTHONPATH=compiler` 只让仓库中的示例和性能工具可被导入，不是安装包依赖。

```bash
.venv/bin/pip install --no-build-isolation -e compiler
PYTHONPATH=compiler .venv/bin/python -m examples.vector.scale --output runs/quickstart/vector
```

输出包含源码 IR、同步前后物理 IR、ASC 和验证报告。增加 `--shared` 可做 CANN
离线编译并生成 host launch 库。上述命令不运行设备。

```python
from xdsl_npu import compile
from examples.vector.scale import scale

artifact = compile(scale.specialize(B=64))
print(artifact.asc)
```

`tiles` 是运行时参数；该例只特化 tile 容量 B，不按输入长度构造 Python 任务。
完整 DSL 源码见 [scale.py](examples/vector/scale.py)。

```bash
.venv/bin/python -m pytest -q compiler/tests
.venv/bin/ruff check compiler
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/scale.py --entry scale --trials 5 --output runs/quickstart/cold
```

## 仓库边界

| 目录 | 职责 |
|---|---|
| `src/xdsl_npu/` | DSL、xDSL IR、同步、代码生成及通用运行工具 |
| `examples/` | 当前可编译的 AST 算子源码和少量使用入口 |
| `tests/` | 回归、同步反例、有限执行模型与设计目标的拒绝测试 |
| `benchmarks/` | 设备验证、独立数值参考、AscendC/ACLNN 对照、编译计时和 profiling |
| `docs/` | 设计、当前状态、下一步计划和带边界的验证记录 |

核心包不导入 examples、benchmarks 或 tests。测试可验证算子示例，并共用无设备
副作用的数值参考；不通过导入运行脚本取得 oracle。`docs/design/` 中的历史源码规格
不作为当前支持证据；实际 FA 源码在 `examples/flash_attention/`，资格以状态页为准。

设备运行前检查全部卡健康、进程、显存和利用率，选择健康空闲物理卡；每进程
显式映射到逻辑 0。不会 reset、换卡绕过错误或使用 `/tmp` 锁。计时和支持范围
以生成代码及设备证据为准；有限模型检查不等于任意规模证明。

只有本目录使用产品 Git。外层环境、reference、缓存和运行数据不纳入产品提交。
