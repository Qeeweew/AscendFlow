# AscendFlow

AscendFlow 是面向 **Ascend 910B / dav-2201** 的 AI 算子编译器。用 Python DSL
描述计算、搬运、buffer 和流水排步，编译器根据真实读写及最后读取生成同步，输出
ASC，再由 CANN 编译。基于 xDSL 0.73.0，使用 CANN 9.1.0，无 MLIR 构建依赖。

## 开始使用

在仓库根目录执行，使用工作区独立环境：

```bash
source ../.venv/bin/activate
python -m pip install -e '.[validation]'
python -m ascendflow --input examples/vector/scale.py --entry scale --output build/quickstart/scale
```

这会生成 source IR、同步前后 IR 和 ASC，**不运行设备**。增加 `--shared` 可生成
CANN 共享库。环境安装、首个 kernel、设备运行和常见错误见[快速上手](docs/guide/quickstart.md)。

## 按任务阅读

| 我想做什么 | 入口 |
|---|---|
| 安装、编译并运行第一个算子 | [快速上手](docs/guide/quickstart.md) |
| 编写 kernel，理解 runtime、buffer 和 stage | [DSL 使用说明](docs/guide/dsl.md) |
| 查找可运行算法及参数 | [示例目录](examples/README.md)、[GEMM](docs/operators/gemm.md)、[Flash Attention](docs/operators/flash-attention.md) |
| 检查所有示例，比较 FA/GEMM 与 ACLNN，计算 TFLOPS | [测试与 profiling](docs/guide/testing.md) |
| 实现 KDA、核对 dot 输入精度并对照 Triton | [decode](docs/operators/kda.md)、[prefill](docs/operators/kda-prefill.md) |
| 修改编译器或同步规则 | [开发入口](docs/development/README.md) |

完整目录见[文档索引](docs/README.md)。安装包和命令名为 `ascendflow`；
`ascendflow` 与 `python -m ascendflow` 调用同一个编译流程。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `src/` | `ascendflow` 包：DSL、xDSL IR、分析、自动同步、ASC/CANN 后端 |
| `examples/` | 算法及分块、缓存、排步政策 |
| `benchmarks/` | 示例编译/运行检查、msprof 工具和可选参考对照 |
| `tests/` | Python/IR 回归、硬件反例、独立有限执行模型 |
| `docs/` | 当前已落实的设计、使用指南、算子合同和支持边界 |
| `build/` | 本机生成产物和原始记录，Git 忽略 |

代码直接位于 `src/`，由项目配置映射为 `ascendflow` 包。参考源码不是构建依赖。
每次文件构建有覆盖 Python→IR→ASC→CANN 的 300 秒硬期限。
构建、数值检查与 profiling 的结果保存在各次工具输出目录；产品文档不收录临时重构记录或性能快照。
