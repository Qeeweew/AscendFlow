# AscendFlow 文档入口

当前同步生成采用物理 cell credit、精确边界与分层派发规则。编译调用链见[源码布局](source-layout.md)，同步语义见[生成规则](generation-rules.md)及[不变量](compilation-invariants.md)。

| 文档 | 职责 |
|---|---|
| [generation-rules.md](generation-rules.md) | 具体规则、适用前提、归纳理由和事实归属 |
| [design.md](design.md) | 必须保持的DSL、循环、硬件和同步语义目标 |
| [compilation-invariants.md](compilation-invariants.md) | 输入前提、构造保证、保持/失效和独立验证的区别 |
| [source-layout.md](source-layout.md) | 真实调用链、代码入口和替换边界 |
| [dsl-multistage.md](dsl-multistage.md) | 用户语法、runtime/constexpr与CV多stage模型 |
| [matmul-target.md](matmul-target.md) | 连续 NN GEMM、FP16/BF16 输入输出、尾块、FIXPIPE 转换与 L2 政策 |
| [gemm-target.md](gemm-target.md) | 连续 NT GEMM、FP16 输入/FP32 输出及输出缓存合同 |
| [attention-target.md](attention-target.md) | FA example的数学、存储与验证合同 |
| [validation.md](validation.md) | 分层验证、工作量、冷编译、CANN及后续设备性能口径 |
| [performance-results.md](performance-results.md) | 当前 GEMM/FA 的已测形状、数值范围、设备性能和证据索引 |
| [deferred-optimizations.md](deferred-optimizations.md) | 已确认的待处理性能问题及后续验收要求 |
| [source-notes.md](source-notes.md) | 参考HEAD与硬件语义来源 |

算法在[examples](../examples/README.md)，基准工具在[benchmarks](../benchmarks/README.md)，语义回归在[tests](../tests/README.md)。设计、计划、测试总数和报告标签都不能代替实际资格。

产品文档按上述职责组织，历史实现由 Git 追溯。源码链接和复现命令以本仓库为边界，
生成数据写入已忽略的 `build/`。测量记录中的本机产物目录只用于溯源，源码、API 合同
和复现入口不依赖这些历史目录。
