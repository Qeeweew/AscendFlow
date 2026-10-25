# 文档索引

初次使用从[快速上手](guide/quickstart.md)开始。以下文档以当前代码为准；
这里只维护已落实的设计、使用指南、算子合同和支持边界；测量结果在工具输出目录
记录日期、配置与产物身份。

## 使用指南

| 文档 | 内容 |
|---|---|
| [快速上手](guide/quickstart.md) | 环境、首个 kernel、CLI/Python API、产物、运行与排错 |
| [DSL](guide/dsl.md) | runtime/constexpr、执行域、物理存储、流水、多级 CV 和支持边界 |
| [设备核数与 workspace](guide/workspace.md) | CANN 核数绑定、512 B 对齐的内部 GM、容量、依赖分析与 stream 生命周期 |
| [Tensor 调用与 host launch](guide/runtime.md) | 编译器签名、自动参数与动态轴绑定、Torch 当前 stream、重复调用和产物身份 |
| [测试与 profiling](guide/testing.md) | CPU 回归、所有示例编译/运行、FA/GEMM 多 shape 的默认 msprof、ACLNN 对照与 TFLOPS |
| [可运行示例](../examples/README.md) | 按算法查找入口、运行器和检查范围 |

## 算子合同

| 文档 | 内容 |
|---|---|
| [GEMM](operators/gemm.md) | 标准连续 `C=A@B`，FP16/BF16、尾块、数值和缓存政策 |
| [Dense NT GEMM](operators/dense-gemm.md) | `C=A@Bt.T`，FP16 输入、FP32 输出及整块限制 |
| [Flash Attention](operators/flash-attention.md) | dense/causal BNSD forward、online softmax、workspace、输入条件及数值容差 |
| [Sparse MLA](operators/sparse-mla.md) | V4.1 分页滑窗/压缩 shared-KV、sink、INT32 元数据与支持边界 |
| [KDA decode](operators/kda.md) | 三阶段 head 流水、运行时 tokens、S metadata、Triton 对照方法 |
| [KDA prefill](operators/kda-prefill.md) | chunk-64、IEEE FP32 Cube、多阶段 CV、块逆及原始 Triton 对照 |

## 编译器开发

- [开发入口](development/README.md)：编译链、生成规则、不变量、验证方法和硬件来源。

每种信息只在对应位置维护：使用步骤在 guide，算法合同在 operators，编译器保证在
development。支持参数不等于该参数已在设备验证；各次工具结果只对其记录的代码、产物与配置负责。
