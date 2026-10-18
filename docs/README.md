# 文档索引

初次使用从[快速上手](guide/quickstart.md)开始。以下文档以当前代码为准；
历史实验通过 Git 查阅，当前测量单独记录日期、配置与产物身份。

## 使用指南

| 文档 | 内容 |
|---|---|
| [快速上手](guide/quickstart.md) | 环境、首个 kernel、CLI/Python API、产物、运行与排错 |
| [DSL](guide/dsl.md) | runtime/constexpr、执行域、物理存储、流水、多级 CV 和支持边界 |
| [测试与 profiling](guide/testing.md) | CPU 回归、所有示例编译/运行、FA/GEMM 多 shape 的默认 msprof、ACLNN 对照与 TFLOPS |
| [可运行示例](../examples/README.md) | 按算法查找入口、运行器和检查范围 |

## 算子合同

| 文档 | 内容 |
|---|---|
| [GEMM](operators/gemm.md) | 标准连续 `C=A@B`，FP16/BF16、尾块、数值和缓存政策 |
| [Dense NT GEMM](operators/dense-gemm.md) | `C=A@Bt.T`，FP16 输入、FP32 输出及整块限制 |
| [Flash Attention](operators/flash-attention.md) | dense/causal BNSD forward、online softmax、workspace、输入条件及数值容差 |

## 当前结果与编译器开发

- [性能报告](reports/performance.md)：Flash Attention 和 GEMM normal 多 shape 的 ACLNN 时间、加速比与 TFLOPS。
- [Causal Attention](reports/causal-attention.md)：因果模式的实现、数值和 ACLNN 对照范围。
- [示例验证](reports/examples.md)：全部公开示例的编译与设备数值检查。
- [开发入口](development/README.md)：编译链、生成规则、不变量、验证方法和硬件来源。

每种信息只在对应位置维护：使用步骤在 guide，算法合同在 operators，编译器保证在
development，实测结果在 reports。支持参数不等于该参数已在设备验证；以结果页列出的范围为准。
