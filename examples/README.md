# 算子示例

算法、数学主体与固定排步政策放在这里；通用编译器不根据示例名选择实现。运行/计时/profiling放在[benchmarks](../benchmarks/README.md)，硬件反例放tests。

下表列出 AST 源码入口；每次构建和设备结果按[验证口径](../docs/validation.md)记录。

| 示例 | 表达的结构 | 工具入口 |
|---|---|---|
| [vector/scale.py](vector/scale.py) | 双缓冲GM→UB→Vector→GM | benchmarks.vector.run |
| [vector/grouped.py](vector/grouped.py) | runtime BNSD商余、cyclic分组和空owner | 同上，--example grouped |
| [vector/prefix.py](vector/prefix.py) | 两 runtime 长度及对齐 prefix（native lowering 尚拒绝） | benchmarks.vector.run_prefix |
| [vector/independent_rings.py](vector/independent_rings.py) | 独立输入/输出ring | benchmarks.vector.run_rings |
| [normalization/rms_norm.py](normalization/rms_norm.py) | 归约、Scalar最后读取 | benchmarks.normalization.run |
| [normalization/softmax.py](normalization/softmax.py) | 行统计、广播scratch、Vector流水 | 同上，--operator softmax |
| [gemm/batched.py](gemm/batched.py) | L1/L0/MAD/FIX多缓冲 | benchmarks.gemm.run |
| [gemm/reduction.py](gemm/reduction.py) | runtime K归约 | benchmarks.gemm.run_reduction |
| [gemm/hierarchical.py](gemm/hierarchical.py) | 输出块循环嵌套K流水 | benchmarks.gemm.run_hierarchical |
| [gemm/dense.py](gemm/dense.py) | 连续 NT 矩阵、配对列片与输出缓存政策 | benchmarks.gemm.run_dense |
| [gemm/matmul.py](gemm/matmul.py) | 连续 C=A@B、FP16/BF16 输入输出与任意正尺寸尾块 | benchmarks.gemm.run_matmul |
| [cv/nested.py](cv/nested.py) | 双缓冲 GM handoff、嵌套 Cube 与共享双 AIV body | 通用 CLI，entry=nested_cv |
| [state/recurrence.py](state/recurrence.py) | 初始化、递推、导出 | benchmarks.state.run |
| [state/grouped.py](state/grouped.py) | 两runtime轴、跨调用复用 | benchmarks.state.run_grouped |
| [state/shared_inputs.py](state/shared_inputs.py) | 共享只读输入 | 同上，--shared-inputs |
| [state/frame_snapshot.py](state/frame_snapshot.py) | snapshot、WAR、最后读取 | 同上，--frame-snapshot |
| [flash_attention/attention.py](flash_attention/attention.py) | 输出Q块、KV及原生子流水，CV多阶段 | 通用CLI及benchmarks.attention.run |

从本仓库根目录、已激活的 Python 环境运行：

```bash
python -m examples.vector.scale --output build/quickstart/vector
```

FA只有一个正常数学主体`attention_target`。默认dense f16 B1/Hq8/Hkv8/D128及BQ256/BK1024/CK128/VR8的数学、workspace和验证要求见[FA合同](../docs/attention-target.md)。Sq/Sk是runtime；不把外层Q块当成编译器专属控制类型。

runner参数存在不等于变体已验证。causal、tail、空序列、varlen、paged/sparse、persistent和continuous多时钟状态分别需要明确合同。透明 DSL helper 和静态尾块区间的现有语法见 [dsl-multistage.md](../docs/dsl-multistage.md)；设计中的其他数学块/宏不构成已实现 API，不在 examples 堆预期失败的方案文件或多份候选算子。
