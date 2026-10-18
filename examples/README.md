# 可运行示例

算法、分块、缓存和排步政策放在 `examples/`；编译器走同一条 DSL→IR→ASC→CANN
路径。先读[快速上手](../docs/guide/quickstart.md)，所有示例的统一检查命令为：

```bash
python -m benchmarks.check_examples --list
python -m benchmarks.check_examples --output build/examples-cann
# 完成全卡检查与显式映射后，再加 --physical <物理卡号> 检查设备结果。
```

| 检查名称 | 源码 / DSL 入口 | 主要展示内容 |
|---|---|---|
| `scale` | [vector/scale.py](vector/scale.py) · `scale` | GM→UB→Vector→GM 双缓冲 |
| `grouped-scale` | [vector/grouped.py](vector/grouped.py) · `grouped_scale` | runtime BNSD、循环分组、空 owner |
| `rings` | [vector/independent_rings.py](vector/independent_rings.py) · `independent_rings` | 输入/输出独立的 5/7 槽环 |
| `masks` | [vector/masks.py](vector/masks.py) · `masks` | interval XOR 64-lane periodic + runtime origin、masked更新、compare/where；runner按已发布shape运行 |
| `mask-rows` | [vector/mask_rows.py](vector/mask_rows.py) · `mask_rows` | 单 tile 行/列 mask：`cols<hi` 前缀、`cols!=lo` 补集、`rows>=cols-q` 行前缀；三个独立拷贝输出 |
| `rms` | [normalization/rms_norm.py](normalization/rms_norm.py) · `rms_normalize` | 归约、Scalar 最后读取 |
| `softmax` | [normalization/softmax.py](normalization/softmax.py) · `softmax` | 行统计、广播 scratch、Vector 流水 |
| `batched-gemm` | [gemm/batched.py](gemm/batched.py) · `batched_matmul` | 多级 L1/L0/MAD/FIX 环 |
| `reduction-gemm` | [gemm/reduction.py](gemm/reduction.py) · `reduction_gemm` | runtime K 归约 |
| `hierarchical-gemm` | [gemm/hierarchical.py](gemm/hierarchical.py) · `hierarchical_gemm` | 输出块循环嵌套 K 流水 |
| `dense-gemm` | [gemm/dense.py](gemm/dense.py) · `dense_gemm` | 连续 NT、FP16 输入/FP32 输出 |
| `gemm` | [gemm/matmul.py](gemm/matmul.py) · `gemm` | 连续 NN、FP16/BF16、静态尾块 |
| `nested-cv` | [cv/nested.py](cv/nested.py) · `nested_cv` | 嵌套混合流水、共享 GM、双 AIV 共用 body |
| `state` | [state/recurrence.py](state/recurrence.py) · `state_sum` | 初始化、FP32 递推、导出 |
| `grouped-state` | [state/grouped.py](state/grouped.py) · `grouped_state_sum` | 两个 runtime 轴、跨调用复用 |
| `shared-state` | [state/shared_inputs.py](state/shared_inputs.py) · `shared_state_sum` | 多调用共享只读输入 |
| `frame-state` | [state/frame_snapshot.py](state/frame_snapshot.py) · `frame_state_sum` | snapshot、WAR、最后读取 |
| `flash-attention` | [flash_attention/attention.py](flash_attention/attention.py) · `attention_target` | QK/softmax/PV/update 多级 CV 流水 |

[gemm/schedule.py](gemm/schedule.py) 是 GEMM 的 inline L2 政策 helper，随 GEMM 验证。
`independent_rings` 用于测试互质环的组合，不是推荐的性能参数。

单个示例可用 CLI 构建，例如：

```bash
python -m ascendflow --input examples/cv/nested.py --entry nested_cv --shared --timeout 300 --output build/nested-cv
```

设备运行器可按[工具目录](../benchmarks/README.md)查找。普通 GEMM 的尺寸、dtype、
对齐与数值条件见[GEMM 合同](../docs/operators/gemm.md)；dense NT 有单独的
[整块合同](../docs/operators/dense-gemm.md)。FA 的 runtime Sq/Sk、workspace、
数值边界与限制见[FA 合同](../docs/operators/flash-attention.md)。

当前逐项编译/运行证据见[示例验证](../docs/reports/examples.md)。性能仅测默认
Flash Attention 与 GEMM normal 的多 shape ACLNN 对照及 TFLOPS，见[性能报告](../docs/reports/performance.md)。
动态 `valid=` prefix 地址等未支持形式保留在拒绝测试中，不作为 runnable example；
runner 接受某个参数不代表该变体已经设备验证。
