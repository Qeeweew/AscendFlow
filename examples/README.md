# 算子示例

这里放 DSL 算子源码。设备测试、数值参考、AscendC/ACLNN 对照、profiling 和
分析脚本在 [benchmarks](../benchmarks/README.md)，同步反例在 `tests/fixtures/`。

## 当前 AST 示例

| 目录/源码 | 展示能力 | 设备验证入口（模块名） |
|---|---|---|
| [vector/scale.py](vector/scale.py) | GM→UB→Vector→GM 双缓冲 | `benchmarks.vector.run` |
| [vector/prefix.py](vector/prefix.py) | 两长度、对齐有效前缀 | `benchmarks.vector.run_prefix` |
| [vector/independent_rings.py](vector/independent_rings.py) | 独立输入/输出槽位周期 | `benchmarks.vector.run_rings` |
| [normalization/rms_norm.py](normalization/rms_norm.py) | 归约、Scalar 最后读取 | `benchmarks.normalization.run` |
| [normalization/softmax.py](normalization/softmax.py) | packed行统计、显式广播scratch、Vector三阶段 | 同上，加 `--operator softmax` |
| [gemm/batched.py](gemm/batched.py) | L1/L0/MAD/FIX 多缓冲 | `benchmarks.gemm.run` |
| [gemm/reduction.py](gemm/reduction.py) | runtime K 归约 | `benchmarks.gemm.run_reduction` |
| [gemm/hierarchical.py](gemm/hierarchical.py) | 串行 output × K pipeline | `benchmarks.gemm.run_hierarchical` |
| [state/recurrence.py](state/recurrence.py) | 初始化、递推、导出 | `benchmarks.state.run` |
| [state/grouped.py](state/grouped.py) | 两 runtime 轴及跨调用复用 | `benchmarks.state.run_grouped` |
| [state/shared_inputs.py](state/shared_inputs.py) | 共享只读 GM | 同上，加 `--shared-inputs` |
| [state/frame_snapshot.py](state/frame_snapshot.py) | 显式 GM snapshot、WAR、最后读取 | 同上，加 `--frame-snapshot` |

从工作区根目录执行，例如：

```bash
PYTHONPATH=compiler .venv/bin/python -m examples.vector.scale --output runs/quickstart/vector
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.run_grouped --frame-snapshot --output runs/quickstart/frame
```

第二条生成同步前后 IR 和 ASC，未指定 `--physical` 不运行设备；加 `--build`
可编译生成代码与独立参考。当前支持与限制见[状态页](../docs/refactor-progress.md)。

## Flash Attention 接入状态

旧 task/tracer examples 与旧 FA 已删除，未保留兼容入口。完整新 DSL FA 尚未
接通，当前没有可运行 FA example；[目标源码与缺口](../docs/attention-target.md)
单独放在设计文档中，不能作为已支持算子。

原有关键机器反例保留在 `tests/fixtures/physical_operators.py`、
`physical_state.py` 和 `cv.py`，只作内部测试输入，不作为用户编程模型或性能入口。
其他目录只展示当前实际可编译的 AST 程序。
