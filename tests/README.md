# 回归测试

从工作区 `/workspace/xdsl-910b` 执行。默认 `pytest compiler/tests` 仍运行全部
回归，不静默跳过旧实现或昂贵检查。修改同步时先选择对应层，稳定后再跑一次全集。
这些命令不运行设备，也不把 Python 回归耗时当成编译性能。

| 范围 | 入口 | 检查什么 |
|---|---|---|
| 条件关系算法 | `sync/test_completion.py`、`sync/test_incremental_completion.py` | 条件交集/差集、增量可达、与有限 BFS 对照；不编译算子 |
| region/call 同步生成 | `sync/test_sequence_sync.py`、`sync/test_sequence_invocations.py` | ready/release、多个 reader、跨调用事件复用、保留循环 |
| frame 与共享输入 | `sync/test_frame_regions.py`、`sync/test_shared_invocation_inputs.py` | 真实别名、零次执行、slot 最后读取、分组地址 |
| 最终 IR 验证 | `sync/test_plan_validation.py` | 删除/移动同步、错误时钟、event 冲突、伪造 ownership 后必须拒绝 |
| 独立有限执行模型 | `models/local_sequence.py` | 解释实际命令、逐字节 RAW/WAR/WAW、二值 event 覆盖和死锁 |

```bash
.venv/bin/python -m pytest -q compiler/tests/sync/test_completion.py compiler/tests/sync/test_incremental_completion.py
.venv/bin/python -m pytest -q compiler/tests/sync --durations=10
.venv/bin/python -m pytest -q compiler/tests --durations=15
```

`sync/conftest.py` 共用四份不可变编译产物。每次 `.plan` 都获得独立可修改 IR，
破坏测试不会污染其他用例。同一多 reader 程序的多个运行时边界只生成一次同步。
编译规模测试和修改源码的反例仍各自编译，不使用 fixture 掩盖编译开销。

有限模型不导入生产版的完成关系或访问摘要；它对小规模实际指令建图。
这能发现不安全的生成计划，但不是任意动态长度的形式化证明。
大常量界测试另检查 1024/1048576 下模板、端点、前沿和关系查询数量保持一致。

编译耗时用 `benchmarks.compile` 的五个独立进程测量，import 和 CANN 单列。
ASC、离线编译和设备性能的入口见 [benchmarks](../benchmarks/README.md)。
