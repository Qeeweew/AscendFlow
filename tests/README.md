# 回归测试

从工作区 `/workspace/xdsl-910b` 执行。默认 `pytest compiler/tests` 仍运行全部
回归，不静默跳过旧实现或昂贵检查。修改同步时先选择对应层，稳定后再跑一次全集。
这些命令不运行设备，也不把 Python 回归耗时当成编译性能。

2026-10-15迁移：旧task/tracer公开入口与兼容测试已删除。原有关键机器反例迁到
`fixtures/physical_operators.py`、`physical_state.py`、`cv.py`，不作为公开算子API。
成功shape扫描裁剪为代表组合，RAW/WAR、最后reader、双AIV、event资源与破坏IR
负例保留；旧FA的2048group/整算子策略测试退出，改为小型CV/状态反例。

新增重点：`test_ast_api.py`验证旧入口不存在；`test_runtime_contract.py`验证
minimum/multiple从AST贯通launcher（CPU测试仅替换设备派发行）；
`test_compile_watchdog.py`验证整个构建总期限/子进程清理/产物身份；
`sync/test_symbolic_cost.py`用独立有限路径验证共享拓扑的域隔离与真实结构计数；
`test_cv_protocols.py`与`test_attention_math.py`分别验证机器协议和独立数值参考。
新FA目标失败回归仍代表未支持，不能计作完整FA成功。

| 范围 | 入口 | 检查什么 |
|---|---|---|
| 条件关系算法 | `sync/test_completion.py`、`sync/test_completion_flow.py`、`sync/test_incremental_completion.py` | 条件交集/差集、增量可达、有限 BFS 对照，以及一个frame编译回归 |
| 直接同步生成 | `sync/test_direct_synthesis.py` | 原生图无候选同步；增量图与实际指令正/负关系差分；条件Wait透明传递；早release不扩大到无关工作 |
| region/call 同步生成 | `sync/test_sequence_sync.py`、`sync/test_sequence_invocations.py` | ready/release、多个 reader、跨调用事件复用、保留循环 |
| frame 与共享输入 | `sync/test_frame_regions.py`、`sync/test_shared_invocation_inputs.py` | 真实别名、零次执行、slot 最后读取、分组地址 |
| 最终 IR 验证 | `sync/test_plan_validation.py` | 删除/移动同步、错误时钟、event 冲突、伪造 ownership 后必须拒绝 |
| 独立有限执行模型 | `models/local_sequence.py` | 解释实际命令、逐字节 RAW/WAR/WAW、二值 event 覆盖和死锁 |

```bash
.venv/bin/python -m pytest -q compiler/tests/sync/test_completion.py compiler/tests/sync/test_completion_flow.py compiler/tests/sync/test_incremental_completion.py
.venv/bin/python -m pytest -q compiler/tests/sync --durations=10
.venv/bin/python -m pytest -q compiler/tests --durations=15
```

`sync/conftest.py` 共用四份不可变编译产物。每次 `.plan` 都获得独立可修改 IR，
破坏测试不会污染其他用例。同一多 reader 程序的多个运行时边界只生成一次同步。
编译规模测试和修改源码的反例仍各自编译，不使用 fixture 掩盖编译开销。

纯关系测试之外，`test_completion_flow.py` 还有一个小frame编译回归，用来
禁止普通编译调用旧 `Graph.path`。直接生成回归另外检查初始图只包含native
指令、选中协议才有完成转移，并比较未要求成立的关系，防止差分只得到两组
全True答案。模板、环宽、条件和层次深度的结构成本仍需持续验收，不能用
“没有调用旧路径查询”代替复杂度证据。

有限模型不导入生产版的完成关系或访问摘要；它对小规模实际指令建图。
这能发现不安全的生成计划，但不是任意动态长度的形式化证明。
大常量界测试另检查1024/1048576下模板、端点、前沿和真实query-domain/条件
事实数量；旧proof_domain_states=0的比较已移除。新计数覆盖叶层最终分析，
父层仍需补全；各域peak之和及增量replay的新计数不能直接与旧统计同比。

编译耗时用 `benchmarks.compile` 的五个独立进程测量，import 和 CANN 单列。
ASC、离线编译和设备性能的入口见 [benchmarks](../benchmarks/README.md)。
