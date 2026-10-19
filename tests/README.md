# 测试职责与运行入口

从产品仓库根目录、工作区独立 `.venv` 运行全部 CPU/Python/IR 回归：

```bash
python -m benchmarks.check_tests --jobs 32 --output build/tests-parallel
```

每次使用新输出目录，全量按文件并行，不复用历史结果。每个普通文件保留
300 秒进程组期限；watchdog 与 CLI helper 自测独立运行。JSON 汇总记录
并发数、总墙钟时间、各文件结果和全部 JUnit 用例。直接 `python -m pytest -q`
仍是完整串行入口。

默认 collection 包含全部测试，没有隐式跳过算子或较慢的有限执行检查。
2026-10-19 的目录整理保留了全部 915 个测试函数、1945 个参数化用例；
该数量是整理前后覆盖对应的记录，不是后续新增或删除测试的配额。
collection 完整和测试体通过分别验证，二者均不代替 CANN/设备资格。

## 按改动选择测试

| 目录 | 负责的合同 | 局部检查入口 |
|---|---|---|
| `frontend/` | AST/DSL、原始轴、静态断言、inline、源 SSA 与拒绝边界 | `python -m pytest -q tests/frontend` |
| `analysis/` | 整数/SMT、地址与原生几何、初始化、依赖、事件域和资源前提 | `python -m pytest -q tests/analysis` |
| `control/`、`loops/` | actor/Stage、递归控制、代次与坐标、空/短循环 | `python -m pytest -q tests/control tests/loops` |
| `passes/` | native/execution lowering、标量与条件重写、支配及原式合法性 | `python -m pytest -q tests/passes` |
| `sync/` | ready/release、最后 reader、cell credit、quorum、事件消费与跨调用复用 | `python -m pytest -q tests/sync` |
| `backend/` | 最终 SSA、ASC 调用和地址、共享 AIV body、物化对应 | `python -m pytest -q tests/backend` |
| `operators/` | GEMM、KDA/Gram、Vector/mask/reduce 等真实算子的 CPU/IR 合同 | `python -m pytest -q tests/operators` |
| `integration/` | 公共 Compilation 合同、默认/causal FA 完整构建及有限 byte/event 执行 | `python -m pytest -q tests/integration` |
| `runtime/` | 输入与 ABI、薄 launcher、cache alias、构建 watchdog 和进程组 | `python -m pytest -q tests/runtime` |
| `tooling/` | runner/mock、产物身份、基准计时边界、工作量报告、文档与示例清单 | `python -m pytest -q tests/tooling` |

先跑受影响的目录，稳定后运行全量。不要用文件排序后按数量切片分组；
目录应直接说明失败归属。FA 的真实构建/大范围有限执行集中在 `integration/`，
日常修改局部整数规则不必借这些集成用例检查一个表达式。
冷编译耗时仍用独立新进程测量，不能以共享 fixture 的运行时间代表编译性能。

不按测试总数删减边界矩阵。0/1/短循环、FIRST/LAST 重合、分支孔洞、真实 alias、
旧 reader 未退休、错误 event ID、删 Wait、少一个 AIV quorum 和跨调用重入
保护不同的合同，必须保留具体反例。旧 solver 的搜索树、节点形状或兼容入口
不作为当前产品合同，也不在测试中恢复第二套编译实现。

## 共享输入、支撑代码与模型

测试文件是消费者，不从另一个 `test_*.py` 导入 helper、输入或 mock。

| 位置 | 职责与边界 |
|---|---|
| `fixtures/` | DSL/source 输入及构造：nested、shared AIV、conditional MAD、event slot 等；可以调用生产准入，但不提供独立正确性预期 |
| `support/` | 生产链 setup、CPU host shim、runner/artifact mock 和诊断入口；fake `.so` 只供身份/转发检查，不得送设备 |
| `models/` | 有限独立执行与 trace/坐标/依赖预期；说明复用了哪些硬件合同，以及不覆盖哪些正确性层次 |

`support/structured.py` 调用生产 frontend/native lowering、依赖和 EventDomains，
没有 pytest fixture，不是独立 oracle。`fixtures/conditional_mad.py` 同样只是
真实 MAD 输入和准入 setup，不把其返回值当成独立证明。

`models/native_control.py` 与 `source_index.py` 只解释 scalar/control/address；
`terminal_trace.py` 记录实际最终 IR 调用，不证明异步完成。
`dependency_edges.py` 展开量化需求，并独立从有限读写 trace 计算 hazard 预期。
`models/structured_domains.py` 解释有限控制与 issue order，不从生产 progress、
rank 或 lifetime 查询取得预期边。

`models/structured_events.py` 从实际最终 IR、真实字节访问与已审计 native footprint
建立 actor/queue/event 有限执行模型；不消费生成器、分配器或 queue_order 的证明。
`check_compiled_events` 使用独立 pre-sync native 快照对应逻辑实例和读版本，
不能将物理发射顺序当作算法次序。`invocations` 表示 controller body 重入，
不隐含一次 kernel launch 的跨 actor 完成边；有限枚举不证明任意 runtime N。

同一个不可变 Compilation 可以供多个用例使用；每次 `.plan` 返回新的 IR。
破坏测试修改自己的副本，不能共享可变 module。编译本身与冷计时测试不借 fixture
缓存通过，也不向产品安装编译缓存。

## 进程、设备与证据

任何单算子构建保留 Python→IR→ASC→CANN 的 300 秒总硬上限，超时清理所属进程组。
`runtime/test_compile_watchdog.py` 和
`frontend/test_frontend_helpers.py::test_cli_source_imports_local_dsl_helpers_without_pythonpath`
测试自身的进程组/期限，不能再套一层 `bounded_process`；普通上述 pytest 命令不受影响。

CPU C++ shim 只执行 scalar 控制并记录调用、地址或 host 参数，不执行 CANN 指令、
tensor 数学或真实异步队列。公开示例 CANN/设备资格在 `benchmarks/check_examples.py`，
日常 profiling 与物理卡检查见[测试指南](../docs/guide/testing.md)。
实际结果必须对应实际 DSL/IR/ASC/库、编译选项、输入和设备；
CPU mock、collection、历史报告和目录清理均不能替代当前产物的设备资格。
