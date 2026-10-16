# 测试入口与验证边界

使用工作区独立 `.venv`。当前产品只有 `compile` / `Compilation` 这一条 source 链路：实际控制与地址、内存 admission、依赖、通知选择、事件资源、progress，再生成最终物理 IR 与 ASC。导出的物理快照不能作为已证明的 source 编译输入。

旧 planner、scope/query/sequence 求解器专属测试、旧局部同步实现和对应 IR 解释器已删除。测试目录不保留这些实现或兼容入口。删除旧 API 测试不等于当前完整链路获得资格；collection 成功也不表示测试体通过。

## 当前职责

| 范围 | 入口 | 主要检查 |
|---|---|---|
| Source 与控制 | `test_frontend.py`、`control/`、`loops/`、`test_compilation.py` | 显式控制、Stage/actor、原始 SSA、source 所有权与不可变产物 |
| Native 与内存 | `analysis/test_native_geometry.py`、`test_native_packets.py`、`test_structured_*.py` | 静态 native 规则、issue packet、真实地址/footprint、bounds、owner、SPR、初始化和数值生产者 |
| 值版本与生命周期 | `analysis/test_logical_valueflow.py`、`test_conditional_accesses.py`、`test_storage_word.py`、`test_storage_lifetimes.py` | 条件 clear/RMW、读版本、PLE 邻接和实际 IR 值保持 |
| 依赖与同步选择 | `analysis/test_dependency_retirement_frontiers.py`、`test_structured_dependencies.py`、`sync/test_notification_selection.py` | 最后真实 reader、捕获/获取端点、实际代次与完整 CV quorum |
| 资源与进展 | `analysis/test_event_*.py`、`test_progress_*.py`、`test_structured_resources.py`、`test_structured_resource_contracts.py`、`test_structured_event_sharing.py` | 精确激活、空/短循环、窗口投影、ID lease、最后消费、真实阻塞路径和 region 接口 |
| 最终 IR 与 ASC | `backend/test_structured_emission.py`、`sync/test_structured_pipeline.py` | 后端消费既有事实、实际 Notification/SCF/Address SSA 与生成调用 |
| 数学、ABI 和测量记录 | `test_attention_math.py`、`test_runtime_contract.py`、`test_attention_runner.py`、`test_performance.py`、`test_work_inventory.py` | 数学参考、host guards、命名 ABI、证据身份、静态工作量与显式分析假设 |

旧硬件反例按职责保留在当前表示中：`sync/test_structured_event_semantics.py` 和 `models/structured_events.py` 包含 native end 非 FIFO、RAW/WAR/WAW、删 Wait、错 ID、过早 Set、跨 body 二元事件复用、Scalar controller 阻塞、双 AIV mode2 quorum、未消费容量与跨队列阻塞环。共享 GM 输入的有限字节检查使用同一 structured 模型。几何与 packet 合同直接使用当前静态对象或实际 SSA；native end 与退休的区别由实际事件模型检查。旧求解器的节点数量、缓存、路径搜索与旧 IR 布局断言不再作为产品合同。

这些覆盖说明测试职责，不代表所有旧场景都已在新编译器获得证明。当前 mixed region、祖先 invocation 和完整 source 集成必须分别满足现有同步契约；未证明的关系应明确拒绝。FA、CANN、设备和性能资格不能由目录清理或小合同通过推导。

## Helper 与模型的独立性

`support/structured.py` 直接调用生产 frontend、native admission、依赖、通知选择和 EventDomains，只负责构造输入。它没有 pytest fixture，也不提供独立的正确性预期。

`models/structured_domains.py` 复用已收集的 ControlFacts、Expr/LoopResidueSelection 求值、需求域/guard 描述与实际任务集合，直接解释有限控制和 issue order。它不从生产 order、rank、lifetime 或 progress 查询取得预期边。

`models/structured_events.py` 读取实际最终 IR，复用已审计的 native footprint 和 target 规则，自行建立逻辑字节版本以及物理 actor/queue/event 执行关系。它不读取 selector、allocator 或 progress 的证明结果。给定参数和有限 body 调用次数的枚举不能代替任意 N 的符号证明。

后端 C++ shim 仅执行 scalar 控制并记录调用与地址，不执行 tensor 数学、CANN 指令或异步设备队列。

## 运行纪律

先执行小规模 Python/IR 检查，再考虑 ASC/CANN 离线编译。运行结果按实际提交及产物记录在 `docs/refactor-progress.md`，不把 collection 当作通过。可从工作区检查当前入口：

```bash
.venv/bin/python -m pytest --collect-only -q compiler/tests
```

collection 不执行 fixture 和测试体。完整 generated pipeline 用例仍需在资源及进展契约闭合后单独验证；现有手工 IR 模型和后端筛选结果不能外推完整 source 资格。

单个进程/算子构建不超过 300 秒；失败停止后续候选并清理所属子进程。设备 runner 位于 `support/`，其运行须遵守工作区卡健康、进程、显存、利用率和显式物理映射要求。
