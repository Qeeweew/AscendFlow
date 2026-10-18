# 测试入口与验证边界

使用工作区独立 `.venv`，从仓库根目录运行：

```bash
python -m pip install -e '.[validation,dev]'
python -m pytest -q
```

示例的 CANN/设备检查和两项 msprof 采集见[测试指南](../docs/guide/testing.md)；
实际结果见[示例验证](../docs/reports/examples.md)。以下说明开发回归的职责。

当前产品只有 `compile` / `Compilation` 这一条 source 链路：实际控制与地址、内存 admission、依赖、通知生成及 credit、事件资源，再生成最终物理 IR 与 ASC。导出的物理快照不能作为已证明的 source 编译输入。

旧 planner、scope/query/sequence 求解器专属测试、旧局部同步实现和对应 IR 解释器已删除。测试目录不保留这些实现或兼容入口。删除旧 API 测试不等于当前完整链路获得资格；collection 成功也不表示测试体通过。

旧实验保存在 Git 的 `1231f4f`。Path/搜索树及其内部形状测试已删除，语义反例已迁移到 `sync/test_generation_contracts.py` 等当前测试。规则见[生成合同](../docs/development/generation-rules.md)，不在测试中保留旧 solver。

## 当前职责

| 范围 | 入口 | 主要检查 |
|---|---|---|
| Source 与控制 | `test_frontend.py`、`control/`、`loops/`、`test_compilation.py` | 显式控制、Stage/actor、原始 SSA、source 所有权与不可变产物 |
| DSL 复用与静态尾块 | `test_frontend_helpers.py` | inline 的词法隔离、源位置、调用预算、静态区间、真实访问与非法 host 调用拒绝 |
| Native 与内存 | `analysis/test_native_geometry.py`、`test_native_packets.py`、`test_structured_*.py` | 静态 native 规则、issue packet、真实地址/footprint、bounds、owner、SPR、初始化和数值生产者 |
| 连续矩阵与末块写回 | `test_matmul.py`、`test_dense_gemm.py`、`analysis/test_structured_tiles.py`、`sync/test_terminal_outputs.py` | 补零、FIXPIPE 转换/裁剪、长步长、swizzle 单射及碰撞反例、LAST 写回和累加器复用 |
| 值版本与生命周期 | `analysis/test_logical_valueflow.py`、`test_conditional_accesses.py`、`test_storage_word.py`、`test_storage_lifetimes.py` | 条件 clear/RMW、读版本、PLE 邻接和实际 IR 值保持 |
| 依赖与同步选择 | `analysis/test_dependency_retirement_frontiers.py`、`test_structured_dependencies.py`、`sync/test_notification_selection.py` | 最后真实 reader、捕获/获取端点、实际代次与完整 CV quorum |
| 资源与进展 | `analysis/test_event_*.py`、`test_structured_resource_contracts.py`、`sync/test_generation_contracts.py` | 精确激活、空/短循环、窗口投影、cell credit、消费回传、bank 预留与 child 边界 |
| 最终 IR 与 ASC | `backend/test_structured_emission.py`、`sync/test_structured_pipeline.py` | 后端消费既有事实、实际 Notification/SCF/Address SSA 与生成调用 |
| 数学、ABI 和测量记录 | `test_attention_math.py`、`test_runtime_contract.py`、`test_attention_runner.py`、`test_performance.py`、`test_work_inventory.py` | 数学参考、输入声明、命名 ABI 转发、证据身份、静态工作量与显式分析假设 |

旧硬件反例按职责保留在当前表示中：`sync/test_structured_event_semantics.py` 和 `models/structured_events.py` 包含 native end 非 FIFO、RAW/WAR/WAW、删 Wait、错 ID、过早 Set、跨 body 二元事件复用、Scalar controller 阻塞、双 AIV mode2 quorum、未消费容量与跨队列阻塞环。共享 GM 输入的有限字节检查使用同一 structured 模型。几何与 packet 合同直接使用当前静态对象或实际 SSA；native end 与退休的区别由实际事件模型检查。旧求解器的节点数量、缓存、路径搜索与旧 IR 布局断言不再作为产品合同。

这些覆盖说明测试职责，不代表所有旧场景都已在新编译器获得证明。当前 mixed region、祖先 invocation 和完整 source 集成必须分别满足现有同步契约；未证明的关系应明确拒绝。FA、CANN、设备和性能资格不能由目录清理或小合同通过推导。

清单完整性、失败即停和默认 profiling 流程由 `test_example_checks.py`、
`test_profile_defaults.py` 检查；本地文档链接由 `test_documentation.py` 检查。
动态 valid prefix 仅是 `fixtures/prefix.py` 的未支持/拒绝用例，不计入 runnable examples。

## Helper 与模型的独立性

`support/structured.py` 直接调用生产 frontend、native admission、依赖、通知选择和 EventDomains，只负责构造输入。它没有 pytest fixture，也不提供独立的正确性预期。

`models/structured_domains.py` 复用已收集的 ControlFacts、Expr/LoopResidueSelection 求值、需求域/guard 描述与实际任务集合，直接解释有限控制和 issue order。它不从生产 order、rank、lifetime 或 progress 查询取得预期边。

`models/structured_events.py` 读取实际最终 IR，复用已审计的 native footprint 和 target 规则，自行建立逻辑字节版本以及物理 actor/queue/event 执行关系。它不读取生成器、分配器或 queue_order 的结果；可以独立 pre-sync snapshot 提供预期 native、地址和读版本。`invocations` 表示原始 controller body 连续重入，不隐含跨 actor 的 kernel launch 完成边；mixed 程序的顺序 launch 由 runtime 外部排序。给定参数和有限 body 调用次数的枚举不能代替任意 N 的符号证明。

后端 C++ shim 仅执行 scalar 控制并记录调用与地址，不执行 tensor 数学、CANN 指令或异步设备队列。

终端pipeline降低后，生成计划使用 `check_compiled_events` 对照独立的同步前
native快照，不能把物理发射顺序当作算法的逻辑次序。模型直接执行最终arith/scf、
通知和地址；loop/stage来源只用于对应逻辑实例。大整数边界、短/空/稀疏offset、
嵌套及公共guard内的event剩余条件由实际SSA的有限执行检查。

## 运行纪律

先执行小规模 Python/IR 检查，再考虑 ASC/CANN 离线编译。运行结果绑定实际源码和产物，不把 collection 当作通过。从本仓库根目录、已激活的 Python 环境检查入口：

```bash
python -m pytest -q tests
```

需要仅查看清单时使用 `--collect-only`，它不执行 fixture 和测试体。完整 generated pipeline 用例检查真实 source 的资源及进展契约；手工 IR 模型和后端筛选结果不能替代完整 source 验证。

单个进程/算子构建不超过 300 秒；失败停止后续候选并清理所属子进程。示例设备 runner 位于 `benchmarks/`，独立设备诊断位于 `support/`；运行须遵守工作区卡健康、进程、显存、利用率和显式物理映射要求。
