# 测试入口与迁移原则

从工作区 `/workspace/xdsl-910b` 使用独立 `.venv`。本目录同时包含当前 source 链路测试、显式历史 oracle 和仍待整理的旧回归；文件存在或 collection 成功不代表用例通过。

当前完整 generated pipeline 仍受 resources 的配对、事件复用与进展契约未闭合阻塞。已有手工最终 IR 模型和后端筛选子集的通过记录，不能据此外推完整 source 链路已通过；最近 M2 迁移只完成 lint、import 和相关测试 collection，未执行测试体。完整算子构建、FA、CANN、设备和性能验证仍处于停止状态。

## 当前 source 测试入口

下表说明测试的目标范围，不表示这些文件已全部通过。当前 source 入口为 `compile` / `Compilation`，沿 `structured_sync` 建立事实并生成最终物理 IR；外部 physical snapshot 仅用于检查，不能授权 source 编译。

| 范围 | 路径/文件 | 检查内容 |
|---|---|---|
| Source 所有权与控制 | `test_compilation.py`、`control/test_structured_physical.py`、`control/test_recursive_native.py` | 单一 source 路径、实际循环/Stage/actor、原始 SSA 与原生操作身份 |
| 地址与内存 admission | `analysis/test_structured_control.py`、`test_structured_addresses.py`、`test_structured_memory.py`、`test_structured_bounds.py`、`test_structured_admission.py` | 实际地址、精确 footprint、bounds、别名、owner、Tile 值与寄存器配置 |
| 依赖与通知选择 | `analysis/test_access_frontier.py`、`test_structured_dependencies.py`、`sync/test_notification_selection.py` | 最后读取、写入与退休义务，实际捕获/获取端点和代次 |
| 资源、共享与进展 | `analysis/test_structured_resources.py`、`test_structured_resource_contracts.py`、`test_structured_event_sharing.py` | 配对、每个参与者的消费、ID lifetime、真实阻塞边；当前仍需闭合 |
| 最终 IR 与 ASC | `backend/test_structured_emission.py` | 实际 Notification/SCF/Address SSA 决定执行，打印消费既有事实；部分用例仍依赖完整资源阶段 |
| Source 集成 | `sync/test_structured_pipeline.py`、`sync/test_structured_event_semantics.py` 中的 `generated` 用例 | 从 source 到最终通知 IR；完整 generated 验证尚未恢复 |
| 独立有限事件模型 | `models/structured_events.py`、`sync/test_structured_event_semantics.py` 中的手工用例 | 实际 IR 的队列、事件、字节版本及 RAW/WAR/WAW 反例 |

`analysis/test_native_effects.py`、`test_native_packets.py` 和 `control/test_recursive_native.py` 含有共享基础层或历史比较用例，不能整文件归入新 source 资格。目录迁移和当前边界见[源码布局](../docs/source-layout.md)与[重构进度](../docs/refactor-progress.md)。

## 历史 oracle 与旧回归

`legacy/` 保存已移出产品的 M2 局部同步实现：`passes/insert_sync.py`、`passes/local_sync.py`、`backend/local.py` 和 `analysis/{contracts,protocols,progress,resources}.py`。其导入显式使用 `tests.legacy`，原产品路径已删除，没有兼容 wrapper，详见 [legacy/README.md](legacy/README.md)。

`analysis/test_local_resources.py`、`test_native_contracts.py`、`sync/test_owned_local_source.py`、`test_structured_local_sync.py` 直接测试此历史实现。它们保留旧契约和反例，不代表当前 source 的 selection、resources、progress 或 emitter 通过验证。`native_packets` 中调用 M2 的用例同样如此。

其他旧 planner、scope/query/sequence 回归与 fixtures 尚未全部迁移。它们通过明确旧入口保留语义检查；旧硬件反例仍需保留，但固定旧节点数、缓存命中或内部模块名的断言不能成为新链路的资格依据。

构建期限测试 `test_compile_watchdog.py` 检查期限、子进程清理和产物身份；`test_attention_math.py` 与 `benchmarks.validation` 提供数学参考。两类检查均不能代替生成代码的数值或设备验证。

## 验证独立性

`models/structured_events.py` 读取实际最终 IR，并使用已承认的 native footprint 与 target 规则解释指令。它自行建立逻辑读写版本和物理 actor/queue 执行关系，不从生产 selector、allocator 或 progress 结果取得预期边。手工用例包括删 Wait、错 ID、过早 Set、跨 body 复用、mode2 quorum 与阻塞环反例。

此模型只检查给定参数和有限 body 调用次数；后端 C++ shim 也只记录控制、调用和地址，不执行 tensor 数学。二者都不构成任意 N 证明、数值验证或设备完成关系证明。

`models/finite_query_native.py` 复用生产 native topology，只能独立检查求解器路径；full/compact 共享转移函数的对照主要检查表示，不能与实际 IR 模型的独立性混为一谈。

最小反例矩阵和各层口径见 [validation.md](../docs/validation.md)，变换保持见[不变量](../docs/compilation-invariants.md)。

## 运行纪律

当前可先收集相关入口，collection 不执行测试体：

```bash
.venv/bin/python -m pytest --collect-only -q \
  compiler/tests/sync/test_structured_pipeline.py \
  compiler/tests/sync/test_structured_event_semantics.py \
  compiler/tests/backend/test_structured_emission.py
```

已有通过记录的手工/后端筛选命令如下；该记录早于本次 M2 迁移，迁移后仅做了 collection。此筛选排除了 generated、owned、foreign 和 launcher 用例，不表示被排除的组已通过：

```bash
.venv/bin/python -m pytest -q \
  compiler/tests/sync/test_structured_event_semantics.py \
  compiler/tests/backend/test_structured_emission.py \
  -k 'not generated and not owned and not foreign and not launcher'
```

待资源契约闭合并恢复构建验证后，按变更选择相关小组，再运行必要的集成回归。`pytest compiler/tests` 会同时覆盖当前与历史测试，不能把整体 collection 或历史通过记录当作当前完整验证。修改 IR 的测试各取独立副本；成本输入和冷编译不复用 cache。

单个算子构建 ≤300s；失败停止后续候选并清理所属子进程。设备 runner 在 `tests/support` 中只负责小规模资格，不能作为大算子性能证据；设备运行规则见 [benchmark 入口](../benchmarks/README.md)。

本检查点新增 `analysis/test_event_windows.py`、`test_event_window_counterexamples.py` 与 `test_event_domain_composition.py` 检查精确membership、固定宽度运算及内向外存在投影；`test_event_progress_regions.py` 和 `test_progress_counterexamples.py` 检查实际区域组合与孙循环漏边。最后一轮相关小合同集成30项通过，生成完整循环用例未在本批恢复。它们不替代完整FA/CANN或设备资格。
