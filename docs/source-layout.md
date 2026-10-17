# 源码职责与重构入口

2026-10-17。此页描述**当前代码**及接手顺序，不把目录拆分当作架构已完成。当前快照与资格见[状态页](refactor-progress.md)，新的生成方向见[计划](refactor-plan.md)。

## 1. 先看实际调用链

```text
compilation.Compilation._build_structured
  passes.structured_sync.synchronize_structure
    lower_native_structure
    memory / bounds / admission / dispatch
    build_structured_dependencies
    select_notifications                  # data coverage only
    allocate_structured_events
      EventDomains                        # 端点/配对/窗口
      prove_progress                      # 选择通知后的阻塞图分析
      establish_lease                     # 消费到复用
      _EventBanks._assign                 # 候选ID和sharing路径证明
    materialize_notifications
    simplify_scalars
  backend.structured._emit_owned
```

这里的后置分析在normal中执行，并非只有debug模式。`owned-construction`报告标签不能改变这个事实。

## 2. 按职责审查，而非重新堆叠pass

| 代码 | 当前职责 | 接手动作 |
|---|---|---|
| `language.py`、`frontend/`、`dialects/` | TileLang风格DSL、结构控制和真实SSA | 保持公共DSL链路；不恢复cube.task/vector.task |
| `passes/lower_native.py`、`native_recipes.py`、`native_addresses.py` | native展开、effects、真实地址与保留循环 | 保留硬件合同；不因重构改变算法或排步 |
| `analysis/structured_memory.py`、`structured_access.py`、`structured_bounds.py`、`structured_admission.py`、`structured_ownership.py` | 实际访问、范围、初始化、alias/owner前提 | 明确事实所有权与失效条件，避免重复解析 |
| `analysis/logical_valueflow.py` | 精确RAW、KILL/UPDATE、代次 | 不与物理复用重新混合 |
| `analysis/storage_packets.py`、`storage_word.py`、`storage_lifetimes.py` | private资格、PLE邻接、WAR/WAW及最后reader | 审查可保留合同；不要重绑值版本 |
| `passes/select_notifications.py`、`analysis/structured_protocols.py` | 依赖覆盖、capture/acquire、逻辑通知 | **主要重构入口：尚非完整协议生成保证** |
| `analysis/event_domains.py`、`event_families.py`、`event_windows.py` | 配对、域、membership/首末及存在投影 | 区分必要输入前提与生成后重新求出的保证 |
| `analysis/event_lifetimes.py`、`event_paths.py`、`structured_event_sharing.py` | 生命周期及共享ID的路径覆盖 | 找出应由生成/组合建立的关系；不能直接关掉 |
| `analysis/event_progress.py`、`progress_regions.py`、`progress_endpoints.py`、`progress_ranks.py` | 后置区域阻塞关系、绑定与进展判定 | 分析职责缺口；不默认继续优化其图算法 |
| `analysis/progress_activations.py`、`progress_graphs.py`、`progress_connectivity.py` | 未完成实验：Path、三值条件、hidden图、共享搜索树 | 可替换/删除；不作为未来架构要求，不复制到legacy目录 |
| `analysis/integer_facts.py`、`progress_conditions.py`、`guard_cover.py` | 数学前提、原式定义性、局部覆盖 | 只保留新生成规则实际需要的能力，不维护无人使用的solver API |
| `analysis/structured_resources.py` | 编排后置分析、bank台账、候选共享、Set/Wait动作 | 分离已有保证的消费与分配新增义务 |
| `passes/materialize_structure.py`、`simplify_scalars.py`、`backend/` | 最终IR、合法纯标量化简、ASC/CANN | 消费完整计划，不新增隐藏同步或第二解释路径 |
| `build.py`、`runtime/process.py` | 单次构建与进程组期限 | 保持300秒硬上限及失败清理 |

公共编译只接收DSL/source；物理快照是检查产物。分析只引用本次拥有的IR及必要不可变结果，不搭建可伪造proof标记或任意可变IR的通用缓存框架。

## 3. 代码整理与迁移纪律

- 停止前源码/测试已保存在`1231f4f`。它是未完成实验快照，当前无须恢复其中任何已被替代的内部接口。
- 不机械搬目录或加转发模块来表示重构完成。先明确输入/输出合同及依赖方向，再按职责替换调用者并删除旧实现。
- 算法、分块与FA变体在`examples/`；手写AscendC/ACLNN和profiling工具在`benchmarks/`；独立有限模型在`tests/models/`。
- 测试保留硬件反例和真实IR语义；旧算法路径数、缓存结构和特定图表示不是兼容性要求。测试内不保存旧solver。
- 原始失败记录、运行数据、临时审查脚本与产物留在外层`runs/`；不进入产品Git。没有根Git。

交接整理仅纠正源码说明，不更改实际调用顺序或关闭检查。接手时先运行`git status --short`和`git log`核对真实状态，保留新增用户修改。
