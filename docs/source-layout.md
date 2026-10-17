# 源码职责和唯一编译链

2026-10-17。规则及事实归属见 [generation-rules.md](generation-rules.md)，实际资格见 [refactor-progress.md](refactor-progress.md)。

```text
compile / Compilation
  synchronize_structure
    lower_native_structure → collect_native_memory
    bounds → admission → dispatch
    logical_valueflow + storage_lifetimes → dependencies
    generate_notifications
      bind_notifications                 # 覆盖、真实位置及原始 ownership edges
      EventDomains                       # 有序原式定义性、坐标往返、配对
      establish_queue_order              # 逐层 physical-step / phase 准入
      construct_cell_channels            # 物理 lifetime credit 归纳
      HandbackRule                       # 无稳定 cell 的有限/ring 生命周期
    allocate_structured_events           # 消费生命周期，建立 bank/ID/共享新约束
    materialize_notifications → simplify_scalars
  backend.structured → ASC
build_file                               # 同一链路 + CANN，单次进程组硬期限
```

`NotificationProgram` 由生成器私有构造；coverage-only selection 不能传给分配器。
`generated-lifetimes` 是报告来源标签，不能作为外部验证凭据。公共 compile
只接受 source，外部 physical snapshot 不进入正常编译链。

| 文件/范围 | 职责与输出 |
|---|---|
| `frontend/`、`dialects/source.py`、`language.py` | TileLang 风格 AST、typed source、逻辑 Tile 和显式 buffer |
| `passes/lower_pipeline.py`、`lower_native.py`、`native_recipes.py`、`native_addresses.py` | 参数化控制及 native recipe；尚未建立异步完成 |
| `analysis/structured_control.py`、`structured_memory.py`、`native.py` | 原始有序算术、真实地址/footprint/effects、目标 native 合同 |
| `analysis/structured_bounds.py`、`structured_ownership.py`、`structured_admission.py`、`structured_host.py` | bounds、别名、逐 actor ownership、初始化、借用 Tile 及 host 前提 |
| `analysis/logical_valueflow.py`、`storage_word.py`、`storage_lifetimes.py`、`structured_dependencies.py` | 精确 RAW、条件 KILL/UPDATE、相邻存储 lifetime、全部最后 reader |
| `analysis/structured_dispatch.py` | 实际 actor、stage、issue_order、嵌套调用位置 |
| `passes/notification_binding.py`、`analysis/structured_protocols.py` | capture/acquire 覆盖；保留自身 origin，不将新增 coverage 当作 ownership |
| `passes/generate_notifications.py` | 唯一通知生成入口，组合规则并输出完整生命周期 |
| `analysis/event_domains.py`、`event_windows.py`、`event_families.py` | 配对及按需的精确成员、FIRST/LAST、非空投影；稳定 cell 不要求 dense rank |
| `analysis/queue_order.py` | 每层零 step DAG 及正 step 前进；child 仅导出 open ports 和 may-block cuts |
| `analysis/cell_channels.py` | 相同物理 cell 和 role 的非重复 credit；原始值/存储 lifetime 归纳 |
| `analysis/notification_lifetimes.py` | 有限使用或明确 queue handback，连接实际首末端口，无任意路径搜索 |
| `analysis/event_sharing.py`、`structured_resources.py`、`event_contracts.py` | 有限 bank、配置/drain 台账、生成 credit 的 ID 分配与新增共享约束 |
| `passes/materialize_structure.py`、`simplify_scalars.py` | 插入确定动作、共享同位置短路 guard 前缀；不改变 native 或 issue order |
| `backend/structured.py`、`backend/cann.py`、`build.py`、`runtime/process.py` | 打印实际 IR；CANN 离线编译与 300 秒组期限 |
| `tests/models/structured_events.py` | 独立有限队列/二值 flag/quorum/字节版本检查，可对照 pre-sync 快照 |

旧 `event_progress`、`event_paths`、`event_lifetimes`、`progress_*` 和
`structured_event_sharing` 已被删除，测试不保留旧 solver。历史源码通过 Git
追溯，不另建 legacy 或转发别名。纯整数查询仍负责真实前提，不能以删除所有
`prove` 调用代替重构。

算法和政策在 `examples/`；测量及独立 AscendC/ACLNN 在 `benchmarks/`。
运行产物、诊断和材料留外层 `runs/`。产品仓库以外不另建 Git。
