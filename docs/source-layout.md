# 源码职责与所有权

当前主线按下表组织。实际功能资格见[状态页](refactor-progress.md)，阶段保证见[编译不变量](compilation-invariants.md)。被替代的模块直接删除；目录与入口收敛不等同于完整功能资格。

## 唯一 source 编译链路

| 层 | 实际模块 | 输入与职责 |
|---|---|---|
| 编排 | `compilation.py`、`passes/structured_sync.py` | 一次调用持有私有 IR，连接阶段、冻结产物；不选择协议 |
| DSL | `language.py`、`frontend/` | AST、类型、runtime/constexpr、`serial/Pipelined/Stage/Cube/Vector` |
| IR | `dialects/source.py`、`physical.py`、`structured.py` | 真实控制、地址 SSA、native 和通知；不包含算子策略 |
| native lowering | `passes/lower_native.py`、`native_recipes.py`、`native_addresses.py` | 保留递归控制、值来源与完整物理地址；不展开 runtime 次数 |
| 控制与访问事实 | `analysis/structured_control.py`、`structured_addresses.py`、`structured_memory.py` | 实际 SSA 坐标、作用域、参与者、原语 effects 与 root/view 几何 |
| 准入 | `analysis/structured_bounds.py`、`structured_admission.py` | 范围、值初始化、alias、配置与原语输入合同 |
| ownership 与 host 合同 | `analysis/structured_ownership.py`、`structured_host.py` | group/sid 写入分区、host 参数和整数范围；不选择同步 |
| 访问分区 | `analysis/structured_access.py` | 按实际物理范围、stride/slot/代次分区；不构造 event |
| region 合成 | `analysis/access_frontier.py`、`structured_regions.py`、`structured_dependencies.py` | sequence/branch/loop 的 RAW/WAR/WAW、首末访问和回边 |
| 排步绑定 | `analysis/structured_dispatch.py` | 固定 Stage 与 issue order 到实际 native 的映射 |
| 通知选择 | `passes/select_notifications.py`、`analysis/structured_protocols.py` | 完整激活域中的 capture/acquire 与 coverage；不分配 ID |
| 事件域 | `analysis/event_domains.py`、`event_families.py`、`event_windows.py` | 实际配对、整条时钟链的参数分区、精确 membership/首末/存在投影；不分配 ID |
| 有限周期 | `analysis/periodic.py` | 结构周期及数学相位求值；不证明原程序的无符号算术定义性 |
| 模条件 | `analysis/modular.py` | 相关多项式整体的余数及比较mask；只用于证明，不生成运行期表达式 |
| 整数关系事实 | `analysis/integer_facts.py` | 固定输入范围及完整有序前提下的规约、范围和比较；同一事实只建立一次，不提供异步完成证明 |
| 局部条件覆盖 | `analysis/guard_cover.py` | 在一项义务内证明条件OR覆盖；有序定义性、整数三分和余数集合，不持有通知或排步 |
| 实际消费路径 | `analysis/event_paths.py` | 组合真实队列/controller接口和路径条件，完整覆盖即停止；生命周期与有限ID共享共用 |
| 事件生命周期 | `analysis/event_lifetimes.py`、`structured_event_sharing.py` | 消费到复用、有限子循环再次调用和有证据的 ID 共享 |
| 派发进展 | `analysis/event_progress.py` | 实际阻塞接口及循环组合；不读取分配 ID |
| 事件分配 | `analysis/structured_resources.py`、`event_contracts.py` | 连接资源阶段、bank/ID 台账、配置预留与最终 drain |
| 物化 | `passes/materialize_structure.py`、`analysis/structured_schedule.py` | 将已确定指令写成真实 SCF 条件、event SSA 和 NotificationOp |
| 标量化简 | `passes/simplify_scalars.py` | 对最终 IR 的纯整数/index arith 根应用 xDSL 通用规则；不移动或重写控制与异步操作 |
| ASC | `backend/structured.py`、`structured_native.py`、`native.py` | 读取最终 IR；不得另用动作表重建执行，不插隐藏同步 |
| CANN 与构建 | `backend/cann.py`、`build.py`、`runtime/process.py` | 一个算子一次进程组期限、离线工具调用、成功产物发布 |

`structured_regions.py` 存共享关系类型；`structured_access.py` 负责范围投影；`structured_dependencies.py` 负责递归合成。这三者不能各自重新解释一次地址、slot 或循环有效域。

资源阶段拆开事实、生命周期、进展和分配，共享一次建立的 `EventDomains`。实际端点的条件队列顺序也由它建立，lease和sharing不重新解释stage/词法位置。`event_paths`依赖纯数学的`guard_cover`，反向依赖禁止。进展区分内部 stream rank 与对外 queue/阻塞接口；子层 native 留在本地 witness，完整嵌套功能资格仍见状态页。

## 必须保持的依赖方向

```text
compilation → frontend / passes / backend
frontend    → dialects / diagnostics
passes      → analysis / dialects / target
analysis    → dialects / target / scalar facts
backend     → 最终 IR + 未改变的 native 合同 / target
runtime     → process / diagnostics
```

循环坐标替换使用共同的 `scalar.substitute`；实际词法归属使用 `ControlFrame`。分析结果不持有回调顶层 planner 的能力。`BoundPlacement` 保留真实端点，`SyncAction` 仅是物化输入，最终执行事实在物理 IR 中。

一份 owned invocation 只建立一次前提。新增通知不改变 native、地址和循环身份，因此可以沿用这些事实；改变访问、排步或激活域时，必须重新建立受影响合同。不能把可伪造的 `verified=True` 当授权。

## 删除边界与工具职责

- `scope_*`、`loop_*`、`sequence_*`、旧 `sync.py/codegen.py/model.py` 及过渡解码器已经删除。
- 不在 `tests/` 保存旧求解器或 emitter，不提供兼容 wrapper；旧 API 专属测试退出，有价值的硬件语义由当前 IR 与独立模型覆盖。未覆盖的资格必须明示。
- 物理 dialect 只注册当前 native、配置、地址、参与者、结构化循环和通知；不注册 QueryChannel、InvocationChannel、ScopeChannel 等旧协议。
- 原语格式化只接受实际 SSA 指针渲染；目标合同只接受 `NativeGeometry`，不再隐式适配旧 Ref。
- 公共编译只接受 DSL/source IR。物理快照是检查产物，不能重新导入充当已证明的输入。
- 基准工具从当前产物提取 ABI、结构 inventory 和显式算法 work。未建立的运行期 native/搬运请求计数必须标明不可得，不能调用旧求解器、填零或冒充设备流量。

算子及分块政策在 `examples/`；数值 oracle、手写 AscendC/ACLNN、profiling 在 `benchmarks/`；独立有限执行语义在 `tests/models/`。参考资料和运行数据留外层，不进入产品包。
