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
| 端点和接口类型 | `analysis/access_frontier.py`、`structured_regions.py` | 实际端点、FIRST/LAST、读版本、条件写、区域接口；无求解器 |
| 逻辑值流 | `analysis/logical_valueflow.py` | 只建立源码 read→writer/代次、KILL/UPDATE 和区域拼接事实 |
| 坐标合成 | `analysis/region_coordinates.py` | 两阶段共用的 guard、绑定、读族和域构造；不选择 writer 或生成同步 |
| 物理生命周期 | `analysis/storage_packets.py`、`storage_word.py`、`storage_lifetimes.py` | 证明闭合 private 生命周期，按实际排步连接 PLE 相邻覆盖与最后 reader |
| 依赖编排 | `analysis/structured_dependencies.py` | 合并不可变 RAW 与物理 WAR/WAW，交给唯一通知路径；不递归重求值流 |
| 排步绑定 | `analysis/structured_dispatch.py` | 固定 Stage 与 issue order 到实际 native 的映射 |
| 通知选择 | `passes/select_notifications.py`、`analysis/structured_protocols.py` | 完整激活域中的 capture/acquire 与 coverage；不分配 ID |
| 事件域 | `analysis/event_domains.py`、`event_families.py`、`event_windows.py` | 实际配对、整条时钟链的参数分区、精确 membership/首末/存在投影；不分配 ID |
| 有限周期 | `analysis/periodic.py` | 结构周期及数学相位求值；不证明原程序的无符号算术定义性 |
| 模条件 | `analysis/modular.py` | 相关多项式整体的余数及比较mask；只用于证明，不生成运行期表达式 |
| 整数关系事实 | `analysis/integer_facts.py` | 固定输入范围、owned有序前提context及唯一递归查询；共享原式检查按当前前缀遍历，不提供异步完成证明 |
| 整数代数与atom身份 | `indexing.py` | 一个有界多项式实现；函数实参的规范key与当前查询raw代表分开，供普通索引与保留函数的整数关系共用 |
| 局部条件覆盖 | `analysis/guard_cover.py` | 在一项义务内证明条件OR覆盖；有序定义性、整数三分和余数集合，不持有通知或排步 |
| 实际消费路径 | `analysis/event_paths.py` | 组合真实队列/controller接口和路径条件，完整覆盖即停止；生命周期与有限ID共享共用 |
| 事件生命周期 | `analysis/event_lifetimes.py`、`structured_event_sharing.py` | 消费到复用、有限子循环再次调用和有证据的 ID 共享 |
| 区域进展合同 | `analysis/progress_regions.py` | 端口、绑定、queue cut 和带条件关系；不执行证明 |
| 条件事实 | `analysis/progress_conditions.py` | 一次进展分析内的数学合取与矛盾答案；不改变有序执行 guard |
| 激活公式 | `analysis/progress_activations.py` | 共享固定二元 AND/OR、保序 guard 叶子、按需反驳与分支、统一 region 工作预算 |
| 端点与顺序事实 | `analysis/progress_endpoints.py` | 完整端点绑定、父调用身份、实际队列关系与stream步长下界；不组合区域图 |
| 区域连通关系 | `analysis/progress_connectivity.py` | 稀疏的有条件路径组合与内部节点消去；不解释执行引擎或循环时钟 |
| 循环进展证明 | `analysis/progress_ranks.py` | stream rank、独立 SCC 条件分支与有限环排除；消费共享条件事实 |
| 派发进展编排 | `analysis/event_progress.py` | 绑定实际阻塞端口，组合子区域与边界；不读取分配 ID |
| 事件分配 | `analysis/structured_resources.py`、`event_contracts.py` | 连接资源阶段、bank/ID 台账、配置预留与最终 drain |
| 物化 | `passes/materialize_structure.py`、`analysis/structured_schedule.py` | 将已确定指令写成真实 SCF 条件、event SSA 和 NotificationOp |
| 标量化简 | `passes/simplify_scalars.py` | 对最终 IR 的纯整数/index arith 根应用 xDSL 通用规则；不移动或重写控制与异步操作 |
| ASC | `backend/structured.py`、`structured_native.py`、`native.py` | 读取最终 IR；不得另用动作表重建执行，不插隐藏同步 |
| CANN 与构建 | `backend/cann.py`、`build.py`、`runtime/process.py` | 一个算子一次进程组期限、离线工具调用、成功产物发布 |

`structured_access.py` 保留每个 operand 的物理访问与条件读，不能把首 clear 的不读分支压成无条件 RMW。`logical_valueflow.py` 只递归一次来绑定数值版本。`storage_lifetimes.py` 消费这些接口和已绑定的 dispatch，不能把读改绑到派发中最近的 writer。

闭合 packet 的资格、单时钟 PLE 邻接与 WAR/WAW 构造分属三个小模块。未证明独立的状态仍保留真实值/外部顺序约束；不是先生成全套源码反依赖再删除。`structured_dependencies.py` 仅编排，旧 `_Builder` 和旧有限 access-frontier solver 已删除。

资源阶段拆开事实、生命周期、进展和分配，共享一次建立的 `EventDomains`。实际端点的条件队列顺序也由它建立，lease和sharing不重新解释stage/词法位置。`event_paths`依赖纯数学的`guard_cover`，反向依赖禁止。进展区分内部 stream rank 与对外 queue/阻塞接口；子层 native 留在本地 witness，完整嵌套功能资格仍见状态页。

进展编排为一次调用持有 `ProgressConditions` 与 `ProgressActivations`；端点、rank 与区域组合共用这些事实。`ProgressEndpoints` 拥有原端点绑定和父调用事实，其接口返回真实条件或保守步长下界；编排器只消费这些关系。数学谓词集合与原始有序 `ActionGuard` 分开，不能用集合排序改写执行短路顺序。连通容器每对端点只存一个共享 AND/OR 根，消去节点时只组合仍活跃的前驱与后继，随后删除该节点及入出边。旧 DNF 条件列表、笛卡尔积、组合回调及旧计数已直接删除；公式的实际 guard 叶子随两端一起重绑定，exact/projection 证据独立保留。原 dense 矩阵、`_Composer` 中的端点/rank 实现和旧类型出口已经删除，测试直接使用各职责模块，不保留转发 alias。

同流水fence只保留选择阶段建立的capture覆盖及端点准入，不进入binary event的case/window/lease构造。分配器按它的原consumer guard发射；该职责拆分不增加或移除实际需要执行的fence。

## 必须保持的依赖方向

```text
compilation → frontend / passes / backend
frontend    → dialects / diagnostics / target scalars
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
