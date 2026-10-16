# 当前重构状态

**单一路径与进展职责拆分已落地，整个重构目标尚未完成。** 被替代的求解器、兼容入口和旧类型出口直接删除。默认 FA 最近一次完整构建仍为300秒超时；本轮有界诊断能完成第一个子区域，但尚无完整 FA、CANN 或设备资格。不恢复算子调优、打包或设备工作。

合同见[编译不变量](compilation-invariants.md)，职责见[源码组织](source-layout.md)，剩余工作见[重构计划](refactor-plan.md)。历史实现由产品 Git 追溯，运行数据留在外层 `runs/`。

## 当前实现

- 唯一入口接收 DSL/source，经 native、访问与准入、逻辑值流、实际 dispatch 存储生命周期、通知、事件资源、最终物理 IR、ASC 到 CANN。物理快照不作为已证明输入重新导入。
- `logical_valueflow` 保留精确 RAW、条件 KILL/UPDATE 与代次；`storage_packets/word/lifetimes` 只对已证明闭合的 private 生命周期按实际派发构造 PLE 相邻 WAR/WAW。所有关系共用 consumer j/source j-distance，边界使用 membership guard，运行期长度不展开。
- `progress_regions` 拥有不可变端口和区域合同，`progress_conditions` 拥有一次调用的数学合取与矛盾答案，`progress_endpoints` 拥有完整端点绑定和实际顺序关系，`progress_connectivity` 组合实际稀疏关系，`progress_ranks` 建立局部 stream/SCC 进展；`event_progress` 组合这些事实。
- 编排器中的旧端点方法、rank、条件判定和 dense 消元已删除。原类型出口没有 alias，测试直接消费新职责模块。被消去的节点及其入出边实际删除，不再参与后续组合。
- 数学集合不替代原始有序 guard；未知条件不视为不成立，保守接口不提升为 exact。预算保持原值，没有 barrier、额外搬运或 FA 特判。
- 向量 prefix 基准迁至公共 `build_file` 和当前产物读取，统一单次300秒期限；不再自行拼接 compile/write/offline_compile。静态无调用的 `indexing.constant_ratio` 已删除。
- 测试只保留一个通知准备入口 `selected`，调用者全部迁移，重复 `selection` 直接删除。准备函数明确不执行完整memory admission或可执行编译，不能将其结果算作该层资格。
- 算法仍在 examples；AIC:AIV=1:2 的共享 AIV body/runtime sid 模型未改变。

## 本轮证据

| 检查 | 结果与边界 |
|---|---|
| 条件事实、稀疏连通、区域与队列接口等6个文件 | 42 passed，3.93秒；含独立小图路径枚举和真实循环反例 |
| 条件进展与父接口2个文件 | 11 passed，2.48秒；rank 迁移及双 AIV 贡献合同 |
| 公共编译、Vector/Cube流水和事件资源3个文件 | 43 passed，31.77秒；真实最终IR、独立有限模型、消费与复用、产物所有权 |
| 准备函数迁移后的事件共享与域组合 | 11 passed，1.93秒；删除的提前闭合检查已由通知选择入口执行 |
| 静态审查 | rank 主体 AST 对照保持；exact、guard 顺序、两轮消元与开放端点审查通过 |
| 格式与引用 | Ruff、diff 检查通过；旧类型消费者已迁移，无转发包装 |

上述四个不重叠测试组共107项，不是完整 suite。端点方法迁移另做21方法AST等价审查；26项受影响接线回归通过（3.00秒，与上述组有重叠），见 [endpoints-integration.log](../../runs/progress-regions/endpoints-integration.log)。独立枚举只覆盖指定小图和规模，不能称任意规模形式化证明。prefix 仅做 Python/结构检查，未运行设备。资源集成日志见 [integration-final.log](../../runs/progress-regions/integration-final.log)。

### 默认 FA 的一次有界诊断

没有重复完整300秒构建。同步计数脚本受55秒协作期限和60秒外层期限约束，在55.10秒抛出期限诊断并退出，未使用异步栈采样。原始记录：[probe-owned.log](../../runs/progress-regions/probe-owned.log)。

- source→EventDomains 22.42秒；44个native模板、726条依赖义务、221条通知、142个参数case、1个packet分区，与拆分前一致。
- 第一个子区域 `$loop2` 完整组合28.52秒。其中有限端点关系13.99秒，混合区域图11.07秒，路径初始化0.80秒，首次稀疏消元1.25秒。
- 区域组合完成后进入 `$loop3`，期限在其端点关系构造阶段到达。累计408个不同数学条件集合、4763次底层比较证明、771次已有矛盾答案复用，784次路径插入/组合尝试。

这些数字定位剩余开销，不是冷编译基准或完整 FA 提速结论。诊断在端点方法迁至独立owner之前采集，迁移保持算法而未重新计时。第一子区域的主要成本发生在节点消元之前，不能把问题全部归因于 dense closure，也不能归因于 runtime 次数展开。下一步在已拆开的端点职责内分析首次条件证明和表达式比较的工作量。

### 已提交基线的功能证据

以下来自 `4b2c432`，不能冒充本轮新的完整资格：

- shared L0C 同步29.32秒、最终 IR 独立模型 n=0..3 通过；公共 CLI/CANN 总31.66秒，CANN2.33秒。模型与构建的前后 IR 逐字对应，见[身份记录](../../runs/lifetime-refactor/shared-build/finite-model.json)、[ASC](../../runs/lifetime-refactor/shared-build/shared_l0c_qk_pv.asc)。这是单AIC小例。
- 默认 FA 完整构建在300秒终止，worker被kill/reap，见[状态](../../runs/lifetime-refactor/fa-default/build-result.json)。没有成功 ASC/CANN，不启动下一候选绕过失败。
- 双 AIV 嵌套 CV 最近完整资格仍为 `dbcef70`；共享 L0C 或小图测试不替代它。

## 支持边界与后续顺序

1. 保持唯一值流、存储和事件路径；新的职责已直接接入并删除旧方法，不引入兼容层。端点关系所有者与区域组合已分开，后续不能把关系证明重新放回编排器。
2. 在结构边界明确后分析剩余编译成本；不提高预算、不改变 FA 排步、不用跨编译命中掩盖复杂度。阻塞解决后才恢复完整 FA 的单次限时构建。
3. progress 仍不支持任意 finite/stream 的联合存在投影。逐边投影不等于共享 ordinal 的联合证明；未知关系保留保守诊断。
4. private 重排只支持精确 cell、完整首写、内部值闭合、同 offset 连续 packet；streaming envelope、跨代状态、动态 ownership、部分周期 reset 等不能凭 private 自动放宽。
5. 完成当前链路功能与编译资格后，才恢复 FA 变体、ASC可读性和设备性能；届时同时对比 AscendC/ACLNN、profiling 与硬件理论指标。

## 复现

在 `compiler/` 执行本轮的受影响回归：

```bash
timeout --kill-after=3s 60s ../.venv/bin/python -m pytest -q \
  tests/analysis/test_progress_conditions.py \
  tests/analysis/test_progress_connectivity.py \
  tests/analysis/test_event_progress_regions.py \
  tests/analysis/test_progress_bound_facts.py \
  tests/analysis/test_progress_queue_interfaces.py \
  tests/analysis/test_progress_counterexamples.py \
  tests/analysis/test_case_progress.py \
  tests/analysis/test_progress_parent_interfaces.py
```

完整算子构建继续使用公共 `build_file/build_module`，覆盖 Python构造至CANN的总期限不超过300秒。超时清理所属进程组，不继续下一候选。只提交产品文件，不提交运行证据、环境或缓存。
