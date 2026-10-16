# 当前重构状态

**重构尚未完成；不保留被替代的实现或兼容路径。** 本轮已拆开源码值版本与物理存储生命周期，通用 PLE 复用通过实际 IR 小例。共享 L0C 子循环已经通过最终 IR 有限模型和 CANN 离线编译；默认 FA 的本次完整构建仍触及300秒硬上限。没有恢复算子调优或设备工作；没有当前 FA 成功构建或性能达标结论。

合同见[编译不变量](compilation-invariants.md)，模块职责见[源码组织](source-layout.md)，下一阶段见[重构计划](refactor-plan.md)。历史由产品 Git 和外层 runs 追溯，不再在此维护多轮流水账。

## 本轮实现

- `structured_dependencies.py` 的旧递归 `_Builder` 已删除，入口只编排值流、dispatch 和存储义务。`access_frontier.py` 删除未使用的有限求解器，仅保留实际端点及 FIRST/LAST 选择类型。没有将旧实现移入 tests，也没有兼容 alias。
- `logical_valueflow.py` 只建立精确 RAW、条件 KILL/UPDATE 和源码区域接口；`CellEffect` 保留各 operand 的真实访问。`mad_first` / `mad_cycle` 的读条件来自原语合同，首 clear 不再取得虚构的旧值；周期大于 1 的 reset 在通用循环值闭包尚未建立时明确拒绝。
- `region_coordinates.py` 统一条件、绑定和域构造。`storage_packets.py` 证明闭合的精确 private 生命周期，`storage_word.py` 建立静态 prologue / 符号 loop / 静态 epilogue 邻接，`storage_lifetimes.py` 产生最后 reader→下一覆盖的 WAR/WAW。源码读版本不变。
- 边界和稳态共用 consumer `j`，source 为 `j-distance`；singleton 用 membership guard 表达，负 distance 合法。没有运行期迭代展开。两端等式比较的 premise/query 统一执行相同整数规约，原始 unsigned 定义域检查仍独立保留。
- 后续 selection、配对、progress、lease、分配和 ASC 沿用唯一链路。算法仍在 examples；两个 AIV 的共享 body/runtime sid 模型未改变。

## 当前测试证据

各组有重叠，不相加作为独立总数；未跑完整 suite。

| 检查 | 结果与边界 |
|---|---|
| 条件访问、native geometry/packet、admission | 99 passed，4 个 real DSL 项未执行 |
| 逻辑值版本 | 14 passed；首次 clear、RMW、空循环、reader families、shared L0C 的独立逐字节读版本 |
| 纯 PLE 邻接 | 41 passed；offset/lexical 排列、短循环、负 distance、静态预算及大 runtime count 不展开 |
| PLE 实际依赖绑定 | 48 passed；n=0..4 的 RAW/WAR/WAW 与独立实际 native 访问轨迹一致 |
| 原依赖回归 | 19 passed，默认 FA 未执行 |
| 条件端点与小型通知选择 | 27 passed，7 deselected；未包含 FA/shared L0C 完整生成 |
| 整数关系与生命周期小例 | 108 passed；包含6个实际同步/拒绝/破坏回归 |
| 本地 Vector/Cube 最终同步 IR | 22 passed，见 [日志](../../runs/lifetime-refactor/local-pipeline-tests.log) |
| 条件端点图 | 7 passed；部分直接路径、互补 guard、缺 peer、无返回的未来分支 |
| 候选实际坐标 / 条件 lease | 分别3项 / 9项通过；测试协议显式区分 Scalar controller 与普通队列 |
| 原事件资源集成 | 16 passed，24.49秒；含真实parent重入、缺失返回路径、mixed与fence合同 |

`tests/analysis/test_storage_lifetimes.py` 包含：独立 scratch 生命周期换序、交错 MTE3 copy、offset 0/1 的 fill/drain 流水（最终 IR 模型 n=0..4）；同 Vector 错误覆盖拒绝；删除实际 MTE3→V release 后独立模型发现 WAR。有限模型不代表任意规模或设备数值证明。

`tests/analysis/test_storage_word_binding.py` 检查实际依赖，不依赖生产 PLE helper 产生预期。`tests/models/structured_events.py` 保持独立逻辑字节版本与硬件队列语义，没有改成消费生产 conditional-read 谓词。

## 共享 L0C 的现状

基线 `dbcef70` 的 FA 失败来源已保存于 [旧队列证据](../../runs/atom-identity/progress-gap/evidence.md)：源码顺序产生 `PV(j−1) FIX → QK(j) MAD`，与真实 `QK(j) → PV(j−1) → QK(j+1)` 排步冲突。新物理阶段只对已证明独立的 plane0 packet 重建相邻关系；plane1 和 PV 累加 RAW 继续保留。

小型 `shared_l0c_qk_pv` 的值、全部实际 storage hazards、选择与 EventDomains 已通过对应检查。旧有限边界表示暴露了 progress 提前存在投影丢掉共享 ordinal 的保守拒绝；统一 PLE 时钟后已越过该阶段。

事件路径已替换为一份条件端点图：候选只生成一次，必要的反向可达性排除无法到达目标的端点，然后保留完整路径 guard 做覆盖。节点/条件状态预算均256，边8192，路径/长度128，候选4096，均未提高；图节点、条件状态、实际边和查询分别计数，不能与旧的重复扫描计数直接作倍率比较。旧前向搜索代码未保留。

n6 的跨父调用 lease 现已通过（事件环宽度4）：单项含 setup 3.35 秒，105个累计图节点、473条实际条件边、26个条件状态、2510次条件边查询，见 [记录](../../runs/lifetime-refactor/lease-graph.log)。该数字是功能诊断，不是冷编译基准。

完整 `shared_l0c_qk_pv` 同步生成29.32秒，最终 IR 独立模型 n=0..3 全部通过，见 [日志](../../runs/lifetime-refactor/shared-graph.log)。公共 CLI/CANN 构建通过，总31.66秒、CANN2.33秒；[同步前IR](../../runs/lifetime-refactor/shared-build/before.xdsl)、[最终IR](../../runs/lifetime-refactor/shared-build/after.xdsl)、[ASC](../../runs/lifetime-refactor/shared-build/shared_l0c_qk_pv.asc)已保存。两份IR与前述有限模型产物逐字相同，直接复用证据，见 [身份记录](../../runs/lifetime-refactor/shared-build/finite-model.json)。ASC仍是10个native模板各一份；双plane及各层双buffer、算法和issue order未改变，没有增加搬运或全局 barrier。

这使通用共享 L0C 小例闭环，但不替代 FA、设备数值或性能资格。默认 FA 的公共 CLI/CANN 构建在300秒硬期限终止，worker 返回-9，清理及记录耗时后总300.14秒；没有成功 ASC/CANN。见 [状态](../../runs/lifetime-refactor/fa-default/build-result.json)。这是失败构建，不是编译性能资格。没有启动下一算子候选或延长上限。

随后一次60秒上限的只读阶段提取在22.87秒完成：lower0.42秒，memory0.07秒，bounds1.49秒，admission5.42秒，dispatch0.01秒，值/存储1.70秒，selection3.90秒，EventDomains9.87秒。44个native、726条逻辑/存储义务、221条通知、142个参数case，1个private packet重排分区；见 [记录](../../runs/lifetime-refactor/fa-stages.log)。这是单独诊断，不能与300秒失败耗时相减作为精确阶段归因；进一步定位资源阶段，暂不继续完整FA构建。 第二次60秒上限的阶段诊断确认22.69秒进入 `progress`，45秒定时栈采样时诊断进程发生SIGSEGV（139），只取得不完整的scalar栈；不能把该栈当作根因。见 [诊断日志](../../runs/lifetime-refactor/fa-progress-diagnostic.log)。后续使用同步阶段/region计数定位progress接口组合，不再依赖该异步采样；没有继续完整FA候选。

## 明确边界

- 独立重排只授权精确 private cell、完整首写、内部读版本闭合、同 offset 且实际连续的 packet。streaming envelope、跨代递推、未证明的条件 packet 不取得该证书。
- 改变父接口需要传播实际首末 reader。当前仅 root-private、无外部访问且无重复父调用的 scratch 允许更换首末接口；其余保持接口或不授予独立重排。
- progress 尚未实现任意 finite/stream 的联合存在投影。统一 PLE 时钟解决当前表示问题，不能扩大为通用能力声明。
- 五 slot 复合映射、部分周期 reset、动态 ownership 等仍有明确拒绝。未知关系不能被当作安全。
- 编译性能分析在结构收尾之后进行。ASC 可读性、FA 变体、设备正确性、ACLNN/profiling/理论硬件性能尚无本轮资格。
- 双 AIV 嵌套 CV 最近资格仍为 `dbcef70` 基线的24.09秒构建；本轮共享 L0C 是单AIC小例，不能替代双AIV或设备资格。

## 复现

在 `compiler/` 执行：

```bash
../.venv/bin/python -m pytest -q tests/analysis/test_logical_valueflow.py tests/analysis/test_storage_word.py tests/analysis/test_storage_word_binding.py
# 实际最终 IR 正/负/删 release 反例
timeout --kill-after=5s 300s ../.venv/bin/python -m pytest -q tests/analysis/test_storage_lifetimes.py
```

每个完整算子构建最多300秒，超时清理进程组并停止该批候选。产品文件只提交 compiler Git，运行证据留外层 runs；没有打包、发布、推送或设备运行。
