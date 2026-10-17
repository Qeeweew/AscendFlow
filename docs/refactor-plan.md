# 同步生成重构计划

2026-10-17。**本计划替代此前以加速事后进展证明为主线的方案。** 当前算法开发已按用户要求停止，本次只整理交接；后续 agent 应从生成算法重新设计。现状与证据见[状态页](refactor-progress.md)。

## 1. 问题与目标

当前 `select_notifications` 只建立数据依赖的capture/acquire覆盖。普通编译随后在资源阶段建立事件域，重建阻塞关系证明progress，再证明lease及共享ID的消费路径。最近的DNF、共享公式、Path和搜索树改造主要降低这些后置分析的成本，未使同步生成本身具有完整、可组合的正确性保证。

**目标是有明确前提的生成规则及其组合，而不是更快地验证一份保证不完整的通知集合。** 规则正确性在开发时论证并回归；每个程序仍须检查适用前提，建立新的必要关系。不能把所有 `prove` 调用都叫重复验证，更不能直接删除检查以获得速度。

## 2. 先完成职责审计

从真实调用链开始，不先增加缓存、图算法或通用证明框架：

```text
compile → synchronize_structure
  → native / admission / dispatch
  → logical_valueflow + storage_lifetimes
  → select_notifications
  → allocate_structured_events
      → EventDomains
      → prove_progress
      → establish_lease
      → bank allocation / prove_finite_event_sharing / event path cover
  → materialize_notifications → simplify_scalars → ASC
```

同时审查依赖/通知是否被不必要地切分成大量独立状态。历史44个native、726条needs和221条notifications只是测量，不是必须保持的输出规模；允许删除不合理的中间状态或冗余同步，但必须保持真实访问、数值、buffer与排步合同。

对每项义务记录四件事：当前由谁建立、依赖哪些输入、下游在哪里重新建立、何种改变才使它失效。必须区分：输入合法性、物理依赖、生成规则前提、协议组合、资源分配的新约束、对已有结果的重复证明、独立测试验证。只在确认是同一事实时消除重复。

`verification_mode="owned-construction"` 目前是产物标签，不证明同步算法通过构造保证正确。检查不能仅凭这个字符串或一个verified标记被跳过。

## 3. 设计并实现最小生成规则

先给出具体规则和反例，再直接编码，不停在报告或常规选择确认。每条规则必须说明：

| 内容 | 要回答的问题 |
|---|---|
| 适用前提 | 哪些实际engine、访问、排步、代次、存在域及ownership允许使用？ |
| 生成位置 | Set捕获哪个真实完成里程碑，Wait阻塞哪个queue/controller？ |
| 建立保证 | 数据ready、最后读取release、配对、consume-before-reuse、可进展各由什么保证？ |
| 循环状态 | 零次、短循环、多slot/generation、prologue/loop/epilogue如何保持？ |
| 资源条件 | 哪些bank/ID/width/quorum约束留给分配器？ |
| 组合与失效 | 父子/顺序/分支如何组合，哪些改写需要重新建立局部事实？ |

最小例包括双缓冲GM→UB→Vector→GM、最后reader/WAR复用、Cube/Fixpipe、嵌套与双AIV CV。不能因这些例子简单就排除现有FA所需的多层固定排步；也不能为FA另设规则。

配对不等于无死锁。例如两端各先Wait对方、再Set自己，数量和ID可以配对而双方都无法前进。新生成规则必须通过排步前提和可组合的进展约束排除这一类结果，不能把问题继续完整交给生成后的通用图求解器。

生成器可以对未知动态alias/ownership要求可验证合同或具体拒绝。不得通过全局barrier、额外搬运、减少重叠、牺牲buffer复用或改算法掩盖缺失保证。

## 4. 循环组合是主要接口

循环保留参数化body，按prologue/loop/epilogue处理；runtime长度不展开。输入前提、一步保持关系、退出导出关系分别明确。外部连线绑定真实ready/release、访问与事件状态，不能将loop exit视为全部引擎完成。

父层只处理实际外部边界与祖先绑定，不重新打开子循环内部路径历史。摘要大小必须可解释并可计量；把整张内部图藏进惰性Path、随后在父查询中再访问，不自动等于完成了这种组合。

多buffer slot、logical generation、stage offset与event width分别建模。空循环继承原状态；短循环不能假借稳态；两个AIV仍共用一份body并由sid区分，各自私有UB和quorum贡献保持。

## 5. 接通资源与物化后删除旧职责

ID分配消费生成/组合已有的生命周期事实，处理实际新增的bank容量、冲突、width与共享约束。一一改名不重做访问和进展分析；改变width、共享成员或等待位置则必须明确影响范围。不能假定分配只是整数改名，也不能保留每个候选都重新搜索整套执行路径的无界工作。

物化只把已确定计划写入实际IR，后端只打印该IR。同步省略依据已有的、独立于被省略同步的覆盖关系，不采用“试删后全图重验”。纯标量优化不移动异步操作或破坏guard定义性。

替代合同和当前消费者接通后，在同一正常链路删除被替代的后置分析及旧API。没有legacy目录、兼容别名、第二编译路径或测试内旧求解器。不要只把现有调用搬进debug开关：必须先说明它承担的每项必要义务由谁接替。

## 6. 验收与完成门槛

- 输入检查、生成规则、组合保持、实际IR破坏和独立有限模型各自负责明确问题；不以xDSL verifier或小模型代替同步正确性论证。
- 普通编译的调用链须能说明每项事实的唯一建立者及失效范围；保留的昂贵分析必须有具体、不可被已有构造保证替代的义务。
- 通过公共DSL入口取得Vector、GEMM/Cube-Fixpipe、嵌套CV和默认FA的对应IR/ASC/CANN产物。旧solver测试、历史小例或改变FA排步不能替代。
- 结构重构完成后再测编译性能。固定源码/政策，先单次定位，再按[验收口径](validation.md)做冷进程测量；既有目标为默认FA source→ASC五次p50≤5秒、max≤10秒，尚未达到，不得降低门槛美化结果。
- 每算子从Python构造到CANN的完整构建硬上限300秒，到时清理进程组；失败先定位，不继续候选或反复跑长测。
- 当前不恢复算子调优、打包或设备压测。后续性能工作需大规模输入、AscendC/ACLNN、最终ASC、profiling与硬件理论指标共同说明。

未完成上述生成合同与真实链路资格，不能将“证明器更快”“通过更多单元测试”或“目录更整齐”标记为重构完成。
