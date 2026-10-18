# 同步生成设计要求

2026-10-17。本页规定必须保持的语义；已实现的生成规则及逐项事实归属见
[generation-rules.md](generation-rules.md)。规则论证、有限反例、ASC/CANN 和
设备结果按[验证口径](validation.md)分别记录。

## 1. 编程模型与当前缺口

使用TileLang风格AST DSL表达计算、物理空间、views、buffer、`serial/Pipelined/Stage`与`Cube/Vector`执行域。两个AIV共享一份body，由runtime sid区分。`cube.task/vector.task`不恢复，内部physical Task IR按lowering需要保留。

循环长度为runtime值，普通常量界循环也不自动展开；只有显式unroll授权展开。算法、FA变体与分块政策留在examples，不进入编译器。

group/Vector peer 的并行独立性属于[源内存合同](memory-contracts.md)，不再以
全局地址分离证明准入。未知区域关系进入 AliasRegion，保留源码内存顺序；
精确数值 RAW、初始化和物理复用仍各自保持。外部 write_disjoint 输出合同
不传播给 scratch，也不提供 ready、release 或事件消费保证。

当前生成器先绑定覆盖与真实位置，检查域和 physical-step/phase 适用前提，
然后以物理 cell credit 或有限 queue handback 建立生命周期。分配器消费
完整 `NotificationProgram`，仅引入有限 bank、ID 和共享约束；它不重新建立
依赖/进展图。父层只消费 child 的真实端口和 may-block cuts。

## 2. 规则正确性与输入前提

规则在开发阶段证明并测试；每次编译检查本程序满足规则前提。输入前提不满足时应给具体诊断，不能把任意排步都先转成通知集合再期待通用求解器完成全部工作。

生成规则要共同处理：物理依赖、capture/acquire的真实位置、queue/controller派发阻塞、配对与consume-before-reuse、必要的seed/drain、资源分配接口。匹配的Set/Wait数量不保证无死锁；数据ready也不保证flag已可复用。

不要先建庞大proof DAG、协议类库或通用solver框架。用现有真实DSL例子建立最小规则及组合合同，接通后直接删除被替代的实现。

## 3. 循环与边界

对N次逻辑迭代，stage偏移s在物理step j对应`g=j-s`，存在条件为`0<=g<N`。归一后的最大偏移为S：

- N=0：没有body native及其配对通知，原有边界状态透传。
- 0<N<S：使用带存在条件的短片段，避免重复的fill/drain。
- N>=S且N>0：prologue为`[0,S)`，loop为`[S,N)`，epilogue为`[N,N+S)`。

这些是执行域分解，不要求展开每步。FIRST/LAST在N=1重合时不重不漏。S=0不表示没有loop-carried release；stage深度、buffer slot数和event width分别建模。

循环应有输入前提、参数化一步保持关系和退出导出关系。前后节点通过实际访问与里程碑连接，不机械地在入口/出口各插barrier。exit不是全部异步操作完成，epilogue也不自动等于引擎退休。

父层只组合真正外部可见的边界及祖先坐标，不重新打开子循环完整路径历史。摘要规模必须可计量；只把内部图封装成一个惰性对象并不能满足这个目标。

## 4. 必须区分的语义

| 事实 | 含义与边界 |
|---|---|
| 逻辑值依赖 | read对应哪个writer/代次；SSA use-def不是异步完成 |
| 物理访问 | root、alias、subview、layout、范围、slot及owner；不同名字不保证不别名 |
| ready | 所需写已完成并对consumer可见 |
| release/credit | 目标buffer的全部相关最后reader已退休，可进入下一覆盖；不必等待整个计算结束 |
| event消费 | 旧Wait已消费通知，才能复用对应硬件状态；与buffer复用分别保证 |
| 最终drain | kernel返回所需收尾；不能替代普通循环的局部生命周期 |

精确RAW和条件KILL/UPDATE不能因物理重排而改绑。WAR/WAW基于给定dispatch和物理生命周期。递推状态、累计/舍入边界保持原语义；FA的FP32求和顺序许可不推广成任意数值变换许可。

Scalar、Vector、MTE1/2/3、Cube和Fixpipe使用dav-2201的真实契约。Set覆盖的退休不能扩展到后续发射，不能泛化同pipe的隐式完成顺序。跨AIC/AIV还须保持GM可见性、controller阻塞和两个AIV的AND quorum；它们不能由OR可达路径替代。

## 5. 资源、优化与物化

协议生成/组合明确其生命周期要求，分配器处理有限bank、配置预留、ID/width和共享约束。改变共享成员或width属于新的合法性义务；单纯ID改名不应重新证明内存与进展。

省略同步直接使用既有覆盖及其前提，不试删后全图验证。不得以减少buffer复用、额外搬运、全局barrier或静默fallback换取易于证明。

xDSL负责SSA、region、rewriter、verifier和pass基础设施；不引入MLIR工程依赖。普通DCE/CSE/代码移动不能忽略异步effects、ownership和完成语义。物化与ASC后端消费已确定的计划，不重新选择同步。

## 6. 复杂度与支持边界

复杂度按native模板T、访问及实际相交A/I、物理依赖D、边界条款B、通知C、资源冲突J、条件条款G与真实耦合周期P说明。运行长度N不成为状态展开维度。独立ring不做全局LCM或参数笛卡尔积。

不预先承诺任意动态别名、任意布尔条件或任意多时钟ownership均可线性处理。必须支持的现有DSL/算子结构不能为了验收而删除；超出已设计规则的情况给出具体支持边界，不能靠算子名字分支。

验收需要生成前后IR、实际ASC/CANN产物、原语反例、有限执行以及工作量与冷编译数据。算法论证、xDSL结构验证、有界模型和设备结果各自只支持其范围内的结论。
