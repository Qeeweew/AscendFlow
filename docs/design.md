# xDSL / Ascend 910B 编译器设计

2026-10-14修订。实现基线为`be957e8`，方案重审前已保存局部顺序流水实现。
本文规定目标架构，不把计划当能力。语法见[DSL规格](dsl-multistage.md)，实施与
退出条件见[重构方案](refactor-plan.md)，实测见[验证记录](refactor-progress.md)。

## 1. 结论与取舍

采用TileLang风格的受限Python AST前端，保留设备上的结构化控制流；用户明确
计算、搬运、布局、缓冲和stage政策，编译器负责物理依赖、异步完成和同步。
在同一个模型内表达核内MTE/Vector/Cube/Fixpipe流水，以及AIC/AIV之间的CV
多阶段流水。自动排步后做，专家排步始终经过同一同步检查。

一个功能只有同时满足下面三项才算完成：

| 验收面 | 必须回答的问题 |
|---|---|
| 源语言 | 能否直接表达算法、状态、各层流水和所有权，不要求手工事件状态机？ |
| 编译成本 | 固定源码与政策下，长度是否只影响设备工作？静态结构扩大时成本是否可解释？ |
| 设备执行 | 最终ASC的搬运、复用、等待、数值次序和重叠是否符合政策？设备证据如何？ |

不以全局barrier、增加搬运、缩小通信块、增加buffer或静默fallback换验收通过。
支持边界可以小，但拒绝必须具体；预算耗尽表示分析未决，不表示程序必然非法。

## 2. 对现有方案的修正

旧方案的语义方向可以保留，但实施路径仍有风险：

1. **有局部正确性，缺区域组合的一致入口。** `lower_source`、`lower_sequence`
   和invocation分支分别接纳不同程序形状。继续增加顶层特例会使每个新控制
   组合重做lowering、同步和codegen。应逐步复用共同区域合同，保留叶证明器。
2. **“不展开循环”不等于长度无关。** `Graph._domain`仍会为查询窗口建立
   translated states；若首尾距离进入窗口，常数百万界仍会膨胀。协议周期LCM、
   region摘要、event候选重查也有独立成本。必须检查内部计数，不能只数IR操作。
3. **连续CV被安排得太晚。** runtime商余解码、query状态与task槽的不同代次
   是架构风险。应在完整FA之前做可执行的小切片；之后仍由真实FA验证整体。
4. **组件容量不等于组合可行。** 目前sequence为各叶分配不相交event IDs，
   可能拒绝物理上可复用的bank。共同资源寿命检查必须早于多个Cube子流水组合。
5. **目标源码不等于语言可用。** FA fixture已有具体数学主体，但缺独立同政策
   AscendC和新AST闭环。宏、views、数值原语不能无限留到“以后补”。

不从零重写所有分析，不先搬目录，不建设通用整数求解器或任务运行时。每次
抽取公共职责，都用旧例及一个真实新组合验证；删除旧入口前保留差分证据。

## 3. 已实现的边界

| `be957e8`能力 | 当前限制/证据 |
|---|---|
| AST单流水、对齐prefix、归约/Scalar、显式L0和runtime K | 各自声明的子集有ASC/CANN/设备记录 |
| serial父×一个local child | 严格GM frame；父体一般前后工作与多child尚不支持 |
| 根部once/stream顺序组合、真实init/use、共享只读GM | 一个actor、一个共同runtime bound加固定count；跨region漂移写拒绝 |
| 序列边界同步、零迭代、最后reader保护 | 四模板state_sum通过1 GiB设备例；冷源码→ASC p50 0.1853 s |
| 最终物理IR重检、预算、不可变Compilation | 已有；属性读取不是源码键编译缓存 |
| 多级CV、规则query×KV连续流、新AST真实FA | 未接通；legacy FA记录不能移用 |

state_sum与独立AscendC三批profile的中位数比为0.99777，仅说明该固定政策
在观测波动内一致，不能推出FA编译性能、任意区域组合或CANN竞争性能已达标。

## 4. 前端与执行模型

公开语法保持`@T.prim_func`、`T.Tensor`、`T.Kernel`、`T.serial/Pipelined`、
`T.Stage`、`T.Cube/Vector`和typed原语。不恢复`cube.task/vector.task`。

必须分开六个概念：

| 概念 | 含义 |
|---|---|
| 逻辑迭代 | 原程序的任务与数值递推；`n/Sq/Sk`默认runtime SSA |
| actor | AIC或AIV，拥有自己的控制派发；Cube作用域内也可以有DMA |
| engine | S/V/MTE1/MTE2/MTE3/M/FIX，决定原生完成契约和event bank |
| stage offset | 最近pipeline内的逻辑代次位移；不代表共同墙钟或完成点 |
| issue order | 每actor在本地step中的派发顺序；固定后不自动更改 |
| storage slot | 物理复用轴；与stage数、event数、query状态代次独立 |

buffer/view保留allocation身份；借用tile值标识写版本，不产生隐式快照。
容量、tile、layout、slots和政策是constexpr，运行时长度及地址不能按调用值
自动特化。常数边界也默认保留循环，只有显式`T.unroll`授权受预算的展开。

采用受限AST宏减少物理样板代码，宏体放examples，展开为普通操作和保留循环。
传入实际buffer、scratch和数值参数；展开需卫生化名字、保留source map及内容hash。
宏不能注册可信同步摘要或直接emit ASC。核心只实现通用原语和目标契约，不识别
FA、softmax或example名字。先实现被真实程序需要的宏子集，不做任意Python。

普通嵌套表示：父actor到达child后派发child控制循环，再继续父工作。child填排
完成只说明延迟工作已派发，不说明异步工作退休；不会隐式把兄弟stage插入
child内部。跨query连续重叠使用明确的连续task流，不承诺靠嵌套自动获得。

CV的一组为1 AIC+2 AIV。各自推进本地step，跨核Wait会阻挡后续派发；等待位置
是性能语义。910B交接采用已验证的GM数据路径，不照搬950直达路径。通信tile、
Cube native tile和Vector行块分别选择，不能把整个通信tile强装进UB。

## 5. IR边界和代码职责

```text
受限Python AST / examples宏
  → xDSL source：scf/arith、refs/views、actor/stage、数值语义
  → xDSL physical regions：保留控制、native访问、地址、派发与ownership
  → 自动同步：义务 → 逻辑协议 → 物理event与实际端点
  → 最终IR重检 → 不可变Compilation → ASC → CANN
```

xDSL是程序真相，不建立另一棵可变程序树；不可变分析索引/摘要可以缓存。
物理同步作为同一IR的操作物化，不另建庞大调度框架。改变访问、layout、guard、
owner、slot或派发顺序会使相应分析失效；最终checker不信任planner已验证标记。
这里的独立重检指重新提取最终IR、不接受planner缓存证据；两者仍可共享底层
访问/图分析代码，不能称为两个独立实现。测试另用独立有限指令/逐字节oracle。

近期重构按职责收敛，暂不强制目录重排：

- 前端解析及类型/数值检查，统一源码位置与constexpr/runtime分类。
- 一个actor绑定环境管理allocation、uniform表达式和frame；region访问器递归
  lower控制，native recipe只处理原语。逐region构造`_Actor`再合并只是过渡。
- 叶区域、顺序组合和循环调用共享边界合同；叶周期证明器保留，父层负责解除
  入口假设、组合ownership与资源寿命，不把整个child变成opaque task。
- codegen按已验证控制和物化同步发射，不自行补barrier、变换算法或重新规划。
- 产物保存源码映射、配置、报告和hash；详尽witness按需导出。

分支只按已实现的有限guard接入；一般if/else、break、数据相关while不因为
xDSL能表示就自动获得同步支持。generic DCE/CSE/代码移动必须尊重异步effects、
SPR和ownership；新变换之后重新验证。

## 6. 区域摘要与同步合同

每条原生操作记录实际root、精确byte footprint、R/W/RW、actor/engine、activation、
generation及源码。padding、layout转换、scratch、Scalar metadata和SPR同样计入。
以下事实不能合并：逻辑结果、写完成、输入最后读取、通知消费、buffer release、
最终drain。只有可信native契约允许提前退休；函数返回或SSA use结束不是依据。

区域摘要包含：入口待满足的初始化/覆盖义务；各byte piece末writer及仍存活
reader families；可信完成前缀；owner/代次/激活条件；未消费通知与bank寿命。
它是带前置条件的传递关系，不是单一`done` token或由用户随意提供的证书。

| 结构 | 组合义务 |
|---|---|
| A;B | A的writer接B首读/首写，所有未退休reader接B覆写；未触及piece继续存活 |
| 循环 | 入口、一次递推、出口、零次恒等路径；保留调用坐标和局部代次 |
| 有限分支 | guard相容的版本合流、每条可达读取初始化、通知与等待激活一致 |
| CV交接 | GM完整贡献ready、所有reader释放、参与者与quorum、可见性和派发gate |

初始化必须是真实写或外部输入合同，末代写不能初始化首代读。只读共享依据
实际effects及外部不并发修改合同；不同tensor名或`access=read`不能证明无别名。
跨owner写需要独占域或明确交接；动态唯一性需可验证输入合同，否则拒绝。

本地HardEvent按核和有向bank计8 IDs，验证旧Wait消费先于同ID下一Set生效。
region控制退出不能释放ID。先构造逻辑channel，再按整个组合的寿命冲突分配ID；
互斥activation仅在证明后共享。候选不足时报告bank/活跃区间/冲突，不缩ring或
加drain。资源搜索有限且可取消；可行性与最少同步不混为一谈。

同pipe只使用dav-2201的真实顺序契约，短MAD等反例可能需要局部等待。跨核通知
计数、reverse ack和内存可见性单独建模，ack不代表GM reader退休。最终drain
覆盖实际未完成工作和通知寿命；子region出口不默认drain。

## 7. 编译成本的设计边界

令N是运行时迭代数，T是native模板数，A是精确访问pieces数，H是实际冲突数，
P_c是某真实耦合分量的静态周期，V_c/E_c是其证明图，Q_c是查询数，W_c是实际
展开的证明窗口，R_c是候选修改后重建次数。需要计量的成本为：

```text
前端/宏/原语物化 + 访问索引与H
 + Σc [图构建 + 查询(Qc,Vc,Ec,Wc) + Rc×失效重建]
 + 独立最终检查 + 序列化/代码生成
```

N不得进入模板、协议周期、查询窗口或缓存键；循环界整数编码仍有位数成本。
访问扫描目标为`O(A log A + touched + H)`，密集alias下H可以二次；一般图查询
为`O(V+E)`，势函数松弛最坏`O(VE)`，guard/周期/边界状态另算。没有整个编译器
线性复杂度承诺，不能仅凭“有限摘要”声称W与N无关。

具体约束：

1. 首尾位置用head/tail相对坐标；查询证明覆盖实际count的边界族，避免展开
   两端之间全部代次。百万常量界和runtime界都测内部状态数量。
2. period只由真实slot/guard/protocol耦合产生。独立存储不做全局LCM；共享bank、
   Scalar gate、CV通知可能重新耦合，不能错误拆开。
3. `task=q*K+k`保留受限商余关系，只对静态slot需要的K余数分类，并单列短K
   边界。这不是任意含runtime除法地址的证明器，不能枚举K个residue。
4. 单次编译复用不可变索引及等价查询，只重查受影响部分；最终checker独立
   提取。摘要规模、分类数、候选数和失败路径都设结构预算。
5. 先修规模，再用CPU profile优化热点；结构稳定、收益明确后才引入原生加速。
   缓存、并行和更高超时不作为首编复杂度修复。

门槛：五个冷进程完整源码→ASC，常用例p50≤1 s/max≤2 s，真实FA≤5/10 s。
Python启动/import、CANN及cache-load另列。真实源码键命中目标≤100 ms，不能用
`Compilation.asc`属性读取充数。新runtime长度复用同binary及launch合同检查。
缓存键包含源码/宏传递内容、constexpr、native合同、编译器/目标/选项；CANN
产物另含工具链。cache miss完整验证，落盘原子写入并检查内容完整性。

## 8. 算子性能与理论分析

FA及变体、数值oracle、AscendC对照和ACLNN适配器均在examples。核心wheel脱离
examples/reference仍可安装编译。通用原语至少用一个非FA场景检验，避免FA特判。

FA以显式online-softmax验收：FP32 m/l/O、p32 denominator、P显式舍入、rescale/add
分开、最终div或reciprocal+mul明确固定。逻辑BK变化可能改变舍入，native累计
分块也需检查次序。同步优化不授权改变数值语义。

先审查最终ASC：运行时循环、native块/搬运次数、Q/K/V与state驻留、各层buffer/ID
峰值、release所在reader、cross-core gate、跨query填排。ASC符合派发政策仍不
等于硬件重叠充分；必要时查CANN编译结果并用profile验证。

三种对照分工：独立同政策AscendC隔离编译器退化；安装ACLNN测竞争差距；
ops-transformer/CATLASS源码解释机制。生成ASC改名不算独立对照，最新源码与
安装CANN不假定一一对应。偏差5%是调查阈值，不是可接受性能损失承诺。

按实际SKU/活跃核/数据类型/频率与native工作量估计条件下界：

```text
t ≥ max(各共享计算/搬运资源的总需求÷对应吞吐上界,
        数据依赖和派发关键路径下界)
```

资源需求包含Cube FLOPs、各类Vector指令、MTE/FIX流量及必要Scalar派发。
共享端口需求先相加，可并行资源不盲目相加；prefill和decode分别估计有效并行度
及填排。dense两次矩阵乘有效FLOPs为`4*B*Hq*Sq*Sk*D`，causal按可见pair计，
padding/重复计算另列。HBM实际字节、GM请求字节和L2命中不能互相替代。未知
持续带宽或Vector吞吐保留未知，platform ini无单位rate不能直接当GB/s。

已有实机记录为910B3、20 AIC/40 AIV、192 MiB L2。作为明确假设的计算示例：
若20 AIC均以1.8 GHz运行，每核每周期完成一个16×16×16的b16 FMA块，FMA计2 ops，
则条件Cube峰值为294.912 TFLOP/s；B1/H32/S8192/D128 dense FA两次矩阵乘有
1,099,511,627,776有效FLOPs，仅Cube理想下界约3.728 ms。它未计Vector、搬运、
派发和填排，不是测得延迟或全kernel可达性能；持续频率与该吞吐假设须另外验证。

固定大例、超过实机L2的buffer轮换、热缓存重复分别测量。记录编译选项、
源码/ASC/库hash、shape/dtype、数值和布局合同、SKU/核/频率、host/device计时
范围、profile开销与批间波动。activity可重叠，单个占比不能确定瓶颈。
gap需对应ASC位置、可检验瓶颈假设及对照证据；算子调参仍暂停。

## 9. 扩展边界与源码依据

| 算子族 | 检验的合同；不代表完整算子已支持 |
|---|---|
| elementwise/归约/normalization/RoPE | 多reader、Scalar、scratch、view别名、原地WAR |
| gather/scatter/MoE routing与finalize | 间接访问、唯一性/atomic、动态ownership、早release |
| grouped matmul/量化/FFN | 空group、共享权重、多输出发布、独立native环 |
| GDN递推状态 | FP32 carry与低精度snapshot分离、状态owner与顺序 |
| dense/GQA/MQA/causal/decode FA | query状态、CV多级流水、有效域、累计/舍入 |
| varlen/paged/sparse/persistent attention | descriptor代次、非矩形访问、空任务与参与者一致 |
| CATLASS分层mainloop | 各级最后reader、跨调用连续流、有限event组合 |

动态队列、全局phase、split-KV、backward/dropout/RNG与通信分别增加合同，
没有空接口式“已支持”。有限交错检查不冒充任意规模证明，xDSL verifier也不
证明异步正确性；设备样本不能代替静态义务检查。

此次只读核对、未pull，参考工作树均干净：TileLang
`6ba65d204c937fcb1c43b1a22b32979506595655`，ops-transformer
`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，CATLASS
`698379e0d4b1b94d12728e7181df2cf83a958e8f`，ops-nn
`64b41ad63a5073bc711ba03802498bfe8bdb77e5`。外层报告固定链接与最新checkout
不同，安装CANN另有版本身份。

依据：[TileLang 910B](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[TileLang FA](../../reference/tilelang/examples/ascend/flash_attention/910b.md)、
[A2 FA设计](../../reference/ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md)、
[arch22 Process](../../reference/ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h)、
[CATLASS生命周期](../../reference/catlass/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)。
更广反例见外层[源码审查](../../docs/xdsl-npu-operator-source-audit.md)，目标契约见
[source-notes](source-notes.md)。参考源码不成为构建依赖，不引入MLIR工程。
