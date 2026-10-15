# 设计：结构化程序上的统一自动同步

2026-10-15 再审；工作保存点 `61c5d06`。本文是下一部分重构的技术合同，
**不是已经实现的统一算法**。实际能力集中记录在[状态页](refactor-progress.md)。

2026-10-15 最新重整：DSL 不保向后兼容，旧 task 用户入口已删除；编译器
内部物理 TaskOp 按需要保留。首尾窗口已迁为符号关系，本轮已共享局部查询域拓扑，
条件事实扫描与父层组合成本仍待优化。本页给语义目标，实际工作顺序以[新计划](refactor-plan.md)为准。

## 目标与纠正

以 TileLang 风格的受限 Python AST 表达计算、存储、局部流水和 CV 多阶段流水。
专家排步与未来自动排步进入同一个物理访问和完成关系模型。自动同步优先，
性能通过最终 ASC、CANN 产物及设备 profiling 验证。

现有源码入口和原生转移规则已有共用实现，runtime 叶流水已直接生成所需协议；
将退出的 legacy 有限路径仍有候选试删，不能将 runtime 结论外推到所有入口。
层次分析的条件状态膨胀仍需替换，局部生成迁移不能代表一般层次 CV 已接通。
继续按程序形状增加证明器，会把组合问题变成不断增加特例。下一次重构要替换
重复的语义和查询职责，而不是增加第四个形状分支或只把文件搬进同一个目录。

同时保留已核实的 native 契约、原语 lowering、ASC emitter 和反例。xDSL 是程序
的唯一可改写表示，分析索引引用它；不建立第二套可变程序树，不引入 MLIR 工程。

## 语言与执行边界

详细语法见 [DSL](dsl-multistage.md)。runtime shape/stride、循环界和 descriptor
是设备 SSA；容量、layout、dtype 和固定 stage 政策是 constexpr。常量循环界也
默认保留，只有显式且有预算的 unroll 才展开。输入长度不参与编译期任务枚举。

`Cube/Vector` 指定 actor，原语决定 Scalar、Vector、MTE1/2/3、Cube、Fixpipe。
Stage 表示逻辑任务的派发偏移，不等于执行引擎、buffer 数、event 数或完成屏障。
嵌套 child 有自己的迭代坐标；离开 child 控制流不能证明其异步访问已经退休。

整个路径为：

```text
Python AST → xDSL 结构化源码 → 保留控制的物理执行 IR
 → 访问/代次/完成约束 → 同步协议和有限资源分配
 → 物化同步 → 从最终 IR 重提取检查 → ASC → CANN
```

## 公共分析模型

下面是必要信息，不预先承诺类名或再创建一组空接口。

| 对象 | 必须保留的含义 |
|---|---|
| 操作模板 | 原生工作、actor/engine、源码位置、所属 region、派发位置 |
| 迭代实例域 | 符号迭代坐标、uniform 参数、activation/guard；不枚举所有实例 |
| 物理访问 | allocation 身份、view/layout 后字节集合、R/W/RW、开始/退休端点 |
| 代次与归属 | owner、slot 映射、generation；fresh frame 的已验证不相交关系 |
| 顺序约束 | 源/目标里程碑、两端实例关系、条件、覆盖范围和硬件强度 |
| 区域边界 | 所需初值、末 writer、尚未退休 reader、可见完成关系、存活通知 |

逻辑值依赖和物理访问分别保留。tensor 名不同不证明不别名；SSA 使用不是异步
完成；借用 view 不是 snapshot。DMA padding、scratch、Scalar 读取及 SPR 状态
也要进入 effects。未知 native effects 直接拒绝，通用 DCE/CSE/移动不能忽略它们。

ready、read-retired、release/credit、notification-consumed、最终 drain 具有
不同语义。每个访问只使用 dav-2201 真实可观测的完成点；没有 native 保证时
不能虚构比操作完成更早的 reader retirement。

## 算法再审：直接生成，约束等待范围，压缩层次状态

审查基线 `61c5d06` 已删除试删与生产路径的旧 `Graph.path`，共用
`native_order`，但当时仍有两个算法问题；下文分别标明当前处理进度：

1. 叶生成器先把所有候选的虚拟指令建图，对每个缺口调用 `route`，再启用相应
   协议。这仍是逐义务搜索。`(新增协议数, mask)` 的选择代价不表示流水性能。
2. 层次分析复制首尾指令片段，并在各节点维护多个 guard 下的源事实集合。
   批量队列降低了重复传播，却没有缩小状态空间；`add_fact` 仍扫描已有 guard
   做包含判断。共享转移代码没有解决这种表示成本。

该基线一次带计数和 profile 的实际编译：分层 GEMM 只有6个 native 模板，最终调用
检查却有1143个节点、2426条边、10079条节点/guard事实记录，最多23个guard/节点。
整个编译有28355次 `add_fact`、115935次 `_includes` 调用。profile 热点在条件
传播与事实维护。计时受 instrumentation 影响，不作为冷编译性能结果。

TileLang `6ba65d2` 的 `OptimizeSyncPoints` 按既定次序检查当前覆盖，保留缺失
关系的对应协议，再增量更新闭包。它没有为每条缺口搜索所有可选协议的路径。
采用这一生成职责划分；不复制其全对闭包，也不把 C++ 当作复杂度解决方案。

### 先确定性能约束，再减少同步数量

最小关系级反例：`R` 是 slot 的最后 reader，`X` 是同流水后续无关工作，`W`
覆盖该 slot。早捕获只要求 R 退休；晚捕获还要求 X 退休。两者都只需要一对
Set/Wait。旧 `route` 可以因为候选编号选择晚捕获，证明安全但扩大等待范围。
这个反例已在旧 `_DomainFlow.route` 复现；它不是已发现的设备 kernel 回退。
当前直接生成回归使用真实双输入 Vector 程序，检查release捕获R并在X前发出。

生成必须以真实访问的最早合法 capture 和最晚合法 acquire 为基准。不能为少
一个 flag 新增对无关工作、其他 slot 或后续 generation 的等待。已有必要协议
已覆盖某义务时可直接省去它，因为没有添加新的顺序。合并 capture 时必须检查
guard、参与者、代次和被等待工作集合，而不只比较同步条数。

不承诺全局最少 event。若当前选择/分配未找到可行方案，应报告具体冲突与未决
原因，不能把启发式失败称为硬件不可能，也不能延迟 release 或增加 barrier 兜底。

### 生成器：访问前沿 → 对应协议 → 更新完成状态

保留 `loop_dependencies`、`region_summary`、`region_sequence` 中的物理访问
分析。按 allocation root、实际字节片段、owner、slot/generation 维护最近
writer 与未退休 readers。逻辑顺序用于提取义务，不作为设备已经完成的证据。

```text
对实际消费者的访问义务分组：
    从真实 native 顺序与已选协议取得该消费者的完成状态
    扣除已覆盖的 RAW/WAR/WAW 义务
    对剩余义务，直接构造对应的 ready 或 release 协议
    只有同一合法 capture 确实包含的义务才合并
    将选定协议的真实转移加入状态，影响后续消费者
检查已选协议的 event 代次、配对、容量和 drain，再分配 ID
物化已选同步；最终 verifier 从实际 IR 重新提取并检查
```

当前已删除 `_proposals → 全候选指令图 → ProtocolSelection.route`。
`_requirements` 只保存实际访问对应的协议规格，没有虚拟xDSL指令或完成效力。
`DirectCompletion` 从native图开始，选中协议才增量添加真实Set/Wait/barrier转移；
既有翻译域原地增加节点与边，不逐协议重建，也不搜索其他未选协议的路线。
物化只创建选中协议的xDSL操作，不留下未选操作的SSA uses。

每个native位置有固定的issue/queue入口与acquire出口。条件同步跳过时保留
输入前缀：例如g=0跳过两代credit Wait，后续ready Wait仍继承此前Scalar派发。
Set只向后传递begin的FIFO关系，不能把发布effect传成后续native的完成屏障。
这些转移与实际物化指令图作正/负关系差分，最终检查仍重新提取实际IR。

处理顺序以真实消费者/时钟的因果位置为基础，同消费者的义务一起处理。
RAW/WAR/WAW 标签只决定需要哪种关系，不决定执行顺序。不同 actor 保留偏序，
不能把 Python 中 AIC/AIV 代码块的先后视作完成顺序。循环携带关系在符号
generation 上处理，不用从第0次迭代执行到 N 的模拟来构造计划。

当前策略先建立循环体关系，再处理携带关系；同消费者先考虑较晚真实源的
capture。消费者存在阻塞派发的acquire时，其必要的同pipe原生fence先处理；
其他位置让外部必要acquire先提供覆盖。该策略不按算子名特判，但选择次序
仍会影响同步数量，不能称全局最优，也不能把分配失败称硬件不可实现。
跨核backlog在已选协议上检查；没有凭空添加ack或全局barrier的修复路径。

双缓冲早release的前后序列（省略无关字段，完整IR由测试程序可复现）：

```text
同步前：load_x(i) → R(i, reads x[i%2])；随后X(i)只访问另一个buffer
同步后：Wait(release_x, i-2)；load_x(i)
        R(i)；Set(release_x, i)；X(i)
```

Set捕获真实R的退休，匹配Wait保护下一次覆盖；X不在该capture内。
回归见 `tests/sync/test_direct_synthesis.py`，不为此新增公开算子或runner。

### 完成状态：保留区分，减少不必要的传播

只需要传播仍服务于内存或 event 义务的完成信息。状态必须区分：

- 实际操作/访问的退休，与 FIFO 派发/开始顺序。
- 真实 capture 覆盖的原生退休前缀，与一个独立任务的结束。
- Set 发布、匹配 Wait 消费和 buffer read-retired。
- actor、时钟、generation、参与者及激活条件。

只有经硬件契约证明包含关系的 capture 前缀可用最大位置压缩。同一流水后一个
任务结束不一定证明前一个退休；不能将所有状态简化成“每引擎一个最大序号”。
多个独立 reader、双 AIV 的贡献也不能合为一个完成位。

原生转移继续由 `native_order` 统一定义：Set 捕获其前序退休，Wait 传递匹配
发布，Scalar gate 影响后续派发；Set effect 不进入后续 native retirement
前缀。SSA use-def 与源码控制退出都不提供这种完成事实。

条件表示先规范化实际 guard，再共享相同条件的转移。避免在每次事实抵达时
遍历所有历史 guard 做包含比较。只细分实际相交的条件；无关环不取全局 LCM。
条件因子化的具体实现仍须通过反例及规模测试，不能把“使用位图”算作完成。
任意条件组合可能膨胀，必须记录碎片数和诊断未决，不承诺通用线性算法。

### 层次摘要：编译 child 一次，组合其边界转移

当前首尾窗口适配器是可保留的回归对照，不再作为一般 nested CV 的扩展基础。
目标摘要由 child 的真实转移在其自身时钟上构成，保留输入前沿、暴露的 capture、
未退休访问以及活跃 event；父层引用这些接口，不为每个 source/destination 或
调用复制 child 的首尾指令图。摘要不得由调用者自报一个可信 child-done。

对规则环，义务直接写为：

```text
MTE2.write_done(i)    → Vector.read_begin(i)
Vector.read_retired(i) → MTE2.write_begin(i+B)
```

需要构造并验证有限的符号状态转移：入口基例、generation 平移后的归纳步骤、
真实出口。状态数量由静态访问/环/条件决定，不能出现 runtime N 因子。零次
child 是恒等转移；slot0 的末读取按 N 的奇偶选择 N−2/N−1，而非统一等待最后
一代。摘要仍必须表达“所有历史 reader 的退休前缀”，不能只留下最后一个 end。

这项摘要实现尚未完成。先用现有双缓冲、frame 和层次 GEMM 验证组合；再用
FA 的早 import release、两 AIV 贡献以及新 query QK 早于旧 query PV 的反例
验证接口。不能未经这些检查就宣称同一个串行摘要已支持一般 CV。

### 验证与反例

最终 verifier 保留，检查实际 IR 的内存关系、event 生命周期和进展；不信任
生成器的覆盖标记。生成与检查共用硬件/条件语义，独立有限执行模型用作差分，
不冒充任意长度的证明。以下反例先于大规模算子测试：

| 反例 | 必须保留的结论 |
|---|---|
| R 后还有无关 X，再复用 R 的 slot | 不为了合并 flag 将 release 移到 X 后 |
| 同 UB 的两个 reader | 一个 reader end 不代表全部 reader 已退休 |
| Set 后有后续 native 工作 | Set effect 不成为后续任务的 fence |
| 两条路径分别只在奇/偶数激活 | 不组成全域完成路径 |
| snapshot child=0 | 保留真实 MTE3→MTE2 ready |
| frame child>0 | 由已有 MTE3→V→MTE2 路径覆盖，不新增直接通知 |
| 偶数 N 的 slot0 release | 等 N−2；最终 state 写仍须覆盖 N−1 |
| 两个 AIV 生产 P | 一方发布不代表全部字节或贡献到齐 |
| 跨调用 event 复用 | Wait 消费完成不代表其 consumer 的读取已退休 |

## 符号关系的可实现范围

首轮限定结构化循环、uniform 非负长度、静态 stage/slot、规则直接 views 和
可验证的 owner 分区。两端关系组合必须匹配中间实例、guard 和完成强度；
只看最小距离或忽略条件的图可达不能替代指定实例关系的覆盖。

`task=query*K+kv` 在 K 为 runtime 时包含参数乘积；`task//K` 也不是普通的
常量除数仿射关系。不能声称交给通用仿射分析就解决。需要为这类规则时钟映射
保留因子化关系及 `K>0, 0<=kv<K` 合同，利用商余唯一性做受限组合；不展开 K
个余数，也不支持任意非线性整数求解。独立测试乘法溢出、零 K 和 query 边界。

多参数 GM 地址同样保留 shape/stride 结构，先证明直接矩形 view 的范围和分区。
普通外部指针 disjoint 合同在 launch 检查实际区间；间接索引、数据相关 ownership
需要更强可验证合同，否则诊断为不支持或分析未决。

统一模型不要求硬件协议相同：local HardEvent 是每核每有向 bank 八 ID 的二值
事件；cross-core 的参与者、计数、quorum、反向 ack 与 GM 可见性另有 dav-2201
契约。消费通知不等于消费方最后读取完成；两个 AIV 的 UB 始终私有。

## 编译复杂度与退出门槛

运行长度 N 不应进入编译规模。目前 frame 的常量界1024与1048576结构回归
通过；这只覆盖已实现子集，不能外推到尚未实现的多时钟 CV。

必须把下列成本分开统计，不再用“稀疏图+位图”概括总复杂度：

| 部分 | 当前或目标成本及限制 |
|---|---|
| 物理访问前沿 | 访问片段 A、真实冲突 H；密集别名本身可产生二次数量的义务 |
| 已删除的逐缺口选路 | 原有Q次route，每次初始化Vd个cost并扫描至多Ed条边；当前无此项 |
| 固定 DAG 的一次传播 | `O((V+E)·ceil(S/w))`，S为实际源事实数；不包括增量轮次和 guard 操作 |
| 当前条件传播 | U次有效更新，每次扫描节点的G个guard，约 `O(U·G)` 次条件/位集合操作；U和G都不是常数 |
| 当前直接生成 | 对H条义务分组并构造选中协议；增量成本仍取决于真实域/源事实/传播次数，不保证线性 |
| 层次组合目标 | 处理真实边界接口与条件；不随 parent 调用数复制 child 内部图 |

图的构建、周期/条件分割、进展检查和 event 分配另计。一般图的势函数求解也
不保证线性。Python 会放大每次操作的成本，但换语言不能消除这些工作量。

验收首先检查结构：runtime N不增长；独立区域增加时工作量按区域相加；增加
静态操作、环宽和嵌套深度分别测量，不以固定的七模板例子代替规模分析。
记录真实/候选节点、边界接口、guard片段、事实更新、条件比较及选路次数。
新生成器候选路线搜索必须为零；“Graph.path为零”已不足以表示完成重构。

当前性能门槛常用例 p50≤1s/max≤2s，完整 FA p50≤5s/max≤10s，五独立冷进程。
门槛不是合理成本的充分条件。import、CANN、设备计时分开；最终 ASC 的搬运、
buffer、早 release 和等待范围同时验收，不通过省略检查或串行化换编译速度。

## 正确性与性能验收

依次检查 IR/输入合同、访问与生命周期、有限交错反例、CANN 离线编译和设备。
最终 verifier 可共用数学工具，但不能宣称与 planner 完全独立；有限模型检查
不等于任意规模形式化证明。失败分为已证非法、不支持、预算未决。

固定政策下先逐项对比 ASC 的 native 指令、累计顺序、搬运、buffer、wait/release、
CV gate 和填排，再对照独立 AscendC 及 timeline。与同政策参考接近，只能支持
编译器保真结论；还需对照 ops-transformer/CATLASS、安装 ACLNN 和硬件工作量。
不得通过新增 packing、全局 barrier、每 query drain 或扩大缓冲取得表面正确。

硬件分析使用真实 SKU、核数、频率依据和 dtype，条件下界为
`max(各共享资源总需求/对应吞吐上界, 依赖与派发关键路径下界)`。
有效/native FLOPs、GM 请求/HBM 实际流量分开；未知持续吞吐保持参数化。
大规模配置、测量口径与 FA 变体次序见[重构计划](refactor-plan.md)。算子调参仍暂停。

## 源码依据

本轮只读核对 HEAD，未 pull：TileLang `6ba65d2`，ops-transformer `4bbfb3b`，
CATLASS `698379e`，ops-nn `64b41ad`。固定提交报告与当前 checkout 分开溯源。
依据为 [910B 契约](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[算子审查](../../docs/xdsl-npu-operator-source-audit.md)及[FA 目标合同](attention-target.md)。
现有 FA 候选政策的 AIV update-before-softmax 与 ops-transformer arch22 可见的
Vec1-before-Vec2 不同，不能称为逐条复刻专家实现。
