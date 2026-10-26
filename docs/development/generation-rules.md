# 自动同步生成合同

本页对应 `generate_notifications` 的实现。规则论证不替代 ASC/CANN 或设备
验证；验证方法见 [validation.md](validation.md)。

## 事实归属

| 义务 | 输入 | 唯一建立者 / 产出 | 消费者 | 失效条件 |
|---|---|---|---|---|
| native effects、真实访问 | 原语、地址、layout、padding、scratch、SPR | native lowering / memory facts | admission、依赖、后端 | 改 native、地址、参与者、layout |
| bounds、初始化与存储实例 | 输入合同和完整 effects | bounds / admission；group/peer并行独立性是源程序前提 | 值与存储分析 | 改 alias、范围、guard、参与者 |
| 可能别名的源码内存顺序 | 原始次序与实际访问 | AliasRegion；不授权数值KILL或精确producer | 存储、通知；不用稳定cell credit | 改访问、guard、循环或并行合同 |
| 可能冲突的 RAW/WAR/WAW | 实际访问、源控制和可选精确分区 | access_order；输入要求、pending reader 与共同 queue 端口 | 通知绑定与生命周期规则 | 改访问、guard、循环、slot 或 dispatch |
| 私有 packet 的物理复用 | 局部完整写入、内部读绑定及实际 stage 顺序 | storage_packets / storage_word；局部完整 packet 的实际邻接 | access_order | 改 native 覆盖、slot、dispatch、packet 前提 |
| 稀疏私有 cell 的完整 ownership round | 原始 body 接口、全部实际访问、相同激活域与本地派发 | cell_ownership；完整 writer/reader 轮次及被替代的原始 requirements | 通知生成、有限 bank、物化 | 改访问、guard、参与者、reader 集合或本地排步 |
| 真实 Set/Wait 位置和覆盖 | 上述必要依赖 | notification binding；保留原始 ownership edge，可增加 coverage | 域、阶段序、credit | 改捕获/等待位置或 activation |
| 实际激活、配对、原式定义性 | 实际位置、祖先坐标、有序 guard | EventDomains；已配对的真实发生域 | 完成覆盖、credit、资源和物化 | 改位置、坐标、guard、ordinal |
| 复用所需精确成员及 FIRST/LAST | 保留的配对协议、真实激活及循环范围 | EventDomains.for_lifetimes；精确 cases/windows | handback、资源和物化 | 改激活、循环或生命周期政策 |
| 无循环等待 | actor-local epoch、queue/controller 路由 | queue_order；每层 phase rank 和 child may-block cuts | credit 构造及输出合同 | 改排步、路由、参与者、Wait 位置 |
| 消费先于复用 | 原始 ownership edge、完整邻接 lifetime，或明确 queue handback | cell_channels / notification_lifetimes；完整事件生命周期 | bank 分配 | 改 credit 成员、width、ordinal、原始 edge |
| bank/ID/共享 | 已确定生命周期、配置和最终 drain 预留 | structured_resources；实际 ID 台账及新增共享关系 | 物化 | 改 namespace、ID、共享成员、width |
| 输出对应 | 唯一生成计划及本次拥有的 IR | materialize；实际 NotificationOp | scalar 化简、ASC | 移动/删除 native 或同步、改 guard |

初始化准入不是 RAW 的重复证明。RAW 绑定不能代替异步完成。数学化简不授权
提前求值 unsigned LAST/rem；原 guard 的定义性先检查，数学集合推理才可使用其
完整条件。没有 `verified=True` 输入，公共 compile 只接受 DSL/source。

bounded dense reader prefix 的空间联合可按 count 的有限上界保守覆盖完整 producer
tile，但 native 字节范围不变，reader family 保留原始 count。捕获实际 LAST 前，
按从外到内已解析的祖先 FIRST/LAST 坐标特化 child count；不使用未选祖先或
上界代次代替真实最后读取。计算 LAST 的 Expr 同样解析 selected ancestor；
canonical occurrence 已有的 j-d 祖先平移不重做。变化的 child 通过共同控制端口组合，不要求该 family 精确闭合。

精确分区按真实地址周期识别物理槽。对于步长可证明为常数 `a` 的
`(a*i+b)%m`，周期为 `m/gcd(abs(a),m)`；步长为 0 或 m 的倍数时周期为 1。
不能按余数表达式的语法将同一物理地址复制为多个独立槽。未知映射继续使用
AliasRegion，原始算术定义性和范围检查不由周期推理代替。

私有 scratch 的独立调用可以包含未访问槽。只有原始 child 接口没有输入读取、
具有完整排序写入及 reader 链，且该 cell 的值不逃逸这组 packet 时，才允许
按实际 dispatch 连接调用之间的物理顺序。分析保留形成队列端口之前的原始接口，
不从端口名字或出口位置反推数据闭合。空路径仍通过真实入口/出口端口传递
旧 ownership 和 pending reader；这些 forwarding RAW 与 WAR/WAW 一起按
storage word 连接，packet 内的源读取关系保持。数据闭合不授权队列端口使用
native cell 的单 ID 归纳，端口仍须取得实际 handback 和有限资源证明。

字节 bounds 组合已经验证的原轴范围、真实线性化和 native 子视图；真实 root
容量可容纳整个原 tensor 时，不重新求解等价的扁平地址边界。容量关系不成立
或未知时仍验证实际 slice 的字节边界，不能用轴 metadata 代替真实 allocation。

## 1. 捕获与获取

每个 retained need 绑定实际位置：native 的 Set 在 source 之后、Wait 在 consumer
之前；控制端口允许在真实 loop/if 调用的 ENTRY/EXIT 插入。控制端口没有 native
完成里程碑，只是发射位置。Set 捕获该 source engine 的已发射完成前缀；普通本地 Wait 只阻塞目标
queue，S Wait 及 2201 `wait_flag_dev(flag)` 阻塞 controller。SSA use-def 不提供
这些完成关系。只有明确声明 implicit completion 的 native 才能省对应事件；
Vector 等同 pipe 依赖保留实际 pipe fence。

Scalar EXIT 端口可保留其全部来源的同步退休前缀；dispatch 逐来源、逐 actor
核对实际 native 的 Scalar 与 implicit-completion 合同。只有来源均满足该合同
才允许沿用同步退休，不能从 EXIT 位置本身推出完成。空路径仍通过入口获取
需要的旧前缀；原接口没有输入读取的同步尾部观察不新建输入 reader。真正的
head reader、异步 reader 和未知来源仍保留。ENTRY、Scalar store 及 Vector/Cube
端口不从这条规则取得同步完成。

完全相同的端点/domain 前提先 bundle，RAW/WAR/WAW 诊断仍分别保留。累计捕获
只选已经需要且确实更晚发射的 source；复用 acquire 只使用已经存在的更早同
queue acquire。被选通知直接引用自身的原始 anchor binding，不把新增 coverage
反过来当作捕获者自己的存储归属。

通用 cell credit 按这些原始 binding 的完整物理 ownership 集合和 role 分组。
先按 Set/Wait、participant、native 操作建立全部保留片段的真实激活索引。
同一位置存在可重叠的多个后继/前驱时，相关片段各自保留独立 credit：保守访问
前沿可能同时保留相邻边和跨条件 child 的更晚边，后者不能冒充相邻 lifetime。
其余无重复端点的相邻片段才按同一物理 word 归纳共享；不同 credit 如需复用 ID，
仍须由资源规划取得真实消费回传证明。条件未知时保守分开，不新增同步或 loop drain。
共同位置的激活交叠只建立一次，分组直接消费该事实。

已取得的完成前缀按实际 route、完整激活类、capture/acquire 的层级发射坐标
建立查询索引。同一个发射 word 只允许最后 lexical 位置变化，其余 scope、
坐标、stage、参与者相对位置及控制边界身份均保留。查询在目标实际 lexical
位置处分段，每段检查相同的完整有序关系；unknown 不作为已排序。
二维区间索引只返回一个已经存在的真实 capture/acquire pair，并保持原来的
最早匹配选择。动态条目按本 word 的实际 acquire 顺序加入；查询不扫描该 word
的全部历史，不把两个 pair 的捕获最大值和获取最小值拼成一条新通知。
各 word 分别判断激活包含关系，任意多种条件/调用域一般仍可形成真实稠密关系。
这些完成覆盖事实不提供 event 消费或 ID 共享资格。

配对域建立后，完成前缀还通过实际 capture/acquire 组合。capture 冻结其之前
已阻塞本 queue/controller 的 Wait 所取得的来源；后来才执行的 Wait 不能回填。
native issue 本身不产生完成事实。同锚点不同动作不因位置相等取得先后关系。
同一实际 capture 在不同需求中可能采用不同逻辑代次。按 actor、源 queue、真实
操作、插入位置和循环路径对应后，仅用已存在坐标间的精确常量平移实例化协议；
sources、consumers、activation 和 metadata captures 同时替换。新实例仍引用
原协议，排除该协议时同时排除全部代次。它不是新参数 case，也不产生硬件动作；
每条原始 binding 在每个 case 中仍独立检查一次。非单位平移或无法对齐时不扩展。
有限 capture/acquire 关系保留各自的原整数条件。查询只要求在原始需求的完整
激活前提下必然取得目标事实。若实际 seed 和路径上全部条件已经是调用方的显式
合取前提，直接沿该路径传播，并保留每个 capture/acquire 的独立条件。按路径长度
归纳即可得到结论；没有 seed 的环不能自证，也不能拆开或条件选取尚未成立的一侧。
其余相关整数及分支关系仍通过既有转移的 Horn 规则建立结论，不展开全部路径。
单输入范围只是带条件的转发边，不另建状态；多输入合流与真实动作使用 Bool
可达状态，避免引入无关整数。证明保留真实支持及完整前提的 owned implication；
规则本身不是已取得的事实。
层级位置索引将不同 sibling region 的内部获取合为共享出口范围；查询只进入
与目标位置重合的 child，开放分支的激活仍保留在每个真实获取上。range 节点
不成为新 gate，空 child 不提供完成事实。有限规则也不把这些共享范围当作已经
完成 child 内部义务的无条件摘要。

机制选择前，每条普通需求先拥有真实 source/consumer binding；CopyGroup 另保留
明确授权的共同 ready 前缀。ownership 替代的跨轮次义务由原始 lifetime 覆盖，
不为它们构造临时通知。selection 及 coverage 共同持有同一份原始 binding；
合并和完成覆盖不替换该身份。接纳一个候选通知前逐 binding、逐参数 case 检查；只为相同真实端点
共享查询，不能用代表 source 替代其他 writer，也不能合取互斥 writer 的激活条件
制造空域。候选图只提出真实支持前沿，查询排除自身及全部已覆盖候选。
成功方案含有新支持时，再仅用已接纳动作尝试完整覆盖，避免 solver core 夹带
额外候选。全部原始义务覆盖成立时，先接纳证书实际使用的全部协议，再发布覆盖；
否则接纳原候选。已接纳协议不再参与覆盖选择，保留集合只增不减。
静态 consumer 位置不代表循环代次间的动态 acquire 顺序；所有顺序仍由完整
坐标、guard 和真实 capture/acquire 证明，不需要后续删边或支持固定集合。
来源种子正向闭包与目标反向闭包之外的规则不参与该次证明；没有真实 seed
的环不能建立完成事实。cell credit、消费、进展和资源许可随后在保留集合上建立。

端点配对先直接比较真实激活，不要求该激活在所有祖先循环上具有周期或 dense
rank；仅在配对本身需要参数特化时建立对应精确 case。完成覆盖之后，保留的非
稳定 cell 协议才建立生命周期所需的成员窗口。已配对域的精确特化保持原等价
事实，不重复证明未改变的配对。这个顺序不授予队列端口单 credit：转发端口仍须
取得完整消费、重入和有限资源证明。无法构造精确窗口时，仅下述任意间隔消费
规则可独立取得宽度 1；不能直接把空 rings 当作宽度 1。

闭合私有轮次在已检查的同参与者、同读取队列及实际 stage/lexical 顺序下，一次
选定最后 reader。退休直接消费原始 RAW binding：覆写许可放在真实 writer 前，
返回许可放在最后 reader 后。共同 ready 可以晚于 writer，不能把覆写许可随之
移到组末尾。全部 reader 和完整 effects 仍属于 lifetime；最后 reader 是同队列
顺序已成立后的退休里程碑，不删除其他 reader。

同一次完整 writer 的多个私有 cell 可以共享一个 persistent free credit，但所有
原始 member lifetime、round 和 binding 继续保留。必须是相同 actor、相同 pipe
路由、完全相同 acquire，且 release 的真实 scope、guard、坐标、family 和代次
绑定完全一致；统一在同 pipe 实际 stage/lexical 最后的 reader 后归还。该 Set
捕获 reader 完成前缀，下一次共同完整 writer 原本就要同时取得全部 cell。
不同参与者、quorum、激活或 writer 不合并，不增加容量，也不重新建立访问事实。

以上规则也适用于已经完整的跨核 quorum：每个 producer 的捕获都覆盖所需
source，每个 receiver 的已有 acquire 都先于所需 consumer，才能复用。
相同 Stage、ActorDomain 或共同写入某个 cell 都不授权合并独立完成边界；
不能为了节省 ID 将多个独立通知改成「最后 capture → 最早 acquire」。

## 2. actor 本地时钟和局部阶段序

源逻辑代次与 actor 发射时钟分离。在每个共同祖先域中，端点的本地 epoch 为
`mapped ordinal + local offset`；不同 actor 的裸 epoch 不构成先后关系。
生成器为最多三个实际 actor 推导常量原点 beta，仅用于进展证明：

```text
proof_epoch(p) = local_epoch(p) + beta[actor(p)]
k + beta[destination] - beta[source] >= 0
```

k 是实际配对 publication 的 epoch 差下界。常数差直接使用；有限符号差从
已有整数事实和实际 guard 建立下界，未知时明确拒绝。用有限差约束松弛求解
原点，不枚举运行时代次、不选择执行排步。不可满足时报告实际端点形成的原点环。
本地 queue/controller 的 Wait→Set 关系及 child may-block transfers 使用相同原点；
忽略 child 的跨 actor 项会破坏循环权重的相消，因此不能省略。

平移后的严格正距离边前进，零距离边必须形成一个 phase DAG。
同 actor 的真实 clock/lexical issue 顺序保持；native end 之间不增加 FIFO 边。
两个 controller 均先 Wait 对方、后 Set 自己仍被拒绝，配对数量不能放行等待环。
原点不进入设备代码，不改变 Set/Wait 发射位置、完成前缀或物理 buffer 使用。
例如 Cube products@1 → Vector import@0 可通过 Vector 原点平移获得证明。

独立通知不共用裸 canonical generation。重复端口的同 step 派发关系直接
来自实际 lexical rank；返回较早 rank 至少前进一个 step。有限 FIRST/LAST
端口的时间范围在本层建立一次，用来排除不存在的短循环回返。相同 native、
actor、phase、实际坐标和 owner/case 的内存分片合成一个 may-block 位置，
域取保守并集、时间范围取凸包；publication 仍逐原始端口检查，不丢配对或坐标。

Wait→Set 的可能阻塞关系按实际 actor/engine 路由建立共享区间索引，不枚举
全体端点对。重复端口共用 lexical 索引；有限选择按实际时间表达式和域范围凸包
分组，保留短循环中不存在的回返及正 step 判断。普通 Wait 只查询目标 engine，
Scalar/cross Wait 查询本 actor 的全部发布队列。同一 child 的内部关系由 child
接口提供，父层仅为再次进入该 child 保留正 step 关系。
区间树的内部顶点只分解可能阻塞关系，不代表发射、完成、消费或存储里程碑，
不成为 child 对外 cuts。原点求解只保留跨 actor 和负的同 actor 约束；非负的
同 actor gate 不约束原点，负的同 actor publication 仍须拒绝。区间索引的
工作量取决于路由端点及不同有限时间类；不同有限选择一般不能合并为线性序列。

child 只导出 actor/queue/controller 输入到输出的 **可能阻塞**关系。父层不导入
其内部图，也不在父查询中读取隐藏路径。正 step 内部关系在 whole invocation
接口中保守合并；可能的条件关系取并集。这是充分准入规则，可能拒绝没有统一
阶段序的特殊条件排步；未知不会变成安全。

这些 may-block cuts 只用于无死锁准入。它们绝不能证明 native 完成、reader
退休或 event 消费。阶段 rank 也不能用于 ID 共享。

## 3. 稳定物理 cell 的 credit

前提：cell 的物理 origin 在 kernel group invocation 中不变；如果 cell 是
某个 dense 子流的联合区域，通知不能在该被联合的内部时钟上重复使用它。
origin 必须是绑定自身保留的精确物理访问/邻接 lifetime edge。控制端口不使用
此规则，必须建立实际 queue handback。不同 role、私有 actor
及物理 cell 保持各自 credit。

同一物理 cell、role 和路由的首末/循环分片，可跨不同 native reader/writer
组成一个 credit。每个片段先对整个同 ownership/role/route 集合检查重复端点；
存在可重叠 fan-out/fan-in 的片段不得与其它 phase 直接组成相邻 word。
其余片段在相同 native 上的 Set 域及 Wait 域分别不相交；不同
native 是 lifetime word 的不同 occurrence。两个 ready consumer 的共同
definition 会使 Set 域重合；多个最后 reader 的共同下一 writer 会使 Wait
域重合，因此它们不会被合并掉。比较使用各端真实坐标，参数 case 仍由域构造负责。

多个精确 cell 可组成共同释放的 ownership 集合：capture 必须在全部旧 reader
发射之后，acquire 必须在所有下一 writer 之前，且每个被合并的 cell/role 在
steady 和 selected boundary 的候选片段中保持同一完整集合。只在边界合成的
集合不取得内部各 cell 的既有 credit。该集合沿用完整物理 lifetime 的归纳，
不是按 phase rank 给不同存储复用 ID；原始 edges、参与者和路由均保留。

每个 cell credit 使用一个本地二值 ID；完整的跨 AIC/AIV credit 使用一个 mode-2
ID，并保留双方实际参与者的原始 edges。复用理由是对物理 lifetime word 的归纳：

* RAW ready 的旧 Wait 先于该 cell 的 reader。下一 writer 必须等全部旧 reader
  退休。因此下一次同 credit 的 writer Set 先于旧 Wait 是不可能的。
* WAR release 的旧 Wait 先于相邻下一 writer。后续 reader 必须取得它的真实新
  定义；中间 RMW/WAW 仍沿完整值/存储关系连接。因此下一 release Set 必须在
  旧 Wait 之后。
* 没有 reader 的 WAW 使用相邻 writer 链；不把它与 RAW/WAR role 无条件合并。

以上归纳使用完整 data/storage coverage 和已准入的阶段序；不以 Set/Wait 同 ID
本身为证明。不同存储不能仅凭 phase rank 共用 ID。

两个 AIV 只有一份 body。UB 和本地 bank 仍逐 actor 私有；mode-2 ready 要两个
receiver 都消费，release 要两个实际 reader 都贡献。少一个成员由域准入拒绝。

不同 Cube native 可以对应两个 AIV 的分片。只有 Vector 端点相同、Cube 端点的
激活域和实际坐标相同、且在每个原始 requirement 前提下都能比较 Cube 发射顺序时，
合并为一个 quorum：发布选最后 Cube native 后的累计完成前缀，获取选首个 Cube
consumer 前的 queue Wait。两个 AIV 的原始 lifetime edges 全部保留；配对、阶段序
和 cell credit 仍检查合并后的实际位置，无法比较顺序或缺少参与者仍拒绝。

quorum 的完成覆盖与物理 ID 共享是不同的变换。前者遵守第 1 节已有位置的
复用规则；后者不改变 Set/Wait 的位置和次数，遵守第 6 节的事件资源合同。
独立发布、完整 quorum、缺第二 AIV 贡献及 ID 共享反例见
`tests/sync/test_quorum_prefixes.py`。

### 3.1 跨空调用的私有 cell 持久 free credit

`cell_ownership` 对以下受限但完整的轮次建立独立合同：

- 物理 origin 在 kernel invocation 中固定，存储为精确的 actor 私有 cell，
  不使用 alias region、streaming 联合区域或条件分区冒充独占 cell。
- 每轮只有一个不读取旧 cell 的完整 writer；全部实际 reader 位于同一 pipe。
  writer 与所有 reader 位于同一实际直接循环，使用其真实 ordinal，
  具有相同的分支路径。跳过共同分支时没有半轮访问。
- 实际 dispatch 中 writer 先于全部 reader，全部 reader 又先于本 cell 周期后的
  下一 writer。周期只取该 cell 的地址周期，不构造不同 ring 的全局 LCM。
  不同 child invocation 是完整的 controller 调用。
- 该 cell 的全部 native 访问均被这些轮次覆盖，没有入口状态、逃逸读取或缺失
  reader。保留每个 reader 的原始 ready requirement；剩余 RAW/WAR/WAW 由
  真实轮次 ownership 承担，原始 needs 仍用于诊断和 coverage 对应。

由实际 ready 端点绑定每轮 acquire 和最后 reader 之后的 return。存在域和
原式 guard 定义性仍检查；不求上一次活跃调用的 dense rank 或空域上的 LAST。
同 pipe 多个 reader 选实际最后发射点作累计 capture，不假设 native 完成 FIFO。

每个 cell 在同一有限 bank 内占一个真实二值 ID。kernel 入口在 body Wait 前
发布 free seed；每次真实 writer 前取得它，全部 reader 退休后归还它。
空调用或未访问槽不发通知，credit 原样保留。kernel 出口回收最后的 free credit，
**先于目标队列的最终 queue→S drain**，保证下一次 kernel invocation 前已消费。
seed 不初始化数据；原有 bounds、初始化和 ready 合同继续生效。

二值消费的归纳是：当前 writer 先取得 free credit，reader 又取得该 writer 的
真实 ready，因此 return Set 必须在当前 free Wait 消费之后。下一 writer 取得
return 才能覆盖，故全部旧 reader 的 ready Wait 也先于下一 ready Set。
每轮保持一个 free token，初始化和最终回收使 kernel 边界的余额为零。
这些是真实 token 和字节依赖的结论，不能由数据闭合标记或阶段 rank 代替。

进展有额外的充分前提：保留的每个本地通知，必须在实际 controller 上先发出
publication、再发出 acquire。生成器保留这些实际位置的 OrderingWitness。
新 free Wait 的来源也总是较早发出的实际 return，或入口 seed。
于是本地队列不依赖未来 controller 发射才能退休；cross Wait 已阻塞 controller，
后续本地 native 之前的 cross gate 必须已消费。跨 controller 进展仍由原来的
queue_order 检查。无法建立这个本地前提时明确报告支持边界，不用新 credit
跳过进展检查。

持久 ID 与配置、普通通知及最终 drain 使用同一资源台账，普通共享不能借走
仍有 token 的 ID。超出八个本地 ID 时明确拒绝。两个 AIV 的私有 UB 和 bank
分别处理、相同指令可共用一份 body；本规则尚不自动生成共享 GM 的 mode-2
持久 credit。多 writer、不同 reader pipe、未闭合的嵌套 family 和变化 quorum
继续保留原来的必要检查，不能由此规则取得支持。
声明旧态读取的 RMW native 也不进入此规则；某个边界代次 clear 不表示每轮
都能独立覆盖，更不能用 free seed 代替递推状态或低精度 snapshot 的定义。

## 4. 无稳定 cell 的通知

真正 streaming 地址之间没有物理 slot 可提供上述归纳。它们使用明确的有限使用
上界，或真实 queue handback：旧 Wait 直接阻塞下一 Set 的 source queue/controller，
或经已经需要的同步返回。消费 origin 经实际 capture/acquire 快照组合，不设
固定 relay 深度。每个实际 Set 的快照仍只取得它之前的 Wait；晚于 capture 的
交接不能回填。完整源/目标存在域、真实 actor clock、FIRST/LAST 和每个参与者
分别保持；没有新增 barrier 或 controller gate。

已建立的消费事实导出到实际取得它的 Wait，保存完整前提和数学证明，再沿真实
queue/controller 传递。相同前沿通过共享位置范围索引和不可变证明复用；新的
开放边界仍在同一图中组合。完成和消费有不同的 origin 类型，完成覆盖不提供
事件复用资格。布尔条件和证明用 DAG 保存，编码及遍历不依赖 Python 递归深度。

单 publisher/consumer、同 actor 的本地通知另有稀疏单 credit 规则。共同 clock
须在真实端点上为单位平移；publisher 不带 family，consumer 只允许已配对的
单次选择坐标。任取两次实际发布，以首个不同共同 clock 分情况：之前的坐标
相等，该坐标增加 `1+gap`（`gap>=0`），后代坐标独立。metadata captures 随坐标
替换，不能假设两次读取的值相同。在新旧端点的完整 activation/ranges 下，
证明旧 Wait 消费先于新 Set；查询排除该通知全部动作。真实 relay 仅取旧调用和
相邻返回 horizon，目标间隔仍任意，必须由真实 queue 顺序跨越。各层证明覆盖
全部严格后继及重入，故任意时刻最多一个未消费发布，虚拟 lane 恒为 0。该规则只在
精确窗口表示不支持时使用，不改变已有合法环的首选政策；证明失败继续拒绝。

环政策与规则能力分开：先用有限次数或 native 边界 handback 构造首选协议；
该流自身无法在容量内构造时才直接用 queue-composition。已有合法首选环的
额外缩环只作为受压时的有限资源选项。每个选项保留原语、动作位置和激活，
冻结自己的 phase/handback；纯编号阶段联合选择并优先首选协议，不能回调
域分析。选择次序、工作上限及当前支持子集见[事件资源](event-resources.md)。

每个 case 的 radix/window 单独保持。首末选取先检查非空，外层 reentry 连接真实
的最后消费与下一 invocation 的首次发出。无法在有限 bank/backlog 内建立合同，
就报告具体拒绝，不添加 barrier 或反向 acknowledgement 来改变重叠。

稳定 cell 本身不需要 dense ordinal。若 streaming 使用者额外需要某个 child
handoff，其成员和 FIRST/LAST 边界才按需建立；已有普遍配对在域特化后保持。
精确 projection 消去被选中的内部时钟，父层保留自由祖先的 membership，
不从展开的 LAST 算式重新推导其已成立的范围。端口只选择所需 FIRST 或 LAST，
并直接携带这份 membership，不在所选坐标上重建一份随即被覆盖的原 activation。
窗口 count 保留原表达式经参数代换后的结构，周期信息不会先被 quotient 展开丢失；
数学查询另用分析视图，原式求值与配对准入保持。候选 handoff 只在旧 Wait 与
下一 Set 的 physical-step 区间内，范围由有限 width 决定，不随 runtime 长度展开。
已建立的边界 cases、位置、消费前沿和 region transfer 随 `NotificationProgram.handbacks`
交给资源规划器继续使用；两阶段引用同一 retained program 的消费分析。原式、
坐标和成员未变时直接消费。新共享关系所需的新增边界在资源规划阶段只建立一次，
编号分配不查询域或 handback。

如果旧 Wait 与返回发布对应同一个实际 native 或控制位置，优先复用已建立的
坐标映射和配对，将返回端直接绑定到该 occurrence，不恢复整个 child 的成员
FIRST/LAST。对应相等条件仍由真实复用域覆盖。可选 relay 的额外精确边界不支持
时只放弃该候选；必需通道、配对和生命周期均保留，无可用回传仍具体拒绝。

## 5. 循环与首尾

循环保留参数化 body。逻辑代次 `g` 在 physical step `j=g+stage` 执行且仅执行
一次；输出为一个时钟循环，各stage用同一存在条件覆盖全部合法实例。源地址、slot、guard、
metadata 与捕获值使用相同坐标变换。runtime 长度不展开。

credit 的第一次 writer 在没有先前 reader 时不生成 Wait；release 只有在真实
下一 writer 存在时才成对生成 Set/Wait。这是未 seed 的有限 credit 表达：N=0
没有虚构通知，N=1 的 FIRST/LAST 重合不重复，最后没有无消费者的 release。
不能为不同循环边界分别建立独立 credit，也不在普通 loop exit 加 drain。

child exit 只结束控制派发，不说明 native 已完成。变化空 child 的外层连接消费
实际队列入口/出口端口，Set/Wait 对同一调用的前缀建立完成与获取；不会反求
最后一次活跃 child。非空稳定 loop 可使用窄的 native reader/writer。MTE1 最后读 L1 后即可 release，MAD/FIX
可以继续；MTE2 最后读 GM 后即可 release，Vector 可以继续。

条件访问只有在全部激活和坐标选择输入在循环入口可求值时，才可能保留窄的
guarded 端点；这包括 child count、FIRST/LAST 选择和不透明 queue 边界，
不能只检查 if 谓词。还须满足该 cell 每轮都有无条件排序写入，或通过整数域
证明访问集合至多在一个 ordinal 激活。后者查询两个不同合法 ordinal 能否
同时激活，不匹配 `i==N-1` 等表达式形状。存在空洞的多次条件写入和晚定义
metadata 继续使用共同 queue 端口，保留跳过分支时的平衡及真实 S 定义位置。

kernel 返回才执行每个实际异步 engine 的 Set→S Wait drain；配置 ID 和 drain ID
进入同一 bank 预留台账。最终 drain 不参与普通 buffer 的早释放证明。

## 6. 分配和物化

一一对应 ID 改名消费既有生命周期，不重建内存依赖和 progress 图。cell 分片的
共同 ID 来自生成时已经成立的同一个 credit。其他有限通知共享 ID 是分配新增
约束，必须有实际消费回传或 invocation-uniform 互斥，不能仅比较 controller 源码位置。
台账从第一次分配就占用完整 credit 的所有成员；对后来才物化的成员也逐一检查
新共享约束，不能以第一片的有限 handback 授权整个循环 credit。

完整的前后两段可用精确 window 选择最后 Wait 与首次 Set，证明消费回传后共享。
相同外层 invocation 内的两个 child burst 还须证明本次前后关系，以及每层下一次
invocation 的最后消费到首次发布；空、短和稀疏 window 不假借稳态存在性。
同一循环、相同 count 的窗口允许成员集合包含关系，共同调用域取较宽窗口，
各个真实端点仍保留自己的 guard。规范时钟到实际端点只能是单位平移；选中的
child FIRST/LAST 还须对该 child 时钟不变。内层 burst 的边界使用自身的精确
FIRST/LAST；外层重入检查两个方向的最后消费到首次发布，覆盖子窗口缺席的情形。
共享候选先检查端点的发射先后关系，再建立实际 queue/controller 消费回传；
发射先后本身不能授权事件共享。无法建立这些前提时继续使用独立 ID 或明确拒绝。
mode-2 在完全相同的实际发布/等待位置、参与者和坐标映射下，还允许互斥槽位
子集组成同一个 FIFO；每对成员分别检查源端和接收端的 guard 互斥，保留各自
完整配对和 quorum。台账仍限制同一 ID 全部完整 credit 的积压。

mode-2 首先允许基于完整消费的时间复用：同路由/参与者的每个旧接收者都经
已经需要的一条完整 quorum 反向回传，先消费再进入每个新发布。各层 LAST/FIRST
及重入均须覆盖；同号的全部完整 credit 两两不干涉，积压取最大值。这个合同
没有新增 Wait 或完成约束，不用 FIFO packing 的积压求和代替。具体边界见
[事件资源](event-resources.md)。

mode-2 计数器还允许一种独立的 FIFO 共享规则：两个完整通道的实际发送者和
接收者集合、source pipe、clock、坐标、激活域、边界 family 和 case 一致，
且逐发送者的发布顺序与每个接收者的消费顺序严格一致。证明使用各端已建立
的实际 issue 关系和每个发送者同 source pipe 的计数贡献 FIFO；不把普通
native end 当作 FIFO，也不移动、合并或减少 Set/Wait。

该规则同时接受一个 AIC 发布给两个 AIV、两个 AIV 发布给一个 AIC。mode-2
按发送者/接收者保留贡献流；AIC 的第 k 次 Wait 消费两个 AIV 各自的第 k 份
贡献。两个发送者可以以不同速度前进，快发送者的后续贡献不能替代慢发送者
的缺席贡献。逐端同序与相同激活域使这些贡献对应同一完整消息，不额外插入
AIV 间的轮次 barrier，也不丢掉任一 AIV 的原始发布或完成前缀。

同号新增的是 counter 效果 FIFO，不能宣称不同 ID 的 Set 效果原本已按发射顺序
完成。当前整组规则核对实际 clock 单位平移和各参与者完整 word，借同 source
pipe 的单调 native 前缀及有限积压建立新增 FIFO 的局部进展；见[事件资源](event-resources.md)。

候选不固定成一份 first-fit 分区。额外结构候选仍经过相同整组检查；不同 clock 的
叶子 word 仅在每个 publisher/receiver 的真实首末端口均证明整次调用顺序时拼接。
积压继续对完整 credit 求和，不用循环控制退出证明通知已消费。
不同 FIFO 子组只有全部跨子组 credit 对取得完整消费回传/整次调用互斥，才可按
子组内部求和、子组之间取最大值组合。分配和最终组检查消费同一份冻结组合。

分配候选 ID 时，对全部已占用与将占用的完整 credit 求积压上界之和；同一个
物理 credit 只计一次，总和不得超过 15。未知积压不是 0。容量检查覆盖之后
才物化的 credit 成员，不只检查当前通知。参与者、激活域或任一发送者/接收者
的消息顺序不一致时不能套用该规则；缺少一个 AIV 贡献仍是协议错误。

本地 bank 是 `(physical actor, source engine, destination engine)`，每 bank 八 ID。
mode-2 的十六个 base ID 来自安装 CANN 9.1 的 2201 `GetfftsConfig` 四位编码；
它们在所有 mode 间共用，超出 15 会 alias。使用真实 namespace 不是放宽 solver
预算；不允许 overflow ID、忽略配置预留或静默重排。

同一实际插入点的两个 AIV 在规划阶段比较完整虚拟动作序列，保留指令种类、方向、
资源/phase、guard、顺序和重复次数；镜像私有资源取得编号相等约束后输出一份代码。
编号后不再因整数恰好相同产生新的合并机会。这里共享的是代码，不合并私有
event bank 或 mode-2 贡献。专属政策和整组 FIFO 的支持边界见
[事件资源](event-resources.md)。

`N.Vector() as lane` 的 lane 就是 runtime SID。共同动作直接在共享 Vector body
执行；只有单 actor 动作才在物化时生成一次 SID 控制，且该控制先于 guard 和
事件 ID 求值。最终 NotificationOp 的 eligible participants 与实际 SCF 控制取
交集，因此物化后的指令采用所在 family 的参与者，不让后端再加第二层 SID 判断。

物化只插入已确定的 native 前后、loop/if ENTRY/EXIT 通知及有序短路 guard，保持 native、地址和循环
身份。随后一次显式投影建立独立终端模块、native 与 SSA 范围映射，保留已准入
的静态 geometry/packet；逻辑事实仍指向原逻辑模块。终端 `lower_execution`
将投影后的 count/stage 表示降低为实际 arith/scf，按每个 actor 的既定 issue
order 映射控制与地址，移动终端 native，保持双 AIV 共用代码。后端只接收
最终 `ExecutablePlan`，不读取原逻辑内存、控制或地址证明表达式。
整数化简复用支配定义、源循环联合范围和实际 runtime/actor/loop/branch，
折叠 min/max、仿射冗余及比较，并外提目标域内求值合法的不变量；
除法定义性及全部 signed 中间量在目标域未知时不外提。
原式定义性先于化简准入，未知或可能溢出的区间不支持折叠。

`simplify_conditions`使用实际loop、祖先和分支前提保持`g`与`g-N`关联，合并
连续指令的共同条件。仅当标量在共同入口完整可求值时才移动计算；通知专用
二次幂余数可改为全定义的位掩码，地址及loop header依赖的原式保持不变。
同种类/路由/参与者的连续事件通知在分支互斥时合并执行域和event ID表达式，
重叠分支仍按原顺序贡献各次 Set/Wait。诊断协议来源全部保留，不重新选择同步。
event选择式在公共guard的剩余域内生成，内部不重复判断共同条件。
有限bank slot的互补选择可化简为无条件指令，不枚举runtime迭代或循环阶段。

同 pipe fence 使用独立的幂等规则：在仅含纯整数 SSA、条件和通知的 packet 中，
同路由、同 eligible participants 的连续 fence 可将执行域取并集，即使条件重叠。
两次 capture 之间没有新的 native 或其他通知，因而覆盖同一个完成前缀；
每条实际执行路径上 `F;F=F`，并集保留原来至少一次 fence 的全部路径。
SID 条件仍属于执行域，不能把一个 AIV 的完成当作另一个的完成。
native、循环、地址/配置及未知 effects 都结束 packet，不跨边界借用完成事实。
合并后的 NotificationOp 保留所有 protocol 来源，`removed_fences` 记录静态合并数；
事件贡献次数、native 顺序、地址和循环身份保持。该规则不依赖 CANN 自动同步。

后端只打印最终IR，每个pipeline/family只有一个物理时钟循环、一份stage body；
相邻同offset的stage共享存在guard。ASC中无阶段枚举、阶段参数、stage lambda
及独立的序言/稳态/收尾/短循环路径；N=0和短循环服从同一执行规则。首末轮、
event slot和其他真实协议条件继续使用实际IR，不把循环退出当完成。
launcher仅输出kernel调用及ABI返回，不生成尺寸/倍数/溢出/指针/对齐/alias
运行时检查，不在其他入口补写检查。编译期仍验证输入合同声明、原始有序
定义性和静态地址/同步前提；调用方满足kernel声明的输入与合法索引求值合同。
独立有限模型按实际字节、队列、二值消费、quorum 和
controller 阻塞检查输出；可另外提供独立 pre-sync snapshot 检查 native/地址/值版本
保留。该模型是有界反例检查，不外推任意 N 的硬件正确性或设备性能。
