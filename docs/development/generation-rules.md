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
| 真实 Set/Wait 位置和覆盖 | 上述必要依赖 | notification binding；保留原始 ownership edge，可增加 coverage | 域、阶段序、credit | 改捕获/等待位置或 activation |
| 存在域、配对、原式定义性 | 实际位置、祖先坐标、有序 guard | EventDomains；一次建立的 cases/windows | credit、资源和物化 | 改位置、坐标、guard、ordinal |
| 无循环等待 | actual physical step、queue/controller 路由 | queue_order；每层 phase rank 和 child may-block cuts | credit 构造及输出合同 | 改排步、路由、参与者、Wait 位置 |
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

## 1. 捕获与获取

每个 retained need 绑定实际位置：native 的 Set 在 source 之后、Wait 在 consumer
之前；控制端口允许在真实 loop/if 调用的 ENTRY/EXIT 插入。控制端口没有 native
完成里程碑，只是发射位置。Set 捕获该 source engine 的已发射完成前缀；普通本地 Wait 只阻塞目标
queue，S Wait 及 2201 `wait_flag_dev(flag)` 阻塞 controller。SSA use-def 不提供
这些完成关系。只有明确声明 implicit completion 的 native 才能省对应事件；
Vector 等同 pipe 依赖保留实际 pipe fence。

完全相同的端点/domain 前提先 bundle，RAW/WAR/WAW 诊断仍分别保留。累计捕获
只选已经需要且确实更晚发射的 source；复用 acquire 只使用已经存在的更早同
queue acquire。被选通知保留它自身的原始 ownership edge，不把新增 coverage
反过来当作捕获者自己的存储归属。

## 2. physical-step 和局部阶段序

在每个共同祖先从外向内比较 `actual ordinal + stage`。严格前进即进入下一
physical step；相等时继续进入同一 child，或在当前层的真实 direct child / native
端点间加入 publication。负向或不能证明非负的 publication 具体拒绝。

本层只加入真实 queue/controller 的 Wait→Set 派发阻塞关系。native end 之间没有
FIFO 边。所有可能的零 step 关系必须有一个 DAG phase rank；非零关系严格增加
外层 physical step。因此完整命令有良基的分层次序。两个端都先 Wait 对方、
后 Set 自己，会在同 step 形成环，被生成规则拒绝，不能由配对计数放行。

独立通知不共用裸 canonical generation。重复端口的同 step 派发关系直接
来自实际 lexical rank；返回较早 rank 至少前进一个 step。有限 FIRST/LAST
端口的时间范围在本层建立一次，用来排除不存在的短循环回返。相同 native、
actor、phase、实际坐标和 owner/case 的内存分片合成一个 may-block 位置，
域取保守并集、时间范围取凸包；publication 仍逐原始端口检查，不丢配对或坐标。

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
组成一个 credit。相同 native 上的 Set 域及 Wait 域必须分别不相交；不同
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

已经完整的 quorum 还可在相同 physical stage、相同 ActorDomain body、激活域、
坐标与边界 family 内合并前缀：逐 actor 选择最后真实 capture 和最早 acquire，
并保留全部原始 requirements/origins。不能跨不同 C/V body 拼接前缀；后续配对、
进展、cell lifetime 与事件资源约束仍正常生成。有限执行与缺第二 AIV 贡献的反例
见 `tests/sync/test_quorum_prefixes.py`。

## 4. 无稳定 cell 的通知

真正 streaming 地址之间没有物理 slot 可提供上述归纳。它们使用明确的有限使用
上界，或真实 queue handback：旧 Wait 直接阻塞下一 Set 的 source queue/controller，
或经已经需要的同步返回。两次 handoff 的组合只在同一实际 native occurrence 上
相接，检查坐标、存在域和 queue 次序；不建立全程序任意路径证明器。

每个 case 的 radix/window 单独保持。首末选取先检查非空，外层 reentry 连接真实
的最后消费与下一 invocation 的首次发出。无法在有限 bank/backlog 内建立合同，
就报告具体拒绝，不添加 barrier 或反向 acknowledgement 来改变重叠。

稳定 cell 本身不需要 dense ordinal。若 streaming 使用者额外需要某个 child
handoff，其成员和 FIRST/LAST 边界才按需建立；已有普遍配对在域特化后保持。
精确 projection 消去被选中的内部时钟，父层保留自由祖先的 membership，
不从展开的 LAST 算式重新推导其已成立的范围。候选 handoff 只在旧 Wait 与
下一 Set 的 physical-step 区间内，范围由有限 width 决定，不随 runtime 长度展开。
已建立的边界 cases 随 `NotificationProgram.boundaries` 传给分配器，原式、
坐标和成员未变时直接消费。分配新共享关系所需的新增边界在分配器内只建立一次。

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

kernel 返回才执行每个实际异步 engine 的 Set→S Wait drain；配置 ID 和 drain ID
进入同一 bank 预留台账。最终 drain 不参与普通 buffer 的早释放证明。

## 6. 分配和物化

一一对应 ID 改名消费既有生命周期，不重建内存依赖和 progress 图。cell 分片的
共同 ID 来自生成时已经成立的同一个 credit。其他有限通知共享 ID 是分配新增
约束，必须有实际消费回传或 invocation-uniform 互斥，不能仅比较 controller 源码位置。
台账从第一次分配就占用完整 credit 的所有成员；对后来才物化的成员也逐一检查
新共享约束，不能以第一片的有限 handback 授权整个循环 credit。

本地 bank 是 `(physical actor, source engine, destination engine)`，每 bank 八 ID。
mode-2 的十六个 base ID 来自安装 CANN 9.1 的 2201 `GetfftsConfig` 四位编码；
它们在所有 mode 间共用，超出 15 会 alias。使用真实 namespace 不是放宽 solver
预算；不允许 overflow ID、忽略配置预留或静默重排。

同一实际插入点的两个 AIV 使用相同有序指令序列时，代码只输出一份。比较包括
指令种类、方向、已分配 ID 表达式及有序 guard，保留每个 actor 的原有顺序和
重复次数；分析协议名仅作为诊断来源保留，不阻止共享。ID、guard 或顺序不同时
保留原有参与者限制。这里共享的是代码，不合并私有 event bank 或 mode-2 贡献。

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
