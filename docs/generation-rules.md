# 自动同步生成合同

本页对应 `generate_notifications` 的实现。资格和未完成项以
`refactor-progress.md` 及实际构建记录为准；规则论证不替代 ASC/CANN 资格。

## 事实归属

| 义务 | 输入 | 唯一建立者 / 产出 | 消费者 | 失效条件 |
|---|---|---|---|---|
| native effects、真实访问 | 原语、地址、layout、padding、scratch、SPR | native lowering / memory facts | admission、依赖、后端 | 改 native、地址、参与者、layout |
| bounds、初始化、alias/ownership | 输入合同和完整 effects | bounds / admission；逐 actor 的物理分区 | 值与存储分析 | 改 alias、范围、guard、owner |
| 精确逻辑 RAW | 原始程序次序、cell、条件 KILL/UPDATE | logical_valueflow；writer/代次及输入端口 | storage、通知绑定 | 改算法次序、定义、guard、循环 |
| 物理 WAR/WAW、全部最后 reader | 值版本、真实 dispatch、closed packet 规则 | storage_lifetimes；邻接 lifetime 及尾 reader | 生成器的 credit 规则 | 改 slot、dispatch、实际读取、packet 前提 |
| 真实 Set/Wait 位置和覆盖 | 上述必要依赖 | notification binding；保留原始 ownership edge，可增加 coverage | 域、阶段序、credit | 改捕获/等待位置或 activation |
| 存在域、配对、原式定义性 | 实际位置、祖先坐标、有序 guard | EventDomains；一次建立的 cases/windows | credit、资源和物化 | 改位置、坐标、guard、ordinal |
| 无循环等待 | actual physical step、queue/controller 路由 | queue_order；每层 phase rank 和 child may-block cuts | credit 构造及输出合同 | 改排步、路由、参与者、Wait 位置 |
| 消费先于复用 | 原始 ownership edge、完整邻接 lifetime，或明确 queue handback | cell_channels / notification_lifetimes；完整事件生命周期 | bank 分配 | 改 credit 成员、width、ordinal、原始 edge |
| bank/ID/共享 | 已确定生命周期、配置和最终 drain 预留 | structured_resources；实际 ID 台账及新增共享关系 | 物化 | 改 namespace、ID、共享成员、width |
| 输出对应 | 唯一生成计划及本次拥有的 IR | materialize；实际 NotificationOp | scalar 化简、ASC | 移动/删除 native 或同步、改 guard |

初始化准入不是 RAW 的重复证明。RAW 绑定不能代替异步完成。数学化简不授权
提前求值 unsigned LAST/rem；原 guard 的定义性先检查，数学集合推理才可使用其
完整条件。没有 `verified=True` 输入，公共 compile 只接受 DSL/source。

## 1. 捕获与获取

每个 retained need 绑定真正的 native：Set 在 source 之后，Wait 在 consumer
之前。Set 捕获该 source engine 的已发射完成前缀；普通本地 Wait 只阻塞目标
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
origin 必须是绑定自身保留的精确值/邻接 lifetime edge。不同 role、私有 actor
及物理 cell 保持各自 credit。

同一物理 cell、role 和路由的首末/循环分片，可跨不同 native reader/writer
组成一个 credit。相同 native 上的 Set 域及 Wait 域必须分别不相交；不同
native 是 lifetime word 的不同 occurrence。两个 ready consumer 的共同
definition 会使 Set 域重合；多个最后 reader 的共同下一 writer 会使 Wait
域重合，因此它们不会被合并掉。比较使用各端真实坐标，参数 case 仍由域构造负责。

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

## 5. 循环与首尾

循环保留参数化 body。逻辑代次 `g` 在 physical step `j=g+stage` 执行且仅执行
一次；原有 stage guard 处理 prologue、loop 和 epilogue。源地址、slot、guard、
metadata 与捕获值使用相同坐标变换。runtime 长度不展开。

credit 的第一次 writer 在没有先前 reader 时不生成 Wait；release 只有在真实
下一 writer 存在时才成对生成 Set/Wait。这是未 seed 的有限 credit 表达：N=0
没有虚构通知，N=1 的 FIRST/LAST 重合不重复，最后没有无消费者的 release。
不能为 P/L/E 分别建立独立 credit，也不在普通 loop exit 加 drain。

child exit 只结束控制派发，不说明 native 已完成。跨 child 和跨外层 invocation
的依赖仍绑定实际最后 reader/首次 writer。MTE1 最后读 L1 后即可 release，MAD/FIX
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

物化只插入已确定的 native 前后通知及有序短路 guard，保持 native、地址和循环
身份。后端只打印真实 IR。独立有限模型按实际字节、队列、二值消费、quorum 和
controller 阻塞检查输出；可另外提供独立 pre-sync snapshot 检查 native/地址/值版本
保留。该模型是有界反例检查，不外推任意 N 的硬件正确性或设备性能。
