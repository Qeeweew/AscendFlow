# 事件资源的事实、政策与编号

事件 ID 是硬件资源映射，不能代替 buffer slot、Stage、消息身份或 ownership。
一个完整 physical credit 的全部成员先取得共同资源身份；不同 credit 的复用
仍需独立证据。逻辑数据、物理存储和通知消费各自保留生命周期。

## 当前编译接口

`NotificationProgram` 建立真实端点、配对、queue order、credit 和环生命周期。
`event_planning.plan_event_resources` 消费它，冻结 `EventResourceProblem`：
虚拟 lane/phase、固定动作位置、全部成员的复用关系、AIV peer 相等约束、
counter 候选组及预留。域和 handback 查询只在规划阶段发生。

`event_coloring` 和 `structured_resources` 只消费有限数字约束，不取得
`EventDomains`、SMT callback 或补造 handback 的入口。分配不能移动 native、
Wait、修改 buffer 数量或添加 controller gate。物化继续同一条 source 编译链。

## 专属优先和有限搜索

local namespace 是 `(actor, source pipe, destination pipe)`，每 bank 8 ID。
两个 AIV 的私有 bank 分开计数；cross mode-2 使用 mixed 组内的 16 个 base ID，
不能通过大于 15 的整数扩容。配置、最终 drain 和持久 token 进入同一台账。

分配先尝试专属连续映射，必要时使用显式有限映射；快速摆放失败后先检查联合
专属着色，再使用已建立的复用约束。DSATUR 采用结构顺序打破平局，全部协议
选项的回溯合计最多 16384 个节点，搜索中没有
域查询。mode-2 在专属映射之后先尝试时间复用，最后才启用并发 FIFO packing。
工作耗尽与已建立约束下无解分开诊断；保守约束图的 clique 不能证明
实际硬件最小需求。非连续映射降为小型有符号 min/max 表达式。

着色顶点是 `(完整 credit, lane)`。同一 credit 的不同 lane 必须异号；有公共
bank 且没有完整复用许可的顶点也必须异号。预留和持久 token 固定在原 bank。
AIV peer 只对编号变量取商：一个共同编号须在每个原私有 bank 中合法，原成员
之间的全部约束继续保留。满足这些冻结约束只能证明有限映射合法，不能替
上游建立者证明生命周期或 sharing 前提正确。

资源规划在虚拟 peer 对应之后检查联合专属摆放；各 bank 计数够用而公共编号
交集不足时，同样准备受影响分量的复用关系。它是准备证据的有限判据，快速
摆放失败不等于专属着色无解。

首选环使用有限次数和 native 边界的必要 handback。只有该流自身无法在容量内
构造首选环时，才直接采用 queue-composition 建立必须的生命周期。已经合法的
首选环不会因可选 queue 证明增强而主动缩小。

逐维首选构造若使整个 case 放不下，会再作一次全维允许 queue-composition 的
最小宽度构造。各维 handback 只依赖其实际端点和本维宽度，独立最小宽度的
乘积也最小，因此不需要枚举嵌套环组合。首选构造成功时不会进入这次重试；
同一 case、维度、宽度和证明模式的结果只建立一次，容量和 handback 规则不变。

受压分量可准备逐维更小的协议，保留其余维度的事实，重新建立改变的 phase、
配对及 handback。镜像 AIV 联合调整并保留 peer 对应。规划冻结按结构顺序累计
缩环的有限选项；分配先穷尽首选协议的资源约束，失败后才尝试这些选项。
当前不枚举所有嵌套环组合、任意相位或所有最小修改子集，有限选项失败仍可能
属于支持边界。它们不改变 native、buffer、通知位置或激活，不增加 Wait。

## 环复用与 local 不干涉的组合理由

每维精确窗口用实际成员 rank 对该维宽度取余，再组成混合进位虚拟 lane。
同 lane 的两个使用必定每维同余。取最外层不同的维度：普通 successor 由该维
完整消费回传连接；跨 child invocation 则连接旧 LAST 与新 FIRST。中间同余
使用按相同关系串接，其余维度依次归纳。有限使用次数不超过宽度时，本调用
没有同 lane 复用，外层重入仍须检查；空窗口没有可借用的首末里程碑。
缩小内维宽度不改变外维真实端点，故可保留外维证明；改变的 phase 仍重新核验
两端坐标对应。逻辑 rank、buffer slot 或源码先后本身均不能证明消费。

local 的两份完整 credit 只有全部成员及实际 occurrence 的占用区间都不干涉，
才取得复用许可。占用从实际发布至匹配 Wait 消费；不以 buffer 退休作终点。
如果整个同号组的各对占用都分离，任意执行点至多有一份占用，结合各流自己的
环合同即可组合。各对的顺序须由同一既有执行关系蕴含，不能各自挑选排步，
或借本次 sharing 产生的边证明自己。activation 在整个 invocation 中互斥才
可直接排除冲突，逐代奇偶条件仍须处理下一成员和各层重入。

这些规则依赖 notification binding 对递归 clock 仅准入常量平移或明确选择的
真实 FIRST/LAST/family 端点，并检查坐标往返及完整激活。未来放宽坐标映射时，
必须重审实际首末顺序，不能只保留可逆性就继续复用整段 invocation 证明。

## AIV 共用代码

规划阶段比较两个 AIV 的完整虚拟动作序列，保留位置、顺序、重复数、phase、
guard 和实际私有资源。镜像资源取得编号相等约束，但 storage credit 和 bank
仍然独立。只有相应的完整序列相等才共用一份指令。编号后不再通过整数恰好
相同发现新的合并机会；各 AIV 的 quorum 贡献不合并。

终端条件 pass 仍可把路由相同、执行域互斥的相邻通知合成一条带标量 ID 选择式的
指令。是否合并不比较 ID，也不要求它们属于同一 credit：双缓冲的两个物理 slot
本来可以属于不同 credit。相同 ID 可简化选择式，但不能删除重叠路径上的贡献，
也不能决定 AIV peer 对应。不同编号的选择式成本仍需分别审查实际 codegen。

在 2201 的终端表示中，互斥 cross 通知若只选择两个常量 ID，使用一个二选一
分支和常量 ID 指令。每次激活仍只执行一次原通知，保留外层完整存在域、所选
编号及发布/消费顺序；同号只化简选择。local、可变 ID 和更多选项继续使用标量
选择式。这是已确定事件程序的目标编码，不重新选择 sharing、环宽或 quorum。
静态通知模板数可能增加，不能据此推断运行时通知次数增加或硬件资源需求变化。

其他有限 ID 选择保留普通标量选择式，已有的单变量仿射映射继续检查完整域内的
对应及溢出。表达式结构更简单不保证 CANN 机器性能；函数大小与 mixed 代码布局
也可能随之改变，目标编码政策须按实际产物验收。

## mode-2 时间复用

`CounterReuse` 对相同路由和参与者的两条完整消息流建立实际消费关系：每个旧
receiver 的 Wait 已消费，才允许每个新 publisher 的 Set。包括同一 clock 内的
episode、各层重入及整段 invocation 的 LAST→FIRST；空 child 不提供边界。
当前 handback 子集要求实际 clock 为单位平移，或无 clock 的单次固定位置。
输入 guard 在整个 invocation 中互斥时也可直接取得不干涉关系。

cross Wait 阻塞整个 controller。当前规则使用一条已经需要的、反方向的完整
quorum 回传：旧消费先于它的发布，它的接收又先于新 Set。逐 receiver/publisher
检查全部组合，不将同 actor 的源码先后冒充跨核完成，也不搜索两段 queue handoff。
后者缺少该直接回传时不提供时间复用资格。

规划对两个完整 credit 的所有成员建立证据；实际同号组的每对 credit 都须有
不干涉关系。于是任一执行点至多一个 credit 尚有未消费贡献，积压取各 credit
上界的最大值，分别保留 AIV publisher 的贡献，不取并发流的求和。原有消费关系
已经蕴含发布效果次序，因此不增加 controller gate 或新的完成约束。

## mode-2 整组 FIFO

并发 packing 使用规划阶段建立的 `CounterGroup`，不同于 local 二值着色。
当前充分规则要求每个 publisher 和 receiver 上的消息属于相同实际 clock、
scope、stage 及坐标映射；所有参与者的完整消息 word 相同。逐 credit 的成员
兼容仍先检查，但两两兼容不能代替整组 word 检查。

同号会增加 counter 发布效果的 FIFO，接收沿既有 controller 顺序；两者分别记账。
实际 clock 必须是保留各层 invocation 的单位平移。不同 stage 间跨迭代交错不属于此规则。互斥成员只有
在已证明的真实位置和激活对应下才能进入同一消息流。

局部进展使用 source pipe 的单调 native 前缀：后一个发布原本已经捕获前一个
发布所需的全部旧 native，且发射在它之后。在符合 15 份积压资源合同的前提下，
旧发布效果没有额外的未来工作前提，因此新增效果 FIFO 不引入局部阻塞环。
这不把旧 Set 的发射当作其完成，也不宣称不同 ID 的发布效果原本已有这个顺序。
整组 word 在各层 clock 的重入和匹配激活过滤后保持一致，不能只比较同一代次。

积压按完整 credit 去重后保守求和，分别记录每个 publisher 的贡献上界，均不
超过 15。两个 AIV 各自贡献，快 AIV 不能替代慢 AIV。删除候选组中的消息保持
word、quorum、FIFO 和积压证据，故分配可选择其子集；最终检查实际同号成员
确为某份完整证据的子集。此规则没有增加 Wait 或新的 controller 排步。
同一 ID 的集合必须整体满足时间不干涉，或属于一份完整 FIFO 证书。不同 ID 可
分别使用这两种合同；同一 ID 上任意混合时间 epoch 与并发子组、更一般的多时钟
交错 packing 尚未支持。

## 证据口径

`Compilation.report.event_resources` 记录实际通知、完整 credit、环宽/phase、
预留、编号、sharing 与物化动作；`protocol_selection` 区分首选环、选中协议及
受影响消息流；`counter_time_reuse` 保存旧消费、新发布、回传及最大积压，
`counter_groups` 则保存并发 word 与求和积压。报告是诊断输出，不能作为外部输入的 verified
标记。独立有限模型读取真实指令、字节版本、二值消费和逐发送者贡献。
CPU 检查、ASC 对照、CANN、设备数值、编译耗时与 msprof 性能分别给出资格。
