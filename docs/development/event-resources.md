# 事件资源的事实、政策与编号

事件 ID 是硬件资源映射，不能代替 buffer slot、Stage、消息身份或 ownership。
physical credit 保留完整的 ownership 和成员身份。首选映射让全部成员共用
资源；资源规划也可将实际成员作为不同 phase 占用，不同 credit 的复用仍需
独立消费证据。逻辑数据、物理存储和通知消费各自保留生命周期。

## 当前编译接口

`NotificationProgram` 建立真实端点、配对、queue order、credit 和环生命周期。
`event_planning.plan_event_resources` 消费它，冻结 `EventResourceProblem`：
虚拟 lane/phase、固定动作位置、全部成员的复用关系、AIV peer 相等约束、
counter 候选组及预留。域和 handback 查询只在规划阶段发生。
`EventIdPools` 在进入规划时保存实际可分配的整数集合及私有 bank 覆盖；所有
缩环选项引用同一份不可变池。local 只允许 0..7、cross 只允许 0..15 的子集，
检查整数类型、重复及 bank 身份；固定预留仍占据原硬件编号，即使它不属于
该可分配子池。编号器保持这些实际数字，不用子池长度重新构造 0 起始的编号。

`event_coloring` 和 `structured_resources` 只消费有限数字约束，不取得
`EventDomains`、SMT callback 或补造 handback 的入口。分配不能移动 native、
Wait、修改 buffer 数量或添加 controller gate。物化继续同一条 source 编译链。

## 专属优先和有限联合选择

local namespace 是 `(actor, source pipe, destination pipe)`，每 bank 8 ID。
两个 AIV 的私有 bank 分开计数；cross mode-2 使用 mixed 组内的 16 个 base ID，
不能通过大于 15 的整数扩容。配置、最终 drain 和持久 token 进入同一台账。

分配先作专属连续映射；已有完整 FIFO word 的分区能直接放入时同样返回
具体映射。这两种构造不枚举所有资源对，不声称全局最少 ID。其余情况把全部
已冻结协议选项和实际 lane 数字构成一份 Z3 有限问题，词典序优先原选项，再
优先每个独立 bank 分量的专属映射。没有 Python DSATUR、编号递归或选项重试。

local 从冻结行 mask 的缺失许可展开实际冲突，保留每个原私有 bank；只对
实际冲突展开 lane 异号约束。已知超额 clique 在构造稠密余部前拒绝。位图成本
还依赖 mask 字长，普通显式数字输入继续按实际公共 bank 检查许可。

counter 同号集必须选已有整组证书之一：完整时间分离使至多一个 credit 活跃，
各 backlog≤15；同一 FIFO word 的过滤子集使所选 backlog 之和≤15；已证时间
组合使各 FIFO child 所选 backlog 和的最大值≤15。模式变量只选已有证书，
不产生完成或消费事实，也不枚举所有兼容子集。

求解沿用 2 秒局部上限并服从编译整体剩余期限。UNKNOWN 和未确定最优偏好
均不授权分配；UNSAT core 标出有限选项、pool、lane 或分离约束，只能说明
这些冻结前提下无解，不能证明实际硬件最小需求。统计分别记录变量、约束、
求解次数与耗时。非连续映射仍降为小型有符号 min/max 表达式。

着色顶点是 `(占用, lane)`，占用可包含完整 credit 或其实际 phase 成员。
同一占用的不同 lane 必须异号；有公共 bank 且没有完整复用许可的顶点也必须
异号。预留和持久 token 固定在原 bank。
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
缩环的有限选项；联合求解以选项序号为首要目标，保留首选协议优先政策。
当前不枚举所有嵌套环组合、任意相位或所有最小修改子集，有限选项失败仍可能
属于支持边界。它们不改变 native、buffer、通知位置或激活，不增加 Wait。

## local 成员 phase 占用

若冻结的完整 credit 冲突约束已有超出实际可用池的 clique，规划为该 bank
准备一份实际成员 phase 选项。这个判据只说明当前粗映射放不下，不说明硬件
最少需求。它不枚举全部兼容子集，也不把同一时刻的观测峰值当作映射许可。

每个 phase 仍指向原 `CellChannel` 和原通知成员，保留其完整 case、配对、环
生命周期和真实 Set/Wait。原 physical-cell 归纳允许同一 credit 的非重复成员
继续同号；这份投影记录为 `CellPhaseSharing`，不能用于不同 credit 之间。
跨 credit 的同号组按实际选中成员重新建立完整消费前沿，包括各层 LAST/FIRST
与重入，编号仅消费冻结的许可。一个早期成员相冲突，不再强迫两个 credit
所有后续成员也异号。整个过程不改变 ready 粒度、native、buffer 容量或 gate。

镜像 AIV 同时准备 phase 选项，保留两个原私有 bank 的编号交集；原有共用
通知序列的次序、重复数和完整 guard 必须继续保持。整数分配失败时才选用
已经冻结的后续选项，所有选项在同一有限问题中选择。当前该规则细分 local
稳定 cell 的通知成员；mode-2 继续使用完整 word/quorum/backlog 合同，普通
runtime ring 内不同 lane 的进一步共享仍须额外证明，不能由此规则推出。

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

事件规划按受压 local bank 或实际 counter route 建立完整 credit 的分层消费前沿。
counter 先尝试现有完整 FIFO word 分区；已放入 ID 池时不求可选时间复用。
同一激活类保留
最早的真实 publisher 和最晚的真实 consumer；合并必须具有相同的完整 guard、
坐标、case、输入范围及参与者合同，并在每个实际 publisher/receiver 上证明
真实端点间的无条件发射包围关系。一个 AIV 的顺序不能代表另一个 AIV。
同一 native 锚点上的两条不同通知没有由此确定的相对顺序，不能据 rank 相等合并。
包围只提供 issue/queue 前缀覆盖，不建立新的完成、消费或 quorum 贡献。

两份前沿的分离仍检查真实尾 Wait 到新头 Set 的 handback，包括同代正向和
每层下一次 invocation 的反向回传。无法证明的粗接口继续细分到实际消息端点，
不另建全历史 pair 表或第二条求解路径。A→B、B→C 的两份许可本身不能推出
A→C：链尾到下一次链首的消费义务仍须独立成立。不同激活类或任意交错结构
仍可能需要稠密的接口检查；本规则不承诺所有程序都具有线性分析成本。

已建立的完整 credit 分离冻结为 `CreditSeparation` 和有限 bitmask 矩形。
编号器只查这份不可变数字关系。纯时间复用同号组逐行检查成员 mask 的完整覆盖，
任一成员缺少对其他 credit 的许可即拒绝；检查和报告保留所选子矩形，不再展开
每对消息。报告以 `LocalEventReuse` 记录 bank、ID、完整 credit 成员与所选矩形，
counter 的 `CounterReuseGroup` 同样保留完整 credit 和所选矩形，其逐 publisher
backlog 为不干涉 credit 的最大值。FIFO-child 组合逐行检查其他 child 的完整
集合，child 内保持 whole word/quorum 和 backlog 求和，再取各 child 上界最大值。
共享的 `event_certificates` / `event_frontiers` 表保存证书和真实端点树。
前沿只保留 head/tail、子树和成员数，不在每次 join 复制完整成员集合；报告按树
保存各实际参与者的 enclosure，不重新展开 counter credit/member 许可对。
这些操作不重新调用域或 handback 分析；逐消息对展开仅用于开发对照测试。
缩环选项只改变 width/phase 时，共用同一输入 owner 的
激活、真实端点、包围和消费事实；完整成员集合或输入 owner 改变即不能沿用。

完成/消费 transfer 共用同一不可变整数输入及完整有序前提的数学服务；有限
规则使用隔离上下文，冻结的消费证据只保存实际支持规则、前提及 owned Answer，
不携带 solver 或分析回调。删除协议后，未变的数学输入可以随域事实投影；
它们不保存协议图或完成/消费许可。消费前沿 word 内相同前提及 before/after 位置关系可复用原查询
服务的数学答案，每次接受仍引用当前候选的实际 Wait、取得 cut 和原始证明。
更弱的激活、不同位置关系或另一输入 owner 都不能借用旧答案。该复用不豁免
原式定义性、溢出、真实 gate 或参与者义务。

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

cross Wait 阻塞整个 controller。消费事实可沿已保留的真实 capture/acquire 继续
传递，各次取得都绑定实际 queue/controller 和位置。每个旧 receiver 到每个新
publisher 的义务保留完整 quorum；同 actor 源码先后、native ready 或删除前的
图不能替代该事实。必要反馈、参与者或实际取得 cut 缺失时不取得时间复用许可。

规划对两个完整 credit 的所有成员建立证据；实际同号组的每对 credit 都须有
不干涉关系。于是任一执行点至多一个 credit 尚有未消费贡献，积压取各 credit
上界的最大值，分别保留 AIV publisher 的贡献，不取并发流的求和。原有消费关系
已经蕴含发布效果次序，因此不增加 controller gate 或新的完成约束。

## mode-2 整组 FIFO

并发 packing 使用规划阶段建立的 `CounterGroup`，不同于 local 二值着色。
叶子 word 的充分规则要求每个 publisher 和 receiver 上的消息属于相同实际 clock、
scope、stage 及坐标映射；所有参与者的完整消息 word 相同。EventDomains 已经
在实际坐标中建立每条消息全部端点的激活对应，所以各参与者实际执行相同的
过滤 word。结构、积压和该共同 word 一次建立；不额外构造 member-pair 顺序表。

不同通知在同一锚点没有自然先后。每个实际 actor/phase 的 tied group 使用各
消息全部 case 的完整激活，逐项累计 seen 和 overlap，证明最多一条激活；
条件 DAG 随消息数增长，不枚举 guard 两两积。无法证明即拒绝该 FIFO 候选，
不选择新动作顺序。不同 guard 可以在共同 word 中过滤；不要求它们字面相同。

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
候选规划保留通用贪心候选，并按完整激活/clock 形状建立额外整组候选；候选允许
重叠，不把第一份分区当作唯一可能映射。结构分桶只选择候选，不代替完整证书。
候选集合有限且不保证全局最优，不枚举所有成员子集。

编号直接选择完整 word 的子集，以 CounterFIFOGroup 保存实际 ID、所选完整
credit 和原证书；不再逐 pair 投影兼容前提。证书保留同锚点完整互斥证明。
不同 clock 的 child word 保留自身证书，parent 另外保存每个参与者的实际
拼接边界；报告按这些对象交接，不复制或重建历史配对表。

数字 FIFO 分区每次优先选择覆盖最多尚未覆盖 credit 的候选。候选被其他 word
部分覆盖后，其旧大小仅是上界，须更新收益，不能继续按原大小消耗剩余资源。
合法 word 的正 backlog 总和不超过 15，更新次数受其实际 credit 数限制。这是
一份有限的映射提案，不保证一般集合覆盖最优；只在提案确实放入池时跳过可选
时间复用，失败仍不能声称硬件最低需求。

不同 clock 的 word 可以保留为 `CounterGroup.episodes`，用每个实际 publisher 和
receiver 上的 LAST→FIRST 发射位置证明整个调用内的顺序拼接。各片段继续使用自身
clock，不比较不同 actor 的裸 epoch。ownership 和编号身份保留；此规则仍对完整
credit 的积压求和，包含跨片段尚未消费的贡献，不把循环退出当成 counter 清空。
不能建立整次调用 word 边界的更一般嵌套/交错结构，不取得此拼接资格。

## FIFO 子组的时间复用

`CounterTimeGroups` 保存多个不相交的完整 credit 子组，各子组有自己的 FIFO
证书或是一份独立 credit。任意两个子组之间，每一对 credit 都须已有完整
`CounterReuse` 证据；没有这些证据的两两兼容图不能授权组合。

于是任一时刻只有一个子组有未消费贡献：子组内部积压按完整 credit 求和，
子组之间取最大值，仍逐 publisher 检查不超过 15。两个 AIV 的贡献保持独立。
冻结后的组合只向分配器交付有限数字选择；最终同号组检查共同消费原组合，
删除未选成员保留资格。不会为获得时间复用添加 Wait、移动 native 或改变 buffer。

## 证据口径

`Compilation.report.event_resources` 记录实际通知、完整 credit、环宽/phase、
预留、编号、sharing 与物化动作；`protocol_selection` 区分首选环、选中协议及
受影响消息流；`counter_time_reuse` 保存旧消费、新发布、回传及最大积压，
`counter_groups` 保存并发 word、顺序片段及求和积压，`counter_compositions` 保存
FIFO 子组、时间回传和最大积压。消费证明和布尔条件以
`consumption_proofs/consumption_predicates` 共享 DAG 引用，保留实际取得 cut、
原始前提及数学答案，不展开全部 relay 历史。报告是诊断输出，不能作为外部输入的 verified
标记。独立有限模型读取真实指令、字节版本、二值消费和逐发送者贡献。
`occupancies` 显式对应整数资源、原 logical credit、实际 phase 成员及 width，
local 复用矩形中的 resource 编号引用这份表。`id_pools` 保留实际可分配数字。
CPU 检查、ASC 对照、CANN、设备数值、编译耗时与 msprof 性能分别给出资格。
