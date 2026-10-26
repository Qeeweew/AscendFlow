# 编译不变量与检查职责

2026-10-21。以下合同由当前唯一 source 链路保持。具体生成规则、前提及
归纳理由见 [generation-rules.md](generation-rules.md)，验证方法见
[validation.md](validation.md)。设备与编译性能须对实际产物分别测量。

## 1. 先区分五类工作

| 类别 | 是否需要对当前程序执行 | 放置原则 |
|---|---|---|
| 输入事实 | 需要 | 访问、别名、排步、定义性等一次建立；输入变化才更新 |
| 规则适用前提 | 需要 | 在生成或组合时检查；不是重新证明规则定理 |
| 新的资源/协议约束 | 需要 | 在引入它的阶段建立，明确影响范围 |
| 未变事实的重复求解 | 不应需要 | 下游消费已有结果，不重新建完整执行/路径图 |
| 独立验证 | 开发测试/debug/外部输入需要 | 不能代替生成规则，也不必每次正常编译重跑 |

每项义务记录输入、建立者、产出、消费者、保持条件及失效条件。不要把所有名为`prove`的函数一概删除；也不要用cache命中率证明职责设计已合理。不建设可伪造的verified标记或庞大通用proof框架。

事件配对与复用成员分析分别建立。完成覆盖只消费真实配对实例；覆盖删除后，仅
保留协议的生命周期构造需要精确窗口。对已证明通用配对的域作合法参数特化时，
保持其等价事实。配对成立不提供旧 Wait 消费或端口重入事实。环复用检查全部
共同 clock 的精确窗口；同 actor 的单发布/单消费本地通知还可逐层证明任意两次
实际使用之间的消费顺序，独立取得有限本地环，不要求 dense rank。缺失窗口本身
不授予复用。同步转移中的自由可达命题使用 Bool；直接前提
传播保留实际 capture 路径，不将 source 的整数值或发射顺序当作完成事实。

source 的槽选择保留实际 allocation、原 selector SSA、槽轴、深度和 generation/
physical 方式。lowering 产生该 selector 对应的真实 modulo 或直接索引；raw view
与结构化 view 进入相同地址/effects/alias 链。Tile slice 沿共享值 DAG 转发原
writer，读取的真实区域仍由 admission 检查，切片不定义新值。

选中的真实 writer 若已定义完整逻辑 allocation 或已建立 enclosure 的完整槽平面，
其 Tile 可供该范围内合法的间接子视图读取；不重复证明调用方动态索引的 bounds。
普通初始化检查也消费同一 source/native enclosure 及真实完整 writer。完整范围排除 allocation padding，
不能由其他 writer 的初始化补大当前 Tile。实际读取仍检查 participant、原 writer
及之后的可能覆盖；该数值定义事实不提供 ready、ownership 或事件消费。

返回部分 row-prefix copy 的 Tile 表示该更新点的目标区域。其保留部分必须在
更新发生时已有定义：admission 在真实 native 更新后、下一个 writer 前，从既有
region/字节 transfer 冻结来源快照，并分别保存实际 participant。完整 payload
只作为值的区域 enclosure；真实 prefix effects 不扩大。后来写入不能回填早先
快照，raw alias 的后续覆盖仍由原 stale-value 检查拒绝。定义快照不提供 ready、
ownership 或 reader 退休，后续同步仍使用真实物理访问。

CopyGroup 的 typed region 对应 physical ReadyGroupOp 的实际 native 成员。
内存事实检查相同 queue、参与者与 activation；RAW 共同捕获最后成员的真实
源前缀，原存储 ownership 和 WAR/WAW 分别保持。内部联系不通过共同 ready
反向等待。资源规划前完成端点与 quorum 检查，终端投影只移除已消费的组描述。
独立有限模型从 source 对应记录额外的共同 ready 义务，但只允许实际 Set/Wait
建立完成顺序；未读取的同组成员也不能提前被宣布完成。

原始义务的真实 source/consumer 和共同 ready 前缀先于通知机制选择建立。
coverage 引用 selection 拥有的同一份 binding，完成覆盖和动作合并保持其身份。
选定通知的 anchors 同样引用这些原始 binding，必须属于本通知覆盖的需求；
新增 coverage 不增加或替换存储归属，复制一份值相等的 binding 不能冒充原始事实。
稳定 cell 的二值 credit 还要求真实相邻 lifetime；保守前沿中的跨 child 非相邻边
不提供此归纳。按完整 ownership/role/route 检查所有片段的重复实际 Set/Wait 位置，
重叠 fan-out/fan-in 的片段保持独立 credit，另行取得消费回传才可复用编号。
闭合私有轮次在输入事实建立时选定真实最后 reader；后续 ownership 直接消费
原始 RAW 端点，不重新绑定或创建临时通知。CopyGroup 的 ready 前缀不替换
覆写前的真实 writer 位置，全部 reader/effects 和退休义务继续保留。

目标原生指令的静态描述由 `NativeSpec` 统一保存，逻辑 memory template 和
终端 native 共享同一份不可变值；实际地址和循环 read condition 各自绑定真实
SSA/代次。opcode、参数或地址 operand 数量改变必须更新描述，不能沿用旧
几何输出另一条指令。packet 仍对可信注册表核验 identity 与 closed mask；
描述共享不取代原生硬件检查、ownership 或同步资格。

源程序实际执行的访问及算术定义性属于 DSL 输入合同。编译器不证明全部
动态原轴/byte bounds，也不要求所有 index 拥有 uniform extrema；静态非法
原式仍在代数化简前诊断。typed SSA 保留实际执行语义，`Expr` 仅是可选分析视图。
ASC 中 unsigned 除法/余数的中间运算在离开该 SSA 定义时恢复 index 的 int64 类型，
包括内联表达式；其后有符号减法、比较和 min/max 不得继承 C++ unsigned promotion。

存储检查保留真实 allocation、原轴线性化、native 子视图/行间空洞、对齐及容量。
源索引作为 SSA 原子参与对应，不展开任意数据计算；不同 active domain 仍保留
自己的控制与 effects，但不再逐域证明调用方的数据合法性。

AST 和 lowering 保持条件、mask、求值顺序。已有且支配插入点的 SSA 直接复用；
纯整数条件选择的分析值保留真实 condition 和两侧值，以数学 `If` 编码；不同
stage 的纯复制在相同输入/坐标下可证明相等。执行 SSA 仍为分支，不外提未选中
的除法，也不把含 load 的选择或循环 carry 赋予该纯值合同。
源动态定义性不再重复求解。编译器新建的通知/时钟/地址算术另有保持义务：
非空前缀先于 LAST、短路前项先于后项，不将 load 或部分函数外提到更大域。
新通知算术在真实物化位置检查定义性和 int64 范围；非负加乘的常量取模可先
约减操作数，避免事件 rank 在取模前溢出，不改变 ID、width 或共享政策。

事件域保留每个 guard 求值前的真实 metadata 前缀，不能以分支/body 内的 load
证明自身激活。在已经假设端点实际执行的子问题内，可使用其局部输入合同；
不能将其升级为循环非空、端点存在或完成证据。条件值在早期 publisher 尚未
定义时，使用既有平衡的 queue-prefix 入口/出口，不提前读取条件，不添加统一 drain。

已核验的 [L0C unitflag](unit-flags.md) word 单独覆盖参与访问的 RAW/WAR，
不建立完整 native 或 pipe-prefix 完成。其条件 FIX 保留真实 reader，不为它
提前捕获 metadata；全部其他访问、边界和 ownership 继续保留。flag 范围、
完整 footprint、更新/消费激活及实际交替次序在访问顺序构造前建立一次，
改变这些前提时失效。独立模型不把细粒度握手提升为 MAD.end→FIX.begin。

空间 enclosure 可由合法源访问及已检查的布局对应建立，无需再证明源 bounds。
它只提供保守空间分离，不改变真实 native 字节、不建立初始化、KILL 或 credit。
区域关系未知或可选分析预算耗尽，保留 AliasRegion 及必要源码顺序；必要协议
检查仍拒绝无法实现的 ownership、quorum、进展或有限事件资源。

MAD 周期 reset 的数值初始化归纳可作用于环形槽。必须证明实际地址、激活和
后代循环次数在整个 reset 区间内与其起点相同；仅在起点完整写入的同一 footprint
才可供后续 RW 读取。该查询消费已有 loop/facts，保持真实 RW effects，不能
提供异步完成、ownership 或最后 reader 退休；缺失起点或区间内换槽仍拒绝。

二维前缀保留实际 count SSA 与静态容量，native 使用同一个 count。部分写不
授权完整初始化；显式补零的 GM 部分读和 L1 完整写分别建模，禁止整块 DMA 多读。
L1 NZ allocation 的物理 shape 和 C0 经统一布局传递，子视图不改变 leading M。
初始化按真实矩形联合及 writer 来源建立；相邻部分写可以拼满，矩形包围盒不能
填补孔洞。完整定义不提供 ready、ownership KILL、reader 退休或事件消费事实。

线性存储的初始化按真实 allocation/participant、读取 loop nest 和完整 guard
建立增量上下文，每个新 writer 在该上下文内只转移一次。已定义常量字节区间
维护精确并集，相邻或重叠区间可合并；跨行孔洞和符号坐标继续保留原 span。
完整分支树仍先按实际参与者及逐代 footprint 闭合，再作循环摘要。数值并集
保留原 writer 身份，不替代逻辑值版本检查、ready 或最后 reader 退休。
读取的 writer 来源以不可变前缀节点共享；读取保留当时的快照，后续 writer 不能
修改旧快照。无条件写与完整分支树的 writer 均按首次建立覆盖的次序去重追加，
正常准入不复制整个历史 tuple；显式诊断迭代仍可取得全部 writer 身份。

公共编译在独立 expression scope 内驻留完全相同的原始 `(kind, args)` 节点，
退出、异常或取消时释放节点表，嵌套 scope 恢复外层。它不保存 range、guard、
alias 或完成证据；数学等价的不同原式不会在此合并。原始 SSA 的求值域继续保留，
新求值另检查有序定义性。查表前检查 key 类型，不能让 Python 的 `True == 1 == 1.0`
将非法常量误认为已构造的合法整数。确定性排序继续使用结构键。

`memo.py` 统一缓存生命周期。公共编译使用 `memo_scope(fresh=True)`，内部独立
入口复用已有 scope；退出、异常及取消均清空各表并恢复外层。`memoized` 自动
管理纯查询查表，调用处显式声明完整语义键（包括输入 facts 的不可变快照）。
`memoized_method` 将 `functools` cache 安装在分析对象自身，owner 无需可 hash，
全局不保留旧分析对象；失败结果不入表，None/False 是正常缓存值。
只对拥有不可变输入快照的分析方法使用该机制。SSA 绑定表、诊断台账和可变
范围的失效不是纯查询，不靠该装饰器推断 IR 未变化；输入变化须重建 owner
或继续执行原有范围失效规则。

同一 scope 的同时坐标替换按完整不可变 binding 快照复用原始 DAG 结果，
不替换到新坐标内部；无变化的结果保留调用方原节点身份。capture 改写保留
原 SSA binding 的结构顺序，只更新坐标与 hash，不重新从已变化的 IR 取顺序。
批量坐标翻译通过 `substitution(bindings)` 一次绑定快照并复用同一 DAG 查询；
后续修改调用方的 mapping 不改变该翻译器。单式 `substitute` 消费同一机制。
整数代数查询按原式、完整输入 facts 快照和 atom identity 复用；改变 multiples、
参数或环境不得沿用旧答案，异常不存为结果，退出编译释放这些表。

BoundPlacement、LogicalNotification、StructuredNeed 的 hash 在各自不变记录中
按需保存，相等语义仍由全部原比较字段决定；hash 碰撞不代表相等。记录替换
重新计算自己的 hash，既有 hash 不授权修改 guard、成员、width 或共享关系。
Comparison、Boolean 和 Symbol 同样只保存自身完整比较字段的 hash；它们不保存
范围、定义性或完成证明。

消费使用独立的 ConsumptionOrigin：真实匹配 Wait 已消费后，它只阻塞指定 queue，
Scalar/cross Wait 按目标合同阻塞 controller。来源完成和消费共用位置、实际
capture 快照及匹配 acquire 的 DAG，不能把数据 ready 替代旧 Wait 消费。
复用窗口通过实际 offset span 和 FIRST/LAST 端口建立，不展开 runtime N。
同锚点不同动作没有自然先后；消费到新 capture 必须严格推进真实位置。

消费前沿保留 origin、实际取得事实的 Wait 或 queue admission cut、完整前提、
证明及该输入作用域拥有的数学答案。二维位置索引只从同一阻塞路由和实际调用
取前沿；在实际 Wait 后取得的事实不能借相同锚点回填更早 capture。改变前提的
查询仍证明全部激活及两个 cut 的顺序，不复用其他作用域的 proved 标记。
origin 的阻塞路由区分 controller 与实际 pipe；cross Wait 的名义 destination
engine 不是它消费的 pipe 前缀。后来的 controller Wait 不能代表更早的 local
pipe Wait 已消费，反方向仍须保留实际派发顺序及完整激活。
成功证明中的每个真实 acquire 导出自己的消费子证明，不把最终结论倒灌到更早
动作。同一 acquire 的条件子证明先合并，再检查完整前提是否覆盖；单侧分支不
产生无条件事实。controller acquire 的事实可由该 actor 后续实际派发取得，pipe
acquire 的事实只沿自己的 pipe 延伸。可选队列组合的证明与仅允许 native 边界的
构造前提分开保存，不能靠查询次序改变正常生命周期的选择政策。
条件证明可以共享；未覆盖开放边界时继续组合该图，缓存阴性结果不作不可达证明。
报告按 DAG 引用保留证据，不重新展开全部 relay 历史。

生命周期生成与资源规划共享同一 retained program 的消费分析，不只复制边界
cases 后重建位置、转移图和前沿。完成选择的候选位置图在产生独立证明后释放；
它不交给消费分析沿用，消费只针对最后接纳的真实动作建立。

完成/消费查询只证明完整原始前提下的目标 cut，不构造所有条件组合下的可达
函数。每个有限转移顶点有独立数学命题；真实 capture 为来源提供 seed，既有
capture/acquire 关系提供带原整数条件的 Horn 规则。固定输入时，最小可达集是
这些规则的最小模型；没有 seed 的环允许全 false，不能自己建立事实。数学命题
不进入运行时 IR、guard 或事件 ID。capture 的位置 word 只分段来源顺序，不提供许可。

完成选择以真实支持前沿为接纳单位。候选位置图只提出支持方案，查询排除自身
及全部已覆盖候选；成功方案含有新支持时，再仅用已接纳动作尝试覆盖，避免
任意 solver core 夹带新的支持。选定支持先一并接纳，再发布覆盖事实。
已覆盖候选不加入保留集合；已接纳协议不再参与覆盖选择或被删除，因此先前支持不因选择失效。
原始义务、真实 reader 和 ownership 全部保持，前沿不承诺任意条件程序的全局最少同步。

数学命题及规则只为实际查询需要的顶点构造，同一规则编码一次。目标查询先取
来源正向闭包与目标反向闭包的交集，没有实际 seed 到目标的结构路径时无需 SMT。
已有 SMT 模型只有满足全部原始前提时才可复用。按该模型传播时，未解释的
guard 保持可能成立；这个保守闭包仍不含目标，才可用该反例否定完整覆盖。
反例只减少无效证明尝试，不提供完成、消费或事件共享许可，也不补全缺失值。
使用隔离的规则系统，以原始前提、来源 seed 和相关保留规则绑定临时作用域；
查询结束撤销这些绑定。未启用规则不放入 solver：其标签只出现为自己规则的
使能前件，取 false 即可将该断言消去，不改变当前目标的可满足性。失败查询不导出或全局保存整图
蕴含式。该系统保留同一整数服务的不可变输入，不污染普通数学查询。成功结果保存 UNSAT core 所选的实际 seed/transfer 规则、origin、
目标命题、前提及精确 owned implication。规则叶不是已取得事实；每个实际
acquire 必须在同一上下文独立证明，才可导出它自己的消费 cut。互斥分支合成的
出口结论不能倒灌成任一分支内部的无条件取得。支持动作删除仍使相应因果证据失效。

转移语法按实际顶点/边增长，不保存 SCC 祖先矩阵、BDD 或路径展开。每个新
origin 的规则上下文仍有与有限关系大小相关的成本，SMT 查询也可能 unknown；
此时不提供覆盖或复用许可。该有限规则合同不表示 parent 已能只消费所有 child
的开放出口；现有参数窗口及 FIRST/LAST 义务仍独立保持。

本地事件与 counter 时间复用使用分层的真实 publication/consumption 前沿。包围依据完整激活和
坐标合同下的无条件发射顺序，实际消费仍由 handback 建立，包括各层重入。
每个实际 publisher/receiver 的包围必须同时成立；同锚点未排序或任一参与者反向
均不能合并。前沿以子树和成员数保持原始成员，不复制完整历史集合。
冻结的完整 credit 分离矩形只提供有限编号许可，不以相邻 pair 传递代替 wrap
义务。未变输入 owner 上的缩环选项可保留端点事实；成员、guard、坐标或输入
事实变化时重建对应接口。详见 [事件资源](event-resources.md)。
最终时间复用同号组检查逐行消费冻结的 bitmask 覆盖；FIFO-child 组合检查每行
对所有其他 child 的完整覆盖，child 内仍消费 whole word/quorum/backlog 证书。
counter 的 peer 数字投影保留原 credit 行，只枚举缺失许可对应的真实冲突。
保留选中的证书子矩形；
报告共享端点树和证书，不把矩形重新物化为全部历史消息对。

## 2. 保证的目标归属与当前缺口

| 不变量 | 目标建立职责 | 当前应核对的代码/缺口 |
|---|---|---|
| 类型与runtime输入合同 | frontend/admission；输入条件由调用方满足 | 编译器检查合同声明；生成的kernel和launcher不验证调用方输入 |
| 完整effects | native lowering/目标原语合同 | 包括scratch、padding、SPR、读写及真实退休 |
| 配置状态 | NativeSpec/registers 的值事实，access_order 的真实读取与覆盖顺序 | RegisterCell 不冒充 tensor；定义、ready、旧 reader 退休分开；入口 setter 的 guard 依赖实际支配 SSA |
| 存储实例与alias | 物理访问、pointer合同与可选分区分析 | group/peer独立性属于源语义；保留root/view/slot/generation及AIV私有性，未知区域保留源码内存顺序 |
| 源值语义 | frontend/Tile 与 admission、局部 packet 合同 | 不要求全程序精确 writer 闭合；dispatch 不能改变原读取值；may-write 不宣称完整定义 |
| 物理复用 | access_order 与局部 packet 的实际 dispatch word | 保留全部 pending reader；控制端口只提供插入位置，真实 Set/Wait 才捕获/获取前缀 |
| ready/release覆盖 | notification binding | 真实 capture/acquire、独立覆盖及原始 ownership edges |
| 配对与事件消费/复用 | 生成器的域、cell credit 与 handback 规则 | 域只检查一次；物理 lifetime 归纳或明确消费回传，rank 不提供消费证据 |
| 可进展 | queue_order 的分层生成准入 | 按路由/有限时间类分解阻塞区间，保留零 step phase DAG 与正 step 前进；索引内部顶点不提供完成/消费，父层只组合 child may-block 边界 |
| bank/ID/width/quorum | 资源规划建立协议及复用约束；编号分配器检查冻结的有限映射 | 时间复用覆盖每个旧receiver至每个新publisher的消费回传；FIFO完整word逐端同序；有完整跨子组不干涉证据时取FIFO子组求和上界的最大值，各publisher≤15；编号选择不触发域或handback求解 |
| 输出对应 | 物化与后端 | 不重选同步，不改native/控制/guard语义 |

上表不表示所有已存在的前置分析都应重写。先对源码确认真实职责，只删除被替代的机制。

## 3. 循环组合必须保持

- 每个合法逻辑代次在每个stage恰好执行一次；坐标替换同时作用于地址、guard、slot、metadata和捕获值。
- N=0透传原状态；N=1的FIRST/LAST重合不重不漏；短循环不能假借稳态条件。不得按runtime长度展开。
- 父层绑定child的真实外部端口和祖先代次，不重新求其内部路径历史；不同代次不能共用裸ordinal条件得出矛盾。
- 控制分析一次保存每个循环进入前的祖先范围；初始化非空证明与slot必访判断
  使用这份调用域，并随实际代次选择替换坐标。不能只用全局参数事实而遗失
  父子次数的关系，也不能借用待证明循环自身的body域排除其零次路径。
- 循环控制退出不证明native完成。跨loop的旧reader与未消费event仍通过对应端口传递，不加统一loop drain。
- 每个逻辑迭代域按 actor family 输出各自的物理时钟循环，stage 以 `g=j_actor-offset` 及其存在条件执行。首末条件属于同一生命周期，不能独立seed/drain。kernel最终收尾单独处理。
- buffer slot数、stage偏移、event width互不代替；跨AIC/AIV使用实际参与者与GM可见性，不能丢掉第二个AIV贡献。
- 选择 child FIRST/LAST 时，先用已解析的全部祖先坐标特化 child header，包括祖先自身的 FIRST/LAST；已经写为 Expr 的计算 LAST 也要解析 selected ancestor。已由 region binding 处理的普通 j-d 平移不能再应用一次。
- 有界 dense read prefix 可用 count 上界形成保守空间联合；native 仍读取真实分块，family、LAST 和 release 使用实际 count。不能用于动态写或假设上界代次被执行。

## 4. 物理、数值与硬件语义

搬运按目标 native 区分对齐、步长单位与有效字节。普通 DMA 的 32 B 块约束
不能外推到矩阵 ND2NZ/FIXPIPE，也不能用矩阵的元素对齐许可放宽普通 DMA。
显式补零的封闭 packet 保留 fill→MTE2 fence→ND2NZ 次序；FIXPIPE 的 dtype
转换只改变目标元素宽度，保留 FP32 累加器的实际 fractal 读取与退休关系。

通用 `mask=` 更新是已初始化目标的 RW 更新，不能作为整片初始化。坐标谓词在目标
每个64 lane计算局部bitmap：全0块不调用native、连续全1块合并repeat、部分块使用
自己的bitmap；不发出空mask，也不把一行的bitmap复用于不同行/块。数据谓词使用
显式packed UB存储、packet局部CMPMASK和已审计的局部`PIPE_V`。封闭packet分别
记录Normal/Counter模式、bitmap/count含义和CMPMASK是否在包内建立；`count_end`
只恢复Normal模式，不恢复mask位值。packed谓词块每64 lane占32 B，只有低8 B是逻辑
位，其余24 B是有定义但不作为lane的padding；padding由真实count-mode Vector Dup
（`mask_zero`）定义，并与mask-free packed compare形成required same-V RAW/WAW，由
通用同步产生实际`PIPE_V`。compare目的view是保守整块`RW` preserve update：只写低8 B
lane word，R表示保留旧padding并依赖真实initializer的ordinary read admission，不让
单次compare单独授权未初始化packed读取。packed NOT是真实count-mode Vector `vnot`，
engine/effects/packet与实际调用一致。不能把scalar清零或其它引擎的工作藏进声明为
Vector的task。
材料化谓词跨 Domain、Stage 或 Iteration 时，描述符与 packed Tile 依赖一起通过
实际 yield 导出。描述符只标识形状和 writer；Tile 依赖仍由普通值绑定、初始化和
同步准入检查真实读取。两者必须指向同一实际 writer 及其精确 allocation。
沿 yield 查询 writer 不证明循环非空、LAST 存在、ready 或 ownership；空循环
不能通过转发描述符制造可读的谓词值。

谓词按有界128位精确整数求值，极端signed起点
不溢出、不shift-by-64；`mask=False` lane保留旧值，不构成完整定义。

GM 缓存别名不创建新的物理存储身份。source 的 cache 提示随同一 root 保留到
physical IR；bypass launcher 用目标 runtime 查询的地址别名绑定原始分配，查询
失败直接返回错误。所有 byte bounds、alias、ownership、逻辑值和同步关系仍针对
同一 allocation，不能用不同虚地址规避真实重叠。

Set只能捕获合法source里程碑，不能把后续发射也当已完成；Wait只取得真实配对贡献。普通queue Wait和Scalar/cross controller阻塞分别建模。数据顺序、flag空闲和无死锁是不同保证。

mode-2 Wait 阻塞接收端 controller。已有完整 quorum 的 capture/acquire 可以覆盖
同一接收端 controller 上随后其他 pipe 的访问，但须保持 producer pipe、全部
参与者、激活和真实端点顺序。该覆盖不新增完成关系或改变原通知的 ownership。
local Wait 只阻塞目标 queue，不能以相同理由覆盖另一 pipe 的 consumer。

组合完成覆盖逐条保存合并前需求的端点和证明对应；证明只落到实际保留的
capture/acquire。已删除动作不能支持后续删除，保留支持动作也不能再被删除。
同一通知的多个 writer、分支和调用分别检查完整激活，不能只证明代表端点。
完成证明不授权旧 Wait 已消费；消费和资源许可在最终通知集合上另行建立。

release定位全部相关最后reader：MTE1读完L1可以释放L1而MAD/FIX继续；MTE2读完共享GM可以释放GM而Vector继续。不能以整个计算完成替代最后读取，不能因不同tensor名称忽略别名。

条件首clear和后续RMW保持各自读写合同。private packet的独立性须由精确cell、已定义读、跨代RAW及外部可见状态等前提建立；private或admission成功本身不够。只允许已经论证独立的生命周期按给定dispatch组织。

分支合流不能拼接互斥路径的完成证据。零次路径保留进入前已有的RAW/WAR/WAW。
数值初始化可由同一 participant 的完整有序分支树建立；变化条件先在共同逐代
footprint 闭合，再检查实际调用非空并汇总循环，不合并不同 AIV 的独立条件。
这项定义事实不提供异步完成、所有权退休或事件消费证据。
区域关系未知时 AliasRegion 保留源码内存顺序，不声称精确 RAW/KILL，也不进入
稳定 cell credit 或 private packet 重排。外部 write_disjoint 合同只用于没有
内部 reader 的输出，不免除 scratch 复用、输入退休或双 AIV quorum。
完整合同见[memory-contracts.md](memory-contracts.md)。

## 5. 保持与失效

| 改变 | 可以保持 | 需要建立/更新 |
|---|---|---|
| 合法纯整数化简 | 数值、地址、effects和执行关系 | 规则适用的类型、溢出、除法与支配前提 |
| 终端单时钟 lowering | 每个actor的指令/地址trace、native身份和既有同步合同 | count/stage到arith/scf的坐标、存在域及控制对应；不再使用旧stage作为输出控制 |
| 连续条件块及互斥通知合并 | 逐actor指令次序、重复次数、事件ID及native effects | 共同入口的标量定义性、条件并集等价及通知分支互斥 |
| subview/layout/范围变化 | 只有明确保持的项 | 涉及的访问、别名、依赖及协议 |
| stage/issue order/参与者变化 | 无关的类型和数学定义 | 涉及的进展、完成、通知及生命周期 |
| 一一对应ID改名 | 逻辑通知及已有执行关系 | bank合法性、冲突及配置预留 |
| 共享两个AIV的相同代码 | 各actor的有序指令、guard、重复次数、私有bank及quorum贡献 | 同一插入点的两个完整动作序列一致；协议诊断名不参与指令等价 |
| width/ordinal/共享成员变化 | 未变的native effects | 相关配对、复用、资源与首尾状态 |
| 删除/移动native或改变guard | 不能盲用原执行事实 | 相关访问、贡献、配对和边界 |
| 生成ASC | 已确定计划 | 仅检查物化对应，不重新选择协议 |

省略同步使用已有且不依赖该同步的覆盖事实；不先试删再全图验证。通用DCE/CSE/移动必须尊重异步effects与ownership。

## 6. 已有数学与原式合同应保留

`IntegerFacts`的答案依赖不可变输入环境和完整有序前提；不同环境、绑定或前提不互借，未知不是安全，异常不缓存失败答案。数学规范化不等于原式可以求值。

原式没有新的 unsigned 节点，或全部节点已由实际支配 SSA 提供时，定义性检查
直接结束；不为此求解各 guard 的真假。仍有未检查节点时，保持原始检查次序，
只有需要跳过后续部分求值时才查询当前前缀是否必假。常量、相同数学值及
完整输入 facts 的保守区间可以直接建立数学关系；其他关系交给完整前提的
SMT 查询。每个 query 使用独立 assertion scope，不将上一查询前提留在下一查询。
数学编码可对正整数常量除数的 ceildiv 使用等价的单次除法，也可消去
`floor((x mod (d*m))/d) mod m` 中在完整输入事实下已知 `d,m>0` 的外层余数。
这些恒等式只改变无限精度查询的表示；定义性及机器范围继续遍历原始 DAG，
不引入机器上的 `x+d-1` 求值，也不豁免原始 `d*m` 的溢出。
固定 solver model 对原始谓词的求值可以复用，但只有同一 model 满足本次
全部前提和完整目标时才能提供 witness/counterexample，不能作为 universal proof。

封闭数学编码器从已编码比较和布尔子式构造 Z3 BoolRef packet，检查其布尔类型与
共同 context，使用所固定 Z3 版本的同一个原生 AST 构造器。空式、单元素式、
操作数顺序、重复项和嵌套保持，不重复执行通用 Python 接口的算术 sort 强制转换。
这不改变输入前提、查询、求解预算或答案；原式定义性和机器范围继续独立检查。

编译器新构造的无符号floor/ceil/rem必须保持有序guard的定义性：例如先证明选择非空，再求LAST；不能先代数消去或合取后续条件，掩盖此前非法求值。前提变化时重新检查受影响原式；同一有序合取内，后续前缀只追加已成立条件，先前已检查的原始 DAG 节点可保持定义性。该复用不跨 guard 调用，不借后续条件证明更早表达式。运行期guard的短路次序不由数学集合排序决定。

输入/输出IR由本次编译拥有，受控变换保持的事实可继续使用；外部或被修改的物理IR不能借旧结果。当前公共compile拒绝外部物理快照，debug_verify也只有xDSL结构检查，不能宣传为独立同步证明。

DSL 定义通过已知 parser 新建并检查的 source IR 直接交给本次 lowering 拥有，
不再重复复制和结构验证。外部 source ModuleOp 仍复制并验证，不读取可变 verified
标记；两者随后进入同一 native、内存、值和同步链。

逻辑pipeline只保存次数、实际stage偏移和发射政策，每个stage只有一个ordinal。
同步确定后，终端lowering将时钟和存在条件变成真实arith/scf，并保持两个AIV共享body。
原始依赖与生命周期仍属于逻辑域；后端只消费保留的native几何及最终SSA，不读取旧控制节点。
同步后逻辑模块和终端模块分别拥有 IR；一次显式投影映射 native 身份和已准入
的 SSA 范围。逻辑分析记录不随终端重写变成失真的最终事实。后端只消费
`ExecutablePlan` 中的实际 SSA、静态 geometry/packet 和只读范围，不持有源
地址或逻辑 frame。终端 native 对应检查不是重新运行同步证明。
通知物化时，计划中的字面常量只复用支配的 `arith.constant`，缺少时创建常量。
数学值相同的 actor SID 块参数或算术结果不能替代该表示；动态表达式仍复用真实
支配 SSA。这样纯标量去重不会经由常量别名把静态 event 指令改成动态 ID 指令，
也不改变原始 guard、通知动作、编号或事件资源政策。
源 SSA 范围在物化前建立一次，物化和终端投影共同消费。新算术优先用实际
作用域的完整区间检查 unsigned 前提和 int64 范围，未知继续必要的数学证明。
只有两侧已经支配插入点（或是合法字面量）的比较，才能由该插入域的前提
消去；不因恒等式删除尚未检查的部分求值或溢出。作用域、actor、坐标和
插入点相同的端点共享支配值集合，不借其他代次或后续 load 的定义。
标量定义移动到更大执行域时，原域的范围事实失效；目标域重新建立完整求值范围后才能外提。

最终 `simplify_scalars` 在静态原式诊断和同步物化之后执行，可选范围不作为准入。整数 CSE 只复用
真实 SSA 支配定义；不变量仅在全部操作数支配目标位置、且目标域内求值合法时
外提，不从 sibling branch 借用定义，不处理浮点、vector 或异步操作。局部区间
来自实际 runtime 合同、actor owner、循环边界及有序 SCF 分支。已删除全程序 host extrema
及由此反推全局参数上界的链路；已有源 SSA 的局部类型事实不授权更早求值。signed 运算的
完整区间可能溢出时退回未知，不重结合算式。
unsigned 单位除法/余数使用精确位向量恒等式，不回溯豁免原 DSL 定义性检查。
本次整数pass保留native、地址、通知和控制身份，SSA操作数只作等价替换。
arith 的静态类型归属和类 traits 按 IRDL 类复用，实例的结果数和整数 operand/result
类型仍逐项检查；此表不保存程序、范围、effects 或优化结果。
继承的标量范围只在本作用域改变了相关 DAG 输入时重建。已观察的源范围
发生变化时使已有结果失效，包括把定义移出原 guard 后扩大范围。
正常链先完成一次域内标量化简和 DCE，再由条件变换后的最终清理统一完成语法 CSE。
随后的`simplify_conditions`在最终实际SSA上合并条件：保留每层loop/stage的
独立身份和祖先前提，以差分约束保持`g<N`的关联，不枚举父子循环阶段组合。
局部条件并集采用区间、吸收及共识化简；有限event slot保留余数之间的关联。
未知条件保留。只有标量在共同入口完整可求值时才能提前计算，不能将原始
unsigned LAST/rem移到其定义域之外。

条件 packet 的单个 case 只按固定范围求交、去除区间包含已确定的谓词；
不为每个输入 guard 单独搜索非空性或相关冗余条件。通知分组之后，
完整 union 的覆盖查询决定公共 guard。event 的仿射 ID 映射须覆盖实际保守范围；
范围不足以唯一决定映射时保留精确的剩余条件选择，不逐个猜测有限 residue。
实际条件仍使用真实外层前提及完整 atom 范围、modulo 和 base 关系；unknown 保留原条件。
原式定义性、共同入口 totality/机器范围与事件配对独立检查，该化简不建立
ownership、完成或通知消费关系。

仅供控制和通知使用的二次幂unsigned余数可精确改写为全定义的位掩码；地址、
loop header、buffer extent及yield依赖的原式不改写。新表达式沿已有scope增量
更新CSE、范围和外提事实，不重跑原式准入、同步选择或整套标量分析。
连续native/内存/同步块共享相同条件；相同路由和参与者的连续事件通知只有在
分支互斥时才合成一条，并按实际slot生成event ID表达式。重叠分支继续保留
各次 Set/Wait 贡献及原次序。连续相同 pipe fence 在没有新 native 或其他通知、
且标量在公共入口可求值时，可按幂等规则合并其执行域并保留全部来源；
native、循环及未知 effects 结束该区间，不跨边界推断完成。

终端 cross ID 的二选一常量分支使用已建立的互斥域及完整外层覆盖，逐激活选择
原来的同一个 ID 和同一次通知；相同作用域内的 guard 必须完整可求值。它不移动
通知、不引入新消费或完成关系，不重新选择编号或合并两个 AIV 的贡献。

后端只消费最终IR。每个pipeline/family打印一个时钟循环和一份stage body，
相邻同offset stage共用存在guard；N=0时所有stage均不执行，短循环使用同一规则。
ASC不输出阶段枚举、阶段参数、stage lambda或独立的序言/稳态/收尾/短路径。
纯整数单次使用表达式可在真实使用点内联，保留短路求值的顺序。
生成的launcher仅转发stream、buffer指针和runtime标量并发射kernel，保留现有
共享库ABI。尺寸、倍数、整数溢出、空指针、对齐和buffer重叠的运行时检查
直接删除，不移动到其他入口。输入声明结构、静态非法、lowering 对应和同步资源检查保留；
调用方满足声明的 shape/alias/原语对齐及实际执行的合法索引和算术前提。

## 7. 测试与正常路径

规则论证和回归放在开发阶段。每程序必要的输入及适用前提检查保留，已建立事实由下游直接消费。独立有限执行模型、实际IR破坏、CANN与设备资格见[验证口径](validation.md)，各自不外推其证明范围。

Path/三值/搜索树实验和相应内部形状测试已删除。实际死锁、unsigned
定义性、祖先代次、event 消费及双 AIV 反例保留在当前表示中；不保留第二套算法。
