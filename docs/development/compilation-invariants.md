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

目标原生指令的静态描述由 `NativeSpec` 统一保存，逻辑 memory template 和
终端 native 共享同一份不可变值；实际地址和循环 read condition 各自绑定真实
SSA/代次。opcode、参数或地址 operand 数量改变必须更新描述，不能沿用旧
几何输出另一条指令。packet 仍对可信注册表核验 identity 与 closed mask；
描述共享不取代原生硬件检查、ownership 或同步资格。

源程序实际执行的访问及算术定义性属于 DSL 输入合同。编译器不证明全部
动态原轴/byte bounds，也不要求所有 index 拥有 uniform extrema；静态非法
原式仍在代数化简前诊断。typed SSA 保留实际执行语义，`Expr` 仅是可选分析视图。

存储检查保留真实 allocation、原轴线性化、native 子视图/行间空洞、对齐及容量。
源索引作为 SSA 原子参与对应，不展开任意数据计算；不同 active domain 仍保留
自己的控制与 effects，但不再逐域证明调用方的数据合法性。

AST 和 lowering 保持条件、mask、求值顺序。已有且支配插入点的 SSA 直接复用；
源动态定义性不再重复求解。编译器新建的通知/时钟/地址算术另有保持义务：
非空前缀先于 LAST、短路前项先于后项，不将 load 或部分函数外提到更大域。
新通知算术在真实物化位置检查定义性和 int64 范围；非负加乘的常量取模可先
约减操作数，避免事件 rank 在取模前溢出，不改变 ID、width 或共享政策。

事件域保留每个 guard 求值前的真实 metadata 前缀，不能以分支/body 内的 load
证明自身激活。在已经假设端点实际执行的子问题内，可使用其局部输入合同；
不能将其升级为循环非空、端点存在或完成证据。条件值在早期 publisher 尚未
定义时，使用既有平衡的 queue-prefix 入口/出口，不提前读取条件，不添加统一 drain。

空间 enclosure 可由合法源访问及已检查的布局对应建立，无需再证明源 bounds。
它只提供保守空间分离，不改变真实 native 字节、不建立初始化、KILL 或 credit。
区域关系未知或可选分析预算耗尽，保留 AliasRegion 及必要源码顺序；必要协议
检查仍拒绝无法实现的 ownership、quorum、进展或有限事件资源。

二维前缀保留实际 count SSA 与静态容量，native 使用同一个 count。部分写不
授权完整初始化；显式补零的 GM 部分读和 L1 完整写分别建模，禁止整块 DMA 多读。
L1 NZ allocation 的物理 shape 和 C0 经统一布局传递，子视图不改变 leading M。
初始化按真实矩形联合及 writer 来源建立；相邻部分写可以拼满，矩形包围盒不能
填补孔洞。完整定义不提供 ready、ownership KILL、reader 退休或事件消费事实。

公共编译在独立 expression scope 内驻留完全相同的原始 `(kind, args)` 节点，
退出、异常或取消时释放节点表，嵌套 scope 恢复外层。它不保存 range、guard、
alias 或完成证据；数学等价的不同原式不会在此合并。原始 SSA 的求值域继续保留，
新求值另检查有序定义性。查表前检查 key 类型，不能让 Python 的 `True == 1 == 1.0`
将非法常量误认为已构造的合法整数。确定性排序继续使用结构键。

BoundPlacement、LogicalNotification、StructuredNeed 的 hash 在各自不变记录中
按需保存，相等语义仍由全部原比较字段决定；hash 碰撞不代表相等。记录替换
重新计算自己的 hash，既有 hash 不授权修改 guard、成员、width 或共享关系。

## 2. 保证的目标归属与当前缺口

| 不变量 | 目标建立职责 | 当前应核对的代码/缺口 |
|---|---|---|
| 类型与runtime输入合同 | frontend/admission；输入条件由调用方满足 | 编译器检查合同声明；生成的kernel和launcher不验证调用方输入 |
| 完整effects | native lowering/目标原语合同 | 包括scratch、padding、SPR、读写及真实退休 |
| 存储实例与alias | 物理访问、pointer合同与可选分区分析 | group/peer独立性属于源语义；保留root/view/slot/generation及AIV私有性，未知区域保留源码内存顺序 |
| 源值语义 | frontend/Tile 与 admission、局部 packet 合同 | 不要求全程序精确 writer 闭合；dispatch 不能改变原读取值；may-write 不宣称完整定义 |
| 物理复用 | access_order 与局部 packet 的实际 dispatch word | 保留全部 pending reader；控制端口只提供插入位置，真实 Set/Wait 才捕获/获取前缀 |
| ready/release覆盖 | notification binding | 真实 capture/acquire、独立覆盖及原始 ownership edges |
| 配对与事件消费/复用 | 生成器的域、cell credit 与 handback 规则 | 域只检查一次；物理 lifetime 归纳或明确消费回传，rank 不提供消费证据 |
| 可进展 | queue_order 的分层生成准入 | 零 step phase DAG、正 step 前进；父层只组合 child may-block 边界 |
| bank/ID/width/quorum | 资源规划建立协议及复用约束；编号分配器检查冻结的有限映射 | 不默认ID共享只是改名；counter 时间复用覆盖每个旧receiver至每个新publisher的消费回传并取最大积压；FIFO需逐端同序、完整quorum及求和积压≤15；编号候选不触发域或handback求解 |
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
谓词按有界128位精确整数求值，极端signed起点
不溢出、不shift-by-64；`mask=False` lane保留旧值，不构成完整定义。

GM 缓存别名不创建新的物理存储身份。source 的 cache 提示随同一 root 保留到
physical IR；bypass launcher 用目标 runtime 查询的地址别名绑定原始分配，查询
失败直接返回错误。所有 byte bounds、alias、ownership、逻辑值和同步关系仍针对
同一 allocation，不能用不同虚地址规避真实重叠。

Set只能捕获合法source里程碑，不能把后续发射也当已完成；Wait只取得真实配对贡献。普通queue Wait和Scalar/cross controller阻塞分别建模。数据顺序、flag空闲和无死锁是不同保证。

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

封闭数学编码器从已编码比较和布尔子式构造 Z3 BoolRef packet，检查其布尔类型与
共同 context，使用所固定 Z3 版本的同一个原生 AST 构造器。空式、单元素式、
操作数顺序、重复项和嵌套保持，不重复执行通用 Python 接口的算术 sort 强制转换。
这不改变输入前提、查询、求解预算或答案；原式定义性和机器范围继续独立检查。

编译器新构造的无符号floor/ceil/rem必须保持有序guard的定义性：例如先证明选择非空，再求LAST；不能先代数消去或合取后续条件，掩盖此前非法求值。前提变化时重新检查受影响原式；同一有序合取内，后续前缀只追加已成立条件，先前已检查的原始 DAG 节点可保持定义性。该复用不跨 guard 调用，不借后续条件证明更早表达式。运行期guard的短路次序不由数学集合排序决定。

输入/输出IR由本次编译拥有，受控变换保持的事实可继续使用；外部或被修改的物理IR不能借旧结果。当前公共compile拒绝外部物理快照，debug_verify也只有xDSL结构检查，不能宣传为独立同步证明。

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
标量定义移动到更大执行域时，原域的范围事实失效；目标域重新建立完整求值范围后才能外提。

最终 `simplify_scalars` 在静态原式诊断和同步物化之后执行，可选范围不作为准入。整数 CSE 只复用
真实 SSA 支配定义；不变量仅在全部操作数支配目标位置、且目标域内求值合法时
外提，不从 sibling branch 借用定义，不处理浮点、vector 或异步操作。局部区间
来自实际 runtime 合同、actor owner、循环边界及有序 SCF 分支。已删除全程序 host extrema
及由此反推全局参数上界的链路；已有源 SSA 的局部类型事实不授权更早求值。signed 运算的
完整区间可能溢出时退回未知，不重结合算式。
unsigned 单位除法/余数使用精确位向量恒等式，不回溯豁免原 DSL 定义性检查。
本次整数pass保留native、地址、通知和控制身份，SSA操作数只作等价替换。
arith 的静态类型归属按 IRDL 类复用，实例的 traits、结果数和整数 operand/result
类型仍逐项检查；此表不保存程序、范围、effects 或优化结果。
随后的`simplify_conditions`在最终实际SSA上合并条件：保留每层loop/stage的
独立身份和祖先前提，以差分约束保持`g<N`的关联，不枚举父子循环阶段组合。
局部条件并集采用区间、吸收及共识化简；有限event slot保留余数之间的关联。
未知条件保留。只有标量在共同入口完整可求值时才能提前计算，不能将原始
unsigned LAST/rem移到其定义域之外。

条件 packet 的单个 case 检查实际可达性、按固定范围求交并消去已证明冗余的谓词；
删除非平凡谓词须由该 case 的其余条件及完整输入前提蕴含。通知分组之后，
完整 union 的覆盖查询决定公共 guard。event 的剩余选择仍使用
真实外层条件及完整 atom 范围、modulo 和 base 关系；unknown 保留原条件。
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
