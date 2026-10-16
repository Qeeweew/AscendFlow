# 编译不变量与变换合同

这是新重构必须满足的合同，**不表示替代实现已完成**。现状只在[状态页](refactor-progress.md)登记，算法见[设计](design.md)。

## 1. 事实只由一个阶段建立

每个变换说明：输入前提、建立的保证、保持的保证、可能失效的保证。规则的正确性论证和实现回归做在开发阶段；每次编译只建立该输入必需的前提和新约束。

normal 不反复重建整个同步图，也不依赖外部可伪造的 verified 属性。输入由一次编译私有持有；当前公共编译仅接收 source；外部物理 IR 和修改后的产物不能沿用这次调用的事实，独立物理输入验证入口尚不作为已支持产品能力。不要先建设通用 proof DAG 或任意可变 IR 缓存框架。

| 不变量 | 唯一建立职责 | 下游如何使用 |
|---|---|---|
| 输入合法 | 前端/admission：类型、runtime合同、坐标、constexpr | launcher 检查需要运行时成立的前提；后续不重新解析源码 |
| effects 完整 | native lowering + 目标原语合同 | 访问包含 scratch、padding、SPR、读写及退休点 |
| alias/owner | 物理访问分析 | root、view、范围、owner 与 slot/generation 不能丢失 |
| 数值版本 | logical valueflow | read 的精确定义、条件 KILL/UPDATE、代次和未绑定的输入 |
| 物理复用 | storage lifetimes | 同一实际 cell 的相邻覆盖及全部末 reader；不得重绑数值版本 |
| 排步固定 | pipeline lowering | actor-local step、logical generation、stage、issue order |
| 循环等价 | PLE 构造与边界组合 | 实例和词法次序不重不漏；零次 identity；回边正确 |
| 完成与可见性 | typed 协议构造 | 只消费已取得的 ready/release/capture 事实 |
| 通知资源合法 | 资源分配 | Set/Wait 配对、consume→reuse、bank/ID/width、quorum |
| 可进展及生命周期 | 协议/资源的进展合同 | 不等待未来无法派发的工作；seed/drain 与真实端点匹配 |
| 数值及性能政策 | 所有受控变换保持 | 无额外搬运/buffer/barrier，精度与固定排步不暗改 |
| 输出对应 | 物化和 emitter | 生成位置、guard、参与者、ID 与已确定计划一致 |

IRDL verifier 负责结构和类型，不能替代异步正确性。有限模型只覆盖指定规模和硬件假设，不能称任意 N 的形式化证明。

用户已明确不要求重构后的ASC文本、loop形态或event编号与旧版本相同。合法循环改写、有效域化简和冗余同步省略均可直接实施；验收看实际访问/数值合同、buffer复用、派发与重叠以及编译/设备证据。抽取原语时的逐字对照只是该局部迁移的检查手段，不能成为后续控制变换的限制。

## 2. PLE 与循环组合的证明义务

1. 对每个 stage，存在且只存在一次物理 step 执行每个合法逻辑代次。坐标替换作用于地址、guard、slot、metadata 和所有捕获值。
2. 长、短、零次路径互斥且覆盖输入域；FIRST/LAST 重合时不复制指令、不改变词法顺序。
3. entry→exit 透明传递不产生新的完成事实；region 控制退出不等于内部异步退休。
4. prologue建立已激活stage的合法进入状态；含首尾guard的一步转移保持carried合同。stage填充结束不代表距离大于stage深度的release已经无条件生效。N=S时主体零次重复是identity；短片段不能借用未建立的归纳假设。epilogue只执行真实剩余stage，导出精确未退休访问和通知状态。
5. sequence 的右项从左项真实出口接收状态，loop 回边使用同样的接线规则；子循环不因为被嵌套就换一种语义。
6. 同一 ring 的末代/首代和 event ordinal 使用各自真实时钟。owner、guard、slot 相关性不能通过分别取模后拼错。

普通SCF for保留真实lower/upper/step及iter_args/yield，不能通过附加pipeline属性改变其执行次数。显式流水stage使用本stage的`g=j-s`及`iv=lower+g*step`；纯索引的重建缓存以实际stage作用域为界，不能从别的stage借用含旧g的表达式。Tile yield转发只保留数值writer身份；PipelineLast必须在实际use处证明非空并绑定该loop的末次逻辑迭代，未用的空循环导出不制造值，也不一概拒绝。

正确性论证写在变换旁；normal 只验证实际适用前提并构造结果，不针对每个 N 再跑归纳检查。

## 3. 完成与同步的关键限制

- source SSA use-def 不证明异步完成；issue/queue/native end/累计退休/通知消费不同。
- read→writer的数值版本关系与物理buffer的退休/复用关系是两类义务。实际dispatch不能把读改绑到最近writer；同样，source中的临时写入次序不能未经生命周期证明就被当作所有物理覆盖的唯一顺序。实现已拆分为 logical valueflow 与 storage lifetimes；完整嵌套资格仍见状态页。
- 完整kill、条件首clear和RMW必须各自保留访问语义。允许已证明独立的private scratch生命周期按既定dispatch排列，不自动授权改变外部可见写、累加递推或状态更新；actor-private、buffer名字不同或一条已有admission成功都不足以证明这种独立性。
- 只在目标原语允许的累计前缀内比较进度；多个流水或参与者不可仅取一个最大序号。
- `mad_first` / `mad_cycle` 的条件读绑定其真实 Stage 所在循环的逻辑 ordinal；首 clear 是完整覆盖，其余代次保留输入值。完整 operand 写入并不自动覆盖 contracted family envelope。
- 独立 packet 必须为精确 private cell（当前不含 streaming envelope），内部所有读已有定义，没有跨 packet/跨代 RAW，阶段同 offset 且在实际派发中连续，内部源码次序不变。头部无旧值输入才能证明从完整 kill 开始；只凭 private 或 tensor 名字不够。
- packet 的外层接口必须保持实际首末边界。当前仅 root-private、无外部访问、无外层循环的 scratch 允许改变边界，并重建物理退出义务；其余需要保持边界或拒绝该独立性证书。
- PLE 邻接按 `(generation + offset, lexical)` 建立，允许负逻辑 distance。长循环使用静态 prologue/epilogue 和符号 steady family；短循环只按静态 stage span 分区，不按 runtime count 展开。静态 span/packet/work 超预算诊断失败，无 barrier fallback。
- Set 采集合法 source capture 的事实；Set 自身的效果不使后续源操作自动退休。
- Wait 只在其真实 admission 后传递已消费通知的事实。CV controller Wait 的派发阻塞单独建模。
- release 捕获目标存储的全部相关最后 reader；GM import 完成可以释放 GM，即使 Vector 仍处理 UB。L1→L0 最后读取可以释放 L1，即使 MAD/FIX 未完成。
- 分支合流只承诺实际路径均成立的事实；缺席路径继承输入，不能拼接互斥路径的证据。
- 两个 AIV 的必需贡献必须齐全；OR 可达路径不能替代 AND quorum。
- 省略尚未插入的冗余同步直接使用已有覆盖。删除已存在同步须有不依赖它的替代 witness，并更新所涉 lease；不试删后全图重验。

## 4. 哪些变换使哪些事实失效

| 变换 | 保持 | 需要更新/重新建立 |
|---|---|---|
| 合法纯整数化简、同值地址表达替换 | 数值、地址与访问、同步关系 | 表达式类型/溢出/除法前提，新的支配位置 |
| 保序的 PLE 控制分解 | 逻辑执行实例、native effects、算法顺序 | 坐标和存在条件、边界/回边绑定 |
| fresh frame / subview / layout 改写 | 只有显式证明保持的项 | 若改变物理范围，更新所涉访问、依赖与资源合同 |
| stage/issue order/参与者变化 | 与调度无关的类型和数学定义 | 涉及的完成、进展、通知、生命周期 |
| 一一对应 ID 重命名 | 访问、逻辑通知、完成拓扑 | bank 合法性、碰撞、保留资源和实际 ID 映射 |
| event width/ordinal/共享成员变化 | 不变的 native effects | 相关配对、消费、lease冲突、初始/尾部状态和进展 |
| 新增 capture/acquire 协议 | 既有正向完成事实及未变访问 | 新协议自身合法性、资源与进展，受影响前沿 |
| 删除/移动 native 或改变 guard | 不能假定旧完成仍成立 | 受影响访问、路径贡献、配对及边界 |
| 输出 ASC | 已确定的执行计划 | 只做按合同物化；emitter 不重新选择同步 |

参数合法域缩小与删除操作的某些激活不是同一件事。后者可能移除必要的 Set 或贡献者，不能直接复用原保证。

## 5. event 状态与收尾

同一物理本地 flag 再 Set 前，前次 Wait 必须已消费。slot 可复用不自动证明 flag 可复用；flag 已消费也只证明它捕获的完成关系。跨核 counter 的容量、贡献者和可见性使用独立硬件合同。

逻辑通知跨 PLE 和外层迭代保持同一账本。不能给每段独立 seed/drain；不能为不存在的循环体消费者产生通知。N=0透明传递已有资源状态，外围资源自身的合法seed/cleanup或继承token收尾仍按各自生命周期执行。父子 event 借用保留区间、端点和共享约束，不默认改成独占 ID。

配置 API 自己可能占有 event，例如 AscendC TPipe 的隐式资源。手写参考与原生 ASC 的资源合同分别记录，不能当作无 effects 的分配助手。

## 6. 验证放在哪里

| 层 | 目的 |
|---|---|
| admission / normal | 输入前提一次、依赖一次、必要的新协议/资源/进展一次，消费受控变换结果 |
| 外部同步 IR | 当前公共compile明确拒绝；未来入口必须从实际输入建立事实，不信任证明标记 |
| debug_verify | 当前仅运行最终xDSL结构检查；不得称为独立同步证明 |
| 回归与有限模型 | 检查变换规则和实现，破坏 IR 后发现错误，覆盖短循环/slot/分支等反例 |
| CANN / 设备 | 实际 ASC/硬件契约、数值及性能，不能由 CPU 模型替代 |

现有基于生产 native topology 的有限 BFS 是求解器差分，不是独立硬件语义 oracle。测试必须标出复用了哪些输入事实，见[验证计划](validation.md)。

## 7. 事件资源阶段的独立合同

- `EventDomains` 建立实际端点/队列/参与者与配对。两个端点的条件合取不代表两者各自均存在；必须双向证明各自激活域相同。
- 同流水fence没有Set/Wait配对、event ID或复用序号。通知选择已建立其真实source-before-consumer和累计capture覆盖；资源阶段保留所有端点归属、route与原始有序算术检查，然后只按原consumer guard物化一次。不为fence构造binary事件参数case、窗口或lease，也不借这些case的空交集过滤执行。
- `EventWindow` 必须等价于实际membership；`first`/`last`仅在非空条件内使用，stride必须对应下一次同ID的活跃使用。未知稀疏guard不能被区间摘要吞掉。child到parent的存在条件由已证窗口显式投影，保留代次关联。
- 同一逻辑通知的参数分区覆盖整个时钟链；分区条件必须在本执行组的整个kernel调用内固定（group可用，sid在实际端点已绑定），不能随父循环重入改变。互斥与覆盖来自整数区间的完整划分，不通过采样推断。空分支也保留，并单独检查两端确实都不激活，不能用两端条件的空交集隐藏不配对。
- 每个分区保留自己的窗口、rank、ring宽度和ordinal。物理bank仅为该通知预留一次最大宽度；实际event编号使用当前分区自己的模数，不能把最大预留宽度当成所有分区的编号周期。条件之间没有调用内切换，跨调用仍需真实drain。
- 多residue的相邻活跃成员不一定间隔一个周期。生命周期按当前residue和ring宽度得到精确下一同ID使用；例如period=5、residues={0,3}不能统一使用`width*5`作为复用距离。
- 参数分区只处理实际未决周期比较，具有明确静态工作预算；不按runtime循环次数枚举，不对不同通知做全局参数笛卡尔积。不能证明的动态系数/ownership仍明确拒绝。
- lifetime只使用真正的Wait消费、后续queue/controller admission以及选中通知；Set效果不门控未来source native。有限child出口不会自动产生全体完成。
- 一项复用义务记为A，真实路径的成立条件记为H_i。必须取得每条`H_i ⇒ consume先于next Set`的实际接口链，再证明`A ⇒ OR(H_i)`；某条路径可能存在不能代替完整覆盖。条件包含端点activation和真实队列顺序，不能仅检查peer存在。
- 队列顺序使用共同祖先上的实际`(coordinate+stage, lexical, phase)`。外层严格更晚或同代继续比较内层；同一native的Wait admission早于Set-after-native。每对端点至多产生O(depth)条条件，不按迭代次数展开。local Wait仅建立目标queue顺序，controller gate与actor身份分别保留。
- 复用与有限ID共享共用`EventPathCover`，保存所有覆盖路径及整数划分证据，不保留旧单bridge/steps合同。每层先判断现有路径是否完整覆盖，再扩展下一层；未覆盖的条件direct路径不能阻止寻找所需的间接路径。访问过的实际端点必须同时保留条件，不能用无条件visited丢掉另一slot相位。
- 数学划分只属于当前证明。uniform参数与loop-varying余数都可参与，但后者不能变成调用内不变的`EventCase`。例如`(p*m+g)%2`作为一个相关多项式处理，不能独立枚举p、m、g后拼接互不兼容的事实。大小比较先在原余数范围求mask，不能用模等价替代整数大小关系。
- 原始有序算术前提先通过定义性检查，再做模规约或空域裁剪。路径构造只接受本次EventDomains拥有的canonical端点；不声称可校验任意外部伪造端点。缺失证明返回未覆盖，非法定义性与预算耗尽分别诊断，不能降级成任意barrier。
- 路径与覆盖各有显式静态预算；一项义务的多次覆盖查询共用节点/谓词预算，失败也计入工作量。候选offset是有限接口搜索范围，不是runtime循环展开，也不是硬件容量证明。不能把整个分析声称为线性：条件路径和整数划分最坏会组合增长，当前通过局部作用域、完整覆盖即停和明确上限约束。
- progress用必须/可能/不可能三种顺序关系。必须边用于建立reuse证据；可能阻塞不能漏掉，其成立条件也不能在parent合并时丢失。已由同一循环覆盖的义务不在root重新投影再证明。
- region进展摘要只导出尚未在本层配对的实际通知端口，以及本region的queue输入和queue/controller阻塞输出。内部native与子层接口消去后留在本地witness；父层不重新导入它们。该摘要只用于可能阻塞与进展，不能给lifetime、memory或buffer复用提供完成证据。
- 非Scalar的局部Wait只阻塞目标queue；Scalar/cross Wait才门控实际controller。两个AIV的接口保持独立，不能以一个Vector族节点替代两个贡献。跨调用坐标的平移作用于整条关系及其guard；不同父代不能在child拼成零步边。
- 空循环不产生内部发布/阻塞贡献；外围同queue/controller本有的次序继续保留。未知投影必须保持为可能关系并标记其证明边界，不能因无法消去条件就丢边，也不能把保守摘要用于证明必然完成。
- bank allocation只能使用已有生命周期或具体sharing witness。同一bank找不到ID表示支持的分配策略未得到方案，不能当作硬件容量下界证明。

最终标量阶段只将带 `Pure` trait 的标量整数/index `arith` 根交给 xDSL 通用 canonicalization/DCE。浮点、向量、除法/取余及所有控制、地址、native、通知不在改写入口内；不做 reassociation、CSE 或跨 block 移动。该阶段在通知物化后调用一次，保持这些执行操作与 region 的身份、次序和数值，不重新建立内存或同步事实。规则与接线通过独立短 IR 回归检查，普通编译不反复做前后全图比对。

## 8. 编译期整数事实的作用域

`IntegerFacts` 持有输入 `Fact` 映射的不可变副本。规约只依赖该环境和表达式；范围/比较的查询键还包含完整、有序的前提序列。相同问题复用已有答案，未知结果仍是未知；不同前提、绑定或输入环境不能互借。表随本次事件分析释放，不跨编译缓存结果，不形成外部可提交的证明标记。原来嵌在事件域中的整数证明实现直接移至此唯一职责，不保留旧方法或兼容入口。

完整前提序列现在先归入本 `IntegerFacts` 拥有的不可变 context：不排序、不去重、不预先解释谓词。`prove/bounds` 入口只负责建立 context，唯一递归实现引用它；递归查询键用 context 身份，不再反复哈希整组前提。不同环境拒绝对方的 context。原始 tuple 的建立/比较仍有与前提大小相关的成本，不能把所有查询称为常数时间。缓存同时保留已证明和未知结果；异常不写入答案表。

与前提无关的派生表达式也由本 owner 建立一次：`_constraint` 保留原始带因子的比较约束，`_difference` 严格执行原来的 `bounded(lhs-rhs)`。它们只按原查询顺序惰性建立，不预扫描后续条件，不增加规约层次。不同 context 可以复用相同原式的这些派生结果，但不能因此互借范围或比较答案；不同整数环境仍隔离。模块级旧 `_term` 已移除，只有 owner 内这一份实现。

整数蕴含本身不证明表达式可求值：`IntegerFacts.ordered_definedness`按完整有序guard建立一次定义性事实，先检查当前表达式的无符号操作数，再允许按已验证前提短路。各真实端点必须分别通过，不能先合取两端条件后互借前提，也不能用数学恒假的域掩盖此前非法求值。窗口参数替换与新存在投影消费同一事实能力；后面的guard不能授权前面的无符号除法。端点绑定事实按完整placement、参与者和替换建立；同一绑定的ambient与coordinates从同一结果读取，不重新解释词法域。事实复用不改变实际通知、可能阻塞关系、资源配额或生成的搬运。

一次比较的左右操作数共用按原始节点 identity 记录的成功检查集合，遍历每个共享子图一次；进入下一个比较时更换集合，因为有序前缀已变。只在孩子及无符号前提全部成功后记录节点，失败仍返回原来的首个缺失条件。不能用代数同值、另一环境或后续前缀代替原式检查。

选中子循环坐标的生效条件随该绑定一起保存。父代次平移、选中FIRST/LAST及其条件必须位于同一坐标系；进入选中子循环的词法条件前，先建立该选择非空。例如count=1、period=2、residue=1时该选择为空，`count-1-residue`为负，不能先计算LAST的unsigned remainder再检查非空。编译器应保留这个拒绝或按已有非空条件正确短路，不能补一个错误的“非负证明”。

区间推导保留原表达式的因子和clamp；例如已知`g-1>=0、k>0`时直接推导`(g-1)*k>=0`，不能先展开为`g*k-k`再丢失因子相关性。代数规约用于等价与常量差匹配。min/max允许有依据的单侧界；无符号floor/ceil/rem区间必须先获得分子非负、分母正的同域前提。两种表示分别用于自己的证明规则，不以区间推导未知取消输入定义性检查。

函数atom的数学身份由同一多项式实现自底向上建立：add/mul使用规范多项式，其他函数以kind和实参的规范身份区分；min/max保留交换律，div/rem不得交换实参。键构造将unsigned函数视为未解释函数，不借全局positive事实替代实际有序前缀。声明可整除的动态商产生分数系数时，也不能因无法重建整数Expr而失去实参同余。

`AtomIdentity`只保存不可变数学key，不保存跨查询raw代表。每次查询从自己的原式选择代表；`normalize`和`bounded`共用本次IntegerFacts的key环境及同一多项式操作。公共多项式的atom和返回表达式仍为Expr，不泄漏typed key或伪造的argument。键、共享DAG与多项式工作受原有预算约束；bounded前处理也按唯一节点遍历。原始有序定义性检查始终读取raw guard，因此数学同值不授予原式求值权，也不授权重写source/emitted DAG。

## 条件事件路径的工作量

同一复用查询使用一份条件端点图。反向拓扑可达性只提供必要条件，不能把未知激活删掉；最终证明仍要求路径条件覆盖完整 assumptions，且每步持有真实 queue/controller gate 与已选通知效果。一个实际端点上的不同 guard 保留；只有条件包含或完整覆盖可吸收。

`path_states` 计图节点，`path_guard_states` 计接受的条件状态，`path_edges` 计实际条件边，`path_edge_queries` 计唯一候选边查询。两类状态都受256限制，实际边8192、候选4096、输出路径/长度128。新计数语义不与旧重复扫描数作直接倍率比较。候选范围以实际 parent dispatch step 为准，使用直接 parent stage，不能使用子循环内部 stage 或假定 logical distance 非负。

## 区域进展组合的不变量

`ProgressConditions` 只拥有已经准入的端点上的数学合取，使用同一次调用的不可变整数环境。集合相同的查询复用同一矛盾答案；未知不等于无解，也不产生安全事实。不同 owner 的条件须在当前环境重新建立答案。原始 `ActionGuard` 及其谓词次序一直保留，集合的规范排序不用于 ASC、端点定义性或短路求值。

`RegionConnectivity` 的边代表非空有向路径；不补零长度自环。消去内部节点 k 时仅组合活跃的 i→k 与 k→j，把条件合取作为一条候选，再由 owner 做矛盾排除和 OR 吸收。内部自环只增加合取条件，跳过其任意次重复已有更弱路径，因此不展开；仍保留的边界自环则必须交给进展证明。该规则只证明布尔连通性，不能用于累计 stream displacement 或退休事实。

每个组合步骤结束后，已消去节点及其关联边不再参与后续组合；未消去边界间的非空路径条件保持等价。实际 guard witness 保留原顺序，推导的保守接口不能提升为 exact。局部 stream rank 与 finite/SCC 排除共用条件 owner，但仍遵循各自时钟和 uniform 条件合同；独立 SCC 不做参数笛卡尔积。原有64个 OR 备选、100000次区域组合和64个 SCC 分支预算不变。

稀疏化不保证最坏线性：若 A(i,k) 表示对应边的条件备选数，组合工作量是消元时实际活跃前驱/后继上的 `Σ A(i,k)·A(k,j)`，稠密图仍可达立方规模并叠加条件备选增长。首次矛盾查询仍逐个谓词检查其余前提；规范集合只消除重复查询，不消除新前提的证明成本。所有这些量由静态接口、表达式和条件分区决定，不按 runtime 迭代次数展开。
