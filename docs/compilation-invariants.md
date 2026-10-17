# 编译不变量与检查职责

2026-10-17。以下合同由当前唯一 source 链路保持。具体生成规则、前提及
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

当前 bounds 按完全相同的 root、原始 axes、offset DAG 和 native geometry
建立 view 合同；每个不同 active domain 仍有自己的义务。所有真实 AddressOp
各自保留映射，不以相同扁平字节范围代替原轴检查。host 以不变 lexical domain
构造标量范围 DAG，保留 owner/iteration 的联合区间；原始除法和源码算术仍独立
检查。Expr 的 free variables 在节点构造时建立，不依赖 range/guard 环境。

公共编译在独立 expression scope 内驻留完全相同的原始 `(kind, args)` 节点，
退出、异常或取消时释放节点表，嵌套 scope 恢复外层。它不保存 range、guard、
alias 或完成证据；数学等价的不同原式不会在此合并。所有原始 SSA 使用仍检查
各自的有序定义性。查表前检查 key 类型，不能让 Python 的 `True == 1 == 1.0`
将非法常量误认为已构造的合法整数。确定性排序继续使用结构键。

BoundPlacement、LogicalNotification、StructuredNeed 的 hash 在各自不变记录中
按需保存，相等语义仍由全部原比较字段决定；hash 碰撞不代表相等。记录替换
重新计算自己的 hash，既有 hash 不授权修改 guard、成员、width 或共享关系。

## 2. 保证的目标归属与当前缺口

| 不变量 | 目标建立职责 | 当前应核对的代码/缺口 |
|---|---|---|
| 类型与runtime输入合同 | frontend/admission；输入条件由调用方满足 | 编译器检查合同声明；生成的kernel和launcher不验证调用方输入 |
| 完整effects | native lowering/目标原语合同 | 包括scratch、padding、SPR、读写及真实退休 |
| alias/ownership | 物理访问与owner分析 | 保留root/view/范围/slot/generation及每个AIV的私有性 |
| 逻辑值版本 | logical_valueflow | 精确RAW、条件KILL/UPDATE与未绑定输入，不能改绑到dispatch最近writer |
| 物理复用 | storage_lifetimes与实际dispatch | 全部相关最后reader，不能以控制退出代替完成 |
| ready/release覆盖 | notification binding | 真实 capture/acquire、独立覆盖及原始 ownership edges |
| 配对与事件消费/复用 | 生成器的域、cell credit 与 handback 规则 | 域只检查一次；物理 lifetime 归纳或明确消费回传，rank 不提供消费证据 |
| 可进展 | queue_order 的分层生成准入 | 零 step phase DAG、正 step 前进；父层只组合 child may-block 边界 |
| bank/ID/width/quorum | 分配器消费已建生命周期，建立新增资源约束 | 不默认ID共享只是改名；不每候选重新搜索整套路径 |
| 输出对应 | 物化与后端 | 不重选同步，不改native/控制/guard语义 |

上表不表示所有已存在的前置分析都应重写。先对源码确认真实职责，只删除被替代的机制。

## 3. 循环组合必须保持

- 每个合法逻辑代次在每个stage恰好执行一次；坐标替换同时作用于地址、guard、slot、metadata和捕获值。
- N=0透传原状态；N=1的FIRST/LAST重合不重不漏；短循环不能假借稳态条件。不得按runtime长度展开。
- 父层绑定child的真实外部端口和祖先代次，不重新求其内部路径历史；不同代次不能共用裸ordinal条件得出矛盾。
- 循环控制退出不证明native完成。跨loop的旧reader与未消费event仍通过对应端口传递，不加统一loop drain。
- 每个pipeline输出一个物理时钟循环，stage以`g=j-offset`及其存在条件执行。首末条件属于同一生命周期，不能独立seed/drain。kernel最终收尾单独处理。
- buffer slot数、stage偏移、event width互不代替；跨AIC/AIV使用实际参与者与GM可见性，不能丢掉第二个AIV贡献。

## 4. 物理、数值与硬件语义

搬运按目标 native 区分对齐、步长单位与有效字节。普通 DMA 的 32 B 块约束
不能外推到矩阵 ND2NZ/FIXPIPE，也不能用矩阵的元素对齐许可放宽普通 DMA。
显式补零的封闭 packet 保留 fill→MTE2 fence→ND2NZ 次序；FIXPIPE 的 dtype
转换只改变目标元素宽度，保留 FP32 累加器的实际 fractal 读取与退休关系。

GM 缓存别名不创建新的物理存储身份。source 的 cache 提示随同一 root 保留到
physical IR；bypass launcher 用目标 runtime 查询的地址别名绑定原始分配，查询
失败直接返回错误。所有 byte bounds、alias、ownership、逻辑值和同步关系仍针对
同一 allocation，不能用不同虚地址规避真实重叠。

Set只能捕获合法source里程碑，不能把后续发射也当已完成；Wait只取得真实配对贡献。普通queue Wait和Scalar/cross controller阻塞分别建模。数据顺序、flag空闲和无死锁是不同保证。

release定位全部相关最后reader：MTE1读完L1可以释放L1而MAD/FIX继续；MTE2读完共享GM可以释放GM而Vector继续。不能以整个计算完成替代最后读取，不能因不同tensor名称忽略别名。

条件首clear和后续RMW保持各自读写合同。private packet的独立性须由精确cell、已定义读、跨代RAW及外部可见状态等前提建立；private或admission成功本身不够。只允许已经论证独立的生命周期按给定dispatch组织。

分支合流不能拼接互斥路径的完成证据。零次路径保留进入前已有的RAW/WAR/WAW。动态alias/ownership无法证明时要求可验证输入合同或诊断拒绝。

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

无符号floor/ceil/rem必须保留原始有序guard的定义性：例如先证明选择非空，再求LAST；不能先代数消去或合取后续条件，掩盖此前非法求值。prefix变化时重新检查受影响原式，同一prefix内共享DAG只需访问一次。运行期guard的短路次序不由数学集合排序决定。

输入/输出IR由本次编译拥有，受控变换保持的事实可继续使用；外部或被修改的物理IR不能借旧结果。当前公共compile拒绝外部物理快照，debug_verify也只有xDSL结构检查，不能宣传为独立同步证明。

逻辑pipeline只保存次数、实际stage偏移和发射政策，每个stage只有一个ordinal。
同步确定后，终端lowering将时钟和存在条件变成真实arith/scf，并保持两个AIV共享body。
原始依赖与生命周期仍属于逻辑域；后端只消费保留的native几何及最终SSA，不读取旧控制节点。
标量定义移动到更大执行域时，原域的范围事实失效；目标域重新建立完整求值范围后才能外提。

最终 `simplify_scalars` 在原式、整数范围和通知定义性准入之后执行。整数 CSE 只复用
真实 SSA 支配定义；不变量仅在全部操作数支配目标位置、且目标域内求值合法时
外提，不从 sibling branch 借用定义，不处理浮点、vector 或异步操作。局部区间
来自实际 runtime 合同、actor owner、循环边界及有序 SCF 分支。合法int64索引
求值隐含的uniform乘积上界直接收紧参数范围，不重新建立地址或同步证明；signed 运算的
完整区间可能溢出时退回未知，不重结合算式。
unsigned 单位除法/余数使用精确位向量恒等式，不回溯豁免原 DSL 定义性检查。
本次整数pass保留native、地址、通知和控制身份，SSA操作数只作等价替换。
随后的`simplify_conditions`在最终实际SSA上合并条件：保留每层loop/stage的
独立身份和祖先前提，以差分约束保持`g<N`的关联，不枚举父子循环阶段组合。
局部条件并集采用区间、吸收及共识化简；有限event slot保留余数之间的关联。
未知条件保留。只有标量在共同入口完整可求值时才能提前计算，不能将原始
unsigned LAST/rem移到其定义域之外。

仅供控制和通知使用的二次幂unsigned余数可精确改写为全定义的位掩码；地址、
loop header、buffer extent及yield依赖的原式不改写。新表达式沿已有scope增量
更新CSE、范围和外提事实，不重跑原式准入、同步选择或整套标量分析。
连续native/内存/同步块共享相同条件；相同路由和参与者的连续通知只有在
分支互斥时才合成一条，并按实际slot生成event ID表达式。重叠分支继续保留
各次贡献及原次序；不能用二值flag或fence的直觉去掉重复指令。

后端只消费最终IR。每个pipeline/family打印一个时钟循环和一份stage body，
相邻同offset stage共用存在guard；N=0时所有stage均不执行，短循环使用同一规则。
ASC不输出阶段枚举、阶段参数、stage lambda或独立的序言/稳态/收尾/短路径。
纯整数单次使用表达式可在真实使用点内联，保留短路求值的顺序。
生成的launcher仅转发stream、buffer指针和runtime标量并发射kernel，保留现有
共享库ABI。尺寸、倍数、整数溢出、空指针、对齐和buffer重叠的运行时检查
直接删除，不移动到其他入口。输入声明、原式定义性、静态越界及同步准入的
编译期检查保留；调用方满足声明的shape/alias/对齐及合法索引求值前提。

## 7. 测试与正常路径

规则论证和回归放在开发阶段。每程序必要的输入及适用前提检查保留，已建立事实由下游直接消费。独立有限执行模型、实际IR破坏、CANN与设备资格见[验证口径](validation.md)，各自不外推其证明范围。

Path/三值/搜索树实验和相应内部形状测试已删除。实际死锁、unsigned
定义性、祖先代次、event 消费及双 AIV 反例保留在当前表示中；不保留第二套算法。
