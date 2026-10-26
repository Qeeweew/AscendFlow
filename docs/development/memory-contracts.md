# 源程序内存合同与自动同步边界

普通 Kernel 的 group 是独立的 SPMD 执行域。程序负责不产生跨 group 的非原子
读写竞争；编译器保持映射和实际地址，不重新求解任意映射的全局唯一性。
两个 AIV 共用 Vector body，各有私有 UB；普通 peer 访问不能依赖未声明的
跨 AIV 通信。AIC/AIV 的共享 GM 交接继续由实际 effects、双 AIV quorum、
ready 与 reader retirement 生成同步。

这些是源程序前提，不是后端可填写的 verified 标记。当前没有通用跨 group
phase/collective DSL；取消全局分离门禁不表示增加这种执行能力。

## 动态访问与求值的输入合同

调用方保证每个实际执行的源访问都在其逻辑 view 和实际存储范围内。未执行的
分支、mask lane 和零行前缀不产生该侧访问。动态索引、metadata、分支结果与
serial carry 无需附加手写范围证明；编译器不重新求解这些输入数据是否始终合法，
也不生成 clamp 或正常运行时逐元素检查。静态明确的非法访问仍诊断。

源 index 保持 signed int64 可表示域；实际 add/sub/mul 求值不得溢出。现有 `//`、
`%` 的有效输入为非负分子、正除数，除数可以依赖运行期数据。标量整数 `&`/`|`
按对应机器位运算执行；分析不理解其范围时保留 SSA。数学分析式不是
机器求值，允许表示未激活坐标；这不授权实际执行非法原式。静态原式在化简前
检查，动态定义性由调用方满足。编译器新增或移动的求值必须另行保持定义性及
机器范围，不能借后面的 guard 授权前面的 load/除法。

可执行表示是实际 typed SSA。分析无法解释或范围查询未知时保留该 SSA，
不将分析表达式子集变成 DSL 准入条件。标量 `and/or` 短路，标量条件表达式
仅求值选中 region；它们与 tile `where` 和 vector mask 的原语合同分别处理。
嵌套 view 保留组合后的真实索引；动态相对选择合法性同样属于输入合同。

编译器继续检查真实 root、原轴到 native 地址的线性化、布局、子视图、有效域与
native operand 对应、内部物理容量以及目标指令能力。对应检查将源索引作为
真实 SSA 原子，不展开它来证明数据范围。普通连续 GM/UB copy 在 GM 起点的
32 B 对齐未知时，使用 CANN 9.1/2201 `asc_copy_*_align` 字节搬运；当前 payload
仍为 32 B 的整倍数、UB 起点仍须满足指令对齐，因此不额外写入 padding。
已确定对齐的访问保留普通 copy；其他原语的布局、对齐及静态 tile 能力不扩大。

合法访问只提供它实际发生时的局部空间前提，不提供全局范围、循环非空、地址
唯一性、初始化、ownership、完成或事件消费。无法精确分区时保留可能别名顺序。

packed predicate 的布尔组合要求输出与全部输入的真实字节范围分离。
`alloc_mask(..., slots=S)` 中不同的物理槽可以共享一个 allocation root；
source 和 lowering 保留实际 selector，native admission 用同一份地址和完整
已求值 branch/loop 前提建立分离。相同范围或未知重叠不能获得许可。谓词的
descriptor 和逻辑 Tile 仍指向原 writer，另一个已初始化兄弟槽不能替代它。
这份空间许可不提供谓词 ready 或源/目标槽的 reader 退休。

## Tensor 声明

内部 GM 由 `N.alloc_workspace` 声明，其容量、512 B 对齐、设备资源绑定和
生命周期合同见 [GM workspace](../guide/workspace.md)。设备资源在 source 构造前
冻结，打包在同步后物化；两者均不替换真实访问的依赖、初始化或最后 reader 分析。

```python
X: N.Tensor((n, 64), N.f16)
Y: N.Tensor((n, 64), N.f16, access="write", write_disjoint=True)
State: N.Tensor((64,), N.f32, access="read_write")
```

| 声明 | 精确含义 |
|---|---|
| `access="read"`，默认 | 入口数据已定义，kernel 不得写；实际 native effects 也检查此权限 |
| `access="write"` | 入口不承诺初始化；可以读取已经由本 kernel 定义的区域 |
| `access="read_write"` | 入口数据已定义，允许原地更新及多次写入 |
| `write_disjoint=True` | 仅外部可写 Tensor；一次 launch 内该 tensor 的各次实际写入互不重叠，包括不同 group、SID 和逻辑迭代 |
| `disjoint_external=True` | 不同外部 tensor roots 不别名；与同一 tensor 内的写入是否重叠是不同合同 |

write_disjoint 不保证每次写完整 tensor、全部输出都被写到、初始化或 ready，
也不约束下一次 launch。算法和调用方满足声明，底层 C launcher 不检查动态访问。
可选的 [tensor runtime](../guide/runtime.md) 校验外部 shape/dtype/layout、标量合同和
已声明的外部 disjoint 区间，不逐个检查间接地址或推导完整输出初始化。
独立有限模型按 invocation 检查具体重复写，能发现错误声明。

当前只对没有内部 reader 的 write_disjoint 输出消除写间 WAW，无需恢复排列的
代数逆。同一 tensor 若还被读取，保留保守源码顺序，包括 RAW/WAR。输出声明
不传播到临时 UB/L1/L0、共享 workspace 或其他 pointer，物理复用照常分析。

## 访问顺序与数值定义

只读 GM 或已初始化 UB 中的 INT32 / INT64 由同一个 `index_load` 原语读取一个实际元素，effects
分别为 4 / 8 字节，返回有符号扩展后的 index SSA。`load_index` 的 inclusive
`minimum/maximum` 都可省略，缺省使用源 dtype 的完整有符号值域。显式范围是
调用方输入合同，不生成 clamp，也不授予写权限或不同索引的唯一性。
范围事实保留真实支配关系和动态捕获，不能授权此前求值或未执行 load；
它们不代替初始化合流、存储分区或通知生命周期准入。
UB 可通过普通 DMA 批量预取整数元数据；Scalar 读取捕获当前存储代次的值，
后续覆盖 UB 不改变该 SSA。它保留普通 Tile 来源、初始化、MTE2→S readiness
和 S→writer 退休义务，不把可写 UB 当作不可变输入元数据。

`N.store_index(value, out=...)` 通过有序 Scalar issue 写入单个 INT32 / INT64 UB
元素，真实 effects 分别为4 /8字节，不写 GM、不经过浮点转换。调用方保证动态
signed index 值可由目标 signed dtype 表示；静态常量溢出直接拒绝，不生成 clamp、
饱和或静默回绕。它与 f32 `store_scalar` 共用 source ScalarWriteOp，并降为
typed `index_store` native；不赋予 S 写入对其他 pipe 的隐式完成。UB 的定义、
S→MTE3/Vector readiness 以及后续覆盖前的 reader retirement 都由普通访问链生成。

`structured_access` 是实际 native footprint 的唯一输入。StorageCell 表示可选的
精确字节区域；AliasRegion 表示无法排除重叠的同一存储实例。两者都由
`access_order` 建立可能冲突的 RAW/WAR/WAW；RAW 在这里是顺序义务，不宣称
某个 writer 是唯一数值 producer。初始化、源 Tile 借用与原式定义性由 admission
独立负责，部分写不能代替完整初始化。

完整分支树可以共同定义读取区域，但合流只在同一实际 participant 内进行。
条件保留实际 SSA、metadata 捕获坐标和有序前缀；逐代写入须覆盖相同的真实
footprint，再沿可证明非空的 writer loop 汇总。对 child loop 不变的条件，
允许先汇总各分支自己的完整循环。循环的非空性使用调用入口域，不借循环
body 内的范围排除零次路径。不同 AIV 的独立条件、不同 metadata 定义、
缺少的叶分支、错位或不完整写入不能拼接成完整定义。

SID 分支按实际 participant 代入。原生内存事实仍检查完整指令合同，再排除
在该 participant 上能证明恒假的访问；其他条件保留原 SSA 与 activation。
共享 GM 可以联合不同 AIV 实际无条件写入的不同区域，不能镜像未执行的 peer
写入来填满缺口，也不能将未知 metadata 条件当真。私有 UB 的定义不跨 AIV
共享。该投影只建立执行归属，不提供 ready、最后 reader 退休或 quorum。

这里建立的是数值初始化，不是异步 ready、精确 ownership KILL 或事件消费。
真实 writer 若覆盖整个逻辑 allocation，其内任意合法动态读取已具备数值定义，
不再证明该读取的动态包含关系。完整覆盖不包含 allocation 尾部的对齐 padding；
部分覆盖仍须匹配实际读取，不能仅凭源访问合法就认定数据已经初始化。
masked/RW 更新先需要旧值有定义；runtime 部分写不以容量上界提供完整定义。
L1 NZ 在逻辑矩阵坐标中保存实际部分定义及 writer 来源；相邻矩形可以合成完整
定义。其他存储的完整定义前沿不收录 runtime 部分写，避免它按相同静态几何
去重掉后续完整写。所有真实写入仍保留 effects、可能覆盖关系和同步义务。
同一只读源可以有多个 reader，实际覆盖前仍须保留所有 reader 的退休义务。

## L1 NZ 布局与区域定义

Load3D A/B 的矩阵寄存器宽度来自原 allocation 的 leading M，不能以请求的子矩阵
行数替换。b16 A 的行切片及完整/跨行 B NN 面板可在保持同一真实访问字节后选择
单条 Load3D，目标仍是紧凑 L0 平面。该 lowering 继续使用寄存器的真实读写、ready
和退休事实，不扩大 effects，也不把配置值相同当作旧读取完成。

L1 NZ allocation 在物理 `BufferOp.matrix_shape` 中保留原始完整 shape。前导轴是
独立平面/slot，最后两轴是矩阵 M/N；`NZLayout` 统一给出容量、偏移和条带几何。
FP16/BF16 的 C0=16，FP32 的 C0=8，每个行块均为 32 B。L0C 的 FP32 16 列
fractal 使用独立合同，不套用 L1 的 C0。allocation 的 M 按 16 对齐，N 按 C0
对齐；具体搬运及矩阵指令仍保留各自的 tile/对齐限制。

view 保留 allocation 的物理 M，不以子视图的行数或实际前缀替换。对 C0 对齐的
列起点，行偏移系数为 32 B，列偏移系数为 `Mphys*itemsize`。部分行矩形在每个
列块内访问一段连续字节，块间距为 `Mphys*32`；块间空洞不构成读写。
地址生成、访问检查及后端使用同一布局描述，后端不从 byte pitch 恢复 M。

`MatrixRegion` 保存所有轴的实际半开矩形和 SSA 起止坐标；`RegionDefinitions`
在相同存储实例、participant 和调用坐标下组合精确矩形并保留 writer 来源。
只有其他轴相同、一个轴相邻或静态重叠时才合并；行洞、列洞和对角遍历不能
用矩形包围盒补齐。非空固定分块循环可汇总连续遍历，分支仍须覆盖所有路径，
不能借兄弟分支或其他 participant 的写建立定义。

例如两次动态写 `x[0:r0,:]`、`x[r0:r0+r1,:]`，其中 `r1=visible-r0`，可共同定义
`x[0:visible,:]`。visible 等于完整 M 时无需额外清零；visible 小于 M 而后续
MTE1 真正读取完整 M 时，尾行仍须初始化，后面的 softmax mask 不能豁免此前
读取。矩形定义仅说明数值已有来源；旧 Tile 被覆盖、ready、最后 reader 退休
和事件消费仍分别检查，不能从完整定义推导存储可复用。

基于合法源访问合同及已检查的布局/子视图对应，可以直接给出连续的外包切片。
例如环形 NZ 矩阵的部分行访问仍被包含在自己的矩阵平面内。分区器可用这些切片
保留有限槽位的分离；它们只划分 AliasRegion 的顺序域，不扩大真实读写字节，
不提供完整初始化、精确 KILL 或持久 cell credit。公共原点必须在 kernel group 内
不变，所有访问使用同一原点及对齐的有限切片；未知时继续保守处理整个 allocation。
共享 GM 每个切片仍保留两个 AIV 的独立域，Cube 参与双方，不能少一个 quorum 贡献。

`AccessTransfer` 保存 head reader、首次可能写、尾部可能写和未退休 reader。
条件写取静态候选的并集，保留输入 writer/reader；不构造条件摘要的笛卡尔积。
无条件写只更新顺序前沿，意味着先前访问已通过 WAW/WAR 接入，不能被用于
给未覆盖的字节建立数值定义。

重复 body 若没有每代必有的写，又存在变化的条件写，就使用共同 invocation
队列端口传递旧前缀，不能只连接相邻代的可选 writer 而丢掉中间空档。
动态空 child 使用同一实际 loop 调用的 ENTRY/EXIT。入口 Wait 获取相关前序，
出口 Set 捕获该 engine 已发射的真实前缀；空 child 也有平衡协议，不构造 native
LAST，不把退出视作完成。多个 reader engine 和两个 AIV 始终保留独立贡献。

已知非空、控制稳定的有限 loop 可保留窄 native FIRST/LAST。它们只是精度优化，
不是同步生成的全程序门禁。只有局部完整的私有 packet 才可按实际 dispatch
重排物理复用；可能别名、streaming、外部状态和未完整定义的 packet 保持源码顺序。
AliasRegion 和控制端口都不直接使用稳定物理 cell credit；它们继续建立有限
使用或实际 queue handback，包含消费先于复用的关系。

## Runtime 行前缀

外部只读 GM tensor 可显式声明 `layout="NZ"`，仅接受静态 FP16/BF16、最后两轴
16 对齐的矩阵平面。allocation、AddressOp 与 memory effects 保留同一 NZLayout、
leading M、列块间距及真实读取字节；GM→L1 原始 DMA 不隐式重排。
部分静态 tile 可显式补零，目标 fill 后的内部 MTE2 fence 只建立同一 packet 内的顺序。
原始 DMA 使用列块数 `N/16`、32 B burst 长度 `M` 和真实源/目标 leading M 间隙；
不套用 Cube 计算块的 256 维限制。列块数至多 4095，burst/间隙必须落入原指令字段，
完整目标仍受真实 L1 容量约束。

二维 `copy(valid_rows=r)` 保留静态物理容量及实际 `r` 两种事实。AddressOp
携带原始轴、容量和真实行数 SSA；native 同时消费同一行数。地址前缀与指令
操作数不一致、静态非法行数或将前缀地址交给整块 native 时明确拒绝。
实际 `0 <= r <= capacity` 是源操作输入合同，不要求编译器证明动态数据；
0 行没有该侧访问。实际源算术仍须可定义，lowering 不得提前求值或丢掉前缀。

L0C→GM 的 `fix_rows` 在两侧携带同一行数 SSA。源是完整 resident M 与最小包围 N0
列块，每列块实际读 `r*64` 字节，列块间距为 `resident_M*64`；目标实际写 r 行、
每行 N 个输出元素，行步长按元素对齐。0 行不发 FIX，不提供输出其余部分的初始化。
它继承普通 FIX 的 FP32 配置、舍入、最后 reader 和退休合同，不能靠容量上界扩大 effects。

GM→L1 的补零目标是完整写入；GM输入只有实际行数的读取。UB→GM 的目标
只是部分写，不能把其容量上界当成完整初始化或整片 Tile producer。GM前缀
使用保守 AliasRegion 保持可能重叠的源码顺序，不虚构精确覆盖/KILL；私有
UB的保守读取上界仍保持真实最后 reader。有限执行模型按实际行数解释字节，
并覆盖零行、部分行、完整行和重入。

GM↔UB 的 `copy(valid_rows=r)` 支持连续或按32B对齐的跨步矩形，两侧保留同一个
实际行数 SSA；GM→UB 走 MTE2，UB→GM 走 MTE3。连续片段按 `r*row_bytes` 发一条
DMA，跨步片段按实际行数发一条二维 DMA；0行不发DMA。未覆盖的 UB 行与行间孔洞
保持原内容，不能由静态容量推断完整初始化，也不隐式补零。需要读取完整 UB tile时，
先显式初始化；普通 UB 读仍沿现有保守定义覆盖规则检查。新路由复用已有
`copy_rows_prefix`、真实prefix访问和ready/最后reader退休规则，不增加全局barrier。

GM→L1 不指定 `pad_value` 时，`copy(valid_rows=r)` 只覆盖目标的逻辑行前缀，
其余目标保持。当前支持 FP16/BF16/FP32、C0 列对齐、至多 256 行的 L1 NZ view。
ND2NZ 的 D 字段为 uint16，当前列数上界为 65520；补零/行前缀与完整搬运
使用同一列宽限制，目标仍受真实 L1 容量约束，不能把计算块维度当作搬运面板上界。
GM 读取是 ND 的 `r` 行；NZ 写入是每个列块内的 `r*32` 字节前缀，保留 resident
行距和真实行偏移；GM 源步长使用对应 dtype 的元素单位。两侧 count 必须对应
同一实际 SSA，0 行不发 DMA。单次部分写只定义真实矩形，多次部分写按上述区域
合同组合；不能按静态容量建立完整定义、精确 KILL 或持久 credit。
需要完整定义时可以拼满真实读取区域，也可以显式初始化后覆盖。指定
`pad_value=0` 的静态或 runtime 补零搬运支持 FP16/BF16/FP32，目标 M/N 均按 16
对齐；先填满目标零值，再按实际行列从 GM 搬入，源步长使用对应 dtype 的元素单位。

metadata 的范围属于真实 load 及其 SID、祖先迭代坐标。事件分析逐 guard 保存
求值前已经支配它的 load/range 前缀，只用这份前缀消去恒真的 metadata 条件，
保留原始有序 guard。支配端点不表示支配此前的 guard：分支或 loop body 内的
load 不能反过来证明该分支会进入或该 loop 非空；未执行的 load 不限制外部输入。
重复出现的 guard 保留最早求值点的前提。不把局部范围注册为全局 constexpr。
事件 handback 已假设旧消费和新发布两个实际端点执行时，分别消费它们的支配
load 范围，按真实 SID 和各层实际坐标绑定一次。消去恒真 activation 不丢弃这份
局部输入合同；不同代次仍可读取不同值。自己的 load 之前没有自己的范围事实。
这些前提不用于证明先前 guard 的可求值性，不借用尚未执行的可选 relay，也不
建立异步完成关系或新的事件共享政策。
同步 Scalar 的 EXIT 可保留实际已退休的 load 身份；没有入口旧值读取的同步
观察无需导出额外入口 reader。异步 reader、真正 head read 和 Scalar store
继续保留各自的前缀与完成义务。

未知区域关系可能减少可用并行性。固定排步不兼容、事件容量不足或控制协议
不能组合时具体拒绝，不加全局 barrier 或改排步。普通 loop 出口没有统一 drain；
kernel 最终退休仍独立保留。

## 每次只读 GM copy 的缓存政策

`copy(..., cache_bypass=p)` 的非缺省选择只接纳 external、read-only GM source。
`p` 是 bool 常量或当前实际求值域内定义的 i1 SSA；source provenance、lowering、
AddressOp 与 ScalarUses 保留同一个选择及其真实依赖，不以分析 Expr 替代执行值。
缺省 `False` 不增加地址选择，继承原 Tensor 的缓存政策。该机制不适用于目的
地址、可写 GM 或私有存储，不通过缓存政策授予写入或别名分离许可。

后端对需要选择的每个 canonical GM root 只准备一份隐藏 device bypass pointer，
由 host launcher 查询 `rtGetL2CacheOffset` 并加到原地址；错误直接返回。已经全局
bypass 的 root 不重复准备别名。公开 host/Torch ABI 不增加参数；device ABI 的
`device_cache_aliases` 记录这些 root。offset、原 allocation 身份、ND/NZ footprint、
effects、初始化、版本、ownership、最后 reader 与退休义务均保持。
两种地址访问同一真实内存，不能据此删除 RAW/WAR/WAW 或增加同步。

独立有限模型检查混合普通/选择读取、空/单次/多轮/重入和双 AIV 的私有选择，
ND/NZ 的实际 native 与事件同普通缓存版本一致。设备缓存政策的性能仍须对具体
产物、输入和原始 CSV 验收；缓存别名支持不构成性能保证。

## 动态标量分支和 L1 初始化

标量 `if/else` 的 Index 结果可以用于地址、长度和 mask。source provenance 检查
条件及两条 yield 的实际定义和支配；每条分支在自己的作用域内计算结果范围，
合流仅导出保守并集，且范围不能引用未逃逸的分支内部 metadata 身份。没有可证明
范围的 loop carry 仍未知。条件求值前的 frame 与结果产生后的 result frame 分开，
条件检查不能借用自己的结果或未来 load；静态非法原式仍独立诊断，动态定义性归输入合同。
这些范围只提供整数事实，不导出 Tile 初始化、ownership 或异步完成。

`N.fill(l1_plane,0)` 支持物理连续的 L1 NZ 浮点平面，降低为 MTE2 的 `zero_l1`。
它精确写入声明的整片字节，使用正常同步和 reader 退休规则；不支持非零值、
带物理空洞的子矩阵或 mask，不把保守外包范围当完整写入。可以在 kernel 入口
初始化驻留池，再用 `copy(valid_rows=...)` 覆盖真实片段；未覆盖行保持已定义值，
算法必须独立保证这些行的数值处理符合其 mask 合同。
