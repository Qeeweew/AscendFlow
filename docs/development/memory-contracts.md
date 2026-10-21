# 源程序内存合同与自动同步边界

普通 Kernel 的 group 是独立的 SPMD 执行域。程序负责不产生跨 group 的非原子
读写竞争；编译器保持映射和实际地址，不重新求解任意映射的全局唯一性。
两个 AIV 共用 Vector body，各有私有 UB；普通 peer 访问不能依赖未声明的
跨 AIV 通信。AIC/AIV 的共享 GM 交接继续由实际 effects、双 AIV quorum、
ready 与 reader retirement 生成同步。

这些是源程序前提，不是后端可填写的 verified 标记。当前没有通用跨 group
phase/collective DSL；取消全局分离门禁不表示增加这种执行能力。

## Tensor 声明

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
也不约束下一次 launch。算法和调用方满足声明，launcher 不添加运行时检查。
独立有限模型按 invocation 检查具体重复写，能发现错误声明。

当前只对没有内部 reader 的 write_disjoint 输出消除写间 WAW，无需恢复排列的
代数逆。同一 tensor 若还被读取，保留保守源码顺序，包括 RAW/WAR。输出声明
不传播到临时 UB/L1/L0、共享 workspace 或其他 pointer，物理复用照常分析。

## 访问顺序与数值定义

只读 INT32 / INT64 元数据由同一个 `index_load` 原语读取一个实际元素，effects
分别为 4 / 8 字节，返回有符号扩展后的 index SSA。其 inclusive 最小/最大值是
调用方输入合同，不生成 clamp，也不授予写权限或不同索引的唯一性。
原轴访问检查保留支配读取的范围关系及已求值分支条件，不能借用随后条件证明
更早的 unsigned 运算合法。条件访问的范围通过不代替初始化合流、存储分区或
通知生命周期准入。

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

这里建立的是数值初始化，不是异步 ready、精确 ownership KILL 或事件消费。
masked/RW 更新先需要旧值有定义；runtime 部分写不以容量上界提供完整定义。
完整数值定义前沿不收录 runtime 部分写，避免它按相同静态几何去重掉后续完整写。
这不删除真实写入的 effects、可能覆盖关系或同步义务。
同一只读源可以有多个 reader，实际覆盖前仍须保留所有 reader 的退休义务。

原轴、布局、子视图和 allocation bounds 通过后，bounds 可以给出连续的外包切片。
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

二维 `copy(valid_rows=r)` 保留静态物理容量及实际 `r` 两种事实。AddressOp
携带原始轴、容量和真实行数 SSA；native 同时消费同一行数。地址前缀与指令
操作数不一致、无上界/负行数或将前缀地址交给整块 native 时明确拒绝。
原轴与 allocation bounds 检查实际非空前缀，0行没有该侧访问；这不豁免原始
标量算术的定义性、溢出或求值域检查，也不允许静态 subview 丢掉前缀。

GM→L1 的补零目标是完整写入；GM输入只有实际行数的读取。UB→GM 的目标
只是部分写，不能把其容量上界当成完整初始化或整片 Tile producer。GM前缀
使用保守 AliasRegion 保持可能重叠的源码顺序，不虚构精确覆盖/KILL；私有
UB的保守读取上界仍保持真实最后 reader。有限执行模型按实际行数解释字节，
并覆盖零行、部分行、完整行和重入。

GM→L1 不指定 `pad_value` 时，`copy(valid_rows=r)` 只覆盖目标的逻辑行前缀，
其余目标保持。当前支持 b16、16 列对齐、至多 256 行/列的 L1 NZ view。
GM 读取是 ND 的 `r` 行；NZ 写入是每个列块内的 `r*32` 字节前缀，保留 resident
行距和真实行偏移。两侧 count 必须对应同一实际 SSA，0 行不发 DMA。部分写
不能初始化完整矩阵，也不能按静态容量建立精确 KILL 或持久 credit。
需要完整定义的调用先补零或拷入完整矩阵，再覆盖已定义存储。

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
