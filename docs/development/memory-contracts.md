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

`structured_access` 是实际 native footprint 的唯一输入。StorageCell 表示可选的
精确字节区域；AliasRegion 表示无法排除重叠的同一存储实例。两者都由
`access_order` 建立可能冲突的 RAW/WAR/WAW；RAW 在这里是顺序义务，不宣称
某个 writer 是唯一数值 producer。初始化、源 Tile 借用与原式定义性由 admission
独立负责，部分写不能代替完整初始化。

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

未知区域关系可能减少可用并行性。固定排步不兼容、事件容量不足或控制协议
不能组合时具体拒绝，不加全局 barrier 或改排步。普通 loop 出口没有统一 drain；
kernel 最终退休仍独立保留。
