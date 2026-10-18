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

## 两种依赖事实

structured_access 保留真实 native footprint。静态 slot、矩形与轴 grid 只是
可选精度，不能建立分区时保留可能别名的源码内存顺序。

- StorageCell 是精确字节区域。logical_valueflow.reads 绑定数值定义；
  LogicalWrite.KILL/UPDATE 仍依赖真实覆盖与读写语义。
- AliasRegion 是同一存储实例的可能别名访问域，没有伪造的字节包络。
  ordered_reads 表示源码顺序前驱，不声称该 writer 就是数值生产者。
  WAR/WAW 由同一递归前沿和 storage_lifetimes 连接。

部分写不能因此成为完整初始化。并行 Vector peers 各自保留源访问域，AIC
的可能共享访问进入每个域，不能以一个 AIV 替代 quorum。AliasRegion 不进入
private packet 重排或稳定物理 cell 的 credit 归纳；其事件消费复用仍由有限
使用或实际 queue handback 建立。

未知区域关系可能减少可用并行性。固定排步不兼容、事件容量不足或控制协议
不能组合时仍具体拒绝，不自动加全局 barrier 或改排步。bounds、原始算术
定义性、初始化、native alias 限制与借用 Tile 生命周期继续独立检查。

## 当前范围

本批移除了 structured_ownership 全局 group/SID 分离准入及 OwnerProof。
admission 记录实际 StorageInstance，保留 pointer alias、初始化与借用 Tile
合同。两种区域均进入原有通知生成、事件分配、物化与 ASC，没有第二编译路径。

任意父代次条件激活、动态空 child 的 family closure 与全局通信尚未重写。
现有 FIRST/LAST、原式定义性、queue_order、cell credit 和 handback 保留。
工作量分别报告 cells、alias_regions、disjoint_writes；保守顺序不代表没有性能代价。

循环实现只使用一份递推组合规则：普通循环和 FIRST/LAST 之间的中间区间共用
`logical_valueflow._repeat`；边界条件按本次 memory snapshot 分类并复用。
没有新增激活区间、周期切分或循环前沿求解器。这是既有实现去重，不是通用
动态控制模型重构完成；上述未支持边界保持具体诊断。
