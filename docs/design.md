# 设计：结构化程序上的统一自动同步

2026-10-15 修订；实现保存点 `3f95139`。本文是下一部分重构的技术合同，
**不是已经实现的统一算法**。实际能力集中记录在[状态页](refactor-progress.md)。

## 目标与纠正

以 TileLang 风格的受限 Python AST 表达计算、存储、局部流水和 CV 多阶段流水。
专家排步与未来自动排步进入同一个物理访问和完成关系模型。自动同步优先，
性能通过最终 ASC、CANN 产物及设备 profiling 验证。

现有源码入口已经统一，region/call 共用增量关系查询与同步选择；叶流水的规划
和层次控制适配尚未统一。
继续按程序形状增加证明器，会把组合问题变成不断增加特例。下一次重构要替换
重复的语义和查询职责，而不是增加第四个形状分支或只把文件搬进同一个目录。

同时保留已核实的 native 契约、原语 lowering、ASC emitter 和反例。xDSL 是程序
的唯一可改写表示，分析索引引用它；不建立第二套可变程序树，不引入 MLIR 工程。

## 语言与执行边界

详细语法见 [DSL](dsl-multistage.md)。runtime shape/stride、循环界和 descriptor
是设备 SSA；容量、layout、dtype 和固定 stage 政策是 constexpr。常量循环界也
默认保留，只有显式且有预算的 unroll 才展开。输入长度不参与编译期任务枚举。

`Cube/Vector` 指定 actor，原语决定 Scalar、Vector、MTE1/2/3、Cube、Fixpipe。
Stage 表示逻辑任务的派发偏移，不等于执行引擎、buffer 数、event 数或完成屏障。
嵌套 child 有自己的迭代坐标；离开 child 控制流不能证明其异步访问已经退休。

整个路径为：

```text
Python AST → xDSL 结构化源码 → 保留控制的物理执行 IR
 → 访问/代次/完成约束 → 同步协议和有限资源分配
 → 物化同步 → 从最终 IR 重提取检查 → ASC → CANN
```

## 公共分析模型

下面是必要信息，不预先承诺类名或再创建一组空接口。

| 对象 | 必须保留的含义 |
|---|---|
| 操作模板 | 原生工作、actor/engine、源码位置、所属 region、派发位置 |
| 迭代实例域 | 符号迭代坐标、uniform 参数、activation/guard；不枚举所有实例 |
| 物理访问 | allocation 身份、view/layout 后字节集合、R/W/RW、开始/退休端点 |
| 代次与归属 | owner、slot 映射、generation；fresh frame 的已验证不相交关系 |
| 顺序约束 | 源/目标里程碑、两端实例关系、条件、覆盖范围和硬件强度 |
| 区域边界 | 所需初值、末 writer、尚未退休 reader、可见完成关系、存活通知 |

逻辑值依赖和物理访问分别保留。tensor 名不同不证明不别名；SSA 使用不是异步
完成；借用 view 不是 snapshot。DMA padding、scratch、Scalar 读取及 SPR 状态
也要进入 effects。未知 native effects 直接拒绝，通用 DCE/CSE/移动不能忽略它们。

ready、read-retired、release/credit、notification-consumed、最终 drain 具有
不同语义。每个访问只使用 dav-2201 真实可观测的完成点；没有 native 保证时
不能虚构比操作完成更早的 reader retirement。

## 算法修正：从逐对可达查询改为完成信息的数据流

2026-10-15 再审：`3debb17` 合并了 region/call 的生成入口，但没有解决核心
工作量。底层 `completion_graph` 已经按真实指令构造稀疏关系；上层却枚举
milestone 对、调用叶路径查询以发现关系，再对发现的关系做另一次搜索。
`CompletionPaths` 缓存这种搜索没有消除第二层图，且扩大了一些查询的作用域。
这条按端点询问“能否到达”的生产路径应被替换，不再以补缓存作为主线。

新的主线为：

```text
真实访问前沿 → 必须满足的完成义务
真实指令/排步 → 稀疏执行关系 → 批量传播完成信息
                                ↓
                       在消费者处比较义务与已有事实
                                ↓
                       只为缺口建立实际同步协议
```

下面是统一算法合同。当前已接通 serial region/call 的传播及局部叶最终验证；
叶生成器试删、旧 invocation 和跨核路径仍待替换。没有改变 DSL 的
`Pipelined/Stage/Cube/Vector`，也没有引入新的用户同步标注。

### 两种前沿，职责分开

已有物理访问分析保留：按 allocation root、字节片段、owner 和 generation
记录最近 writer 及尚需保护的 readers。读要求对应 writer 完成，写要求旧 writer
及全部冲突 readers 退休。`region_summary` / `region_sequence` 已有这个基础，
不另造一套“统一访问框架”。多个 reader 不因同属一个 pipe 就删成最后一个。

新增的执行数据流回答：**到达某个真实指令里程碑时，哪些工作保证已经完成？**
它是编译期的条件事实，不是模拟设备当前执行到了哪里，也不是预测完成时间。

最低限度保留三类不同事实：

- 派发/开始顺序：按每 actor 的实际 issue 和每 pipe 的 FIFO 契约传播。
- 原生工作退休：一个具体访问完成，或该 pipe 在某位置之前的原生工作全部完成。
- 通知消费：某 event 的这一代 Wait 已消费；用于证明下次 Set 的资源寿命。

“退休前缀”是所有先前原生工作的完成集合，不是“最后派发的任务已完成”。
只有实际 Set/Wait、pipe barrier 或硬件明确提供的隐式完成关系才能使它成为
后续任务开始时的已知事实。不存在默认的 region-done 或 per-call drain。

### 按真实契约传播，不通过两两查询发现边

先从每条真实指令建立局部转移，顺序关系只连接相邻位置及必要边界。条件跳过
使用透明的前沿传递，不枚举所有跨 guard 的绕过边。基本规则为：

| 位置/操作 | 可以传播的事实 |
|---|---|
| 原生操作开始 | 其真实派发与接收流水入口已保证的完成事实 |
| 原生操作结束 | 开始时的事实，加该操作自己的实际完成 |
| 退休前缀 | 前一前缀与本次原生完成的集合；它本身不阻挡后续派发 |
| Set 的完成 | 对应硬件保证的源流水退休前缀、实际前驱事实及通知发布 |
| 匹配 Wait 的消费 | 匹配代次和参与者的发布事实，以及此次消费事实 |
| pipe barrier | 该 barrier 契约覆盖的前序原生完成，传至后继开始位置 |
| Scalar 阻塞等待 | 按已核实契约影响其后的派发；不追溯阻塞已经派发的工作 |

源码访问依赖是**需要建立的关系**，不提前放入“已有完成事实”。否则会出现
用尚未生成的同步证明可以省去它自己的循环论证。普通任务发射也不使其 end
事实自动进入同 pipe 后继任务的 begin。

在无环片段上一次按拓扑序传播；只传递消费者及 event 寿命实际需要的事实。
不同 actor 使用偏序，不按 Python 源码排列成一个总执行序列。新增已确定协议
只更新受影响后继，不调用另一个 `prove(source, target)` 搜索器来发现边。

一条消费者的全部 RAW/WAR/WAW 义务一起处理。同一源流水、同一时钟中已经
证明有包含关系的退休前缀可以合并；不同 reader/时钟/参与者保留独立事实。
同一消费者的缺口确定后，才选择合法的源捕获位置和接收等待位置。沿真实路径
已经传过来的完成事实直接满足义务，不再先插满通知再逐一试删。

事实可用小整数 ID、位集合或经过支配证明的前缀表示；表示形式服从语义。
不能把所有事实压成一个每引擎最大序号。选择顺序仍可能影响同步数量；不承诺
全局最优，但交付必须对照既有 ASC，不能接受无解释的额外等待。

### 循环、条件与层次的组合

普通循环保留符号坐标。双缓冲的关系直接表示为：

```text
MTE2.write_done(i) → Vector.read_begin(i)
Vector.read_retired(i) → MTE2.write_begin(i+2)
```

循环体交付“入口前沿到各个可见里程碑的转移”，而非一个退出完成 token。
规则环的基例处理尚无旧代次的槽；归纳步骤验证 `i → i+1` 后同一个转移仍成立；
出口用真实最后访问位置，例如 slot0 的 `N−2` / `N−1`。零次循环是入口状态的
恒等传递，first/last 和尾部条件单独保留。不展开 N 次，也不把常量界变成展开许可。

不能直接运行无界 worklist 求循环闭包：代次会不断增加。首个实现限定已有
常量 stage/slot、规则直接访问子集，显式求出上述转移不变量，再检查基例、
归纳步骤和出口。静态环宽与真正耦合的条件决定分析规模；独立环不取全局 LCM。
当前 serial adapter 用静态首尾片段保留被查询的真实里程碑，中间只携带
每条转移都保持的 issue/queue/native-retirement/implicit 前缀。此包含不变量
按步归纳，event 的 Set effect 不跨省略部分传递。未包含的中间跨流水路径
不能提供覆盖证据；这是受限摘要，不是一般层次循环的完整求解。有限模拟
仍只是差分检查，不能替代这个归纳依据。

分支保留事实的 guard。无条件汇合处，只能把每条可行路径都保证的事实提升为
无条件事实；需要两个贡献者的真实同步汇合则必须收齐两者。不能混淆这两种合并。
摘要引用 child 的真实首尾和中间捕获点，父层无需逐端点重新询问 child 内部路径。

CV 继续使用同一完成信息传播规则，但有独立的 AIC/AIV issue 流、各自的原生
流水和 task/query/child 坐标。Score-ready、P 两个 AIV 的完整贡献、PV-ready、
各 GM ring 的最后 import reader 分别表达。控制退出不串行化下一 stage；
新 query 的 QK 可以早于旧 query 的 PV。跨时钟映射必须有独立合法性依据，
不能复用当前串行 region 的词法顺序假设。

### 七模板 frame 示例的直接推导

snapshot 的 MTE3 reader 必须在首次 Vector 覆写 state 前完成，因此该真实
Wait 把 snapshot 完成事实带入 Vector 流。随后 slot0 的 Vector release 又把
该事实带入 reload 所在的 MTE2 流。N>0 时，reload 的 snapshot-ready 已经满足；
N=0 时这条路径不存在，必须单独等待原 MTE3 snapshot。这里是沿现有协议传播
事实并比较缺口，不需要查询所有 region 端点之间是否可达。

奇数 N 的 slot0 release 捕获最后一次 Vector 更新；偶数 N 只捕获 N−2，
所以后续最终 state 加法仍需覆盖 N−1 的更新。后一依赖不能因为输入 slot0
已经可复用就消失。这是同一完成信息传播必须直接给出的两种条件结果。

### 资源与最终验证

local binary event 使用 `Set(e,k) → Wait(e,k) → Set(e,k+1)` 的具体资源义务，
再按真实生命期分配有限 ID；cross-core 的参与者、计数和可见性使用其自己的
硬件转移。通知消费不能作为 buffer read-retired 的代替。同步正确性不能靠
任意全局 barrier、额外搬运或改变 buffer 数量得到。

最终 verifier 从生成 IR 重新提取真实转移，执行同样的批量数据流，再检查
内存义务、通知资源和进展；不信任 planner 提供的“已证明”状态。需要独立的是
事实来源，不是另建一套昂贵的逐对搜索。保留独立小规模字节/event 模型作差分。

### 用现有反例否定过强简化

| 反例 | 算法必须保留的区别 |
|---|---|
| 两个 Vector reader 后覆盖同一 UB | 后一个开始/结束不自动证明前一个退休；真实前缀捕获才覆盖两者 |
| Set 已发出，随后同 pipe 又有原生工作 | Set 只覆盖自己的前序；其 effect 不能当作后续工作的 fence |
| child=0 的 snapshot 回读 | 恒等转移不能生成不存在的最后 reader；保留真实 snapshot-ready |
| child 为偶数时 slot0 复用 | 等 N−2 的实际 reader，不扩大成等最后 slot1 的 N−1 |
| 一条路径仅偶数激活、下一条仅奇数激活 | 两个条件不能组成一条全域完成路径 |
| 两个 AIV 共同生产 P | 一方发布不能覆盖另一方的字节和贡献 |
| 新 query QK 早于旧 query PV | query 编号不授予整个 query 的串行完成关系 |

## 符号关系的可实现范围

首轮限定结构化循环、uniform 非负长度、静态 stage/slot、规则直接 views 和
可验证的 owner 分区。两端关系组合必须匹配中间实例、guard 和完成强度；
只看最小距离或忽略条件的图可达不能替代指定实例关系的覆盖。

`task=query*K+kv` 在 K 为 runtime 时包含参数乘积；`task//K` 也不是普通的
常量除数仿射关系。不能声称交给通用仿射分析就解决。需要为这类规则时钟映射
保留因子化关系及 `K>0, 0<=kv<K` 合同，利用商余唯一性做受限组合；不展开 K
个余数，也不支持任意非线性整数求解。独立测试乘法溢出、零 K 和 query 边界。

多参数 GM 地址同样保留 shape/stride 结构，先证明直接矩形 view 的范围和分区。
普通外部指针 disjoint 合同在 launch 检查实际区间；间接索引、数据相关 ownership
需要更强可验证合同，否则诊断为不支持或分析未决。

统一模型不要求硬件协议相同：local HardEvent 是每核每有向 bank 八 ID 的二值
事件；cross-core 的参与者、计数、quorum、反向 ack 与 GM 可见性另有 dav-2201
契约。消费通知不等于消费方最后读取完成；两个 AIV 的 UB 始终私有。

## 编译复杂度与退出门槛

令 A 为实际访问片段数，H 为真正的存储冲突关系数，V/E 为真实执行关系图
的节点/边数，F 为需要传播的完成事实大小。分析不以端点两两组合来发现 E。
无环、固定条件片段内的数据流目标成本为 `O((V+E)·F + H)`；位集合可以按机器
字压缩 F。这是局部片段的成本，不是包含任意分支、符号循环和资源分配的总界。

多个同步的增量传播还取决于事实更新次数，循环不变量求解、条件碎片及真实
密集别名都有额外成本。不能把“worklist 单调”说成总体线性，也不能靠提高预算
掩盖状态增长。运行时长度 N 不得成为模板数、事实数或循环处理次数的因子。

现有 frame 只有七个原生模板；即使需要一百多个语义端点，这仍是小问题。
保存点 `3debb17` 的 72 次覆盖查询触发 13525 次实际关系求值，这是多余工作。
当前 serial frame 已移除该嵌套查询；局部最终验证也按条件域批量传播，
不再逐义务搜路径。剩余旧生成与跨核路径继续按同一方向替换。

验收同时记录真实边数、完成事实/条件片段数、传递次数、旧路径查询次数和
五冷进程耗时。首个替换的 frame 生产路径不得再调用逐端点叶路径搜索。
保留循环的常量界 1024 与 1048576 不应导致结构计数按次数增长。

常用例源码→ASC 仍要求 p50≤1s/max≤2s，并比较阶段计时；达到 1s 仅是已有
门槛，不等于七模板程序的开销已经合理。完整 FA 目标 p50≤5s/max≤10s。
import、CANN、launch 单列；最终验证保留。ASC、buffer、搬运及等待范围必须
同时验收，不能以降低检查强度或串行化换取编译速度。

## 正确性与性能验收

依次检查 IR/输入合同、访问与生命周期、有限交错反例、CANN 离线编译和设备。
最终 verifier 可共用数学工具，但不能宣称与 planner 完全独立；有限模型检查
不等于任意规模形式化证明。失败分为已证非法、不支持、预算未决。

固定政策下先逐项对比 ASC 的 native 指令、累计顺序、搬运、buffer、wait/release、
CV gate 和填排，再对照独立 AscendC 及 timeline。与同政策参考接近，只能支持
编译器保真结论；还需对照 ops-transformer/CATLASS、安装 ACLNN 和硬件工作量。
不得通过新增 packing、全局 barrier、每 query drain 或扩大缓冲取得表面正确。

硬件分析使用真实 SKU、核数、频率依据和 dtype，条件下界为
`max(各共享资源总需求/对应吞吐上界, 依赖与派发关键路径下界)`。
有效/native FLOPs、GM 请求/HBM 实际流量分开；未知持续吞吐保持参数化。
大规模配置、测量口径与 FA 变体次序见[重构计划](refactor-plan.md)。算子调参仍暂停。

## 源码依据

本轮只读核对 HEAD，未 pull：TileLang `6ba65d2`，ops-transformer `4bbfb3b`，
CATLASS `698379e`，ops-nn `64b41ad`。固定提交报告与当前 checkout 分开溯源。
依据为 [910B 契约](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[算子审查](../../docs/xdsl-npu-operator-source-audit.md)及[FA 目标合同](attention-target.md)。
现有 FA 候选政策的 AIV update-before-softmax 与 ops-transformer arch22 可见的
Vec1-before-Vec2 不同，不能称为逐条复刻专家实现。
