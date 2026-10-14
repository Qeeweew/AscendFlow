# xDSL / Ascend 910B 编译器设计

2026-10-14，重新审查基线 `b98d541`，开始时产品工作树干净。本文取代此前设计，
不改变已提交代码的能力声明。语言合同见 [DSL 规格](dsl-multistage.md)，迁移次序见
[重构方案](refactor-plan.md)，已有实验见 [验证记录](refactor-progress.md)。

## 1. 产品目标与本次修正

目标是一种接近 TileLang 的、可以明确控制 NPU 执行组织的 Python DSL。
用户写计算、搬运、布局、存储和流水政策；编译器推导实际访问之间的同步。
**语言可表达、编译成本可接受、机器执行保真，是同一个功能的三个验收条件。**
自动同步先行，自动排步后做；专家固定排步仍走相同验证。

上一版有可保留的基础，也有需要纠正的取舍：

| 原方案的问题 | 本版决定 |
|---|---|
| 用一个父 serial × 单 child 特例代表区域组合 | 将顺序组合、循环递推、分支合流的合同写清，再推广已验证的局部证明 |
| 要求所有 GM 按 invocation 分区 | 分开只读共享、独占输出、交接 scratch；以真实 effects 证明，不能只看形状 |
| 把跨 query 连续流水与动态 persistent 一起延期 | 规则的 query×KV 连续流进入本轮验收；动态任务队列仍单独延期 |
| 用更多单层例子代替最终语言验证 | 先固定真实 FA 源程序和预期派发，再用最小非 FA 反例补齐必要语义 |
| “不按 N 展开”被当成编译性能问题已解决 | 同时约束 proof states、周期耦合、查询/重建次数、冷编译和内存 |
| 泛化、宏、缓存、性能基线都留到后期 | 按依赖分步接入；固定 AscendC/ASC 验收从目标源码开始，而非最后才补 |

继续保留受限 AST、xDSL 核心基础设施、显式物理 effects、完成前缀、最终 IR
重检及已有反例。不会重建环境、迁移 TileLang pass 集合，或先建设通用调度框架。

## 2. 基线事实与尚未成立的假设

| `b98d541` 的事实 | 不能据此声称 |
|---|---|
| AST 单流水 Vector、对齐 prefix、归约/Scalar、显式 L0、runtime K 已接通 | 任意 Python、一般分支、任意布局/尾部已支持 |
| local serial 父 × 单 child，GM frame 和跨调用 credits 已接通 | 多 child、父前后 effects、嵌套 CV 已支持 |
| 父子例 6 个 native 模板，runtime outputs/panels 不展开 | 任意静态结构的证明都足够快 |
| 父子冷编译 p50 0.6889 s，960 MiB 例与 AscendC 延迟比 1.00028 | 真实 FA 的编译/竞争性能已达标 |
| legacy FA 在 examples，有 4K/8K 设备记录 | 新 AST 路径已经实现 FA，或旧性能数据可直接迁移 |

代码约束可直接核对：`lowering.lower_source` 限制父 body 只有一个 pipeline；
`invocation.extract_invocation` 要求每个 GM root 都有 frame；`Loop.protocol_period`
仍合并同步条件周期；`Compilation` 已有分阶段计时、最终检查和不可变产物，
`budget` 已有协作取消。不能再把这些已实现项列成从零开发，也不能忽略其限制。

本次查看的最终父子 ASC 保留两个设备循环，跨调用 FIX→MAD，L1 credit 来自
MTE1，只有 kernel 末尾一次 PIPE_ALL；短 MAD 的 PIPE_M 有独立机器依据。
这份代码没有跨 query CV 连续派发，不能把“没有循环内 PIPE_ALL”当成重叠充分的证明。
代码、计时和 hash 对应 [层次流水证据](runtime-hierarchy.md)。

## 3. 语言设计：保留熟悉语法，消除语义歧义

公开入口为 `@T.prim_func`、`T.Tensor`、`T.Kernel`、`T.serial/Pipelined`、
`T.Stage`、`T.Cube/Vector` 和 typed 原语。`T` 是现有 language 的导入别名。
不再使用 `cube.task/vector.task`；不为同一种执行模型引入第二套任务 builder DSL。

- `n/Sq/Sk` 默认是运行时标量；tile、dtype、layout、slots、stage policy 是显式
  constexpr。编译器不执行设备函数体，不按输入长度创建 Python task 对象。
- 常数边界也保留循环；只有显式 `unroll` 在预算内展开。运行时参数不能因为
  调用时已知就偷偷进入特化键。动态 shape、索引和边界是 SSA，不是表达式字符串。
- `Stage(s)` 是所属 pipeline 的代次位移；`Cube/Vector` 是执行者；MTE2/MTE1/
  MAD/FIX/V/S 是内部引擎。stage 数、buffer 槽数、event 宽度没有自动相等关系。
- 逻辑语句顺序规定算法、读写版本和数值递推；每 actor 的 `issue_order` 只规定
  实际派发顺序。同步不能为了可行性悄悄改变用户固定的顺序、tile 或容量。
- buffer/view 保留分配身份；typed tile result 是某次写的借用，不是隐式副本。
  必须保存旧值时显式存储。局部状态初始化必须是真实写，不能伪造 initialized。
- 常用 GEMM/softmax 等表达由 examples 中受限 AST 宏组成。宏只展开 typed 原语，
  接受显式 scratch，保留源码映射和数值边界；核心不识别算子名。

初期接受物理存储较显式这一代价，以保证性能可控。先做好明确的原语及可审查
的例子宏，再考虑自动布局/自动分配；不把一个 opaque matmul/softmax 当成简洁性的替代。

## 4. 两级流水及两种任务组织

CV 流水的逻辑任务可以有 QK、softmax、PV、update 四个阶段；每个阶段内部又有
搬运/计算的小流水。通信块、Cube 原生块和 Vector 行块分别指定。两个 AIV 的
UB 各自私有。910B 的 CV 数据交接明确经过 GM，不能套用 950 的直达通路。

### 嵌套控制

父执行者到达 child 时派发其控制循环，退出后继续父工作。child 控制退出没有
“所有异步指令完成”的含义；出口携带实际 reader/writer/通知的存活关系。
普通嵌套不自动将父的兄弟 stage 插入 child，也不自动跨 query 消除填排开销。

### 规则连续任务流

专家实现可能让下个 query 的 QK 与上个 query 的末段 PV/update 重叠。应允许在
设备上保留一个规则的 task 循环，并从 task 解码 query/KV；不在 Python 展开乘积。
这是已知规则域的派发政策，不要求动态工作窃取或软件任务队列。

初始支持限定 `tasks=queries*K`、uniform `K>0`、`q=t//K`、`k=t%K`，按同一个
被捕获的 K 解码。分析保留 `t=q*K+k, 0<=k<K` 的分段关系；只对静态 slot 周期
需要的 `K mod B` 分类，不建立 K 个证明余数。短 K 另按静态stage/依赖距离的
边界分类，不能仅靠余数抹掉启动/收尾差别。K=0 必须在除法外处理空域。
无法证明此模式时明确拒绝，不能改为逐 query drain 或将 K 强制 constexpr。
该能力尚未实现，必须以独立切片验收后用于 FA。

两种组织共用 effects、代次、ownership 和事件契约，但派发图不同。
自动调度未来可以选择组织；本轮仅实现用户明确指定的组织。

## 5. IR 与区域组合的具体合同

```text
受限 Python AST
 → xDSL typed source（arith/scf、refs/views、stage/actor、数值参数）
 → xDSL physical regions（保留控制、显式 native effects/存储/派发）
 → 推导义务、规划同步、物化实际 Set/Wait/Notify
 → 从最终 IR 重检 → immutable Compilation → ASC → CANN
```

以 xDSL IR 为程序真相，不新增与其平行的可变程序树。分析索引、关系和摘要可以
用不可变 Python 数据结构；改变 layout、guard、owner、schedule 即失效。

每条物理操作必须有 allocation root、精确字节 footprint、R/W/RW、执行者/引擎、
激活域、代次和源码来源。补零、转换、scratch、Scalar metadata、SPR 设置都算 effects。
分别记录派发、写完成、读退休、通知消费和最终 drain；SSA use-def 不代替完成。
早释放只能取实际最后 reader 的可信完成点，不能猜指令内部的提前读取时刻。

区域摘要的输入为入口可见版本/仍存活读者、参数/owner 域和资源状态；输出为：

| 字段 | 必须保留的内容 |
|---|---|
| 入口义务 | 实际读取依赖哪些已写字节；首次覆盖需哪些旧读者退休 |
| 出口访问 | 每个精确 byte piece 的 writer 与未被覆盖的 reader families |
| 完成关系 | 哪个可信里程碑覆盖哪些访问，不能缩成一个 done |
| 激活与代次 | parent invocation、local generation、guard、first/last、零域 |
| 资源余量 | 未消费通知、待归还 credit、bank/ID、生效与消费次序 |

组合规则先实现可证明的有限子集：

1. **顺序 A;B**：连接 B 的入口与 A 的出口，保留 B 不触及的存活访问。只有已证明
   覆盖历史读者的完成前缀才能替换 reader family；不能仅取“最后派发的 reader”。
2. **循环**：分别证明入口、单次递推、出口与 N=0 恒等分支。迭代次数是参数；
   最后一次写不能初始化第一次读。边界待解决义务传给父层，到根仍未解决则拒绝。
3. **分支**：按 guard 合流有效版本和通知；读必须在所有可达路径初始化。
   early return、break、数据相关循环先拒绝，不通过任意 barrier 合流。

先支持有限层级、规则索引、受限边界谓词；不先引入通用 Presburger 求解器。
现有 `region_summary/invocation_sync` 是可复用的特定证明，不是任意区域组合定理。
推广时必须能通过现有父子例，再覆盖 prelude→child→epilogue、只读共享和两个 sibling。

只读共享输入允许跨 invocation/owner 重读，前提是整个 kernel 的真实 effects 中
没有与它重叠的写；外部并发修改不在调用契约内。输出需要独占写域，交接 scratch
需要明确生产者/消费者及 release。`access="read"` 声明和 `initialized=True` 都不能
单独证明只读性；未知 alias/ownership 需要可验证契约或具体诊断。

## 6. 同步与硬件资源

从实际 RAW/WAR/WAW、初始化、slot 复用、SPR 和通知寿命生成义务，再用可信
隐式完成路径及有界协议候选覆盖。资源分配参与可行性检查，物化后重新提取。
planner 搜索失败与程序确定不合法必须区分；不能用 fallback 掩盖未知。

| 资源/事实 | 验证要求 |
|---|---|
| 本地 HardEvent，每物理核每有向 bank 8 IDs | 配对、旧 Wait→同 ID 新 Set 生效；保留 ID；真实生命周期共享 |
| 同引擎派发 | 只采用已审查的完成契约；包括短 MAD 反例，不能统一当 FIFO 完成 |
| cross-core wait | 是 actor 后续派发的 gate，纳入死锁/等待位置分析 |
| notify | 承接指定源 pipe 前序完成；不 fence 后续源操作 |
| 1 AIC + 2 AIV | 分片、完整贡献、quorum、GM 可见性与各 reader retirement |
| cross-core backlog | 按所用 mode/目标契约验证；反向通知消费 ack 不等于 buffer release |
| UB/L1/L0 | 每 AIV 192 KiB；AIC L1 512 KiB，L0A/B 各64 KiB，L0C 128 KiB |

最终 drain 只收尾实际未完成工作和资源；子 region 出口不默认 drain。
必要的局部 M/V 等完成等待可以存在，但每条都要有义务依据。禁止通用 DCE/CSE/
移动忽略异步 effects；后续存储合并也必须重新验证物理别名和容量。

## 7. 编译成本是架构约束

固定源码/constexpr 时，改变运行时长度不改变模板、证明图、ASC 和编译键。
不以“设置超时”代替正常速度，也不以缓存掩盖首次编译复杂度。

记 A 为展开到精确 pieces 后的访问数，H 为实际冲突数，Vc/Ec 为一个真实耦合
证明分量的图规模，Qc 为查询数，Sc 为候选修改次数。成本要按下面的项目测量：

```text
AST/native物化 + 访问相交索引/实际H
 + Σ分量(图构建 + 势函数/可达性查询 + 候选修改后的失效重查)
 + 最终IR重检 + emission/序列化
```

目标访问索引 `O(A log A + touched + H)`；最坏 H 可以是二次规模。普通图查询
一次为 O(Vc+Ec)，带 guard/整数位移的实际搜索按 proof states 单列；势松弛最坏
可达 O(Vc*Ec)。周期 LCM、分支路径、摘要 pieces 和候选搜索还可能放大成本。
不能把这条目标公式当作当前实现的已证明复杂度，更不能声称整个编译器线性。

优化顺序：消除 N 实例化 → 按实际耦合分量分解周期 → 归并等价义务/退休前缀 →
单次编译内复用分析/仅重查失效查询 → 固定预算资源搜索 → 实测热点的原生加速。
共享 bank、Scalar gate 或访问关系可能把两个分量连接起来，不能为了缩图错拆。
正确的 child 摘要按结构、参数绑定形式及入口合同复用，不按运行次数重建。

父子例已有单次样本 total 0.661 s：规划约0.262 s，最终验证0.282 s，snapshot
0.047 s，frontend0.022 s，lowering0.044 s。主要成本仍在同步/验证，不在源码
语法糖；这些是已有样本，不是本轮重新测试。完整五进程口径另为 p50 0.6889 s。
6 个模板对应48,455 proof states，也说明后续组合必须监测证明放大。

| 门槛（同机、固定版本、无并发压测） | 要求 |
|---|---|
| 完整源→ASC，至少5个冷进程 | 常用内核 p50≤1 s/max≤2 s；真实 FA ≤5/10 s |
| 长度不变性 | 多 runtime 轴分别变化；常量百万边界保留循环；无 N 个 task/证明状态 |
| 编译统计 | 分阶段时间、模板/pieces/H、周期、图、查询/重建、摘要大小、RSS |
| 预算 | 60 s 单 case 外部截止 + 内部 deadline/结构预算；返回 phase/计数 |
| 真缓存 | 命中目标≤100 ms；返回不可变产物，不重复证明；导入/加载另计 |

缓存键包含源码与宏传递内容、constexpr、目标/native 契约、编译器和选项；CANN
产物另含工具链身份。runtime 参数值和指针不入键；其 launch 检查每次执行。
先实现可度量的进程内缓存，再实现持久缓存、原子写入和失效测试。现有产物属性
读取不是编译缓存。新性能报告同时列 Python/import、完整调用与 CANN 时间。

## 8. 算子性能：先查最终代码，再结合设备解释

算子主体、宏、oracle、AscendC/ACLNN 适配器均放 examples。核心只保留语言、IR、
目标原语、同步/验证、codegen 和通用工具链。核心 wheel 必须可脱离 examples/
reference 安装。每新增原语至少给一个非 FA 的语义用例，避免算子名字特判。

FA 是本轮主验收程序，而非编译器内建能力。ops-transformer 的 A2 设计将通信
块128×1024、Cube块128×128、Vector块8×1024分开；源码还有跨 query taskId 和
延迟 descriptor。因此必须验收大通信块、独立子块、状态/alpha 寿命、规则连续流。
不能通过缩 BK、隐藏 GM packing 或每 query drain 达到迁移表面完成。

数值合同固定 FP32 m/l/O、p32 denominator、显式 P cast 和独立 rescale/add。
最终归一化要记录采用 div 还是 reciprocal+mul，不能因为数学等价就替换既有
实现。逻辑 BK 改变 online 分组与舍入，是新数值配置；native 分块也必须核对
实际累计次序，不能一概视作无数值影响。

三类证据分别回答：最终 ASC 是否复现固定 policy；同契约 AscendC 是否说明
编译器引入退化；ACLNN 与 ops-transformer 是否揭示仍有竞争差距。
AscendC 对照必须独立可运行并审查其预占事件；不能把生成代码改名作为对照。
上游源码模板的调用顺序不能直接当作库内部硬件时间线。

性能固定大例和统计方法见 [计划](refactor-plan.md)。先比较实际 native 次序、
工作量、搬运、存储复用、等待位置和填排，再解释 profiler。手排延迟差5%是
排查阈值，不是“专家性能已实现”的结论；与 CANN 的 gap 必须单独保留。

理论下界按真实 SKU/启用核/频率与工作量计算：

```text
t >= max(Fcube/Pcube, Vector各类工作量/对应吞吐上界,
         各搬运资源实际字节/对应带宽上界, 依赖关键路径下界)
```

共享资源上的需求先合计；可重叠资源不能简单相加。dense FA 两次矩阵乘的有效
FLOPs为 `4*B*Hq*Sq*Sk*D`；causal按可见pair计，原生padding工作另列。
GM 请求字节与 HBM 实际流量分开；热缓存与大于实机 L2 的轮换分别测量。
20 AIC×1.8 GHz×8192=294.912 TFLOP/s 仅是条件模型，不是所有910B的实测峰值。
未知持续带宽/Vector吞吐不能捏造。activity 可重叠，需结合 timeline/等待及对照
分析；只看某引擎占比不能确定因果。

## 9. 参考与范围

本轮只读核对，未 pull：TileLang `6ba65d204c937fcb1c43b1a22b32979506595655`；
ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`；CATLASS
`698379e0d4b1b94d12728e7181df2cf83a958e8f`；ops-nn
`64b41ad63a5073bc711ba03802498bfe8bdb77e5`，三个可更新参考库工作树干净。
外层报告的 `98f82c3` 等固定链接不是当前 checkout；安装 CANN 也不自动对应最新源码。

重点依据：[TileLang 910B](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[FA数值/连续流](../../reference/tilelang/examples/ascend/flash_attention/910b.md)、
[ops-transformer A2设计](../../reference/ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md)、
[arch22 Process/SetExtraInfo](../../reference/ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h)、
[CATLASS独立环](../../reference/catlass/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)。
其他算子的反例及安装头文件依据见 [source-notes](source-notes.md)。

normalization/Scalar、RoPE别名、gather/scatter、MoE release/routing、动态GMM、
量化/FFN、GDN carry/snapshot、paged/sparse任务都用于检验语义边界，不要求本轮
实现完整库。动态 ownership、全局phase、通信、训练backward/RNG分别定义合同后
再扩展；不可宣称有空接口就已支持。有限交错检查不是任意N证明，xDSL verifier
也不是同步正确性证明。设备性能始终需要实际代码与设备证据。
