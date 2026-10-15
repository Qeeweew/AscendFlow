# 重整计划：统一 DSL 到设备的完整编译链

2026-10-15。以[实际状态](refactor-progress.md)和用户最新要求为准，取代此前
“继续优化 legacy FA”与“先取消首尾窗口”的安排。当前已完成旧公开API删除、输入minimum/multiple合同、局部查询域拓扑共享
和通用构建期限；随后接通runtime轴SSA、商余寻址和cyclic局部ownership。
实际测试与编译结果见状态页。以下保留整个重构的目标和退出
条件，一般nested CV与完整FA仍未完成。

## 目标、边界与次序

只保留一套面向用户的 TileLang 风格 AST DSL：tensor/view、显式存储、
`serial/Pipelined/Stage`、`Cube/Vector` 和 typed 运算。用户表达多阶段流水及
必要的派发政策，编译器负责物理访问、异步完成、buffer 退休与事件资源。
不同执行引擎的局部流水和 CV 流水使用同一套可组合的语义。

**不要求 DSL 向后兼容。** 直接删除 `cube.task/vector.task` 类用户入口、
legacy tracer 和对应兼容路径，不提供 alias、弃用壳或 fallback。物理 TaskOp
是编译器内部表示，可以保留；这不是继续支持旧 DSL。

工作顺序是：入口与合同收敛 → runtime 访问/控制与一般 CV 组合 → 完整 DSL FA
→ 同主体连续流水与变体 → 多算子性能。event 表示、真实成本计数和测试整理
从第一阶段并行跟进，不能留到最后；也不要求先建成庞大通用求解器才能接 FA。
每步都交付实际源码、前后 IR、ASC 和相应验证，不以增加设计 fixture 代替实现。

## 1. 删除旧入口，建立可验收的新入口（本轮已实施）

- 移除顶层 `xdsl_npu.Kernel` 和公开 `xdsl_npu.builder.Kernel`；生产 lowering
  仍使用的最小 physical 构造器改为内部模块，不留下同名兼容导入。
- 删除 `language.trace_func`、执行设备 Python 体的 tracer、隐式版本分配和
  隐式 L0 的旧 `gemm` 语法。保留新 AST 的 `language.Kernel`、
  `Cube/Vector/Stage/Pipelined` 及共用 typed 构造能力；不能按名字机械删除
  当前前端也在使用的 `_Trace` 上下文或 `_matmul` 实现。
- 旧 examples/runner 退出产品入口。测试按目的迁移：硬件与同步负例迁到
  内部物理 fixture，用户语法测试改为 AST，只测试旧兼容行为的用例删除。
  独立 AscendC、数学 oracle 和有界解释器保留，不依赖旧 FA builder。
- 删除旧 attention suite 前，先把其全链300秒 watchdog、子进程清理和失败
  即停止逻辑提取到通用构建入口并迁移回归，不能在清理期间失去总期限保护。
- 本轮之前未提交的六个 block-sum 文件先明确保存其 diff/证据，再处理删除。
  不为保留这个实验延续旧 FA，也不覆盖用户工作。

验收：安装包只有新用户语言入口；旧 import/语法不再可用；正常 AST 示例继续
走公共 `compile()`；关键 RAW/WAR/WAW、错误代次、event 配对与双 AIV 负例未丢。
旧 FA 不再能运行不是回归，不以绕行内部 builder 的“新 example”替代迁移。

## 2. 完成 runtime 输入、索引、控制和存储合同

先接通真实 FA 需要的有限语言子集，同时用 GEMM、向量尾块和递推状态反例检验：

| 层 | 具体实现 | 必须拒绝或检测 |
|---|---|---|
| 输入/ABI | shape、stride、dtype、正值/非负、整除、参数顺序、workspace 与 external alias 前提进入 IR/launcher | 溢出、容量不足、违反可检查合同；不可证动态 ownership |
| 索引/descriptor | 保留 BNSD 维度结构、head/group 映射、受限 runtime 乘积与 divmod 的来源/范围 | 不能通过任意字符串放行非线性表达式；零除数和越界明确诊断 |
| 控制与时钟 | query/task/child 分开，Stage 位移整个 descriptor；uniform first/last、零次和短循环有实际激活域 | 不执行 Python 循环展开 runtime N；metadata 读取不能作为纯标量复制 |
| 访问/布局 | root、subview、有效域、native 实际 footprint、recast、布局转换与 padding 全部保留 | 不同 tensor 名字不证明不别名；mask 不能缩小真实 DMA footprint |
| 存储生命周期 | 区分环槽、跨 query 状态、只读共享和 fresh frame；按真实最后 reader 复用 | 不把 child 退出或某个 consumer 完成当作全部访问退休 |

运行长度不成为 constexpr；tile/layout/dtype/静态政策可特化。先支持带明确前提
的直接访问，再扩展间接访问；不提前建设任意非线性整数求解器。
各 group 的任务数不等、部分 group 零工作从首个 FA 闭环就要支持；256个
query tile 分给20组已经产生12/13差异，不能追加长度整除组数的限制来回避。

## 3. 一般多阶段 CV 与同步组合

复用现有 `native_order` 和已接通的符号完成表示，延伸到真实 nested mixed/CV。
已有首尾窗口已经取消，不重新实施；也不引入 FA 专用同步器。

固定的分析职责依次为：

1. 从实际物理访问提取 RAW/WAR/WAW、generation 和 ownership 义务。
2. 用同流水真实契约与已选协议扣除已覆盖关系，直接确定必要同步。
3. 分开验证 ready、最后读取/release、event 消费/复用和最终 drain。
4. 分配有限硬件资源，物化同步；最终从实际 IR 独立检查内存、事件和进展。

父子关系保留 actor/clock/guard 和暴露的完成边界；共享不可变拓扑与必要边界，
不为每个源、查询域复制整张图，不把所有 actor 强行排成一条串行时钟。
现已接入AST词法scope和fixed child完成投影，local sequence实际使用同一投影，
固定行步长列切片及其跨child访问frontier也已进入最终检查。以下下一步必须
在已有实现上扩展，不能再写一套脱离lowering的summary原型。直接固定child的
真实callsite、祖先地址别名、父Stage/issue policy及双AIV端点组合已经接线。
这一真实路径已完成最终IR破坏检查、CANN与多长度设备闭环，下一步解决两个反例：

- 子event寿命必须跨父调用计算；按完整通知ordinal选择ID，而不是在每个child
  中无条件重启同一组ID。先从当前拒绝的`nested_copy`验证Wait消费，再扩大共享。
- 共享L0C须先建立逻辑定义/read配对与RW recurrence，再选择独立定义的物理
  占用次序，生成相邻定义的WAW与全部最后reader的WAR。不能将所有访问按issue
  重排。`QK(t+1)→PV(t)`有负逻辑代次差；同步epoch与实际访问generation必须
  分开表示。用公共parent step可以表示它，但仍需保留端点的启动/排空激活域。
  `n=1`和最终PV连续占用必须通过独立trace，外部最终可观察值不得改变。

这两项通过后再接first/last与独立query/kv时钟；保留原FA source主体做验收。
双 AIV 的完整贡献、MTE2 import 的早 release、FIX 与共享 L0C 冲突都要表达。
源码控制退出不插入 child drain；不以增加 buffer、搬运或 barrier 接通。
先核实每个 native 的读退休、写完成和配置资源合同；当前主要使用 task end，
不能把未来包含多个 native 的 recipe 合成一个不透明 completion。只有硬件
依据允许时才区分更早的读退休，不能凭 SSA use-def 推定异步完成。

关键反例优先于长算子测试：最后 reader 后仍有无关计算、两个 reader、Set 后
继续工作、奇偶条件路径、零次 child、跨 query 新 QK 先于旧 PV、只有一方 AIV
到达、旧 Wait 尚未消费就再次 Set。超出当前证明能力和真实资源冲突分别诊断。

退出条件：完整 dense FA 的 nested 多级流水可编译、离线构建并通过小例设备
检查；最终 ASC 保留 QK/PV 的子流水与 Vector 行块循环。失败时停在具体合同
反例上修正公共表示，不再退回 legacy FA 做性能优化。

## 4. event ID：先正确表示与分配，再通用化简

现有分段周期编码及 xDSL `canonicalize → cse → canonicalize` 保留作为已有
能力，不能重复宣称待接入。后续分三层解决：

- **资源层**：把本地 channel、跨 region 活跃 event、入口 SPR 配置同步与收尾
  都纳入可审查资源清单。局部二值事件和跨核计数通知保持真实硬件契约。
- **分配层**：在实际通知寿命上证明旧 Wait 已消费后才能复用；规则环直接产生
  常量、rotation 或有界条件分段表达。覆盖跨 channel/region 可共享与真实并发
  超容量反例，不先枚举输入长度张表再压缩，不承诺任意程序全局最优着色。
- **表达层**：对纯整数 IR 做通用 canonicalize/CSE；需要时增加有范围/同余
  前提的通用规则。处理 guard 与 ordinal 时显式保持源 generation，不做 ASC
  字符串替换或某个 FA 表的特判，不移动异步 effects。

验收同时检查实际 Set/Wait ID、bank、代次、资源寿命和标量表达成本。已有有限
表可作小域差分 oracle，不能继续扩成产品主路径。源码短不等于等待范围更小，
也不等于 Scalar 性能已改善。

## 5. 编译耗时：按当前结构修，不归咎于 Python

runtime N 不参与静态规模。需要分别计量静态任务 T、访问片段 A、节点 V/边 E、
查询域 D、各节点条件事实 F、环周期与层次深度：

- 改动前每查询域复制拓扑带来约 `D*(V+E)` 建图成本；本轮已共用一份
  `V+E`拓扑，各域事实仍独立，传播与事实成本没有消失。
- `_remember` 对既有事实做包含扫描，单节点累计插入有 `F²` 风险；这是源码
  上界风险，必须用结构曲线核实，不能冒充已测渐近结论。
- 按 root 的周期/区间切分、父子组合及规划内部重复检查另计；最终独立 verifier
  保留。正常 `compile()` 已把最终 analysis 交给 emitter，不能再误报它重复验证。

本轮已补叶层真实domain/拓扑/条件片段/比较/更新/传播计数并共享拓扑；下一步
规范化条件集合、减少同一次规划重复提取，补父层同口径计数。优化需同时证明未扩大
等待范围或改变存储政策。缓存、C++ 重写和并行编译不作为结构失控的补救。

验收包括同 binary 多长度、独立区域成本、模板/深度/周期分别增长；定向用例
稳定后五冷进程测源码→ASC，import、CANN、写盘和 launch 分列。既有目标为常用
内核 p50≤1s/max≤2s、完整 FA p50≤5s/max≤10s，当前未达成完整 FA 验收。
全链单次构建硬上限为300秒，到时停止整个候选序列；第1步已要求先将旧 suite
的 watchdog 提取为通用入口，此处验收各入口确实共用总 deadline，避免 Python
与 CANN 各自另有一段预算。

## 6. FA 接入和变体矩阵

算子在 `examples/flash_attention/`；完整数学主体包含 QK、online softmax、P
舍入、PV、O 更新与归一化。head mapping/mask/状态更新由 example 表达，编译器
不按算子名/shape 生成隐藏任务。FP32 sum 树允许调整，按误差合同验收，不要求
位级一致；其他状态更新和舍入边界不能悄悄改变。

| 阶段 | 变体/政策 | 新增合同与验收 |
|---|---|---|
| FA-A | 完整 tile dense MHA，先f16再bf16 | runtime Sq/Sk、group不等长/零工作；嵌套 QK/PV/Vector、三个 GM 环、双 AIV；源码→设备完整链 |
| FA-B | 同主体 continuous、GQA/MQA、cross attention | runtime task divmod、跨 query 状态、共享 KV/head 映射 |
| FA-C | causal/offset、tails、decode | 有效域、全 mask 行、padding 初始化、空 owner；decode 低并行度政策独立记录 |
| FA-D | varlen、paged KV、sparse | segment/metadata epoch、间接范围、页/块有效性、真实任务 generation 与 ownership |
| FA-E | persistent、split-KV | 任务领取与状态生命周期；跨核或多 kernel partial 合并、launch/global phase 合同 |

decode/persistent 是调度维度，不能写成某个新函数存在就自动支持所有mask/dtype。
每个组合标注 frontend/lowering/sync/CANN/数值/性能状态；不一次复制一堆变体
文件，也不在未接通前把失败 fixture 改名为可运行 example。

## 7. 多算子性能：通用优化与明确政策

从完整 DSL 路径做性能归因，并持续覆盖 Vector 搬运/elementwise/归约/normalization、
GEMM/量化 FFN、RoPE/gather/scatter/MoE、动态 grouped matmul、GDN 递推状态，
以及 attention。后三类的间接访问/动态 ownership/有序状态需各自落实，不能
为了统一抽象改写算法。

每项性能工作先给最终 ASC 与源码专家实现差异，再做一次有假设的改动：
搬运与布局、buffer 复用、向量指令组织、流水重叠、分组负载。可调政策写在
example 的少量明确参数中；编译器新增优化必须按通用语义/硬件条件匹配，并有
FA 之外的实例或反例。禁止 shape→答案表、按名字特判和无界参数扫描。

固定大规模起点：MHA B1/H8/Hkv8/Sq=Sk4096/D128/f16，GQA
B1/H8/Hkv2/Sq=Sk8192/D64/bf16；接通后加入 H32/D128、cross4K×8K、
MQA 和 decode32K。Vector/GEMM/状态沿用已有大例，不为漂亮数字缩小规模。
分别记录热缓存与工作集超过 L2 的条件，不能仅凭 N 大就宣称测得 HBM 带宽。

对照必须包含生成 ASC、独立 AscendC、ops-transformer 等参考源码及安装 ACLNN、
profiling。相同 policy 的手写参考用于定位编译器损失，专家最佳 policy 用于
定位算法/调度差距；二者不同就明确记录。三批交替、warmup≥5/repeats≥30，
报告设备 kernel p50/p90 与批间波动；host stream interval 另列。

用核数/频率、Cube 工作、Vector 指令/数据量、UB/L1/L0 容量、L2 工作集及实测
pipe 活动定位瓶颈。条件峰值和理想下界单列；pipe 活动不能相加为总延迟，GM
请求字节不是实测 HBM 流量。超过 ACLNN 5% 是继续调查线，不是自动允许损失；
噪声覆盖写未决，未通过写失败。

## 8. 测试、工具与协作交付

测试按合同 → 同步关系/实际 IR 破坏 → 独立小域模型 → ASC/CANN → 设备分层。
移除旧接口兼容测试的同时，逐项保留仍有效的语义反例；有限模型不冒充任意
规模证明。常规修改先跑受影响层，集成稳定后才跑一次必要广泛回归。

统一产物身份：DSL 源码、编译器版本、IR、ASC、ABI/数值/ownership 合同、CANN
选项和库 hash；性能记录绑定同一产物及 shape/dtype/设备/输入/计时口径。
examples 只留可运行算法，测试 fixture 和设计模板不混入；运行材料留外层 runs。

多 Agent 按工作区 AGENTS.md 执行：`gpt-6.1-sol / max`，主 Agent 管公共接口
和集成，子 Agent 负责边界清楚的编码/反例/测试；一文件一写入者。设备工作
单一负责人排队，冷编译测量不与其他 CPU 压力任务混跑，不让每个 Agent 各跑
一遍全量测试。每个交付报告真实支持范围、命令、证据和剩余缺口。

旧API删除和关键反例迁移已完成；runtime输入标量与原始轴SSA、BNSD商余及cyclic
局部任务和普通f32 `fill(±inf)`已接通。下一步保留这些合同，接一般nested CV及
跨query状态，处理first/last初始化与父子完成边界。完整FA必须沿同一路径
通过同步与设备验收，不继续扩展内部有限测试路径或恢复旧FA性能搜索。
