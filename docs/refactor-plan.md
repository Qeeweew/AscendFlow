# 编译流程重构实施计划

2026-10-16，最初源码审查基线 `fdff8c5`。本计划替换此前“继续 GM→L1、构建 BQ256、
设备复测”的执行顺序。**算子优化继续暂停；本轮已实施编译流程重构，正在按固定源码
验收。可用实现与原计划的进一步复杂度压缩分开记录，不能将 R0–R6 整体标为完成。**
技术设计见[design.md](design.md)，变换合同见
[compilation-invariants.md](compilation-invariants.md)。提交、测试、冷编译结果与资格状态
统一见[状态](refactor-progress.md)；本页不以单次 probe 或机制落地宣布性能达标。

## 1. 要交付的结果

同一公开 DSL 经结构化编译路径生成 ASC。目标是普通编译一次建立物理访问和依赖，
按 region 合同构造同步，联合分配事件，保持不变量地物化；不在 child、parent、query、
allocation、final 各自重证整个程序。当前已消除受控正常路径中的独立终检重建和若干
重复事实建立；不同结构的后端、图求解与分阶段资源分配仍存在，后文明确列出边界。

同时做到：runtime 长度不参与分析展开；静态 fragment/guard/period 的成本可解释；
生成代码保持可读；保持既有排步、早 release、buffer 复用和 CV 重叠能力。
编译快、正确、机器代码无回退分别验收，不能用削减能力换编译数字。

本次不重新设计算子算法，不修改 FA tile、Vector 政策、GM/L1/UB 存储政策，也不进行
算子参数搜索。已有 examples、基准和失败记录保留，BQ256 仍是未完成编译的暂停候选。

## 2. 分阶段实施及退出条件

### R0：固定基线和成本口径

已落地：流程审查、变换不变量与失效边界，以及正常编译的 `analysis_work`、完成前沿、
纯域运算、进展 rank、资源池等工作计数。计数标明建立、复用、传播、比较和实际输出；
不把已有事实的消费再次算成一次独立验证，也不把省去终检等同没有分析成本。

`benchmarks.compile` 已支持显式 AST entry/constexpr、normal/deep 对照和每 trial
全新进程。默认五次，单次明确标为 probe；记录源码/package/ASC SHA、内部阶段、
源码读取到 ASC 与内部 snapshot 的总时间、完整进程 wall time 和真实峰值 RSS。
import、身份计算、产物磁盘写入另列；每组只保存一份产物，不复用跨编译分析 cache。

必须分别记录：T/A、访问相交 I、H、域/条件 K、unique queries、构造次数、重复复用、
origin 位长、实际 transfers/comparisons、静态窗口副本、资源约束、输出大小和各阶段时间。
RSS 用真实进程峰值，不能把各 flow 的峰值相加当作总峰值。CPU profile 时间与无探针
冷编译分开；N、静态 child count、tile 容量是三条不同规模轴。

继续验收：新分析仍须补齐对应的输入规模和工作量，不能由当前统计推定所有 I/K/H
维度已统一计数。每个超过预算的阶段应说明哪个输入维度放大了哪个输出。
历史 141 秒 FA 数据保留来源，不机械重跑旧路径；当前固定实现按第 4 节完成冷测。

### R1：分析所有权和重复事实的显式交付

已落地：`Compilation` 先取得输入的私有副本，再用单次编译拥有的 `PhysicalFacts`
把 lowering 建立的 loop/scope/sequence 描述和依赖交给规划。源值 producer 检查与
leaf 同步复用同一已建立依赖；显式不同的 live-in 合同不混用。

`scope_resources.py` 以具名的内部结果交付 child resource、parent pool、counter 与
配置合同。query 消费同一 invocation 的结果，只建立新增跨 query 义务。交付检查
实际 op/operand/attribute、块内顺序、clock/cut/guard/ID 对应，不能用 report flag、
历史 JSON 或可变 Module 的身份替代。源码或物理输入进入下一次编译时重新建立事实。

单次编译内还共享不可变 CountDomain 的 canonical/intersection/inclusion 纯运算；
它不保存图证明，也不跨编译保留结果。独立 `verify_plan` 等入口仍从实际 IR 建立事实。

继续验收：保持这些交付和失效反例；新增变换必须声明保持或使哪些事实失效。这里
落实的是已接通路径的所有权交付，不代表所有后端和每类派生分析已合成一个通用结果。

### R2：一次物理访问分析与紧凑 region transfer

已落地：源值读取检查与规划复用已提取的物理描述/依赖；`region_summary.py` 与
`scope_access.py` 对非 definition root 选择更便宜的固定重复摘要。stationary/periodic
且 N>2P 的受支持 root 扫描实际存储周期 P，保留精确 byte cuts、首末 writer、零次
identity 和完整 ReadSpan；fragment/visit 预算按实际静态分析记录计费，没有放宽上限。

`summarize_fixed_storage` 默认仍建立全部 `DefinitionRun/ReadBinding`。scope 只对
实际 `definition_roots` 显式请求完整历史；这些绑定不能用退出 last writer 替换。
有限漂移、特殊模式、短固定 count 等不适合周期化的 root 保留原精确 fixed 扫描，
不能为了压缩拒绝原来合法的 footprint。规则行的孔洞与 actor/slot 身份仍分别保留。

无 serial-query 的长 reader span 逐 cell 投影真实 first acquire 和 tail retirement
prefix，各 native family 分开；不能把最后 task.end 或最大的序号当成全体退休。
query 路径当前只压缩有匹配周期 writer 覆盖的 root，**长 ReadSpan 仍保留完整 fixed
读者表示**。已有 region/query transfer 复用相同访问模式，尚未统一所有层的摘要表达。

sequence 摘要还支持调用者显式选择输出：`materialize_cells=False` 省去不用的 sequence
输出 cells，`obligation_source_regions` 在构造物理 Obligation 前筛选来源。query 只请求
来源 region `"0"` 的义务，同时保留自身的 cells/parts 及完整物理范围、域和端点信息。
`_transfers` 的全部 reader/history/domain 推导不变，默认公共摘要仍输出全部内容；统计
分别记录实际物化与避免构造的对象数。这项投影不等于省去了相应语义分析或退休证明。

继续实现的原目标：统一 effects/definition 与 child/parent/query 的组合合同，进一步
压缩需要历史版本的固定重复、query 长 span 和不规则描述；不能声称所有 fixed count
成本已与次数无关。Vector 双缓冲、GEMM、原地状态和嵌套 child 的义务继续与独立小型
字节模型对照，访问构造不应因 planning/final 或 ID-only 更新重复建立。

若 descriptor 压缩改变访问孔洞、丢失部分 reader，或把 child 退化成整体 done，阶段失败。

### R3：有类型的完成转移与构造式同步

已落地：leaf 的 `DirectCompletion` 按共同 native milestone 合同选择协议，对可由
capture/acquire 构造直接建立的关系消费该构造合同；scope/sequence/query 将其结果与
真实 typed cuts 组合。Issue、QueueBegin、RetireCapture、ImplicitCompletion 继续分离，
Set effect 不进入 native retirement prefix，child exit 也不变成整体 done。

已有稀疏 fixed-child 投影保留同一完成偏序的闭包；本轮进一步按需/批量注册 origins，
共享目标 coverage，合并相同 guard/origin 的区间，维护 delta，并索引稀疏相交前沿。
append-only 图复用进展 rank，对新增约束及受影响后继增量松弛；失败诊断仍可完整检查
图约束。实际传播、比较、origin 位长、新边和 rank 松弛均保留计数，不以缓存报告代替
本次工作。

尚未完成的原理想边界：parent 的候选协议插入仍检查 coverage，guarded/anchored/
query 等层仍用具体图和符号完成前沿求解。现在减少的是重建、重复查询与无关前沿比较，
并未将所有结构改成只组合小型输入/输出 transfer，也未消除普通编译内全部全图传播。
后续应进一步收敛这些表示，只对确有需要的局部关系求解；不能改名后继续隐藏 origin×node
或域/窗口副本成本。

循环用启动/一步保持/退出合同，条件 guard 保留孔洞。进展和资源是独立义务；不能因为
内存关系成立就默认完成。同步捕获只扩大到真实所需里程碑，禁止多加等待换事件容量。

继续验收：相同物理政策下，同步安全与“不能多等待”的反例均通过；region 组合不引入
child/query drain。原来已支持的父 child 插入 cuts、跨 query release 都有表达。
尚未接通的模式明确报 unsupported，不暗中退回旧求解器，也不把拒绝称重构成功。

### R4：统一资源寿命与分配，先合同后 ID

已落地：child lifetime 与 parent pool 的实际分配结果交给 invocation/query 复用；
parent 的 self-reuse 查询跨 bank 批量执行，有限宽度、共享与受限借用保留确定规则。
ID-only 赋值检查协议对应，不重建 memory analysis 或重证已完成的内存关系。
child width 需要调整时复用物理描述/依赖，用覆盖实际候选端点的父投影选择合法资源；
query width 在 1..8 的硬件资源范围内有界选择，通常复用小窗口，只在需要时扩大静态
窗口，ID 赋值不重建已建立的 query 完成图。

保留普通共享、受限 parent/child 借用、配置保留、FIRST/LAST anchors、跨 query
整池配对和 CV counter 上限；资源的消费检查仍与内存安全分开。

尚未完成的原理想边界：child、parent、query 仍分阶段分配；query 仍从实际空闲 ID
中选择资源，宽度变化可能需要更大投影/窗口。这不是一次建立全部逻辑通知后统一联合
求解整个 scope 的 width、共享和 ID。后续应继续统一寿命约束与失效范围，而不是新增
同内存图的重复规划回路，或让既有 child ID 分配永久限制后续合法共享。

继续验收：ID-only 置换不触发 memory analysis；width/共享变更只使实际资源相关项失效。
安全置换、真实碰撞、guard 孔洞、跨 query 未消费通知和 CV backlog 均有反例。
已接受 BQ128 不能因新分配器把 ID 独占而被拒绝；未达此条件不能切换默认路径。

### R5：保持不变量的物化及正常/深度检查分离

已落地：公开 `compile(..., debug_verify=False)` 的 loop、retained scope/query 与
sequence 路径消费规划和物化持有的实际结果，正常出口不再调用对应独立 verifier 重建
整套证明。materializer 的实际端点/guard/ID 与原生 op/clock/address 对应检查仍执行；
ASC 由同一受控结果生成。`before_ir` 在同步变换前保存，全部 IR/report snapshot 工作
计入源码到 ASC 总时间及 `snapshot_s`，不能因原地变换或移出计时缩小成本。

`compile(..., debug_verify=True)` 显式增加对实际计划 IR 的独立终检，normal/deep
比较 ASC 与语义报告，工作量/来源字段分别解释。`synchronize=False` 的专家物理排步、
`verify_plan`、公开物理 emit 等入口仍独立检查；导出的可编辑 plan 没有可复用验证状态。
没有通用 `verify=False` 入口，不接受 planner bool 或伪造 verified 属性。

边界：有限静态/旧 mixed 等仍经 `sync._finish` 的路径保留独立检查；所有权重构也不
自动消除 R3 所列的规划内图求解。后续按同样合同逐条收敛，不能把当前可用 normal
路径写成全部后端已经只执行纯边界 transfer。验证破坏、修改输入与 normal/deep 差分
继续作为回归，结果见状态页。

### R6：收敛前端/标量流水与删除重复生产路径

已落地：`lower_source` 一次收集 lexical source scopes，交给 region/scope lowering；
删除 `_legacy_work` 与按 AST/frontend 标记选择不同遍历器的生产分支。带标记和未带
标记的合法 source IR 使用相同 clock/control admission，固定 child body 保留为一份。
`frontend` 属性只保留既有显式 L0 输入合同的用途，不再选择整套 source walker。

source→native 读取映射使用实际 producer/root/range 绑定，并通过 R1 的 owned facts
交给同步。physical Task IR、专家排步和独立物理验证入口继续存在，它们不等于已删除的
旧 `cube.task/vector.task` 公开 API，也不应为删除内部类而削减机器语义覆盖。

既有整数 canonicalize/CSE、event 常量、地址公共子表达式、原始轴合法性和 uniform DAG
继续保留；proof Expr 的数学整数与 ASC i64 语义仍须区分。这里没有另建一套新的标量
优化器。scope/sequence/symbolic 等后端的结构分派仍存在，R2/R3/R4 的进一步统一
不能因为删掉旧 walker 就一并标完。异步 effects、早 release、所有权与数值合同继续验收。

## 3. 现有模块的去留

| 当前职责 | 本轮实际落地 | 仍要继续收敛的目标 |
|---|---|---|
| `native_order.py` | 共用 native milestone 合同 | 在不同组合器中继续保持同一硬件合同 |
| `model/loop_model/scope_access` | owned 描述/依赖复用；选择性周期摘要 | 统一 effects/definition 与多层 region 绑定 |
| `region_summary/region_sequence` | pattern 复用、精确 byte frontier、完整所需历史 | 压缩 query 长 span、定义历史与更多合法 fixed 描述 |
| `ParentCompletion` | 持有实际 child 投影；按需 typed cuts；宽度适配复用物理事实 | 用更紧凑 child transfer 代替更多内部投影成本 |
| guarded/anchored/query | 批量 origins、共享 coverage、前沿/域运算压缩 | 统一域/时钟条款组合，减少跨层图传播 |
| parent/query/case 资源回路 | invocation 资源结果交付；ID-only 保持内存事实 | 一个联合逻辑通知模型处理配置/child/parent/query |
| normal 出口 | loop/scope/sequence 组合 owned 结果；deep/外部入口独立检查 | 按合同迁移仍用 `_finish` 的有限静态后端 |
| scoped/sequence/symbolic/mixed 分派 | 共用源 scope admission，仍有结构后端 | 继续统一公共阶段，支持域保持明确，不加算子特判 |
| 旧 source walker | `_legacy_work` 与属性遍历分流已删除 | 保留机器反例和独立有限 oracle，不恢复兼容生产路径 |

按职责替换而非先搬目录。每次合入注明已替换入口和仍存在的旧调用；不允许永久两套
生产算法都保留，正常编译还同时跑两遍“作比较”。

## 4. 复杂度与编译性能验收

首先验收工作量，再做冷计时。下表仍是总体验收目标；R2 保留的 definition 历史、query
长 span 和不规则 fixed 扫描等尚不满足全局次数无关目标，不能为满足表格拒绝这些合法
输入，也不能将其标为已完成：

| 规模轴 | 要看到的结果 |
|---|---|
| 同一 runtime N 输入 1、2、小边界、4K、8K、大数 | 同一个 binary；静态模板/分析结构不随运行值增长 |
| 普通固定循环 count 增大 | body 一份、首末常数变化，不枚举所有历史；显式 unroll 另计 |
| T/A 增长 | 规则流的增长与模板/访问及真实输出相关，不能平白增加 origin×node 状态 |
| 不相关 ring 周期增多 | 按相连分量组合，不全局 LCM |
| guard/alias/reader/层次复杂度增大 | 明确记录 K/I/H/J 的增长；真实困难不隐藏、重复工作不伪装为必需 |
| ID-only 修改 | 内存提取/依赖构造新增次数为 0 |
| 父层加入调用 | 复用 child 语义转移，只做实际绑定与新边界，不重跑 child verifier |
| 普通输出 | 不运行独立全图终检，完整建立者计数仍真实可见 |

规则无别名族做 T=8/16/32/64 的规模测试，按实际操作计数约束近线性增长；交错范围
族另外验证输出敏感成本。不能给任意 guard/资源图承诺线性，也不能靠全局阈值拒绝
所有较复杂但原来已支持的 kernel。

稳定集成后才做五个独立冷进程，源码/参数到 ASC 的目标：常用局部/GEMM p50≤1秒、
max≤2秒；现有 dense FA p50≤5秒、max≤10秒。阶段实现不代替这组性能验收，实际结果
见[状态](refactor-progress.md)。import、文件读取、前端、全部分析、标量化简、snapshot、
CANN 各自记录；总量不能靠移动阶段降低，CPU profile 与无探针冷测分开。

当前 `benchmarks.compile` 的 worker 默认总期限为 **300 秒**，可显式缩短，范围
(0,300]；不再是旧的60秒 worker 上限。期限从新进程启动前开始，覆盖 import、源码
读取/构造、编译、内部 snapshot、身份与产物写入，整个子进程组共用，成功或失败退出
均清理残留子进程。失败、超时、源码/package/ASC 身份变化立即落盘，停止后续
trial/mode/input。normal/deep 分组，每组五个新进程才是正式统计；单次是 probe。

从工作区根运行一个已实现的冷测入口示例：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/scale.py --entry scale --parameters '{"B":64}' --mode both --trials 5 --timeout 300 --output runs/compile-cold
```

源码到 ASC 冷测不含 CANN；需要完整文件构建时，Python/分析/ASC/CANN 合计仍遵守
单次300秒硬上限，较短的阶段预算不能替代它。上限是保险线，不是合格性能标准。

## 5. 测试重构与时间纪律

- 每阶段先运行直接受影响的合同/负例；稳定集成后做一次必要广泛回归。不要每个局部
  改动都重编全 FA，也不扫描一堆等价 shape。
- 关系/依赖、native effects、协议构造、event 资源、materialization 破坏、独立有限
  模型分层。新测试验证合同/反例，不镜像私有函数实现。
- 有限模型只在小 N/小状态枚举，保留实际二值位、字节访问和死锁；不导入生产依赖
  算法当 oracle，不冒充任意长度证明。
- 共用不可变 fixture，但编译规模测试和改变源码的反例不能用 artifact cache 取巧。
- 小型反例首先贯通 normal/deep，再用已有 FA 源码作为综合编译压力用例；不新增同算子
  的候选 example 或临时脚本。运行数据统一留外层 runs。
- 无新的代码改动/失败/未决问题，不重复已通过的长检查。某个阶段卡住先解释结构成本，
  禁止“扩大预算→再试完整构建”的循环。

以下为现有定向入口示例，执行结果以状态页为准；按改动范围选择，不一起机械执行：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_compact_fixed_summary.py compiler/tests/test_scope_access.py
.venv/bin/python -m pytest -q compiler/tests/test_scope_resource_reuse.py compiler/tests/test_scope_planning_reuse.py
.venv/bin/python -m pytest -q compiler/tests/test_compilation.py compiler/tests/test_compile_benchmark.py
```

最终冷计时使用已接通的测量工具和实际源码参数，禁止跨编译 cache；冷进程不代表清空
OS 文件 cache。新增入口/选项仍须实现后才给出复现命令。

## 6. 生成代码和设备资格

当前暂停算子优化，本节仅定义未来编译器替换的资格，不安排现在运行设备。
固定源码和政策，先核对 ASC 的 native/DMA 调用、buffer/scratch、实际地址、wait/release
切点、CV gate、循环填排、最终 drain。可以改变 ID 数字和等价整数文本，不能增加搬运、
barrier、query drain，不能把所有输入释放推到输出完成。

设备阶段再用同一库覆盖多种 runtime 长度，保留全输出小规模检查、大规模4K/8K验证和
计时。数据需关联 DSL/IR/ASC/库hash、选项、shape/dtype/芯片/计时口径。设备前检查
全部卡健康、进程、显存、利用率，显式物理卡→逻辑0，不 reset 或换卡绕过错误。

编译器重构首先与暂停前相同政策比较无回退；达到 ACLNN 的≤5%差距是后续算子性能目标，
不能用“重构无回退”替代。恢复该工作时仍同时比较独立 AscendC、ACLNN、profiling 与
硬件理论指标，不只看源码或单个 PMU 百分比。

## 7. 协作、交付和停止条件

按根 AGENTS 的 gpt-6.1-sol/max、多 Agent 一文件一写入者规则。主 Agent 固定公共合同、
集成和证据口径；独立任务可分给访问/前端、事件、不变量反例。冷计时独占，设备统一排队。
本轮源码实现与回归已按文件所有权分工；进入独占冷测后冻结源码与 CPU 测试。文档可
继续核对，具体测量时段和结果由主 Agent 统一记录，不在不同页面各写一套成绩。

每阶段交付：小补丁、已保持/新增的不变量、失效范围、复杂度计数、相关回归、生成前后
对照和未解决项。只提交 compiler 产品文件，不纳入外层环境、实验记录，不打包或推送。

以下情况停止进入下一阶段：当前支持能力丢失；等待范围扩大；借用/ownership 被简化
掉；已迁移的 normal 路径重新运行独立终检；复杂度下降仅因跨编译缓存或未计时阶段；
硬超时。规划内尚存在的图求解继续按 R3 计数和压缩，不能藏成“组合完成”。先报告具体
反例和规模，再修正设计，不能通过继续调算子掩盖编译器问题。
