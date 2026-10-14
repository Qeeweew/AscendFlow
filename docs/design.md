# xDSL / Ascend 910B：语言、同步与性能的共同设计

2026-10-14重新审视；实现基线为 `872bf6e`。本文规定目标和取舍，实际能力以
[验证记录](refactor-progress.md)为准；语法见[DSL规格](dsl-multistage.md)，执行顺序见
[重构计划](refactor-plan.md)。继续使用xDSL 0.73.0、ASC、CANN9.1、dav-2201；
不引入MLIR/TVM构建依赖。算子调参暂停，固定配置性能比较属于重构验收。

## 1. 结论与当前缺口

保留受限Python AST、显式存储和自动同步核心。重构的下一难点是**可组合的
层次流水**：需要同时表达CV多阶段、每个actor内部的DMA/计算流水，以及跨
循环调用仍存活的buffer和event。它不是给现有单循环再套一层语法。

三个验收条件必须同时满足：用户能读懂并控制执行组织；首次编译成本由程序
静态结构决定；最终ASC和设备执行保持预期的搬运、数值边界、复用及重叠。
正确性、编译时间、设备性能分别报告，任何一项不能代替其他项。

| 已有证据 | 尚未解决，不能外推 |
|---|---|
| AST单循环Vector、对齐尾部、归约/Scalar捕获 | 一般分支、循环外初始化/收尾的物理lowering |
| 显式L1/L0/C与运行时K归约，首清末导 | 父query/output循环与子K循环的组合 |
| 分离存储周期；5/7槽例冷编译约0.173秒 | 协议周期仍有LCM；跨channel共享与可取消搜索未完成 |
| K归约冷编译0.242秒，960 MiB例与手排AscendC约等时 | 不能说明真实FA的编译速度或设备性能达标 |
| legacy FA已在examples，有4K/8K设备记录 | AST真实FA迁移未完成；简单CV例没有online softmax |

此前方案的问题在于把“区域摘要”写成黑盒、缺少父子派发语义，并把过多能力
集中在R2一个阶段。本版固定最小语义，拆开可独立验收的步骤；保留历史编号，
不重做已经通过的单循环切片，也不把局部进展登记为完整R2。

## 2. 前端：TileLang风格、一种设备语义

公开语言采用 `@T.prim_func`、`T.Kernel`、`T.Pipelined`、`T.Stage`、
`T.Cube/Vector`、显式allocation/view和typed operations。`T`只是导入别名，
现有 `N` 不需要改名。用户不写 `cube.task/vector.task`、event ID或credit状态机。

| 概念 | 用户写法/选择 | 编译器责任 |
|---|---|---|
| 算法 | copy、matmul、reduce、cast、状态更新 | 类型、初始化与数值顺序 |
| 物理存储 | memory space、layout、slots、views | 精确footprint、别名、容量、ownership |
| 任务域 | runtime循环、条件、owner坐标 | 保留结构、捕获descriptor、验证激活域 |
| 执行组织 | stage位移、actor、每actor issue order | 派发、填排、必要同步与排空 |
| 特化 | 显式constexpr tile/layout/dtype/policy | 稳定特化键、合法性与资源检查 |

默认长度是device scalar，tile形状/布局/槽数/排步是constexpr。设备函数体不在
Python执行；常量循环也默认保留，只有显式 `unroll` 受预算展开。禁止解析失败
退回tracing。host工厂可选择配置，但不能在编译器内部按实际输入长度构造task列表。

普通buffer更新足以表达carry；借用tile value指向特定存储版本，不是隐式快照。
需要保留旧值必须显式存储，不能靠SSA名字假定它仍存在。宏放examples，按受限
AST展开普通typed操作，保留来源、scratch和数值策略；不另建算子编译后门。
一般宏尚未实现，迁移前不因使用方便放开任意Python调用。

### 三种独立的流水量

- **Stage位移**：stage s在该actor的物理step j处理逻辑代次g=j−s。
- **Buffer槽**：例如 `x[g % 2]`，决定哪些代次共享物理字节。
- **Event宽度**：由真实Set/Wait生命周期决定，与前两者分别验证。

stage不是硬件pipe，也不是时间长度。`Cube`是AIC作用域，里面可有MTE2、MTE1、
MAD、FIX；`Vector`是AIV作用域，里面可有Scalar、MTE2、V、MTE3。
AIC与AIV各自推进step，stage编号相等不表示两个控制器同时到达。
显式排步和未来自动排步使用同一依赖/资源验证；本轮不实现自动排步。

## 3. 结构化IR与层次组合

```text
Python AST
  → typed source IR（xDSL SSA、arith/scf、storage、stage/actor）
  → physical IR（保留循环/guard，native effects、dispatch与ownership）
  → 同步计划物化到physical IR
  → 从最终IR重检 → immutable Compilation → ASC → CANN
```

不新增一套与xDSL平行的可变程序树。索引、frontier、候选和摘要可用不可变
Python数据结构；变换前后的权威程序仍是IR。native recipe只能展开固定机器
操作，源码循环不能借recipe名义按N实例化。首版保留最小操作集合，不预建
通用图调度器、自动tiling框架、运行时软件队列或全局通信框架。

### 3.1 两种顺序与多个完成点

逻辑任务内的语句、循环递推和值关系定义计算/状态更新顺序。issue policy定义
控制器实际派发顺序；同步补足两者间必要的完成关系，不能把派发顺序当完成。

每条native操作记录：actor/pipe、源码位置、逻辑坐标、guard、allocation root、
layout及精确字节访问、R/W/RW、SPR效果和可信完成点。必须区分：

```text
发射/标量捕获；指定输出write_done；指定输入read_retired；
物理slot可复用；通知已消费；actor控制到达出口；kernel最终drain。
```

当前单条native操作的读退休与写完成取审查过的完成点；没有硬件依据，不捏造
指令内部更早release。早释放来自选择真正最后reader，而非把最后reader换成
整个计算。通用DCE/CSE/代码移动不得忽略异步effects、ownership或完成语义。

### 3.2 嵌套的准确含义

子pipeline有自己的逻辑坐标和step。父actor派发到该region时，运行其控制循环，
然后继续下一个父stage；**控制流退出不隐含等待子循环全部异步工作完成**。
仍在途的读写和通知被传到父层。首版不会自动把兄弟stage插进子循环，也不会
把所有循环乘成统一微步时钟。需要跨调用连续预载时，以后用显式连续任务流
表达；普通嵌套不自动提供这种调度能力。

代次使用 `(region identity, parent invocation, local generation)`；同一child
第二次被调用时，`i=0`不是整个kernel的第一代。slot可按内层、外层或两个轴
显式选择。region边界既不重置存储版本，也不重新seed已有event。

### 3.3 区域摘要的最小合同

摘要不是一个 `done` token，也不是child第一/最后task两点。需要保留下列关系：

| 摘要项 | 对父层的含义 |
|---|---|
| 参数与域 | 入口捕获的bound、parent坐标、owner、激活条件；空域单独表示 |
| 入口frontier | 哪些实际区域需要既有写版本/旧reader退休 |
| 访问关系 | 精确byte pieces与guard；不把union holes包成整个allocation |
| 出口frontier | 每区域的末次writer、仍需考虑的reader families与迭代关系 |
| 里程碑关系 | 各输入read-retired、各输出ready、控制出口间已经证明的完成路径 |
| 存活资源 | 未消费通知、待归还credits、ID bank及下次invocation的复用义务 |

构建规则先限制为：正步长规则循环、入口不变bound、静态owner、仿射及常数
floor/mod地址、明确first/last/尾部条件。对child先证明内部义务，再把这些关系
连到父的前后实际访问；参数换名不得改变child epoch。保留摘要不能证明的
边界义务，交父层解决；到根仍未解决则拒绝。父层核验后才能分配/复用ID。

零次摘要是“没有child访问，保留入口frontier”，不是“输出已经ready”。有读
无写的区域必须保留累计读取退休义务；单个最后read完成未必涵盖历史read。
末代写只改变出口，不能变成周期前驱初始化第一代。这些规则已由单层末代
反例约束，任意层次的闭合性尚未证明，必须逐个受支持结构实现与验收。

首个层次例固定为父output×子K reduction，检查L1末次MTE1、L0末次MAD、
L0C末次FIX分别导出。其次才组合两AIV的分段CV；不先建设任意Presburger求解器。

## 4. 同步、目标契约与资源

从每个alias root的精确访问前沿产生RAW/WAR/WAW和初始化义务，结合真实
guard/代次/ownership。未知动态别名不能默认为安全；要求可验证契约或拒绝。
copy padding、reduce scratch、Scalar metadata与SPR读写同样计入effects。

处理顺序：访问规范化 → 义务 → 已有硬件完成路径 → 有限协议候选及位置 →
event宽度/资源分配 → 删除有证据的冗余同步 → 物化 → 最终重检。
资源检查参与候选选择；一个贪心计划失败不等于程序数学上不可行。固定政策
下不能偷偷增加buffer、搬运、全局barrier或改变issue order。

| dav-2201约束 | 必须保留的验证 |
|---|---|
| 同pipe隐式顺序按指令/shape成立 | 短MAD不能无条件采用完成顺序 |
| local二值event，每有向bank 8 IDs | Set/Wait配对；旧Wait先于同ID新Set生效；seed/drain |
| 跨channel/child共享ID | 真实生命周期、同actor/bank和保留ID；不能仅看词法不重叠 |
| cross-core Wait阻挡后续派发 | 完成图包括actor dispatch gate，检查循环等待 |
| Notify承接此前源pipe工作，不是源fence | 不阻挡以后无关源工作；通知消费不等于buffer读退休 |
| 一个mixed group含1 AIC和2 AIV | 分片、quorum、GM可见性、release须包含实际参与者 |
| mode2有限backlog | 使用已审查协议；当前保守IDs0..7，不从编码宽度推断全部支持 |
| 910B无直接L0C→UB/L1→UB/UB→L1路径 | CV明确经过GM交接，不能套用950硬件 |

UB192 KiB/每AIV、L1 512 KiB、L0A/B各64 KiB、L0C128 KiB。硬件依据见
[源码记录](source-notes.md)。AscendC对照还要计入库预占event；例如TPipe预置
M_MTE1 IDs0/1/2，不能和手排credit碰撞。目标契约是证明前提，不是模型自证结论。

## 5. 编译时间：成本模型和预算

固定源码和constexpr配置，改变运行时N只能改变调用参数，不改变IR模板、
证明图、ASC或特化键。该约束对所有runtime轴分别成立；静态大N也不能触发
隐式展开。大整数表达式有位数成本，不等于按N实例化。

记T为native模板、A为精确访问片段、H为真实冲突义务，Pc为某个实际耦合
分量的余数规模，Cc为候选数，Vc/Ec为其完成图，Qc为证明查询数，Sc为候选修改次数：

```text
编译成本 = AST/类型/native物化
         + 访问索引及实际相交片段/H
         + 各区域/耦合分量的图构建与证明
         + 有预算的候选修改/受影响查询
         + 一次最终IR重检 + emission/快照。
```

访问索引目标为 `O(A log A + touched + H)`；H本身可二次增长。
图构建约 `O(Σ Pc*(Tc+Cc))` 只描述局部表示，不是整个编译器复杂度。
通用整数势松弛最坏可达 `O(Vc*Ec)`；重复候选/guard域可能放大Qc、Sc及状态数。
不承诺任意程序线性复杂度。分层摘要也必须按输出pieces/关系计费，不能用
“有摘要”掩盖指数增长，更不能枚举父子runtime长度的笛卡尔积。

first/last条件必须按参数化边界归一化，不能把实际N−1当作证明历史窗口。
同一child的静态分析按结构/契约及参数绑定形式复用，不按runtime invocation
次数重复构图。最终checker可重新生成自己的摘要缓存，但不复用planner的
已验证标记；改变layout/guard/ownership/排步时相关摘要失效。

优先顺序固定：保留循环 → 分开storage/protocol周期 → 精确frontier/退休前缀
→ 复用单次编译内不可变分析 → 受影响查询重查 → 有预算资源搜索。
只有profile证实某纯Python热点仍主导时，才考虑原生加速；并行化和缓存不能
修复错误的O(N)模型。语义与验证不能为了计时被省略。

每次记录AST、lowering、依赖、规划、最终验证、ASC、snapshot时间，以及T/A/H、
周期、图、proof states、查询/重建次数和RSS。当前已有部分计数；缺失的阶段
拆分、内部取消与真正源码编译缓存列为待办，不能把artifact属性访问称作cache hit。

在同一开发机、Python/xDSL版本及无并发测试负载下，完整源码→ASC初期门槛：
常用内核cold p50≤1秒，代表性FA≤5秒，至少5个独立进程，max≤相应目标2倍。
记录CPU/版本/编译器hash；import、CANN和诊断导出单列，补充进程端到端时间。
真缓存命中目标≤100 ms，键含源码/宏/constexpr/目标契约/编译器/选项；CANN产物
另含工具链身份，runtime值不入键。未实现缓存前不把该指标标为通过。

CPU单case默认60秒外部预算，内部还有结构/查询预算；后续搜索必须能检查
deadline并留下当前phase/计数。超预算给出具体诊断，不让用户一直等，也不自动
回退有限展开或加barrier。当前常量周期预算840、proof states预算100万只是
实现护栏；提高数字不算解决复杂度。

## 6. 算子性能：表达能力必须在ASC和设备上成立

算子、宏、oracle、AscendC/ACLNN适配器全放examples。core仅保留通用语言、IR、
target/native契约、同步、验证、ASC与工具链。需要新增原语时先给通用语义和
非FA反例，不按算子名特判。核心独立安装不得依赖examples/reference。

研究源码当前HEAD：TileLang `6ba65d2`、ops-transformer `4bbfb3b`、CATLASS
`698379e`、ops-nn `64b41ad`，本轮工作区均干净，未pull。外层审计固定链接中的
ops-transformer `98f82c3` 是历史版本；安装ACLNN二进制也不等于该checkout。

### FA暴露的必要语言能力

ops-transformer [A2设计](../../reference/ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md)
中的通信块128×1024、Vector行块8×1024、Cube原生块128×128是三个量。
[arch22实现](../../reference/ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h)的 `Process`、
`SetExtraInfo`、`ProcessVec1`分别展示多任务填排、descriptor携带、Vector行循环。
这是实现要求的来源，不表示产品已经复现其全部执行组织。

逻辑BK从128增至1024时，在整除条件下KV更新/交接次数减少8倍，但Score/P
总字节未必减少8倍；PV导出、状态更新、通知粒度及占用都会变化。BK同时改变
online-softmax分组和P舍入边界，不能当成无数值影响的编译优化。因此先保持
算法BK，允许native/row子块独立，再对固定同契约AscendC比较。

第一份AST FA必须是真实FP32 m/l/O递推、未舍入p32的denominator、显式P cast，
以及QK→Softmax→PV→UpdateO的CV多阶段。先固定一个专家policy：AIC新QK派发
先于旧P gate，AIV旧UpdateO先于新Score gate；不把这一policy冒称所有CANN
模板的顺序。当前参考 `Process` 可见Vec1先于Vec2，实际Matmul库派发另需核查。
门槛是能表达并验证所选顺序，随后以生成代码与profiling说明优劣。

### 三类比较各回答一个问题

| 比较 | 判据 |
|---|---|
| 源policy/IR/最终ASC与手排实现 | native次序、有效/实际工作量、搬运、容量、复用、wait位置是否保真 |
| 同数值/布局/排步的可运行AscendC | 重构是否引入设备退化；同卡、同缓存/计时口径 |
| 安装CANN/ACLNN及对应上游源码研究 | 竞争力差距与可能原因；记录不同算法/tiling，不能归因于同步器一项 |

大例固定4K f16 MHA、8K bf16 GQA、长KV decode，具体shape见计划。
独立三批、每批至少30个设备样本，报告p50/p90与批间波动；超过手排基线5%
需解释，噪声覆盖阈值则未决。热重复与大于实机L2的轮换分别测；工作集大
不自动证明HBM受限。host、ACL stream与device task计时分别报告。

理论分析使用实际SKU/活跃核/频率和原生工作量。在可信吞吐上界条件下：

```text
t_device >= max(F_cube/P_cube, Vector各类工作量/对应吞吐上界,
                HBM_bytes/BW_HBM, 各搬运路径bytes/BW_path,
                依赖关键路径下界)
```

共享端口/引擎上的工作先按资源需求合计；可重叠引擎不能简单相加。
dense FA有效matmul FLOPs为 `4*B*Hq*Sq*Sk*D`；causal按可见pair计，padding及
实际执行native FLOPs另列。GM请求字节不是HBM流量。20 AIC×1.8 GHz×8192
ops/cycle=294.912 TFLOP/s仅是条件模型；持续带宽/频率未知时保留未知。
activity ratios可重叠，须结合timeline、gate位置与对照解释，不能单凭占比定因果。

## 7. 验证范围和失效处理

结构/类型/输入契约 → 物理依赖/代次/事件 → 最终IR重新提取检查 → 独立有限
字节/交错oracle → ASC/CANN → 设备数值/性能。xDSL verifier不是同步证明；
有限oracle不是任意N证明；planner/checker共享分析需用独立反例持续约束。

诊断区分：确定的访问/死锁违例；缺失输入契约/不支持控制域；资源搜索未找到
方案或超预算。后两者不冒称程序必然不安全。每条关键同步提供反例、前后IR、
实际指令序列、足够性与等待范围说明。

除FA外，用normalization/Scalar捕获、CATLASS多级环、GDN carry/snapshot检验
当前重构；RoPE/gather/scatter、MoE finalize/routing、动态GMM/量化/FFN、
paged/sparse/persistent attention保留为扩展约束。动态索引唯一性、跨核owner、
全局phase和通信完成不能由核内event冒充，也不为未来能力先添加空支持接口。

产品Git只保存源码/测试/examples/docs；环境与原始运行数据在外层。
设备执行先检查所有卡，显式物理卡映射逻辑0；不reset、不用/tmp锁、不换卡
掩盖错误。可复用既有hash一致的证据，文档修改不触发无意义的设备重跑。
