# 重构方案：先固定语言与性能合同，再替换实现

2026-10-14，代码基线 `b98d541`，工作树起始干净。本方案落实
[总体设计](design.md)和[DSL规格](dsl-multistage.md)。本轮重写文档，不恢复算子
调参，也不将文档列为重构已完成。此前 R0–R5 记录保留作历史；以下 P0–P5 是
新的工作包，明确区分已完成资产与待实现内容。

## 1. 保留、替换与完成标准

保留现有 AST/typed source、原生契约、精确访问、单流周期分析、完成前缀、
最终重检、不可变 Compilation、预算机制以及设备/负例证据。
不要先重排整个目录或重写全部同步器；逐步将当前父子特例变成受限区域组合。

| 当前限制 | 替换位置与策略 |
|---|---|
| `lower_source` 只接受一个外层流/限定父循环 | 保留 xDSL scf/region，lower 原语和控制分别处理；不建立另一棵可变程序树 |
| `_Actor` 同时处理分配、frame、work收集、发射 | 随区域组合提取明确职责，复用原生recipe；不复制第二份lowering |
| `Loop.invocation`、专用 `carry` 与跨调用桥 | 先作为已验证协议实现复用，再接到共同边界合同；不是立刻删除有效代码 |
| 每个 GM root 都按 parent 分区 | 以 effects 区分只读共享、独占写、交接scratch；独占frame仍是合法特例 |
| 全局协议周期及重复查询 | 按真实耦合关系分量化；保留bank/gate跨分量义务，度量失效重建 |
| finite/trace 与结构化双路径 | finite只保留显式legacy/测试；生产AST无隐式分流，替代证据齐全后删除重复实现 |

完成必须同时满足：真实 FA 可用新 DSL 表达并运行；支持规定的多级和规则连续
CV 政策；编译长度无关且达到耗时门槛；最终代码保持专家政策；大例有独立
AscendC、ACLNN、profiling及硬件工作量分析；核心可独立安装。
与手排等时只是保真条件，不能作为已经消除 CANN 竞争差距的结论。

## 2. 工作顺序

```text
P0 固定目标源码/数值/派发/性能合同（不重复大调研）
 → P1 顺序区域组合 + init/use + 只读共享
 → P2 多级CV组合 + 第一份真实AST FA
 → P3 规则跨query连续流 + 专家政策保真/大例性能解释
 → P4 基础FA变体 + API/缓存/生产路径收敛
 → P5 变长、分页、稀疏及动态persistent扩展
```

编译复杂度、ASC审查和固定性能对照贯穿 P1–P4，不等最终 FA 才暴露问题。
完整算子库、自动排步和无界调优不是这些步骤的前置条件。

### P0：把最终用例变成可审查的输入

此步产物是固定的 examples 目标源码/配置和期望表，不是庞大的新框架。
选择一个真实 online-softmax FA 配置，先明确：

- BNSD输入/输出，FP32 m/l/O、p32 denominator、P cast、最终归一化recipe；
  query cache、alpha、三个GM交接环、local scratch的各自代次和布局。
- BQ128/BK1024/D128、native/row块分别指定；明确哪种累计/归约次序与
  AscendC对照相同。BK128遗留算法不是BK1024的位级等价替代。
- 两份执行组织：query内嵌套、规则跨query连续流；按actor列出0/1/2/多任务
  激活表、固定issue order、不可提前的gate和真实release。
- 对照 ops-transformer arch22 的 Process/SetExtraInfo/Vec1/Vec2 与实际
  Matmul调用边界；固定独立 AscendC 源码。不能复制自动生成ASC作为独立基线。

旧 FA 的归一化使用 reciprocal+mul，迁移必须显式保留或另立数值配置，不能
仅写 O/l 后默认后端任意选择。当前文档中的候选update-before-softmax policy
也不能自动当作已复现CANN；先列出差异再固定所选对照。

退出条件：前端、lowering、同步、原语、性能五类缺口逐条映射到 P1/P2/P3。
未支持的源码应明确标记设计fixture，不能标成可运行example或通过测试。
不要求用户逐项批准普通实现选择；合同具体后直接进入 P1。

### P1：先证明可组合的局部区域

第一份实现提交聚焦 `init → child → use`，不是再增加一个没有边界工作的
独立算子。将初始化/输出保留为真正物理操作，接入同一effects与同步机制。

实施内容：

1. `frontend/source_ir/lowering`：保留普通typed工作、循环和入口捕获；支持
   单actor顺序区域。分配身份跨region不变，零child仍执行pre/post。
2. `region_summary/invocation_sync/loop_model`：用共同入口/出口合同接现有叶
   分析；导出每piece的writer、reader families及通知寿命，解决相邻区域义务。
3. 增加只读共享GM合同，扫描实际writes并保留别名检查；不放宽未知owner。
4. `loop_codegen/performance`：按真实控制emit/count，不因child=0跳过整个
   parent，不在region边界插PIPE_ALL；最终检查从物化IR重新推导。

最小正例：一次初始化、多次局部更新、循环外导出；Q cache供多个reader；两个
sibling复用scratch；FP32 carry与舍入snapshot用最小GDN式切片检验。
不在此步要求完成整个GDN算子或通用if/else，否则主线会被算子库建设拖住。

必要反例：child=0读未初始化；首读错用末代writer；只等最后派发reader；
把FP16 snapshot当FP32 carry；readonly声明下仍有写；跨调用重seed；跨bank误共享。
原父子GEMM要作为共同规则特例继续通过，保留其现有ASC/设备证据或说明具体差异。

退出条件：前后IR、实际同步序列及足够性说明齐全；独立有限字节/指令oracle覆盖
0/1/短于ring/跨ring；至少一个大规模设备例对照独立AscendC；冷编译达常用门槛。
如果组合只能靠whole-child等待成立，此步未通过，先修合同而非堆新特判。

### P2：多级CV与真实FA同时推进

依赖 P1。先将原分段CV迁入AST验证跨核合同，同时用同一能力接通真实FA，
不把“分段乘法/cast例已运行”作为阶段结束。

实现目标：

- 父Pipelined包含多个stage，每stage可有保留的serial/Pipelined子region；
  原生块/Vector行块与CV通信块独立。动态bound默认留在设备。
- 一组1AIC/2AIV的固定分片、真实完整贡献ready、MTE2读退休release；
  actor dispatch gate进入组合完成/进展分析。资源按整个组合分配。
- FA必需的first/last和有限guard合流、fractal子块、row reduce/cast及受限
  AST宏随用例接通。对未知layout/guard仍拒绝，不以opaque宏绕过证明。
- Q/m/l/O初始化、alpha延迟消费、最终归一化/输出全部走共同物理effects。
  例子有FP32递推，不能以假softmax替代；避免整通信块挤入UB。

反例：Score只发第一FIX就ready；漏一个AIV贡献；把Wait消费当GM release；
child事件与父/兄弟bank冲突；新Score gate挡住本应先发的旧Update；归约跨BK
误合并；未定义空域使用末次结果。有限oracle注明边界/状态数/是否截断。

退出条件：真实FA经AST→ASC→CANN→小例数值、固定大例和同契约AscendC/profiling；
BK1024/较小native及row块确实存在于最终代码；冷编译≤5/10 s。此时可以如实
报告query内嵌套实现，但尚不能宣称跨query专家组织已复现，必须继续 P3。

### P3：规则连续流与专家组织保真

此步从旧 R5 的“persistent”范围中提前：规则连续流不是任意动态任务队列。
以uniform K及同源商余解码实现 `tasks=queries*K`，保留有限模板，不建立K个
证明状态。将query/KV/child及slot的关系加入规范化，限制在可证明模式。

实现顺序：先验证非FA连续双输出/归约反例，再用P2同一个FA数学宏切换派发
政策。跨query保持Q/state归属和必要初末工作；允许末次PV/Update与新QK重叠。
不把查询边界变成每轮重新seed/drain，不用额外buffer降低资源难度。

退出条件：

- queries=0/1/2/17，K=0/1/2/7/1024，同binary；K0不除零，K1既first又last；
  K与环宽不整除仍正确；两runtime轴、百万常量界均不展开。
- 最终ASC对照固定专家政策，列出native数、加载/导出、跨核gate、各层release、
  buffer/ID峰值、跨query边界和唯一最终drain。必要局部barrier单独解释。
- 大例3轮AscendC/ACLNN/profiling比较；冷编译和结构成本达标。不能将
  将输入改成panel-major或预转置的算子与原布局CANN直接算成“同步器差距”。

性能差距分类：数值/通信粒度不同、跨query填排、可避免的wait、数据路径/布局、
指令效率、并行度/缓存与计时口径。每个待解决gap绑定最终ASC位置和profiler证据。
固定policy相对AscendC保真但仍落后ACLNN，可以是此功能的有效验收结果；
“性能总目标完成”仍需要解释并处理剩余可归因于编译器的差距。
算子调参保持暂停，不因测到gap就开始无界搜索。

### P4：基础变体与产品收敛

基础forward变体逐项验收：f16/bf16、MHA/GQA/MQA、prefill/decode、causal
绝对位置/非方形对齐、对齐与非对齐尾部、空序列/全mask的明确合同。
同一binary只对声明的runtime参数复用；dtype/layout/tile特化可以不同。
不能把所有变体压到一个未经验证的模板，也不靠CANN fallback获得支持标记。

缓存分两步：P1/P2期间可接入进程内源码/配置键缓存及失效测试；到P4验收持久
缓存和独立安装。runtime值不进入特化键，launch契约检查每次执行；源码、宏、
constexpr、target/native合同、编译器/选项/CANN变化必须失效。测冷编译/真命中，
`Compilation.asc`属性访问不算命中。缓存不可省略cache miss的最终验证。

已经迁移的算子只走结构化生产路径；有限实例化留测试/显式legacy，去掉隐式
marker分流，复用并最终收敛重复分析。wheel在不含examples/reference的临时
环境安装并编译普通程序；CI分纯Python、CANN、设备，不隐式依赖开发workspace。

退出条件：基础变体矩阵、独立安装、冷/热编译、回归、最终ASC和大例性能证据齐全。
历史文档标明适用路径；不把legacy能力表当作新DSL支持列表。

### P5：动态任务与更复杂的attention

顺序为 varlen → paged → sparse → 动态persistent；每项分别审查metadata读取、
descriptor捕获、别名/所有权、非矩形域、参与者和收尾。规则连续流已在P3，
不与动态任务混为一谈。split-KV/global phase需要单独的可见性与参与者合同。

MoE routing/finalize、gather/scatter、动态GMM、量化/FFN、完整GDN继续提供
反例和扩展需求；不因接口能写就宣称实现。backward/dropout/RNG/通信需单独
范围；不加入本轮基础forward完成标准，也不取消未来覆盖目标。

## 3. 每步共同的编译验收

| 维度 | 固定实验 |
|---|---|
| 输入长度 | 0/1/2/4/4096/1048576，多个轴独立变化；同一源码/constexpr无额外模板 |
| 静态规模 | 分别增加模板数、层级、访问pieces、guard、真实alias；记录增长曲线 |
| 周期 | 独立互素ring与真实耦合ring分别测；不得将runtime K作为枚举周期 |
| 事件资源 | 8-ID压力、共享bank、Scalar gate与资源搜索预算；区分unsat和未决 |
| 冷编译 | ≥5独立进程，源捕获/AST至ASC p50/max及RSS；import/CANN/诊断另列 |
| 热编译 | 真源码键命中≤100 ms；新输入长度不重编译，合同仍检查 |
| 失败路径 | ≤60 s单case外部上限；内部取消、结构预算和失败后的清理测试 |

常用内核p50≤1 s/max≤2 s；代表性FA≤5/10 s。指标是门槛，不是当前全部通过。
记录CPU/系统/Python/xDSL版本、源码/package/hash、最慢phase、T/A/H、proof states、
摘要大小、图节点/边、路径查询/重建次数。对runtime值要求结构不变；对static
复杂度给出实际成本来源，不能用一句O(N)或“已做摘要”替代分析。

先保留循环/缩小真实耦合域，再优化可测热点。缓存、并行编译和原生扩展只用于
剩余确定热点；不跳过最终验证、不靠调大预算让长时间搜索继续。失败一次有
定位数据后，先改小结构反例，不反复等待同一无预算大例。

## 4. 大规模设备与性能验收

| 用例 | 固定大规模输入 |
|---|---|
| Vector | 2^24个f32元素；多核完整工作集 |
| normalization | 8192×4096 f32，固定reduce和Scalar捕获政策 |
| 父子Cube | 20组×16outputs×256panels，M32/N64/BK64 f16，输入960 MiB |
| Dense FA | B1/Hq8/Hkv8/Sq=Sk4096/D128 f16 |
| GQA/MQA | B1/Hq8/Hkv2或1/Sq=Sk8192/D128 bf16 |
| Decode | B1/Hq32/Hkv8或1/Sq1/Sk32768/D128；报告可用并行度 |

小例穷尽边界/逐元素，大例完整finite/guard、公开分层数值采样；CPU资源允许时
补全量参考。随机输入、非对称输入、极值/尾部与结构化精确样本分别测试。
采样不是全输出误差证明；不同block rounding的ACLNN使用声明容差，不做位级等价断言。

设备先检查全部卡健康、进程、显存和利用率；选择健康空闲卡，每进程物理映射
逻辑0；不reset、无/tmp锁、不换卡掩盖错误。每版本三批独立测量，至少
5 warmup+30设备样本，报告p50/p90与批间波动，交替版本顺序控制漂移。

手排比值偏离5%进入归因；噪声覆盖差异就标未决，不取最好样本。热重复与
超过实机L2的buffer轮换分别记录；上述FA输入并非天然超过L2。profiling有自身
开销，诊断与计时配置必须相同；host提交、ACL stream区间、device task分别报告。

每次记录源/宏/policy、IR/ASC/库hash、工具链/flags、shape/dtype、数值合同、
布局与额外packing、设备SKU/活跃核/频率、cache政策和原始profile定位。
独立AscendC负责保真，安装ACLNN负责竞争比较，上游源码负责机制研究；三者
的版本不能互相冒充。最终代码审查固定包含：

- 搬运路径/次数、native矩阵块/归约/舍入、Scalar发射开销与padding工作。
- Q/state/scratch驻留范围、L1/L0/UB及event峰值；是否凭空增加搬运或存储。
- ready、实际最后reader release、gate位置、prologue/epilogue和最终drain。
- 生成的运行时循环/条件数量；固定策略与实际代码是否一致。

理论分析使用实际资源需求及条件吞吐上界。有效FLOPs与原生FLOPs、GM请求与
HBM流量分别列出；同一引擎工作先合计，可重叠引擎不盲目相加。带宽/频率
未知就明确保留未知。用timeline和固定对照检验瓶颈假设，不把activity占比
当因果证明；编译器结构改进也不能直接称为设备加速。

## 5. 提交与眼前的下一步

每个实现切片提交：可运行example、正/负例、源码→前后IR→ASC命令、支持与
拒绝范围、编译指标及所需设备证据。优先复用已有hash一致的证据，不因文档
改动重跑设备。只提交compiler产品文件；runs/环境/cache/构建产物在外层；不推送。

当前 `b98d541` 已保存全部现有实现；本次单独提交方案及入口文档修正。
接下来先完成P0的目标源码/契约表，再实施P1的 `init→child→use` 与只读共享。
P0不另开长调研，P1不扩成完整算子库；每个新抽象都要能解释真实FA的对应访问。
新方案不代表重构已完成，P1–P4仍需实际代码与三类验收共同闭环。
