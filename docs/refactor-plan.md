# 重构计划：从单层流水到可用的多阶段DSL

2026-10-14重新制定，代码基线 `872bf6e`。本计划落实[总体设计](design.md)和
[DSL规格](dsl-multistage.md)，保留R0–R5编号以便追溯历史。算子调参继续暂停；
固定policy、ASC对照和设备profiling是重构验收的一部分，不是另开调优任务。

## 1. 从实际进度出发

| 阶段 | 当前状态 | 不能替代的后续门槛 |
|---|---|---|
| R0 保存与基线 | 实现已保存；Vector、归约、Cube/K归约有手排对照 | 递推/层次CV/真实FA同契约基线仍需补齐 |
| R1 AST单循环 | Vector、双参数对齐prefix、归约、Scalar捕获已通过声明子集 | 一般有effects控制、宏、嵌套还未lower |
| R2 层次流水 | 周期分离、显式L0、动态K与限定父output×子K闭环已验证 | 通用循环外工作、状态/guard/子块、一般层次CV尚未完成 |
| R3 真实dense FA | legacy算法/设备数据可作输入 | 新AST路径、runtime KV、专家多级排步、大规模验收未完成 |
| R4 基本变体/收敛 | 尚未完成 | GQA/MQA、causal/tails；移除隐藏双生产路径 |
| R5 动态任务变体 | 独立后续范围 | varlen/paged/sparse/persistent与动态ownership |

`872bf6e`包含此前未提交的K归约工作及证据说明。221项相关回归、五次冷编译
和ASC hash核对已完成；数值/性能证据见[运行时归约](runtime-reduction.md)。
这些不是全产品回归或新FA性能结论。后续不重跑整套调研，也不重新实现R1。

## 2. 执行顺序与每步退出条件

```text
当前单循环基础
  → R2a 父输出循环 + 子K归约 + 跨调用生命周期
  → R2b 状态/guard/布局子块 + 非FA反例
  → R2c 分段CV组合 + 调用边界资源
  → R3 真实dense FA + 大块通信 + 三方性能分析
  → R4 基本变体 + 唯一生产路径
  → R5 动态metadata与复杂attention
```

复杂度计数、deadline与固定ASC回归随R2a一起接入，后续每步扩展；不等到FA
才处理编译爆炸。宏和独立安装检验随实际示例需要完成，不先造通用框架。

### R2a：第一个真正的父子流水

当前主程序的物理/设备/固定性能闭环已通过，证据见[运行时层次流水](runtime-hierarchy.md)。
第一项中的通用循环外typed工作仍需完成，与R2b的初始化/收尾共同推进；不能
把当前只含一个child的parent结构当作全部区域组合支持。以下保留原交付门槛。

程序：每AIC处理runtime `outputs`个output，每个output有runtime `panels`次
K累加。父用 `serial`、子用 `Pipelined`；复用已有首清末导、L1三槽/L0两槽。
输入继续明确panel-major，不加GM packing；零panels保留对应输出。

实现顺序：

1. 在physical IR保留父子region及SSA bound；给代次附scope与parent invocation。
   加入循环外typed工作与多runtime轴存储范围。动态row pitch先支持结构化
   tensor坐标及可检查的非负extent/stride，不扩成任意非线性地址求解。
2. 对一个已验证child导出入口/出口访问frontier、精确ready/read-retired关系和
   存活资源。child内部仍用已审查的单循环分析；其上界和父坐标作为参数。
3. 父组合child与相邻实际读写，分别证明L1→MTE2、L0→MTE1、L0C→MAD复用。
   父再次进入child不能重新seed同一未消费event；ID分配看组合后的生命周期。
4. 从最终嵌套IR重新验证，不信任planner的摘要标记；独立有限oracle直接枚举
   小域字节/指令，与生产摘要无共享依赖推导。unsupported region仍明确拒绝。

退出条件：

- 同一binary分别变化outputs=0/1/2/17、panels=0/1/2/7/1024；独立轴不进入
  特化键。静态百万级边界也不展开；模板、proof规模对runtime值不变。
- 故意遗漏跨调用FIX→clear、丢失child最后reader、错误重seed、空child
  输出ready、将parent/child epoch混用，都有最小反例及明确诊断。
- 展示父子before/after IR和实际ASC；无每child全pipe drain，L1释放不被扩大
  到MAD/FIX。只在kernel返回收尾，其他等待必须能对应实际访问/资源义务。
- 固定大例20组×16outputs×256panels、M32/N64/BK64 f16，输入960 MiB；
  同契约AscendC、设备oracle和profiling通过。数据量可复用先前量级，不能
  用单个output测试代替跨调用复用。性能差异绑定最终ASC解释。
- 包含前端的五次冷编译达到常用内核门槛；超预算能结束并留phase/结构计数。

此步只承诺词法嵌套派发和异步出口，不承诺把父兄弟stage穿插进child，也不
承诺跨query持续预载。若局部摘要无法证明组合，报告具体缺失关系再修正，
不能以child整体完成或额外buffer作为默认替代。

### R2b：状态、边界控制与精确子块

以两个程序推进：GDN风格FP32 carry/低精度snapshot；同一L1块供多个native
reader的分层Cube。它们检验FA也需要的能力，但不依赖FA算子特判。

交付：循环前初始化、循环后输出、first/last及限定if/else的effects合流；
每次入口bound捕获；精确fractal子块和子视图范围；parent状态与child scratch
的独立代次。通用分支不一口气放开，未证明guard配对/初始化时拒绝。

退出条件：零/一次/多次carry正确，snapshot不回写低精度carry；Scalar捕获后
原buffer可释放，但延迟consumer所需descriptor仍属于正确代次；非连续view
不以bounding box扩大访问；末代写不污染steady frontier；完整row reduce和
cast边界不移动。新增源码宏只展开typed原语并显式传scratch，不隐藏allocation。

对照固定GDN切片/分层Cube的ASC和设备结果。完整动态state ID、任意gather
唯一性不属于此步；旧静态GDN证据不能移作新路径通过记录。

### R2c：CV多阶段与层次资源组合

迁移现有分段CV到AST：外层QK/Vector/PV/Update阶段，内层native/Vector row
循环保留。两AIV分别读写明确GM分片；Score/P/PV完整贡献才ready，各GM槽的
release对应真正MTE2最后读取。三个交接环与各级local环共用同一分析。

退出条件：

- 新QK所有子块在旧P gate之前派发；AIV选定的旧输出工作先于新Score gate。
  固定issue order不能被同步放置偷偷破坏；跨核Wait作为dispatch gate检查。
- 错quorum、只发布第一子块、一个lane跳过通知、返回credit只代表Wait消费、
  child重入事件冲突和漏drain均有反例。有界交错注明状态数、范围和是否截断。
- 存储周期与协议周期继续分开。实际无关的协议分量不强制全局LCM；共享
  scalar gate/ID bank的分量不能为省图被错误拆开。资源搜索有确定预算和取消。
- 相同输入的各native/row分块不改变逻辑通信BK或浮点顺序；ASC无额外转置、
  搬运、整体child屏障。相关cold计时、固定AscendC及设备profiling齐全。

该例仍不是FA，必须继续R3。跨channel共享优先覆盖真实组合所需的银行压力，
不先承诺任意全局最优event分配。无法找到解时区分资源不足证据与搜索未决。

### R3：真实dense Flash Attention

从已有examples算法迁移，保留真实online softmax：FP32 m/l/O，denominator
读取p32，P在指定逻辑BK边界显式RN到operand dtype，再PV；最后归一化输出。
Q cache、state、CV通信槽、native/row scratch各自有生命周期。

交付顺序：先完成一个固定policy全链路；随后在保持逻辑数值契约的条件下，
验证通信BK1024与较小native/row块独立表达。优先研究ops-transformer A2
128×1024/8×1024组织；它是选配置的依据，不是把源码模板名当作性能保证。
同契约AscendC配置及oracle必须在迁移前固定。

退出条件：

1. 新FA源码和宏只在examples，普通typed操作；核心无FA名字分支/同步旁路。
   小例逐元素，固定大例B1/Hq8/Hkv8/Sq=Sk4096/D128 f16做公开分层采样、
   全输出finite及边界检查。随机极值/非对称输入补充规则整数样本。
2. 同一内核运行多KV长度，至少两个独立runtime轴不触发展开。源码→ASC
   cold p50≤5秒、max≤10秒；import/CANN单列，完整RSS/结构计数可解释。
3. 专家policy与最终ASC逐项对照：native/搬运数、GM路径、缓冲、cast/累加、
   wait位置、最后读取、prologue/epilogue。通信BK不足或逐query全drain不能
   成为迁移完成的隐藏条件；不能用全局barrier或额外GM packing补洞。
4. 同契约AscendC、安装ACLNN、上游源码研究三者分别给结论。保存3轮profiling，
   分解有效/原生工作量、各资源请求、条件理论下界和未解释gap。

本阶段可得出“固定配置保真，但仍落后ACLNN”的结论；这不等于性能总目标
完成。把差距区分为算法/通信粒度、等待/重叠、指令效率、硬件/缓存口径。
若需要改变逻辑BK或数值算法，形成后续独立优化任务，当前不无界搜tile。

### R4：基本FA变体与唯一生产路径

分项迁移GQA/MQA、causal、对齐及非对齐tails、明确空域/全mask行为。
Hq/Hkv整除及映射、非方形causal左上/右下对齐、absolute position进入契约。
不能用一个含糊causal开关替代多个oracle；分支不能造成两个AIV通知不平衡。
范围为forward；backward/dropout/训练RNG需要新语义，不在本次默认支持内。

固定大例：B1/Hq8/Hkv2/Sq=Sk8192/D128 bf16；MQA另测Hkv1。
增加Sq1/Sk32768/D128的decode，用批/head并行度解释性能；不以prefill有效
FLOPs衡量decode。non-square causal和尾部同时覆盖数值与通知。

已迁移示例统一走AST结构化编译，有限实例化留测试/显式legacy入口；最后
删除迁移范围内的隐藏marker分流与重复生产分析。先替代回归/证据再删旧代码。
核心wheel在无examples/reference路径环境可安装、编译一个普通程序。
真正源码编译缓存与正确失效在此阶段验收，artifact属性读取仍单独标记。

### R5：动态任务与其他高性能变体

varlen → paged → sparse → persistent分别立项，逐项定义metadata读退休、
任务descriptor捕获、非矩形域、参与者和动态ownership。persistent跨query
连续任务流须保留first/last与state归属，不能复用单query循环结果冒充支持。
动态GMM、MoE gather/finalize、量化/FFN及全局phase同样按已验证子集推进。
这些约束现在纳入设计审查，不提前暴露不可用接口，也不把R4写成“所有FA变体”。

## 3. 统一验收口径

### 编译复杂度

| 轴 | 必测/必记 |
|---|---|
| runtime长度 | 0/1/2/4/4096/1048576，多个轴独立变；同一binary及模板不变 |
| 静态结构 | native模板、精确pieces、真实重叠、深度、guard分支分别增长 |
| 周期/资源 | 独立与真实耦合互素ring；protocol predicates；8-ID bank压力 |
| 计时 | 至少5个独立冷进程，AST到ASC p50/max；进程启动/import/CANN分别记录 |
| 内存/搜索 | RSS、义务/图/证明状态/查询、候选与失效次数、最慢phase |
| 有效预算 | 60秒CPU case；内部deadline/状态预算；超限保存诊断并终止 |

常用内核p50≤1秒、max≤2秒；代表性FA≤5/10秒；真实cache hit目标≤100 ms。
结构增长不要求严格线性，但必须给成本来源，禁止输入长度线性实例化。
不反复运行未改代码的无预算慢例；先在较小结构定位热点，再用同一固定例复测。

### 设备性能与理论解释

大规模Vector固定2^24 f32元素、normalization8192×4096；Cube/K/FA按各阶段
固定配置。相同计时/缓存政策下，三批每批至少5warmup+30samples，报告设备
p50/p90及批间波动。手排中位数退化超过5%进入差异分析，噪声不支持判断时
标未决；不得筛选最优样本冒称通过。host/stream间隔不替代device task时间。

每条记录绑定源代码/宏/policy、IR/ASC/库hash、CANN编译选项、shape/dtype、
数值契约、实际device/SKU/核数/频率、cache政策、kernel名及profiler数据。
大于L2的工作集轮换另测；若持续HBM带宽或流量未知，报告GM请求率而非带宽。

差距分析必须同时回答：是否执行相同算法/工作量；是否复现专家等待和重叠；
哪个资源需求或依赖链限制性能；理论吞吐采用了什么硬件假设；哪些结论尚不能
由profiling支持。上游checkout不能冒充安装ACLNN的源码映射。

## 4. 实施边界与交付

模块按职责随功能提取，保持 `frontend/source_ir → lowering → physical IR →
loop analysis/planning/check → codegen` 方向。先实现R2a的调用边界，再从
`lowering.py/loop_model.py`提取被反复使用的区域分析；不先大规模搬文件。
已有native effects、线性完成前缀、精确frontier和最终重检保留并做差分回归。

每步提交包含实现、可运行example、负例、命令、before/after IR与ASC检查、
编译指标、设备证据和明确拒绝范围。只提交compiler产品文件；外层runs、
缓存、构建和环境不进Git，不推送。全部卡健康/占用检查后才设备运行，物理卡
显式映射逻辑0；不reset、不用/tmp锁、不换卡绕过错误。

**下一步完成R2a剩余的循环外工作及R2b状态/guard/布局子块。** 父output×子K
主闭环已有证据，不能以新增单层算子替代剩余工作。全部重构仍未完成；完成
标准是R2/R3/R4共同验收，而不是某个局部测试数或一份设计文档。
