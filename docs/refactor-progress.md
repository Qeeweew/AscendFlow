# 重构的验证记录

## R2 第一个切片：存储周期与协议周期分离

在 `7ace2dc` 上改为按物理分配分析地址余数，按实际同步执行条件构造完成图。
同端点的byte-cell义务先归并，整数同余求交保留各协议余数的首次真实代次；
最终checker从实际IR重新读取。新增证明域分配前预算、明确结构诊断和报告
计数。实现、前后同步及命令见[周期分离](factored-periods.md)。

同源码5次冷进程对照：5/7槽例完整源码→ASC p50从3.9671秒降到0.1726秒，
max从3.9765秒降到0.1886秒；peak RSS中位数约303→37 MiB。完成图2275→65
节点。Vector、normalization、CV、分段CV及5/7槽例的最终ASC全部逐字不变。
本次是编译性能改善，没有声称设备执行加速或完整FA迁移完成。

新例离线/shared编译通过，在物理1→逻辑0用同一binary运行12组长度，最大
16777216个f32元素；误差0、guard通过。241项扩大回归通过（57.76秒），最后
相关54项复验通过（22.23秒），计数不相加。外层证据在
`runs/refactor/factored-periods/`，最终冷编译使用 `cold-*-final/`。

29/31槽的地址分析现在可以分别完成；该固定排步无法证明8个本地event可安全
复用，仍明确拒绝。对同一分配上29/31周期别名，会报告实际耦合周期899的
预算问题。另有仅偶数代ready导致奇数代RAW缺失的反例，以及独立有限字节/
硬件DAG oracle。这些区分性能预算、别名和硬件事件寿命，不靠barrier兜底。

当前仍用LCM联合实际协议条件，尚未按协议连通分量分图；一般嵌套控制、分层
生命周期摘要、Cube显式L0/布局、跨通道event共享和完整可取消搜索仍未完成。
下一步继续R2的结构化物理区域与动态内层GEMM；R3/R4真实FA及变体和大规模
同契约AscendC/profiling验收继续保留为必需任务。

## R1 第三个切片：归约、Scalar 捕获、专家排步及 AscendC profiling

在 `10d345c` 上增加 typed reduce（显式 scratch/order）、Scalar UB read 和
捕获值乘法，统一进入原生 effects、generation 和最终同步验证。统计量最后
读取归于 Scalar read，Scale 只消费已捕获 f32 值；缺少 V→S ready、capture
晚于 consumer 或错 stage 均拒绝。`issue_order` 可按 Cube/Vector 固定 Stage
派发顺序，数值顺序继续按源程序定义。详细实现、数值/ASC和复现命令见
[AST normalization](ast-normalization.md)。

整数/布尔 `scf.if` results 和 `scf.for` iter_args/yields 已在前端接通，独立
解释测试覆盖零次循环、入口上界捕获、分支合流和 constexpr unroll 更新。
一般分支/嵌套循环的物理同步仍未接通；延迟 Scalar 描述符还需要版本化存储。

已完成的验证：

- 216项相关归约、前端、prefix、loop、编译和既有物理/算子回归通过（50.81秒）。
  增补空排步、CV policy、运行时工作量及 Vector profiling 检查后，相关30项
  通过（6.01秒）；不是把两个测试数相加作为全量覆盖。
- 独立 AscendC 源码与生成代码均通过 CANN9.1/dav-2201 离线/shared 编译；
  在物理卡1→逻辑0运行0/1/2/4/256 rows/group，最大8192×4096 f32。两者输出
  hash 一致，CPU参考最大绝对误差2.384185791015625e-7，guard通过。
- 每个版本3轮独立 msprof、每轮5 warmup+30 samples：设备 p50 的中位数
  为218.314 / 234.316 µs，生成版本比本例 AscendC 对照延迟低约6.8%。本例
  不含 gamma/bias，不能代替完整 RMSNorm/ACLNN 或 FA 的性能结论。
- 5次完整源码到 ASC 冷编译 p50=0.2895秒、max=0.2900秒、RSS约37.2 MiB。
- 原 prefix binary 补齐两独立参数下精确0/1/2/4/4096/65536 tiles 的门槛，
  最新17组调用均通过。记录在 `runs/refactor/r1-prefix-exact-gates/`。

性能分析绑定真实910B3的40 AIV、192 KiB UB/AIV、192 MiB L2；固定配置使用
32 AIV、UB65856字节/AIV，GM请求256 MiB。AIV持续频率、HBM流量/带宽仍未知，
报告只给工作量、请求口径intensity与条件下界形式，没有伪造硬件利用率峰值。
数据在 `runs/refactor/ast-normalize-comparison/`、`ast-normalize-profiles/`、
`ast-normalize-cold/`；回归日志在 `ast-normalize-checks/`。

下一步是 R2 的结构化物理控制/层次摘要、独立 buffer 时钟和可组合状态生命周期。
R0 其余算子的源码基线、完整例子迁移、唯一生产路径和 R3/R4 的真实 FA/变体/
大规模性能验收仍有未完成项。当前没有宣称整个重构完成。

## R1 第二个切片：原始元素长度、双参数与真实尾块

在 `b79025e` 基础上，增加非负 uniform 标量表达式的物理 lowering，保留
`min(n,m)` / `ceildiv(length,B)` SSA；GM extent 使用各自的原始长度。新例子
`examples/ast_prefix.py` 对 min(n,m) 个 f32 元素进行缩放，不修改输出后缀。
`valid=min(B,length-t*B)` 被传入搬运与计算，生成真实长度的 CANN native 调用。
完整语义、同步前后和最小反例见[运行时尾部](runtime-prefix.md)。

当前 prefix 证明要求同一 allocation 的访问使用相同的有效前缀、dtype 和
等宽对齐 cell；仅最后一个 generation 可部分有效。混入 full reader、不同
mask、错位 alias、跨 owner 或未知 recipe 时拒绝。此前周期依赖/事件证明在
该明确条件下复用；最终 IR 重检不信任 source 的 valid 声明。

物理模板已使用独立 `Templates` 数据类型，移除 `Program(count=0)` 容器。
uniform DAG 每节点只生成一次，避免嵌套三目表达式造成代码膨胀；证明输入有
4096项/128层预算。host launcher 增加长度/step/extent 溢出、DMA有效长度、
指针范围/对齐和外部区间不重叠检查；实际分配大小仍属于调用者提供的契约。

验证：

- 相关前端、source、loop、编译、prefix 回归104项通过（42.52秒）；随后对
  等价上界复用和 ASC 等待范围增加回归，prefix 23项通过（1.97秒），记录于
  `runs/refactor/ast-prefix-checks/prefix-final.log`。全目录 ruff/diff 检查通过。
- 独立有限测试从实际 SSA 和逐字节访问构造冲突/初始化义务，与实际派发/
  完成 DAG 对照；不调用生产依赖分析产生答案。仅在明确有限样本内成立。
- CANN9.1/dav-2201 离线/shared 编译通过。全部卡检查后使用物理卡1→逻辑0，
  同一 binary 的14组 n/m、每组5次通过，最大处理16,777,224个 f32 元素，
  误差0、未处理后缀及 guard 不变。负数、非对齐尾部、极大长度、重叠与错位
  指针等6类调用在 host 返回 -1。没有设备吞吐率结论。
- 完整源码到 ASC 的首批5次冷进程中位数0.1541秒、最大0.1673秒，RSS约31.5 MiB；
  当时同时进行回归，记录作为探索测量。最终独立计时中位数0.1542秒、最大
  0.1592秒、RSS约31.5 MiB，另存 `ast-prefix-cold-final/`。
- 生成代码只有3个 native 模板、4条局部通道、每 bank 两个 ID；与原 full-tile
  AST 示例的同步行逐条对照，不增加等待或扩大 release 到整个计算结束。

设备产物源码 hash 为
`4c774101b2cdc3a9ad1404ce8104a5066fba6c099dcbf4db696aa9d98c3d62f4`。
源码/IR/ASC/CANN 选项在外层 `runs/refactor/ast-prefix-final/`，设备记录在
`runs/refactor/ast-prefix-device/`，回归日志在 `runs/refactor/ast-prefix-checks/`。

R1 仍缺标量 SSA 合流/携带、归约设备例和完整 AscendC 性能基线；当前前缀
子集不支持非32字节 DMA 尾部或多 owner。R2 层次流水摘要、R3/R4 真实 FA
迁移和设备性能验收均未完成，不能把本切片作为整个重构完成。

## R1 首个切片：AST → 自动同步 → ASC → 设备

R0 的原型已保存为 `d3a00d0`。新 `N.prim_func` 返回捕获源码的 `PrimFunction`，
编译时解析 AST，设备函数体不在 Python 执行；`N.from_source` 是源码不可取得时
的明确入口。旧示例/回归改用显式 `N.trace_func`，没有 parser 失败后的 fallback。

已实现两遍参数解析、`N.int64`、显式 constexpr 特化、固定范围的常量捕获、
源码位置诊断、显式槽位和 AST statement/展开预算。`Pipelined` 保留 body 一份，
即使边界是常量或零；只有 `N.unroll` 进行受预算限制的展开。设备条件和 serial
循环形成真实 `arith/scf`，内含 typed memory operations，序列化后可重新解析。
目前它们的同步 lowering **明确拒绝**；标量合流/iter_args 也尚未接通。

`examples/ast_vector.py` 的单外层、完整64元素 tile、显式双槽程序已经经过现有
物理访问/代次/event 检查并生成 ASC。与同配置 legacy 程序对照，统一函数名后
ASC 完全一致；没有新增搬运、loop barrier 或缓冲。其源码、自动同步前后 IR、
ASC、离线 ELF/shared 库和 CANN 选项在外层 `runs/refactor/ast-vector/`。

全部8张卡健康、无进程、利用率0后，选择物理卡1映射逻辑0；runner 再次检查。
同一 binary 运行0/1/2/4/4096/65536/262144 tiles，最大16,777,216个 f32 元素，
每例预热1次、重复5次，误差0，输入只读及输出 guard 通过。记录在外层
`runs/refactor/ast-vector-device/device.json`。这些是 correctness 证据，所记时间
是含提交间隙的 ACL stream interval；单 AIV/小 tile 不是吞吐率目标验收。

编译 benchmark 新增 `--entry scale`，从 Python 定义模块读取/捕获、AST 和源 IR
构造计至 ASC，import/CANN 分开；每 case 有60秒进程预算，至少5个冷进程。
已有产物属性读取明确标记为非编译缓存。首批完整输入到 ASC 中位数0.2317秒、
最大0.2463秒、RSS约31.1 MiB；原始数据包含 package/source/ASC hash，位于
`runs/refactor/ast-vector-cold/`。最终源码复测为中位数0.2253秒、最大0.2274秒、
RSS约31.4 MiB，另存 `ast-vector-cold-final/`；最终生成 ASC 的 hash 与设备已
验证产物一致：`26e6a53204100b776c9f47c3b2fa2efe12e0cc7a7cca961cd5a9a56883c3025d`。

相关回归104项通过（38.85秒）；随后增加 serial 内实际访存及 decorator 契约
检查，最终前端/source 回归45项通过（11.13秒），ruff 全目录通过。
日志在外层 `runs/refactor/r0-checkpoint/ast-*.log`。未重跑的旧设备记录未冒充
当前新路径证据；最终再次编译的 ASC 与设备实际运行库的源码 hash 对照后记录。

R1 仍未完成：raw 元素长度与 `ceildiv` 的物理 lowering、精确尾部、两个独立
运行时参数的设备闭环、标量 SSA 合流/携带、归约例、可验证输入契约和完整
性能基线。R2 的层次摘要、R3/R4 的真实 FA 迁移和性能验收仍待执行。

复现（工作区根目录，设备命令先检查全部卡）：

```bash
.venv/bin/python compiler/examples/ast_vector.py --output runs/refactor/ast-vector --shared
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_vector.py --entry scale --trials 5 --output runs/refactor/ast-vector-cold
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/runtime_vector.py --frontend ast --output runs/refactor/ast-vector-device --library runs/refactor/ast-vector/scale.so --physical 1 --counts 0 1 2 4 4096 65536 262144 --repeats 5
```

## R0：实现现场保存与重新验证

设计提交 `a80aadd` 后重新检查符号循环工作树。执行
`../.venv/bin/pytest -q tests/test_loop_sync.py tests/test_source_language.py tests/test_compilation.py --durations=10`，
66项通过，38.45秒；包含之前未复测的图搜索顺序微调。原始日志位于外层
`runs/refactor/r0-checkpoint/tests.log`。`git diff --check` 通过；全目录 ruff
发现已有 `tests/test_planning.py:110` 的 pairwise 写法问题，另步修正。

本次 checkpoint 保存此前全部符号循环源码、回归和 runner，不宣称 R0 的全部
基线或新设计已验收。后续首先接入不执行用户函数体的 AST 前端，保留自动同步
和最终实际 IR 检查。固定 AscendC 对照及完整前端编译口径仍待补齐。

## 以下为设计修订时的历史记录

2026-10-14 的[总体设计](design.md)、[DSL 规格](dsl-multistage.md)和
[重构计划](refactor-plan.md)已重写；它们规定后续目标，不能作为实现证明。
本次文档修订没有改运行时代码、运行编译回归或启动设备实验。算子调参暂停。

## 修订时的实现快照

修订开始时产品 HEAD 为 `7c5d0a9`，其后有未提交的 `loop_*`、前端/lowering/codegen、
benchmark 和回归修改。下面符号循环记录来自该未提交原型，不代表 HEAD 单独
包含这些能力，也不表示最新工作树已经完整复验。

| 能力 | 实际状态 |
|---|---|
| TileLang 风格作用域 | tracing 原型已能构造 typed source IR；新受限 AST 前端尚未实现 |
| 运行时循环 | 单个连续外层循环，至多一个 uniform runtime size；完整 native tile |
| 嵌套/索引 | 静态内层展开上限64，动态嵌套拒绝；runtime 轴映射受限 |
| slot/周期 | `versions` 隐式关联；全局 LCM 上限840及模板预算，尚未因子化 |
| 同步/验证 | 周期 frontier、guarded completion 图、整数势函数；最终重读 IR，但共享分析 |
| 资源 | 符号路径尚无完整跨 channel 生命周期共享，不能套用有限路径的支持声明 |
| 算法 | 算法/oracle/算子适配器已在 examples；FA 仍是遗留物理构造 |
| CV 示例 | QK→乘法/cast→PV，含分段例；没有 online softmax，不是 FA |
| 编译缓存 | 不可变产物可重复读取；不等于已完成按源码键的真正编译缓存 |

符号分析的新记录位于外层 `runs/refactor/loop-complexity/`：三个例子的 ASC
离线/shared 编译成功。Vector 同一 binary 已运行0/1/2/4/4096/65536 tiles，
最大8,388,608个 f32 元素，误差0且 guard 检查通过；CV 同一 binary 已运行
0/1/2/4/4096 tasks，误差0。**最新符号分段 CV 只有离线编译证据，未做设备验证。**
历史静态分段 CV 的设备结果见下节，不能移用为新路径的证据。

| 较早三次冷进程中位数，N=4096 / 1048576 | 编译秒数 | peak RSS 约值 |
|---|---|---|
| Vector | 0.0666 / 0.0665 | 27 MiB |
| CV | 1.603 / 1.599 | 154 MiB |
| 分段 CV | 3.346 / 3.373 | 217 MiB |

这些计时从已有 source IR 开始，排除 Python import、源码 tracing/AST 构造和
CANN；不满足新版完整前端计时口径。随后线性 prefix 图将分段图约35k边降到
3894边、2394节点，生成 ASC 保持不变；`linear/*/compilation.json` 的单次探索
结果分别为0.0808 / 1.328 / 2.592秒，不是正式多次验收或真实 FA 编译性能。

`regressions.log` 有线性改造前360项通过记录；后续局部测试、失败修复及图搜索
微调不等于当前工作树全回归已通过。最终搜索顺序微调需在 R0 重新验证。本轮
不把旧测试次数、设备 correctness 或 ACL stream interval 当作吞吐率达标证据。

## 历史静态分段 CV 设备证据

2026-10-14：2 groups、4 tasks、3 segments，f16 输入/f32 输出；CANN 9.1 /
dav-2201 离线/shared 编译成功。检查全部卡后物理卡1映射逻辑0，执行10次，
独立 CPU 参考最大绝对误差0。数值契约是
`sum_chunk RN_f16(2 * QK_chunk^T) @ V_chunk`，没有 softmax。
源码、IR、ASC、编译选项、健康记录和 hash 位于外层 `runs/refactor/segmented-cv/`。

该历史 ASC 中全部新 QK segment 在旧 `probabilities_ready` Wait 前发射，
内部预取先于上一个 segment 的 Cube；提前至首个 export 后发布完整 Score
的负例被拒绝。这验证特定排步/数值，没有测吞吐率，也未证明新版 AST/符号嵌套。

## 以下为早期重构记录

旧实现保存在 `06fce4d`，原计划保存在 `9b40e2f`。下列“下一步”“尚未完成”
描述对应当时阶段；当前边界以上述快照为准，实施顺序以新版计划为准。

初始固定 FA 输入 B1/H1/Sq128/Sk2048/D128、950个 task 的 cProfile 记录为：
AutoSync 100.03秒、再验证4.95秒、含重检的 emission 5.58秒、witness/JSON
9.53秒。约8292万次调用，完成图构造320次、闭包361次、`Graph.edge` 约2807万次。
输入 SHA256 为 `08656eaa89ab69f71748ed9b5ee108dbb89f9891334d55884dc90306df4062e8`，
数据在外层 `runs/refactor/cpu-baseline.log` / `.prof`。这是带 profiler 的单次
定位，不能与上表的无 profiler 时间直接计算加速比。

## 同步分析和 Python 开销

物理访问只枚举自身的精确切分区间。完成图用线性前缀表示此前 native retirement；
Notify effect 不成为后续 source task 的完成约束。候选规划复用完成 DAG，以具体
路径和受影响义务重查被删除的边；最后的独立 checker 仍从实际最终 IR 提取
全部 effects、ownership、event 生命周期和 drain，不读取规划路径缓存。

新增有限随机 DAG 删除差分、native 图规模、前缀捕获反例；本轮局部/物理扩展/
source/规划相关135个回归通过，CV和分段旧回归45个通过。范围是有限回归和
有界执行模型，不是任意规模形式化证明。

新 Compilation 接口拥有独立输入，返回不可变代码、IR字符串和检查摘要。
普通输出复用最后一次完整检查的解析结果；外部可变 IR 的公共 emission 和
verification 仍重新检查。返回的可编辑 plan 是新对象，不携带缓存证明状态。
逐访问 witnesses 是显式诊断选项，普通编译不导出它。

旧固定输入的规划耗时已由原带 profiler 的100秒级降到正常运行约2–14秒；这
两个口径不能直接计算加速比。输出格式化从重复验证中分离后，探索测量的
ASC formatting 为20–64毫秒。常用内核1秒、4K/8K FA5秒的验收仍未完成，
尤其8K旧物理流规划仍超时，需要继续依据新的 CPU profile 修正热点。
正式多次 CPU benchmark 和峰值RSS将单独记录；当前不得宣称编译性能已达标。

## 源语言闭环

静态三阶段 Vector、双向 CV、多 segment CV 已接通。28个 source 回归通过，
覆盖零/一/两/四次循环、内外 clock、最后结果的 slot、覆盖后的 borrowed tile、
同 root 的独立输入视图、提前 ready 和逻辑轴越界。
分段 CV 的历史 CANN 和设备证据见本页上方。它是正确性验证，没有计时结论。

算法/oracle/算子适配器已迁出核心，通用 CLI 和冷进程 benchmark 已建立。
精确区间 frontier 改为索引表并合并同源同目标的相邻区间；不跨孔洞合并。
198个 frontier/source/物理/CV 回归通过，另有逐字节有限差分验证。

用户指出：主编译路径不能按输入长度展开。现有有限代次规划仅保留为遗留回归，
不再通过缓存把它当作最终循环编译模型。下一步 source/physical IR 保留运行时
SSA 上界；以周期访问 frontier、带代次位移的完成路径及整数势函数证明循环。
有限实例化仅用于测试。

仍待完成：符号循环同步；全部算子迁新 DSL；核心独立安装测试；
AscendC可运行对照及 profiler/hardware 比较流程。
这些是未完成工作，当前提交不能作为整个重构完成的声明。
