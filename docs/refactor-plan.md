# 重构方案：以可组合语法、编译成本和机器执行共同验收

2026-10-14修订，实现基线`be957e8`。依据[设计](design.md)与[DSL规格](dsl-multistage.md)。
保留P0–P5编号以追踪已有工作，但修订依赖顺序；不将已做的工作重新列为从零开始。
算子调参继续暂停，固定政策的代码审查、正确性和性能对照继续作为重构验收。

## 1. 保存点与实际进度

| 工作包 | 现有资产 | 尚缺 |
|---|---|---|
| P0目标合同 | examples中完整FA目标源码、数值与派发表、容量检查 | 独立同政策AscendC；fixture不能运行 |
| P1局部组合 | 根部once/stream、共享ref、真实init/use、只读GM；1 GiB闭环 | 父serial体组合、统一绑定环境、跨region资源复用、snapshot切片 |
| P2多级CV/FA | 旧CV/分段反例、native与cross-core基础 | 新AST嵌套、完整softmax/FA、同政策对照 |
| P3连续任务 | 目标query/KV解码源码与有限派发表 | runtime商余的分析合同及完整编译/设备路径 |
| P4收敛 | 不可变Compilation、通用benchmark、算子已在examples | 宏/真缓存/基础变体/重复路径收敛及独立安装验收 |
| P5动态扩展 | 源码研究与反例需求 | 动态descriptor/ownership/全局phase各自的实现 |

原始实现历史保留，不回滚用户工作。`be957e8`保存本轮接手时的未提交实现，
本次方案重写另行提交。测试与设备证据见[progress](refactor-progress.md)。

## 2. 路线调整与停止条件

旧次序将连续流与公共资源问题留到完整FA以后，容易在局部例子上持续堆特例。
修订后的工作依赖为：

```text
现有保存点
  → P0补齐具体政策/独立参考（限于已有目标，不开新大调研）
  → P1a统一局部区域入口与父体组合
  → P3a提前验证规则task/query时钟和编译规模
  → P1b共享event寿命 + P2a层次CV切片
  → P2b真实AST FA（同时接入最小宏）
  → P3b同数学主体的连续CV FA及大例性能归因
  → P4基础变体、缓存、产品收敛
  → P5动态扩展
```

P3a是必要的架构检验，不是用小例替代P2/P3的FA验收。P0的参考适配及编译统计
可随实现更新，但在FA设备性能结论前必须有可运行对照，不再无限挂账。

每个切片同时交付源码、同步前后IR、最终ASC及分析计数；涉及设备行为时补
CANN/正确性/固定性能证据。任何一次超预算先保存phase/计数并缩成反例，修正
算法或明确子集，不重复长时间等待、不提高预算后继续试同一个大例。
如果新功能需要第四个按顶层形状分流的lowering入口，先审查公共区域合同。
如果靠改变tile、buffer数、布局、数值或drain才通过，登记为合同变更，不算原任务完成。

## 3. P0：把语言和性能目标固定为可执行合同

保留[FA目标源码](attention-target.md)，补齐以下缺口：

- 固定一份BNSD、BQ128/BK1024/D128、Cube CK128、Vector VR8的政策。目标参数
  是表达能力验收配置，不宣称最优。输入Sq/Sk为runtime，dtype/tile为constexpr。
- 核对全部QK/PV累计次序、reduce树、FP32 carry、p32 denominator、P舍入和最终
  reciprocal+mul；已有legacy与目标数值差异逐项写明，不只验证数学公式相同。
- 完成独立AscendC参考，显式写原语/手排事件、同样布局和buffer数、同样计时
  边界。不能从生成ASC反向复制，也不能靠高级API隐含改变tiling后仍称同政策。
- 对照ops-transformer当前arch22 `Process/SetExtraInfo/ProcessVec1/ProcessVec2`
  与A2设计文档。选用update-before-softmax政策时明确它与Vec1-before-Vec2源码
  的区别；以实际生成代码和profile判断效果，不能把模板调用顺序当完整时间线。
- 将目标源码缺口映射到具体source操作、view、control、sync和codegen责任点。
  核心不出现FA名字特判；容量计算不能代替native布局和event可行性检查。

退出条件：数学/存储/派发合同可审查，参考能独立离线编译并跑小例；记录其
编译器、flags、workspace、事件占用和源码hash。fixture继续明确标为未支持，
只有真实通过AST→ASC后才能替换为可运行example。

## 4. P1：从现有局部实现收敛到区域组合

### P1a：一个绑定环境，普通工作与循环共用入口

不再复制`_Actor`能力。按以下小步迁移，每步保持原例可用：

1. 分离allocation/uniform/frame绑定、native原语lowering和region遍历；为同一
   actor创建一个绑定环境，根部sequence不再生成多份临时module后按名字合并。
   实际SSA身份与source allocation身份必须保留，文本相等不代表值相同。
2. 将once/stream和旧invocation接到共同入口/出口合同，先实现
   `serial parent { init; child; use }`。父调用坐标及child入口bound保留为SSA，
   空child执行init/use，parent=0则整个body不执行。
3. 增加同一父体中的两个child及只读GM跨调用重读。独占输出检查实际联合
   owner域；未知漂移写拒绝。跨region借用值/Scalar捕获仅在版本证明接通后允许。
4. 最终checker重新提取所有区域、端点和真实初始化，解除叶入口假设；不能
   将内部`live_in`变成用户可绕过检查的参数。

先覆盖有限深度、规则frame、uniform bound与first/last控制；不同时做一般
多参数非线性求解。父输出轴与子归约轴独立保留，不做其长度乘积的Python对象。

退出例：保留state_sum、父子GEMM；新增父体真实init/use、多reader只读cache、
两child复用scratch。每例都有0/1/短于ring/跨ring与错误首读、漏reader负例。
原hash应一致；如改变，逐条解释ASC差异并按影响补设备证据。

### P1b：事件寿命和数值状态成为组合的一部分

当前不同region分配不相交ID只是可用起点。改为先提取逻辑channel的Set生效、
Wait消费和跨调用存活关系，再建立每有向bank的冲突关系；用有预算的确定性
分配复用已证不重叠的ID。不能因host/Scalar进入下一region就宣布旧通知退休。

必要用例：两个Cube子流水单独合法、合起来机械独占ID会超8，但真实寿命允许
复用；另一个实际并发寿命超8的版本必须拒绝。分支互斥、零次路径及下次调用
重seed都需验证。无法证明可复用可拒绝，但这种拒绝必须阻止相关FA政策验收，
不能把硬件可行的性能目标悄悄改小。

FP32 carry/低精度snapshot用一个GDN式最小切片验证：snapshot不改变下一代carry，
多个异步reader分别退休。不是要求实现完整GDN算子。

退出条件：统一最终checker、独立有界指令/逐字节oracle、资源容量负例；至少
一个新增组合完成CANN/设备对照；常用冷编译门槛通过。P1a后即可先做P3a，
避免为了所有通用组合延迟关键架构检验。

## 5. P3a：提前检验连续流的参数关系

先接一个非FA程序：uniform K、`task=query*K+kv`，有query级初始化/归约/输出、
task级双槽和child级不同ring。沿用已有计算原语，不新增算子算法。

分析保留受限商余关系及owner域；只对静态slot必要的K余数分类，短K单列。
不是声称任意runtime除法和动态alias都能证明。首尾查询不得生成与K或queries
成正比的窗口，也不能把K要求为constexpr来通过。

退出条件：

- 同一binary运行queries=0/1/2/17，K=0/1/2/7及长K；K0有定义且不除零，
  K1同时first/last，query槽和task槽不同，K与ring互素也正确。
- 编译规模实验分别扩大两个runtime轴，保留模板、证明域和查询数；百万常量
  界的源码值/hash允许不同，但结构计数不能按长度增长。
- 前后IR/ASC包含一个连续task控制，不在query边界重seed/drain；相关源码
  进入与P1相同的区域模型，不能另设attention/task-stream特判后端。

本步如果无法达到规模要求，先修关系表示，再进入复杂层次CV。

## 6. P2：层次CV与真实FA

### P2a：接通多级执行及跨核边界

在固定1AIC/2AIV组内，父pipeline stage可以有普通工作和保留的子serial/
Pipelined。扩展P1同一合同，纳入actor派发gate、分片、quorum、GM可见性与reader
release；所有event按组合后的寿命分配。有限例验证任一贡献漏失、提前ready、
reverse ack误作release、child覆盖父scratch、wait挡住本应先发工作等反例。

通信块/native块/Vector行块独立；所有native loop默认留设备。只实现FA真实
源码需要的translated view、first/last合流与原语，每个原语配非FA语义反例。
短分段CV仅是调试切片，不能作为P2结束。

### P2b：第一份完整AST FA

将fixture变成可运行examples，用同一数学主体分别接query内嵌套与后续连续流。
补齐宏最小子集，使用户能读出四个stage及各自scratch和政策；宏展开后的低层
源码/IR也能审查。宏不得隐藏同步、ownership或额外搬运。

第一份完整例必须有真实softmax、m/l/O、alpha延迟消费、P cast、末次归一化，
不能只跑QK/PV或“乘法代softmax”。明确最大值、归约、exp和舍入的数值检查。

退出条件：源→物理IR→同步→ASC→CANN→小例完整数值→固定大例；同政策
AscendC可运行并有profile。最终代码有BK1024/CK128/VR8分层，workspace随固定
slots和groups有界，未增加输入packing。冷编译≤5/10 s。
如果只有query内嵌套通过，应如实登记P2完成，继续P3b，不能宣布专家连续流水已复现。

## 7. P3b：完整FA连续CV与性能归因

复用P2数学主体和P3a关系分析，跨query保留Q cache、FP32 state和alpha的正确
归属；旧query末PV/Update与新QK可重叠。不要为连续流复制另一份数学实现。

最终ASC逐项对照独立参考：native指令及累计次序、搬运次数/字节、各bank存储、
GM handoff、精确release、gate位置、prologue/epilogue和唯一kernel最终drain。
同步可以语法不同，但覆盖的里程碑与允许的重叠必须解释清楚；不追求源码文本相同。

然后做固定大例的AscendC/ACLNN/profile比较。分类记录gap：编译器多余wait或
搬运、不同通信/数值政策、Scalar/native效率、跨query填排、核利用率或缓存/
计时差异。每个假设绑定ASC位置和设备证据；局部重构引入的退化必须修复。
同政策与AscendC一致而仍落后ACLNN，可以证明保真，但不能宣布总体性能目标完成。
算子参数搜索继续暂停，不因一个gap开启无界调优。

## 8. P4/P5：变体与产品收敛

P4基础forward分项验收：f16/bf16、MHA/GQA/MQA、prefill/decode、causal绝对位置
和非方形对齐、对齐/非对齐tails、空序列/全mask。每项单独定义数值/参与者/
有效域，不把所有变体塞入未经验证的万能模板。dtype/layout/tile可以特化，
声明为runtime的长度不能重新编译。未支持的组合明确诊断，不静默fallback。

缓存先接进程内内容键及失效测试，再接持久缓存。独立测试源码/宏/constexpr/
目标/选项/native契约/CANN改变导致miss；runtime参数变化仍命中，launch检查
每次执行。冷编译先达标，真命中目标≤100 ms，import/加载单列。

宏可用性在P2验收，缓存机制可在P1/P2有稳定键后接入，不把二者都等到最后。
P4完成独立wheel安装：无examples/reference/原workspace也能编译普通用户源码。
生产AST只有保留循环的路径；finite只留显式legacy/测试。删除重复路径前核对
替代证据，历史文档明确路径。CLI/benchmark是通用工具，不引用FA构造器。

P5为varlen→paged→sparse→动态persistent，逐项增加descriptor读取/捕获、间接
footprint、ownership和空任务参与者合同；split-KV/global phase另有可见性与
驻留/参与者条件。MoE、GMM、量化/FFN、GDN反例继续检验通用性。backward、
dropout/RNG和通信单独排期，不宣称现有接口已支持。

## 9. 编译复杂度和加速的固定验收

每次记录机器/CPU、Python/xDSL版本、package/source/ASC hash、分阶段时间、
峰值RSS、T/A/H、周期、证明窗口/状态/图边、查询/重建数、摘要pieces和候选数。
分析规模与wall time一起看，不能仅缓存最后结果掩盖首编。

| 维度 | 固定实验和判断 |
|---|---|
| runtime长度 | 同一源码/政策改变多个轴0/1/2/4096/1048576；编译键/IR/ASC不依赖值 |
| 常量边界 | 1024与1048576保留循环；允许常量文本变化，不允许模板/查询窗口随长度膨胀 |
| 静态结构 | T、region层数、精确pieces、guard分别增加，绘增长曲线定位主项 |
| alias | 无交集、局部交集、全交集分开；H二次时如实报告，不称线性 |
| 周期 | 独立29/31环与真实耦合环对照；不误做全局LCM、不漏共享bank/gate |
| 组合 | parent×child×sibling，摘要大小随静态结构而非执行次数变化 |
| event | 8-ID压力、可复用/不可复用、预算耗尽和失败后清理 |
| 时间 | ≥5独立冷进程源→ASC；常用p50≤1 s/max≤2 s，FA≤5/10 s |
| 失败路径 | 默认60 s外部deadline+内部轮询/结构预算，报告phase/计数，不提高预算掩盖问题 |

加速顺序：消除长度进入分析→限制实际耦合周期/边界窗口→复用摘要与查询→
减小候选重建→优化已profile热点→必要时原生内核。可并行的独立编译不是单次
复杂度修复；不得为加速跳过最终检查。尚无证据需要引入新的原生构建依赖。

## 10. 大规模设备与性能验收

| 用例 | 固定大规模配置 |
|---|---|
| Vector/状态累加 | Vector至少2^24个f32；已有state_sum X=1 GiB可作局部组合基准 |
| normalization | 8192×4096 f32，固定reduce/Scalar合同 |
| 父子Cube | 20组×16outputs×256panels，M32/N64/BK64 f16，输入960 MiB |
| Dense FA | B1/Hq8/Hkv8/S4096/D128 f16；主大例B1/Hq32/Hkv32/S8192/D128 bf16 |
| GQA/MQA | B1/Hq32/Hkv8或1/Sq=Sk8192/D128 bf16 |
| Decode | B1/Hq32/Hkv8或1/Sq1/Sk32768/D128，单列实际并行度 |

D64与D128、causal与dense、prefill与decode分开记工作量，不能用单个峰值代表。
更长case仅在固定矩阵不足以解释L2/并行度时追加，不做无界shape或参数搜索。
小例覆盖0/1/边界和完整数值，大例做完整finite/guard及公开分层误差检查；资源
允许时做全量参考。采样不能称全输出误差证明，随机/极值/非对称/精确样本分开。

设备前检查所有卡健康、进程、显存、利用率，选择健康空闲卡；每进程显式
物理→逻辑0映射，不reset、不用/tmp锁、不换卡掩盖设备错误。
每版本三批独立测量，交替顺序，至少warmup5/repeats30，报告p50/p90与批间波动。
profiling与非profiling计时分别记录；host提交、ACL stream和device task区间不混用。

热缓存与超过实机L2的buffer轮换分别测；大Sq并不自动保证输入工作集超过L2。
AscendC与generated使用相同缓存/计时条件，ACLNN明确是否包含workspace准备或
layout变换。偏差超过5%触发归因；噪声覆盖差异则标未决，不挑最好样本。

硬件分析与每份性能报告同时交付：

- 实际SKU、活跃AIC/AIV、容量/频率来源及持续频率未知项。
- 有效与native/padding FLOPs、各类Vector工作、GM请求/各级搬运及重复加载。
- 按共享资源汇总需求，以吞吐上界和依赖关键路径计算条件下界；不得把GM
  请求速率当HBM带宽，或将platform无单位rate代入GB/s。
- 对照timeline/活动率及等待位置解释瓶颈。活动率可重叠，不单凭MTE2/Cube占比
  下结论；带宽或Vector峰值未知则列参数化下界和需要的测量，不捏造数值。
- 保留CANN源码、安装库、独立AscendC和生成代码各自身份，不能互相冒充。

## 11. 下一次开发从哪里开始

先按P0补齐FA参考和政策差异表，直接用现有fixture而非再写宏观报告；随后
P1a抽取actor绑定环境，接通父体`init; child; use`。紧接P3a验证双runtime轴
及query/task/child代次关系，再推进资源复用与层次CV。每步完成具体退出条件
才登记，不以测试总数或文档完成代替功能交付。

工作区根目录的可复现检查命令：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_sequence_sync.py compiler/tests/test_region_sequence.py
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_state_sum.py --entry state_sum --trials 5 --output runs/refactor/state-sum/cold-review
.venv/bin/python compiler/examples/run_state_sum.py --output runs/refactor/state-sum --build
```

运行数据、构建产物、环境均在外层，不纳入产品Git；只提交compiler中的产品
文件，不推送。已有hash一致的设备证据可以复用，文档修改不触发无意义重跑。
