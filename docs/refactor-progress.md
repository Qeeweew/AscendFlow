# 当前状态与未解决问题

2026-10-16。本页是唯一当前状态入口。[设计](design.md)和[计划](refactor-plan.md)描述目标，不作为实现证据。

## 1. 当前工作和代码现场

用户已恢复重构，局部同步基线为`959f79e`，源码事实交接已提交为`a5ed7e6`。M0文档整理已完成，M1基础组件保存于`a111140`，effects/原语接线保存于`8773953`；M2局部同步已有可运行的新入口。公共入口仍未切换；新scale/批量GEMM已有CANN离线结果，没有运行设备。算子优化和旧求解器局部调优继续停止。

此前 `7ddaa54ec060a92e219d0462df162939b8b32d72` 的交付未达标。重构前全部实验和文档已保存到 `bbc1458`，标题为“Checkpoint paused synchronization experiments and structured-loop refactor plan”，不是资格声明。根目录没有产品Git。

保存在bbc1458的实验包括compact query patterns、virtual child cuts、局部transfer、边界批读和ownership区间合并，以及对应测试。它们不是新计划的已完成阶段；公共同步路径当前仍保留它们，待M2–M4替换。不能把它们默认为最终架构或称为全量已验证。

本轮文档从36份收敛为10份。原文、四个入口README及SHA256清单保存到[外层历史归档](../../reports/compiler-refactor/20261016-before-structured-loops/README.md)，不纳入产品Git。运行数据没有迁移、删除或改写。

### M1当前实现

- dialect已拆到`dialects/{source,physical,types,io}.py`，删除旧文件与反向注册依赖；无旧模块兼容转发。
- frontend已拆parser/builders/syntax/contracts，language只提供公开声明与捕获入口；移除frontend↔language以及对model诊断/标量助手的依赖。
- 原语与机器常量在target/native，f32字面语义在target/scalars；统一diagnostics支持类别/阶段/位置/witness，预算失败标为ANALYSIS_LIMIT。
- 构建期限和进程组清理下沉runtime/process，CANN驱动迁backend/cann；保留整个构建上限与失败清理。
- `passes/lower_pipeline.py`在真实SSA循环上建立最多4个PLE相位，不复制body，保留普通前后work、相邻循环、nested runtime以及serial lower/stride/carry结构。它当前是显式控制规范化入口，不是公共编译已接受全部这些组合。
- `analysis/frontier.py`为声明的前缀域保存至多一个rank；acquire/must-merge/quorum与通知identity明确，Clock按声明身份而非名字比较。尚不推断任意guard包含或跨clock关系。
- `analysis/regions.py`只代入固定边界wiring和局部保证；要求保留真实consumer，不能因后面的要求更强就删除早等待位置。`analysis/loops.py`只验证显式候选的一步保持和单独basecase，不求全闭包/不动点，不冒称完整exit summary。

本轮定向结果：构建watchdog24项；dialect/API25项；frontend113项及runtime注解16项；原语/effects18项；budget/fill18项。合并后控制/前沿/region/编译入口85项通过，随后加入loop归纳和同名clock反例后的新analysis/loops组77项通过。各组有重叠，不合计成独立总数；未跑全量回归。相关Ruff与diff检查通过。

拆分前后scale的ASC及source/before/plan IR逐字相同。真实FA仅做了source/control检查：276个操作、6个循环模板、3层嵌套、21个相位描述，body克隆和迭代展开均为0。证据在[控制阶段产物](../../runs/structured-loops/m1-control/results.json)；该阶段约4ms，不是FA source→ASC耗时，也没有重新取得≤5s资格。

随后把loop domain数据下沉`analysis/iteration.py`，source算术admission与PLE共用真实lower/upper/stride，修正负下界被误当0、非零起点被误拒、非零上界空循环仍检查不执行除法的问题；相关39项通过。`Actor.view`仍有旧零起点地址假设，所以控制/admission支持nonzero lower/stride不代表完整设备路径已支持。

M1尚缺控制有效域与实际native完成/入口出口的接线，以及由真实effects推导有限layout、carried候选和精确出口。一步保持结果是带前提的合同：相同guard标签本身不证明FIRST/LAST等条件在下一代仍成立，必须由控制层建立合法successor域。M2–M6未完成。

### M2局部闭环（仍有范围缺口）

- `passes/native_recipes.py`从旧`_Actor.emit`抽出copy、element/BRCB、reduce、Scalar和显式L0 matmul，返回真实task、定义位置、每个源输入的全部实际reader。旧logical Tile检查已消费这些reader，避免仅检查第一个reader而漏掉后续覆盖。控制、地址绑定及legacy L1 operand展开仍在旧adapter。
- `analysis/native.py`把已admit的单actor局部流水绑定到实际TaskOp/operand/root SSA、精确Ref及SPR effects。原语合同和view根在首次解码时建立，后续绑定clock不再重解码。嵌套/CV/动态任务尚不由此入口处理。
- `analysis/hazards.py`一次消费已有物理依赖，保留RAW/WAR/WAW、真实首尾cut、距离/余数/末次域和实际root。显式区分consumer logical generation与physical step坐标；没有通过名字证明不别名，没有枚举运行长度。
- `analysis/transitions.py`使用目标合同给出单指令native/Set/Wait/pipe fence转移。退休端口只是完成时的条件状态，派发不能取得它；Set effect不回注后续源流水。filtered implicit只覆盖满足目标合同的原语，不能借large MAD证明此前short MAD退休。已有capture的坐标代换同时处理prefix和handoff ordinal，不重放历史。

上述基础组件合并后的analysis、loop归纳、recipe及小例公共compile/API组共137项通过（5.89s）。后续实现如下，各阶段测试有重叠，不相加成全量回归总数：

- `analysis/dependencies.py`成为唯一周期访问扫描，在writer/pending-reader更新时直接保存ReadCarry；RAW/WAR已覆盖的WAW不进入必需协议。`mad_first/mad_cycle`清零代没有实际read，部分覆盖继续保留WAW；不做WAW与全部reader的事后两两搜索。
- `passes/local_sync.py`消费实际admission需求，按固定派发次序选择缺口。每个预声明engine/stage/activation键至多保存一个当前capture；同一consumer可使用其本来就需要的较晚源capture，但不会把早consumer的等待推到后面。普通queue与Scalar controller传播分别处理。
- `analysis/progress.py`只对最终选定指令建立一次整数排序合同，不回答内存覆盖查询。V个milestone变量、E条约束，至多V轮、O(VE)工作量；不是全来源闭包，也不含运行次数N。
- `analysis/resources.py`按真实consume→nextSet链建立最小可证width，然后分配ID。反向bank按接口索引，普通Wait不冒充controller gate；当前跨channel共享仍明确不支持。ID分配不重建内存关系。
- `passes/insert_sync.py::synchronize_local`和`backend/local.py::emit_local`已连接真实DSL lowering之后的局部物理IR、同步生成、资源、实际物化及ASC。测试禁止它们调用旧autosync、DirectCompletion、completion_graph和verify_loop；旧验证器只作单独迁移交叉检查，不能当独立硬件语义oracle。
- ASC直接消费同一个`LoopScheduleAttr.phases`，使用普通for或prologue/steady/epilogue及独立short分支，去掉稳态stage判断；不在后端再维护一份相位分解公式。phase有效域用于化简同步guard、删去恒假指令。原始IR body不克隆，ASC可能常量倍增，需要计量输出/编译成本。用户允许这些输出形式变化，不以旧ASC文本一致为交付条件。

本轮相关合并组148项通过（9.97s）；最后统一canonical phase来源后，输出与真实集成组59项通过（3.02s）。各组有重叠，非全量回归。真实scale/Cube固定N从2^10改到2^20，模板/协议/进展变量及selection/resource工作量相同。Scalar管线复用MTE2→S ready对controller的贡献，省掉重复MTE2→V通知，lease通过真实Scalar桥闭合。

明确边界：该入口目前只接受一个已admit的本地actor/clock；terminal-only native、尚未显式展开的Scalar SPR setup、嵌套/CV、跨调用lease仍未接通。legacy local LoopChannel/LoopSync编码和native/launcher打印工具暂时复用，不代表M4统一IR/后端完成。主线的query/child/parent求解器仍未删除；FA尚不走新入口。`lower_and_synchronize_local`现已把源码所有权、逻辑Tile检查与新局部同步交接成一个入口：每次编译保留同一Loop和完整DependencyAnalysis，原语合同、物理提取和访问扫描各建立一次。外部物理IR入口仍每次独立接纳，不信任verified属性或跨编译缓存。递归控制仍通过旧lower_source，尚未替换其结构分派。

可复现的CPU新入口（产品目录内执行）：

```python
from examples.vector.scale import scale
from xdsl_npu.passes.insert_sync import lower_and_synchronize_local
from xdsl_npu.backend.local import emit_local
plan = lower_and_synchronize_local(scale)
print(emit_local(plan))
```

源码事实交接新增10项回归；与局部同步、scope lowering和编译入口合并49项通过（10.74s），Ruff及diff检查通过。验证覆盖真实Scale/Cube逐native一次合同、一次Loop提取/依赖扫描、不可变统计与快照、外部修改重新接纳、逻辑Tile覆盖诊断；未重跑FA、CANN或设备。

提交`959f79e`对应的真实scale和`examples/gemm/batched.py`源码/同步前后IR/ASC、源码hash及CANN日志在[本地闭环产物](../../runs/structured-loops/m2-local/)。构建由该目录的`build_probe.py`与`runtime.process.bounded_process`执行，单次整个Python→CANN进程组上限300秒。小例离线结果不能替代FA编译门槛或设备正确性/性能。

`959f79e`的单次新进程结果（不作为此后事实交接改动的重新离线资格）（source→ASC含源码载入和IR快照，Python包import另列在JSON；不是五次冷测）：

| 真实DSL | source→ASC | 其中同步 | 整个Python→CANN构建 | 资格 |
|---|---:|---:|---:|---|
| scale，B64/f32、两个UB slot、runtime tiles | 0.262s | 0.0072s | 2.827s | CPU与CANN离线通过 |
| batched_matmul，M32/N64/K64、f16→f32、20 groups、L1三槽/L0两槽、runtime tiles | 0.485s | 0.0178s | 3.088s | CPU与CANN离线通过 |

两个结果均未设备执行，不是FA的加速成绩。对应ASC可直接查看[scale.asc](../../runs/structured-loops/m2-local/scale/scale.asc)和[batched_matmul.asc](../../runs/structured-loops/m2-local/batched_matmul/batched_matmul.asc)。

恢复后的顺序：先把真实source控制、native lowering与已建立的effects/依赖交接成一个入口；再完成嵌套loop边界、terminal/SPR、CV和共享lease；然后切换公共compile并删除旧专用求解器及过渡编码/打印适配；最后执行默认FA五次冷编译、相关算子回归、CANN和大规模设备资格。继续保持FA≤5s/≤10s等验收门槛。

### 递归 native 结构降级（尚未接入同步）

`passes/lower_native.py::lower_native_structure`现可直接消费真实DSL：普通`scf.For/If`保留初值、结果和yield；显式Pipeline降到`PipelineLoopOp/StageOp`，每stage绑定自己的logical ordinal/iv。循环前后/相邻/内部普通native走同一递归遍历，不调用旧lower_source/region/scopes分派。显式issue_order仍保留，循环体不按N展开。

`passes/native_addresses.py`用原Buffer root和完整SSA byte offset表达地址，保留所有祖先坐标、源axis及subview displacement。同root不同基址不再被合并到单capture。`analysis/value_bindings.py`保留实际writer、所有native reader以及PipelineLast有限DAG；Stage/Domain只转发，SCF carry保持零次初值。Task scope按真实循环/Stage/分支共享父链接，不复制访问历史。source除法检查只为实际存在的除法收集祖先范围，仍检查未使用的原除法，不强迫无关scalar carry化成旧access公式。

递归结构基线已提交为`d561378`；后续接入单Mixed控制树、参数化ActorDomain和按family保存的issue_order，仍返回`NativeStructure`而不是可执行计划。**动态bounds、ownership、原语/SPR接纳、stale Tile及LastTile非空使用条件、递归边界同步还没有接通；不能直接生成ASC。** 跨Stage Scalar引用、SCF Tile carry/merge、valid prefix、动态native row pitch和partial fractal明确拒绝；不是静默选择旧路径。现有局部同步入口仍消费旧lower_source的物理表示，下一步必须把这两部分接在一起。

相关集成和index组59项通过（2.52s）；随后补充未使用除法/无关carry反例，iteration组9项通过（0.56s）。新物理IR45项、地址26项、值绑定13项分别通过；各组不合计成全量结果。真实Scale派发/地址、Cube排步与writer/reader、嵌套前后work、零/短循环有有限轨迹对照。固定N=2^10与2^20保持相同native/loop模板及构造工作量。该阶段未运行FA同步/ASC、CANN或设备，不给出整编译加速结论。

用户已明确下一步的CV代码形态：AIC:AIV=1:2时两个AIV共用一份body，通过runtime sid区分地址/角色。同步分析仍区分两参与者的私有UB、ownership和quorum，不以复制AIV IR来区分身份。新结构降级已落实：每个source Vector域只生成一个带sid参数的physical域，不生成aiv0/aiv1副本。GM root跨域保持同一SSA，UB实例需由(root,participant)区分，不能把sid自动加到UB基址。共享loop的物理step由各actor控制器独立推进，不含全局锁步。旧mixed生成器尚未切换。

新增shared AIV与递归集成26项通过（2.21s），物理结构组65项通过（0.97s）；均非设备验证。真实默认FA也已通过新结构降级：6个loop模板、44个native模板、2个Cube域和2个Vector域（softmax/update各一份），Sq/Sk保持runtime，迭代展开0。单次构造到native结构约0.492s，含IR快照/源码清单约0.674s，**不包含同步或ASC，不是≤5s编译资格**。源码和IR身份、源码清单、复现脚本在[中间结构证据](../../runs/structured-loops/m3-native/attention/result.json)。源码仍是原默认FA，未改变算法或调参。

CANN契约核对：当前ASC C API的`asc_get_block_idx()`为group id，`asc_get_sub_block_id()`为sid；C++ `AscendC::GetBlockIdx()`在AIV是group*ratio+sid，不能混用两套公式。共享AIV代码不改变mode2通知的两个实际参与者贡献，不能因单body减少quorum，也不能默认给flag加sid。下一步是实际native/地址接纳与递归边界同步，随后统一资源和后端。

## 2. 编译耗时：明确未达标

固定默认FA政策：B1/Hq8/Hkv8/D128、f16、BQ128/BK1024/CK128/VR8、groups20，Sq/Sk保持runtime。目标为五个独立进程source→ASC p50≤5s、max≤10s。

| 代码/运行身份 | 结果 | 能说明什么 |
|---|---|---|
| 已提交7ddaa54对应冷测 | source→ASC p50 53.114s，max 54.309s，5次 | 上次交付未达标；不是当前实验成绩 |
| 未提交边界压缩早期probe | source→ASC 31.022s，1次 | 单次编译可完成，不能当五次p50或最终达标 |
| 未提交最近采样诊断 | wall 37.840s，内部compile 37.707s，规划34.699s | 仍很慢；含诊断，不与冷测混为同一统计样本 |

证据：外层[旧五次汇总](../../runs/compiler-refactor/comparison.json)、[单次probe](../../runs/compile-threshold/fa-boundary-probe/benchmark.json)、[最近采样](../../runs/compile-threshold/frontier-profile/profile-samples.json)及[工作量](../../runs/compile-threshold/frontier-profile/profile-report.json)。

最近采样：76个native模板；父流水972节点，跨外层任务的分析窗口2926节点。父层13个域、1082个来源、109次传播批次。

| 层 | 最终条件事实 | 沿边传播 | 事实比较 |
|---|---:|---:|---:|
| 父流水 | 27,069 | 548,885 | 1,749,299 |
| 外层任务窗口 | 18,394 | 459,096 | 1,894,151 |

两层已超过100万次传播，尚不含局部child分析。压缩节点没有消除重复工作，不能继续将“节点少了”作为完成证据。最近lowering约1.574s、emission约0.798s、snapshot约0.472s，说明同步以外也占用≤5s预算。

上述两次实验的ASC SHA均为 `e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`，与默认旧基线一致。这是该配置的输出一致证据，不是新的CANN/设备测量。

## 3. 问题清单

| 编号 | 当前问题 | 新计划中的处理 |
|---|---|---|
| A1 | serial/KV/fixed-child结构特判，QueryChannel与专门副本图 | M1/M3/M4统一PLE、边界、回边并删除query路径 |
| A2 | 多层全来源条件传播、增边重放与区间比较 | M1先确定有限前沿/归纳规则，M2/M3替代正常路径 |
| A3 | child/parent/query分阶段资源求解 | M3统一逻辑lease，保留共享/borrow与硬件机制区别 |
| A4 | 93模块平铺，职责闭环，emitter依赖planner | M1开始按职责迁移，M4统一输出；见源码结构页 |
| A5 | 源语言接受的carry/组合在lowering才拒绝 | 统一admission、源位置和能力矩阵 |
| A6 | 原语合同多处维护、Load2D展开/化简藏在emitter | 透明recipe及唯一原语合同，纯标量显式pass |
| A7 | normal/deep分离局部落地，但旧分派仍存在 | 保留所有权原则，删除重复后端；外部输入独立验证 |
| A8 | 测试依赖旧图细节，独立性表述过强 | 语义反例迁移、有限模型分级、结构成本测试 |
| A9 | 历史报告、无效命令和现状互相矛盾 | 本轮归档、重写索引；今后只更新本页状态 |
| A10 | FA编译和相对ACLNN性能均未达标 | 先编译结构验收；设备资格及算子调优另阶段执行 |

## 4. 当前可用路径与资格范围

| 范围 | 现有实现/证据 | 边界 |
|---|---|---|
| AST DSL→xDSL→ASC | Vector、normalization、GEMM、状态和默认dense FA示例已接通旧主线 | 新统一循环算法尚未实现；语法通过不代表任意组合通过 |
| 普通编译事实所有权 | 已提交版本有normal/deep分离、资源结果交付和纯整数化简 | 不等于统一算法或成本达标 |
| CPU回归 | 7ddaa54对应旧记录有分段2831项；停止实验有定向回归 | 不把历史总数写成当前工作树全量通过 |
| 新局部路径 | scale和独立批量GEMM经新选择/资源/PLE输出，已CANN离线编译 | 不是完整重构、FA或设备资格；Scalar SPR/terminal/nested仍有缺口 |
| 历史CANN | 7ddaa54记录FA/GEMM/scale离线构建成功 | 不能代表新FA路径已通过 |
| 默认FA设备 | 对应历史产物有短域全输出、4K/8K采样与msprof | 当前未提交分析实验没有新的设备资格；采样不是全输出 |
| Attention变体 | runner可表达部分dtype/head/tile政策 | 不代表causal、tail、varlen、paged/sparse、persistent等全部实现 |

公开 `cube.task/vector.task` 已移除，不恢复兼容。动态间接alias/ownership、任意guard、continuous多时钟状态等仍需通用合同和明确诊断，不能通过新接口名字宣布支持。

## 5. 历史设备差距与证据保留

默认4K f16 FA的旧三批msprof设备task中位数为生成代码605.780µs、ACLNN512.152µs，慢18.3%。独立手写AscendC为1191.220µs。这是对应旧源码/政策/设备的历史结果，不是编译器重构加速结果。

原始摘要仍在[performance.json](../../runs/refactor/cv-order/performance.json)及[op指标目录](../../runs/refactor/cv-order/op-metrics)。历史文档原文见外层归档的 `compiler/docs/attention-performance.md`。此前清理已删除库、部分profiler原始数据和旧候选目录，需重建才能再次运行；不恢复旧优化任务。

未来比较必须直接看生成ASC、手写AscendC、ops-transformer/ACLNN和硬件条件下界，详见[验收](validation.md)。当前M0完成、M1实施中，M2–M6未完成。
