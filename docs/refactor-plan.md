# 编译流程重构实施计划

2026-10-16，基线 `fdff8c5`。本计划替换此前“继续 GM→L1、构建 BQ256、设备复测”的
执行顺序。**算子优化已停止；当前交付为源码审查和详细方案，没有修改编译器实现。**
技术设计见[design.md](design.md)，必须遵守的变换合同见
[compilation-invariants.md](compilation-invariants.md)，事实见[状态](refactor-progress.md)。

## 1. 要交付的结果

同一公开 DSL 经一条结构化编译路径生成 ASC。普通编译一次建立物理访问和依赖，
按 region 合同构造同步，联合分配事件，保持不变量地物化；不在 child、parent、query、
allocation、final 各自重证整个程序。

同时做到：runtime 长度不参与分析展开；静态 fragment/guard/period 的成本可解释；
生成代码保持可读；保持既有排步、早 release、buffer 复用和 CV 重叠能力。
编译快、正确、机器代码无回退分别验收，不能用削减能力换编译数字。

本次不重新设计算子算法，不修改 FA tile、Vector 政策、GM/L1/UB 存储政策，也不进行
算子参数搜索。已有 examples、基准和失败记录保留，BQ256 仍是未完成编译的暂停候选。

## 2. 分阶段实施及退出条件

### R0：固定基线和成本口径

交付：当前流程调用表、唯一分析计数、原始/派生计数定义、pass 读写与失效表。
本轮文档完成了源码审查部分；后续代码仅补缺失的轻量统计，不新增全量 profile 循环。

必须分别记录：T/A、访问相交 I、H、域/条件 K、unique queries、构造次数、重复复用、
origin 位长、实际 transfers/comparisons、静态窗口副本、资源约束、输出大小和各阶段时间。
RSS 用真实进程峰值，不能把各 flow 的峰值相加当作总峰值。CPU profile 时间与无探针
冷编译分开；N、静态 child count、tile 容量是三条不同规模轴。

退出：每个超过预算的阶段能说明哪个输入维度放大了哪个输出，不再只返回“case超限”。
不为补齐基线重跑五遍 141 秒 FA；既有数据注明历史来源，必要的新 baseline 每例一次。

### R1：分析所有权和重复事实的显式交付

主要位置：`compilation.py`、`scope_analysis.py`、`scope_sync.py`、
`scope_query_resources.py`。引入少量具名、私有结果和 revision/变更范围；不建大型 pass 框架。

先消除一个可以独立验证的明确重复：同一 final snapshot 的 child resources、parent
pools、counter lifetime 结果由 invocation 交给 query，query 只处理跨 query 新义务。
不缓存整个 `verify_plan`；不把历史 JSON、planner bool 或可变 Module 的身份当证明。

退出：同快照每项合同只建立一次；修改 cut/guard/ID/clock 的反例能使相应事实失效；
正常结果和显式深度检查一致。当前 final 仍保留，暂不以移走它宣称编译问题解决。

### R2：一次物理访问分析与紧凑 region transfer

主要位置：`lowering.py`、`scope_access.py`、`region_summary.py`、`region_sequence.py`、
`scope_boundary_summary.py`、`scope_query_summary.py`。

- 源值读取绑定、物理 overlap 和后续同步共用一次 effects/definition 分析。
- 按精确 root/range/guard 维护 frontier；同端点义务共享求解，footprint 保持独立。
- 规则 strided/fractal descriptors 不提前拆成逐字节状态；固定重复采用可组合 transfer。
- query、parent、child 通过同一摘要组合，保留 native 内部 typed cuts 和零次 identity。
- 只绑定父层真正需要的端点，保留动态参数、slot 相位、历史 read bindings。

退出：Vector 双缓冲、GEMM、原地状态、嵌套 child 反例的义务与独立小型字节模型一致。
访问构造不在 planning/final 内部反复执行；固定 count 从小数到大数不线性复制访问历史，
除非源码显式 unroll 或硬件原语必须静态分段，并对此明确计数。

若 descriptor 压缩改变访问孔洞、丢失部分 reader，或把 child 退化成整体 done，阶段失败。

### R3：有类型的完成转移与构造式同步

主要位置：`native_order.py`、`loop_sync.py`、`scope_sync.py`、
`child_completion.py`、`scope_*completion.py`。

先做三个小例：早 release 的 Vector；L1/L0/L0C 不同寿命的 Cube；两 AIV 的 CV
交接。用同一套 Issue/QueueBegin/RetireCapture/ImplicitCompletion 规则，再组合父子循环。

完成前沿保留跨通知传递的事实；对已有保证直接引用，对缺口构造确定协议。正确构造的
输出本身建立完成关系，不调用全图搜索确认自己。小型偏序确实需要求解时，只针对唯一
需求一次处理，记录实际状态，不把旧全源传播换个名字藏进去。

循环用启动/一步保持/退出合同，条件 guard 保留孔洞。进展和资源是独立义务；不能因为
内存关系成立就默认完成。同步捕获只扩大到真实所需里程碑，禁止多加等待换事件容量。

退出：相同物理政策下，同步安全与“不能多等待”的反例均通过；region 组合不引入
child/query drain。原来已支持的父 child 插入 cuts、跨 query release 都有表达。
尚未接通的模式明确报 unsupported，不暗中退回旧求解器，也不把拒绝称重构成功。

### R4：统一资源寿命与分配，先合同后 ID

主要位置：`scope_events.py`、`scope_event_pools.py`、`scope_event_borrow.py`、
`scope_query_resources.py`、`scope_query_planning.py`、`loop_sync.py`。

逻辑通知固定后一次收集整个 scope 的寿命/进展约束。width 和 ID 是输出，不再先让
child 独占 ID、父层扩宽、重新构造内存图，再为 query 复制整图。

保留普通共享、当前受限 parent/child 借用、配置保留、FIRST/LAST anchors、跨 query
整池配对和 CV counter 上限。分配失败只在资源约束内有界处理，不重跑整个 compilation。

退出：ID-only 置换不触发 memory analysis；width/共享变更只使实际资源相关项失效。
安全置换、真实碰撞、guard 孔洞、跨 query 未消费通知和 CV backlog 均有反例。
已接受 BQ128 不能因新分配器把 ID 独占而被拒绝；未达此条件不能切换默认路径。

### R5：保持不变量的物化及正常/深度检查分离

主要位置：`sync.py::_finish`、`Compilation`、`codegen.py`、各层 codegen。
materializer 按已固定端点/guard/ID 映射生成 IR，不再选择协议。先用破坏/差分回归证明
实现遵守该映射，再让普通编译直接消费结果，移出重复全图检查。

外部手写/导入可变 physical IR 仍从实际 IR 建立事实；debug/reference 显式执行独立
终检。不能新增通用 `verify=False` 绕过入口；不能将 planner report 直接改名成证书。

退出：normal 和 deep 的输出语义一致；普通路径没有独立全图重建；外部修改 IR 无法
复用旧分析。所需阶段全接通才切换默认，开发比较使用显式选择，不设静默 fallback。

### R6：收敛前端/标量流水与删除重复生产路径

前端语法保留，内部 `frontend` 属性分流、`_legacy_work`、有限 source walker 在相关
机器反例迁移后删除。physical Task IR、专家排步和独立物理验证入口继续存在，它们
不等于旧 `cube.task/vector.task` API。

统一 scope collection、clock/control admission 和 source→native 读取映射。
整数 canonicalize/CSE 显式组织，区分 proof Expr 数学整数和 ASC i64 语义；event 常量、
地址公共子表达式、原始轴合法性、uniform 表达式 DAG 均保留。emitter 只打印确定结构。

退出：公开 DSL 一个入口、运行时循环保留、异步 effects 不被通用优化删除/移动，
不再因一种 call 形状启动一套独立完整求解链。删除对应旧 production 分支；通用数学
工具和独立 test oracle 保留，不为删文件减少语义覆盖。

## 3. 现有模块的去留

| 当前职责 | 迁移后 |
|---|---|
| `native_order.py` | 小型硬件转移核心，补齐合同来源与反例，不复制新版本 |
| `model/loop_model/scope_access` 重复访问提取 | 唯一 physical effects/依赖分析与 region 绑定 |
| `region_summary/region_sequence` | 紧凑 transfer 基础，保留历史读取绑定和精确范围 |
| `ParentCompletion` 多次全 child 投影 | 已建立的 child transfer + 按需 typed cuts |
| guarded/anchored/query 多层重建 | 一种域/时钟条款组合；独立通用求解仅用于确有需要的局部关系或reference |
| parent/query/case 不同 event 分配回路 | 一个逻辑通知资源模型，配置/child/parent/query 统一约束 |
| normal `_finish` 全量重建 | 受控物化合同；独立全图检查转为显式入口 |
| 按 scoped/sequence/symbolic/mixed 结构分派的多套生产分析链 | 统一结构化流程；不同支持域明确合法化，不增加算子特判 |
| 旧 source/finite 编译兼容路径 | 迁完反例后删除；有限执行模型留在 tests |

按职责替换而非先搬目录。每次合入注明已替换入口和仍存在的旧调用；不允许永久两套
生产算法都保留，正常编译还同时跑两遍“作比较”。

## 4. 复杂度与编译性能验收

首先验收工作量，再做冷计时：

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
max≤2秒；现有 dense FA p50≤5秒、max≤10秒。是目标，当前未达到。import、文件读取、
前端、全部分析、标量化简、snapshot、CANN 各自记录；总量不能靠移动阶段降低。

当前 `benchmarks.compile` worker/子进程上限为60秒，不能用于收集历史141秒FA。
重构计时工具须明确总期限和子进程清理；完整文件构建仍遵守单次300秒硬上限，超过
立即终止并报告，不自动再跑下一候选。上限是保险线，绝不是合格性能标准。

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

例如以下是已有的定向入口，**本轮没有执行**；未来按改动范围选择，不一起机械执行：

```bash
.venv/bin/python -m pytest -q compiler/tests/sync/test_direct_synthesis.py
.venv/bin/python -m pytest -q compiler/tests/test_scope_events.py compiler/tests/test_scope_event_pools.py
.venv/bin/python -m pytest -q compiler/tests/sync/test_plan_validation.py
```

最终冷计时使用既有工具或明确修订后的工具；新入口/选项实现前不写伪复现命令。

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
本轮三个 Agent 只做了独立源码审查，未运行测试或修改代码。

每阶段交付：小补丁、已保持/新增的不变量、失效范围、复杂度计数、相关回归、生成前后
对照和未解决项。只提交 compiler 产品文件，不纳入外层环境、实验记录，不打包或推送。

以下情况停止进入下一阶段：当前支持能力丢失；等待范围扩大；借用/ownership 被简化
掉；normal 偷调旧全图求解；复杂度下降仅因缓存或未计时阶段；硬超时。先报告具体反例
和规模，再修正设计，不能通过继续调算子掩盖编译器问题。
