# AscendFlow 编译器重构方案：统一语义域与 SMT 查询

日期：2026-10-19。审计基线：`fb39c663ba5efb864ec6c73a4717f8184b75a834`。
本文是待实施方案；源码审计与隔离原型已经完成，Z3 尚未接入产品编译链。
后续实际进展、提交与验收证据写入 `smt-refactor-progress.md`，不得将本文目标当作已实现能力。

## 1. 结论与重构目标

当前实现有明确的设计欠账：同一整数表达式、作用域和代次，在范围、地址、
存储、事件域及最终优化中使用多套部分重叠的解释和手工绑定；有些关键前提
依靠调用者记住，并未在接口边界落实。重复补局部规则会扩大回归面。

Z3 可以显著收敛整数蕴含、条件覆盖、互斥与部分别名推理。重构应真正删除
被替代的证明代码，形成一个查询入口，而非在旧推理器后面再叠一个兜底求解器。
硬件 effects、源值与存储复用、异步完成和无限循环组合仍由编译器明确建模。
求解器无法恢复被错误去重丢掉的 writer，也无法修正错误的硬件完成前提。

交付目标：

1. 一套原始整数与执行域语义；非平凡数学查询统一服务，保留廉价语法化简。
2. 每项事实有明确输入、建立者、适用域、消费者和失效条件；未变事实直接复用。
3. 初始化、源值、物理 ownership、ready、reader retirement、事件消费分别建立。
4. 循环与分支使用可组合接口，父层不恢复子层内部执行历史。
5. 保持唯一 DSL/source→native→自动同步→最终 SSA→ASC→CANN 编译链。
6. 恢复 FA 回归，保留 KDA、GEMM 与其他真实 DSL 算子的精度和流水性能。
7. 用实际删除项、查询与构建成本、独立语义反例验收简化，不以换目录或增加包装层验收。

不更换 xDSL，不引入 MLIR 工程；不建设 legacy 路径、兼容别名或可伪造的 verified 标记。
本方案中的职责名称不要求一一新增文件，不建设通用大型 proof 框架。

## 2. 已核实的现状与问题

### 2.1 当前完整编译链

`Compilation` 调用 `synchronize_structure`：

```text
DSL / source ProgramOp
  lower_native_structure        原式检查、source lowering、原生指令与访问
  collect_native_memory         canonical control、地址、effects
  admit_memory_bounds           原轴、字节范围、对齐、片上容量
  admit_structured_memory       初始化、逻辑 Tile、读写与配置合同、host 整数范围
  bind_structured_dispatch      actor 的真实 stage / issue order
  build_structured_dependencies 可能冲突顺序、最后 reader、局部存储 packet
  generate_notifications
    notification_binding        capture/acquire、quorum、前缀合并
    EventDomains                activation、坐标、配对与存在域
    queue_order                 physical step 与 child may-block 接口
    cell_channels / HandbackRule 消费与事件复用
  allocate_structured_events    bank、ID、预留、新共享约束
  materialize_notifications
  lower_execution               终端单时钟 arith/scf、两个 AIV 共用代码
  simplify_scalars / simplify_conditions
  ASC emitter / CANN
```

这条主链值得保留。不能将已有必要检查笼统认作重复求解，也不能先删除检查再补合同。

### 2.2 规模和职责分散

基线 `src/` 为 85 个 Python 模块、27,030 行；analysis 为 10,514 行。
下列整数、条件和对应 IR 重写相关文件合计 4,726 行：

| 文件 | 行数 | 主要职责 |
|---|---:|---|
| `scalar.py` | 559 | Expr、facts、原式求值与基本推导 |
| `indexing.py` | 742 | 多项式、整除与联合范围、原轴和原式检查 |
| `analysis/integer_facts.py` | 398 | 前提下的 bounds、prove、ordered definedness |
| `analysis/modular.py` | 424 | 有界同余多项式和 residue 谓词 |
| `analysis/periodic.py` | 133 | 周期与相位分析 |
| `analysis/guard_cover.py` | 293 | 整数/同余分支构造覆盖证明 |
| `analysis/structured_host.py` | 155 | host uniform 范围 |
| `passes/simplify_scalars.py` | 905 | 最终 SSA 的区间、关系、化简、外提 |
| `passes/simplify_conditions.py` | 1117 | 区间并集、差分关系、余数、条件合并与重写 |

4,726 行不是可删除行数：里面包含必须保留的语法操作和代码生成。
优先替换查询主体，再合并共享域与表达式设施。

### 2.3 具体欠账和证据

| 问题 | 已核实证据 | 重构要求 |
|---|---|---|
| scope 身份不完整 | 初始化去重曾使用 `template.frame.parent`，合并可独立为空的兄弟循环；`cbc08d8` 修为实际 frame | 统一 activation/occurrence 身份，禁止各模块自行弱化作用域 |
| 数学中间式泄漏到可执行整数构造 | `_joint_affine_interval` 将有理系数 bias 送入 `_to_expr`，默认 FA 编译失败 | 求解表示和可执行表达式分离；查询不依赖重建多项式 |
| 相同失败在多处各自修补 | `integer_facts`、`region_coordinates`、`structured_access`、`event_windows` 按同一异常字符串回退原式 | 统一结构化查询结果；移除按报错文本控制语义的路径 |
| 参数/代次性质依赖名字 | 多处以 `name.startswith("$")` 区分 uniform 与执行坐标 | 使用有身份及种类的 symbol；名字只用于诊断 |
| 坐标转换规则多处维护 | RegionCoordinates、notification binding、EventDomains、queue order、terminal lowering 分别处理选择与代次平移 | 共用显式 occurrence mapping；一次平移不能重复应用 |
| 最终 IR 与旧事实混在同一容器 | terminal lowering 原地改变 module，返回对象仍暴露原 logical control facts | 交接清楚区分 logical facts 与最终 executable plan，提供稳定身份映射 |
| 事件分配同时尝试共享证明 | allocator 对既有成员反复调用 handback/覆盖推理 | 生命周期兼容关系一次建立，分配消费关系，仅为新增共享建立新义务 |
| 原生定义分散 | recipes、native contract、packet、emitter 各自识别 opcode/geometry | 收敛公共原语描述；保持独立硬件合同核验 |
| 输入假设容易误用 | kernel 曾把 full clear 后 partial store 声明为 write_disjoint | 区分输入假设和已证事实；开发期用实际访问校验可检查的声明 |
| 验收边界不完整 | 已知 FA 编译失败长期被目标小组排除 | 公共示例回归必须成为集成门槛，不能以排除后通过代替总体资格 |

FA 回归已经定位到 `e77a6b2`，2026-10-18 17:28:28 UTC，KDA decode 支持改动。
父提交 `a9fa8b8` 的同一 FA 准入通过；该提交失败；仅在诊断进程中把
`ratio == 0` 回退为 `ratio <= 0` 可恢复准入。这只定位根因，不是产品修复方案：
重构必须保留合法负方向范围推理，不得通过整体禁用新增能力解决回归。

## 3. 隔离 SMT 原型的证据与限度

实验位于工作区外层 `runs/compiler-architecture-audit/`，未修改产品依赖或编译实现。
使用官方 PyPI 的 `z3-solver==4.15.4.0` AArch64 wheel，适配容器 glibc 2.35。
该目录不是产品运行或构建依赖。

对 Gram、output、scan、gram_cross 的真实 Python/IR/ASC 编译进行只读调用抽取：

| 项目 | 结果 |
|---|---:|
| prove 调用次数，含已有缓存命中 | 122,582 |
| 按 owner、原式、完整前提区分的不同蕴含查询 | 2,164 |
| Z3 证明成立 / 给出反例模型 | 1,144 / 1,020 |
| 旧推理器证明成立 / 未证明 | 1,141 / 1,023 |
| 旧 true、SMT sat 的不一致 | 0 |
| 单次设定 50 ms 时限下的 unknown | 0 |
| 不同蕴含查询总耗时 | 约 1.64 s |
| 查询中位数 / p95 / 最大值 | 约 0.68 / 1.30 / 2.31 ms |
| 条件覆盖调用 / 不同覆盖查询 | 7,874 / 50 |
| SMT 覆盖 / 不覆盖 | 22 / 28，与原实现一致 |
| 50 项覆盖查询总耗时 | 约 0.052 s |

额外成立的三项蕴含来自可判定不可达的前提组合；不能把它们描述为新增三种算子支持。
已有缓存命中不是重新执行完整证明，因此不能将 122,582 次都算作昂贵重复求解。
但 7,874 次覆盖调用只有 50 项不同输入，说明复杂覆盖构造和分配调用有明显收敛空间。

FA 当前失败的实际表达式是 `8 * (Sq // 256) - group`，其原始输入合同含
`Sq >= 1`、`Sq % 256 == 0`，实际 group 范围为 0..19。
Z3 可直接证明其范围及 `8*(Sq//256) == Sq//32`，不需要把整数整除展开为有理系数。
原型还检查了短路 LAST、原式溢出与数学恒等式的区别、无符号余数位掩码、
重叠 clear/store、issue order 与 reader retirement 的区别。

这些结果仅说明局部数学查询替换可行。尚未实现完整 solver 编译链、生产
definedness/overflow 编码、SMT 语义验证器、端到端性能或新的 CANN/设备资格。
样本不涵盖所有算子与参数；Z3 的 sat/unsat 也依赖编码和硬件合同正确。

## 4. 目标语义边界

### 4.1 统一 symbol、domain 与 occurrence

复用现有 source/physical IR，不堆叠新的完整 dialect。
以少量不可变记录表达下列信息，具体文件可按源码收敛：

* symbol 的稳定身份和种类：runtime parameter、group、participant、loop ordinal、
  metadata capture、carry。metadata 必须关联实际 load 与祖先代次；不得因同名合并不同读取。
* lexical domain：父域、当前新增的有序 guard、loop bound、input contract。
  新前提生成新 context，兄弟域隔离。
* occurrence：native/边界身份、actor、实际祖先 loop bindings、activation。
* coordinate map：logical ordinal、selected FIRST/LAST、translated generation、physical step
  之间的显式转换。替换同时用于地址、guard、slot、metadata 和捕获值。

source 原式 DAG 保留求值顺序和位宽；求解器使用数学 Int/Bool 或适当的 BitVec
视图，不将求解器规范形反向当作原始程序。

### 4.2 一套约束查询服务

对外只提供语义明确的查询，如 implies、satisfiable、equivalent、disjoint、covers。
结果区分 proved、counterexample、unknown，携带查询类别、context 身份、来源与耗时。
禁止把“未证明”等同于“已否定”，禁止用布尔 false 混淆超时和具体反例。

基本编码：

```text
implies(C, P):  UNSAT(C ∧ ¬P)
equivalent(C, A, B): UNSAT(C ∧ (A ≠ B))
covers(C, alternatives): UNSAT(C ∧ ¬OR(alternatives))
disjoint(C, A, B): 不存在合法实例和 row/byte 位置使物理访问相交
```

编码器按原 DAG 共享节点；上下文按 lexical scope 共享增量约束。缓存必须包含
完整前提、数学/机器语义、位宽、变量身份和查询类型。未知、异常不形成安全事实。
超时从算子剩余总期限派生，并有受控的单查询上限与独立统计。

廉价常量折叠、相同节点匹配和用于输出的基础语法化简保留。
非平凡正确性查询统一走此服务；不长期维持另一套启发式证明器作为第二正常路径。
迁移期对照只用于开发验证，随后删除被替代代码。

### 4.3 definedness、数学关系与机器整数分别处理

对无符号除法/余数，原式要求 numerator >= 0、divisor > 0。
逐层验证原始 DAG，不能用恒等式消掉非法子式。
短路表达式的定义性是 `Def(A) ∧ (A ⇒ Def(B))`，不能借后面的条件证明前面的求值合法。
`N > 0 && (N-1)%2 == 0` 与逆序 guard 的定义性不同。

数学 Int 不模拟溢出；SSA/C++ 索引的位宽和 signed/unsigned 语义必须单独检查。
`(x+1)-1 == x` 的数学证明不授权在 x=INT64_MAX 时求值原始加法。
BitVec 用于真实位向量语义和位掩码等精确恒等式；不能将 C++ signed 溢出当成合法回绕。

原始输入的合法整数求值前提可以消费并记录；新生成或被外提的表达式必须在
新的执行域重新建立定义性与范围，不能把该表达式自身的合法性当作证明前提。
SMT 对除零等操作的数学总函数行为不能被误用为源语言合法性。

### 4.4 存储与值保持三个独立职责

* must-initialized：每个实际读取字节在全部可达路径上有合法定义。
* source value：dispatch 变化后仍读取原来的逻辑值；部分更新和 RMW 保留真实版本。
* physical lifetime：全部旧 reader 退休后才能覆盖物理槽；控制退出不表示异步完成。

三者消费同一真实访问与 occurrence，不彼此替代。
先按 allocation/actor/group/root 排除不可能别名，再对有必要的区域使用 SMT 查询。
行和列可用受界整数变量表达相交，不展开每个 byte 或 runtime iteration。
unknown 保留可能冲突顺序；不能当作 disjoint 或完整初始化。

`write_disjoint` 等显式输入假设必须与 proved facts 区分。开发验证检查可表达的
真实重复写反例；正常路径不要求重新证明任意数据相关算法的全局唯一性。
full clear 后 partial update 不能声明为单次独立写。非法输入不送设备。

### 4.5 循环与同步采用组合构造

保留目前 physical-step、稳定 cell credit 和真实 handback 的方向，重构其共享
语义接口；不恢复全程序 Path/搜索树/后置删除同步再求解的路线。

子区域明确导出：输入需求、可能访问、出口 pending readers/writes、实际
queue/controller 边界及可能阻塞传递。数值定义与完成保证分别保留。
N=0 和未取分支透传入口状态；N=1、短 pipeline 和 FIRST/LAST 重合使用同一规则。
父层组合 child 接口，不展开 runtime N，不重新访问 child 内部证明历史。

SMT 用于 activation 互斥、条件覆盖、坐标等价和局部次序前提。
有限零 step 图的 DAG 检查继续使用普通图算法；无限循环安全需要组合规则或归纳，
有限 BMC 没有反例不构成任意 N 的证明。

Set 只能捕获 source queue 的已发射完成前缀；本地 Wait、Scalar Wait 和 cross
controller Wait 的阻塞语义不同。ready、release、event consume 分别有实际端点。
两个 AIV 仍共用一份代码，以 runtime SID 区分地址；各自的 UB、最后读取和 quorum 贡献独立。

### 4.6 生命周期与事件 ID 分配分开

已建立的 cell 生命周期、配对和实际 consume→next Set 关系形成通道事实。
完全相同前提和成员的共享兼容关系只建立一次；分配器消费兼容结果，不随每个候选 ID
重复构造相同 guard cover 和 handback。新增共享必须建立新增约束。

普通确定性分配可保留；若需要，可对已完成合法性分析的有限冲突图使用 Z3/CP-SAT
选择 ID。它只解决有限资源放置，不能创造缺失的消费回传。
local bank 为 actor/source/destination，每 bank 8 个 ID；配置与 drain 预留计入。
mode-2 只有 0..15 base ID，双方 AIV quorum 与实际 counter backlog 合同必须保留。
连续 ID ring 是当前输出政策，不能误当硬件额外容量或用 overflow ID 绕过限制。

### 4.7 最终优化与后端

同步确定后消费 logical plan，产出仅含最终 SSA、地址、已审计几何和已确定通知的
executable plan。旧 logical control 事实不再通过同一对象伪装成最终 IR 的事实。
使用稳定 native ID 和受控映射交接，不产生第二条可执行编译路径。

标量与条件优化共用数学查询与实际支配/执行域；保留 SSA CSE、DCE、LICM 和紧凑
guard 输出。SMT 不直接输出 runtime 调度或任意量词公式；代码构造与证明查询分开。
内存、native、参与者、配置和异步 effects 保持，相关变化才使事实失效。

原生公共字段向小型 NativeSpec 收敛：opcode、dtype/bank 几何、effects、原生
packet 状态、完成合同与格式化入口。CANN 指令约束仍需可信硬件核验，不能用
同一错误 spec 的两个读取者互相充当独立 oracle。前端算法与 tiling 政策继续放 examples。
build、deadline 和 CANN 包装层已有明确职责，保持简单，避免顺手重写。

## 5. Z3、其他工具及代码删除边界

| 部分 | 建议 | 可删除/合并的内容 | 必须保留 |
|---|---|---|---|
| IntegerFacts | 改为统一 SMT 查询适配 | 自制关系证明、重复 bounds 决策、分散异常回退 | 原始 DAG、context、定义性、预算与诊断 |
| guard_cover / modular | 用 Bool/Int 蕴含替换覆盖搜索 | 手写 trichotomy/residue 证明树及仅服务证明的模多项式 | 硬件真实 mask 编码、实际 runtime 谓词构造 |
| indexing / host ranges | 用完整整数原式与查询 | 为证明而展开整除的有理系数重建、重复局部范围证明 | 原轴合同、地址布局、原式合法性、廉价语法构造 |
| scalar / condition passes | 共用域和 query service | 各自的关系闭包、重复 bounds/互斥证明 | SSA 重写、支配、输出成本与 effects 屏障 |
| event windows / families | 缩小为必要的周期/成员构造 | 纯决策性质的参数分裂与重复证明 | 稀疏事件的 rank/count/FIRST/LAST 构造与归纳前提 |
| access/alias/initialization | 共用真实区域与 SMT 查询 | 重复区域关系求解与手工 scope 弱化 | 数据流、版本、最后 reader、保守冲突顺序 |
| queue order / cell credit | 保留组合算法并收紧接口 | 重复坐标解释和适用前提求解 | 硬件队列模型、无死锁组合、消费生命周期 |
| ID allocation | 先缓存合法兼容关系，必要时有限 CSP | 候选 ID 内重复 handback/覆盖工作 | bank、预留、quorum、真实消费 |
| frontend / native lowering / ASC | 有边界地保留与收敛 | 散落的公共 opcode 属性与冗余转换 | DSL 语义、指令选择、真实 footprint、实际代码生成 |

Z3 对当前 Int/Bool 查询与少量 BitVec 性质合适。ISL 适合仿射集合与常量除数的
Presburger 投影，但动态 uniform divisor、变量乘积和机器位向量不都落在其范围。
SymPy 可做表达式变换，不能独自承担带前提与机器定义性的安全查询。
CP-SAT 可用于有限 ID/调度放置，不替代无界整数和异步生命周期建模。
本次不同时引入多个求解器；先用一个小的 Z3 接口收敛语义。

符号乘积、变量除数与量词可能使 SMT 求解困难。必须保持查询局部、量词尽量消去、
子区域按合同组合，并统计 unknown。必要准入无法证明就具体拒绝；可选优化无法证明
则保留原式。不能用无限调大预算、屏障或改变调度换取通过。

## 6. 实施顺序、交付物与退出条件

| 阶段 | 实际工作 | 本阶段验收 |
|---|---|---|
| P0 基线与反例 | 冻结 HEAD、现有未提交内容；记录 FA 错误、现有实例和性能身份；建立进展台账 | 本次 FA 引入点、兄弟循环、clear/store、短路与溢出都有明确反例和期望；不排除已知失败冒充全量通过 |
| P1 统一整数服务 | 正式依赖、原 DAG→Z3、完整 context、Int/BV/定义性、结构化结果和诊断；替换 IntegerFacts 查询并修复 FA | 默认 FA 完整 DSL→ASC 编译恢复；正/负方向和动态 divisor 均保留；相关反例、公共例子 IR 回归通过；删除被替代推理主体 |
| P2 统一域和代次 | symbol/occurrence 身份、共享坐标映射、空域/分支接口；合并 binder/domain/order 的重复解释 | 0/1/短循环、兄弟/嵌套、metadata、FIRST/LAST 和双 AIV 的独立 trace/字节模型通过；无裸名字决定语义 |
| P3 存储和值合同 | 初始化、源值、ownership 分责；区域关系与输入假设接口；收敛 NativeSpec 公共字段 | 部分写、mask/RMW、多 reader、跨 loop 覆写、共享 L0C、别名与重叠声明的正反例通过；原生几何/packet 微测试保留 |
| P4 同步和资源 | 循环组合消费共享域；SMT 替换 cover 决策；共享兼容关系缓存，有限 ID 分配与生命周期分离 | 原有同步反例、事件错误破坏测试、bank/quorum/reentry 通过；同步数量和 Scalar 输出有解释；不扩大硬件 ID 容量 |
| P5 终端收敛 | executable plan 边界；scalar/condition 查询统一；移除旧手工证明与异常字符串回退；更新文档 | 最终实际 IR 的独立检查通过；只有一条正常路径；提交前列出删除、保留及新增代码和查询统计 |
| P6 集成资格 | 全量回归、公开 DSL 示例、ASC/CANN、必要设备数值与性能验证 | 无排除的已知 FA 失败；当前真实产物具备清楚的构建/设备资格；编译成本与性能报告可复现 |

阶段按依赖顺序推进，紧密关联的小项可在同一提交完成。每个提交应有具体语义
改动和对应验收，不为保留旧 helper 或旧文件结构写镜像测试。
中间迁移不能关闭尚无替代合同的现有检查；替代规则和回归接通后再删除旧实现。
如果尚有未完成阶段、失败或缺少设备资格，进展与最终结果必须明确标出。

## 7. 验证矩阵和性能口径

### 7.1 必须保留与扩展的反例

* 原式：unsigned 非法子式被代数抵消、除零、guard 逆序、signed 溢出与合法 BitVec 恒等式。
* 域：独立空兄弟循环、嵌套 count 依赖、0/1/短 pipeline、稀疏 guard、未取分支透传。
* 内存：原轴越界、native padding/stride、部分写和 RMW、full clear 后 partial store、不同 view 共用 root。
* 异步：最后 MTE1 reader 早释放、MAD/FIX 共享 L0C、MTE3 仍读 UB 时循环退出、全部 pending readers。
* 事件：删 Set/Wait、错误 slot/guard、二值 flag 未消费复用、跨调用 reentry、配置与 drain 预留。
* 参与者：两个 AIV 共用代码、私有 UB/bank、缺一个 AIV contribution 的反例、GM 真实可见性。

独立有限模型读取实际最终 IR，并与独立 pre-sync native 快照比较实例、地址、读版本、
事件消费和 queue/controller 因果。新增 SMT 验证不能共享生产构图与全部假设后声称独立。
有限验证只覆盖指定实例；无限循环的普遍合同仍通过组合规则及适用前提建立。
source→native 的数值与硬件语义另由 CPU oracle、原生检查和必要设备实测覆盖。

### 7.2 构建与设备纪律

每个算子的 Python 构造、分析、ASC、CANN 及子进程属于同一次 300 秒硬期限，
超时终止所属进程组并停止该候选。较长的代理总任务时限不替代此限制。
先纯 Python/IR，再 ASC、CANN；未通过前序检查的候选不送设备。
运行前检查所有卡的健康、进程、显存和利用率；显式物理卡→逻辑 0。
设备错误停止，不 reset、不换卡绕过；不使用 `/tmp` 锁。

至少恢复 16 个公开示例的真实编译入口；重点覆盖 FA、GEMM、KDA decode/prefill、
softmax、RMSNorm 的原有语义与当前合法参数范围。FA FP32 求和次序按现有许可，
KDA 状态递推、输出 dtype 和已声明 rounding 边界保持。
不得用手写 ASC、历史产物、旧路径或预期编译失败 fixture 代替当前 DSL 支持。

### 7.3 编译与运行性能

分别报告源码构造、分析各阶段、求解器翻译/调用/缓存、终端优化、输出、CANN、
进程总耗时及 RSS。不得通过移动计时边界、跨编译缓存或扩大旧预算制造提速。
统计 distinct query、调用次数、cache hit、sat/unsat/unknown、耗时分布和最大上下文。
default FA 的 source→ASC 延续五次冷进程 p50≤5s、max≤10s 目标，同时报告
GEMM/scale/softmax/RMSNorm 的绝对值；未达标明确记录。

KDA 主要回归配置为 B=1、S=4096、H=32、K=V=128、chunk64，另含 S=4077 尾块。
基线 Gram 1240.62 us、prefill kernel sum 3699.80 us，来自
`build/kda/op-prefill-gram-aiv-paired-b1-s4096-basic/metrics.json`。
比较保持数学、dtype、输入、设备与频率口径，采用 `msprof op BasicInfo`、默认
kernel 重放；PipeUtilization 仅解释各单元活动，不等于 FLOPS 利用率。
原产物链接及 SHA、配置与输入在 `build/kda/prefill-gram-aiv-paired-b1-s4096/optimization-summary.json`。
重构后必须使用新代码实际生成的产物；复用库只能明确标为未重编译部分，不能作为本树资格。

## 8. 最终交付清单

1. 唯一编译路径上的实际重构代码、必要正式依赖及已删除旧实现清单。
2. 已修复的 FA 回归及范围/循环组合/同步资源反例，完整测试结果与失败解释。
3. 当前 source/IR/ASC/CANN 的身份、命令、构建时间与必要设备证据。
4. Z3 的编码边界、预算、unknown 政策、查询统计和可信前提说明。
5. 更新后的 source-layout、compilation-invariants、generation-rules、validation，反映实际代码。
6. `smt-refactor-progress.md` 逐阶段记录提交和“已实现/已编译/已设备验证/方案/明确拒绝”。

审计原始材料：`/workspace/xdsl-910b/runs/compiler-architecture-audit/` 中的
`captured-queries.json`、`smt-results.json`、`capture.py`、`smt_probe.py` 与 watchdog；
FA 相邻提交定位材料为 `/workspace/xdsl-910b/runs/fa-regression-audit/`。
这些是诊断材料，不是新编译器依赖。

参考：[Z3 Arithmetic](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)、
[Bitvectors](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/)、
[Programming Z3](https://z3prover.github.io/papers/programmingz3.html)。
