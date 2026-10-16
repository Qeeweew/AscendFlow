# 编译流程重构：控制分析规模，按不变量组合变换

2026-10-16。历史源码审查基线 `fdff8c55b0cddc51331c637ffecc75ce0e34401f`，
本轮实施基线 `216b7117c3bd5c1d9e2f91ac182115bd5e7004e3`。
本文区分已落地的变换合同与尚未完成的边界 transfer 目标；**不宣称完整形式化证明**。
实际状态和本轮冷编译测量见
[状态页](refactor-progress.md)，实施顺序见[重构计划](refactor-plan.md)，
pass 合同见[编译不变量](compilation-invariants.md)。

用户已停止算子优化。当前已实现私有物理事实交付、结构化正常路径的 normal/deep
分离、资源结果复用、增量排名、条件前沿与编译内数学值共享、合格固定摘要和源工作
收集统一。一般 guard/region/query 求解仍在，legacy finite physical 入口仍保留。
旧 BQ256/GM→L1 算子优化计划停止；本轮编译器测量不代替设备验收。

## 1. 先明确问题，而不是继续补热点

历史基线把原生语义、依赖构造、完成关系、事件寿命、分配后检查反复转成图查询。
本轮已移出结构化正常路径中同一事实的独立终检与若干资源重证；单次分析的物理分片、
条件域、里程碑、父子投影和传播仍会放大状态。Python 对象和大整数会放大成本，
但换语言不能消除这些工作量。

历史默认 FA 源码到 ASC 构建为 141.225 秒：

| 阶段 | 秒 | 口径 |
|---|---:|---|
| frontend / input clone | 0.135 / 0.011 | AST 与编译器私有副本 |
| lowering | 1.625 | 原生物理 IR |
| planning | 79.685 | 含规划内部分析/检查，并非纯插入指令 |
| final check | 58.583 | 从同步后实际 IR 重建分析 |
| emission / snapshot | 0.753 / 0.433 | ASC、IR 与报告 |

来源：外层 `runs/refactor/cv-order/build/compilation.json`。这是历史单次构建，不是
当前 HEAD 的五次冷编译。规划和终检合计约 97.9%；**即使全部移走终检，仍约 82.6 秒**。
因此只删 final verifier、换 C++、加 cache 或放宽预算，都不是完整解法。

该历史默认 FA 有 76 个 native 模板、12 个 call；Sq/Sk 是 runtime 参数，没有按长度
枚举。另一个旧 BQ256 候选在当时源码上的只读诊断如下，并非本轮 normal 测量：

| 统计 | 值 |
|---|---:|
| guarded topology 节点 / 边 | 8,469 / 12,895 |
| flow / origin | 5 / 1,599 |
| condition facts | 117,189 |
| fact attempts / comparisons | 774,448 / 1,528,601 |
| propagation passes / transfers | 96 / 792,389 |
| 累计query注册次数 / charged cases | 9,501 / 16,384 |
| 有效 `_reach` 调用 / 不同参数元组 | 14,166 / 5,900 |

来源：`runs/refactor/l1-reuse/case-probe.json`。这些是编译分析状态，**不是 kernel
必需的运行状态**。这是一个 guarded 实例，不是整次编译所有分析器的总和。`_reach`
还包含 anchored origin 初始化，case 计费与它不是同一集合；两者之差不能精确称为
无效 case，去重比例也不能直接推成可获得的加速比。
query注册次数包含图更新后partial/false答案的重注册，并不是全局唯一key数。

此前的 transfer 共享、稀疏投影、查询合批、单调答案保留和 singleton 简化继续保留。
本轮进一步改变事实所有权、物化交付和内部表示。历史 final-only 统计不能直接与
本轮 planning 加单次完成的累计统计相减；对应阶段与冷测结果由状态页单独列出。

## 2. 实际调用链与三种不同问题

当前结构化 source 正常路径如下。结构分派仍然存在，不表示所有层次已经统一为一种
理想 transfer；完成的重构是各建立者把已接受结果交给后续，而非独立重建同一命题。

```text
Compilation：limit + completion_math（单次编译生命周期）
  PhysicalFacts；PrimFunction.to_ir → clone → lower_source(_facts)
    collect_source_scopes → 统一 _work → lower_regions / lower_scopes
  autosync
    scope：_plan_scope_parent
      _plan_leaf / plan_child_variants → 已完成 child analysis
      _scope_access → query boundary summary → ParentCompletion
      复用已有覆盖；构造可直接匹配的协议；求解其余新关系
      child width/borrow → parent pool → InvocationResourceProof
      _install / _read_protocols：物化对应
    scope query：plan_query_memory → QueryMemoryPlan + QueryEventReuseProof
      _complete_scope_plan → 组合旧结果，只求解新增 query 资源边界
    loop：_plan_leaf → _leaf_result / InvocationPlan.complete
    sequence：_plan_boundaries / plan_calls → 已选 bridge 与覆盖结果
      _install / _read_bridges + native snapshot：物化对应
  已建立 report/analysis
  若 debug_verify=True：另从最终 IR 执行 verify_plan
  _emit_checked → 不可变字符串 snapshots
```

`Compilation(synchronize=False)`、公开 `verify_plan`/检查型 codegen 入口从实际 physical
IR 建立事实。legacy finite、mixed、static dispatch 分支仍经 `_finish` 独立终检；
本轮未删除这些生产入口，也未把它们称为已经完成的结构化 normal 路径。

定位：[compilation.py](../src/xdsl_npu/compilation.py)、
[scope_sync.py](../src/xdsl_npu/scope_sync.py)、
[scope_query_resources.py](../src/xdsl_npu/scope_query_resources.py)。

需要分清：

1. **不同命题的必要分析**：内存安全、通知消费、CV 可见性和无死锁各自有义务。
   内存依赖正确不等于 event 可复用，也不等于参与者不会互相阻塞。
2. **已移出的重复证明**：结构化正常路径不再独立重验已完成的 leaf、parent、sequence
   和 query memory。child resource、parent pool、counter invocation 结果交给 query；
   query memory 与新 ID rotation 的规划结果交给最终组合。ID/名称替换保持 native
   效果和 cut，按替换合同更新结果，不因未改动的访问重建依赖。
3. **单次求解本身过大**：片段、条件和源事实扩张，即使只运行一次也昂贵，必须改表示。

所有权交付处仍检查实际协议的端点、guard、ID、operand、block/order 与 native snapshot。
这是物化对应和结果绑定检查，不是重建访问/完成图；线性 IR 对照本身也有成本。
`PhysicalFacts` 只服务一次私有编译，按真实模型和 live-in 前提交付提取/依赖结果，
不是面向任意可变 Module 的通用验证缓存。外部可编辑 IR 和显式 `explain()` 重新检查。

历史审查使用的 TileLang 参考提交为 `6ba65d204c937fcb1c43b1a22b32979506595655`；
`insert_sync.cc::OptimizeSyncPoints` 先消费已有覆盖、为缺口加入确定协议，固定端点后
计算消费证据供宽度选择复用。借鉴职责顺序，但它也有闭包、重试和耦合，不能以 C++
实现或文件更短作为复杂度证明，不迁移其整套 pass。

## 3. 复杂度变量和真实增长来源

| 符号 | 含义 |
|---|---|
| N | runtime 输入长度/迭代数；不得成为静态枚举维度 |
| T | native 模板数，不含 runtime 实例 |
| A | 精确访问描述数，包括布局、scratch、SPR |
| B、I | 必须区分的范围片段数、访问与片段实际相交记录数 |
| H | 真实 RAW/WAR/WAW 等依赖记录数 |
| K | 前沿条款数，区分 clock、guard、owner、generation、里程碑 |
| P | 同一相连约束分量必须区分的周期/余数类数 |
| V、E | 分析图实际节点、边 |
| O、D、F | origin 数、分析域数、条件事实总数 |
| U、Q | 累计事实插入尝试、语义查询数 |
| C、J | 逻辑事件数、需处理的寿命冲突/共享约束数 |
| W | 静态query证明窗口的副本数，不是runtime query数 |
| Z | 输出 IR/ASC 大小 |

当前成本至少包含：

- `scope_access.py` 对合格 stationary root 使用周期摘要；需要历史 definitions、
  非 stationary 地址或当前 query 不能消费的长 reader span 仍用完整固定表示。
  固定 child 次数 × ref 行段 × group 坐标的乘积因而尚未普遍消除。
- 访问按端点切分后，宽访问覆盖多个细分区间。一般交错范围的相交记录 I 可达
  `Θ(A²)`，不能只统计 B，更不能把每个字节当一个必要状态。
- `region_sequence.py` 还按 count 边界、period 和 frontier 组合；共享 pattern 后，
  真实物理区间与义务仍需实例化。
- 旧 child 投影的端点平方步骤已由 `7a32407` 改为稀疏投影；1,800 点链生成 1,799 边。
  不应再把它写成当前仍执行的平方算法。但端点数量、投影成本和条件传播尚未消失。
- 条件前沿对小桶/dense mask 直接扫描，大且稀疏时才用 origin 范围索引。相连区间
  可以减少 F 和下游工作，但最坏仍扫描桶内 f 项；累计包含/合并工作可达平方。
  `fact_comparisons` 与 `fact_index_steps` 分列，不能把少做比较当作总成本必然下降。
  大整数 `&`/`|` 的成本仍随 origin 位长增长。
- 全 origin 传播包含 V × D × 条件碎片 × origin 信息。不同 guard 无包含关系时不能
  简单合并；一般布尔条件的互不支配集合可能指数增长。
- 图增边使部分/失败答案重算，触发 replay，影响后续资源查询；总代价必须包含各轮，
  不能只报告最后图的规模。
- event pool 同 bank 两两组合、宽度和 parent/child 首末借用会增加查询。物理 ID
  只有 8 个不意味着数百个逻辑通知的分配是常数时间。
- 不相关 buffer 的周期取全局 LCM 会放大状态。只合并真实相连的约束分量；确有耦合
  时保留并报告，不靠省略 guard/余数降低数字。

`completion.py::coalesce` 已从反复两两合并改为 period/residue 分组排序扫描，exact
点可桥接相邻片段，保留孔洞。若有 g 组、s 个 exact 点，仍含 g×s 的归组筛选与各组
排序/扫描，不能一律称线性；工作内存为片段与点的线性规模。`_uncovered` 仍含阈值
切点×组合 period×guards。

`Graph.potential` 已保留 append-only 排名证书，只加入新边并松弛其受影响后继；增大
stride 时同比缩放旧排名，保持旧严格余量。返回 tuple snapshot，后续变更不改旧结果。
复制/扩展排名仍需 O(V)，一般约束松弛与 cycle diagnosis 最坏仍为 O(VE)。
`completion_math()` 在一次编译中共享规范 CountDomain 及 intersection/includes 的纯
数学结果，退出即释放；不缓存图、coverage 或 verifier 结论。distinct 值/关系数仍需
计入空间，命中/构造与关系对数由 `completion_math` 报告，不使用跨编译预热掩盖成本。
`scope_query_planning._query_width` 在证明窗口不足时仍会重建 QueryCompletion，以静态窗口 W 复制
parent拓扑，单次仅复制就要 O(W(V+E))；历史默认BQ128的W=3将4979个parent节点扩为
14945个query节点。W不是runtime query数，但必须计入总成本。

所以现状不能诚实简写为一个 `O(T)` 或 `O(T²)`。**不随 N 增长只是必要条件，远不是
编译够快的充分条件。** 各阶段必须记录输入、实际输出、中间状态和重复次数。

### 尚未全部达到的有条件目标

规则流水（仿射/规则 strided 足迹、结构化 guard、有限 stage/slot、固定参与者）：

| 阶段 | 输出敏感目标 |
|---|---|
| 前端/标量规范化 | O(AST + 表达式 DAG + 实际重写工作) |
| 连续区间访问索引 | O(A log A + I) |
| 依赖与 region transfer | O(I + H + K 的实际合并工作) |
| 协议构造 | O(H + 对外边界条款 + 新协议数 + 实际条款匹配/条件求交工作)，不含全源全点闭包 |
| 事件分配/进展 | O(C log C + J + 有界资源求解工作 + 排名约束求解工作) |
| 物化/输出 | O(T + 新协议数 + Z) |

这不是任意动态程序的线性时间保证。fractal/间接访问求交另计；K 的合并、J 都可能
平方；一般偏序资源冲突不能当区间着色。不能把昂贵全图求解藏进 `covers()`、
`summary()` 后声称只有一遍扫描。明确的非法、当前不支持和分析预算未决须分开诊断。
对能用同域capture前缀rank比较的义务，覆盖可用常数次比较；需要索引区间查询时计
O(log K)，扫描reader antichain/guard条款时计实际长度，不能一律算O(1)。已有进展排名
被变换保持时只处理新增约束；确需一般求解时仍将最坏O(VE)列入成本。

## 4. 编译阶段与所有权

只保留 source IR、physical IR、同步后 physical IR 三个可打印边界，使用 xDSL 的
SSA/region/rewriter/pass；不可变分析索引引用同一份私有 IR，不建立第二套可变程序树。

| 阶段 | 工作 | 输出保证 |
|---|---|---|
| A 输入归一化 | AST、runtime/constexpr、类型、alias/owner 合同 | 输入合法、假设显式 |
| B 标量规范化 | 纯整数 DAG、canonicalize/CSE | 数值/地址等价，effects 不变 |
| C 原生 lowering/固定排步 | 原语、layout、scratch、engine、stage | 完整物理效果、数值边界、确定 issue 结构 |
| D 依赖/region transfer | root/range frontier，RAW/WAR/WAW | 实际访问的唯一义务及边界摘要 |
| E 同步构造 | ready/release、首末、局部/CV 协议 | 覆盖内存义务，导出事件寿命和进展约束 |
| F 资源/进展 | 配对、width/ID、ownership、seed/drain、阻塞 | 资源合法，原内存/排步事实保持 |
| G 物化/ASC | 具体同步映射、标量表达式输出 | 位置、guard、参与者、ID 与确定计划一致 |

这是职责划分，不表示表内每项都已成为独立线性 pass。当前使用 `PhysicalFacts`、
leaf analysis、`InvocationPlan`、bridge selection、`InvocationResourceProof` 与
`QueryMemoryPlan` 交付已建立事实；尚无通用 revision/依赖失效框架。会改变访问的
布局、buffer 复用和算子政策须在相关事实建立前固定，晚改不能继续沿用旧结果。

## 5. 前端、表达式和可读 ASC

保持 TileLang 风格 `prim_func/serial/Pipelined/Stage/Cube/Vector`，不恢复旧 task DSL。
详见[DSL](dsl-multistage.md)。stage、buffer slots、event width 是三个不同量。

当前 AST 的 serial/Pipelined 各构造一次 region；仅显式 unroll（≤64）复制。
`collect_source_scopes` 与 `_work` 已统一 source context、clock、stage、guard 和 native
work 收集，重复 `_legacy_work` walker 已删除。`frontend` 属性仍用于明确的语法/recipe
admission，不再选择另一套 work walker。`symbolic=False` 和有限 physical 后端仍在，
结构化 lowering 也仍按 loop/sequence/scope 组织；不能把 walker 统一称为所有后端归一。
固定 child 的压缩资格见下一节。

runtime 长度、stride、metadata 坐标保持设备 SSA。dtype、tile 容量、layout、显式
固定政策才参与特化；常量界也不自动 unroll。native recipe 必需的静态分段单独计数。

现有 canonicalize/CSE 位于 `asc_scalar.py` 的纯整数临时 func，不是全程序 pass。
应显式组织这个阶段，但单纯前移不算减少总工作。`scalar.Expr` 的数学整数/非负合同
与 ASC i64 signed 除法语义不同；必须明确溢出、除数、原始轴和 dominance 条件。
已有 uniform DAG 临时变量输出予以保留。

异步 copy/MAD/FIX/Vector、event、SPR setter 不能因 SSA 结果未使用而 DCE，不能按
相同 operands 做 CSE，也不能忽略 ownership 移动。通用优化只进入已知纯的子图。

分配后 width=1 的 event 输出常量；连续 ID 轮换优先 `base + ordinal % width`；
非连续映射确有需要时保留小映射。证明整数映射完全等价后，不重新证明 kernel 同步。
禁止 regex 改 ASC、算子名或特定 shape 特判。emitter 只消费确定结构，不再次选择协议。

## 6. 依赖和层次边界：压缩表示，不缩小语义

访问身份为 root、精确 footprint、owner、generation/slot、activation、milestone。
逻辑值版本绑定来源，物理 root/range 找复用冲突；SSA use-def 不证明异步完成。

连续范围用分组端点扫描，规则二维/fractal 优先保留紧凑 descriptor，仅在必要求交处
细分。padding、stride 孔洞、recast 和 subview 必须准确；包围盒导致过度等待也不合格。

每个范围的逻辑访问 frontier：

```text
read  → 从最近定义建立 RAW；加入 reader
write → 从旧定义建立 WAW；从待保护 readers 建立 WAR；更新定义/reader 集
```

“最近”和“更新”只是依赖构造，**不是宣布设备已经完成**。只有 E 阶段的真实硬件关系
才允许覆盖。不同 engine/owner 的 reader 不可因源码较早而删掉。

同端点、完整条件、强度的多字节义务共享一次同步求解，并保留 footprint 列表；不为
每个 cell 复制完成图，也不把不同条件和范围随意并集。

当前 region 组合由物理摘要、child completion 和 event 结果共同保留 entry reads、
first/last writes、pending readers、slot 相位、guard 和完成/通知边界，并非一个已经
承载全部合同的通用 `Transfer` API。父层导入实际需要的边界；内部 scratch 在 child
解决。**不能只保留 entry/exit**：当前父同步和借用会插在 child 的实际 native
`admit_issue/admit_queue/after_queue` 切点，需按需导出有类型的中间端口，保持排步。

本轮 `select_fixed_storage` 仅为合格 stationary root 选择周期扫描：地址不随 child
generation/count 漂移，无 clipping/tail、last-only 或特殊 MAD RMW；需要保留历史
definitions 的 root 和 call-local frame 不压缩。短固定域保留原完整扫描。周期路径用
一个精确 period 和首末 transfer 表达重复 reader，保留 `ReadSpan`，不授予完成关系。

query 消费者尚不能通用承接长 native reader retirement span，因此
`allow_read_spans=False`；只有每个 reader 都有匹配周期 writer 的合格 root 可压缩，
其余保持 full 固定表示并计入原访问预算。drifting、definitions 等模式仍有与固定
count 相关的扫描工作；不能宣称所有固定 child/count 已压缩。选择先于 row tuple
分配，原 fragment、definition、visit 预算仍生效。

N=0 不制造 first/last 或 drain；child 控制返回不能证明内部异步访问退休。推广到任意
循环的 entry 参数化体转移、首末和 slot 组合仍是目标，须先论证摘要族闭合。

尚未全面接通的边界 transfer 方向如下；它不是本轮实现的 API 清单：

1. `AccessIndex[root]`：精确范围/规则descriptor索引，以及带域的定义和reader前沿；
   独立存储的相交查询不会遍历其他root。
2. `Transfer`：输入参数、需要的typed cuts、输出条款和未决义务。相同native body的
   转移共享一次；call仅绑定真实root、clock、owner与坐标，不共享这些调用身份。
3. `FactRef`：指向一次建立的规则结论，记录直接前提。数据不是“所有origin在每个节点
   的位集”；只有对外frontier/实际acquire所需事实进入对应状态。

构造时对相同完整语义条款建立唯一ID。相邻范围只有在访问/域/里程碑都相等时共用一条
transfer；物理范围仍保留。按已证capture包含关系去除被支配reader，而不是无条件保留
“最后一条reader”。没有包含证明时保留antichain，并把它的实际长度计入K。

结构化条件不立即展开成所有分支组合。分支内事实保留原guard；同一域的两条可能路径
在join处只能保留双方都保证的完成信息，不可直接并集。不同互斥域可并列为带guard条款，
不能把EVEN上的reader A和ODD上的reader B并成ALL上A、B都完成。只有消费端实际需要时
才对相关条件求交，记录碎片增量。一般布尔条件仍可能昂贵，这个表示不承诺消除真复杂度。

## 7. 同步构造与循环归纳

分开保存 D 的访问义务和 E 的已知完成前沿。按真实 issue 结构处理 acquire：

```text
义务已被相同域/强度的事实覆盖 → 直接引用，不生成同步
否则 → 在真实最后访问后 capture、目标访问前 acquire
       可由构造规则直接匹配的义务交付保证，其余新关系继续求解
组合已有 region 摘要/完成结果 → 处理循环边界 → 导出资源/进展约束
```

不建立所有候选协议图，不全插后试删，不承诺全局最少 event。同步选择改变性能，不能
把“少一条同步”当唯一目标。

完成前沿必须区分 `IssuePrefix/QueueBeginPrefix/RetireCapturePrefix/ImplicitCompletion`
以及通知消费。只有同 actor/clock/guard 且已证明完整 capture 包含关系时才能压成前缀。
`native_order.py` 的 task.prefix 是 `join(task.end, before.retire)`，**不是最后一个
native 的 end**；Set.effect 捕获 retire，但不能回灌成后续 native 的退休或开始保证。
Wait 仅向真实 admission cut 之后传播，不能反向给过去的操作添加完成关系。

`establishes_directly`、bridge selection 和 owned query plan 已把可直接建立的关系及
已求得的 coverage 交给后续。一般 guarded/anchored/query 关系仍用现有符号求解器，
包含 origin×节点×条件的传播维度；本轮没有把它整体换成理想边界 transfer。

其中 `SymbolicCompletion` 已使用条件 interval frontier：只有在完整 CountDomain 上
可比较且相连的区间才能合并，并只移动共享 origin bits。双方都需残留不同 bits 时
不拆成三条记录；不同 guard、clock/domain 或 origin 不借用彼此的覆盖。pending 接收
本次有效支撑，并消费前沿的精确编辑；已发送的旧宽区间不因合并而重新入队。
只有已有 native monotone carry 可以延到 horizon，普通 end/Set 不获得新归纳权。

受控 `NodeActivations` 在写入时禁止修改旧 key 和为已导入旧节点补 activation，读取
只检查 append log 新项；外部普通 dict 仍执行完整旧节点突变检查。这减少重复扫描，
不把外部输入默认视为 immutable。不可压缩 reader/guard antichain 的实际长度仍计入 K。

通用循环组合需包含 slot 代次、entry credits、pending readers、通知消费和退出状态，
并满足下列合同；这不是通用 body transfer 已完成的声明：

1. 零次和 startup：没有虚构 reader/token；短循环和未使用 credits 有定义。
2. 任意合法 g→g+1：同一体转移保持访问、代次和事件合同。
3. exit：所有对外访问/通知按合同退休，最终 drain 恰好位于真实边界。

有限 stage/slot 决定有限边界类别，不能按 N 或独立周期的乘积枚举。guard 保留结构化
分支和孔洞；动态 stream 按实际 task generation 推进。一般数据依赖流尚未接通，明确
拒绝，不写成已支持。不同 actor block 的 Python 顺序不是 CV 完成关系。

后续的通用循环 transfer 目标是以 entry frontier/prefix 为符号参数解释一次 body，得到
`S_out=F(S_in,g)`，
再将backedge连接为slot/通知的代次关系。规则环应直接求解“g与g+B使用同一slot”的
关系；reader跨度保存为带步长的范围，由归纳合同建立退休覆盖，不枚举范围内每次读取。
不能靠不断展开g直到状态看似稳定，更不能用小N样本代替这一步证明。

后续仍需回答三个具体问题：这种 transfer 族是否对当前嵌套 CV 的组合封闭；
内部cut参数是否足以保持当前borrow所需事实；对规则FA是否真正将K降至边界/依赖规模。
如果必须退回全源传播才能回答，该边界 transfer 原型不通过，应修改表示后再推进，
不将问题藏到 F 阶段。

内存安全、Set/Wait 配对仍不保证无死锁。固定排步另建立阻塞依赖和 token 义务；零距
等待环拒绝，跨代环还需启动 token 与参与者激活。不能只凭 distance>0 宣称进展，
也不能补全局 barrier/ack 改变用户政策。当前只接受满足已实现 native、配对、资源和
排名合同的协议；通用 body transfer 的归纳闭合仍需上述论证。

等待范围反例（省略 ID 和收尾，只说明放置）：

```text
before: load X[g%2]; R(g, 最后读X); U(g, 无关计算)
        load X[(g+2)%2] 将复用X

after:  if g>=2: Wait(release,g-2)
        load X[g%2]; Set(ready,g)
        Wait(ready,g); R(g); Set(release,g); U(g)
```

release 捕获 R 的真实退休，不等 U。若后续 cast 又读 X，最后 reader 必须改为 cast；
若两 AIV 都读，必须保留两者贡献。MTE1 读完 L1、MAD 读完 L0、FIX 读完 L0C 分别
建模，不全等到 PV 输出完成。局部 event 不自动授予 CV ownership 或 GM 可见性。

## 8. 事件分配只解决资源问题

当前已实现同快照资源结果交付：child 分配结果、parent pool 收据及 invocation counter
backlog 组成 `InvocationResourceProof`；query 复用这些事实，只加入 old/new query
边界、mixed pool、counter 边界及新 query event 的义务。`QueryEventReuseProof` 交付
实际 width/ID rotation 的已完成结果，不在最终组合时再问一遍相同关系。

分配器仍分 child width/borrow、parent pool、query 三层，query 也仍有静态拓扑副本。
一次收集整个 scope 的逻辑通知约束并统一求解仍是后续目标，不能把收据复用称为该
联合模型已完成。各层均保留下列资源合同：

- local 二值 event：旧 Wait 消费必须先于同 ID 再 Set；width 依据真实通知 ordinal
  和允许重叠推导，不能简单等于 buffer slots。必要的 1..8 宽度试探只处理资源约束。
- CV counter 使用真实计数、参与者和可见性合同，不能套 local 二值规则。本地每核
  有向 bank 8 ID；跨核物理空间和当前实现使用子集单独列出。
- 保留已支持的 parent/child 空档共享；必须证明旧 child Wait→parent Set、parent
  Wait→新 child Set，条件缺席和跨 query 同样处理。不能以独占 ID 或额外等待退化。
- 只有具有已证总序切点的寿命才用区间扫描；偏序用显式约束。一般冲突着色可能困难，
  允许针对新资源关系的有界求解及符号传播；不得为未改动的访问重复提取依赖或重证
  已有内存命题。
- ID-only 改写保持逻辑拓扑，但共享必须先证明整个池的消费关系；不能通过“分配到
  同 ID”凭空增加顺序，再用该顺序证明自己合法。
- 改 Set/Wait 数量、cut、guard、ordinal、width 会改变寿命，需重算相应资源/进展，
  不能假装只是 ID 改名；端点和访存不变时不重算内存依赖。
- 资源不足报告实际 bank/存活关系/未决原因；启发式无解不等于硬件无解。不增加 barrier、
  搬运、buffer 或延后 release 兜底。

## 9. 正常编译与深度验证的边界

已接通的结构化 normal 路径从合法输入，经保持不变量的构造/变换得到确定产物。
后续直接使用已有事实，不再全局证明“刚才那个正确变换是否正确”。实现是否遵守合同，
由针对性反例、差分、独立有限执行模型、debug 深度验证和设备资格检验。

本轮已通过 owned leaf/sequence/invocation/parent/query 结果与物化对应，移出这些
正常路径的独立全图终检。`debug_verify=True` 仍从最终实际 IR 独立建立分析；公开
`verify_plan`、外部已同步 physical IR 和显式 explain/codegen 检查不消费旧内部结果。
legacy finite/mixed/dispatch 的 `_finish` 独立终检仍在。

任意导入/修改的 physical IR 重新建立信任，不接受 `verified=True`、序列化 summary
或 frozen dataclass 包住可变 Module 作为已验证依据。内部结果只在私有 IR 的受控
变换及已检查 snapshot 对应范围内有效。详见[不变量清单](compilation-invariants.md)。

## 10. 性能和能力边界

本轮不恢复算子优化。未来编译器替换的资格保持同一 DSL/政策，比较搬运、buffer、
MAD/Vector 工作、舍入边界、release 范围、CV 阻塞位置、填排与最终 drain，再测设备。
没有多余 flag 不等于没有性能损失，也不能通过降低接收能力获得漂亮编译数字。

独立 AscendC、ACLNN、ops-transformer 源码、PMU、理论算力/带宽各自提供证据；
后续测量仍用大规模输入和同一 runtime binary，不以小 kernel 的 launch 时间作吞吐结论。
默认 FA 的历史 ACLNN 差距 18.3% 仍未解决。重构无回退与达到 ACLNN 目标是两项验收。

本轮已落地的是事实交付、正常/独立检查边界、保持映射的物化、增量排名及具体表示
压缩。一般 guard/region/query 求解尚未全面改为理想边界 transfer，复杂 antichain、
物理相交与资源约束仍可能超线性。编译耗时、ASC 对应、CANN 和设备各自验收；
测量统一见状态页，不以微测或移出终检本身宣称秒级编译、完整统一或性能无损。
