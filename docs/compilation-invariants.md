# 编译不变量、变换合同与验证边界

2026-10-16。历史审查基线 `fdff8c5`，本轮实施基线 `216b711`。
这是[流程设计](design.md)中已落地合同与继续约束后续变换的不变量清单。结构化
normal 路径已交付 owned facts、资源和物化结果；独立 verifier 保留为外部输入与
显式 deep 检查入口，legacy finite/mixed/dispatch 仍有独立终检。当前状态和冷测见
[状态页](refactor-progress.md)。

**原则：输入事实只建立一次；正确变换直接交付保证；后续使用保证，不重复全局求证。
只有相关前提变化才失效。实现错误交给回归、独立模型和调试检查发现。**

## 1. 什么叫“变换已证明正确”

对每个 pass 给出以下合同，而不是每次编译重新证明整个 pass：

```text
{输入不变量 I，额外前提 P}  transform  {输出不变量 I'，保持的事实 R}
```

分清三层：

- **规则的论证**：为何在前提成立时该变换保持语义；按 pass/硬件原语写一次。
- **当前程序的前提**：例如地址不别名、除数正、当前资源池足够；编译时建立一次。
- **实现的检验**：测试验证代码遵守规则，不把生产中的反复全图搜索当作实现正确性保证。

当前合同由实现、针对性回归、破坏性反例和独立有限模型支持，不能宣称整套编译器
已经形式化验证。文中的“证明”包括带明确前提的算法论证和可组合分析结果，不要求
引入定理证明器或大型证明框架。

普通路径不为每次 ID 改名重跑内存分析，也不逐 pass 重新播放整个 proof DAG。构造器
保证类型/字段一致性，必要的新约束求解一次；复杂的一致性审计放在 debug/reference。
最终 emission 若只是保持映射的纯输出，不承担重新推导同步的任务。

## 2. 不变量目录

| 编号 | 保证 | 建立位置 / 主要依赖 |
|---|---|---|
| INV-01 输入与域 | 类型、runtime/constexpr、参数范围/整除、循环语义、实例存在性明确 | 前端/外部 IR 入口；参数和控制结构 |
| INV-02 来源与范围 | root/view provenance、真实 native 字节足迹、padding、scratch、SPR 完整 | 原生 lowering；地址、layout、dtype、原语合同 |
| INV-03 值版本 | 每个读取绑定正确定义；RMW continuation 和借用不是 snapshot | 物理依赖分析；root、版本、控制域 |
| INV-04 依赖前沿 | 同一物理字节/代次的 RAW/WAR/WAW 完整，旧 readers 不因写出现而凭空退休 | D 阶段；INV-02/03、owner、slot |
| INV-05 排步 | actor/engine、clock、stage、issue order 和内部插入 cut 固定 | C 阶段；策略和 native lowering |
| INV-06 完成事实 | issue、开始、退休捕获、隐式完成、通知消费严格区分 | native transfer / 协议构造；dav-2201 合同 |
| INV-07 内存安全 | 每个访问义务由相同实例域/范围/强度的完成事实满足 | E 阶段；INV-01..06 |
| INV-08 ownership/可见性 | 指定版本的 writer/reader 参与者完整；release 与可见性机制对应 | 局部/CV 协议；地址、owner、target |
| INV-09 事件等价 | 逻辑通知和实际 event 一一匹配，旧消费先于复用；计数不溢出 | F 阶段；ordinal、bank、width、ID、参与者 |
| INV-10 进展与退出 | 阻塞依赖可进展，startup/zero/tail/drain 完整，无虚构未来消费者 | 协议组合和资源求解；控制、同步与通知 |
| INV-11 数值/性能政策 | dtype、舍入/状态语义和固定排步、搬运、buffer 政策保持 | lowering/优化合同；算法及专家政策 |
| INV-12 输出对应 | 物化同步和 ASC 与已接受计划的位置/guard/参与者/ID 一致 | materializer/emitter 的变换合同 |

不是所有阶段都重查整张表。每项事实有唯一建立者，后续只消费或明确失效。
IRDL verifier 只建立其负责的结构事实，不能代表 INV-07/09/10。

## 3. 关键语义规则

### 3.1 物理访问、定义与最后读取

`ReadBinding/DefinitionRun/ReadSpan` 的历史定义关系应保留。退出 frontier 只描述后续
可能冲突的访问，不能用退出的 last writer 给之前的所有 read 重新绑定。

原地化或 layout 变换必须更新真实 footprint。tensor 名不同不证明不别名；同一 root
不同代次也会因 slot 取模复用。未知动态别名或 ownership 要求明确输入合同或拒绝。

最小反例：UB `score_in` 从独立输出变为原地处理后，最后读取延到 cast，旧 Muls 后
release 不再合法。但 GM Score 的最后 reader 仍是 MTE2 import，不能把 GM release
也一起延到 cast。不同存储有不同退休点，不能传播一个整 task 的“最后读取”。

### 3.2 native 完成前缀

当前仍使用 `native_order.py` 的分离规则，依赖其 dav-2201/CANN 原生合同：

```text
task.prefix = join(task.end, previous.retire)
Set.effect 采集 previous.retire
Wait 消费匹配 Set，向实际 admission cut 之后传递事实
```

`task.end` 不是此前所有 native 的完成；Set.effect 不进入后续 native 的 retire 或
FIFO-begin 前缀。不同 actor/clock/guard/参与者的前缀不可只取最大序号。

已有通知取得的跨 pipe 事实可以由后续合法 capture 携带。这是消除冗余同步、保留现有
容量和性能的重要能力；新算法不能退化成每对 reader/writer 独占一个通知。

### 3.3 同步构造而非再求证

对具体义务 `retire(R,g) → begin(W,g+d)`，协议构造器要求：

- R/W 实例在相应 guard 下存在，clock 映射正确。
- Set cut 实际捕获 R 的退休；Wait cut 先于 W 的访问。
- 机制允许该 actor/engine/参与者组合，并提供所需可见性。
- domain/ordinal 对齐，新增通知由资源阶段处理。

`establishes_directly` 和 bridge selection 对可匹配的实际端点/域交付构造保证；
其余义务仍通过既有符号完成关系求解。已完成的 coverage 与请求一起交给后续，不为
同一受控变换重建全图。构造不自动保证资源和进展，这两项由 INV-09/10 解决。

通过已有事实省略同步不改变计划。若后续主动删除/合并已经插入的同步，则必须持有
**不依赖被删通知**的替代依据；更新所有引用和资源寿命。当前不做全局试删循环。

### 3.4 region 组合与循环

当前由物理摘要、child completion 和资源结果共同保留四类 typed prefix、物理版本/
reader、通知状态及实际 native cuts；尚未统一为一个通用边界 `Transfer` API。
仅 entry/exit 摘要无法保留当前父同步插入和 parent/child ID 借用。

循环合同区分 startup、参数化一步保持和退出；N=0 不产生虚构访问/通知。条件缺席
不等于旧通知已经消费。不同 clock 只有经明确坐标映射才可组合；固定 count 不授权
源码隐式 unroll，也不表示全部固定分析已做到与 count 无关。

`select_fixed_storage` 只压缩合格 stationary root：需要完整历史 definitions、frame、
drifting/count-dependent 地址、clipping/tail、last-only 和特殊 MAD RMW 保留完整固定
表示。压缩路径保留周期 word、首末与 `ReadSpan` 的全部读者，不将其视作退休证明。
query 尚不能通用消费长 reader retirement span，故禁用这种 span 的压缩；仅有匹配
周期 writer 的合格 root 可压缩，其余沿用 full 固定表示。选择在 fragment 分配前完成，
原访问、definition 和 visit 预算不变；一般 region/query 求解仍可能超线性。

参数域缩小通常保留旧的全域正确性；**删除 native 或同步的某些激活不等于参数域缩小**，
它可能切断原本携带事实的路径或使 Wait 没有 Set。必须重新处理相关协议。

### 3.5 event 资源

本地二值 event 的状态合同：

```text
Empty → Set.effect(k) → Full(k) → Wait.consume(k) → Empty
```

跨核 counter 使用真实参与者及累计规则，不能使用这个二值模型替代；当前合同还需
分别约束参与者 backlog（现有 mode-2 检查上限 15）。输入 ready、buffer release、
通知 consumption、最终 drain 是不同事实。

资源共享合法性必须建立在**分配前已有的实际完成/消费关系**上。不能给两个 channel
相同 ID，再从这个相同 ID 推导有序，循环证明它们可共享。

保留当前受限 parent/child 借用：distance=1、width=1、每个实际 child ID 至多一个
borrower；child self、parent self、两条跨边界关系均需成立。child 中间缺席时保留
FIRST/LAST anchors；跨 query 还要有整池状态归纳。只查相邻活跃 epoch 容易漏孔洞。

本轮 child 分配批次和 parent pool 的已建立结果由 `InvocationResourceProof` 交付，
counter invocation backlog 建立后也交给 query。query 只求解新增 old/new、mixed pool
和 counter 边界；`QueryMemoryPlan`/`QueryEventReuseProof` 交付已完成 memory 和新
ID rotation 结果。分层分配仍在；这不是整个 scope 已改为一次联合资源求解。

### 3.6 进展

证明 RAW/WAR/WAW 和 token 配对仍可能死锁，例如先发出一个阻塞当前 controller 的
Wait，而它所需 Set 在该 controller 后面才可能发出。

`Graph.potential` 使用 `φ(node,c)=stride*c+rank(node)`；c 是该 residue family 的周期
坐标，实际 generation 为 `residue+period*c`。约束为
`rank[b]-rank[a] >= 1-stride*delta`，保证每条实际边严格增加。

当前 append-only 图保留旧证书，只提交新增边，并松弛受影响后继；新边不满足旧排名
不等于死锁。节点数增长需增大 stride 时，同倍缩放旧 ranks，使旧严格余量继续成立。
排名返回不可变 tuple；旧引用不会被后续扩展原地修改。排名复制/扩展仍有 O(V) 成本，
一般松弛及独立 cycle diagnosis 最坏 O(VE)，不承诺任意图线性。

serial query 可按外层代次归纳：query 0 无向前 Wait；跨 query 的 Wait 只依赖更早
query；每个 query 内部、通知状态及配对已经正确。不能把“所有距离正”推广到任意 CV。

### 3.7 条件完成前沿与数学值

`SymbolicCompletion` 的记录为同一 flow/node 上的 `(lo, hi, CountDomain) → origin mask`。
合并须在完整 count guard 上证明区间相连及端点可比较，且只移动共有 origin bits；
不同 guard、孔洞、domain/clock 或 origin 的事实不能互相借用。两边均有独立残留 bits
时保留原紧凑记录。pending 由本次接收的有效支撑及已接受的精确编辑维护，避免把
已发送的旧宽区间重新入队；普通 end/Set 不因区间合并获得 monotone horizon 归纳。

小桶/dense mask 直接扫描，大稀疏桶可使用 origin 范围索引；这是相同语义的数据结构
选择，不是正确性 fallback。比较、索引、合并及 transfers 单独计数。一般 antichain
及 origin×节点×条件传播仍在，不能由该前沿命名推成完整边界 transfer 或线性复杂度。

`completion_math()` 仅在本次编译共享不可变 CountDomain 和 intersection/includes 的
纯数学结果；异常或正常退出均释放上下文。它不保存 IR、图覆盖或 verifier 结论，
外部新编译不会继承上次命中。canonical 幂等、关系缓存和等周期快捷路径保持原预算，
不能跨 EVEN/ODD 或边界孔洞扩大集合。distinct 值/关系对和复用次数显式报告。

## 4. 变换保持/失效矩阵

| 变换 | 保留 | 新分析或失效范围 |
|---|---|---|
| 纯 SSA 重命名 | 经双射映射的全部语义 | 更新身份引用，不求解完成图 |
| 已证整数等价改写 | 地址/guard 的语义及下游事实 | 只建立一次类型、范围、溢出/除法前提；不能丢掉原始非法轴诊断 |
| 新增局部合法同步 | 物理访问、依赖；未修改激活时旧内存完成路径 | 新关系增量、资源和进展；新增等待不保证进展单调 |
| 同 bank、同 width、共享成员不变的 ID 双射 | 访问、内存拓扑、寿命、进展 | 用置换规则更新实际 ID 和配置占用绑定 |
| event 表达式等价化简 | 包括资源配对在内的语义 | 表达式域内等价，emission 映射更新；不重证 memory |
| 改 width/ordinal/共享成员 | native 访问；满足配对规则时逻辑完成拓扑 | 所涉 lease、slot ordinal、首末/跨 query 配对及进展 |
| 移动/删除/改变 Set/Wait guard | 未改动的 physical footprints | 依赖其完成路径的 memory、pairing、resources、progress |
| 改 stage/issue order | 纯地址/footprint 可保留 | FIFO/admission/capture、相关 memory witness、resources、progress |
| 改 view/root/原地复用 | 无关 root 的分析 | 受影响 overlap、定义绑定、最后读、同步、容量及上层摘要 |
| 改 native 参数/recipe/layout/dtype | 无关 region 的分析 | footprint、数值边界、native transfer、buffer 和上层依赖 |
| 改 loop/clock/activation/owner | 无关 clock/region 的事实 | 实例、slot、参与者、完成、资源和进展 |
| 换 target 或硬件语义版本 | target 无关的源算法事实 | 所有依赖原硬件合同的原生、完成、可见性和资源事实 |
| 导出可变 IR 后重新传入 | 不信任历史内部证明 | 从实际输入建立一次可信状态 |

只失效实际依赖的事实；若某次变换真实影响整个 scope，该 scope 的重算是必要的，
不能承诺永远 O(1)。目标是不存在无修改、无新命题的全量重证。

## 5. 分析所有权与实现方式

不引入可伪造的 `verified=True`。xDSL operation 仍可变，`frozen dataclass` 包着一个
可变 Module 不能证明它未变。延续 `Compilation` 的输入 clone 与 artifact 字符串边界：

- `Compilation` 私有持有输入 clone，产物只导出不可变字符串；`.plan` 重新 parse。
  `PhysicalFacts` 在本次调用交付已提取 loop/scope/sequence 与按实际 loop/live-in
  前提建立的依赖。它保持模型对象生命期，不是对任意 mutable Module 的验证缓存。
- `collect_source_scopes` 一次收集上下文，`lower_regions`/`lower_scopes` 共用 `_work`；
  clock/stage/guard/native 读取不再由第二个 legacy source walker 重解释。前端属性仍
  可做具体 admission，结构分派和 finite physical 入口没有因此消失。
- `_plan_leaf` 返回已完成 analysis，`_leaf_result` 组织其结果；case 名称/ID 替换由
  `rename_leaf_plan` 保持 command indices、guard 与 native cuts，并更新实际 ID 报告。
  width、共享成员或 cut 改变依矩阵建立新资源前提，不能冒充 ID-only 改写。
- `InvocationPlan`、bridge selection 和 sequence call plan 持有实际请求与已完成覆盖。
  物化后读回实际协议并对照 native/operand/block/order snapshot；检查已返回域是否
  覆盖请求不等于重建完成图。未登记的任意 mutate 不受此保持合同保护。
- `InvocationResourceProof` 绑定 program/access/analysis/parent、实际协议、child pair
  和 invocation snapshot。query additions 是明确允许的变更；native、地址、cut、guard、
  clock 或旧资源变化使该结果失效。外部输入由独立建立者生成新结果，不接受 report bool。
- `QueryMemoryPlan` 绑定实际 query 声明、memory checks/answers 和完成快照；新事件的
  rotation 结果由 `QueryEventReuseProof` 交付。后续只建立新 query 资源边界，报告分别
  列出本次求解与组合复用，不把同一结果重复计为新证明。
- `QueryCompletion.append_protocols` 只接受实际IR中既有协议序列之后的追加，核对
  non-query snapshot和旧协议精确前缀。保留真实词法forwarding；仅新增弧重放其源事实，
  不重建native图或origin。完整正答案在增边下保持；partial/false保留原case并失效重读。
  window不足时才建立更大静态图；删除、重排、改旧guard/cut不能用append合同放行。
- `summarize_sequence` 的输出投影只省去调用者明确不消费的Cell/Obligation构造。
  所有transfer、reader历史与域检查保持；被省去的cell不能交给下一层推导invocation依赖。
- `NodeActivations` 在写入时验证旧绑定不可改变；已导入节点的“没有 activation”也被
  固定，读取只处理追加项。普通 dict 或不具该合同的输入仍逐项验证旧节点突变。
- emitter 消费固定位置/guard/参与者/ID。外部 IR parser、手写物理同步或 `.plan`
  修改后的再输入都是新输入，进入独立验证；内部所有权结果不序列化成信任标记。

当前没有通用的分层 revision/proof-DAG 失效框架；保持项由上述受控接口和结果绑定
建立。一般 region/query 图求解也未全面替换。后续扩展必须落实表中的实际前提与
失效范围，不能先取 `RegionTransfer`/`SyncPlan` 等名称便宣称已经完成。

## 6. 哪些检查放在哪里

| 模式 | 做什么 | 不做什么 |
|---|---|---|
| normal（已接通的结构化路径） | 一次输入合法化，所需依赖/协议/资源/进展分析，owned 结果组合、物化对应和输出 | 独立重建已保持的整套 memory/completion 图 |
| import physical IR | 从实际外部 IR 建立结构、访问、同步、资源、进展事实 | 接受序列化“已验证”标记 |
| debug/deep verify | `debug_verify=True` 独立从最终 IR 检查，保留 deep report/计时；定位失效遗漏 | 把独立重建成本算作已接通 normal 路径的必要工作 |
| tests/model | 最小反例、破坏 IR、有限交错、规则归纳前提及边界测试 | 用有界结果冒充任意 N 证明 |
| device qualification | ASC/CANN/硬件行为与性能证据 | 用 xDSL verifier 通过替代设备正确性 |

结构化 loop、sequence、invocation、scope/query 已按上述合同交付正常结果，
`Compilation(synchronize=False)` 和公开独立检查入口始终读取实际 IR。这不是通用
`verify=False` 开关；正常路径仍建立全部必要的新访问、资源和进展义务。

legacy finite/mixed/static-dispatch 仍经 `_finish` 独立终检，尚未统一到该正常路径。
`verification_mode` 表示入口策略，不能单凭该字符串断言任意 backend 已移走终检。
normal/deep 比较保留语义字段，将 `analysis_work`、`completion_math`、建立来源及
实际工作计数单独比较；不因移走重复证明就要求这些计数相等，也不一概忽略整数型
访问/依赖/域/ID 等语义字段。冷测与资格结论由状态页给出，不在此预先承诺耗时。

## 7. 首批必要反例及其用途

| 反例 | 必须发现的问题 |
|---|---|
| 双缓冲 R 后有无关 U | release 捕获 R，不扩大到 U；新增最后 reader 时正确更新 |
| 两 MAD 共享 K/V，Q/P 分 plane | K/V 等两个实际读者；Q/P 独立复用，不能按名字不别名 |
| shared L0C 的 QK clear/PV FIX | clear 等实际旧 FIX 读，不能被新逻辑 SSA 掩盖 |
| child 零次、一次、FIRST/LAST 中间孔洞 | identity、首末配对、非相邻 event lease |
| 两 AIV 仅一方消费 | 缺失贡献不得被当作完整 release/credit |
| 先阻塞 Wait、后发其 Set | 配对成立但进展失败 |
| ID 置换、碰撞、width 改动 | 安全重命名复用事实；碰撞拒绝；width 只重算相应资源 |
| 移动同步后复用旧 snapshot | stale witness 必须失效，不能由旧 report 放行 |
| 单一 N 参数换大数 | 模板/静态分析规模不按 N 增长；仅执行模型可在小 N 展开 |
| stationary 与 drifting 固定摘要、历史 definitions、query 长 reader span | 合格模式压缩且读者完整；其余保留 full 表示和原预算，不借压缩扩大完成 |
| 增量排名的旧 snapshot、节点增长、追加回边 | 保持旧严格约束；只松弛受影响后继；真实坏环仍拒绝 |
| 数学值的残余类孔洞、异常退出、再次编译 | 集合语义与 period 预算不变；不跨编译保留 arena |
| owned activation 与外部 dict 改写 | 旧节点新增/改变/删除 activation 均不能借历史事实通过 |

已有反例优先迁移，不新增一套重复的整算子测试。详细测试与阶段退出条件见
[实施计划](refactor-plan.md)。
