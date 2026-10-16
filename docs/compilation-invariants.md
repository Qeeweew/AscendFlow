# 编译不变量、变换合同与验证边界

2026-10-16，审查基线 `fdff8c5`。这是[新流程设计](design.md)的核心合同清单，
尚未实现新的 proof/result API，也未移除当前 verifier。

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

目前只有部分硬件规则和局部回归，不能宣称整套编译器已经形式化验证。文中的“证明”
包括带明确前提的算法论证和可组合分析结果，不要求引入定理证明器或大型证明框架。

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

沿用 `native_order.py` 的分离规则，并在迁移前核实其 dav-2201/CANN 来源：

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

满足前提后，其输出就是这条完成关系。后续无需重新用全图 reachability 确认。
构造不自动保证资源和进展，这两项由 INV-09/10 一次解决；不能用内存关系代替。

通过已有事实省略同步不改变计划。若后续主动删除/合并已经插入的同步，则必须持有
**不依赖被删通知**的替代依据；更新所有引用和资源寿命。首版不做全局试删循环。

### 3.4 region 组合与循环

transfer 包括四类 typed prefix、物理版本/reader、通知状态，以及按需导出的 native
内部 cuts。仅 entry/exit 摘要无法保留当前父同步插入和 parent/child ID 借用。

循环规则证明 startup、参数化一步保持、退出；N=0 是 identity。条件缺席不等于旧
通知已经消费。不同 clock 只有经明确坐标映射才可组合；固定 count 也不授权按 count
复制全部分析状态。

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

### 3.6 进展

证明 RAW/WAR/WAW 和 token 配对仍可能死锁，例如先发出一个阻塞当前 controller 的
Wait，而它所需 Set 在该 controller 后面才可能发出。

规则流水可使用排名 `φ(node,g)=stride*g+rank(node)`，检查所有实际依赖严格增加。
新增边符合已有排名即可保留它；不符合时求解受影响约束，不能立刻当作死锁，也不能
假定 O(V+E)。当前 general potential relaxation 最坏 O(VE)。

serial query 可按外层代次归纳：query 0 无向前 Wait；跨 query 的 Wait 只依赖更早
query；每个 query 内部、通知状态及配对已经正确。不能把“所有距离正”推广到任意 CV。

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

- 一次 compilation 私有持有 IR；内部 pass 通过受控入口修改，外部没有共享引用。
- 分层记录 effect/address、domain/clock、schedule、sync、resource、target revision。
  一个事实只依赖相关 revision，不对每次读取做全 IR hash。
- pass 返回结果及明确变化范围；同一构造规则可直接传递其保持项。未登记的内部任意
  mutate 不受此合同保护；debug 用结构指纹/重建差分发现实现违反约定。
- 逻辑义务和唯一 witness 关联保留；依赖索引用于失效。只保存实际产生的依据，不制造
  全源全点 proof DAG，不在正常路径重新遍历/验证所有上游依据。
- invocation 的 child/pool/counter 已建立结果直接交给 query 层；query 只处理新边界
  义务。报告中标明“组合复用”和“本次建立”，不虚增检查次数。
- 物化是固定映射的受控变换，不再由生成器重新猜协议。外部 IR parser、手写物理同步
  或 `.plan` 修改后的再输入视为新的输入，进入显式 import verification。

建议结果内容为 PhysicalFacts、RegionTransfer、SyncPlan、ResourcePlan；这些名称只是
说明职责，不要求先建四套空类和插件框架。资源与 native 事实分离后，ID-only 改写不得
调用访问提取或内存依赖求解，这是可测的算法约束。

## 6. 哪些检查放在哪里

| 模式 | 做什么 | 不做什么 |
|---|---|---|
| normal（目标流程） | 一次输入合法化，所需依赖/协议/资源/进展分析，受控变换和输出 | 从同步后 IR 重新构造整套 memory/completion 图 |
| import physical IR | 从实际外部 IR 建立结构、访问、同步、资源、进展事实 | 接受序列化“已验证”标记 |
| debug/deep verify | 独立从最终 IR 检查；对每类变换差分；定位失效遗漏 | 作为普通编译不可避免的默认成本 |
| tests/model | 最小反例、破坏 IR、有限交错、规则归纳前提及边界测试 | 用有界结果冒充任意 N 证明 |
| device qualification | ASC/CANN/硬件行为与性能证据 | 用 xDSL verifier 通过替代设备正确性 |

目标 normal 入口仍有逻辑完备的保证，并非一个 `verify=False` 开关。当前没有这条新
实现；旧 final 只有在替代合同、materialization 对应及针对性回归接通后才逐段退出。
旧独立验证器暂作显式 reference 路径，迁移结束删除生产中的重复入口和隐式 fallback。

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

已有反例优先迁移，不新增一套重复的整算子测试。详细测试与阶段退出条件见
[实施计划](refactor-plan.md)。
