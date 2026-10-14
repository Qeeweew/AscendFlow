# xDSL / Ascend 910B 编译器设计

2026-10-14 修订。本文规定下一阶段目标；[实现记录](refactor-progress.md)记录实际
证据。[DSL 规格](dsl-multistage.md)和[重构计划](refactor-plan.md)分别规定源语义与
实施门槛。本文不表示这些能力已经实现。本轮只修订文档，算子调参继续暂停。

## 1. 要交付的产品与本次修正

目标是让用户写出能复现专家流水组织的 NPU 程序，由编译器根据实际访问和
生命周期生成同步。可读的源码、可用的编译速度和设备性能是同时成立的验收
条件。同步正确但编译数分钟，或编译很快却扩大等待范围，都不能验收。

采用 TileLang 风格的 Python DSL，保留 `Pipelined / Stage / Cube / Vector`
作用域；使用 xDSL 0.73 的 SSA、region、arith/scf、effects 和验证基础设施，
直接输出 ASC，调用 CANN 9.1 / dav-2201。没有 TileLang/TVM/MLIR 构建依赖。

本次审查确认，原方向有可复用部分，但还不是完整方案：

| 原型现状 | 本次设计修正 |
|---|---|
| 单个外层循环已符号化；嵌套 pipeline 仍静态展开 | 前端和物理 IR 都保留循环层次，常量边界也默认不展开 |
| 单一 runtime size、受限轴表达式 | 多个运行时标量，支持 `ceildiv(Sk, BK)` 和正常块索引 |
| `versions` 默认绑定最近 pipeline | 用显式存储轴及 `slot = iv % B` 表达，保留所属迭代变量 |
| 有限与符号两套生产路径 | 迁移为单一结构化路径；有限实例化只保留测试用途 |
| CV 示例只有 QK、乘法/cast、PV | 真实 online-softmax FA 提前成为设计验收程序 |
| 全局周期 LCM、候选反复求证 | 按依赖连通分量处理周期，增量检查并设显式预算 |
| 重读最终 IR 的 checker 与 planner 共享关键算法 | 保留最终重检，另建独立有限访问/交错 oracle 暴露共同遗漏 |
| benchmark 从已序列化 IR 开始 | 正式验收包括 DSL 解析、源码 IR 构造到 ASC 的全部编译阶段 |

不重写已审查的 native effects、精确物理区间、完成前缀图和不可变编译产物，
但要适配其适用边界。当前 `Program(count=0)` 充当模板容器的约定不进入新规格。

## 2. 前端与排步契约

### 2.1 一种生产前端

选择受限 Python AST 前端。`@N.prim_func` 捕获源码，设备函数体不在 Python
执行；设备 `if/else`、循环和标量表达式生成 xDSL region/SSA。普通 Python
留在函数外作配置与工厂。没有“无法解析就退回 tracing”的第二套语义。

`N.constexpr` 表示显式特化参数；长度、group 数等默认是运行时标量。片上 tile
形状、布局、槽位预算和 stage 排步首版要求编译期固定。即使调用时知道长度，
也不把它偷偷放进特化键。源码不可取得时明确报错，可提供显式源码字符串入口。

语法只支持必要的 Python 子集、受限常量捕获和注册 DSL/macro 调用。宏在
examples 中定义，展开为有类型的普通操作/region，并保留来源位置；禁止递归、
无限展开和执行任意 host 副作用。形状注解经过两遍符号解析，不用任意 `eval`。

### 2.2 用户表达什么，编译器决定什么

| 用户可见 | 编译器负责 |
|---|---|
| 算法、精度/舍入边界、显式状态更新 | 类型、数值依赖及初始化合法性 |
| memory space、布局、视图、槽位和输入契约 | 精确物理 footprint、别名、容量与 ownership |
| 循环、stage、Cube/Vector 域、可选 issue order | 填排、排空、native 展开及 dispatch |
| 明确的 copy/matmul/vector/reduce 操作 | ready、最后读取、release、event 分配及最终 drain |

逻辑依赖是算法顺序；stage 是逻辑任务的派发位移；actor 是 AIC 或 AIV；pipe
是 Scalar、Vector、MTE1/2/3、Cube、Fixpipe 等真实执行引擎。四者不能合并。
一个 Cube region 可以包含搬运、MAD 和 FIX；region 结束不等于这些工作完成。

专家固定排步与将来自动排步都使用同一结构化表示及验证器。自动排步本轮不做。
固定排步无法满足依赖时给出路径诊断，不擅自改 stage、添搬运、扩槽或串行化。

### 2.3 循环、状态和嵌套

`serial` 和 `Pipelined` 默认保留循环；只有 `unroll` 明确请求静态展开，并受
结构预算限制。支持的索引从仿射、常数除数 floor/mod/ceildiv、min/max 开始。
循环上界在入口求值一次；可变 metadata 的读取不能作为纯表达式随意重算。

slot 是物理存储轴，generation 是逻辑写入代次。它们并不相等：同一 slot 被
不同代次复用。嵌套循环保留多维代次 `(query, kv, chunk, ...)`，不能映射成
依赖运行时长度的 Python instance 列表。query state、KV 交接、L1/L0 预取可以
分别使用独立轴和生命周期。普通 buffer 显式 init/update/output 足以表达状态，
暂不建设通用 State 框架；标量循环携带值使用 scf 的 iter_args/yield。

循环外可以写 Q cache 初始化及最终归一化。父子 region 边界不隐含全流水
barrier 或全子循环 drain；只等待当前用途确实依赖的里程碑。零次循环不得
读取不存在的最后一个结果，必须有初始化值或覆盖该情况的分支。

## 3. 一条结构化编译路径

```text
受限 Python AST / 显式源码
  → typed source IR（结构化控制、逻辑值、存储、stage/actor）
  → physical execution IR（仍保留循环、native effects 与所有权）
  → 同步规划并物化 events（同一 physical IR）
  → 最终合法性检查 → immutable Compilation → ASC → CANN
```

不为每层再建一个庞大通用框架。xDSL `arith/scf` 承担纯标量与控制流，自定义
最少的 NPU 存储/操作/排步属性。native recipe 可以展开固定数目的机器操作，
但不得按运行时长度展开；较大编译期循环也需显式 unroll 才能展开。

### 3.1 物理操作与完成语义

每个 native operation 记录 actor/pipe、发射位置、guard、精确读写集合、别名
root、layout 映射、slot 表达式、代次、SPR effects 及可信完成点。

必须分别表示：逻辑值产生、发射时标量捕获、输入最后读取、输出写完成、
buffer 可复用、通知可消费、kernel 返回前的 drain。当前 native 契约默认把
单条指令的 read-retired/write-done 放在同一可信完成点；没有依据时不得捏造
更早的指令内部 release。早释放来自选择真正的最后 reader，而非缩短机器指令。

带异步 effects 的操作不能被普通 DCE/CSE/LICM 删除或移动。允许独立验证的
纯索引化简；对计算、搬运、ownership 或完成点的优化必须重建受影响的证明义务。

### 3.2 依赖和区域摘要

从物理 footprint 构造 RAW/WAR/WAW，既含逻辑生产消费，也含跨代次复用。
不同名字可能别名，同 root 的不相交区域也不能被迫整体串行。布局转换、holes、
对齐与分片 ownership 进入访问集合；不得用包围盒代替真实范围来掩盖问题。

子 region 摘要导出：入口要求、精确访问及 guard、循环携带依赖、特定输出
ready、各输入最后读取/复用、仍存活的通知资源、零/一次/稳定段/尾部条件。
摘要必须保留可供父层使用的早释放点；“整个子循环完成”不能替代全部出口。
这个可组合摘要是待实现工作，现有单循环整数势函数求解器只是可复用的叶子技术。

实现从每个 alias root 的精确读写前沿开始，求交得到带迭代域/距离的冲突关系；
随后在保留 dispatch gate 的完成图上验证生产、读取与复用路径。简单内层循环
先导出参数化首/末访问及携带关系；父层合成后仍检查边界 guard 和资源状态。
无法用受支持的仿射/常数模表达这些关系时拒绝摘要，不能仅保留首末任务然后
假设中间都安全。首先用结构化 GEMM/递推反例确认摘要规则，再扩展复杂任务域。

按实际耦合关系处理周期：不相交 ring 的周期保留因子形式；仅在依赖需要时
枚举局部余数。若耦合分量周期过大，返回明确分析预算诊断，不能恢复成全程序
LCM 展开。复杂控制域同样受预算限制，不声称任意嵌套程序都能低成本证明。

### 3.3 同步与资源一同规划

依赖义务 → 目标允许的完成路径 → event/credit 候选及位置 → 资源分配与精简
→ 物化 → 最终验证。资源分配参与规划，不把一次贪心删边后的失败当作不存在
可行方案。首版可用确定性的有限搜索，必须区别以下诊断：

- 已找到违例：给出具体冲突访问、代次、缺失完成路径或死锁环。
- 输入超出可证明子集：说明动态别名、guard、ownership 或索引缺少何种契约。
- 规划器未找到资源可行解/超预算：给出候选、ID 压力及搜索范围，不声称数学上不可行。
- 数值契约不合法：例如越过显式 cast、改归约树或未初始化的 carry。

二值 event 的 Set/Wait 必须配对，旧 Wait 先于同 ID 下一 Set；跨 channel
共享 ID 需要生命周期不重叠证明。buffer 槽数、软件 stage 深度和 event 环宽度
分别计算。ready、credit、末次未消费通知的 drain 不互相代替。

### 3.4 dav-2201 契约边界

| 约束 | 实现要求 |
|---|---|
| 同 pipe 的发射顺序未必等于完成顺序 | 按操作/shape 采用经审查的隐式顺序；短 MAD 不能套用无条件 Cube 完成序 |
| 本地 event bank 每有向 bank 8 IDs | 检查配对、复用和末尾残留，不用全局 barrier 兜底 |
| cross-core Wait 阻止该 actor 后续各 pipe 派发 | 排步图必须包含 dispatch gate，避免 QK 被旧 P Wait 阻挡 |
| Notify 捕获此前源 pipe 完成，不是源 fence | 后续源工作可继续；发布和生效分开建模 |
| 1 AIC / 2 AIV 的 GM 交接 | 验证分片、参与者/quorum、可见性与返回 credit；两 AIV 不虚拟成一个 reader |
| mode2 通知及有限 backlog | 延用已审查协议；当前实现保守采用 IDs 0..7，4-bit 编码不等于已验证全部16 IDs |
| 910B 无直接 L0C→UB、L1→UB、UB→L1 搬运 | CV 用明确 GM handoff；不能借其他芯片路径消除真实代价 |

UB 192 KiB/AIV、L1 512 KiB、L0A/L0B 各64 KiB、L0C 128 KiB 是该目标容量
输入。API 与可见性以安装的 9.1.0/dav-2201 契约及设备证据为依据。详见
[源码依据](source-notes.md)，不能反过来用模型检查“证明”模型内假定的硬件行为。

## 4. 编译时间也是产品契约

固定源码、tile 和排步下，改变运行时长度 N 不能改变模板数、图规模、生成代码
或编译次数；输入长度不是特化键。常量边界检查的位数成本不构成按 N 展开。
这不意味着编译成本对程序静态规模必然线性。

| 量 | 含义及主要成本 |
|---|---|
| T、A | native 模板数、精确访问片段数；由源码和显式展开决定 |
| H | 真实冲突义务；重叠多时本身可达二次规模 |
| C、P | 同步候选数、局部耦合余数规模；不使用所有 ring 的全局 LCM |
| V、E | 完成图节点/边；线性 prefix 避免重复连接全部历史退休 |
| D、R | guarded 域及受影响证明查询次数；可能主导复杂程序耗时 |

区间索引按实际触及 cell 和 H 付费，可期望 `O(A log A + touched + H)` 的
前沿构造，不能保证 H 很小。局部图在固定 quorum 下可约为 `O(P(T+C))`；这
只是图构建成本。整数势函数的通用松弛最坏可能 `O(VE)`，候选试探仍可能导致
多次查询，不能以局部图公式宣传整个编译器为线性。

优化顺序：保留循环 → 精确访问索引/完成前缀 → 复用不可变分析 → 按变动范围
重查 → 有预算的候选/资源规划。禁止依靠缓存掩盖首次编译规模爆炸。每阶段
记录时间、峰值 RSS 和上述结构计数；取消/超预算在阶段边界和内层搜索可响应。

正式计时包括 AST/宏及源 IR 构造、lowering/native 展开、effects、规划、最终
验证和 ASC；Python 进程启动/import、CANN、可选完整图导出分别计时。区分：
冷编译、相同源码再次请求命中真正的编译缓存、读取已有产物属性。后者不是缓存
验收。缓存键含源码/宏/捕获常量、constexpr、目标契约、编译器、后端及选项；
CANN 产物另含工具链版本。运行时标量排除在特化键之外。

暂定目标：固定开发机常用内核 p50≤1秒、代表性 FA p50≤5秒、真实缓存命中
p50≤100毫秒；同时报告至少5次 cold 的 max 和 RSS。作为初期回归门槛，max
不超过上述目标2倍；超出时附结构计数/profile 定位，不能删验证或补 barrier。
这些都是待验收目标。当前源码 IR benchmark 和探索测量不满足这一完整口径。

## 5. 算子与设备性能贯穿重构

所有算法、数值 oracle、算子专属 host/ACLNN 适配和 AscendC 对照放在 examples。
核心只保留 DSL、IR、目标契约、同步/验证、ASC 和通用工具链。不能识别“这是
FA”来补正确性，也不能给某个示例单独写一条未经相同验证的编译路径。

| 验收程序/研究反例 | 对模型的要求 |
|---|---|
| 双缓冲搬运/elementwise、归约/normalization | last read、scratch 别名、尾部与归约顺序 |
| RoPE、gather/scatter | 不规则 view、原地更新；未知间接冲突明确拒绝 |
| MoE routing/finalize | 动态分区所有权、早释放、全局阶段不能当核内 event |
| grouped matmul、量化/反量化、FFN | 动态任务边界，完整归约后舍入/量化，混合精度 |
| GDN 递推切片 | carry/snapshot 区分，状态代次与读取退休 |
| dense/GQA/MQA/causal FA | online softmax、多阶段 CV、独立 query/KV/chunk 生命周期 |
| varlen/paged/sparse/persistent attention | metadata 捕获、非矩形任务域、跨 query state ownership |
| CATLASS 分层 GEMM | L1/L0/L0C 不同释放点、父子流水组合 |

首轮重构只实现向量/归约、递推小例、分层 Cube 与真实 dense FA；随后扩展
GQA/MQA、causal/tails。其余是必须检验设计的反例和后续门槛，不是已支持算子库。

### 5.1 固定排步先保真，再调优

每阶段对固定专家配置检查：源算法 → 物理计划 → 最终 ASC 的 native 指令、
GM 请求、buffer 容量/复用、等待位置和数值顺序。结构差异必须逐项说明；允许
经证明删冗余同步，不允许隐含额外搬运、扩大等待或缩小 tile。最终 ASC 是首要
可审查产物；代码符合意图之后仍需设备计时判断真实性能。

性能对照包含两个不同基线：同数值契约、可运行的 AscendC 源码基线，以及
安装 CANN 的 ACLNN 产品基线。ops-transformer checkout 不是已安装二进制的
源码证明。引用其流水/通信块设计时记录实际 kernel/tiling、HEAD 与适用条件。

大规模固定集合至少含 B1/Hq8/Hkv8/Sq=Sk4096/D128 f16 MHA，
B1/Hq8/Hkv2/Sq=Sk8192/D128 bf16 GQA，以及长 KV decode；小例只定位错误。
同时记录热数据重复与轮换工作集大于 L2 的条件，双方使用相同计时/缓存政策。
数值大例采用公开采样策略和全输出检查；采样通过不能称逐元素证明。

每条性能记录绑定源码、ASC/库 hash、编译选项、shape/dtype/布局、tile、舍入与
mask 契约、设备/SKU/频率、实际 kernel 名、warmup/重复数和 profiler 原始数据。
区分设备 task 时间、ACL stream 区间与 host 端到端时间，不混用提交间隙。

### 5.2 理论下界用于解释差距

在同一计时和工作量口径下，估计：

```text
t_lower ≥ max(F_native / P_cube,
              各类 Vector 工作量 / 对应吞吐率,
              HBM_bytes / BW_HBM,
              各实际搬运路径 bytes / BW_path,
              依赖关键路径延迟)
```

不同引擎可重叠，不直接相加。该下界的数值只有在吞吐/流量前提成立时有效。
dense FA 有效 matmul 工作量为 `4*B*Hq*Sq*Sk*D`；causal 按实际可见 pair 计，
原生 padding/chunk 的实际工作量另列。GM 指令请求字节不等于 HBM 流量，L2
命中、通信块和复用均影响后者；未知计数或持续带宽保持未知。

当前安装 910B3 配置为20 AIC/40 AIV、64 GiB HBM、192 MiB L2。
`20*1.8GHz*8192 = 294.912 TFLOP/s` 是附带每周期操作数与频率假设的估计，
不是实测持续峰值。区分标称峰值、条件估计和校准上限，按实际卡/频率更新。
pipe activity 比例会重叠，只能结合 timeline、等待位置和对照解释瓶颈。

设备运行前检查所有卡健康、进程、显存、利用率；单进程显式物理卡映射逻辑0。
不用 `/tmp` 锁，不 reset，不通过换卡绕过错误。文档修订本身不启动设备实验。

## 6. 验证与交付边界

1. AST/类型/布局/范围/输入契约及 xDSL verifier；这只验证结构和明确契约。
2. 符号依赖、初始化/代次、ownership、event 生命周期、控制域与收尾检查。
3. 从最终物化 IR 重新提取并检查义务；不信任 `verified=True` 或陈旧 witness。
4. 测试专用独立有限实例化：直接枚举字节访问冲突与硬件状态交错，覆盖故意
   删边、早 ready、早 credit、错 slot、错 guard、少参与者和遗漏 drain。
5. ASC 顺序/footprint 对照、CANN 离线编译、设备数值及固定规模性能证据。

第3层与 planner 共享目标契约及部分代码，不称完全独立证明；第4层不得复用
生产 `loop_dependencies` 来生成它要检验的正确答案。有界枚举注明范围与截断，
不外推任意 N；符号检查也只在已实现索引域、机器前提和输入契约下成立。

无法证明的动态别名/ownership 可要求可执行检查的输入契约或拒绝编译。跨组
交接、任意动态 task stream、任意 extern effects 和通用自动排步尚未实现，
不建立空接口宣称支持。每阶段都交付前后 IR、同步序列、原因和等待范围解释。

研究输入为工作区报告及只读源码，不是包依赖。本次核对 HEAD：TileLang
`6ba65d204c937fcb1c43b1a22b32979506595655`，ops-transformer
`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，CATLASS
`698379e0d4b1b94d12728e7181df2cf83a958e8f`，ops-nn
`64b41ad63a5073bc711ba03802498bfe8bdb77e5`。外层报告固定链接可能属于更早
提交；具体路径、硬件反例及契约依据见[源码依据](source-notes.md)。
