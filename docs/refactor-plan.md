# 重构方案：以真实多级流水逐步收敛

2026-10-15。依据[设计](design.md)与[语言规格](dsl-multistage.md)。
`ed87249` 保存接手时的父体组合实现；本次只重写方案、核对结果和补冷编译测量，
未继续实现或调参。整体重构仍未完成。

## 1. 当前基线和路线改变

| 已有资产 | 保留/尚缺 |
|---|---|
| AST 捕获、runtime 循环、物理原语、叶同步和最终检查 | 保留；不能以新目录结构替代已验证能力 |
| 共享 ActorBindings、根部 sequence、父×单 child | 保留，父体新组合已接入原 sequence 路径 |
| 父体 init/多个 child/use、相邻调用 scratch 保护 | 严格 frame 子集已有；无 frame 只读共享、跨 sibling 写 frame 仍拒绝 |
| 独立 AscendC FA、目标数学/存储/派发源码 | 保留；尚未从新 AST 编译，参考性能不代表专家目标 |
| legacy mixed、FA 及变体设备记录 | 作为差分资产；不计入新 AST 支持 |
| 真缓存、宏、一般层次 CV、连续 task/query 时钟 | 待实现 |

新父体的21项测试已通过；generated/独立 AscendC 共20条小规模设备结果全部
max_abs_error=0，含空父、空子及舍入反例。五个冷进程源码→ASC p50=0.8804s，
max=0.8824s，RSS=38320KiB。不是全量回归、任意规模证明或大例性能结论。

前版 P0–P5 的内容不全部作废，但不再按“先补完所有局部通用能力，再接 FA”推进。
新顺序按风险和可运行产物验收，每步都触达真实语言与最终代码：

```text
G0 基线封存与证据校准
 → G1 公共区域合同：替换现有局部形状分支
 → G2 架构切片：规则双轴 + 层次 CV + 必要事件共享
 → G3 真实 AST FA：嵌套/连续共用数学主体 + 同政策性能
 → G4 基础变体、专家 gap、缓存和产品收敛
 → G5 动态任务/间接访问/全局 phase
```

G1 不要求完成一般控制流；G2 不要求完整 softmax；G3 必须是真实 FA。每项通过
明确退出条件才登记。若需要第四套按顶层形状选择的 lowering/sync/codegen，先
修公共合同；若固定政策只能靠加 buffer、搬运或 drain 才通过，该项仍未完成。

旧编号映射：P0参考已基本具备；P1a剩余→G0/G1；P3a与P2a、P1b必要事件部分→G2；
P2b/P3b→G3；P4→G4；P5→G5。未迁移的 GDN snapshot 切片保留为 G2/G4 的非 FA
验证，不把整个 P1b 框架做完才允许 CV 进入。

## 2. G0：保存成果并校准证据

代码保存已完成。下一次开发先收尾，不重启上一轮已经完成的进程：

- 核对父体新增协议、两个 child、Scalar/通知反例及小设备记录；跑相关回归和 lint。
- 修正 `runtime_work` 聚合父体时从首个 once 继承的 `tasks_per_invocation=1`，
  用真实分 region 工作量表达；未修前不拿它做性能账本。
- 复核旧 state_sum/hierarchical_gemm/reduction_gemm ASC hash；不变则复用旧设备
  证据，有变化逐项解释并补相应验证，不盲目重跑全部设备任务。
- 父体固定大例 B4096/G32、outputs16/tiles128（X=1GiB），generated 与独立
  AscendC 完成数值和三批 profile。已有小例结果不能提前填写大例结论。

退出：保存点、支持矩阵、计时/工作量和运行 artifact 身份一致；不在此阶段扩
新语言或开始 shape/tile 搜索。原始运行数据始终放外层 runs。

## 3. G1：公共区域合同，用替换验证重构

目标只覆盖现有能力加真实 FA 必需的局部组合，不做一般整数求解器。

| 步骤 | 具体修改 | 验收产物 |
|---|---|---|
| 统一遍历 | 将 lower_source/sequence 的结构识别接到同一 region visitor；一 actor 一绑定环境 | 单流、once/stream、父体例走同入口，旧 ASC/数值不退化 |
| 统一边界 | loop/invocation/sequence 共用入口要求、末 writer/待退休 readers、条件、事件存活表示 | 同一最小例换合法嵌套位置不需新专用后端；删除重复边界逻辑 |
| 最小 ownership 扩展 | 从实际 effects 识别跨调用只读 GM；保留真实 group 域和逐 ref 范围检查 | 共享权重重复读通过，加入写/错误 frame/越界即拒绝 |
| 多参数地址 | 接规则 runtime stride/extent，不把所有 GM 压成单 frame | FA BNSD view 无预打包，两个 runtime 长度保持 SSA |
| 最终重检 | 从物化 IR 解除 live-in 假设，验证实际初始化和边界 | 漏首写、漏 reader、提前 credit 均有独立反例 |

只读共享不能由 `initialized=True` 推断。现有 invocation 的 `access_groups=1`
依赖已剥离 frame 的前提，放宽 frame 时必须逐 ref 重审，不能只删除 reject。
跨 sibling 写同 GM frame 在共同前沿完成前继续拒绝。

保留已有叶证明器，通过适配器逐步迁移；用同 source 的最终 ASC、义务与有限
oracle 对比，而非仅比较测试总数。不先重排目录，不一次性合并所有 Python 文件。

退出：真实 init/child/use、多 child 共享 scratch、只读 cache 通过；旧路径中
相同职责实际删除；常用例冷编译门槛通过。G1 后立即进入 G2，不扩一般 if/while。

## 4. G2：尽早让多级 CV 检验架构

三个切片逐步汇合为同一目标程序，使用 G1 的区域遍历和摘要，不另写 task-stream
或 attention 后端。各切片失败时先最小化，不在大程序上反复盲等。

### G2a：规则双轴及代次

使用已有 copy/add 或 matmul 原语，uniform K，`task=query*K+kv`；query 状态初始化/
导出、task 双槽、child 独立环同时存在。runtime K 不改成 constexpr。

覆盖 queries=0/1/2/17、K=0/1/2/7/长K、query/task 槽不同、K 与环互素。保留一个
连续 device task 循环，不在 query 边界重新 seed/drain；K0 路径有定义且无除零。
只对 slot 必需的静态余数分类，证明窗口不得枚举 K 或 queries×K。

退出：同 binary 多长度；常量1024/1048576的模板/图状态/查询数不随次数增长；
前后 IR 和 ASC 保留正确父/子坐标与 first/last。无法达到时先修关系表示。

### G2b：层次 CV 与实际资源

1 AIC+2 AIV，外层四 Stage，至少一个 Cube Stage 含保留的子 pipeline，一个
Vector Stage 含行分块；混合前后普通工作、共享 L0C、实际 GM handoff。切片可以
不含 softmax，但必须有真实各级读写，不能将 Stage 当黑盒 done。

纳入双 AIV 分片/quorum、GM 可见性、每级最后 reader 和 cross-core gate。事件
按整个组合寿命分配，必要时在已证不重叠的 child 间复用 ID。先实现确定性有限
分配，不追求最优着色；真实不足与搜索未决分别报告。

必须有两种容量例：各 child 独立合法、机械独占会超8但寿命允许共用；实际
并发寿命超8而必须拒绝。不能提高 buffer、缩块或添加 child drain 换通过。

### G2c：合并检查点

同一个双 runtime 轴、连续 task 程序具备外层 CV、内层 native/Vector 流水，
跨 query 复用 cache/state。加入非 FA 的 FP32 carry/低精度 snapshot 或原地
state-cycle 最小例，保证接口不围绕 Score/P/PV 特化。

退出必须同时具备：

- 源语法可审查，各 Stage/actor/child 的 generation 明确，无延迟 descriptor 手工代码。
- 前后 IR、真实同步序列和最终 ASC；漏任一贡献、提前 ready/release、ack 当退休、
  下一调用重用未消费 event、wait 挡住必要派发等负例。
- 单槽 L0 的 MAD-before-next-load 合同保持；数据/数值及原生搬运不变。
- CANN 编译、小设备正确性、固定大工作量与手写 AscendC/profile 对照。
- 编译规模不依赖 runtime 两轴，成本可定位；合并例目标按 FA 的5s/10s门槛。

这是架构出口。纯 source verifier、局部四阶段模拟或 legacy CV 运行均不能替代。

## 5. G3：接通完整 FA，而非再建一套框架

复用 [attention target](attention-target.md) 的明确数学主体及
[独立 AscendC](attention-reference.md)。先 nested 再 continuous，二者数学代码
共用；G2已解决的循环/组合/资源不可在这里重新特判。

最小宏与必要原语在此接通：实际 refs/scratch/constexpr 参数、卫生化内联、来源
链；maximum、packed reduction、broadcast 等必须至少有非 FA 语义例。宏不注册
可信同步摘要，不直接 emit ASC，不隐藏 scratch 或调用整算子 CANN。

完整 forward 包含 FP32 m/l/O、真实 softmax、p32 denominator、alpha 延迟消费、
P cast、PV 按 CK 顺序累计、最终指定 reciprocal+mul/cast。不得以乘法代 softmax，
不得为了迁移改逻辑 BK、reduce 树或舍入边界。QK/PV 的逐条数值合同见目标文档。

退出：源→IR→同步→ASC→CANN→小例完整 oracle→固定4K/8K大例，nested 和 continuous
都可运行；通信BK1024/nativeCK128/VectorVR8保持，workspace 随固定 slots/groups
有界，没有输入 packing。完整冷编译≤5s/10s；同政策 AscendC 三批 profile 完成。

性能验收先看最终 ASC：原生指令/累计次序、每级搬运与重复加载、buffer/ID峰值、
GM交接、release端点、gate位置、跨query填排、最终drain。再看profile是否发生
预期重叠。独立参考本身落后时，登记专家 gap，不能宣布整体性能目标已完成。

## 6. G4：基础变体、性能差距和产品收敛

基础变体按独立合同验收：f16/bf16、MHA/GQA/MQA、prefill/decode；随后 causal、
非方形绝对位置、tails、空序列和全mask。每项定义 valid 域、数值 identity、
通知参与者及错误输入，不将 legacy 变体自动标为新 AST 支持。

专家 gap 不用参数搜索掩盖：对照 ops-transformer arch22/CATLASS 的代码组织，
安装 ACLNN 的同 shape/dtype/layout 结果，以及既有专家执行政策。区分：

| 差距类型 | 处理 |
|---|---|
| 编译器多余等待、搬运、错误 gate、冗余派发 | 缩成反例并修复公共编译逻辑，保持政策 |
| reference 政策/算法/数值边界不同 | 明列差异；相同政策保真结论不外推专家性能 |
| 硬件利用率、带宽、Vector或Scalar瓶颈 | 结合工作量、时间线及硬件上界验证假设 |
| 计时/cache/profile条件不同 | 统一口径后比较，不挑最好样本 |

当前算子调参仍暂停。若未复现专家政策/重叠或 gap 未解释，性能目标保持未完成；
不以“架构无损”或与慢参考一致作为退出依据。

真缓存独立实施并测试失效：源码/宏传递内容、constexpr、目标/native合同、
数值选项、编译器/CANN身份变化必须 miss，runtime长度变化复用 binary，launch
合同每次检查。冷路径先达标，命中≤100ms；属性访问不可充当cache benchmark。

完成独立 wheel 安装：没有 examples/reference/原工作区仍可编译用户源码。CLI与
benchmark保持通用。旧 finite 路径只保留显式legacy/有限oracle用途，生产AST
不静默fallback；删除旧实现前核对替代证据，产品支持矩阵统一更新。

## 7. G5：动态任务与其他算子合同

varlen → paged → sparse → dynamic persistent，每步增加 descriptor读取/捕获、
间接footprint、真实task generation、ownership及空任务参与者。动态grouped
matmul、MoE routing/finalize、量化/FFN、GDN分别提供反例，不要求立即建全算子库。

scatter唯一性/atomic、状态owner冲突、split-KV全局合并、全局phase与通信完成
分别处理。无法证明的动态别名要求可验证合同或拒绝，不用任意barrier掩盖。
backward/dropout/RNG另立计划，不宣称预留接口等于支持。

## 8. 编译成本验收矩阵

每次记录机器/CPU、Python/xDSL、源码/包/ASC hash、分阶段时间、RSS、模板/访问/
冲突数量、各周期、最大/总证明域、边/查询/重建数、摘要pieces与候选数。

| 轴 | 固定实验与判断 |
|---|---|
| runtime长度 | 同binary运行0/1/2/4096/1048576，编译键与ASC不取决于值 |
| 常量界 | 保留循环的1024与1048576，允许常量文本不同，结构计数不按长度增长 |
| 静态规模 | 模板4/8/16/32、region层数、sibling与pieces分别增加，定位增长主项 |
| 别名密度 | 无交集/局部交集/全交集分开；H二次时如实记录 |
| 周期 | 独立29/31环与真实耦合环，避免全局LCM及错误拆分 |
| 组合 | parent×child×sibling，不按执行次数复制摘要 |
| 资源与失败 | event容量、guard爆炸、取消/预算；错误后无成功缓存和残余状态 |
| 冷时间 | ≥5独立进程源→ASC；常用1s/2s、完整FA及合并CV5s/10s（p50/max） |

加速顺序固定：消除长度进入分析→限制真实耦合窗口→摘要与等价查询复用→减少
候选重建→profile优化热点→必要时原生加速。不得省略最终检查。外部60s截止前
内部预算应给phase/计数；超预算保存反例，不反复等待或提高上限。

## 9. 大规模性能矩阵与计时

| 类别 | 固定配置 |
|---|---|
| Vector/状态 | 至少2^24个f32；已有state_sum及父体X=1GiB |
| normalization | 8192×4096 f32 |
| 父子Cube | 20组×16outputs×256panels，M32/N64/BK64 f16，输入960MiB |
| Dense FA | B1/H8/S4096/D128 f16；主例B1/H32/S8192/D128 bf16 |
| GQA/MQA | B1/Hq32/Hkv8或1/Sq=Sk8192/D128 bf16 |
| Decode | B1/Hq32/Hkv8或1/Sq1/Sk32768/D128，单列并行度与填排 |

D64/D128、dense/causal、prefill/decode分别核算。不为增加数字而无界扩shape；
必要时追加能分辨缓存/带宽/并行度的固定例，说明假设。小例做全量数值，大例
全输出finite/guard并公开误差检查范围；资源允许全量oracle，采样不称全量正确。

每版本三批独立测量，交替顺序，至少warmup5/repeats30，报告p50/p90/批间波动；
profiling与非profiling分开，host、ACL stream和device task时间不混用。热缓存
及超过实机L2的buffer轮换分别测；Sq大不保证输入工作集超L2。

记录源码/生成ASC/库/CSV hash、工具链及flags、shape/dtype/layout/数值合同、
物理卡/SKU/活跃核/频率依据、workspace和计时范围。AscendC与generated条件
相同，ACLNN明确workspace准备/布局变换是否计入。超过5%触发归因；噪声覆盖
则未决，不视为允许的性能损失。

硬件分析同时交付有效/native FLOPs、Vector指令需求、各级搬运和GM请求字节；
HBM实际流量另记，不能混作带宽。共享资源需求汇总，计算条件下界及关键路径，
用timeline/等待位置验证；活动率可重叠，不单看一个占比断言瓶颈。未知持续
带宽/Vector峰值明确留空，不造数。实际下界方法见总体设计。

设备前检查所有卡健康、进程、显存和利用率，选健康空闲卡；每进程显式物理→
逻辑0映射。不reset、不用/tmp锁、不换卡规避设备错误。

## 10. 每一步的交付和复现

每个切片一起交付源码example、最小正/负例、同步前后IR、实际Set/Wait序列、
ASC、阶段成本和明确支持/拒绝表；设备相关切片再加CANN/数值/性能证据。说明
为何同步足够，以及没有扩大到无关reader或整child完成。

工作区根目录命令（设备运行另按全卡检查选择物理卡）：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_sequence_invocations.py compiler/tests/test_sequence_sync.py
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_grouped_state_sum.py --entry grouped_state_sum --trials 5 --output runs/refactor/parent-sequence/cold-next
.venv/bin/python compiler/examples/run_grouped_state.py --output runs/refactor/parent-sequence --build
.venv/bin/python compiler/examples/attention_target.py --output runs/refactor/attention-target
```

最后一条只生成目标源码/有限派发表，不能当FA编译成功。下一次实现先完成G0
遗留收尾，再用G1替换重复职责，紧接G2，不再追加无关局部算子。只有compiler
产品文件进入独立Git；环境、缓存、产物、运行数据在外层。不恢复根Git、不推送。
