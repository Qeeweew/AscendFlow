# xDSL / Ascend 910B 编译器设计

2026-10-15 重审。代码保存点 `ed87249`，前一已提交基线 `0798819`。
本文规定目标与取舍；实现事实见[验证记录](refactor-progress.md)。配套文档为
[DSL 规格](dsl-multistage.md)和[重构方案](refactor-plan.md)。

## 1. 目标与本次修正

提供 TileLang 风格的 Python DSL，直接表达计算、存储、核内流水和 CV 多阶段
流水。用户固定专家政策，编译器自动完成依赖、buffer 复用和事件协议；未来自动
排步也消费同一套契约。采用 xDSL 0.73.0 → ASC → CANN 9.1 / dav-2201。

每个功能同时接受三项检验：语言能否自然表达；冷编译成本是否可控；最终机器
程序是否保留所需搬运、缓冲、数值次序和重叠。不能先建通用分析框架，最后再补
语言和性能。前版方向基本正确，但实施约束不足：

| 当前源码或证据 | 本版调整 |
|---|---|
| `lower_source` 按顶层形状转入 sequence/invocation，边界分别解释 | 用相同区域合同逐步替换形状分流，每步删除实际重复职责 |
| 父体组合已有新实现，旧文档仍写未接通 | 分开保存点、已验证子集和目标，保留现有成果 |
| `Graph._domain` 仍物化有限证明状态，边界查询可放大 | 同时约束窗口、图规模、查询/重建数；保留循环只是必要条件 |
| 局部例持续增加，多级 CV 和真实 AST FA 仍未接通 | 下一个架构检查点必须包含层次 CV 与两个 runtime 轴 |
| FA fixture 的单槽 L0 曾有错误 issue 顺序 | fixture、手写参考也要审查，不能当正确性公理 |
| 同政策参考可以正确但很慢 | 分别验收编译器保真和专家性能，前者通过不能代替后者 |

保留已验证的原语契约、叶证明器、反例、代码生成和 runner。不重做整个后端，
不迁移 TileLang 全部 pass，不引入 MLIR 工程。本次方案重写不恢复算子调参。

## 2. 语言、控制和存储

公开语言采用 `prim_func / Tensor / Kernel / serial / Pipelined / Stage / Cube /
Vector`。设备函数解析为 AST，不通过执行函数体生成每个任务。具体示例见 DSL
规格，不恢复 `cube.task/vector.task`。

| 概念 | 用户表达 | 编译器责任 |
|---|---|---|
| 数学与状态 | 运算顺序、精度、舍入、carry、snapshot | 保持逻辑版本和数值边界 |
| 存储 | 空间、layout、capacity、view、显式 slots | provenance、原生字节访问、容量/别名 |
| actor | AIC/AIV 与行分片 | 合法放置、参与者和 ownership |
| engine | 由 typed 原语和路线确定 | S/V/MTE1/MTE2/MTE3/M/FIX 的完成契约 |
| 流水政策 | stage offset、各 actor 的 issue order、嵌套/连续流 | 位移整代数据/descriptor/guard，生成填排 |
| 同步 | 普通 DSL 不写 flags/credits | ready、最后读取、release、配对、event 复用、收尾 |

Stage 不是执行单元、共同物理时钟、buffer 数或完成屏障。Cube 是 AIC 作用域，
内部还有 MTE2/MTE1/M/FIX。AIC 与两个 AIV 独立推进，cross-core Wait 会阻挡等待
actor 的后续派发，其位置直接影响重叠。

长度、offset、有效域默认 runtime SSA；tile、片上容量、layout、dtype 和固定
政策为 constexpr。常量循环界也默认保留，只有显式且受预算的 unroll 才展开。
声明为 runtime 的参数不因某次调用可见其值而特化。

先提供可控原语，重复数学块通过受限 AST 宏组合。宏在 examples/用户库中，
卫生化展开并保留来源，内部普通循环保留；不得成为 opaque task 或隐藏同步、
DMA、scratch。后续 convenience API 也必须降到相同物理合同。

局部流水和 CV 流水可以嵌套：通信块、Cube 原生块、Vector 行块分别选择。
BQ128/BK1024/CK128/VR8 是当前表达能力 fixture，不是最优参数承诺。普通嵌套
保持词法派发：actor 派发完 child 控制后继续父工作；child 退出不意味着异步
退休，不隐式插入 drain，也不把兄弟 stage 自动穿插进 child。

跨 query 连续流由源码明确给出 `task=query*K+kv`，保留 query/task/child 三种
代次，不能靠把父循环展开实现。首批只支持 uniform K 和规则分区；K=0 有独立
算法路径，不执行除零。嵌套与连续组织共用数学主体和同步合同，只有嵌套通过
不能宣称复现专家连续流水。

## 3. 程序表示与区域组合

```text
受限 Python AST + constexpr + examples 宏
  → xDSL source：refs/views、arith/scf、actor、pipeline/stage、数值语义
  → xDSL physical：保留控制、原生任务、实际访问、派发、owner/代次
  → 同步义务 → 逻辑协议 → 事件资源 → 物化同步
  → 最终 IR 重提取检查 → 不可变产物 → ASC → CANN
```

xDSL 是可改写程序的唯一真相；分析快照是带 revision 的不可变索引，不是另一棵
可变程序树。同步操作加入物理 IR，无需先建额外调度运行时或十几层 dialect。

目标遍历按原语、顺序、循环、受限分支、actor 组合，program shape 不选择另一套
语义。每 actor 共用 allocation/scalar/frame 绑定，native lowering 只处理原语。
将现有 loop/invocation/sequence 接到共同合同，保留叶分析；不是文件改名后仍
重复解释初始化、边界或事件。

区域摘要是带前置条件的传递关系，保留真实 root/byte piece 的入口要求、末 writer、
所有未退休 reader，以及初始化、owner/代次、guard、零次恒等路径、可投影的完成
关系和未消费通知。`A;B` 连接 A 的实际前沿与 B 的首访问；循环验证入口、递推、
出口、零次；分支按可达 guard 合流。不能压成一个 child-done 节点。

重复摘要和条件关系共享不可变 DAG；摘要膨胀触发预算和具体诊断。不得扩大到
整 allocation 后加 barrier 冒充组合成功。一般分支、动态别名、动态 ownership
只有在合同实现后才支持，xDSL 能表示不等于后端能证明。

## 4. 自动同步合同

每个原生访问记录 root、布局后的实际字节集合、R/W/RW、actor/engine、activation、
generation、开始与退休端点。padding、DMA 边界、scratch、Scalar metadata 和 SPR
同样计入。借用值标识写版本，不是隐式 snapshot。

RAW 连接写完成与读开始；WAR 连接所有旧 reader 退休与新写；WAW 连接相关写入。
SSA use-def、tensor 名、函数返回、控制出口不能替代异步完成证明。ResultReady、
ReadRetired、NotificationConsumed、slot release、final drain 分别建模，早退休
必须来自可信 native 契约。

双槽输入在 Vector 最后读完后可供 MTE2 复用，无需等输出 MTE3；输出槽自身必须
等 MTE3 读退休。L1/L0A/B/L0C 分别由 MTE1/M/FIX 最后 reader 归还，不能以整个
计算结束统一替代。多 reader 要全部覆盖，最后派发者不一定最后退休。

本地 HardEvent 每核每有向 bank 有八 ID，Set/Wait 是二值协议。证明旧 Wait 消费
先于同 ID 下一 Set 生效；buffer 环宽、stage 和 event 宽度独立。先生成逻辑
协议，再按实际寿命冲突分配 ID；不同作用域不能无条件独占或共享。确定性有限
搜索失败区分已证超容量与搜索未决，不增加 buffer 或加 drain 绕过。

dav-2201 mixed group 为 1 AIC+2 AIV，各 UB 私有。CV 使用合法 GM 路线，ready
覆盖所需范围全部 writer，release 覆盖全部 reader；双 AIV quorum、通知计数、
reverse ack、可见性分别验证，ack 不等于 read-retired。不套用 950 的直达路径
或 mutex。相同 pipe 的隐式完成也仅限核实的硬件契约。

最终 checker 从物化 IR 重提取义务，不接受 planner 的已证明标记；允许共享
基础分析，不能称为两套独立证明。另用有限指令/逐字节 oracle、协议变异反例
验证实现。有限检查不冒充任意规模证明。generic DCE/CSE/代码移动必须尊重
异步 effects、ownership 和 SPR，相关改写后重新检查。

## 5. 编译复杂度与性能预算

N 为 runtime 执行次数，S 为静态语法/原生模板规模，A 为访问 pieces，H 为冲突
数。每个真实耦合分量 c 有静态周期 P、证明窗口 W、图点/边 V/E、查询数 Q：

```text
Tcompile ≈ O(S) + 访问索引(A,H)
           + Σc [构图(Vc,Ec) + Qc×查询成本(Vc,Ec,Wc) + 失效重建]
           + 最终检查 + 序列化/发射
```

N 不得进入模板、窗口、候选数量或编译键；整数编码位数另计。访问索引目标避免
无冲突的全对扫描，密集别名 H 仍可能 O(A²)；普通遍历 O(V+E)，当前势函数和
域推理另计。不能宣称全编译线性，也不能只数 IR 操作就声称长度无关。

| 风险 | 对策与实验 |
|---|---|
| 大常量界进入头尾距离 | head/tail 相对坐标；1024/1048576 比较内部状态/查询数 |
| 无关环导致周期乘积 | 分解真实耦合；共享 bank/gate 的耦合不能漏掉 |
| 两 runtime 轴乘积展开 | 保留关系域，只分必要静态余数类，禁止枚举 K |
| 父子摘要与 guard 爆炸 | 关系 DAG 共享；pieces/候选/深度预算和增长曲线 |
| 候选反复重建闭包和 module verify | 每阶段一次索引；按依赖失效，等价查询规范化 |
| cache 掩盖冷路径慢 | 属性访问、真 cache hit、源码冷编译分别测 |

新增同步改变可达性，不能跨 revision 复用负查询或失效结果；最终检查独立重
提取。先修规模与重复工作，再 profile 热点，结构稳定后才考虑原生加速。并行
可提高独立编译吞吐，不能修复单次复杂度。

五个冷进程源码→ASC：常用例 p50≤1s/max≤2s，真实 FA p50≤5s/max≤10s。这是
工程门槛，未达即登记未完成。Python 启动/import、CANN、加载和 launch 单列；
记录 RSS、最大单域/总状态、查询/重建数、源码/包 hash。外部默认 60s 截止配合
内部结构预算与取消；耗尽即诊断，不提高预算后继续盲等。

真缓存键包含源码和宏传递内容、constexpr、目标、数值政策、native 契约、编译器
版本/选项，CANN 产物另含工具链。runtime 值只做 launch 合同检查。miss 完整验证，
hit 检查产物完整性；命中目标≤100ms，当前不可变产物属性读取不算此能力。

## 6. 算子性能的三层证据

算子、FA 变体、oracle、手写 AscendC、ACLNN 适配器在 examples。核心 wheel 不
依赖 examples/reference，不允许识别 FA 名字的 lowering/codegen 分支。

1. **编译器保真**：固定政策对照独立手写 AscendC；最终 ASC 对比指令、搬运字节、
   布局、buffer、累计顺序、release、gate 和填排，然后做设备计时。
2. **专家性能**：对照 ops-transformer/CATLASS 具体架构路径、既有专家政策和
   安装 ACLNN。参考本身慢时，“与参考一致”不能结束性能目标。
3. **硬件解释**：将 ASC 位置、profile、资源需求和可检验瓶颈假设对应起来。
   多余同步属于编译器缺陷，通信/算法/数值政策差异分别记录。

先看最终 ASC；CANN 仍可能改写，需设备证据，必要时看 CANN 产物。禁止额外
packing、扩大 buffer、per-query drain、改变逻辑 BK 或静默调用 ACLNN。固定
政策比较属于重构验收，算子调参继续暂停。

理论下界按真实 SKU、活跃核、频率、dtype 与工作量估算：

```text
T ≥ max(每个共享资源的总需求 / 吞吐上界, 数据依赖与派发关键路径下界)
```

分别核算 Cube、各类 Vector、Scalar、MTE/FIX/GM 与填排；共享资源需求先相加，
可并行资源不盲目相加。有效/native-padding FLOPs 分开；dense FA 两次 matmul
有效 FLOPs=`4*B*Hq*Sq*Sk*D`，causal 按真实可见 pairs 计。GM 请求不等于 HBM
字节，热 L2 与大于实机 L2 的轮换工作集分别测。

已有记录为 910B3、20 AIC/40 AIV、192MiB L2。若假设 1.8GHz、每 AIC 每周期
一个 16³ b16 FMA 块、FMA 计2 ops，条件 Cube 上界为294.912TFLOP/s，
B1/H32/S8192/D128 dense 的 Cube-only 下界约3.728ms。这不含 Vector/搬运，
不是全 kernel 可达时延；频率与吞吐假设须标来源。未知带宽/Vector 吞吐保留
参数化估计，不把无单位 platform rate 当 GB/s。

大规模矩阵、三批交替测量和误差口径见重构方案。5% 是差异调查线，不是可接受
性能损失承诺；噪声覆盖则记未决。最新开源 checkout 与安装 CANN 分别记身份。

### 尚需实证的设计风险

- 区域摘要可能保留不足而丢早 release，也可能过大而拖慢组合。G2必须同时通过
  多reader反例、真实事件容量例和规模实验；不能只证明保守正确就接受性能损失。
- uniform商余和多参数stride的受限关系域仍待实现。若分类随K增长，应修改关系
  表示，不能把K改成constexpr、展开任务或强制改变输入layout。
- 对任意形状都追求最少同步/最优ID分配不在首轮范围；固定目标政策必须通过，
  其他搜索耗尽明确诊断。一个分配算法失败不等于硬件不可实现。
- 当前FA政策没有专家性能保证。G3验收保真后仍须G4解释差距；如政策本身不足，
  先用代码/profile明确原因，再讨论调整，不能将编译器支持范围写成性能承诺。

## 7. 实现边界与来源

| 保存点 ed87249 | 状态 |
|---|---|
| 单 local 流、runtime prefix、归约/Scalar、L0/Cube/FIX、父×单 child | 已有各自子集编译/设备记录 |
| 根部 once/stream、真实 init/use、共享只读 GM | 已贯通，state_sum 有1GiB独立 profile |
| serial 父体 init/多个 child/use、跨调用 local scratch | 新严格 frame 子集；父组合21项测试、小设备对照通过 |
| 该新例冷编译 | 五次源码→ASC p50 0.8804s/max 0.8824s，RSS 38320KiB；不是 FA 结果 |
| 父体无 frame 共享只读 GM、跨 sibling 写同 frame | 仍拒绝；不能以 initialized 标志绕过 |
| 跨 region ID 复用、一般层次 CV、规则 task/query 时钟 | 未接通；legacy mixed 不等于新 AST 支持 |
| AST 宏、新 AST 完整 FA、真编译缓存 | 待实现；fixture 和独立参考不算迁移完成 |

此表保留设计保存点的状态。后续G1已统一源码区域入口，并接通
[跨父调用共享只读GM](shared-invocation-inputs.md)；最新结果见验证记录。

保存点不代表本轮全量回归或新例大性能验收，父体 aggregate 工作量字段也有
待清理项。更广反例持续覆盖 normalization/RoPE 的 scratch/多 reader、MoE 的
间接 ownership/早 release、GMM/量化/FFN 的空组与联合发布、GDN carry/snapshot、
paged/sparse/persistent attention 的 descriptor 和参与者。全局 phase、split-KV、
backward/dropout/RNG、通信各自立合同，没有空接口式支持。

本轮只读核对、未 pull，ops-transformer/CATLASS/ops-nn 工作树干净：
TileLang `6ba65d204c937fcb1c43b1a22b32979506595655`；ops-transformer
`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`；CATLASS
`698379e0d4b1b94d12728e7181df2cf83a958e8f`；ops-nn
`64b41ad63a5073bc711ba03802498bfe8bdb77e5`。

依据：[TileLang契约](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[FA阶段](../../reference/tilelang/examples/ascend/flash_attention/910b.md)、
[arch22 FA](../../reference/ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h)、
[CATLASS mainloop](../../reference/catlass/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)、
[MoE finalize](../../reference/ops-transformer/moe/moe_finalize_routing/op_kernel/moe_finalize_routing_fp_db.h)。
arch22 Process 可见 Vec1 先于 Vec2，当前 update-before-softmax fixture 不能称为
逐条复刻；MoE 在 Add 后 release 输入也不意味着其全部 barriers 应照搬。
外层[源码审查](../../docs/xdsl-npu-operator-source-audit.md)的固定提交是历史溯源，
参考源码不成为产品构建依赖。
