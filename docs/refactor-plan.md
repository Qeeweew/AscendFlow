# 下一部分重构：直接同步生成与符号边界摘要

2026-10-15 再审。工作保存点 `61c5d06`，不是重构完成点。
状态以[实现与验证](refactor-progress.md)为准，算法约束见[设计](design.md)。

## 本次决定

停止把“逐义务搜索改成批量传播”当作完整方案。当前叶生成器仍搜索候选路线，
层次传播仍复制大量带条件状态；二者必须分别处理。先修生成器的选择依据，再
修当前主要耗时的层次表示。算子调参继续暂停，不扩大公开 examples。

保留 AST/物理访问/原生契约/emitter 和独立反例。`native_order` 共用规则、
生产路径移除旧 `Graph.path` 是有用基础；heap/bitset 仅作为实现工具。既有
首尾窗口完成图保留作迁移对照，不继续用它承接一般 nested CV。

## 1. 重构同步生成与组合

### 1a. 从具体访问义务直接生成协议

替换 `_proposals → 全候选图 → ProtocolSelection.route`：

1. 按真实消费者、时钟和激活条件收集其全部物理访问义务。
2. 用 native 顺序及已选协议的完成状态扣除已覆盖部分。
3. 对缺口直接确定源 capture、目标 acquire 和必要参与者；同一合法 capture
   内合并。只有已选协议更新完成状态，不为所有可能同步建指令节点。
4. 处理选中协议的二值 event/counter 资源义务；物化之后从实际 IR 独立重提取。

性能约束优先于 flag 条数：不得为了合并同步把最后 reader 的 release 延到
无关工作之后；已有必要路径覆盖则可以省去直接通知。按 RAW/WAR/WAW 全局
排序不是通用算法，source/consumer 的实际位置与 guard 必须进入分组。

先落关系级早/晚 capture 反例，再接已有双缓冲 Vector、Cube/Fixpipe 和局部
CV 回归。使用现有源码与 runner；每个 sync 变化对照最终 ASC。验证未选协议
不能提供事实，Scalar gate不能追溯阻塞，两个 AIV不能借用彼此的发布。

退出条件：生产生成器没有候选路线搜索、试删及全候选指令图；正确的 ready/
WAR/复用/drain 保留。仅禁用 `Graph.path` 或降低冷编译时间不足以通过本步。
本步首先修生成策略，不预先声称能消除已测得的层次传播热点。

### 1b. 缩小层次状态，消除逐事实的条件包含扫描

对现有 frame 和 hierarchical GEMM，按相同硬件语义构造一次 child 边界转移：
入口前沿、实际暴露的 capture、未退休 reader、活跃 event，保持其符号时钟。
父层组合接口，不因 source、调用或父层查询复制 child 的首尾指令窗口。
条件规范化后共享相同转移，只有实际相交的条件才细分；不取独立环的全局 LCM。

先完成当前规则环的入口/归纳/出口和N=0合同。不能删除最终 verifier，不能用
缓存返回旧证明，也不能把一般循环问题交给无界 worklist。摘要的具体表示仍
需实现检验：若必须依赖整体 child drain 或输入长度展开，应在这个反例上修正
设计，不扩充一个新的 CV 专用查询器。

最先验收已有 frame 的精确关系：

- N=0：真实 snapshot-ready；N>0：沿已有 MTE3→V→MTE2 路径覆盖。
- slot0：偶数 N 等 N−2，奇数等 N−1；最终 state 写仍覆盖真实末次更新。
- 跨调用：分别检查 buffer 最后读取、内部 event 的旧 Wait→新 Set。

同时保留两个 reader、Set后续工作、互斥条件、错误代次的反例。必须通过实际
生成器与最终 IR 检查两条路径；单独的摘要原型不算实现完成。

### 1c. 成本与迁移验收

基线数据：6模板 GEMM 的最终调用图1143节点、10079条节点/guard事实；整个
编译115935次条件包含比较。证据在外层 `runs/refactor/algorithm-review/`。
这是有 instrumentation 的工作量统计，不是冷编译性能统计。

验收顺序固定为：

1. 关系级与破坏 IR 的定向反例，分钟级全量回归不参与每轮算法试验。
2. N=1024/1048576结构相同；独立区域的成本相加；静态模板、环宽、条件和深度
   分别递增。统计实际节点/边界接口/guard片段/更新/条件比较，不能只报耗时。
3. 11例 ASC 对照，尤其早 release、buffer 数与搬运。改变等待范围必须说明依据。
4. 代码稳定后五冷进程 frame/GEMM；完成一个可用迁移后再运行一次完整回归。

叶、region、call 采用共同语义和生成原则，不要求所有问题都被压成一个大图。
保存点中尚未广泛验证的批量修改不能使用历史全量测试的结果冒充当前验证。

## 第1步与真实 CV 的衔接

用 FA 的 QK/PV、三个 GM 环、两个 AIV贡献和早 import release 检验摘要接口。
时钟保留 task/query/child 身份，支持新 query QK 早于旧 query PV 的偏序；
不能由串行 region 已通过推断 CV 已通过。语言仍使用 TileLang 风格的
`Pipelined/Stage/Cube/Vector`，不把 event或完成 token 推给用户维护。

## 2. 在真实 FA 上接通 nested 多级 CV

直接使用 [目标数学主体](design/attention_target.py.in)，不以无 softmax 的 CV
样例替代。按其实际编译阻塞依次完成下列合同，均进入第 1 步公共模型：

- BNSD 多参数 runtime 地址、只读 Q/K/V 共享、group/lane 分区和精确 views。
- AIC/AIV 外层四阶段，QK/PV 内部 retained pipeline，Vector 行块 pipeline。
- 共享 L0C、L0 单槽最后读取、三个 GM 交接环、双 AIV 完整贡献及内存可见性。
- 所需 typed maximum/reduce/broadcast/fill/copy；scratch、数值次序和 effects 可见。
- 分支 first/last 与零工作 owner 的真实初始化、event 配对、最终 drain。

初始政策保持 BQ128/BK1024/CK128/D128/VR8；AIC qk0→pv1，AIV update2→softmax1。
它是可审查候选政策，尚不等于 ops-transformer 最佳政策。单槽 L0 的
load/MAD/operand(/FIX) 次序保持；不扩大环、不减小通信块、不加输入 packing。

完整 online softmax 保持 FP32 m/l/O，p32 denominator、P 显式 RN、PV 的 CK
累加顺序、独立 rescale/add、最终 reciprocal+mul/cast。共用数学块若需要宏，
只实现受限 AST 内联，不隐藏异步工作或可信同步摘要。

退出要求：新 AST → 前后 IR → 自动同步 → ASC → CANN；小例全输出数值、输入/
guard 检查通过；大例固定政策与独立 AscendC 对照。最终 ASC 必须保留三层循环、
原有 buffer/搬运和早 release；child 控制退出不插入 drain。

## 3. 同一数学主体接通 continuous，并完成性能验收

运行时 `task=query*K+kv`、商余 descriptor 和 query/task/child 坐标各自保留。
K>0 合同及零 K 分支独立处理；不枚举 K 个余数，不把长度改成 constexpr。
新 query 的 QK 必须能在旧 query PV 之前派发；不复用串行 CallProof 的词法假设。

事件按整个程序的实际寿命分配，保留有条件的跨 region ID 共享。覆盖“独占会
超过八 ID、寿命允许共用”和“真实并发超容量”两个资源反例；搜索未决不能
谎报硬件不可实现。去掉冗余同步必须有关系覆盖证据，不承诺全局最优着色。

退出要求：nested/continuous 共用算法主体和 native 合同，同 binary 多长度；
不按 query 边界重新 seed/drain；与手写参考逐条解释 ASC 和设备 timeline 差异。
同步之前、之后、删边理由、未覆盖诊断与冷编译成本一起交付，不能只报告测试数。

## 编译规模和耗时门槛

| 维度 | 必须检查 |
|---|---|
| runtime 长度 | 0/1/2/4096/1048576 不进入编译键；合法值复用 binary |
| 常量循环界 | 1024 与 1048576 的模板/关系/查询量不按次数增长 |
| 静态规模 | 模板、region 深度、sibling、访问 pieces 分别递增，定位主项 |
| 周期 | 独立互素环不取全局 LCM；真实耦合不能错误拆开 |
| 失败与资源 | 预算/取消给 phase 与计数，不生成成功产物或静默 fallback |
| 冷编译 | 五独立进程；常用 p50≤1s/max≤2s，完整 FA p50≤5s/max≤10s |

记录 source/package/ASC hash、Python/xDSL、CPU、RSS、阶段耗时与结构计数。
import/CANN/launch 单列；属性读取不是编译缓存。先修长度相关复杂度和重复查询，
再依据 profile 优化热点。缓存、原生加速和并行编译不能作为规模失控的遮盖。

## 大规模设备性能门槛

| 用途 | 固定配置 |
|---|---|
| 保留的 Vector/状态回归 | X=1GiB；normalization 8192×4096 f32 |
| 保留的 Cube 回归 | 20组×16 outputs×256 panels，M32/N64/BK64，输入960MiB |
| FA 主例 | B1/H8/Sq=Sk4096/D128 f16；B1/H32/Sq=Sk8192/D128 bf16 |
| 后续 GQA/MQA | Hq32/Hkv8或1，Sq=Sk8192，D128 |
| 后续 decode | Hq32/Hkv8或1，Sq1/Sk32768，D128；单列低并行度 |

小例全输出数值；大例全部 finite/guard，数值检查覆盖范围明确记录，采样不能
称全量。三批交替测量，warmup≥5/repeats≥30；p50/p90、批间波动、profiling 与
非profiling分别报告。固定大规模不等于超过 L2，另核算工作集与必要的 buffer 轮换。

每项对比有三份证据：生成 ASC 与独立 AscendC 的排步/搬运/缓冲；CANN 开源
实现及安装 ACLNN 的相同 shape/dtype/layout；profile 与硬件资源需求分析。
先归因编译器多余同步/搬运，再区分政策和数值差异。超过 5% 是调查线，绝非
允许损失；噪声覆盖记未决。未知 HBM/Vector 持续吞吐保持未知。

## 后续范围与停止条件

上述主线通过后，再迁移 f16/bf16、MHA/GQA/MQA、prefill/decode、causal、非方形
与 tails；然后为 varlen/paged/sparse/persistent、split-KV、动态 GMM/MoE/GDN
分别落实 descriptor、间接访问、ownership 或全局 phase 合同。旧 FA 变体不会
自动成为新 AST 的支持能力，预留接口不算实现。

本部分不做无界调参、整套 TileLang 移植或一般整数求解器。若需要按输入长度
枚举、隐藏搬运、扩大缓冲或增加无依据 barrier 才能通过，就停在明确反例上
修改公共表示；不通过继续增加局部示例宣称进展。

每个可运行提交同时更新[状态页](refactor-progress.md)。只有产品文件进入独立
Git；日志、生成代码、profile 在外层 runs。设备前检查所有卡，每进程物理卡
映射逻辑0，不 reset、不换卡规避错误、不使用 /tmp 锁。
