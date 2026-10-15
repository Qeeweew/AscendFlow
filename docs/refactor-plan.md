# 下一部分重构：统一同步关系，接通真实层次 CV

2026-10-15。基线已保存为 `3f95139`。本计划替代旧 P/G 阶段推进顺序，
状态以[实现与验证](refactor-progress.md)为准。

## 本轮与下一部分的边界

本轮整理目录、导入、公共验证工具和文档，保存已有代码及证据；不修改同步
算法，不恢复算子调参。目录整理本身不记为同步统一或 FA 能力完成。

下一部分的主线只有一条：**同一套符号访问与完成关系 → 同一套同步生成/验证
→ 真实 FA 的 nested 和 continuous 两种组织。** 最小反例放 tests，直接从
主目标提取；不再把每次局部修补扩成一个公开算子和一套独立 benchmark。

## 1. 替换重复的同步语义

先以现有代码建立明确替换关系，不预建空模块：

| 当前职责 | 重构要求 |
|---|---|
| `region_lowering.source_body` 的形状准入 | 保留 xDSL region 层次；准入由已实现合同决定，不复制或展开运行实例 |
| `invocation_sync.Site`、`sequence_sync.Point`、`sequence_calls.Mark` | 统一端点身份、实例域、guard、milestone 和相对坐标 |
| `BoundaryProof.covered/site_path`、`CallProof.direct/covered` | 消费同一关系索引和覆盖查询；显式区分串行域与重叠 CV 域 |
| loop/sequence/invocation 的访问边界 | 同一 root/range 前沿及初始化/reader-retirement 合同 |
| event 局部分配及跨区域检查 | 同一逻辑生命周期表示，硬件 bank/参与者是协议属性 |

先迁移现有路径并删除对应重复逻辑，保留 native effects 和 emitter。若新路径
仍要根据顶层形状选择另一套完成语义，这一步不算完成。允许迁移期间有薄适配器，
每个适配器记录剩余职责和删除条件；不以共用类名作为统一的证据。

第一批关系用例必须包括：空 child、奇偶 slot 最后读、跨 region 条件传递覆盖、
双 reader、事件跨调用复用，以及 continuous CV 中新 query QK 早于旧 PV 的顺序。
关系可表达不等于完整程序可编译，后者在第 2/3 步验收。

退出要求：现有正/负回归保持；漏 ready、漏 release、错误 fresh、ack 当退休
均拒绝；11 个迁移基线 ASC 保持，变化必须逐项说明。修掉 frame snapshot 已知
多余通知时，要同时保持 child=0 正确，不能只按正长度删边。

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
