# 下一部分重构：统一同步关系，接通真实层次 CV

2026-10-15。基线已保存为 `3f95139`。本计划替代旧 P/G 阶段推进顺序，
状态以[实现与验证](refactor-progress.md)为准。

## 本轮与下一部分的边界

目录、导入、公共验证工具和文档整理已完成。region/call 的增量查询
改造已保存。随后已接通 serial region/call 的批量传播和局部叶最终验证，
frame 冷编译 p50 降至 0.532s，ASC 不变。第 1 步仍有叶生成器试删、旧 invocation
与跨核路径要替换；一般层次 CV 和完整 FA 尚未接通。
不恢复算子调参。具体源码及测量见状态页，目录整理本身不算同步统一完成。

下一部分的主线只有一条：**同一套符号访问与完成关系 → 同一套同步生成/验证
→ 真实 FA 的 nested 和 continuous 两种组织。** 最小反例放 tests，直接从
主目标提取；不再把每次局部修补扩成一个公开算子和一套独立 benchmark。

## 1. 替换求解方式：实际执行关系上的批量完成传播

2026-10-15 再审后，这一步替代“继续优化 RegionCompletion 的端点搜索”。
`3debb17` 是可回溯的旧实现保存点。缓存和范围剪裁可以缓解耗时，但不作为
下一交付的算法方案；算法合同及反例见[设计](design.md)。

### 保留与替换

| 现有部分 | 处理 |
|---|---|
| AST、物理访问与别名、固定排步、ASC emitter | 保留，不改变 DSL 或算子数学 |
| `region_summary` / `region_sequence` 的字节访问前沿 | 保留并补足组合边界；不重新命名成另一套访问框架 |
| `completion_graph` 的原生指令/队列/退休契约 | 提取为共同转移规则，直接构造真实稀疏执行关系 |
| `RegionCompletion.candidates/direct` → `LeafRelations.site_path` → `Graph.path` | 从新生产路径移除这条嵌套搜索链 |
| `CompletionPaths` 按查询来源维护可达前沿 | 用作用于真实执行关系的批量完成信息传播替换 |
| `sequence_plan` 的缺口生成与 event 分配分层 | 保留职责；覆盖判断改用消费者处的完成事实 |
| 最终 verifier | 从实际 IR 重提取，再批量传播；不使用 planner 缓存或逐义务重新搜图 |

### 内部实现顺序

1. 先对已经生成的真实 IR 实现完成信息传播，以当前独立有限模型和反例核对
   同流水开始/退休、Set 捕获、Wait 消费与 Scalar gate。边来自硬件转移和实际
   指令相邻关系，不能通过枚举端点对再调用证明器来发现边。
2. 加入已有规则循环的符号转移：基例、归纳步骤、空循环和出口。只处理实际
   访问相关的 slot 代次；不展开长度，不对无限代次做无界 fixed point，不把
   “小窗口通过”当成任意长度证明。child 摘要保留尚未完成的工作与实际捕获点。
3. 消费已有访问前沿产生的义务，在每个消费者处批量求缺口，再生成真实同步。
   已确定的协议更新执行事实；未选择的候选没有证明效力。实际流水顺序不能
   被词法 region/query 编号替代。
4. 在同一转移表示中保留 event 发布/消费义务及资源分配，接通最终 IR 重检。
   不能省掉最终验证来达成编译预算，也不能用额外 event/barrier 掩盖模型缺口。

第一交付必须同时接通生成与最终验证；仅有一个新验证器、关系类或快的微型
测试不算完成。新路径的准入由已实现语义决定，不按算子名字特判，不静默回退。

### 首个实际验收

使用已有 `frame_state_sum`、双缓冲 Vector 和 Cube/Fixpipe 回归，不新增公开
玩具算子或 runner。frame 的七个原生模板是成本与条件精度的基准：

- `N=0`：snapshot 的 MTE3→MTE2 ready 必须存在。
- `N>0`：snapshot 完成已经由 MTE3→Vector→MTE2 的真实协议传递，不再加直接通知。
- slot0 的最后读取：偶数 N 为 N−2，奇数 N 为 N−1，不能等待无关的 slot1。
- 偶数 N 的最终 state 写仍需要覆盖最后 Vector 更新，不能因 slot0 已释放就省去。
- 下一父调用复用 state 要等真实 MTE3 reader 退休，不等“整个 child 完成”。

同时检查双 reader、Set 后续工作、互斥 guard、错误 event 代次的负例；保持
现有源码的循环、buffer、搬运与数值顺序。所有同步变化逐条解释实际 ASC。

退出要求：新 frame 生产路径的逐端点叶路径查询为零；记录真实边/事实/条件
片段/转移次数，比较 N=1024/1048576 的结构规模和五冷进程阶段耗时。运行相关
定向回归，稳定后一次全套；设备仅在生成代码有需要验证的语义变化时安排。
不得把“低于 1s”当作算法成本已经合理的充分证据。

## 第 1 步与层次 CV 的衔接

层次是符号转移的组合：不同 child clock 保持身份，出口不制造 done；父层使用
明确的边界及中间完成事实，不再逐点打开 child 重新证明。先用 FA 的实际 QK/PV
与三个 GM 环检查该接口，尤其是两个 AIV 的完整贡献和早 import release。
如必须引入整体 child drain 或全局时钟展开才能组合，就修改转移表示，而非
增加一个 CV 专用证明器。未完成这种检验前，不宣称算法覆盖一般层次 CV。

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
