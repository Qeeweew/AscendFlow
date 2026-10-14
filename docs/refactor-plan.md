# 重构实施计划：语法、编译复杂度与流水性能共同验收

2026-10-14 重写。本计划取代此前“静态 DSL 包装 → 有限图提速 → 最后迁移算子”
的顺序。以[总体设计](design.md)和[DSL 规格](dsl-multistage.md)为目标，不表示
实现已经完成。算子调参暂停；固定配置的代码/设备对照属于重构保真验收。

## 1. 当前现场与可复用资产

本次修订开始时产品 HEAD 为 `7c5d0a9`。更早 checkpoint 为 `06fce4d`，旧计划提交为
`9b40e2f`，多阶段原型为 `ec45a38`。工作树已有符号循环、codegen、示例及回归
修改，必须单独检查、保存，不能回退覆盖。根目录不建 Git，只提交产品文件。

| 保留并验证 | 必须替换/收敛 |
|---|---|
| 审查过的 dav-2201 native effects、布局、精确范围 | 用 Python 循环执行/展开设备循环的前端 |
| 线性 completion prefix、访问索引、受影响义务重查 | 单 runtime 参数、最近循环隐式版本绑定 |
| 最终实际 IR 重检、不可变 Compilation | `Program(count=0)` 模板约定、全局周期 LCM |
| ASC/CANN 工具链、设备 runner、数值 oracle | 有限/符号两条长期生产路径及重复分析 |
| examples 内算法与已记录的源码/设备基线 | 用已序列化 IR 计时代表全部前端编译 |

单外层符号循环已经有局部可运行原型，不再从零重写；但嵌套/分支/多状态摘要
尚未成立，现有 FA 仍走遗留路径。当前 checker 重新提取最终 IR 是必要防线，
其共享算法意味着它还需要独立测试 oracle，不能直接称为独立形式化证明。

## 2. 阶段与依赖

```text
R0 固定现场/基线
  → R1 AST + 单结构化循环 + Vector/归约闭环
  → R2 嵌套摘要/状态 + Cube/CV 资源闭环
  → R3 真实 dense FA，固定专家排步与大规模设备验收
  → R4 GQA/MQA/causal/tails，收敛唯一生产路径
  → R5 动态 metadata/paged/sparse/persistent（独立后续范围）
```

性能记录从 R0 开始，每阶段检查最终 ASC。不是等 R4 结束才发现语法或分析迫使
流水串行。R3 的数值主体和固定 AscendC 配置在 R0 就确定，避免以同步小例代替
真实算子的设计验收。以下门槛未通过，就修复该阶段，不继续堆算子变体。

### R0：保存现场并建立可比较基线

交付：

1. 先 review 当前 diff，再单独提交现有实现 checkpoint；记录 HEAD、脏文件和
   未验证变更。不得把本次 docs-only 提交称为当前运行时代码已完整保存。
2. 固定 Vector、归约、递推切片、分层 Cube、CV 及 dense FA 的源码、数值契约、
   tile/layout/slots/排步。保存当前能生成的 before/after IR、ASC 和编译选项。
3. 建立统一阶段计时/结构计数输出；保留旧慢输入用于定位，但不反复等待无预算
   的大例。记录现有路径哪里按输入展开，哪里有重复分析或资源搜索热点。
4. 选定可运行的同契约 AscendC 对照。优先适配已有专家/ops-transformer 实现；
   若无法独立运行所选源码，记录缺口，先保留可运行手排基线，不能拿 ACLNN
   二进制冒充该源码结果。FA 对照至少保持逻辑 BK、P 舍入和 FP32 更新顺序。

退出条件：相关实现回归通过或失败已定位；最新图搜索改动重新验证；每个基线
都有可复现入口和明确测量口径。旧记录的测试通过次数不能替代本次工作树结果。

### R1：受限 AST、显式槽位和运行时循环

交付：

- `@prim_func` AST 捕获、两遍签名解析、源码诊断；多运行时标量、constexpr、
  基础 if/for/算术，生成 xDSL arith/scf 与最小 NPU region。
- `serial/Pipelined` 不执行 Python 迭代，不按常量或运行时 N 默认展开；
  显式 `unroll` 有结构预算。buffer leading slot axes 和范围/别名检查。
- 现有单循环符号分析接入结构化 IR；放弃伪造 `count=0` 容器，保留已审查
  native effects 和最终检查路径。首阶段暂不支持的嵌套必须精确诊断。
- examples 中双缓冲 raw-length Vector、归约/normalization 原语小例；尾部
  先支持可实现的对齐子集，不以 padded shape 掩盖原始长度。

退出条件：

1. 同一生成 binary 运行 N=0/1/2/4/4096/65536 个 tile，独立改变至少两个 runtime
   参数；AST 到 ASC 不含按 N 实例化，模板/图/code size 与 N 无关。
2. 输入范围、溢出、未知调用、错 scope/slot、别名覆盖、零次结果未初始化有负例。
3. 前后 IR 和实际 ASC 展示 ready/WAR/最后读取/drain；CANN 离线成功；设备
   Vector/归约正确。不能只以 xDSL verifier 通过验收。
4. 冷编译包含前端，达到常用内核目标；保留与手排 ASC 的搬运/缓冲/等待对照。

### R2：分层摘要、循环携带状态和资源规划

交付：

- 保留嵌套循环、actor/guard 和多维代次；区域摘要导出精确访问、入口条件、
  last-read/ready/reuse 与存活资源，不以 child 全完成代替所有出口。
- 独立周期按耦合关系求解，取消全局 LCM；分支/零次/一次/稳定段/尾部各有
  检查规则。纯 scalar 捕获与有 effects 的 metadata 读取分开。
- event 位置/宽度/ID 一起参与规划；本地共享 ID 需要寿命证明，CV 保留真实
  2 AIV 参与者及 gate。支持范围有限时明确报错，不能当不存在可行排步。
- GDN 风格 carry/snapshot 微例、动态内层分层 GEMM、完整贡献后发布的多 segment
  CV，均放 examples；从这三类程序检验摘要，避免只为 FA 设计。
- 独立有限实例化访问 oracle 与有界交错测试，不调用生产依赖分析产生答案。

退出条件：

1. 外层、内层长度分别变化不改变模板规模；两个无关 ring 的周期不相乘。
   有耦合大周期和复杂 guard 时，预算/取消可响应并给出具体结构计数。
2. 删除早 release 所需的某个 reader、提前 ready、错代次、通知不平衡、ID
   提前复用、漏 drain 都有可重现反例；最终 materialized IR checker 能拒绝。
3. AIC 完整新 QK 的派发在旧 P gate 之前；L1/L0/L0C 的释放点分开，能证明
   输入早复用时不等待整个计算或父 region 结束。ASC 对照和设备数值均通过。
4. 同一固定配置的设备性能没有无法解释的退化；大型 native body 的编译图
   规模、proof 查询和 RSS 有记录，不以小模板结果外推复杂 FA。

### R3：真实 dense Flash Attention，性能保真提前验收

交付：

- `examples/` 中完整真实 FA：Q cache、FP32 online m/l/O、P 显式舍入、QK /
  softmax / PV / UpdateO 多阶段，运行时 KV 长度和嵌套 native/row 子流水。
- 单独 qk/pv/reduce 宏只表达算法与 typed operations，显式传 scratch；
  不加入编译器算子名分支或手工同步旁路。
- 一份固定专家配置先验证实现保真，再检查较大通信 BK 的分层表达；不能用
  小 BK、逐 query 全 drain 或额外 GM 转置作为迁移完成的隐含前提。
- 同契约 AscendC 和 ACLNN 分别运行并 profiling，绑定相应源码/库/tiling。

退出条件：

1. 小例逐元素 oracle；固定大例 B1/H8/Sq=Sk4096/D128 f16 的采样数值及完整
   finite 检查通过。误差阈值、归约树/exp 实现和采样策略在运行前固定。
2. 同一内核覆盖多种 KV 长度；全部源码到 ASC 冷编译达到代表性 FA 目标，
   不能只测旧 source.xdsl 到 ASC，不能将 artifact 属性访问记成编译缓存命中。
3. 生成 ASC 复现所选专家搬运、容量、复用和 gate 顺序；差异逐项记录。
4. 提交设备 task 时间、profiling、实际 native FLOPs/流量与条件理论下界。
   固定手排配置下无新增等待/搬运仍不等于性能相同，必须有设备证据。

本阶段允许结论为“重构保真成立，但仍落后 ACLNN”，前提是量化并解释差距，
不能宣称达到最终性能目标。此时另列通信粒度、流水/容量或指令效率问题；
恢复算子调优要形成下一独立任务，不在重构中无界搜索 tile。

### R4：基本 FA 变体及唯一生产路径

交付：GQA/MQA 的 head 映射、causal、对齐/非对齐尾部及空域契约逐项实现。
这里的 FA 范围为 forward；GQA 明确 Hq/Hkv 整除和映射，非方形 causal 明确
左上/右下对齐与绝对位置，不能只以一个含糊的 causal 开关共用数值 oracle。
新增 B1/Hq8/Hkv2/Sq=Sk8192/D128 bf16 固定 GQA 大例；长 KV decode 单列
吞吐模型，不能用 prefill 的有效 FLOPs 解释 decode。

退出条件：

- 各变体前端只使用普通参数、控制流和原语；没有新增算法特判。
- causal 全 mask 行/尾块/两 AIV 参与不对称有明确数值及通知契约，不能死锁。
- 迁移边界内所有示例走同一结构化 compiler；普通公开入口不再按隐藏 marker
  选择有限后端。未迁移算法继续明确标为 legacy，不能覆盖为新 API 示例。
- 有限 planner 移至测试参考/显式 legacy 工具；在承诺支持的旧示例迁移和
  回归替代完成后删除生产分支。旧物理 IR 导入必须显式选格式并完整验证，
  不提供静默 fallback。核心独立安装不依赖 examples/reference。

### R5：后续动态任务与高性能变体

varlen、paged、sparse、persistent FA、动态 grouped matmul/MoE 分阶段做，
每项先定义 metadata 读取、task descriptor 捕获、动态 ownership 和交接契约。
连续 query 的 state first/last 必须随任务移动；跨组/全局阶段不能沿用核内 event。
这些不阻塞 R1 的局部闭环，但不能在 R4 后被自动登记为“已支持各种 FA”。

## 3. 统一验收矩阵

### 编译可用性与复杂度

| 维度 | 固定测试/记录 |
|---|---|
| 运行时 N | 0/1/2/4/4096/1048576，固定源码/constexpr；检验模板、图和 ASC 不变 |
| 源码结构 | 模板数量、嵌套深度、access pieces、真实 overlap、guard 分支数分别增长 |
| 周期/资源 | 独立与耦合 ring，互素 slot 数；event 资源压力与寿命共享 |
| 阶段计时 | AST/宏、源 IR、native lowering、effects、规划、最终验证、ASC、可选导出 |
| 内存/搜索 | peak RSS、T/A/H/C/P/V/E/D/R、重建/失效次数、最慢义务 |
| 运行方式 | 至少5个冷进程、p50/max；真正编译缓存命中另测；import/CANN 分开 |

目标：常用内核 p50≤1秒，固定代表性 FA p50≤5秒，真实缓存命中≤100毫秒；
初期 max 门槛为相应目标2倍，RSS 和结构增长需解释。独立 N 变化不允许线性
实例化，即使缓存后很快也不通过。结构变大可变慢，但必须给出上述成本分解。

CPU 实验先设60秒单 case 总预算及可取消搜索；超时保存最近阶段/计数/profile
摘要后结束。优先用较小静态结构定位热点，再对同一修复复测；禁止持续重复
运行未改代码的同一个大例等结果。内部查询预算与总 wall time 都记录。

### 设备性能

固定手排基线保真与 ACLNN 差距是两项指标。前者用于防止重构退化，后者体现
最终竞争力。默认退化阈值为同卡、相同缓存/计时政策下中位数超过基线5%；
先用重复批次估计波动。波动覆盖阈值时结论为未决，不能强报通过或反复择优。
至少报告 p50/p90 和误差范围，设备时间与 host/stream 时间分开。

大规模 Vector 可固定2^24个 f32 元素；normalization 可固定8192×4096，按
算法容量选择固定合法 tile。FA 大例见 R3/R4，decode 后续固定 Sk≥32768。
热重复和总工作集大于192 MiB L2 的轮换分别记录，避免以缓存差异误判 HBM。

每次实验一并审查最终 ASC、可运行 AscendC 和 profiler；从有效/原生 FLOPs、
各路径流量、依赖关键路径及实际硬件信息解释差距。未知 HBM/L2 持续带宽不
填猜测值，不把 GM 请求量直接除以标称 HBM 带宽当作可靠运行时间。

## 4. 代码边界与交付纪律

按职责逐步提取，不先一次性改所有目录。前端/结构化 lowering 接替 `language.py`
和 `lowering.py` 的展开职责；`loop_*` 成为结构化同步的内部实现，复用必要的
graph/frontier 原语。`sync.py` 的双路径分流在 R4 收敛；`model.py/check.py`
的有限逻辑保留期间标明 legacy/test。避免让源码模块和物理模块互相调用形成环。

每个小步提供：实现 diff、适用范围、精确拒绝诊断、相关测试、编译/运行命令、
前后 IR/ASC 与性能证据。文档只引用工作区外运行数据，Git 内只放产品源码、
测试、示例和文档。环境、缓存、构建、profiler 原始数据不提交；不 push/publish。

设备执行前检查所有卡；选健康空闲卡并显式物理卡映射逻辑0。不 reset、不用
`/tmp` 锁、不换卡掩盖错误。需要新增目标机器契约时先源码核查，再离线/设备
验证；既有模型证明不能替代硬件依据。

实施现场以[最新验证记录](refactor-progress.md)及实际Git为准。R1单循环闭环和
R2的周期分离、显式Cube存储已完成相应切片；下一重点是R2嵌套物理控制、动态
内层K与层次生命周期摘要。R0其余基线和R3/R4验收没有被这些切片替代。
不要绕过这些门槛迁移全部FA、扩展动态任务框架或恢复算子调参。
