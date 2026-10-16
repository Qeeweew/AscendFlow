# 下一阶段：编译成本、向量路径与 Attention 变体

2026-10-16。当前完整 dense f16 FA 已通过公开 DSL、真实同步IR、CANN 和设备数值验证。
已完成和限制以[状态页](refactor-progress.md)为准。算子主体在 examples，编译器只增加通用语义。
旧 `cube.task/vector.task`、legacy builder 和兼容路径不恢复。

## 1. 先降低真实编译成本

最新完整FA源码到ASC为128.96秒，其中规划75.14秒、最终独立检查50.95秒；
此前为169.10秒。运行时Sq/Sk仍为SSA变量，当前76个native模板、12个call固定。
规划前重复parent验证已删除，新query单ID检查也不再直接建立最大窗口。
最终独立检查保留，当前主要剩余成本是静态条件完成事实的传播与包含扫描。

下一步按以下顺序修改，逐项保留同步指令、物理访问和buffer布局等价证据：

1. 在单个flow/node保存已接受origin bits的并集；全新且完全不相交的bits无需扫描
   其他记录的包含关系。有交集仍走原算法，不能合并domain或扩大coverage。
   先测独立origin在不同静态位置汇入同一node的反例：当前比较m(m-1)，目标消除
   这类无关扫描；保留大整数bitset字长成本。此项目前仅有方案，FA实际收益待测。
2. 合并同次分析中完全相同的静态工作：本次boundary transfer已由4503个cells共享667个
   唯一effect序列；继续检查projection及condition事实表示，不跨不同domain合并路径。
3. 按源/目标实际端点批量登记查询，不逐byte传播；每次真正插入协议后才更新完成关系。
   不做“删一个同步→重跑全验证”的必要性搜索，不提高预算，不用C++重写掩盖算法规模。
4. 稳定后做五个独立冷进程，从源码读取到最终验证/ASC计时，import/CANN/写盘另列。
   常用kernel目标p50≤1秒/max≤2秒，完整FA目标p50≤5秒/max≤10秒。

所有算子构建仍用统一≤300秒硬期限覆盖构造、分析、ASC及CANN。超时停止整个候选序列。

## 2. 以源码和profiling解释性能差距，再改Vector政策

起点固定B1/H8/Hkv8/Sq=Sk4096/D128/f16。必须保留实际DSL/IR/ASC/库hash、CANN选项、
设备及输入、计时与缓存口径，三批generated/独立AscendC/ACLNN交替profiling。
block8分层归约已接通真实DSL并完成三批验证：device task中位时间为
825.44/1194.27/512.21µs，较旧R64降低27.8%，仍慢于ACLNN61.2%。
Vector活动已由698.11降至380.45µs；继续优化前要核对CV时间线，不仅比较指令数。

源码依据为ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e` 的arch22，
不是950路径或安装版ACLNN的已知构建提交。已确认的差异包括：

- 当前显式block8→packed merge/BRCB/8行输出更新；上游SoftmaxFlashV2历史量每行8份，
  D128 Vec2按64行、直接Div。先检查Vector活动与生成指令，按UB容量和真实读写寿命选块。
- 当前AIV update→softmax、serial query；上游Vec1→Vec2、taskId跨query连续。
  派发尾部不等于硬件drain，需时间线判定重叠损失。
- 当前Score/P/PV分离、O驻UB；上游f16 ND可复用Score/P地址并把中间O写回GM。
  不能未经访问/ownership证明照搬alias，也不能靠额外搬运换取正确性。

每次只做一个有测量依据的政策/通用变换。FP32 sum树可调整，按误差合同验收；
P/O舍入边界及递推状态更新顺序仍须保留。不得按算子名、shape或生成字符串特判。
对比硬件时用真实核数/频率、Cube工作、Vector活动、存储容量及L2工作集；
条件峰值/理想下界单列，pipe活动不能相加，GM请求量不是实测HBM流量。
超过ACLNN 5%仍为失败/继续调查线；三批波动交叠则记未决。

下一候选先审查example的Vector issue顺序：上游先Vec1再Vec2，而本例先update再softmax。
通过时间线判断旧PV等待是否阻碍新softmax/P就绪，再决定是否改源码政策；
不得绕过alpha双槽和最后读取、增添buffer、或直接复制一张手工event表。

## 3. 继续完善通用同步与ASC表达

- 已接通query内存/已有parent-child事件/跨核counter强边界反馈。新query协议当前仅local；
  条件缺乏反馈、bank满、动态别名或ownership无法证明时明确拒绝。
- 下一步处理合法但资源不足的跨query复用：现有mixed recurrence在MTE1→MTE2 bank用满后拒绝。
  先证明通知寿命的可共享区间或连续ordinal，再改分配；不添全局barrier、buffer或drain。
- 专家排步和自动生成继续共用实际IR verifier。逻辑RAW、物理alias、异步完成、最后reader、
  event消费及退休分别保留；Scope/Query/child三种同步在实际native旁统一按IR顺序检查和输出。
- 通用纯整数canonicalize/CSE已接入；query短count case按端点等价证明合并，FA新增ID为常量6。
  capture `/1`已在Expr结构渲染中消除且保留合法域证明；其他冗余继续按公共规则处理。
- 统一编译产物的结构化ABI元数据，避免launcher物理参数顺序与DSL形式参数不同导致调用者猜测。

## 4. Attention变体按真实合同逐项接入

| 顺序 | 范围 | 验收 |
|---|---|---|
| A | 默认dense f16，当前已通 | 保留已测四个短域与4K；先解决编译和性能差距 |
| B | bf16、GQA/MQA、cross attention | 正确head映射、共享KV只读、相同二进制多长度；各自CANN/数值/性能 |
| C | 同主体continuous | runtime-cycle首末、跨query状态、未触slot和全部Wait消费；不能伪装成parent首末 |
| D | causal/offset、tails、decode | 有效域、全mask行、padding真实DMA footprint、低并行度政策 |
| E | varlen/paged/sparse/persistent/split-KV | metadata epoch、间接范围、任务generation/ownership及跨kernel归并 |

默认固定政策四个点已验证不等于所有constexpr/变体都支持。示例仅放一份数学主体；
未实现政策留测试/设计合同，不复制一堆不可运行example。
继续用GEMM、向量归约/normalization、RoPE/MoE、动态GMM/量化FFN和GDN检验抽象，
不能为统一模型改变它们的有序状态或数值合同。

## 交付门槛

每个阶段交付实际补丁、针对性反例、可复现命令及明确支持/拒绝范围。常规pytest不反复长编译FA；
完整FA由一次总期限受控的纵向构建和设备证据验收。同步关系用独立有限图交叉检查，
有限规模结果不冒充任意程序证明。只有结果绑定当前代码/IR/ASC/库才算本阶段证据。

多Agent继续按根AGENTS.md一文件一写入者；主Agent审查接口/集成并独占排队设备任务。
冷编译期间不并发CPU压力任务；不安装、打包、推送或发布作为验收替代。
