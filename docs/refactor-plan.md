# 自动同步重构实施与剩余验收

2026-10-17。本页区分已接通的实现和仍须以产物验收的项目；结果见[状态页](refactor-progress.md)。原 `1231f4f` Path/搜索树实验不再是实施路线。

## 已实施的职责替换

1. 审查了真实 native、内存准入、逻辑 RAW、存储 WAR/WAW、通知绑定及后置资源链。事实的输入、建立者、产出、消费者和失效条件逐项列在[生成规则](generation-rules.md)。
2. `generate_notifications` 先建立真实 capture/acquire、域配对及分层 physical-step/phase 前提，再构造完整 cell credit 或明确 queue handback 生命周期。两端先 Wait 再 Set 的匹配死锁在本层拒绝。
3. 循环保持参数化 body；父层只组合实际 open ports、FIRST/LAST 与 may-block cuts，不导入 child 内部路径历史。零次、短循环、首末重合、父代次翻译和相邻循环均有反例。
4. 分配器只消费 `NotificationProgram`，建立 bank/ID/共享新约束。完整 credit 的全部成员同时进入台账；配置和最终 drain 保留独立预留。
5. 物化和 ASC 后端消费确定计划。旧 progress、Path、三值查询、任意执行路径及对应 solver 形状测试已删除，无 legacy、兼容别名或第二链路。
6. 独立有限模型可对照 pre-sync 快照，检查最终 native、地址、读版本、二值消费和 mode-2 quorum。真实默认 FA 与嵌套双 AIV 模型接入回归；算法和政策留在 examples。

## 每程序仍需执行的前提

输入 effects、bounds、初始化、alias/ownership、值/存储 lifetime、真实位置、原式有序定义性、配对、phase 准入，以及资源新增约束均保留。阶段 rank 不能授权 flag 复用；局部 loop exit 不能替代最后 reader；报告标签不能授权外部物理 IR。

相同 view、标量 DAG、绑定域和实际插入位置的事实只建立一次。发生地址、guard、native、参与者、width 或共享成员变化时，必须按[不变量](compilation-invariants.md)更新受影响事实。

## 完成门槛

- 单一公共 DSL→IR→ASC→CANN 的当前 Vector、默认 GEMM、嵌套 CV 与默认 FA 产物，以及独立语义回归。
- 默认政策固定、Sq/Sk 不特化；source→ASC 包括源码读取/构造、所有分析/化简/输出与内部快照。import、进程总时间、CANN 单列。
- 默认 FA 五次冷进程 p50≤5 秒、max≤10 秒。单次 probe 不提供这项资格；接近门槛后才进行五次，不降低阈值或缩小算子。
- 每算子完整构建≤300 秒，失败先定位并清理所属进程组，不继续下一候选。
- 当前不进行算子性能调优、设备压测、打包或发布。后续设备性能需较大输入、最终 ASC、AscendC/ACLNN、profiling 和硬件条件指标。

生成合同接通、ASC/CANN 通过和编译性能达标是独立状态。未闭合项目必须保留在状态页，不能将局部提速或测试数增加写成重构完成。
