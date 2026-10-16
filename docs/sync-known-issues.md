# 同步与编译复杂度：当前问题

2026-10-16，审查基线 `fdff8c5`。算子优化停止；当前主线是
[编译流程重构](refactor-plan.md)。[设计](design.md)解释复杂度，
[不变量清单](compilation-invariants.md)规定事实的建立、保持与失效。

| ID | 当前问题 | 替换方向与验收 |
|---|---|---|
| SYNC-01 | origin/guard事实仍可能在节点桶反复扫描，累积超线性 | typed prefix/结构化条件/需求端点；记录实际K、插入和比较，不仅改cache |
| SYNC-02 | native语义已共享，但child/parent/query分析对象和部分拓扑重复构造 | 参数化region transfer与内部cuts；不能只保留whole-child完成 |
| SYNC-03 | planning和final重建；final内部还重复child/pool/counter验证 | 同快照事实一次建立；query仅处理新增跨query义务；正常流程按合同传递 |
| SYNC-04 | 各统计作用域/峰值/重复计数易误读 | unique、累计、单实例、全pipeline、CPU/RSS分列；case不等于依赖 |
| SYNC-05 | 一般动态CV、多时钟、连续跨query状态未完成 | 保留现有dense与受限借用，未知ownership明确拒绝；不以简化语义缩减现有能力 |
| SYNC-06 | 新不变量/构造算法尚无实现，不能直接删除现有检查 | 局部合同、反例、物化对应接通后逐段退出重复检查 |
| SYNC-07 | 仍有finite/source兼容分析和多种组织分派 | 迁移机器反例后删除旧生产分流；独立有限oracle留tests |
| SYNC-08 | 已支持部分ID共享/借用，但width/parent/query阶段反复资源查询 | 联合逻辑寿命模型，分配只改ID时不重算memory；保留FIRST/LAST与跨query整池义务 |
| SYNC-09 | 配置、local、cross-counter资源必须统一参与计划 | 现有入口配置检查已存在；新资源模型保留，不恢复旧“未纳入”的描述 |
| SYNC-10 | 独立有限模型仅覆盖受限native/actor集合 | 扩展按实际硬件规则；有界检查不冒充任意程序证明 |
| SYNC-11 | 普通region summary部分路径枚举固定count；scope组合按行/cell放大 | 紧凑足迹/transfer，计数增长与真实描述/输出相关 |
| SYNC-12 | 事件安全和进展与memory TRUE混在查询流程 | 分开INV-07/09/10；增同步不保证无死锁，ID相同不产生完成边 |

近期已有改动：sequence完整pattern共享、fixed child稀疏投影、query coverage合批、
append-only完整正答案保留、singleton查询化简。它们没有完成以上架构替换，不能继续
通过预算补丁宣称总体复杂度已解决。

历史GEMM图计数、旧阶段结果和被替换方案保存在Git及
[旧状态](history/refactor-progress-before-query-20261016.md)。这些不是当前HEAD的耗时。
当前设备资格、五次BQ256拒绝和准确证据目录见[状态页](refactor-progress.md)。
