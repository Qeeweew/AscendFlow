# 同步与编译复杂度：当前问题

2026-10-16。编译流程重构的实现与冷测结果见[状态页](refactor-progress.md)，
变换保持的事实见[不变量清单](compilation-invariants.md)。算子优化仍暂停。

| ID | 本轮变化 | 剩余工作与边界 |
|---|---|---|
| SYNC-01 | 相同完整guard的连续frontier合并；pending传播直接使用实际增量；纯数学域值在一次编译内共享 | 不同guard的antichain仍可能超线性；不能把合并前沿等同于消除origin×node传播 |
| SYNC-02 | 普通编译保留child分析、父层投影；sequence和invocation复用完成关系；query追加通知保留同一图与origin | parent和query仍有不同适配图；静态窗口不足时重建；尚未替换为完整参数化边界transfer |
| SYNC-03 | 普通AST路径不再独立重建最终分析；child/pool/counter结果交给query | deep与外部同步IR按实际输入独立检查；低层有限physical入口仍保留既有检查流程 |
| SYNC-04 | 增加唯一提取/依赖构造、增量rank、数学值和资源查询计数；冷测记录进程RSS | 局部统计不是整个pipeline累计成本；不能相加各flow峰值充当RSS |
| SYNC-05 | 保留dense CV、早release、受限parent/child事件借用 | 一般动态ownership、连续跨query状态、任意多时钟/稀疏task stream仍不支持 |
| SYNC-06 | owned结果、物化对应和normal/deep接通，增加实际IR破坏回归 | 不声称编译器已形式化验证；未知输入不能复用内部结果或伪造verified标记 |
| SYNC-07 | source统一scope collection，删除`_legacy_work`和旧finite source walker | 低层有限physical诊断入口与多个完成关系适配器仍存在；独立有限oracle保留 |
| SYNC-08 | 同一资源请求跨分配重试复用；无空闲bank直接进入既有借用规则；ID改名不重证memory | child→parent→query仍分阶段分配，不是一次完整联合资源求解 |
| SYNC-09 | local、配置保留、CV counter及跨query整池义务继续参与资源分析 | 二值event与counter规则仍须分开；分配同ID不能产生新的完成事实 |
| SYNC-10 | 增加增量排名、摘要、物化绑定与有限模型差分 | 有界模型只覆盖测试域，不能替代任意N证明或设备资格 |
| SYNC-11 | 合格stationary固定循环摘要压缩；非query长reader保留first acquire/tail prefix；query不构造无需消费的摘要输出 | query长ReadSpan、漂移/特殊footprint和需要完整历史定义的路径仍使用精确固定表示；保留的物理义务仍须聚合 |
| SYNC-12 | 精确端点的capture/acquire直接建立完成关系；进展排名只更新新增/受影响约束 | coalesced不同端点、资源消费和一般偏序仍需要分析；增同步不保证进展 |

Sq/Sk保持runtime参数；本轮没有改变FA示例算法、tile、DMA或buffer政策。默认FA的
生成ASC与基线相同，编译时间下降不代表ACLNN设备性能差距已解决。
一般guard、资源共享和布局相交的复杂度仍可能较高，后续方向见[计划](refactor-plan.md)。

历史BQ256的五次拒绝、设备结果和旧规模诊断保留在状态页及Git；未通过的候选不因
本轮普通路径加速而自动取得编译/设备资格。未放宽分析预算，也未增加barrier兜底。
