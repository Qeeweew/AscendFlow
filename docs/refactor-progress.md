# 当前状态与未解决问题

2026-10-16。本页是唯一当前状态入口。[设计](design.md)和[计划](refactor-plan.md)描述目标，不作为实现证据。

## 1. 当前工作和代码现场

用户已停止算子优化以及上一轮同步求解器局部补丁。本轮授权工作是：审查整体结构、清理文档、重写大幅重构计划。**本轮未修改编译器/测试/算子代码，未运行编译、CANN或设备。**

产品HEAD为 `7ddaa54ec060a92e219d0462df162939b8b32d72`，标题为“Reuse owned synchronization analyses and reduce cold compilation work”。该交付已被用户判定不达标。根目录没有产品Git，全部现有未提交代码实验保留。

工作树上的未提交实验包括compact query patterns、virtual child cuts、局部transfer、边界批读和ownership区间合并，以及对应测试。它们不是新计划的已完成阶段；下一轮逐项判断语义工具能否复用，不能把它们默认为正确的最终架构。停止前只完成部分定向回归，没有当前工作树的完整新资格。

本轮文档从36份收敛为10份。原文、四个入口README及SHA256清单保存到[外层历史归档](../../reports/compiler-refactor/20261016-before-structured-loops/README.md)，不纳入产品Git。运行数据没有迁移、删除或改写。

## 2. 编译耗时：明确未达标

固定默认FA政策：B1/Hq8/Hkv8/D128、f16、BQ128/BK1024/CK128/VR8、groups20，Sq/Sk保持runtime。目标为五个独立进程source→ASC p50≤5s、max≤10s。

| 代码/运行身份 | 结果 | 能说明什么 |
|---|---|---|
| 已提交7ddaa54对应冷测 | source→ASC p50 53.114s，max 54.309s，5次 | 上次交付未达标；不是当前实验成绩 |
| 未提交边界压缩早期probe | source→ASC 31.022s，1次 | 单次编译可完成，不能当五次p50或最终达标 |
| 未提交最近采样诊断 | wall 37.840s，内部compile 37.707s，规划34.699s | 仍很慢；含诊断，不与冷测混为同一统计样本 |

证据：外层[旧五次汇总](../../runs/compiler-refactor/comparison.json)、[单次probe](../../runs/compile-threshold/fa-boundary-probe/benchmark.json)、[最近采样](../../runs/compile-threshold/frontier-profile/profile-samples.json)及[工作量](../../runs/compile-threshold/frontier-profile/profile-report.json)。

最近采样：76个native模板；父流水972节点，跨外层任务的分析窗口2926节点。父层13个域、1082个来源、109次传播批次。

| 层 | 最终条件事实 | 沿边传播 | 事实比较 |
|---|---:|---:|---:|
| 父流水 | 27,069 | 548,885 | 1,749,299 |
| 外层任务窗口 | 18,394 | 459,096 | 1,894,151 |

两层已超过100万次传播，尚不含局部child分析。压缩节点没有消除重复工作，不能继续将“节点少了”作为完成证据。最近lowering约1.574s、emission约0.798s、snapshot约0.472s，说明同步以外也占用≤5s预算。

上述两次实验的ASC SHA均为 `e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`，与默认旧基线一致。这是该配置的输出一致证据，不是新的CANN/设备测量。

## 3. 问题清单

| 编号 | 当前问题 | 新计划中的处理 |
|---|---|---|
| A1 | serial/KV/fixed-child结构特判，QueryChannel与专门副本图 | M1/M3/M4统一PLE、边界、回边并删除query路径 |
| A2 | 多层全来源条件传播、增边重放与区间比较 | M1先确定有限前沿/归纳规则，M2/M3替代正常路径 |
| A3 | child/parent/query分阶段资源求解 | M3统一逻辑lease，保留共享/borrow与硬件机制区别 |
| A4 | 93模块平铺，职责闭环，emitter依赖planner | M1开始按职责迁移，M4统一输出；见源码结构页 |
| A5 | 源语言接受的carry/组合在lowering才拒绝 | 统一admission、源位置和能力矩阵 |
| A6 | 原语合同多处维护、Load2D展开/化简藏在emitter | 透明recipe及唯一原语合同，纯标量显式pass |
| A7 | normal/deep分离局部落地，但旧分派仍存在 | 保留所有权原则，删除重复后端；外部输入独立验证 |
| A8 | 测试依赖旧图细节，独立性表述过强 | 语义反例迁移、有限模型分级、结构成本测试 |
| A9 | 历史报告、无效命令和现状互相矛盾 | 本轮归档、重写索引；今后只更新本页状态 |
| A10 | FA编译和相对ACLNN性能均未达标 | 先编译结构验收；设备资格及算子调优另阶段执行 |

## 4. 当前可用路径与资格范围

| 范围 | 现有实现/证据 | 边界 |
|---|---|---|
| AST DSL→xDSL→ASC | Vector、normalization、GEMM、状态和默认dense FA示例已接通旧主线 | 新统一循环算法尚未实现；语法通过不代表任意组合通过 |
| 普通编译事实所有权 | 已提交版本有normal/deep分离、资源结果交付和纯整数化简 | 不等于统一算法或成本达标 |
| CPU回归 | 7ddaa54对应旧记录有分段2831项；停止实验有定向回归 | 不把历史总数写成当前工作树全量通过 |
| CANN | 7ddaa54记录FA/GEMM/scale离线构建成功 | 本轮没有新CANN结果 |
| 默认FA设备 | 对应历史产物有短域全输出、4K/8K采样与msprof | 当前未提交分析实验没有新的设备资格；采样不是全输出 |
| Attention变体 | runner可表达部分dtype/head/tile政策 | 不代表causal、tail、varlen、paged/sparse、persistent等全部实现 |

公开 `cube.task/vector.task` 已移除，不恢复兼容。动态间接alias/ownership、任意guard、continuous多时钟状态等仍需通用合同和明确诊断，不能通过新接口名字宣布支持。

## 5. 历史设备差距与证据保留

默认4K f16 FA的旧三批msprof设备task中位数为生成代码605.780µs、ACLNN512.152µs，慢18.3%。独立手写AscendC为1191.220µs。这是对应旧源码/政策/设备的历史结果，不是编译器重构加速结果。

原始摘要仍在[performance.json](../../runs/refactor/cv-order/performance.json)及[op指标目录](../../runs/refactor/cv-order/op-metrics)。历史文档原文见外层归档的 `compiler/docs/attention-performance.md`。此前清理已删除库、部分profiler原始数据和旧候选目录，需重建才能再次运行；不恢复旧优化任务。

未来比较必须直接看生成ASC、手写AscendC、ops-transformer/ACLNN和硬件条件下界，详见[验收](validation.md)。当前仅M0文档整理完成，M1–M6未开始。
