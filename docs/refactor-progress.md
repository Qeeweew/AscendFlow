# 当前状态：停止算子优化，重构编译流程

2026-10-16。代码审查基线 `fdff8c55b0cddc51331c637ffecc75ce0e34401f`，审查开始时产品
工作树干净。用户要求停止算子优化，改为梳理复杂度、编译不变量和重构方案。

**本轮只完成源码审查与文档重写；没有修改编译器代码、移除检查、运行编译/测试/
profiling/设备，也没有打包或推送。新算法尚未实现。** 三个 Agent 均仅做只读审查。

入口：[设计](design.md)、[不变量](compilation-invariants.md)、[实施计划](refactor-plan.md)。
这些文件替换旧的“继续 GM→L1 优化”顺序，专项性能页中的旧下一步不再是活动任务。

随后按用户要求清理了外层`runs/`：保留`refactor/cv-order`、`query-boundaries`和
`l1-reuse`中的近期源码/ASC、编译记录、CSV/性能摘要与失败诊断，删除旧实验、二进制、
缓存及profiler原始trace/数据库。保留目录是历史证据，已不是可直接运行的完整build；
库需重新构建，其他历史runs路径可能已删除。清理明细见外层`runs/README.md`。

## 当前代码实际支持

公开 `prim_func → source xDSL → physical IR → autosync → 实际 IR 独立终检 → ASC → CANN`
路径已经接通默认 dense f16 Flash Attention。数学与政策只在
[examples/flash_attention/attention.py](../examples/flash_attention/attention.py)。
旧 `cube.task/vector.task`、公开旧 builder/tracer 和旧 FA 入口已删除，无兼容层。

Sq/Sk 是 runtime 参数，不按长度枚举；query、KV、固定 child 时钟保留。默认 BQ128
有76个 native模板、12个call、20组AIC/双AIV。Vector先softmax再update；PIPE_ALL仅
actor返回3处，没有query drain或额外搬运。

已接通纯整数局部 canonicalize/CSE、constexpr assert、dtype/head输入合同、受限
parent/child event ID借用。借用仍要求完整通知消费及FIRST/LAST边界，不是任意动态共享。
当前普通编译仍执行独立完整终检，本轮没有改成新方案中的 normal 模式。

## 最近完整设备资格：历史 BQ128

产物：外层 `runs/refactor/cv-order/build/`；ASC SHA256：
`e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`。
CANN9.1/dav-2201/-O2。它是明确的历史产物，不等于当前HEAD已重新完成资格。

| 项目 | 结果 |
|---|---|
| 源码→ASC | 141.225s；planning79.685s、final58.583s |
| 同次完整构建 | 147.301s，未超300s |
| 小规模全输出 | Sq:Sk=128:1024、384:3072、768:1024、512:2048；误差≤3.052e-5 |
| 大规模采样 | 4K/8K同一binary；误差≤1.526e-5，全输出有限、输入/保护区通过 |
| 4K三批device task p50的中位数 | 605.780µs；旧政策822.660；独立AscendC1191.220；ACLNN512.152 |
| 性能资格 | 比旧政策快26.4%，仍比ACLNN慢18.3%，未达≤5%目标 |
| bf16 GQA Hq8/Hkv2 | 同一DSL政策CANN/shared通过148.596s；没有设备资格 |

原始正确性/计时记录在 `runs/refactor/cv-order/device/`、`performance.json` 等，
指标口径及ops-transformer源码依据见[attention-performance.md](attention-performance.md)。
硬件/profile证据不能仅由编译器架构设计替代。

## 暂停候选：BQ256/BM128 K/V复用

example已有配对query政策、runner合同和小型CPU回归。其源码请求预算GM→L1为520MiB，
对照历史默认PMU为776MiB、ACLNN为544.3125MiB。**520MiB不是设备实测收益。**
候选没有成功生成完整ASC/shared binary，没有CANN或设备结果。

五次拒绝均发生在300秒前，目录在 `runs/refactor/l1-reuse/`：

| 构建目录 | 秒 | 拒绝位置 |
|---|---:|---|
| build | 22.697 | sequence endpoint预算 |
| build-shared-sequence | 22.184 | fixed child endpoint relation预算 |
| build-sparse | 59.493 | parent选择期间guarded case预算 |
| build-incremental | 89.502 | child event lifetime期间guarded case预算 |
| build-singleton | 99.837 | parent借用期间guarded case预算 |

`rejected-builds.json`只列前四次；第五次以`build-singleton/build-result.json`为准。
保留失败数据，不继续扩大预算或重跑候选。case probe只是既有只读诊断，不能称成功构建。

## 已合入的通用改动与剩余问题

| 提交 | 已实现内容 | 没有证明什么 |
|---|---|---|
| 7966bce | 受限parent通知借用fixed child event空档 | 任意条件/资源图均可共享 |
| 743ae8c | 相同完整访问pattern共享sequence transfer | 逐物理区间/义务成本消失 |
| 7a32407 | 固定child稀疏投影 | 父层条件/端点总量已受控 |
| 6b6c4cf | QueryCompletion按endpoint/domain合批coverage | GuardedCompletion也全改成合批 |
| 5cc71fa | 增边时保留完整覆盖正答案 | partial/false及新命题无需处理 |
| fdff8c5 | singleton端点跳过无关边界枚举 | 非singleton边界成本已解决 |

既有针对性测试保存在各提交中；本轮没有重跑，不把历史通过当作新重构通过。

当前明确问题：

- normal pipeline仍多次建立访问、child证明、父层完成关系和资源事实；甚至final内部
  对同一未变快照重复验证。分层不变量及所有权没有贯穿。
- 固定child计数、访问行片段、条件域、period和query静态窗口仍放大状态；不随runtime
  长度展开并不代表规模合理。诊断出现8469节点、117189条件事实、约153万record访问。
- 当前HEAD默认FA的冷编译分布/RSS未知；历史141秒不能冒充最新测量。
- 一般动态ownership/task stream、continuous跨query状态、稀疏激活尚未完成。
  causal/tails/decode、MQA、paged/sparse/persistent等变体没有完整资格。
- 数学正确性、事件资源合法性、进展和设备性能是不同验收；不能靠一次verifier通过
  或取消verifier解决全部问题。

## 下一步开发范围

按[计划R0–R6](refactor-plan.md)先建立分析所有权与不变量，再替换访问/完成表示和资源
求解，最后切换normal流程、删除重复生产路径。不得先关闭现有final，或恢复算子调优。

后续每步报告实际工作量、保持/失效的合同和有限回归；不再用“继续测试/继续优化”代替
具体解释。算子性能问题保留，待编译流程具备可用性后按新的任务安排处理。
