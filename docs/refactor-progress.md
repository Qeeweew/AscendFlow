# 当前实现、资格与未完成项

2026-10-17。同步生成已替换旧 Path/progress 链路，公共 DSL→IR→ASC 只有一条路径。
**重构尚不能宣布完成：默认 FA 冷编译目标尚未达到。** 下面分别记录实现、离线编译和性能；没有设备验证或算子性能资格。

## 代码与用户工作

接手 HEAD 为 `01ddce954b5bc269dc0d39e65d96bc1702164742`，实验快照为 `1231f4f`。
接手已有的 `tests/models/structured_events.py` 修改、`tests/test_model_reference.py`
和 `analysis/generation_order.py` 原文保存在外层
`runs/generation-redesign/incoming/`。独立 pre-sync 对照已集成；临时 generation_order
原型已由当前规则替换，没有复制进 legacy。实际提交身份以产品 Git 为准。

外层没有 Git，未创建根 Git；reference、旧 TileLang 工作区和容器没有修改或重建。
CATLASS、ops-transformer、ops-nn 的实际 HEAD 已再次核对，见 [source-notes.md](source-notes.md)。

## 已实施

- [生成规则及事实归属](generation-rules.md)：逐项输入、建立者、产出、消费者和失效条件。
- 真实 native capture/acquire、有序 unsigned guard、坐标往返和完整 mode-2 quorum。
- 分层 physical-step/phase 准入，拒绝匹配但互相先 Wait 的死锁；父层只消费实际边界。
- 由相邻值/存储 lifetime 构造稳定 cell credit。非稳定地址使用有限使用或明确 queue handback。
- credit 的全部成员进入有限 bank 台账；新 ID 共享检查全体成员，保留配置与最终 drain。
- 两个 AIV 共用一份 Vector body，由 runtime sid 区分地址、私有 UB 和 quorum 贡献。
- 物化不改变 native、地址、循环身份或 issue order；ASC 只打印最终 IR。
- 已删除旧 `event_progress`、`event_paths`、`event_lifetimes`、`progress_*`、
  `structured_event_sharing` 及其 solver 形状测试，无兼容入口或第二编译链。

`generated-lifetimes` 是来源标签，不能授权外部 physical IR，也不替代规则论证、
独立模型或硬件证据。输入与规则前提检查保留，未把必要检查搬进 debug。

## 独立回归

实际最终 IR 模型检查队列、controller、二值消费、mode-2 四位 ID/十五贡献容量及字节 RAW/WAR/WAW。
独立 pre-sync 快照提供预期 native、地址和读版本；生产生成器不提供 oracle 的预期边。

默认 FA 模型使用原始默认政策，Sq/Sk 未特化，有限实例为
`(128,1024)`、`(128,2048)`、`(128,3072)`、`(512,2048)`，检查 group 0 和 19，
包含空 owner、两个 query invocation、双 AIV 和多 KV 复用。它不执行浮点运算，
不外推所有输入或设备性能。`invocations` 表示 controller 原始 body 重入，
不隐含 mixed kernel 的跨 actor launch 完成；顺序 launch 由 runtime 排序。

首轮全套为 1566 passed、29 failed、1 error（146.17 秒）。用原 HEAD 独立复现的
相关旧测试为 51 passed、21 failed、1 error（15.47 秒），记录在
`runs/generation-redesign/baseline-failures.log`：主要为旧诊断字符串和两项从未接通的
positive 测试（prefix、循环内 allocation）。现已明确它们的拒绝边界，并修正
一个未对齐 alias fixture，使其实际触发 alias 检查。没有把这些拒绝算成 DSL 支持。

最终全套回归和冻结产物记录正在生成，完成后在本页补齐。

## 当前编译证据与计时边界

默认 FA 政策保持 B1/Hq8/Hkv8/D128、BQ=BM128、BK1024、CK128、VR8、GROUPS20、f16。
source→ASC 包含源码读取/构造、全部分析/化简/输出和内部快照；import、进程总耗时、
文件发布和 CANN 分别记录。没有跨编译结果缓存。

冻结前的公共源码构建已取得四项 IR/ASC/对象/库（`runs/generation-redesign/qualification/`）：

| 算子 | Python 内部编译 | CANN | 状态 |
|---|---:|---:|---|
| Vector scale，B64 | 0.307s | 5.348s | 离线编译通过 |
| 默认 batched GEMM | 0.561s | 5.417s | 离线编译通过 |
| examples/cv/nested.py | 0.954s | 5.696s | 离线编译通过 |
| 默认 FA | 11.507s | 8.965s | 离线编译通过 |

上述 Python 内部时间不包括源码构造，不能充当完整冷编译时间。之后修复了完整
credit 台账、共享 view/实际位置事实和精确整数系数；最终资格将绑定保存后的提交。

完整 source→ASC 单次 probe `fa-coefficient-probe` 为 **10.6469s**；此前
`fa-current-probe` 为 67.5209s（47.2971s 在旧的逐端点队列准入），均是各自源码的
真实成功 ASC 构建。带 cProfile 的 74.5243s 仅作诊断，不能当冷编译成绩。
当前未达到五次 p50≤5s、max≤10s，未执行五次，不降低门槛。

## 支持与具体拒绝

支持的规则覆盖双缓冲 GM→UB→Vector→GM、真实最后 reader/WAR、slot/代次复用、
Cube/Fixpipe、参数化嵌套和相邻循环、共享双 AIV CV 以及本页默认 FA 结构。
数学/依赖前提、完整 quorum、phase rank 或有限资源不能建立时具体拒绝。

未知动态 alias/ownership 需要可验证输入合同；外部根的 disjoint pointer 合同保留。
数据相关控制、无范围的 scalar carry、未封闭的多时钟/稀疏 activation、prefix native
地址、循环内 allocation、continuous FA 状态边界仍不宣称支持。没有全局 barrier、
额外搬运、减少 overlap、调大旧 solver 预算或 FA 名称/shape 特判。

## 历史失败只作溯源

`91eb40d` 的 125.44s、`475f49b` 的 56.87s、Path 接入但共享树之前的 56.69s
均是失败耗时，未产生 ASC/CANN，不能与成功构建作提速倍率。
原证据在 `runs/scalar-cost/`、`runs/factored-progress/`、
`runs/lazy-progress/fa-default/` 和 `runs/lazy-progress/build-source.json`。
旧 `4b2c432` 的 shared L0C、`dbcef70` 的双 AIV 记录不提供当前资格。
