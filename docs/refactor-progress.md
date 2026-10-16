# 当前重构状态

本页是当前状态入口。合同见[编译不变量](compilation-invariants.md)，实施顺序见[重构计划](refactor-plan.md)。

**先完成代码重构，再分析编译性能。** 算子优化、profiling 和冷编译基准尚未恢复。整体重构仍未完成，不能以代码删除、小合同通过或历史产物替代当前完整资格。

## 已收敛的产品结构

```text
DSL/source
  → 保留控制与地址 SSA 的 native IR
  → 物理访问、准入与 ownership
  → region 依赖
  → 逻辑通知
  → 精确事件域、生命周期、进展、资源分配
  → 实际同步 IR
  → ASC → CANN
```

- 公共编译只接受 DSL/source；后端只打印最终 IR 的实际控制、地址、原语参数和通知。
- 按最新要求，旧 planner、scope/query/loop/sequence 路径、过渡解码器和旧物理协议直接删除。没有兼容 wrapper，也没有测试目录中的历史求解器。实现历史由 Git 追溯。
- `physical.py` 仅保留当前 native、配置、地址、参与者、循环和通知。`NativeGeometry` 只表达完整静态 footprint；原语格式化不再解释旧 Ref、prefix/count 地址。
- 测试清退了 137 个旧文件，保留当前 source/IR 合同、几何与 packet 准入、实际 SSA recipe、独立字节/event 模型。没有将删除旧测试描述为新链路全部通过。
- 基准工具统一读取当前产物；`source-work-v1` 区分模板清单与算法工作量。旧运行期计数器已删除；尚未重建的 native、搬运与同步请求计数明确标为不可得，不能据此报告 native 吞吐或带宽。
- Attention runner 检查一份共享 Cube/Vector 控制结构及各参与者的私有存储；host launcher 保留源 buffer 参数名，实际编译器输出与 runner 的命名 ABI 有独立回归。

## 同步重构的准确边界

| 部分 | 已实现 | 尚缺资格或结构 |
|---|---|---|
| 控制与访问 | runtime 循环、Stage、标量 carry、完整地址 SSA、物理范围、初始化与 ownership；共享 AIV body 使用 runtime sid | 未证明的动态别名/ownership 明确拒绝 |
| 依赖与通知 | RAW/WAR/WAW、最后 reader、slot/generation、首末有效域；统一 LogicalNotification | 默认 FA 的完整资源结果未建立 |
| 事件域与生命周期 | Set/Wait 等价域、精确窗口及内向外存在投影、consume→reuse、双 AIV quorum、显式资源台账 | 一般重复 channel 的 ID 共享仍有限制 |
| 循环进展 | 内部 StreamProgress 与 finite boundary 分开；父调用保留真实 +1 步；prologue 正反例通过 | 对外 channels 仍携带内部 native 端点，真正边界摘要尚未完成 |
| 最终 IR 与 ASC | 真实 SCF/Notification/Address SSA；单份 AIV 代码，无隐藏全局 barrier | 完整嵌套/CV、默认 FA 以及相关 CANN/设备资格未恢复 |

进展阶段已移除重复的 stream channels 和冗余接口索引，每个 region 在一份组合图上顺序消去内部与边界节点。但当前 `_channels` 仍导出隐藏 native 的 Set→Wait 端点，父层再次接收这些端点，**不能称为边界抽象已完成**。

下一步只导出两类接口：ancestor 拥有匹配的实际通知端口，以及本 region 的 queue 输入、queue/controller 阻塞输出。内部 native、窗口与 rank 留在本地 witness；这些接口只描述可能阻塞/发布路径，不代表异步完成或 buffer 退休。

## 本次验证及限制

- 15 项小进展合同通过：原 region 反例、prologue 正反例、父循环复用与 CV quorum。CV 三项只验证实际参与者域和缺失贡献拒绝，不证明完整 CV progress。
- 独立手工最终 IR 模型及 native 合同的小组已执行；修复了迁移暴露的 packet 身份准入缺口，并将 block8 recipe 断言改为实际 SSA root 和不重叠字节区间。
- 后端定向检查验证实际地址、控制和通知 SSA、纯标量 pass 保持 effects、MAD 首块初始化；host C++ 检查不能替代设备执行。
- Attention 元数据检查已迁移共享 body 布局；新增实际编译器 launcher→runner ABI 回归。元数据 fixture 不代表完整 FA 编译。
- 七项工作量记录/schema 检查通过。清退后的全局 collection 成功；没有执行完整测试集。
- 当前有限 GM→UB→Vector→GM 用例经过公共编译链路及 CANN 离线编译，产物在 [runs/refactor-no-legacy/finite](../../runs/refactor-no-legacy/finite/)。它不包含循环或 CV，不作为双缓冲、GEMM、FA 的资格。
- 本轮没有设备执行或性能结果。

旧完整 FA 同步曾达到 300 秒硬上限，未进入 CANN。后来的嵌套 CV 小合同在取得 19 个通知及事件域后，进展阶段达到其 45 秒测试上限；之后未重跑该候选。两者都不是硬件容量不足的证据，也不构成当前代码的性能结论。

旧完整 QK/PV 共享 L0C 跨 query 的全部启动/稳态/收尾场景尚未逐项在新路径取得资格。通用硬件反例的覆盖不能代替这些实际算子测试。

## 后续顺序

1. 完成真正的 region queue 边界摘要，禁止父层重新导入子层内部图；独立验证跨引擎路径不丢失、父调用代次及短/零循环。
2. 通过同一公共链路验收双缓冲 Vector、Cube/Fixpipe、嵌套/CV、默认 FA；每个完整构建仍有 300 秒硬上限。
3. 必要 CPU 与 CANN 功能资格闭合后，才标记结构重构完成并分析编译性能。
4. 编译性能问题处理后再恢复算子性能工作，比较最终 ASC、AscendC/ACLNN、profiling 与硬件理论指标。

源码、回归和产品文档只提交独立 compiler Git；外层运行数据不提交，不推送或发布。
