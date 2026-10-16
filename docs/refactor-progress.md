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
| 循环进展 | 内部 stream rank、实际 open 通知端口与参数化 queue/controller 接口分开；父层只消费 child.interface | 大例完整构建尚未完成，任意条件投影不据此宣称支持 |
| 最终 IR 与 ASC | 真实 SCF/Notification/Address SSA；单份 AIV 代码，无隐藏全局 barrier | 完整嵌套/CV、默认 FA 以及相关 CANN/设备资格未恢复 |

进展阶段不再向父层导出隐藏 native。每个 region 只导出 ancestor 拥有匹配的实际通知端口及自身 queue 输入、queue/controller 阻塞输出；实际 native、窗口和 rank 留在本地 witness。一次导入绑定同时处理端点与 guard，rank 与组合图复用同一结果。

跨调用 gate 由拥有时钟的祖先按真实坐标连接，不能在 child 拼成零步边。独立 n=1/n=2 外层等待反例验证了这一点；原有父层组合已经能正确拒绝该例，未宣称修复新的漏检。额外显式复制 queue 调用实例的候选已经删除，没有保留第二条路径。这些接口只描述可能阻塞/发布，不代表异步完成或 buffer 退休。

## 本次验证及限制

- 原 20 项小进展/接口合同通过；最终受调整影响的新增 n=1/n=2 两项通过。独立 queue 接口文件共7项已通过。CV 域检查只验证参与者与缺失贡献拒绝，不代替完整 CV 构建。
- 独立手工最终 IR 模型及 native 合同的小组已执行；修复了迁移暴露的 packet 身份准入缺口，并将 block8 recipe 断言改为实际 SSA root 和不重叠字节区间。
- 后端定向检查验证实际地址、控制和通知 SSA、MAD 首块初始化。新的通用整数标量pass已经接入物化后的实际管线，6项规则/接线回归及1项期限机制回归通过；前端与源语言相关57项通过。host C++ 检查不能替代设备执行。
- Attention 元数据检查已迁移共享 body 布局；新增实际编译器 launcher→runner ABI 回归。元数据 fixture 不代表完整 FA 编译。
- 七项工作量记录/schema 检查通过。清退后的全局 collection 成功；没有执行完整测试集。
- 有限 GM→UB→Vector→GM 用例已通过公共链路与 CANN，产物在 [finite](../../runs/refactor-no-legacy/finite/)。随后实际 [Scale 双缓冲示例](../examples/vector/scale.py)也通过同一路径和 CANN，产物在 [scale-current](../../runs/refactor-no-legacy/scale-current/)，运行长度仍是参数。这两项都不是设备或 CV/FA 资格。
- 已检查 Scale 最终 ASC：保留3个native模板、输入/输出双槽及4条ready/release通知，没有增加搬运。event ID仍含较多min/max和重复取模；新通用标量pass只处理可安全应用的stock规则，没有宣称输出已充分简化或性能已达标。
- 一次期限机制测试中的 `compile(scale, timeout_s=3)` 曾到达期限。该机制测试改用有限三native输入，原现象仍记录为待后续编译性能分析项；更换测试输入不是性能改善。
- 实际批量 GEMM 的6个native模板、L1/L0/L0C多缓冲及Cube/Fixpipe经公共链路和CANN离线编译通过，产物在 [gemm-current](../../runs/refactor-no-legacy/gemm-current/)。这是独立矩阵批次，不是完整attention或多层K-reduction资格。
- 本轮没有设备执行或性能结果。

最近完整嵌套 CV 构建触及300秒硬期限，worker被终止，已确认进程组无残留；状态见 [nested-cv/build-result.json](../../runs/refactor-no-legacy/nested-cv/build-result.json)。没有发布ASC/CANN产物，现有日志不足以定位全部耗时。本批停止，不继续FA或其他构建候选。

更早的完整FA同步曾达到300秒上限，未进入CANN；窗口/进展重写期间的嵌套CV局部检查也曾达到45秒测试上限。它们都不是硬件event容量不足的证据。代码结构改造已落地，但完整CV/FA资格和编译耗时问题仍未完成。

旧完整 QK/PV 共享 L0C 跨 query 的全部启动/稳态/收尾场景尚未逐项在新路径取得资格。通用硬件反例的覆盖不能代替这些实际算子测试。

## 后续顺序

1. 保存当前单一路径、region接口和通用标量pass检查点，不恢复旧实现。
2. 在这份结构已稳定的代码上，定位未完成构建的阶段与编译工作量；先做有界的阶段诊断，不继续完整算子候选批次或无界测试。
3. 根据实证处理编译性能问题并完成嵌套CV、默认FA的功能与CANN资格。当前的超时不能算功能通过，亦不能算重构目标全部完成。
4. 之后再恢复算子性能工作，比较最终ASC、AscendC/ACLNN、profiling与硬件理论指标。

源码、回归和产品文档只提交独立compiler Git；外层运行数据不提交，不推送或发布。
