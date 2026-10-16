# 当前重构状态

本页是唯一当前状态入口。设计见[design.md](design.md)，阶段合同见[compilation-invariants.md](compilation-invariants.md)，执行顺序见[refactor-plan.md](refactor-plan.md)。

## 本轮顺序

用户已恢复实现，并明确要求：**先完成代码重构，结束后再分析编译性能。** 当前不做profiling、冷编译基准、缓存调优或算子优化；仅运行必要的语义回归与离线功能验收。每次完整算子构建的300秒上限继续有效。

本轮起点为`acb54b2`，当前检查点以产品Git的HEAD为准。整体重构仍未完成。工作区根目录不建Git。不能把依赖分析闭合、某个组件的测试通过，或历史产物当成当前完整编译器资格。

## 唯一目标主线

```text
source/native结构
  → 物理访问、范围、ownership、值与输入合同
  → region依赖（实际scope、generation、有效域）
  → 逻辑通知（capture/acquire、硬件机制、参与者）
  → 统一事件生命周期、进展与分配
  → 最终物理IR（真实通知、条件和event SSA）
  → ASC
```

同步选择不分配ID，资源阶段不重新发现内存依赖。有限子循环内使用与父循环再次调用时的复用属于同一资源合同；不能将有限child强制改成无限单时钟模型，也不能在每个child出口增加drain。

后端只读取最终物理IR，不再通过一份动作列表和一份展示IR分别解释执行。新增通知保持原native、地址和循环身份；已有事实继续用于其未改变的前提。

## 实际状态

| 阶段 | 当前实现 | 尚缺内容 |
|---|---|---|
| DSL与native结构 | 保留runtime循环、SCF scalar carry、Stage排步、完整SSA地址；AIV共用body并用sid | 任意动态控制/别名不据此宣称支持 |
| 物理事实与准入 | 原语合同、axis/byte/alignment、初始化、Tile/LastTile、ownership、配置和host整数合同 | 后续扩展必须提供明确输入合同 |
| region依赖 | RAW/WAR/WAW、ReadCarry、FIRST/LAST、动态slot与首末代条件，使用实际native端点 | 未证明的符号别名、非周期空child等明确拒绝 |
| 逻辑同步 | 循环内/跨循环关系统一为LogicalNotification；已删除新入口的local/boundary分派 | 完整算子资格仍依赖资源阶段收口 |
| 事件资源 | 精确窗口及内向外存在投影已接通；按实际region组合进展；有证据的有限ID共享 | finite/祖先端口与本层重复流的混合接口、仅由child接口引入的变动端口族尚未闭合 |
| 最终IR与后端 | 原地物化真实SCF/NotificationOp；后端改读实际SSA与通知 | 与完整资源结果联合验收 |
| 公开入口与旧路径退场 | 唯一source编译；删除旧物理分支/公共别名/CLI checker；M2的7模块移出产品 | 剩余旧planner及旧计数消费者仍待退场；必需示例完整资格未完成 |

默认FA已接入访问/准入/依赖阶段，**不代表当前新路径已经完成FA同步、CANN构建或设备验证**。其他Attention变体也不能继承这个阶段的资格。

## 已有功能证据及其边界

- 递归结构、原始轴/字节范围、ownership、初始化、Tile与短/零循环均有针对性CPU回归。组件回归不合计成全产品通过数。
- 新region依赖包含独立有限访问trace反例：双缓冲、RMW、首末代、短child与compound slot、只读family。有限检查不是任意规模形式化证明。
- 后端功能回归使用host C++/UBSan检查实际控制、地址、singleton AIV条件与guard内算术；不是设备证据。
- 在本轮最终IR收敛**之前**，公开source入口的Scale与独立批量GEMM曾通过CANN离线编译，产物在[runs/structured-loops/m4-public](../../runs/structured-loops/m4-public/)。该记录不能代替当前改动的重新离线资格。
- 本轮没有设备执行或性能结论。

## 最近一次集成的明确阻塞

- 在本轮窗口/进展改写前，最近一次默认 FA 完整同步达到300秒硬上限并以124退出；未取得完整资源结果，未到CANN。该候选已停止，不继续后续完整算子构建，不延长超时或回退旧求解器。
- 配对与ID代次负例已补：仅禁用Wait、删除中间代Wait、改变端点代次均不能通过。
- progress 已移除旧的root/FIRST-LAST整图投影。实际region保存局部rank、条件端口、child出口及controller调用顺序；多跳出口标明保守包络。已修复经过孙循环的真实阻塞路径漏出问题。复杂条件SCC及未证明的混合接口明确拒绝，不能宣称一般嵌套进展已经证明。
- 旧窗口仅从若干guard推断first/count，现已删除。新窗口阶段已接通精确membership、首末、后继和非空投影，并按实际scope先于action guard的次序处理表达式定义域。22项窗口数学/固定宽度反例及3项内向外投影测试通过；这些小合同不代表完整FA资格。
- 最后一轮小合同集成为30项通过，覆盖窗口组合、region进展、配对/代次、有限ID共享和公共产物隔离；2个生成循环ID用例因本批完整算子构建已停止而未运行。日志在[runs/structured-loops/resource-contract-integration.log](../../runs/structured-loops/resource-contract-integration.log)。这不是全产品通过数。
- 新最终IR有限模型及后端的独立子集38项通过；generated Scale/GEMM用例曾被生产progress拒绝，不能算模型已验证生成路径。M2迁移后的190项只完成收集，没有执行测试体。构建期限测试已改用有限tile隔离进程机制，并修复新报告缺少kernel字段导致的产物读取失败。

## 下一步保持顺序

1. 完成有限初始化/末次/祖先端口与循环重复流的组合，不再增加按算子或query命名的分支。
2. 迁移剩余旧native adapter、物理计数及benchmark解码器，随后移走尚留在产品中的旧planner闭包；旧语义测试还需要按当前IR与历史oracle区分。
3. 重新建立实际Scale、GEMM、嵌套/CV、默认FA的完整功能资格及必要CANN离线资格；当前没有新路径设备结论。
4. 整体结构重构完成后再分析编译性能。失败更快或拒绝更多输入不能作为性能改善。

## 完成结构重构的门槛

1. 内部与边界依赖使用同一通知表示，保留完整激活域；不再经过损失first/count/dynamic phase的适配。
2. 同一资源阶段解释有限调用内和跨调用的event生命周期、双AIV贡献及最终收尾；实际不足或未证明必须诊断。
3. 所有执行指令都在最终物理IR中可见，ASC没有隐藏同步、额外搬运或另一套位置解释。
4. Scale、Cube/GEMM、嵌套/CV及默认FA经过同一公开链路，保留原算法和排步合同；相关CPU反例与必要CANN离线验收完成。
5. 删除被替代的主线路径与适配，更新相应测试和文档。不能用保留多套普通后端的方式宣布完成。

以上完成后，再独立分析编译性能与执行性能。此前默认FA的53秒级旧编译结果以及约18%的旧设备差距只作历史基线；没有重新取得当前实现的性能资格。

## 历史记录

重构前现场保存在`bbc1458`。旧文档及来源保存在[外层历史归档](../../reports/compiler-refactor/20261016-before-structured-loops/README.md)；历史数据不纳入产品Git。暂停的action专用化简原稿保存在[外层未完成实验归档](../../reports/compiler-refactor/paused-action-simplifier/)，不作为产品pass。
