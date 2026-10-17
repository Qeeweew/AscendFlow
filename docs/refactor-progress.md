# 交接状态与证据

2026-10-17。**用户已停止算法开发，要求重新设计Set/Wait生成。重构未完成。** 本次整理只保存代码、纠正文档并提供新agent prompt；不继续FA构建或算子优化。

## 1. 代码身份

| 版本 | 含义 |
|---|---|
| `475f49b` | 先前提交：共享AND/OR替换DNF，但仍提前组合区域路径 |
| `1231f4f` | 停止时源码/测试快照：惰性Path、hidden图、三值查询和共享source搜索树；**未完成实验，不是认可的最终架构** |
| 交接整理提交 | 在上述快照上纠正文档与源码说明；以`git log`查看实际HEAD，不修改同步算法 |

没有回滚或隐藏已有工作；没有新增legacy目录、兼容入口或第二编译路径。后续agent可以直接替换失败设计，不必维护这些内部API。Git快照仅供溯源，不代表功能资格。

## 2. 必须重新处理的根因

`select_notifications` 的函数合同明确为data coverage only；普通编译接着在 `allocate_structured_events` 调用 `EventDomains`、`prove_progress`、`establish_lease`，分配ID时还可能调用事件路径覆盖/共享证明。通知选择尚未提供完整、可组合的协议进展和生命周期保证。

此前迭代主要优化上述后置求解器。缓存、factored公式、惰性路径和搜索树没有解决生成算法的职责缺口。新方向见[计划](refactor-plan.md)。不能直接关闭检查；也不能继续把当前Path方案写成已经确定的未来架构。

`Compilation.report["verification_mode"]` 当前仍使用 `owned-construction` 标签；它描述产物来源/私有IR装配，**不能作为同步已由构造保证正确的证据**。相关消费点在`benchmarks/attention/run.py`和对应测试，后续应连同报告合同审查。

## 3. 完整FA构建的实际结果

默认`attention_target`：B=1、HQ=HKV=8、D=128、BQ=BM=128、BK=1024、CK=128、VR=8、GROUPS=20、f16；Sq/Sk为runtime参数。下表都是失败耗时，**不能用作成功编译提速比较**。

| 源码状态 | 结果 | 证据 |
|---|---|---|
| `91eb40d` | 125.44秒，父区域DNF备选超限 | [scalar-cost](../../runs/scalar-cost/fa-default/build-result.json) |
| `475f49b` | 56.87秒，父区域急切消元工作超限 | [factored-progress](../../runs/factored-progress/fa-default/build-result.json) |
| Path接入、共享source搜索树之前 | 56.69秒，已进入finite progress查询，路径见证BFS触区域预算 | [lazy-progress结果](../../runs/lazy-progress/fa-default/build-result.json)、[日志](../../runs/lazy-progress/fa-default/build.log)、[源码指纹](../../runs/lazy-progress/build-source.json) |

三次均未产生成功ASC/CANN产物。**最终`1231f4f`中的共享source搜索树之后，没有再次完整构建FA。** 不得把前一次失败或后续短测试当作该版本完整资格。

## 4. 停止前的局部证据

这些结果由上一轮执行，本次整理未重跑；各组有重叠，不能相加为独立覆盖率。

| 检查 | 实际结果 | 限制 |
|---|---|---|
| 端点/父子区域/进展集成 | 48通过，4.19秒 | 搜索树改动之前 |
| 资源与公共Vector/Cube链路 | 46通过，26.35秒 | 不代表默认FA或当前CANN资格 |
| 最后图、条件、case与接口小组 | 62通过，2.38秒 | 搜索树之后；有界/单元与小集成 |
| 条件事实清理 | 7通过，0.52秒 | 删除无生产调用的union/subset等旧辅助入口 |

三节点全部边mask/hidden集合、七级diamond、非空自环、旧快照不变、双AIV贡献及guard次序等测试有保留价值；测试期待应是硬件/语义合同，不能要求重构后保留旧求解器形状或缓存结构。

最近历史shared L0C模型/CANN记录为`4b2c432`，双AIV嵌套CV为`dbcef70`。它们只能说明对应历史源码，不代表本工作树已获CANN或设备资格。

## 5. 可用的诊断资料

外层`runs/factored-progress/`有实际父图与数学条件快照、拓扑重放、同形节点/消元顺序/吸收实验。固定输入曾为44 native、726 needs、221 notifications、142 cases、1 packet分区；父图284节点、2159条原边、150个隐藏节点。

这些资料说明后置求解成本如何产生，不要求新agent继续研究这些算法。纯拓扑或Boolean表示的结果只对相应抽象成立，不证明真实NPU程序正确；快照不能导入编译器充当已证明输入。运行数据、ASC、缓存和报告均留外层，不进入产品Git。

## 6. 下一位agent的入口

接手时先检查实际HEAD和未提交修改，再读[计划](refactor-plan.md)、[不变量](compilation-invariants.md)与[源码职责](source-layout.md)。先审计生成义务和复杂度，明确规则后直接实现；不默认恢复停止的后置证明器优化。
