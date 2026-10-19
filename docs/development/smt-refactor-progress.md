# 统一语义域与 SMT 重构进展

2026-10-19。实施方案见 [smt-refactor-plan.md](smt-refactor-plan.md)。
代码审计基线为 `fb39c663ba5efb864ec6c73a4717f8184b75a834`，方案提交为
`bdf65313f03bb66fe1c45ebe8ef1a958246bd02a`。本文记录实际状态，不代替验收。

## 当前资格

| 阶段 | 状态 | 已有内容与缺口 |
|---|---|---|
| P0 基线与反例 | 部分记录 | 已保留接手补丁、旧调用日志和 FA 失败；完整反例矩阵仍需核对 |
| P1 整数查询 | 主体已实现，FA ASC 已恢复 | 已接通正式 Z3 依赖、原式数学查询与覆盖；公共例子的完整资格仍待集成 |
| P2 域与代次 | 待实施 | 尚未接通统一的 typed symbol、occurrence 与坐标映射 |
| P3 存储与值 | 待实施 | 保留现有合同，尚未完成计划中的收敛 |
| P4 同步与资源 | 部分修复 | 已保留释放 credit 的物理身份，恢复默认 FA 分配；共享兼容关系收敛仍待实施 |
| P5 最终链路 | 待实施 | 尚未完成终端边界和旧推理实现删除 |
| P6 集成资格 | 未取得 | 默认 FA 已有 ASC；尚无本次完整 CANN、设备数值或性能资格 |

## 2026-10-19 接手与监督恢复

原 pi 调用 `49ecf69244bf` 的监督进程丢失，但 pi 及其诊断子进程继续运行。
接手时没有阶段提交。按 PID 和进程开始时间检查、停止所属进程后保存现场，
再用同一显式会话续接为 `02a05708df5b`；随后停止该调用，将实施改为按阶段
交付并审查。模型保持 `deepseek/deepseek-flash`，thinking 为 `high`。

现场材料位于工作区外层：

- `runs/pi-subagent/49ecf69244bf/recovery.json`：进程身份、清理结果。
- `runs/compiler-architecture-audit/pi-recovery-20261019/partial-work.patch` 和
  `smt.py`：首次接手时的工作树快照。
- 同目录 `validation-extract.json`：从原会话抽取的命令与错误摘要。
- `runs/pi-subagent/02a05708df5b/events.jsonl`：续接调用的原始事件记录。

这些材料仅供诊断，不是产品依赖。已保留所有产品修改，没有将未验收实现
回滚或宣称为完成。

用户随后明确停止 pi，由主代理直接接手。最后调用 `644c143812b8` 已取消并
回收，未再委派。其最后提出的稳定 cell 单向 handback 改动没有覆盖循环回绕，
已移除，没有将该提案当作可用的 ID 共享合同。

## 已观察的验证结果

续接调用的日志有一组整数、索引和 execution lowering 短测 `144 passed`。
这是局部证据，尚未构成完整 P1 或编译链验收；测试入口和删除的私有实现
测试仍需按原式语义审查。

默认 FA 的实际构建已经出现后续失败：`notification_106` 无法在 AIC 的
`MTE1 -> MTE2` 本地 bank 中分配一个连续 ID，容量为 8，包含原有预留。
分配器尚未证明足够的生命周期共享关系；尚未生成可用于本树验收的完整 FA ASC
或进入 CANN、设备验证。不得扩大硬件容量、删除必要同步或排除该项来通过。

当前优先事项：接通并验证原式定义性与数学查询边界，修复默认 FA 完整构建，
删除已被替代的推理主体，形成第一个有明确验收结果的代码提交。后续每个阶段
应追加提交、实际命令、产物身份、删除项和未完成内容。

## 主代理补充的整数语义检查

`tests/analysis/test_smt_semantic_boundary.py` 的 12 项检查已在当前工作树通过
（0.54 s），ruff 检查和格式检查通过。覆盖数学零不能授权负数 unsigned
求值、动态零除数、短路 LAST 的顺序、代数恒等式中的溢出中间值、存在性与
蕴含结果分离、前提与输入环境隔离、相邻字节范围及完整 64 位余数掩码。

命令为 `.venv/bin/python -m pytest -q tests/analysis/test_smt_semantic_boundary.py`
（从产品根目录使用工作区 `.venv` 的绝对路径执行）。原始日志在工作区外层
`runs/smt-refactor/parent-smt-semantic-boundary-fixed.log`。这组检查尚依赖未提交
的查询服务，不代表默认 FA、完整编译链或设备资格。

## 主代理完成的第一段实现

正式依赖为 `z3-solver==4.15.4.0`。`IntegerFacts` 的非平凡关系与 guard union
coverage 使用同一查询服务，输入事实在每个服务中建立一次，查询通过 push/pop
隔离前提；数学关系、原式定义性和机器整数范围保持分离。查询时限消费当前
编译剩余时间，整个构建仍由独立进程组 watchdog 限制到 300 秒。

删除 `guard_cover` 的手写整数/余数分支搜索和无生产消费者的 `modular.py`。
测试保留原始 guard 的独立有限整数求值及遗漏分支、余数相关性和非法 unsigned
反例；删除的是旧证明树结构与计数器形状的镜像检查。

FA 的后续资源问题来自局部释放通知的前缀合并：多个物理 cell 的边界释放被
组成一个新 ownership 集合，无法沿用原有循环内 credit。新增统一的
`StorageCell.physical_key`，局部 WAR/WAW 前缀合并保留相同物理 ownership
集合，仍经原有配对、循环 credit、队列和 bank 检查。没有新增共享证明、扩大
ID 容量或改变 native issue order。

验证与范围：

- `runs/smt-refactor/owned-smt-cover-02/build.log`：206 项通过，含实际默认
  FA admission、索引、整数、有序定义性、guard cover 和已有循环共享正反例。
- `runs/smt-refactor/owned-credit-ownership-02/build.log`：29 项通过，含八个
  物理释放 cell 的嵌套循环重入；独立有限事件/字节模型覆盖两次调用及空外层循环。
- `runs/smt-refactor/owned-definedness-dag-01/build.log`：9 项原式 DAG 与有序
  定义性回归通过。
- `runs/smt-refactor/owned-fa-source-01/benchmark.json`：当前真实默认
  `attention_target` 的单次冷进程 source→ASC 通过，26.6407 秒；ASC SHA256 为
  `2eb5d8368c28ca7d4e2bc5f302ed797c2c26dcbf5989ebfe2d8a3c00405554e8`。
  它不是五次冷进程性能验收，且尚未达到 p50≤5 秒目标。

当前 source 为 85 个 Python 模块、26,976 行，比审计基线净减 54 行。
这只是第一段替换；尚不能称大幅简化，终端两个 pass 等重复推理仍待删除。
上述日志均属于工作区外层 `runs/`；ASC/IR、输入、编译器树身份和阶段计时
随实际构建保存。CANN 与设备状态未被这些 Python/ASC 结果代替。
