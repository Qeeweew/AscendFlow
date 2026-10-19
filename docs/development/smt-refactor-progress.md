# 统一语义域与 SMT 重构进展

2026-10-19。实施方案见 [smt-refactor-plan.md](smt-refactor-plan.md)。
代码审计基线为 `fb39c663ba5efb864ec6c73a4717f8184b75a834`，方案提交为
`bdf65313f03bb66fe1c45ebe8ef1a958246bd02a`。本文记录实际状态，不代替验收。

## 当前资格

| 阶段 | 状态 | 已有内容与缺口 |
|---|---|---|
| P0 基线与反例 | 部分记录 | 已保留接手补丁、旧调用日志和 FA 失败；完整反例矩阵仍需核对 |
| P1 整数查询 | 主体已实现，FA ASC 已恢复 | 已接通正式 Z3 依赖、原式数学查询与覆盖；公共例子的完整资格仍待集成 |
| P2 域与代次 | 部分实现 | 已接通 SSA 身份、显式 symbol 种类和 metadata/carry 捕获代次；跨模块坐标接口尚待收敛 |
| P3 存储与值 | 待实施 | 保留现有合同，尚未完成计划中的收敛 |
| P4 同步与资源 | 部分修复 | 已保留释放 credit 的物理身份，恢复默认 FA 分配；共享兼容关系收敛仍待实施 |
| P5 最终链路 | 主体已实现，集成中 | 终端条件查询已统一；逻辑模块与终端模块已隔离，后端只读取 executable plan；全链资格仍需完成 |
| P6 集成资格 | 未取得 | 当前默认 FA 已通过 ASC/CANN；analysis 636 项通过，整组仍有失败且无本次设备资格 |

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

## SSA 身份、捕获代次与编译内查询复用

第一段实现提交为 `2080d9d`。其默认 FA 的 Python→ASC→CANN 完整构建在
`runs/smt-refactor/owned-fa-cann-01/build-result.json` 中通过，31.4193 秒；
库 SHA256 为 `c49e976572e7a9ea5be4285478b729f970032ebe789a64a3e40676c592ed255d`。
使用 CANN 9.1.0、dav-2201，尚未设备执行。

本段将参数、group、SID、loop ordinal、metadata 与 carry 改为显式种类。
编译器符号绑定真实 SSA/control 定义；打印名字不参与定义身份，不再用 `$`
前缀判断 uniform 或执行坐标。metadata/carry 捕获实际 group、SID 与祖先 loop
ordinal；同时替换只应用一次，求解端以捕获坐标为自变量建立实际读取函数。
不同代次不能借用同一值，相同代次的等价坐标仍指向同一次读取。

完整输入事实快照相同的整数查询服务只在一次编译内共享，退出时释放。
原始表达式驻留仍只共享语法，不保存范围或正确性事实。报告新增 backend
查询统计；它计数的是消费者本地缓存之后的查询，不能当作全部 prove 调用数。

验证与实际限度：

- `runs/smt-refactor/owned-symbol-owned-queries-02/build.log`：89 项通过，含
  symbol/捕获反例、原式语义、实际通知 trace、ownership 重入、execution
  lowering 和 structured admission（含真实默认 FA）。ruff 和 diff 检查通过。
- `runs/smt-refactor/owned-fa-symbols-01/benchmark.json`：默认 FA 单次冷进程
  source→ASC 25.9319 秒，通过；ASC SHA256 与第一段实现相同。本段尚未重跑 CANN。
- 此次真实编译共用一个四符号输入环境，backend 调用 4,148 次、distinct 3,876，
  unknown 为零。条件简化 6.6621 秒、标量简化 2.7015 秒，仍有显著重复分析。
- 该结果不是五次冷进程验收，尚未达到 p50≤5 秒目标；P2 坐标接口、P3、P4
  资源分配职责及 P5 终端旧推理器的收敛继续实施，不将本段称作完整重构。

## 终端条件查询收敛（集成中）

终端 guard 保留紧凑区间/合取/并集语法及实际 SSA 原子；可达性、蕴含与覆盖
改为同一整数服务的布尔组合查询。删除手写差分闭包、整数 gcd 可行性、成对
余数枚举、递归 residue 搜索和覆盖减法。只按已证明不回绕的算术展开 affine；
非二次幂 unsigned remainder 仍要求实际入口的非负前提。移位或合并后的标量
是否能在公共入口求值，继续由原有合法性和 materializer 检查，数学证明不替代它。

初版仍逐一搜索旧算法的候选，默认 FA 虽通过却退化到 44.4670 秒。随后将
候选合并改为语法对齐、对实际覆盖一次查询，复用固定约束的编码；反例保留
固定的 solver model，仅在需要诊断时解码。没有保留旧 solver 作为 fallback。

标量区间与 affine 视图复用到前提变化为止；范围表经只读视图暴露，更新通过
显式接口。已查询的范围、父域或源域范围改变时失效，兄弟域保持隔离；给新
SSA 增加尚未查询过的事实不清空已有结果。定义被外提后，原 guard 范围必须
改为目标域范围。新增回归覆盖父域变化和源事实放宽，不能复用旧的窄范围。

当前证据（尚非完整 P5/P6）：

- `owned-terminal-smt-lazy-model-01/build.log`：69 项通过，含独立最终执行
  trace、重叠通知次数、两个 AIV、unsigned 短路、原式整数和失效反例。
- `owned-fa-terminal-lazy-model-01/benchmark.json`：单次冷进程 source→ASC
  24.0488 秒，通过；峰值 RSS 137,820 KiB。标量简化 1.6591 秒，条件简化
  7.2444 秒。仍未达到五次冷进程 p50≤5 秒目标。
- 生成 ASC SHA256 为 `2f20a86da3ebab1ddd22470aa8a0c69c26a8ded10d3470532f333cb49d113fd8`。
  FA 最终通知 127 条，前一版本 133 条；尚待本段 CANN 与设备验证。
- 完整回归正在集成，已观察到失败；未将局部通过记为全量资格。

## 域接口与前缀安全的集成修正

同一物理 cell 的不同 writer 不能因同处一段 queue prefix 就合并 ready：
后一个 writer 可能等待前一个 reader 退休。前缀合并现保留真实 value writer
身份；release 对应其下一 writer。未知 alias 不提供可选合并的独立性前提。
原有独立 byte/event 检查恢复了 interleaved scratch 的合法流水。

runtime 参数的 int64 上界改为由实际 SSA 类型提供给查询服务，不再填入
逻辑 `Fact.maximum`。机器表示界不是 dense reader 的空间收缩上界；混用曾将
动态整条读取收缩到巨大的静态区域，丢失 nested copy 的值区分。原有动态
容量拒绝和 nested copy 的独立重入验证均保留。

选定祖先后，同步 family 的 count/residue 同时特化；处于祖先之外的 invariant
producer 也使用实际可获得的 selected consumer header，不能留下另一次
调用的裸 ordinal。`owned-selection-integrated-01/build.log` 的 79 项通过。

域配对、互斥和可达性改为完整合取/存在性查询，删除逐个 predicate 取反的
循环。服务可复用已经取得的 SMT model，但必须重新检验该 model 满足完整
查询上下文；它仅给出存在见证或反例，不能证明全称性质，也不补全未知值。
报告区分实际 solver check、model reuse 和完全相同查询的 cache hit。

当前验收边界：

- `owned-analysis-integrated-01/build.log`：analysis 目录全部 636 项通过。
- `owned-query-reuse-01/build.log`：182 项数学/原式/布尔/符号/条件与标量回归
  通过。与其他分组有重叠，不将各组执行次数相加当作 unique test 数。
- `owned-fa-current-cann-01/build-result.json`：当前默认 FA 完整 DSL→ASC→
  CANN shared library 通过，27.2882 秒。构建时另有 CPU 回归在运行，因此
  该数字不是冷编译性能比较。ASC SHA256 为上节的 `2f20a86...`，SO SHA256 为
  `4bd5349ba9b34a76b144872c03464720ab52dcbf6a3c1f3070dddb9d511c2452`。
- `owned-fa-checkpoint-01/benchmark.json`：当前默认 FA 单次冷进程 source→ASC
  22.3002 秒，通过，ASC SHA256 与本段 CANN 构建相同；仍未达到五次冷进程
  p50≤5 秒目标，也不将本次单次结果当作 p50。
- `owned-causal-diagnostic-03/worker-result.json`：真实 causal FA 的诊断构建
  source→ASC 52.5652 秒通过，两套输入环境均无 unknown；不是干净计时。
  随后完整回归的因果 FA fixture 也构建通过（61.6561 秒），并继续执行了
  实际最终 IR 的有限 byte/event 检查。
- 两次整组 pytest 都达到整组 watchdog 的 300 秒上限，已清理进程；没有
  获得整组通过结果。第二次执行至 1,138/1,930 项时停止，失败节点完整保存在
  `owned-integration-02/failure-nodes.json`，没有将未执行项或错误计为通过。
- 尚有两个真实同步问题待处理：八 panel 的 local prefix 在 V→MTE2 bank
  分配不足；可能为空的 child prefix 尚未建立 same-engine source→fence
  的层级 issue order。另有一项 DMA 拒绝测试的旧诊断正则需要更新。分组
  复现见 `owned-integration-failures-03/build.log`；仍保留原测试和必要检查。

当前源码 86 个 Python 模块、27,268 行，比审计基线增加 238 行。旧条件证明
主体确已删除，但身份、显式结果和缓存失效合同也增加了代码，尚未实现整个
编译器的大幅净简化。P3、资源共享收敛、最终 executable plan 边界、五次冷
编译目标以及新代码的设备数值/性能验收均未完成。

## 层级顺序与一致 ownership 集合

空 child 的失败来自不可达的循环回边：先证明局部步数相等，会在空 activation
上取得空真值，再按词法位置错误地判为逆序。现在对完整层级词典序关系做
Bool/Int 查询，不将空域上的中间相等关系当作后续决策。原始 guard 定义性仍
由原阶段准入；空循环端口、零次透传和实际同步指令没有删除。

局部释放前缀使用统一的 `StructuredNeed.ownership_key`。多个 cell 可以共同
释放，但必须在全部候选片段（包括 steady 和 selected boundary）中保持同一
完整 ownership 集合；仅边界合并、内部仍分散的候选保留各自 credit。由此恢复
八 panel、双槽程序，同时保留八 cell 嵌套重入的已有约束。cell channel 的
手写等式替换和逐个谓词取反分离查询已删除，改用一次完整的共存查询。

分配器将成对兼容关系与候选 ID 搜索分开：固定成员、域和生命周期的关系只
建立一次，包括已拒绝的可选共享；成功关系保留真实 consume→Set handback。
报告增加 `sharing_pairs` 与 `sharing_reuses`，不扩大 bank 容量或删除预留。

`owned-sync-integration-01/build.log` 中同步目录、资源合同与 AST Cube 分组
共 256 项通过，238.18 秒。该分组已包含默认与 causal FA，且没有排除原失败。
这次分组在兼容关系缓存接入前启动；缓存接入后的专项结果另行记录。
回归新增 child count 为 `parent%4` 的多次执行，并在八 panel 用例中检查两次
调用、循环回绕和删除真实释放 Wait 的反例。DMA 测试仅更新拒绝消息正则，
仍检查非法 GM→L1 transpose。

缓存接入后 `owned-resource-relations-01/build.log` 的 34 项通过；共享测试直接
记录真实 `share_events` 调用，要求同一固定成员对不因候选 ID 重复建立关系，
并继续执行独立 byte/event 重入检查。ruff、format 和 diff 检查通过。

## 显式终端边界

同步后逻辑模块保留原 count/stage 和 admission/dependency/resource 记录。
终端转换使用独立模块，一次显式映射 native 身份、实际 SSA 与已准入的范围。
`ExecutablePlan` 只携带最终模块、native 的 operand geometry/packet、参与者及
只读最终 SSA 范围，检查 native 一一对应及逻辑控制已消除。它不是外部输入
的 verified 标记；公共 compile 仍只接受 source，debug 输出仍含禁止执行声明。

后端删除 `NativeMemoryFacts`、`ControlFacts`、`MemoryTemplate` 与原始地址
表达式的输入依赖；native 格式化始终从终端 operand 取得地址与 scalar。
`ScalarUses` 只接收实际模块和 native 参与者，lowering/输出不再借用旧 memory。
独立 byte/event 模型改为重新读取实际最终 IR，不借用旧逻辑 frames 作终端事实。

`owned-terminal-boundary-02/build.log` 中 backend、execution lowering、实际同步
语义、空 child、八 panel 与依赖分组 170 项通过，52.92 秒。新增的隔离回归另见
`owned-terminal-isolation-01/build.log`：修改保留的逻辑 native scalar 和地址，
禁止再查询逻辑坐标与分析入口，终端 ASC 仍保持相同。未将这两组外推为全量
或设备资格；真实默认 FA 的冷构建与 CANN 继续验证。
