# 当前重构状态

**重构尚未完成。** 唯一 source 编译链和进展职责拆分已落地，本轮继续把整数前提及派生事实归入唯一 owner。默认 FA 已从此前300秒超时推进到125.44秒明确诊断：父区域路径条件超过64个 OR 备选，尚未生成可验收的 ASC，也未进入 CANN。不能把这个失败耗时作为成功编译指标。

合同见[编译不变量](compilation-invariants.md)，职责见[源码组织](source-layout.md)，剩余工作见[重构计划](refactor-plan.md)。算子调优、打包和设备任务没有恢复。

## 已落地的结构

- 唯一入口接收 DSL/source，经 native、访问与准入、逻辑值流、实际 dispatch 存储生命周期、通知、事件资源、最终物理 IR、ASC 到 CANN。物理快照不作为已证明输入重新导入。
- `logical_valueflow` 保留 RAW、条件 KILL/UPDATE 与代次；`storage_packets/word/lifetimes` 按已证明闭合的 private 生命周期构造 PLE 相邻 WAR/WAW。边界共用 consumer j/source j-distance，不展开 runtime 次数。
- 进展分为区域合同、端点与顺序事实、数学条件、稀疏连通、rank 和组合编排。被替代的编排器方法、dense 消元、类型出口和重复测试准备入口已删除，没有 alias 或第二求解器。
- `IntegerFacts` 现在按完整有序 tuple 建立 owned context，递归范围/比较查询只引用该 context。换序、重复谓词及不同输入环境保持各自身份，不提前解释后续谓词。
- `_constraint` 惰性建立原始带因子的约束，`_difference` 惰性建立原来的 `bounded(lhs-rhs)`。相同原式在多个 context 中复用派生结果，但范围/比较答案仍绑定完整 context。旧模块 `_term` 删除，无兼容包装。
- 有序定义性按当前比较的原始节点 identity 遍历共享子图；左右操作数共用成功集合，下一个比较重建集合。原始 unsigned 前提、首个诊断及短路顺序不变。
- 算法仍在 examples；两个 AIV 共用 body/runtime sid，源算法、buffer、issue order、通知与资源预算未改变。

## 本轮证据

136项短回归通过（2.52秒）：整数关系、owned context、原式定义性和进展条件；加强一个观测断言后19项通过（0.95秒，与前组重叠）。规则正文经展开派生 helper 后的 AST 对照保持，Ruff和diff检查通过。新测试检查前提/环境隔离、异常重试、raw factor/unsigned语义和共享 DAG 工作量。

递归范围查询在0/1/64条前提的测试中，缓存key均不再哈希任何 premise Comparison；这只说明递归key的成本，首次建立完整context和实际约束扫描仍有成本。相同差值在不同有序context下不再重复构造，不意味着省略对应的证明。

`cf44eec` 的107项结构/公共编译/Vector-Cube/资源回归及26项重叠端点接线回归是上一提交的证据；本轮没有重跑完整suite，也没有设备资格。

### 固定输入的阶段测量

同一默认 `attention_target`、同一55秒协作/60秒外部诊断，无跨进程结果缓存。44个native模板、726条依赖义务、221条通知、142个case、1个packet分区保持不变。运行期 Sq/Sk 仍是变量。

| 阶段（秒） | cf44eec | 仅context与定义性 | 再加入派生事实 |
|---|---:|---:|---:|
| source→EventDomains | 22.42 | 20.47 | 17.88 |
| 第一个子区域有限端点关系 | 13.99 | 9.39 | 4.08 |
| 第一个子区域混合区域图 | 11.07 | 7.43 | 3.16 |
| 第一个子区域总计 | 28.52 | 19.31 | 8.20 |

最终诊断完成 `$loop2..5` 四个内部子区域；55.14秒在父区域 `$loop1` 的图构造阶段停止。累计1735个条件集合、18712次底层证明、8793次已有答案复用、3077次路径插入/组合。见 [progress-derived.log](../../runs/scalar-cost/progress-derived.log)。这些是局部诊断，不能作为完整冷编译或设备性能结论。

此前对8次端点关系的同步 profiling 记录了约17.8万次事实查找与4.9万次Expr构造，用于定位成本，见 [relations-profile.log](../../runs/scalar-cost/relations-profile.log)。曾尝试额外通用结构key包装，阶段耗时反增至39.51秒；该方案及测试已从产品代码完整撤掉。没有保留旧/新两套表达式路径。

### 当前完整 FA 构建结果

公共 CLI 使用默认 constexpr：B=1、HQ=HKV=8、D=128、BQ=BM=128、BK=1024、CK=128、VR=8、GROUPS=20、f16。总期限300秒，覆盖Python构造、ASC与CANN；未改变输入合同或排步。

结果为 **failed，125.44秒，returncode=1**：`attention.py:98:13` 的父区域在稀疏消元期间触及 `region-interface OR alternatives exceed their explicit budget`。见 [build-result.json](../../runs/scalar-cost/fa-default/build-result.json) 与 [build.log](../../runs/scalar-cost/fa-default/build.log)。子进程已结束，没有继续其他候选。源码及改动指纹见 [build-source.json](../../runs/scalar-cost/build-source.json)。没有成功ASC/CANN产物。

这里的64是同一端点对的路径条件备选数，不是64次运行期迭代。当前把 `incoming AND outgoing` 展开为合取备选的笛卡尔积；七级独立二选一路径就可能产生128个展开项。下一项必须改关系表示，不能提高该预算掩盖组合增长。

## 剩余顺序与边界

1. 将区域路径激活条件改为共享、因式化的布尔关系，保留 `old OR (incoming AND outgoing)`，避免在接口生成时展开所有路径组合。原始有序guard、数学激活条件与exact/projection证据分别保留；这是待实现任务。
2. rank与条件查询按需消费该表示，保留循环时钟、父调用绑定和原预算。因式化不能自动解决任意finite/stream联合存在投影，也不能把保守摘要当必然完成。
3. 完成上述结构与反例后，再恢复一次有期限的默认FA资格构建。成功后才比较其他算子的完整编译耗时；失败不继续下一候选。
4. private重排、动态ownership、周期reset等支持边界不扩大。未知关系不能当作安全，无全局barrier、额外搬运或算子特判。
5. 当前 shared L0C 最近完整模型/CANN资格仍为 `4b2c432`，双AIV嵌套CV为 `dbcef70`。局部提速或短测试不能替代新链路的这些资格。编译资格完成后再恢复设备性能工作。

## 复现

在 `compiler/` 执行受影响回归：

```bash
timeout --kill-after=3s 60s ../.venv/bin/python -m pytest -q \
  tests/analysis/test_integer_contexts.py \
  tests/analysis/test_integer_definedness_dag.py \
  tests/analysis/test_integer_facts.py \
  tests/analysis/test_progress_conditions.py
```

本次完整构建命令（当前已知失败，修复前不直接重复执行）：

```bash
../.venv/bin/python -m xdsl_npu.cli \
  --input examples/flash_attention/attention.py --entry attention_target \
  --output /workspace/xdsl-910b/runs/scalar-cost/fa-default --compile --timeout 300
```

产品文件进入 compiler Git；环境、缓存、构建产物和运行数据留在外层。
