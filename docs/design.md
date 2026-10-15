# 设计：结构化程序上的统一自动同步

2026-10-15 修订；实现保存点 `3f95139`。本文是下一部分重构的技术合同，
**不是已经实现的统一算法**。实际能力集中记录在[状态页](refactor-progress.md)。

## 目标与纠正

以 TileLang 风格的受限 Python AST 表达计算、存储、局部流水和 CV 多阶段流水。
专家排步与未来自动排步进入同一个物理访问和完成关系模型。自动同步优先，
性能通过最终 ASC、CANN 产物及设备 profiling 验证。

现有源码入口已经统一，region/call 共用增量关系查询与同步选择；叶流水的规划
和层次控制适配尚未统一。
继续按程序形状增加证明器，会把组合问题变成不断增加特例。下一次重构要替换
重复的语义和查询职责，而不是增加第四个形状分支或只把文件搬进同一个目录。

同时保留已核实的 native 契约、原语 lowering、ASC emitter 和反例。xDSL 是程序
的唯一可改写表示，分析索引引用它；不建立第二套可变程序树，不引入 MLIR 工程。

## 语言与执行边界

详细语法见 [DSL](dsl-multistage.md)。runtime shape/stride、循环界和 descriptor
是设备 SSA；容量、layout、dtype 和固定 stage 政策是 constexpr。常量循环界也
默认保留，只有显式且有预算的 unroll 才展开。输入长度不参与编译期任务枚举。

`Cube/Vector` 指定 actor，原语决定 Scalar、Vector、MTE1/2/3、Cube、Fixpipe。
Stage 表示逻辑任务的派发偏移，不等于执行引擎、buffer 数、event 数或完成屏障。
嵌套 child 有自己的迭代坐标；离开 child 控制流不能证明其异步访问已经退休。

整个路径为：

```text
Python AST → xDSL 结构化源码 → 保留控制的物理执行 IR
 → 访问/代次/完成约束 → 同步协议和有限资源分配
 → 物化同步 → 从最终 IR 重提取检查 → ASC → CANN
```

## 公共分析模型

下面是必要信息，不预先承诺类名或再创建一组空接口。

| 对象 | 必须保留的含义 |
|---|---|
| 操作模板 | 原生工作、actor/engine、源码位置、所属 region、派发位置 |
| 迭代实例域 | 符号迭代坐标、uniform 参数、activation/guard；不枚举所有实例 |
| 物理访问 | allocation 身份、view/layout 后字节集合、R/W/RW、开始/退休端点 |
| 代次与归属 | owner、slot 映射、generation；fresh frame 的已验证不相交关系 |
| 顺序约束 | 源/目标里程碑、两端实例关系、条件、覆盖范围和硬件强度 |
| 区域边界 | 所需初值、末 writer、尚未退休 reader、可见完成关系、存活通知 |

逻辑值依赖和物理访问分别保留。tensor 名不同不证明不别名；SSA 使用不是异步
完成；借用 view 不是 snapshot。DMA padding、scratch、Scalar 读取及 SPR 状态
也要进入 effects。未知 native effects 直接拒绝，通用 DCE/CSE/移动不能忽略它们。

ready、read-retired、release/credit、notification-consumed、最终 drain 具有
不同语义。每个访问只使用 dav-2201 真实可观测的完成点；没有 native 保证时
不能虚构比操作完成更早的 reader retirement。

## 一套算法如何处理层次和 CV

1. 一次遍历保留 region/actor/clock 层次，建立访问和派发索引。
2. 按共同 storage root 与精确范围求访问交集，建立符号 RAW、WAR、WAW 关系。
   环形复用使用真实 generation；已有初始化及零次执行路径一起检查。
3. 加入已核实的同流水顺序、派发与完成关系。跨 region 的查询使用相同约束
   表示；顺序、循环、条件的组合只负责域与边界，不各自定义另一套完成语义。
4. 检查必要关系是否已经被路径覆盖；只为缺口选择合法本地或跨核协议。
   每条删除或合并的等待必须有条件和范围均匹配的覆盖依据。
5. 为逻辑协议证明配对、复用和收尾，再分配硬件资源；检查等待是否阻挡必要
   派发。协议和 ID 选择改变时，重新检查受影响关系，不继承过期的证明缓存。
6. 从物化 IR 重提取访问、同步及事件生命周期，验证完整计划。planner 的
   “已证明”标记不能当作 verifier 输入结论。

区域组合可以自底向上计算前沿，但完成关系始终使用相同端点和实例域。条件
分支保留条件，空循环是恒等路径；循环携带访问不能被局部出口吞掉。摘要不能
把所有 reader 压成一个 child-done，也不能把“父调用串行”当作 CV 的全局事实。

当前 serial region/call 的具体实现：`RegionCompletion` 一次建立真实端点索引，
按 scope、pipe 和 milestone 限定 native 候选；`CompletionPaths` 按源端点和
条件域保存可恢复的可达前沿。`sequence_plan.select_bridges` 查询已有覆盖，只为
差集生成协议，并把确定的边加入前沿。新增边只唤醒已经到达它源端点的查询；
未选择的候选不能给其他候选提供证明。合并同端点的等价条件后再分配 event ID。
最终 verifier 从实际 IR 重新建分析，不继承规划阶段的前沿。

这借鉴了 TileLang `AnalyzeNodeList` 的层次组织，以及 `OptimizeSyncPoints`
先查覆盖、保留必要边后增量加入 `DepClosure` 的顺序。没有移植其有界距离全闭包、
特殊协议重试或跨层时钟投影；当前单 count/serial adapter 也不因此具备一般 CV
能力。候选的消费端/最后 reader 优先次序是一种选择政策，不保证同步全局最少。

例：双缓冲输入 `x[i%2]` 需要

```text
MTE2.write_done(i) → V.read_begin(i)          # ready
V.last_read_retired(i) → MTE2.write_begin(i+2) # release
```

输入 release 不等输出 MTE3；输出槽另由其最后 MTE3 reader 保护。L1、L0A/B、
L0C 分别按 MTE1、MAD、FIX 的实际最后读取释放，单槽共享也不能通过增槽解决。

在 continuous CV 中，`QK(query+1,0)` 可以在旧 query 的 PV/Update 之前派发。
serial 调用的词法顺序优化只有在域确实串行时可用，不能推广到这条路径。

## 符号关系的可实现范围

首轮限定结构化循环、uniform 非负长度、静态 stage/slot、规则直接 views 和
可验证的 owner 分区。两端关系组合必须匹配中间实例、guard 和完成强度；
只看最小距离或忽略条件的图可达不能替代指定实例关系的覆盖。

`task=query*K+kv` 在 K 为 runtime 时包含参数乘积；`task//K` 也不是普通的
常量除数仿射关系。不能声称交给通用仿射分析就解决。需要为这类规则时钟映射
保留因子化关系及 `K>0, 0<=kv<K` 合同，利用商余唯一性做受限组合；不展开 K
个余数，也不支持任意非线性整数求解。独立测试乘法溢出、零 K 和 query 边界。

多参数 GM 地址同样保留 shape/stride 结构，先证明直接矩形 view 的范围和分区。
普通外部指针 disjoint 合同在 launch 检查实际区间；间接索引、数据相关 ownership
需要更强可验证合同，否则诊断为不支持或分析未决。

统一模型不要求硬件协议相同：local HardEvent 是每核每有向 bank 八 ID 的二值
事件；cross-core 的参与者、计数、quorum、反向 ack 与 GM 可见性另有 dav-2201
契约。消费通知不等于消费方最后读取完成；两个 AIV 的 UB 始终私有。

## 编译复杂度

令 S 为静态模板数，A 为访问片段数，H 为真实冲突关系数，R 为区域数，D 为
关系表达式规模。运行时长度 N 不应成为对象数、证明窗口或周期枚举的因子。

```text
成本 = 前端/物理化(S) + 访问相交(A,H,D)
     + 关系组合/覆盖(区域、边、条件与查询数量)
     + 协议/资源分配 + 最终重检 + 发射
```

密集别名可能产生 O(A²) 个冲突；条件投影、关系组合和 event 分配也可能昂贵。
不宣称整个算法线性。独立周期按实际耦合分量分析，不取所有 ring 的全局 LCM。
缓存按不可变关系与分析 revision 建立，失效规则明确；不能靠热缓存掩盖冷路径。

当前增量查询不再因每次加入 bridge 重建端点池和整条搜索。新增边的传播只访问
已经到达源端点的前沿；不同目标可以复用同源、同条件域的搜索。它仍可能为多个
源维护不同前沿，同一 scope 内仍有端点两两 native 证明，条件差集也会产生片段。
因此不能宣称整体线性，或仅凭重构断言冷编译变快。报告中的 completion sites、
frontiers、expansions、native relation checks 是最终验证阶段计数，耗时测量另行
包含规划与最终验证两者。

验收同时测模板/关系 pieces/节点/边/查询/重建/访问数、RSS 和五个冷进程时间。
保留循环的常量界 1024 与 1048576 不应导致结构计数按次数增长；整数位数成本
单独承认。常用例源码→ASC 目标 p50≤1s/max≤2s，完整 FA p50≤5s/max≤10s。
import、CANN、launch 单列。内部预算给具体阶段诊断，外部 60s 截止；不盲目提高
预算重试。任意符号程序的终止性或最优同步不在首轮承诺内。

## 正确性与性能验收

依次检查 IR/输入合同、访问与生命周期、有限交错反例、CANN 离线编译和设备。
最终 verifier 可共用数学工具，但不能宣称与 planner 完全独立；有限模型检查
不等于任意规模形式化证明。失败分为已证非法、不支持、预算未决。

固定政策下先逐项对比 ASC 的 native 指令、累计顺序、搬运、buffer、wait/release、
CV gate 和填排，再对照独立 AscendC 及 timeline。与同政策参考接近，只能支持
编译器保真结论；还需对照 ops-transformer/CATLASS、安装 ACLNN 和硬件工作量。
不得通过新增 packing、全局 barrier、每 query drain 或扩大缓冲取得表面正确。

硬件分析使用真实 SKU、核数、频率依据和 dtype，条件下界为
`max(各共享资源总需求/对应吞吐上界, 依赖与派发关键路径下界)`。
有效/native FLOPs、GM 请求/HBM 实际流量分开；未知持续吞吐保持参数化。
大规模配置、测量口径与 FA 变体次序见[重构计划](refactor-plan.md)。算子调参仍暂停。

## 源码依据

本轮只读核对 HEAD，未 pull：TileLang `6ba65d2`，ops-transformer `4bbfb3b`，
CATLASS `698379e`，ops-nn `64b41ad`。固定提交报告与当前 checkout 分开溯源。
依据为 [910B 契约](../../reference/tilelang/docs/compiler_internals/ascend_910b.md)、
[InsertSync](../../reference/tilelang/docs/compiler_internals/ascend_910b_insert_sync.md)、
[算子审查](../../docs/xdsl-npu-operator-source-audit.md)及[FA 目标合同](attention-target.md)。
现有 FA 候选政策的 AIV update-before-softmax 与 ops-transformer arch22 可见的
Vec1-before-Vec2 不同，不能称为逐条复刻专家实现。
