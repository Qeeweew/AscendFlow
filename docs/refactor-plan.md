# 编译流程重构计划

本计划按最新指令执行：**先完成结构重构，再分析编译性能。** 不用跨编译结果缓存、算子特判、扩大超时或恢复旧求解器绕过合同缺口。一次编译内的不变事实应建立一次并复用。状态及运行证据只在[状态页](refactor-progress.md)登记。

## 1. 收敛一条编译链路

```text
DSL/source
  → retained native结构与SSA地址
  → 物理访问、ownership、值与输入合同
  → sequence / branch / loop依赖
  → 逻辑通知
  → 事件域 → 生命周期与进展 → bank/ID
  → 最终物理IR
  → ASC → CANN
```

已完成的结构调整：普通 `compile()` 只接受 DSL/source；旧物理输入分支和公共 `autosync/verify_plan/emit_asc` 已删除。后端读取最终 IR 中的真实条件、event SSA 和通知，不再单独解释动作表。被替代的planner、过渡实现及旧物理IR直接删除，不保留历史实现目录。

测试和基准消费者统一迁至当前产物。公共入口切换与代码删除本身不代表新路径已获得全部功能资格。

## 2. 当前重点：分开值依赖与物理复用

参数case、多余数窗口、条件路径覆盖、统一queue-order和函数实参同余已经接通。真实双runtime循环/双slot及嵌套CV通过了对应层验证。当前默认FA的拒绝已定位到更早的**依赖构造合同**：`structured_dependencies`按source顺序固定全部RAW/WAR/WAW，`structured_sync`随后才绑定dispatch。实际派发重排独立临时值时，source的物理反依赖会产生不必要的顺序约束；不能通过关闭progress或删一个event解决。

具体证据：共享L0C plane0在稳态的实际发射为`QK(j) → PV(j−1) → QK(j+1)`；当前依赖却要求`PV(j−1) FIX → QK(j) MAD`。这条WAR选成n15后，与同一QK的M→FIX RAW形成真实等待环。n15还合并了plane1的必要WAR，不能整条省略。有限实例及原始端点见[状态页](refactor-progress.md)。这证明当前选中同步不可进展，不证明FA数学算法或给定排步本身不可行。

按以下顺序重构，不再建设另一条同步路径：

1. **明确每个读取得的值。** 从现有source访问与准入事实保留精确read→reaching writer、范围、条件和代次；显式Tile与raw view都要有合同。当前ReadAdmission只保存若干writer操作，不能当作完整版本/生命周期证书。不能把读改绑到dispatch中最近的writer。
2. **区分完整覆盖与递推更新。** 完整kill建立新值生命周期；RMW保留其旧值输入、累加次序及条件。`mad_first`的首clear事实目前只用于admission，依赖阶段尚未消费；须用原语的条件读写合同表达首clear和后续RMW，不能按算子名字补特判或删除后续累加RAW。
3. **在已有排步上确定物理复用。** 将上述值实例映射到实际dispatch，对相同root/owner/byte cell推导相邻生命周期、全部最后reader和下一覆盖者。只有证明完整且独立的private scratch生命周期才允许改变source中的存储反依赖；actor-private本身不够。外部可见写入、递推状态和未证明别名保持其必要顺序或明确拒绝。
4. **循环仍按接口合成。** 由真实prologue/loop/epilogue的值和存储接口建立边界、回边及generation，不按runtime长度展开。不能把所有反向边统一改distance，也不能把子循环出口当异步退休。plane0与plane1分别建立实际复用链，随后才允许累计capture合并。
5. **共用后续同步合同。** 新的RAW值义务与physical WAR/WAW义务仍进入现有selection、事件域、路径覆盖、progress与分配器。替换被混用的source-order职责，不保留旧求解器作为fallback，不修改FA算法、缓冲、issue_order或最小输入规避问题。

先验收通用反例：独立完整覆盖的private值允许既定重排；读版本被覆盖、别名未知、外部写顺序和RMW递推不能因此放宽。`shared_l0c_qk_pv`目前只测了依赖/选择，必须补实际同步与有限队列验收，不能把已有局部通过称完整支持。保持数值/slot/参与者与通知资源检查后，再恢复默认FA完整资格。

事件层继续保持已建立的不变量：case必须覆盖实际membership；原始端点分别检查定义性与配对；consume→reuse必须有完整条件路径；parent只消费child接口；queue cut只表达阻塞，不提供完成；bank找不到方案不是硬件容量下界。具体合同集中在[编译不变量](compilation-invariants.md)，不在此复制第二份算法。

## 3. 收尾产品组织和功能资格

- 直接删除被替代的planner闭包及旧API专属测试。原语合同只接收几何，ASC只读取最终SSA；硬件反例迁至当前IR与独立模型，不保存旧实现。
- 基准消费者读取当前产物；结构模板数、算法工作量、实际运行期请求和设备流量分别标识，缺失计数明确报告。
- 测试分别承担输入合同、变换保持、实际IR破坏、独立有限执行语义和CANN构建；历史oracle不算新source资格。
- 使用同一公共链路验收双缓冲Vector、Cube/Fixpipe、嵌套/CV及默认FA。保留源算法、缓冲和排步合同；两个AIV共用一份body，用runtime sid区分。
- 每次完整算子构建最多300秒，超时清理整个进程组并停止该批候选。先定位并处理实际阻塞，再恢复对应资格构建，不直接重试或继续下一候选。
- 最终IR、ASC和实际CANN产物必须相互对应。当前debug_verify仅检查IR结构，不能充当独立同步证明。

当前主干代码组织、单一路径和旧职责退场已经落实；在该结构上处理重复整数证明和端点绑定后，嵌套CV已恢复完整构建。默认FA及设备资格仍需独立完成，不以小例或历史产物替代。

## 4. 结构改造落地后分析编译性能

固定当前源码、选项和输入；对成功产物记录IR/ASC，对未完成构建先记录实际停止阶段和有界诊断。测量各阶段工作量和耗时。分别考察native模板、物理分区、region边界、逻辑通知及资源关系数；runtime长度作为变量，不按执行次数展开。

已定位的重复工作收敛为同一 `IntegerFacts` 环境和完整端点绑定事实；表达式节点保存结构hash/排序信息，替换按共享节点遍历。状态量由静态模板、接口和不同前提决定，不由runtime次数决定。复杂度仍需计入各region的接口两两关系、图消去及不同有序条件集合；不能笼统称整个同步算法线性。结构排序的相等比较仍可能遍历子结构，不宣称所有表达式操作都是常数时间。

用新进程的实际完整构建衡量效果，不用跨编译命中、减少验证范围、换算子排步或静默fallback作为改善证据。编译性能通过后再恢复算子性能工作，比较AscendC/ACLNN、profiling及硬件理论指标。

职责图见[源码组织](source-layout.md)，每步保持与失效条件见[编译不变量](compilation-invariants.md)。
