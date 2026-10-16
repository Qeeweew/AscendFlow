# 编译流程重构计划

本计划按最新指令执行：**先完成结构重构，再分析编译性能。** 不用缓存、算子特判、扩大超时或恢复旧求解器绕过当前合同缺口。状态及运行证据只在[状态页](refactor-progress.md)登记。

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

已完成的结构调整：普通 `compile()` 只接受 DSL/source；旧物理输入分支和公共 `autosync/verify_plan/emit_asc` 已删除。后端读取最终 IR 中的真实条件、event SSA 和通知，不再单独解释动作表。M2 的7个过渡模块已从产品源码移入明确标识的历史测试目录。

尚未全部退出的旧 planner、旧计数工具不能成为新路径的 fallback。公共入口切换本身也不代表新路径已获得全部功能资格。

## 2. 当前重点：把事件合同补完整

### 激活域与循环窗口

`EventDomains` 一次建立实际端点、队列、参与者和 Set/Wait 激活配对。`EventWindow` 必须与实际 membership 等价，提供有条件的首末成员、同ID后继和存在投影。metadata 不能比实际发出的指令更窄；未知 carry、动态稀疏guard或未证明的别名不能静默忽略。

从内向外使用已证明的 child 存在投影构造 parent 窗口，保留共享坐标的关联。零次路径不计算没有定义的LAST；数学表达式正确也不能替代实际i64中间运算安全。

### 生命周期与资源

`event_lifetimes.py` 只处理实际消费到下一次复用的路径，使用上述窗口；不再分析内存依赖。区分有限调用内与下一次调用，不能把child出口视为全部退休。

分配器只维护 bank、ID、配置预留及最终 drain。有限通知共享ID必须留下真实 consume→next Set witness；不能把尚未实现的共享证明报成硬件容量下界。禁止为分配方便增加Scalar gate或全局barrier。

### 条件进展

`event_progress.py` 按实际region调用归属组合阻塞/发布接口。source与consumer来自不同parent代次时必须归属parent，不能误当child内部。

必须顺序用于reuse证明；可能阻塞不能遗漏，也不能丢掉其成立条件。child总结不导出虚假的异步完成。parent不能把每个内部通知的FIRST/LAST独立拼成零步图，也不再重复证明已由同一实际循环覆盖的义务。

先完成动态LAST死锁、短循环正例和嵌套接口反例，再集成实际例子。有限模型仅是独立回归，不能替代任意规模证明。

## 3. 收尾产品组织和功能资格

- 清退剩余旧 planner 闭包之前，先迁移共享native adapter、旧物理计数及benchmark解码消费者；保留硬件反例，不要求旧用户API兼容。
- 测试分别承担输入合同、变换保持、实际IR破坏、独立有限执行语义和CANN构建；历史oracle不算新source资格。
- 使用同一公共链路验收双缓冲Vector、Cube/Fixpipe、嵌套/CV及默认FA。保留源算法、缓冲和排步合同；两个AIV共用一份body，用runtime sid区分。
- 每次完整算子构建最多300秒，超时清理整个进程组并停止该批候选。默认FA最近已触及此上限，当前只继续代码重构和最小合同回归，不继续完整算子候选。
- 最终IR、ASC和实际CANN产物必须相互对应。当前debug_verify仅检查IR结构，不能充当独立同步证明。

只有主线语义、必要功能资格和旧职责退场都完成，才将这次重构标为完成。缺失合同、编译失败和未运行的验证要在状态页明示。

## 4. 完成后才分析编译性能

届时固定源码、生成IR/ASC、选项及输入，测量各阶段工作量和耗时。分别考察native模板、物理分区、region边界、逻辑通知及资源关系数；runtime长度作为变量，不按执行次数展开。

先确定真实复杂度来源，再决定是否需要改变算法或实现。不会用缓存命中、减少验证范围、换算子排步或静默fallback作为改善证据。编译性能通过后再恢复算子性能工作，比较AscendC/ACLNN、profiling及硬件理论指标。

职责图见[源码组织](source-layout.md)，每步保持与失效条件见[编译不变量](compilation-invariants.md)。
