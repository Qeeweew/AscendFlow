# 待处理的性能问题

## 重叠条件下的重复 PIPE_V 屏障

2026-10-17 记录；本轮保留现状，后续单独修改和验收。

当前默认 FA 生成的 ASC 有 28 个 `asc_sync_pipe(PIPE_V)` 静态调用点。
`build/fa-metrics-20261017/l0a-pair/attention_target.asc` 的 452–458 行为：

```cpp
asc_sync_pipe(PIPE_V);
if (npu_scalar_78 || npu_scalar_79) asc_sync_pipe(PIPE_V);
if (npu_step_73 >= 3) asc_sync_pipe(PIPE_V);
asc_brcb(...);
```

两个条件与无条件屏障重叠，中间没有新的 Vector 指令，重复等待同一个
Vector 完成前缀。相似片段在该产物的 375–378、384–387、391–394、
460–466 和 516–519 行。行号仅定位本次产物；ASC 的 SHA256 为
`f1a4071cc531845d3f687c40d976635636df3ae04be9f07b2b235ae5503172d9`。
CANN 9.1.0 的 2201 C API 将 `asc_sync_pipe_impl(pipe)` 定义为
`pipe_barrier(pipe)`。

当前 `src/passes/simplify_conditions.py::_packet` 只合并互斥条件中的相同
通知，条件重叠时保留原指令次数；
`src/analysis/structured_resources.py::_combine_actions` 同样保持逐 actor
的完整指令序列及重复次数。这是 Set/Wait 贡献次数所需的合同，不能直接
扩展成所有通知去重。PIPE_V fence 的幂等性需要独立的变换规则。

后续在 IR/同步变换层建立通用规则：按真实 pipe 完成前缀、执行域与中间
effects 判断可合并的 fence，保留 coverage/provenance，并覆盖分支、循环、
中间有新 Vector 工作及 Set/Wait 不可去重等反例。用独立有限模型验证变换
前后，再做 ASC/CANN 和同卡性能回归。不在后端按字符串删除屏障，不关闭
必要的同步或资源检查。当前没有实现该优化，也没有测得其设备收益。
