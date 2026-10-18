# 同步性能优化状态

## 重叠条件下的重复 PIPE_V 屏障

2026-10-18：已在 IR 条件变换中实现同完成前缀的 fence 合并。

`simplify_conditions::_packet` 只接纳纯整数 SSA、条件和通知。相同 pipe、
相同 eligible participants 的连续 fence 之间没有新 native 或其他通知，
因此可将 guard 取并集；重叠路径上的多次相同 fence 按 `F;F=F` 合并。
SID 条件和原式定义性仍保留，合并指令汇集全部 protocol 来源。
遇到 native、循环、地址/配置或未知 effects 就结束区间。Set/Wait 继续仅合并
互斥分支，重叠的本地二值事件和跨核贡献不去重；后端不删除 ASC 字符串。

当前默认 FA 的 PIPE_V 静态调用点由 28 降至 20，ASC 由 557 行降至 531 行。
新旧 source IR 和同步前 IR 完全相同，native 仍为 52 个；本地 Set/Wait 和跨核
通知的静态次数及全部 309 个 protocol 来源保持。单次 DSL→IR→ASC→CANN
对象及共享库构建 28.001 s，
使用同一 300 s 进程组硬期限；这不是冷编译性能资格或设备加速证据。
本机产物在 `build/pipe-v-coalescing-20261018/`，ASC SHA256 为
`3f442bc853196df869270f5034b26a5cc8ead014388d5a539d9a627a04e5ab21`。

条件变换回归包含条件包含/重叠/并集、双 AIV 的不同 SID 执行域、native/循环/
未知 effects 边界，以及本地和跨核 Set/Wait 不可去重。独立有限模型检查实际
字节依赖，删除合并后仍必要的 fence 会出现 unordered WAW。
全量 1812 项测试、CANN 和六组全输出设备检查通过。同卡三批交替测量中，
4K/8K 设备 task p50 的中位数分别降低约 0.12%/0.16%；4K 批次范围重叠，
本次不宣称显著加速。
设备数值和性能的本次记录见[性能与设备验证记录](performance-results.md)。

当前规则保守地保留跨 native、跨循环和跨其他通知的 fence；这些情况需要
另外建立完成覆盖，不能仅凭指令名字或条件相似删除。

## 2026-10-17 的问题与基线

当时默认 FA 生成的 ASC 有 28 个 `asc_sync_pipe(PIPE_V)` 静态调用点。
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

同日将最终 O 写回合入 accumulate stage 后，默认 FA 的 ASC SHA256 为
`29b38ccc6f2e1a2eb2c8e116d47cacfbb37cdc409b84e8ee3ecdd25dc5aa49ff`，
仍有 28 个 PIPE_V 调用点，上述 452–458 行重叠屏障仍在。
该产物位于 `build/gemm-simplify-20261017/terminal-stages/fa-256/`；本次 stage
合并没有修改 fence 去重规则。

当时 `src/passes/simplify_conditions.py::_packet` 只合并互斥条件中的相同
通知，条件重叠时保留原指令次数；
`src/analysis/structured_resources.py::_combine_actions` 同样保持逐 actor
的完整指令序列及重复次数。这是 Set/Wait 贡献次数所需的合同，不能直接
扩展成所有通知去重。PIPE_V fence 的幂等性需要独立的变换规则。

以上是历史基线，不代表当前仍有 28 个调用点；旧产物和性能不能代替本次
合并后的产物及设备验收。
