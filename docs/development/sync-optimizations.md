# 同步优化边界

当前 `simplify_conditions::_packet` 会合并相同完成前缀的连续 pipe fence。
相同 pipe、相同 eligible participants 的 fence 之间没有新 native 或其他通知时，
可将 guard 取并集；重叠路径上的相同 fence 按 `F;F=F` 合并。
SID 条件、原式定义性和全部 protocol 来源仍保留。

遇到 native、循环、地址/配置或未知 effects 就结束该区间。跨这些边界删除 fence
需要另外建立完成覆盖。Set/Wait 仅合并互斥分支；重叠的本地二值事件和跨核贡献
保留次数，不能套用 fence 的幂等规则。后端不修改 ASC 字符串来删除同步。

回归在[条件 pass 测试](../../tests/passes/test_condition_pass.py)及独立有限字节/event 模型中，
包括重叠条件、SID 域、未知 effects 边界和必要 fence 删除反例。
完整保持合同见[编译不变量](compilation-invariants.md)。

该规则已由 `64dff27` 实现。旧的调用点数量、逐批性能和优化过程通过 Git 查阅；
当前设备数值与两项 kernel time 只在[示例验证](../reports/examples.md)和
[性能报告](../reports/performance.md)维护。
