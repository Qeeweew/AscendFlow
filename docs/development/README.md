# 编译器开发

先完成[环境与上手](../guide/quickstart.md)，再按下列顺序阅读。设计目标和规则论证
不代替 CANN 或设备实测；结果见[当前报告](../reports/examples.md)。

| 顺序 | 文档 | 要回答的问题 |
|---|---|---|
| 1 | [源码布局](source-layout.md) | 当前唯一编译链在哪里，各模块建立哪些事实？ |
| 2 | [循环与同步设计](design.md) | 值、物理存储、异步完成与循环边界如何区分？ |
| 3 | [生成规则](generation-rules.md) | Set/Wait 如何从 cell credit、边界和派发规则生成？ |
| 4 | [编译不变量](compilation-invariants.md) | 哪些事实被保持，哪些改写使其失效，谁负责检查？ |
| 5 | [验证方法](validation.md) | 输入检查、有限模型、CANN 与设备分别证明什么？ |
| 6 | [硬件与源码依据](source-notes.md) | dav-2201 的可信前提来自哪里？ |
| 7 | [同步优化边界](sync-optimizations.md) | fence 合并的适用条件和不可跨越的边界是什么？ |

日常命令和两项性能采集统一在[测试指南](../guide/testing.md)，测试目录职责见
[tests](../../tests/README.md)。不要恢复历史 solver、旧公开 task API 或第二条编译路径。
