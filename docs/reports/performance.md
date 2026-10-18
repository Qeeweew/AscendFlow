# 两项 kernel time

2026-10-18，Ascend910B3，物理卡 2 → 逻辑卡 0，CANN 9.1.0，dav-2201，`-O2`。
两项均使用当前 AscendFlow 生成的 kernel，各采集一次默认参数 msprof。

| 用例 | 配置 | kernel time（µs） |
|---|---|---:|
| Flash Attention | Sq=Sk=4096，B1/Hq8/Hkv8/D128，FP16，默认 BQ256/BK1024 | **506.512** |
| GEMM normal | M=N=K=4096，连续 NN，FP16 输入输出，正态数据，normal 缓存 | **484.540** |

时间取设备 task 时长中位数，排除 host 提交间隙。沿用运行器默认次数：FA 排除
1 次预热，统计 3 次执行；GEMM 排除 5 次预热，统计 30 次执行。同一次采集内的
重复执行不作为独立试验；本页不做 ACLNN 性能比较或历史三批门槛结论。

两项 sampled oracle、完整输出有限性、输入不变及 guard 均通过。小规模全输出
检查见[示例验证](examples.md)；大尺寸的采样不等于全输出数值验证。

## 复现与证据

环境与卡状态检查见[测试指南](../guide/testing.md)。选定健康空闲物理卡并映射后执行：

```bash
python -m benchmarks.profile_defaults --output build/kernel-times --physical 2
```

本机原始记录位于 `build/docs-validation-20261018/kernel-times/`，总表为
`results.json`；两项子目录各含 `artifact/`（DSL/IR/ASC、库、CANN manifest）与
`profile/`（原始 CSV、`device.json`、`summary.json`）。构建分别为 300 秒硬期限。
[measurement.json](measurement.json)保存配置、源码/库/CSV 身份、选项和记录路径。

编译器基于 `64dff27`；本次未改变编译器可执行逻辑，CLI 仅做格式调整，AST 一致性
记录在摘要中。历史多尺寸、多 dtype、PMU 和逐批实验通过 Git 查阅；本页维护这两项
本次采集结果，其余示例仅作编译和运行检查。
