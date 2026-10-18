# 示例编译与设备验证

2026-10-18：**16/16 个公开示例 CANN 编译通过，53 项小规模全输出设备检查通过**。
本页记录功能验证；kernel time 仅在[两项性能报告](performance.md)维护。

环境为 Ascend910B3、CANN 9.1.0、dav-2201、`-O2`，物理卡 2 显式映射逻辑 0。
运行前后检查全部八张卡，所用卡健康空闲。编译器 HEAD `64dff27`，本次未修改编译器可执行逻辑（CLI 仅格式调整）；
被测算法及运行器的实际源码 hash 保存在原始汇总中。

| 检查名称 | 本次全输出配置 | CANN / 设备 |
|---|---|---|
| scale | tiles=0、1、7 | 通过 / 通过 |
| grouped-scale | S=16、48、128，含空 owner | 通过 / 通过 |
| rings | tiles=0、1、7、35，输入/输出 5/7 槽 | 通过 / 通过 |
| rms | 每组 rows=0、1、7，D4096 | 通过 / 通过 |
| softmax | 每组 rows=0、8、24，D1024 | 通过 / 通过 |
| batched-gemm | tiles=0、1、7，32×64×64 | 通过 / 通过 |
| reduction-gemm | K panels=0、1、7 | 通过 / 通过 |
| hierarchical-gemm | outputs:panels=0:0、1:0、1:1、3:7 | 通过 / 通过 |
| dense-gemm | 256×512×256，FP16→FP32，normal | 通过 / 通过 |
| gemm | 129×257×259，FP16→FP16，normal，M/N/K 尾块 | 通过 / 通过 |
| nested-cv | n=0、1、2、7，每项连续执行两次，双 AIV | 通过 / 通过 |
| state | tiles=0、1、7，另含顺序舍入反例 | 通过 / 通过 |
| grouped-state | outputs:tiles=0:0、1:0、1:1、3:7，另含舍入反例 | 通过 / 通过 |
| shared-state | 同上，共享只读输入 | 通过 / 通过 |
| frame-state | 同上，原始 frame snapshot | 通过 / 通过 |
| flash-attention | Sq:Sk=256:1024、768:3072，默认 FP16 / BQ256 | 通过 / 通过 |

各 runner 验证全部输出、输入不变及 allocation guard；独立 oracle、容差和数据
规则保留在对应 runner/算子合同中。单次构建总耗时为 **6.579–30.442 s**，每次
Python→IR→ASC→CANN 共用 300 秒进程组硬期限。设备状态、数值和 CANN 产物
均来自本次真实 DSL 路径。

Python/IR 全量回归 **1816 passed，312.30 s**；文档链接、Ruff 与格式检查通过。
全量测试总耗时与每算子 300 秒构建硬期限分别计算。

源码入口见[示例目录](../../examples/README.md)。动态 `valid=` prefix 尚不支持，
已从 runnable examples 移入 `tests/fixtures/prefix.py`，原有拒绝测试保留。
BF16、大尺寸、其他政策和未列出边界不由这 53 项检查自动取得设备资格。

复现见[测试指南](../guide/testing.md)。本机原始记录位于
`build/docs-validation-20261018/examples/results.json`；各示例子目录保存命令、
日志、DSL/IR/ASC、CANN manifest/库和 `device*.json`。小型可追溯摘要为
[measurement.json](measurement.json)，原始构建和 profile 不纳入 Git。
