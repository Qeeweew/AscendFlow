# GEMM / FA 性能与设备验证记录

2026-10-17，记录通用 b16 GEMM、DSL helper 和末块输出 stage 合并后的验证。
接口与算法合同分别见 [NN GEMM](matmul-target.md)、[NT GEMM](gemm-target.md)
和 [FA](attention-target.md)；复现工具见 [benchmarks](../benchmarks/README.md)。

## 测量合同

Ascend910B3，物理卡 2 映射逻辑卡 0，CANN 9.1.0，目标 dav-2201，CANN
编译选项 `-O2`。运行前后检查全部八张卡；最终均健康空闲。
每组在同一卡交替运行 generated / ACLNN 三批，每批预热 5 次、计时 30 次，
预热及计时复用该进程内的同一 allocation，不 flush。表中时间为三批 msprof 设备 task p50 的
中位数，排除 host 提交间隙；ACL stream interval 另行记录。

验收固定为最慢 generated 批次 p50 / 最快 ACLNN 批次 p50 ≤ 1.00。
比较器重新校验实际 DSL、IR、ASC、编译选项、库、输入、设备及原始 CSV
身份。ACLNN 使用容器安装库，其身份不能由上游参考源码代替。

## 当前性能

| 算子 | 尺寸 | 输出类型 | AscendFlow µs | ACLNN µs | 最严格批次比 |
|---|---|---|---:|---:|---:|
| 标准 NN GEMM | M=N=K=4096 | FP16 | 485.090 | 501.970 | 0.96747 |
| 标准 NN GEMM | M=N=K=4096 | BF16 | 484.678 | 501.832 | 0.96769 |
| 标准 NN GEMM | M=N=K=8192 | FP16 | 4023.630 | 4134.402 | 0.97574 |
| 标准 NN GEMM | M=N=K=8192 | BF16 | 3935.526 | 4020.120 | 0.97979 |
| dense NT GEMM | M=N=K=4096 | FP32 | 486.380 | 505.888 | 0.96200 |
| dense NT GEMM | M=N=K=8192 | FP32 | 4041.952 | 4242.124 | 0.95841 |
| FA，BQ256 | Sq=Sk=4096 | FP16 | 509.668 | 528.638 | 0.96754 |
| FA，BQ256 | Sq=Sk=8192 | FP16 | 1762.556 | 1896.550 | 0.93030 |

八组均通过门槛。标准 GEMM 的 A/B/C 同为 FP16 或 BF16；ACLNN adapter
通过 `aclGetDataType` 查询三个实际 tensor 描述符，FP16=1、BF16=27。
dense 的 A/Bt 为 FP16、C 为 FP32，adapter 构造时检查输出描述符为 FP32=0；
其 runner 记录没有单独保存 `tensor_dtypes` 字段。两类 GEMM 均使用
`cubeMathType=0`。FA 为 B1/Hq8/Hkv8/D128、非 causal、标准输入。

标准 GEMM 4K 使用 normal 输出缓存，8K 使用 bypass，并按 example 的
L2 子矩形遍历；dense 的 4K/8K 同样分别使用 normal/bypass，保持自己的
NT 政策。任意正 constexpr M/N/K 的正确性支持不等于所有尺寸都满足此性能
门槛；本表只覆盖列出的尺寸、dtype、政策和产物。未将几微秒的批次波动
单独归因于 stage 合并。

## 编译与正确性

全量回归 **1799 passed，311.23 s**，无 skip/error。共完成 35 次
DSL→IR→ASC→CANN 构建：主 GEMM 24 次，6.449–17.092 s；其它算子 11 次，
6.293–28.806 s。每个算子单次构建均使用覆盖全部阶段的 300 s 硬上限。
全量测试耗时与单算子构建期限分别记录。

| 全输出设备检查 | 数量 | 范围 |
|---|---:|---|
| 标准 NN GEMM | 72 | FP16/BF16、尾块、长步长、非 32 B 对齐地址；normal/exact/halfway |
| dense NT GEMM | 4 | FP32 输出；generated/ACLNN、normal/exact |
| runtime K reduction | 24 | FP16/BF16 × NT/NN；空、单块、多块 |
| hierarchical GEMM | 18 | FP16/BF16；空、单块、多块及嵌套输出 |
| 默认 BQ256 FA | 12 | 四组 Sq/Sk × standard/uniform/sharp |
| BQ128 FA | 2 | standard；独立于默认政策记录 |
| 合计 | 132 | 完整数值输出、输入不变及 guard 检查 |

标准 GEMM 的 72 项中，exact 32、normal 36、halfway 4；exact 和 halfway
要求与舍入后的独立 FP64 oracle 精确相等。普通输入的数值容差遵守对应
算子合同。另有 24 项有限 byte/event 模型回归，覆盖空流、单块、LAST
输出及跨输出/跨调用的累加器复用。

4K/8K 性能用例使用 sampled oracle。两类 GEMM 使用独立 FP64 oracle，
分别检查 193/385 行的全部列，即每 128 行块前六行加末行。FA 使用遵守其
数值边界的独立 online-softmax oracle，分别检查每 head 192/384 个 query 行，
具体行号保存在 `query_rows_per_head`。这些大尺寸没有执行全输出数值 oracle；
另检查完整输出有限性/hash、输入不变及 guard。

标准 GEMM 四组大尺寸各三批的完整输出 hash 均与同 dtype ACLNN 相同。
默认 FA 的四组 standard 全输出用例，Q/K/V 与完整 O 的 hash 均与 stage
合并前相同；FA 与 ACLNN 按数值容差验收，不要求 bitwise 相等。

## Pipe 指标与历史缓存调查

以下为本次标准 GEMM 的 `PipeUtilization` 活跃比例，取各批汇总的中位数；
每格为 AscendFlow / ACLNN。

| 尺寸 / 类型 | MAC | MTE1 | MTE2 |
|---|---:|---:|---:|
| 4096 / FP16 | 0.9830 / 0.9680 | 0.7550 / 0.8310 | 0.9610 / 0.8330 |
| 4096 / BF16 | 0.9840 / 0.9675 | 0.7560 / 0.8310 | 0.9615 / 0.8320 |
| 8192 / FP16 | 0.9650 / 0.9540 | 0.7410 / 0.8115 | 0.9810 / 0.8810 |
| 8192 / BF16 | 0.9600 / 0.9490 | 0.7365 / 0.8050 | 0.9860 / 0.9110 |

MAC 与 MTE2 均高度活跃；pipe 可重叠，比例不能相加分解延迟，也不能单独
据此认定某个 stall 原因。本次没有新增 L2/HBM PMU 测量，算法 FLOPs、
逻辑驻留工作集和 GM 请求均不当作实测 HBM 流量。

历史 NT GEMM 缓存调查属于提交 `c95dd2e` 的产物，保存在
`build/gemm-20261017/{result.md,evidence.json}`。当时 normal/bypass 对照保持
device kernel body 一致，只改变 launcher 的输出地址绑定；8K 输出旁路使
L2 `write_cache_miss_allocate` 中位计数从 524288 降至 0，r0 读 miss 分配比例
约从 6.80% 降至 3.24%。这些是当时的归因证据，不是本次新 ASC 的 PMU 结果。
末块 stage 合并已改变当前 ASC 和设备二进制，旧产物不能代替本次资格。

## 产物索引与待处理项

本机产物根目录为 `build/gemm-simplify-20261017/`，不纳入 Git。记录中的绝对
CSV 路径用于身份校验，复核时保留原目录。构建记录生成时的 HEAD 为
`c95dd2e`，源码为当时未提交工作树；其源码快照与 hash 定义被测内容，
不能只用父提交号识别。后续文档整理不修改原始测量记录。

| 相对产物根目录 | 内容 |
|---|---|
| `final/evidence.json` | 主 GEMM 编译器/算法/runner 身份、构建、数值、性能及限制 |
| `final/large-{4096,8192}-{f16,bf16}/` | source/before/after IR、ASC、CANN manifest、库及源码快照 |
| `final/correctness/` | 小尺寸、尾块、长步长和地址偏移的全输出记录 |
| `final/comparison-{size}-{dtype}.json`、`final/profiles-{size}-{dtype}/` | 主 GEMM 三批比较、设备记录及原始 CSV |
| `terminal-stages/evidence.json` | dense、reduction、hierarchical、FA 的构建和全输出证据 |
| `terminal-stages/comparison-{dense,fa}-{size}.json`、`terminal-stages/profiles-{dense,fa}-{size}/` | 其它性能用例的三批比较及原始数据 |
| `terminal-stages/sources.json`、`terminal-stages/validation-sources.json` | 编译器、算子及数值验证源码 hash |
| `final/tests.log`、`final/tests.xml`、`final/quality.json` | 全量测试、Ruff 和格式检查 |
| `final/code-comparison.json`、`final/source.patch` | stage 合并后的代码变化与测量时源码 diff |
| `final/final-npu-smi.txt`、`final/final-health.json` | 最终全卡健康检查 |

当前默认 FA ASC 的 SHA256 为
`29b38ccc6f2e1a2eb2c8e116d47cacfbb37cdc409b84e8ee3ecdd25dc5aa49ff`，
位于 `terminal-stages/fa-256/attention_target.asc`。仍有 28 个静态 PIPE_V
调用点，452–458 行的重叠屏障仍在；问题及后续验收要求见
[待处理的性能问题](deferred-optimizations.md)，尚未实现 fence 去重。
