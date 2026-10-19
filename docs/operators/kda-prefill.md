# KDA prefill

实现为 [prefill.py](../../examples/kda/prefill.py)、
[prefill_prepare.py](../../examples/kda/prefill_prepare.py)、
[prefill_gram.py](../../examples/kda/prefill_gram.py) 和
[prefill_solve.py](../../examples/kda/prefill_solve.py)。全部 kernel 使用普通 DSL、
xDSL 自动同步和 ASC/CANN；算法与流水政策均在 examples，编译器没有 KDA 特判。

## 数值与输入合同

本轮合同为 Q/K/V/Output FP16，Gate/Beta/Initial/Final FP32，K=V=128、
H=HV、BT=64，启用 Q/K L2 归一化，Gate 为非正的 log 衰减值。
state 布局为 `[sequence, head, V, K]`。
每条序列的 chunk 数相同且非零；最后一个 chunk 可以不足 64 tokens，长度可以不同。
`TOKENS: N.int64(minimum=1)` 是运行时值，Starts/Lengths 为不可变 GM i64 元数据。
调用方保证每个 chunk 在对应序列内且 `1 <= valid <= min(64, TOKENS-start)`，
各序列只有最后一个 chunk 可以不满 64 tokens。
当前 runner 拒绝零长度和不同 chunk 数的序列，尚未交付 grouped heads、speculative
state indexing 或 BF16 的设备资格。

原始 Triton 的 **矩阵乘两侧输入 dtype** 已逐项审计，见
[输入精度表](kda-prefill-precision.md)。跨块 Gram、块逆合并、W 与历史状态投影
使用 FP32×FP32 Cube，设置 IEEE 模式。W/U、状态增量与两个输出乘法使用
FP16×FP16，FP32 累加。Gate cumsum 使用 FP32 Vector 前缀累加；同一 16×16
对角块的 Gram 使用 FP32 元素乘、exp 和归约，与原始路径一样不调用 dot。
FP32 carry 与 FP16 snapshot 是不同的存储和值，snapshot 不反馈到递推状态。

## 流程和优化

| 阶段 | 计算及流水 |
|---|---|
| prepare_full / prepare | 非末块使用整块跨步 DMA；各序列末块保留三阶段 token 流水与 runtime 尾块；整块归一化，FP32 gate 顺序前缀累加 |
| gram | 四个 16×16 对角块驻留 UB；4/8/12/16 行因果前缀减少无效 exp/product，保持每个分数的 FP32 归约；四块求逆合为 16×64 Vector 计算，复用不变 packed mask，整块写回 Aqk |
| gram_prepare | 纯 Vector，按三组严格下三角行面板打包 FP32 左右输入 |
| gram_dot16 / gram_dot32 / gram_dot48 | 纯 Cube，M=32、K=128、N=16/32/48；两阶段双缓冲 L1 导入与 IEEE FP32 dot |
| gram_store | 纯 Vector，应用 beta、有效行与负号，写入块逆和输出乘法输入 |
| solve_merge16 / solve_merge32 | 同一 DSL kernel 的两次 constexpr 特化；两层 FP32 Cube 块合并，三阶段双槽流水 |
| transform | 整块 Vector 运算，显式保留 beta*K*expG、beta*V、KG、QG 的 FP16 舍入边界 |
| wu | FP16 Ai @ RHS，双槽导入/计算流水，生成 W/U |
| scan | 每个 chunk 内五阶段 CV 依赖；FP32 历史投影、FP16 residual/snapshot、Cube 更新与输出、FP32 carry；有状态的 chunk 顺序执行 |

块逆使用 `[[D0,0],[-D1*A10*D0,D1]]` 合并 16→32→64，乘法输入保持 FP32。
跨块 Gram 改为同一 stream 上每批依次执行 Vector→三种 Cube→Vector，
每批最多 512 个 chunk/head 任务，复用 54 MiB workspace。
加上该批主要输入和输出，估算活跃数据约 102 MiB；192 MiB L2 是分批预算，
实际命中率由 kernel 重放下的 L2Cache 计数报告，不以容量估算代替采集。
batch=1、S=4096、H=32 时共 4 批，13 种特化、28 次 kernel launch。
该阶段通过 stream 顺序交接，没有逐矩阵 mixed C/V 通知。
输入块、临时乘积与输出块分开存储，避免原地覆盖破坏流水的值和最后读取关系。
块合并临时乘积、32×32 C→V 交接使用每组两槽 workspace；旧 MTE2 reader 退休后
即可回收 GM 槽，不等待后续 Vector cast/store，从实际槽位生命周期约束通知复用。
每个 mixed kernel 的两个 AIV 共用 body，通过 runtime lane 区分地址与角色。
所有 ready/release、S 写入发布、GM 交接及事件资源约束均经同一同步生成链处理。

为此补齐的通用能力包括 FP32 Cube 及对应 NZ 字节几何、b16 转置 L1→L0A、
单元素 `store_scalar`、窄行广播、相关元数据边界、精确跨步矩形初始化覆盖，以及
同一 actor body 内完整双 AIV quorum 的前缀合并。没有关闭必要的同步检查。
本轮另补齐常数除数下多项式形式余数的上下界，支持分批 workspace 的地址证明；
原始 unsigned 表达式定义性与同步准入继续检查。
UB 跨步矩形的普通 `N.copy` 另接通 Vector bitwise copy，保留精确行 footprint、
不重叠和对齐准入及闭合 mask 合同。四块求逆合批的布局、执行与证据见
[Vector 合批报告](kda-prefill-vector-batch.md)。

## batch=1、S=4096 的当前结果

物理卡 2，1800 MHz；精度、输入、原始参考与官方 wheel 身份逐项核对。
`msprof op --aic-metrics=BasicInfo` 保持默认 **kernel 重放**，warm-up=0。
一次调用的全部 kernel Task Duration 之和：

| 实现 | 时间 μs | 原始 Triton / 当前实现 |
|---|---:|---:|
| AscendFlow 原版 `1845d4c` | 7578.520 | — |
| L2 分批版 `a625f83` | 5373.460 | — |
| 当前 AscendFlow | 4684.580 | 2.213× |
| 原始 Triton-Ascend | 10367.480 | 1.000× |

当前比 `a625f83` 减少 12.82% 时间，比 AscendFlow 原版减少 38.19%。输出/状态 relative RMSE
为 0.00174171 / 0.00116390，均满足独立 CPU FP32 递推的 `<0.005` 门槛。
同库将 runtime 长度改为 4077，末块长度为 45，再次通过，误差为
0.00174330 / 0.00120644。
13 个特化完整构建 125.903 s；当前 DSL 捕获与编译产物的 source IR 逐字节一致。
当前分阶段时间与单元利用率见 [Vector 合批报告](kda-prefill-vector-batch.md)，
`a625f83` 的 L2 hit/miss 和复现命令见 [L2 分批报告](kda-prefill-profile.md)。

## 历史设备结果（1845d4c）

以下为改写分批 Gram 之前的九阶段实现，不代表当前算法的小规模设备资格。
对应冻结产物可核验；当前 runner 使用新的 phase ABI，不能直接启动旧库。

设备为 Ascend910B3，物理 2→逻辑 0，1800 MHz，CANN 9.1.0，
[官方 Triton-Ascend 3.6.0](https://github.com/triton-lang/triton-ascend/releases/tag/v3.6.0)。
原始参考为未修改的 vllm-ascend `2581a69bcc16ee6af56e72985a15799bb5f61808`。
性能使用 `msprof op --aic-metrics=BasicInfo`，汇总一次调用的全部算子 kernel
Task Duration，包含各实现内的打包、清零等辅助计算，排除输入 reset、数值验证与
host 提交间隙。单次采集没有统计置信区间。
下表 H=HV=32、K=V=128，Q/K/V/Output 为 FP16，Gate/Beta/state 为 FP32。

| Batch | 每序列 tokens | AscendFlow (μs) | 原始 Triton (μs) | 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 378.760 | 486.800 | 1.285× |
| 4 | 512 | 3774.800 | 5161.140 | 1.367× |

batch=4 输出 relative RMSE=0.00168435、状态 0.00116318；Triton 分别为
0.000542813、0.000376406。同一套库在 `[449,477,505,512]` 长度（总 tokens=1943）
再次通过，输出/状态 relative RMSE 为 0.00166339/0.00128617，包含末块长度 1/29/57/64。
两个发布配置的完整九阶段构建分别耗时 86.640 s（batch=1）和 94.219 s（batch=4），
均包含 Python→IR→ASC→CANN，统一 300 秒硬上限。89 项针对性回归通过。

128 tokens、batch=1 的优化诊断：首版六阶段 643.140 μs，整块 Vector/Gram
改进后 500.460 μs，连续跨块流水与分块逆的诊断版为 371.900 μs。
发布版统一使用能支持长流水的每组两槽交接，最终时间以上表为准。
batch=1 输出 relative RMSE=0.00155499、最终状态
0.00108982；原始 Triton 分别为 0.000530733、0.000374423，均通过独立 CPU FP32
递推 oracle 的 `<0.005` 门槛。FP32 Cube 独立随机输入检查的 relative RMSE 为
1.609e-7，输入不预先舍入到 b16/HF32。

诊断证据为 Git 忽略的 `build/kda/`：

- `op-prefill-baseline/metrics.json`：首版 CSV、时间和数值记录；旧首版构建目录后来部分重编，不作为冻结库交付。
- `prefill-opt5/`、`op-prefill-opt5/`：优化版全部 DSL/IR/ASC/库身份、BasicInfo 和数值记录。
- `op-triton-prefill-b1-t128/`：原始 kernel 的 IR、npubin、编译参数、CSV 和数值记录。
- `comparison-prefill-opt5-b1-t128.json`：逐字节输入、平台、原始源码与官方 wheel 身份核对后的比较。
- `prefill-release-b1-t128/`、`prefill-release-b4-t512/`：冻结交付产物，包含完整构建 watchdog、IR、ASC、库与 SHA256。
- `op-prefill-release-b1-t128/`、`comparison-prefill-b1-t128.json`：小规模发布版 BasicInfo 与原始 Triton 比较。
- `op-prefill-release-b4-t512/`、`op-triton-prefill-b4-t512/`、`comparison-prefill-b4-t512.json`：大规模 BasicInfo 和全量数值比较。
- `prefill-check-b4-ragged/results.json`：同库改变 runtime tokens、各序列尾块的数值记录。

尾块记录与 batch=4 发布版的九个 ASC/共享库 SHA256 已逐项核对完全相同。

## 复现

使用工作区独立 venv，从产品仓库执行：

```bash
source ../.venv/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.kda.prefill --build-only --physical 2 \
  --artifacts build/kda/prefill --output build/kda/prefill-device --lengths 4096
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/prefill --output build/kda/prefill-ours
python -m benchmarks.kda.profile_prefill --physical 2 --backend triton \
  --artifacts build/kda/prefill --output build/kda/prefill-triton
python -m benchmarks.kda.compare_prefill \
  --ascendflow build/kda/prefill-ours --triton build/kda/prefill-triton \
  --official-release /workspace/xdsl-910b/materials/triton-official/release.json \
  --output build/kda/prefill-comparison.json
```

完整 pipeline 构建和每次运行分别有单次 300 秒进程组硬上限。运行前检查所有卡的
健康、进程、显存和利用率，每进程显式映射物理卡至逻辑 0；设备错误直接停止。
`--run-lengths` 可复用同一份库改变 tokens 数和尾块，只接受已编译的 sequence/chunk 域。
数值检查先于报告性能；stream interval 另有标记，不用它代替 kernel 时间。
batch=4 构建时将 `--lengths` 改为 `512 512 512 512`；该库可用
`--run-lengths 449 477 505 512` 运行尾块配置，输出目录使用新路径。
