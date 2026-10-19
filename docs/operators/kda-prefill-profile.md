# KDA prefill：batch=1、S=4096 的 kernel 重放记录

2026-10-19。实现和合同见 [KDA prefill](kda-prefill.md)。本轮基线为
`1845d4c`，以下“当前”指随本文提交的 DSL、索引边界和 benchmark 修改。

## 测量条件

- Ascend910B3，物理卡 2 映射逻辑 0；20 AIC、40 AIV，L2 为 192 MiB。
- Batch=1，S=4096，H=HV=32，K=V=128，chunk=64。
- Q/K/V/output FP16，Gate/Beta/initial/final state FP32；启用 Q/K 归一化。
  原始 FP32 dot 的两侧输入保持 FP32，Cube 使用 IEEE 模式；不改为 HF32 或 b16。
- CANN 9.1.0，`dav-2201`；ASC 编译为 `-O2 -fPIC -std=c++20`
  和 `-mllvm -cce-aicore-dcpreload-args=false`。
- 原始 vllm-ascend：未修改的 `2581a69bcc16ee6af56e72985a15799bb5f61808`；
  使用 [官方 Triton-Ascend 3.6.0](https://github.com/triton-lang/triton-ascend/releases/tag/v3.6.0)。
- 三类采集全部使用 `msprof op` 默认 **kernel 重放**，`--warm-up=0` 与旧基线一致。
  BasicInfo、PipeUtilization、L2Cache 分开采集；不混用各次采集的 Task Duration。
  所有比较 kernel 的 CSV 实测频率均为 1800 MHz。
- 主性能口径为一次 prefill 调用全部 kernel 的 Task Duration 之和，包含打包和实现内
  清零，排除 host 间隙、输入 reset 和数值验证。每个 backend 仅一次 BasicInfo 采集，
  没有统计置信区间。此前 application 重放的诊断数据不参与下述结果。

## 算法与 workspace

跨块 Gram 从逐矩阵 mixed C/V 交接改为每批依次执行纯 Vector prepare、
三种纯 Cube dot、纯 Vector store。同一 stream 保证批内依赖及下批覆盖之前的最后读取。
Cube dot 使用两阶段双缓冲 L1 流水，M=32、K=128、N=16/32/48。

一个任务是一个 chunk/head。批大小由
`H * gcd(CHUNKS, 512 // H)` 确定，只包含完整 chunk；当前为 512 个任务、4 批。
各批复用以下 FP32 workspace：

| 分配 | Shape | 大小 |
|---|---|---:|
| GramLeft | `[512,3,32,128]` | 24 MiB |
| GramRight | `[512,96,128]` | 24 MiB |
| GramScore | `[512,32,96]` | 6 MiB |
| 合计 | | 54 MiB |

该批 Qp/Kp/Gp 约 32 MiB，两个主要输出约 16 MiB；加上 workspace，
主要活跃数据约 102 MiB。这是布局和分批预算，不能单凭它证明缓存命中。
每个任务跨块 dot 为 `2*32*128*(16+32+48)=786432` FLOP，
共 1.611 GFLOP；其 workspace 输入和输出合计 108 KiB/任务，四批共 216 MiB。

其他改动：非末块整块跨步 DMA，末块保留 runtime token 流水；块内 Gram 使用
4/8/12/16 行因果前缀，省去大部分无效上三角运算；16×16 对角逆留在 UB 内完成，
移除独立 solve 和 A workspace；scan 非末块整块输出。
gate 前缀保持原顺序；FP32 carry 与 FP16 snapshot 分离；chunk 状态仍按顺序递推，
没有跨 chunk 的 scan 流水。当前共 13 个特化、28 次 kernel launch。

编译器增加常数除数下展开余数的数学上下界，支持 slab offset 的地址证明。
原式求值定义性、物理访问、ownership、最后 reader、事件生命周期均走原同步链，
未添加 KDA 专用 pass 或关闭检查。

## BasicInfo 比较

| 实现 | kernel 时间 μs | Triton / 当前 |
|---|---:|---:|
| AscendFlow 基线 `1845d4c` | 7578.520 | — |
| 当前 AscendFlow | 5373.460 | 1.929× |
| 原始 Triton-Ascend | 10367.480 | 1.000× |

当前比 AscendFlow 基线减少 29.10% 时间。比较程序已核对输入逐字节 SHA256、
shape/dtype、精度门槛、物理卡、平台配置、实测频率、参考源码与官方 wheel 身份。

| BasicInfo 阶段 | 基线 μs | 当前 μs |
|---|---:|---:|
| prepare（含当前 full 与末块） | 1003.480 | 544.020 |
| 块内 Gram + 对角逆 | 3315.040 | 2296.500 |
| 跨块 Gram 全部阶段 | 1421.440 | 603.820 |
| 两级块逆合并 | 342.020 | 342.460 |
| transform | 212.500 | 217.940 |
| W/U | 81.900 | 82.120 |
| scan | 1202.140 | 1286.600 |

scan 在这次 BasicInfo 采集中更慢，不将它列为已测得的优化收益。

## Cube 与搬运

以下均来自单独的 PipeUtilization 采集，该次 kernel 时间合计 5199.480 μs。
表中比例为对应 core 活跃区间的样本均值，流水可重叠，不能相加为 100%。

| 阶段 | launch 数 | 合计 μs | Cube % | MTE2 % |
|---|---:|---:|---:|---:|
| Gram Vector prepare | 4 | 187.120 | — | 89.487（AIV） |
| Gram Cube dot | 12 | 233.680 | 13.466 | 87.160（AIC） |
| Gram Vector store | 4 | 137.480 | — | 38.873（AIV） |
| 块逆合并 | 2 | 324.060 | 6.384 | 66.684（AIC） |
| W/U | 1 | 73.740 | 27.618 | 97.341（AIC） |
| scan | 1 | 1221.580 | 8.955 | 24.291（AIC） |

基线跨块 Gram 的 Cube 为 2.596%，AIV Scalar 为 84.205%；
当前 Gram Cube dot 的 Scalar 为 5.663%。跨块 Gram 的主要等待已从逐矩阵交接
转为输入/输出搬运：当前 dot 的 Fixpipe 也达到 74.370%。
Cube 占比有所改善，距离高利用率仍有差距。

整条 pipeline 的设备容量归一化 Cube 忙碌比例为
`3562.382 core-us / (20 * 5199.480 us) = 3.426%`，基线为 2.466%。
它包含所有纯 Vector 阶段及核间不均衡，排除 host 间隙；不同于单核 Cube ratio。
当前块内 Gram/求逆仍耗时 2298.040 μs，Vector ratio 73.565%；
prepare_full 的 Scalar ratio 为 83.025%，顺序 scan 的 Cube ratio 只有 8.955%。
这些是后续主要优化对象，不能以跨块 dot 的局部提升宣称整条计算已达到硬件上限。

## L2Cache 原始计数

L2 指标已在 **kernel 重放**下采集成功。下表合并同阶段全部 launch/core 的
`r0`、`r1` 读取计数，以 `sum(hit)/(sum(hit)+sum(miss_allocate))` 计算，
不对各核百分比作未加权平均。

| 阶段与核 | Read hit | Read miss allocate | 读取命中率 % |
|---|---:|---:|---:|
| Gram prepare / AIV | 1007772 | 983044 | 50.6211 |
| Gram dot / AIC | 228 | 1572876 | 0.0145 |
| Gram store / AIV | 284120 | 50184 | 84.9885 |
| 块逆合并 / AIC | 295295 | 196285 | 60.0706 |
| 块逆合并 / AIV | 261184 | 1040 | 99.6034 |
| scan / AIC | 2687056 | 327680 | 89.1307 |
| scan / AIV | 2827034 | 614406 | 82.1468 |
| W/U / AIC | 19 | 655361 | 0.0029 |

scan、块逆合并和 Vector store 的复用均被记录。
Gram dot 在该采集下几乎全 miss，不能声称已测得它的跨阶段 L2 收益；
54 MiB workspace 的容量估算不能代替此结果。低命中率本身不表示指标采集失败。

## 数值、构建与证据

| 运行 | output relative RMSE | state relative RMSE |
|---|---:|---:|
| 当前 AscendFlow，S=4096 | 0.00174171 | 0.00116390 |
| 原始 Triton，S=4096 | 0.000551696 | 0.000373511 |
| 同一 AscendFlow 库，runtime S=4033 | 0.00173612 | 0.00109892 |

全部对独立 CPU FP32 recurrence 校验，门槛保持 `<0.005`。S=4033 使用相同
64-chunk 编译域、相同 ASC/共享库，覆盖 63 个完整 chunk 和长度为 1 的末块。
这不外推为 BF16、不同 chunk 数/序列或 grouped heads 的设备资格。

完整 Python→IR→ASC→CANN 构建为 126.016 s，共用一次 300 s 进程组硬上限。
最终源码捕获与 13 个冻结 `source.xdsl` 逐字节一致。相关索引、bounds、控制、
runtime 标量与 prefill primitive 回归 153 项通过；Ruff 与 diff 检查通过。

证据位于 Git 忽略的 `build/kda/`，每份 profile 保留原始 CSV SHA256 和设备数值记录，
设备记录包含每个实际 ASC/共享库的 SHA256 和 CANN 命令：

- `prefill-l2-fused-b1-s4096/`：完整构建、IR/ASC/库、pipeline manifest、当前源码一致性记录。
- `op-prefill-l2-fused-b1-s4096-{basic,pipe,l2cache}/`：三次独立 kernel 重放记录。
- `comparison-prefill-l2-fused-b1-s4096.json`：原始 Triton 的资格与性能比较。
- `prefill-l2-fused-b1-s4033-check/results.json`：同库 runtime 尾块检查。
- `op-prefill-baseline-b1-s4096-{basic,pipe}/`、
  `op-triton-prefill-b1-s4096-basic/`：原版基线和未修改 Triton 的原始记录。

冻结 pipeline manifest SHA256：
`7a35f7372520f44867e4bed80e22aeb8d90ac5e440e7720a13409c61e7ba52e8`。
比较记录 SHA256：
`980eeb61960a53be3a6d0cf069008df806c8d412f690bbdeae49e11fd2eb9a62`。

## 复现

从产品仓库使用独立 venv；每个输出使用新目录。worker 在运行前检查所有卡的健康、
进程、显存与利用率；设备运行和每次采集同样受 300 s 进程组期限约束。

```bash
source ../.venv/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.kda.prefill --build-only --physical 2 \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-device \
  --lengths 4096
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-basic
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-pipe \
  --metrics PipeUtilization
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-l2 \
  --metrics L2Cache
python -m benchmarks.kda.profile_prefill --physical 2 --backend triton \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-triton
python -m benchmarks.kda.compare_prefill \
  --ascendflow build/kda/reproduce-basic --triton build/kda/reproduce-triton \
  --official-release /workspace/xdsl-910b/materials/triton-official/release.json \
  --output build/kda/reproduce-comparison.json
python -m benchmarks.kda.prefill --physical 2 \
  --artifacts build/kda/reproduce-4096 --output build/kda/reproduce-tail \
  --run-lengths 4033 --warmup 0 --repeats 0
```
