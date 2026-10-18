# Flash Attention / GEMM：多 shape ACLNN 对照

2026-10-18，Ascend910B3，物理卡 2 → 逻辑卡 0，CANN 9.1.0，dav-2201，`-O2`。
以下均为 FP16 输入/输出，采用当前 AscendFlow DSL 生成代码，对比本机安装的 ACLNN。
每个 shape、每个后端各采集一次默认参数 msprof；所有时间来自本次设备 task，单位 µs。

## Flash Attention

`B=1, Hq=Hkv=8, D=128`，BNSD、dense、非 causal、无 dropout，默认 BQ256/BM128/BK1024/CK128/VR8，20 组（每组 1 AIC / 2 AIV）。
ACLNN 使用 `aclnnFlashAttentionScore`（keepProb=1、innerPrecise=0、sparseMode=0），两边 scale 均为 `FP32(1/sqrt(128))`。
shape 列为 `Sq×Sk`；表内四组使用同一份 AscendFlow 二进制，Sq/Sk 保持 runtime 参数。
causal 与非 causal 的默认性能验收统一只取 1K/2K/4K/8K 正方形。

| shape | AscendFlow µs | ACLNN µs | 加速比 | AscendFlow TFLOPS | ACLNN TFLOPS |
|---|---:|---:|---:|---:|---:|
| 1024×1024 | 60.620 | 59.560 | 0.983× | 70.85 | 72.11 |
| 2048×2048 | 164.240 | 165.640 | 1.009× | 104.60 | 103.72 |
| 4096×4096 | 505.472 | 525.668 | 1.040× | 135.95 | 130.73 |
| 8192×8192 | 1756.556 | 1887.140 | 1.074× | 156.49 | 145.66 |

## GEMM normal

`C[M,N]=A[M,K]@B[K,N]`，连续 NN，A/B/C 均为 FP16，FP32 累加，正态输入均值 0、标准差 0.5。
ACLNN 使用 `aclnnMatmul`、`cubeMathType=0`，实际 A/B/C tensor dtype 均查询为 FP16。
shape 列为 `M×N×K`；AscendFlow 使用默认 BM128/BN128/BK256/CK128、两个列片、20 个 AIC。

| shape | AscendFlow µs | ACLNN µs | 加速比 | AscendFlow TFLOPS | ACLNN TFLOPS |
|---|---:|---:|---:|---:|---:|
| 1024×1024×1024 | 12.700 | 17.440 | 1.373× | 169.09 | 123.14 |
| 2048×2048×2048 | 68.870 | 73.824 | 1.072× | 249.45 | 232.71 |
| 4096×4096×4096 | 484.990 | 500.580 | 1.032× | 283.39 | 274.56 |
| 8192×8192×8192 | 4024.070 | 4157.324 | 1.033× | 273.23 | 264.48 |
| 2048×8192×4096 | 486.372 | 498.710 | 1.025× | 282.58 | 275.59 |
| 8192×2048×4096 | 484.262 | 502.030 | 1.037× | 283.81 | 273.77 |

`normal` 指正态输入。保留 example 的 `OUTPUT_CACHE="auto"`：本次 8192³ 的 C 为 bypass，
其余五组为 normal，A/B 均为 normal。ACLNN 使用其自身分块和缓存策略。

表内 FA 加速比为 0.983–1.074×，其中 1K 为 0.983×；GEMM 为 1.025–1.373×。
大于 1 表示本次 AscendFlow 更快，小于 1 表示更慢；接近 1 的差异仅作本次观测。

## TFLOPS 计算口径

FMA 计两次运算，计算算法有效矩阵 FLOPs：

- GEMM：`FLOPs = 2*M*N*K`。
- FA：QK 和 PV 各 `2*B*Hq*Sq*Sk*D`，合计 `4*B*Hq*Sq*Sk*D`；当前 value 维度等于 D。
- `TFLOPS = FLOPs / (time_us * 10^6)`，两个实现使用同一分子。
- `加速比 = ACLNN time_us / AscendFlow time_us`。

例如 4096³ GEMM 为 137,438,953,472 FLOPs，以本次 484.990 µs 计算得到 283.39 TFLOPS。
FA 的分子不计 softmax、指数、归约、online 状态更新和数据搬运，分母包含整个 forward kernel 时间；
因此表中为算法有效 TFLOPS，不能当作 native 指令吞吐或硬件峰值利用率。

## 计时、数值与复现

每组先 AscendFlow 后 ACLNN，使用同一卡、相同 shape/dtype、实际输入 hash 和相同 oracle 行。
两个后端沿用各自 runner 的相同默认次数：FA 预热 1 次、统计 3 次；GEMM 预热 5 次、统计 30 次。
排除预热，取设备 task 时长中位数；一次 launch 有多个 task 时相加。进程内复用同一 allocation，
不 flush 缓存。ACL stream interval 包含 host 提交间隙，不作为表中时间。

表内 10 组 shape 均通过 sampled 数值 oracle、完整输出有限性、输入不变及 guard 检查。
原始归档保留当时全部 12 组 shape、24 次后端采集；其中两组不等长 FA 不再属于默认性能验收，本次未重新测量。
FA runner 对完整输出执行 `np.isfinite(actual).all()`，原始记录与汇总中的 `finite_elements` 为全输出计数；GEMM 记录 `finite_output=true`。
FA 的实际 `softmax_scale_f32` 和小端 FP32 字节 `softmax_scale_f32_bits` 同时保存在汇总中。
FA 容差 rtol=0.002、atol=0.0002；GEMM 对实际 FP16 输入使用独立 FP64 oracle，
容差 rtol=(2^-10)/2+1e-5、atol=0.001。采样行及误差保存在每项记录中，采样不等于全输出 oracle；
小规模全输出证据见[示例验证](examples.md)。一次采集的重复执行不属于独立试验，未执行历史三批门槛验收。

先完成[环境和全卡检查](../guide/testing.md)，显式映射健康空闲物理卡到逻辑 0，再运行：

```bash
python -m benchmarks.profile_defaults --list
python -m benchmarks.profile_defaults --output build/kernel-comparison --physical 2
```

本次共 7 次 DSL→IR→ASC→CANN 构建（FA 复用一次，GEMM 六次），每次有覆盖全部阶段的 300 秒硬期限。
编译器与算法源码为 `12c21d7`；本次仅变更测量工具与文档，实际脚本 hash 另行记录。
运行前后检查全部八张卡，所用卡健康空闲。未修改时钟；保留只读硬件快照，不将入口时钟当作持续运行频率。

`build/performance-aclnn-20261018/performance.json` 在本地保存实际 shape、FLOPs、未舍入时间/TFLOPS、输入/库/CSV hash、
CANN 选项、源码身份和数值范围。本机原始数据在 `build/performance-aclnn-20261018/`：
`results.json` 为总表，`artifacts/` 为 DSL/IR/ASC/库，`<shape>/<generated|aclnn>/` 为原始 CSV、
`device.json` 与 `summary.json`，`tables.md` 为自动生成的比较表。JSON 汇总、原始构建和 profile 均保留在本地 `build/`，不纳入 Git。
