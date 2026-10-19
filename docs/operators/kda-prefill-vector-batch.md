# KDA prefill：四块 Vector 合批

2026-10-19，基于 `a625f83`。输入为 batch=1、S=4096、H=HV=32、K=V=128，
FP16 Q/K/V/output、FP32 gate/beta/state。物理卡 2→逻辑 0，Ascend910B3，
CSV 实测 1800 MHz，CANN 9.1.0/dav-2201。数值与输入合同见 [KDA prefill](kda-prefill.md)。

## 小 Vector 操作的来源与改动

此前四个 16×16 对角块分别求逆。每轮先广播 16 个系数，乘法使用 16 次 repeat，
每次仅启用 16 个 FP32 lane；随后四级加法树处理 128、64、32、16 个元素，
最后行更新也只有 16 个元素。四个独立块分别执行 14 轮行递推。
`aiv_vec_ratio` 计量 Vector 忙碌周期，不是有效 lane 数或峰值 FLOPS 的占比。

当前将四个矩阵按列并排放入 UB 的 `[16,64]` 布局：

```text
inverse[j, block*16+c] = X_block[j,c]

for i = 2 .. 15:
    coefficients = copy(inverse[i,:])
    coefficient_matrix[j, block*16+c] = coefficients[block*16+j]
    product = inverse * coefficient_matrix          # 1024 个 FP32 元素
    tree = product[:8,:] + product[8:,:]              # 512
    tree[:4,:] += tree[4:8,:]                         # 256
    tree[:2,:] += tree[2:4,:]                         # 128
    tree[0,:] += tree[1,:]                            # 64
    inverse[i,:] = coefficients + tree[0,:]           # 64
```

系数用 BRCB、Vector broadcast 和跨步 UB copy 准备；乘法和每一级求和都使用完整的
64 个 lane。四个块共享 14 轮循环，各自的 FP32 乘法及 `j` 维加法树顺序保持不变。
依然求 `(I+A)^-1`，行依赖仍有 14 轮；当前没有跨 head/chunk 的求逆合批。

Gram 的结果先保留为 UB 的两个 16×16 矩阵，统一清除上三角后再保存：

- Aqk 从每个 chunk/head 的 64 次逐行 GM 写回改为 4 次整块写回。
- 对角 Gram 写入上述 `[16,64]` 布局，四块全部就绪后统一求逆。
- 两个不变三角 mask 在每个 AIV 启动时通过 `N.materialize` 生成一次，热循环复用
  packed UB 位图，通过 Vector select 完成掩码更新。避免在每个矩阵中用 Scalar 重建位图。

每个 AIV 的 UB 分配为 **113152 字节（110.5 KiB）**，低于 192 KiB。
外部 workspace、13 个特化与 28 次 launch 均保持原数量。

## 通用编译器支持

普通 `N.copy` 增加 UB 跨步矩形的 Vector bitwise copy lowering。
原生合同为每行 32–256 字节、至多 255 行、行距和起点按 32 字节对齐，repeat stride
最多 255 个 32 字节块；不支持转置。真实行和间隙进入 bounds、初始化、alias、最后读取
及同步分析，源/目标必须证明不重叠。packet 显式设置 normal mode 和精确列 mask。
没有算子名特判、手写 ASC、标量逐元素搬运或关闭同步检查。

## 性能

全部采集使用 `msprof op` 默认 **kernel 重放**，warm-up=0。BasicInfo 与
PipeUtilization 分开采集，表内只比较同一指标组。时间为全部 kernel Task Duration
之和，排除 host 间隙和数值验证；单次采集不提供统计置信区间。

| BasicInfo 实现 | Gram＋16×16 求逆 μs | prefill 总时间 μs |
|---|---:|---:|
| `a625f83` | 2296.500 | 5373.460 |
| 四块合批、尚未复用 mask | 1987.580 | 4992.220 |
| 当前：四块合批＋mask 复用 | **1748.060** | **4684.580** |

当前总时间比 `a625f83` 减少 **12.82%**，比更早的 `1845d4c` 减少 38.19%。
原始 Triton-Ascend 为 10367.480 μs，当前加速 **2.213×**；比较程序已重新核对
逐字节输入、平台、时钟、精度门槛、未修改参考源码和官方 wheel 身份。

以下为独立 PipeUtilization 采集的块内 Gram＋求逆：

| 指标 | `a625f83` | 四块合批 | 当前：另复用 mask |
|---|---:|---:|---:|
| 时间 μs | 2298.040 | 1980.020 | **1737.460** |
| AIV Vector 忙碌占比 | 73.565% | 76.214% | **87.050%** |
| AIV Scalar 占比 | 32.580% | 36.046% | **14.529%** |
| AIV MTE2 占比 | 8.332% | 9.738% | 11.177% |
| AIV MTE3 占比 | 20.142% | 4.251% | **4.758%** |

该阶段时间减少 **24.39%**。这些是对应 core 活跃区间的样本均值，各流水可以重叠，
不能将比例相加。Gram 和求逆没有分开计时，不能把合计时间当成纯求逆时间。

当前 Pipe 采集总 kernel 时间为 4605.800 μs。按整条 prefill 的设备容量归一化，
`sum(单元 busy core-us)/(物理核数 * 总 kernel 时间)` 为：

| 单元 | AIC，20 核 | AIV，40 核 |
|---|---:|---:|
| Cube / Vector | 3.867% | 42.425% |
| Scalar | 13.964% | 25.105% |
| MTE1 | 4.241% | — |
| MTE2 | 16.675% | 19.189% |
| MTE3 | <0.001% | 11.630% |
| Fixpipe | 10.598% | — |

其余 12 个特化的 ASC 与 `a625f83` 逐字节一致。跨块 Gram、Cube 块逆合并、W/U
和顺序 scan 的算法未在本轮调整，整条 pipeline 的 Cube 利用率仍低。
本轮未重新采集 L2Cache；此前 L2 指标冻结在 [L2 分批报告](kda-prefill-profile.md)，
不作为当前版本的新测量。

## 数值与构建证据

| 运行 | output relative RMSE | state relative RMSE |
|---|---:|---:|
| 当前 S=4096 | 0.00174171 | 0.00116390 |
| 同一库、runtime S=4077 | 0.00174330 | 0.00120644 |

都通过独立 CPU FP32 recurrence 的 `<0.005` 门槛。S=4096 的 RMSE 及最大绝对误差
与 `a625f83` 一致；不据此声称所有中间值位级相同。
4077 tokens 使用相同 64-chunk 编译域，末块长度 45，覆盖两个完整、一个部分和一个空的
16×16 子块；13 个 ASC、共享库和编译选项与正式 profile 完全一致。

完整 Python→IR→ASC→CANN 构建 **125.903 s**，单次 300 s 进程组硬上限。
最终 DSL 捕获与 13 个冻结 source IR 完全一致。184 项针对性 CPU/IR 回归通过，覆盖原生几何、
闭合 mask、字节版本模型、重叠拒绝、地址对齐、转置拒绝和 packed predicate。

证据位于 Git 忽略的 `build/kda/`：

- `prefill-vector4-mask-b1-s4096/`：完整构建、IR/ASC/库、pipeline manifest、当前源码核对。
- `op-prefill-vector4-mask-b1-s4096-{basic,pipe}/`：最终两次独立 kernel 重放采集。
- `comparison-prefill-vector4-mask-b1-s4096.json`：与已核验原始 Triton 记录的比较。
- `prefill-vector4-mask-b1-s4077-check/results.json`：同库 runtime 尾块资格。
- `prefill-vector4-b1-s4096/`、`op-prefill-vector4-b1-s4096-{basic,pipe}/`：mask 复用前的中间测量。

冻结 pipeline manifest SHA256：
`a7c26e93c3cefbf0ffbcf083f917edbd7f432ab081b8e6aad34d26988dafd153`。
比较记录 SHA256：
`f4644fcc1c65e30d43f79a3af3d4813fcf5c5539afe32e8518e979b5afd917e2`。

从产品仓库使用独立 venv，复现命令沿用 [L2 报告](kda-prefill-profile.md#复现)，
所有构建和采集输出目录使用新路径。尾块命令将 `--run-lengths` 改为 `4077`。
