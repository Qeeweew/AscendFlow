# 标准连续 GEMM：FP16 / BF16 输入输出

`examples/gemm/matmul.py::gemm` 计算 `C[M,N] = A[M,K] @ B[K,N]`。
A/B/C 都是普通连续行主序 ND，`DTYPE="f16"` 或 `"bf16"`，三者同类型。
M/N/K 为 constexpr，可取任意正尺寸并包含 M/N/K 尾块；仍须满足可分配内存、
合法 int64 索引及 SDK 步长范围。零尺寸不在合同内。没有外部 packing、B 转置、
GM workspace 或额外转换 kernel。

默认 BM128、BN128、两个列片、L1 BK256、Cube CK128、20 个 AIC。每个 Cube 完整持有
输出块并按 K 顺序用 FP32 L0C 累加，最后一次 FIXPIPE 写回时转换一次。
L1 使用 384 KiB，L0A/B 各 64 KiB、L0C 128 KiB。B 的两个列片在 GM→L1 时
合成一次 256 列搬运，在 L1 中切片；两个累加器分别保留最后 MAD 和 FIX 退休点。
算法分块和缓存政策位于 example；编译器不按 GEMM 名称选择路径。

主体用 `T.tile_ranges` 绑定每轴的 `(base,count,valid)`，删除手写 M/N/K
整块和尾块枚举；有效宽度仍为 constexpr。N 尾部的多个 BN 列片现在共用
同一份 A 和一次宽 B 搬运、同一条 K 流水，逐列片裁剪 FIXPIPE。
L2 坐标政策在 [schedule.py](../../examples/gemm/schedule.py) 的 `@T.inline`
helper 中；其 AST 在正常前端内联，原轴、字节访问、ownership 和同步检查照常执行。
K 流水只有 load/compute 两个 stage；最后一个 panel 在 compute 末尾执行
FIXPIPE，不为只执行一次的写回增加 export stage。MAD→FIX 与 accumulator
下一次清零前的最后读取关系仍由正常同步分析建立。

默认 `L2_TILE=0` 按半个 L2 的输入预算选择子矩形边长：
`edge=max(1,L2_BYTES//(4*K*(BM+BN*COL_PARTS)))`。M/N 方向各取不超过
该边长的完整 tile 数；不能整除时该轴采用一个 tile。先遍历 L2 子矩形，
再遍历其中的输出 tile，仍以 `t=group+tile*GROUPS` 均衡分核，不加跨核 barrier。
整个 N 轴已在一个子矩形内或 M 轴仅一个 tile 时，直接使用等价的行主序解码。
每个宽 tile 仍导出两个 BN 列片，M/N 尾块保留完整行带的 owner。

4K 的完整输入为 64 MiB，使用整个网格；8K 的子矩形为 2048×4096，
对应 A[2048,8192] 的 32 MiB 和 B[8192,4096] 的 64 MiB，共 96 MiB。
这些是逻辑驻留工作集，不是实测 cache 命中量。`--l2-tile` 可在 benchmark
中固定两轴 tile 数，便于用同一数学和编译路径做对照。

该策略参考 ops-nn `a7539b121effe3c7b32a1e5e38de5b620b2e4ac9` 的
[`CalcTile` / `DoL2CacheTiling`](https://gitcode.com/cann/ops-nn/blob/a7539b121effe3c7b32a1e5e38de5b620b2e4ac9/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp)、
[`UpdateBlockCnt` / `CalcGMOffset`](https://gitcode.com/cann/ops-nn/blob/a7539b121effe3c7b32a1e5e38de5b620b2e4ac9/matmul/mat_mul_v3/op_kernel/mat_mul_base_block.h)。
复用其两层分块和映射思路；没有复刻 `UpdateBasicIndex` 的完整错位策略。不依赖参考
工程或调用其 kernel。日常 kernel time 只采集当前生成 kernel；ACLNN 参考用于专项诊断。

```python
from examples.gemm.matmul import gemm
program = gemm.specialize(M=129, N=257, K=259, DTYPE="bf16")
```

## 精度与尾块

`T.copy(L0C, GM)` 根据目标 dtype 自动选择 FIXPIPE 的 `quant_pre`：

| L0C | GM 输出 | quant_pre |
|---|---|---|
| FP32 | FP32 | NoQuant |
| FP32 | FP16 | F322F16 |
| FP32 | BF16 | F322BF16 |

FP16/BF16 使用 RN ties-to-even，没有 scale 或 ReLU。GM 行步长按输出
元素大小换算，不能写死除以 4。较小的 GM 目标只导出累加器左上角有效矩形；
源读按 N0 fractal 和真实 source_M 保留范围、间隔及最后读取关系。
普通 DMA 不获得隐式转换许可。

`T.copy(GM_view, L1_plane, pad_value=0)` 明确定义完整驻留 NZ 平面：读取
有效 GM 矩形，补零其余元素。封闭 MTE2 packet 先做 L1 fill，再执行局部
`PIPE_MTE2` fence，最后 ND2NZ，防止 fill 覆盖随后加载的有效值。
该 fence 不替代外部 ready/release 协议。M/N/K 边界区域走同一 DSL 数学
主体和同一编译链，不使用运行失败后的替代实现。

## 对齐与物理访问

普通 GM/UB DMA 仍按 32 B 块建模。b16 ND2NZ 的 GM 输入及 b16 FIXPIPE 的
GM 输出有独立的按元素地址、宽度和行步长合同；L1/L0 fractal 和对齐要求
保留。分析按实际 native 汇总每个地址的最大对齐，不统一放宽所有 copy。
ND2NZ 的 source leading dimension 为 uint16，超出范围时使用逐行描述符
和真实字节地址，不截断行步长。

4K/8K 性能输入的 A/B 每行搬运均为 256 个 b16 元素，即 512 B；两者
行步长分别为 8192/16384 B，各 tile 的 GM 偏移也是 512 B 的倍数。
runner 的 allocator 在原 allocation 内显式向上对齐至 512 B，再加入 512 B
guard；计时使用 `offset_elements=0`，因此基址与每行起点均保持 512 B 对齐。
B 的 128 列切片只发生在 L1，GM→L1 仍是一次 256 列搬运。C 的每个
FIXPIPE 列片宽 256 B，不能据此推断 A/B 输入搬运未对齐。奇数 leading
dimension 和尾块可能不满足这些性能条件，但仍按元素对齐的正确性合同执行。

L1 NZ 支持按 16 对齐的二维矩形。物理起点为
`row * 16 * itemsize + column * full_rows * itemsize` 加领先 slot 偏移。
每个 N0 block 的有效宽度为 `selected_rows * 16 * itemsize`，相邻 block
间隔为 `full_rows * 16 * itemsize`，保留孔洞和原轴 bounds。
稳定 FMATRIX 宽度的 strided NN B 可使用 Load3D，相应 A 可共用初始化；
物理 IR 保留入口配置 writer、S→MTE1 握手、宽度检查和资源预留。

二维 ND rectangle 身份在地址解码时建立。streaming 分离使用真实坐标的
单调行/列间隔；不同区域使用完整 active domain 的矩形范围。带 group 的
serial 起点只在保持真实 ownership residue 后组合。不同循环名、不同
tensor 名或构造器本身都不提供分离或完成保证；碰撞、错位与未知范围仍拒绝。

ND grid 的高低位拆分保留原始 div/rem 原子；只有低位范围及完整 radix
都成立时，才建立对应 digit permutation 的单射坐标。ownership 和 streaming
分离使用跨访问一致的排列，不能混用各访问的不同编号。原轴 bounds、原始
unsigned 定义性、真实字节访问和最后 reader 仍独立检查；swizzle 参数本身
不是正确性凭据。

## 缓存政策

`OUTPUT_CACHE="auto"` 时，A/B/C 的逻辑总字节数超过 `L2_BYTES` 才旁路
只写一次的 C，为输入复用保留缓存；否则使用 normal。A/B 均为 normal。
默认预算 192 MiB，runner 对照实际设备容量；也可显式选择 normal/bypass。
4K 总大小 96 MiB，8K 为 384 MiB。旁路使用同一 allocation 的 runtime
地址别名，查询失败返回错误。

## 构建和验证

```bash
../.venv/bin/python -m benchmarks.gemm.run_matmul --build --output build/gemm/nn-bf16 --m 129 --n 257 --k 259 --dtype bf16
# 先检查全部卡；物理卡编号仅为演示，按实际状态选择。
ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -m benchmarks.gemm.run_matmul --output build/gemm/nn-bf16 --physical 2 --backend generated --scope full --offset-elements 1
```

每个构建走 DSL→IR→ASC→CANN 唯一链路和单次 300 s 硬期限，保存 DSL、
完整编译器源码快照、source/before/after IR、ASC、库、选项和 hash。
另保存 matmul 与 schedule 的算法源码快照及 hash，覆盖内联 helper 的真实来源。
`--backend aclnn` 使用相同连续布局及输入输出类型，并通过 `aclGetDataType`
查询 A/B/C 描述符：FP16 为 1，BF16 为 27，`cubeMathType=0`。

oracle 以实际存储后的输入做独立 FP64 `A @ B`。正态输入均值 0、标准差
0.5，容差为半个输出相对 epsilon 加 1e-5、atol=0.001；记录未舍入误差及
舍入差异。有界 integer/8 输入和正负 halfway 输入要求与舍入后的 FP64
oracle 精确相等，没有因尾块失败而放宽容差。
小形状检查全输出。大形状选择 sampled 时，记录每个 128 行块前六行和最后
一行，检查这些行全部 N 列；另检查完整输出有限性/hash、输入不变、前后
各 512 B guard。地址可额外偏移 0–15 个 b16 元素。

日常性能记录固定为默认 4096×4096×4096 FP16、正态输入，各采集一次默认
msprof；具体命令与预热/执行次数见[测试指南](../guide/testing.md)，当前 kernel time
见[性能报告](../reports/performance.md)。原有 ACLNN 三批比较器保留作专项诊断，
不是日常验收步骤。任意正尺寸的正确性支持与某一配置的性能记录分别限定范围。
