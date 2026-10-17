# 连续矩阵 GEMM

`examples/gemm/dense.py::dense_gemm` 计算连续 FP16 `A[M,K] @ Bt[N,K].T`，
输出连续 FP32 `C[M,N]`。没有输入 packing、额外转置 kernel、split-K 或
GM partial accumulator。M/N/K 为 constexpr；当前 native 行步长仍要求静态。

默认 BM128、BN128、COL_PARTS2、L1 BK256、Cube CK128，20 个 Cube。
M/N/K 分别是 128/256/256 的正倍数；不支持尾块。每个 owner 循环处理输出
块，每块按 K 顺序累计。相邻两个列片共用 A，保留各自的 L0C、最后 MAD 和
FIX 退休点。L1 384 KiB，L0A/B 各 64 KiB，L0C 128 KiB，无 GM workspace。
仅最后一个 K panel 写回，FIXPIPE 位于 compute 末尾的 LAST 条件内，
与 compute 共用 stage；pipeline horizon 为 `PREFETCH+1`。

普通 ND 输出 tile 的核间 ownership 和跨迭代分离使用通用的 tile-grid 坐标
合同：实际二维 footprint 与 axes/offset 一致、原点按 tile 宽度对齐、所有
访问采用同一 grid，再证明 owner residue 和 streaming cell 家族互不重叠。
该几何事实在地址解码时建立，下游复用；不提供完成或逻辑值版本的保证。
L1 NZ 的全行、16 列对齐切片保留原轴范围，物理列偏移为
`column * full_rows * itemsize`；仍经独立 bounds、alignment 和生命周期检查。

## 输出缓存政策与运行时地址

算法的缓存政策位于 example：`OUTPUT_CACHE="auto"` 时，输出大小超过
`L2_BYTES` 才使用 `cache="bypass"`，否则为 `normal`。默认 L2 预算为
192 MiB，runner 对照实际设备容量；可显式选择 normal/bypass 做归因。
编译器不按 GEMM 名称、shape 或阈值选择缓存政策。

`Tensor(cache="bypass")` 是通用的 GM allocation 提示，沿 source/physical
IR 保留。它不创建新的 allocation，也不改变字节 effects、alias、ownership
或同步身份。launcher 接受正常分配的原始指针，查询当前逻辑 device 的
`rtGetL2CacheOffset`，按 CANN 9.1.0 / 2201 的地址别名合同绑定旁路指针。
查询失败或 offset 为零时返回错误，不退回普通缓存。普通缓存的 launcher
仍只有原来的 kernel 调用和 return。跨外部参数的 disjoint 合同针对真实
物理 allocation，不能将其不同缓存别名声明为互不相交。

依据为安装 SDK 的 `runtime/rt_external_kernel.h` 和 2201 的
`GlobalTensor::SetL2CacheHint` / `L2CacheAlter`。上游 ops-nn 的
`matmul_v3_l2_cache.cpp` 也按输出大小选择 C 缓存，不能据此宣称安装二进制
与参考源码或 tiling 完全相同。旁路别名仍需要当前设备的完整数值及 guard 验证。

## 验收与复现

数值使用独立 FP64 oracle，均值 0、标准差 0.5 的正态输入按
rtol=0.001、atol=0.001；有界
integer/8 输入要求精确相等。短域做全输出检查，大域 sampled 明确记录实际
行号，另检查完整输出有限性/hash、输入不变与 512 B 尾 guard。
性能比较同一卡、相同布局/dtype/输入、预热后复用同一分配，使用 msprof 设备 task
时间，要求 AscendFlow/ACLNN ≤ 1.00。ACL stream interval 另列，包含提交间隙
及缓存 runtime 查询开销；不把它当 kernel 时间。

```bash
../.venv/bin/python -m benchmarks.gemm.run_dense --output build/gemm/4096 --build
../.venv/bin/python -m benchmarks.gemm.run_dense --output build/gemm/8192 --build --m 8192 --n 8192 --k 8192
# 先检查所有卡；下列物理卡编号仅演示，按当前检查结果选择。
ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -m benchmarks.gemm.run_dense --output build/gemm/4096 --physical 2 --backend generated
```

`--backend aclnn` 调用安装的 aclnnMatmul（FP16 A、Bt.T view、FP32 C、
cubeMathType=0）。每个构建经唯一 DSL→IR→ASC→CANN 路径和单次 300 秒硬期限，
保存 source/before/after IR、ASC、库、选项、DSL 和编译器源码快照及 hash。
设备记录绑定这些产物、runtime cache alias、硬件与输入身份；复用产物不重新编译。

每个 shape 收集 `trial-0` 至 `trial-2`，各含 generated/aclnn 的
`summary.json` 与 `device/device.json`，可用严格门槛比较：

```bash
../.venv/bin/python -m benchmarks.gemm.compare_dense build/gemm/profiles/4096 --output build/gemm/comparison-4096.json
```

比较器重新校验 IR/ASC/库、安装 ACLNN、输入/采样范围、grid、物理卡和 CSV
的身份。三个生成批次中最慢的 p50 也必须不慢于最快 ACLNN 批次的 p50；
统计同时报告三批 p50 的中位数，不放宽 1.00 门槛。

具体已测形状、数值范围、三批性能及历史缓存调查集中在
[性能与设备验证记录](performance-results.md)。性能资格按对应产物记录限定；
小 tile 或 panel-major 输入的结果不替代普通大矩阵资格。
