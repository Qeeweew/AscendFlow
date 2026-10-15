# R2：显式 Cube 操作数和分层缓冲生命周期

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

[ast_cube.py](../examples/gemm/batched.py) 接通 AST 的 L1/L0A/L0B/L0C 分配、显式搬运、
原生 matmul 和 Fixpipe。运行时参数是每组任务数；每个任务执行一个完整矩阵
乘法。这是嵌套 K 流水的前置闭环，不代表动态内层 GEMM 或 Flash Attention 已迁移。

## DSL 和布局

```python
a1 = T.alloc_l1((3, M, K), T.f16)
b1 = T.alloc_l1((3, N, K), T.f16)  # 用户提供的 Bᵀ
a0 = T.alloc_l0a((2, M, K), T.f16)
b0 = T.alloc_l0b((2, K, N), T.f16)
c0 = T.alloc_l0c((2, M, N), T.f32)
for t in T.Pipelined(tiles, stages=4):
    with T.Stage(0, name="load"):
        av = T.copy(A[t, :, :], a1[t % 3, :, :])
        bv = T.copy(Bt[t, :, :], b1[t % 3, :, :])
    with T.Stage(1, name="operands"):
        aa = T.copy(av, a0[t % 2, :, :])
        bb = T.copy(bv, b0[t % 2, :, :], transpose=True)
    with T.Stage(2, name="compute"):
        value = T.matmul(aa, bb, accum=c0[t % 2, :, :], initialize=True)
    with T.Stage(3, name="export"):
        T.copy(value, C[t, :, :])
```

可运行示例另按 group 划分 GM，并固定物理派发顺序为旧 export → 新 load →
operands → compute。矩阵维度 N 与 DSL 命名空间 T 分开，避免名字遮蔽。

- L0A 为 ZZ，L0B 为 ZN，L1/L0C 为对应 dtype 的 NZ；leading axes 选择完整
  矩阵平面。lowering 检查平面大小、stride、偏移与所有索引项，不把线性行列
  切片当成 fractal 子视图。不支持的平面内切片明确拒绝。
- `copy(..., transpose=True)` 表示逻辑 `[N,K]→[K,N]` 转置，目前仅支持
  完整 L1→L0B。它选择 NT 装载路径；NN 的 `[K,N]→[K,N]` copy 使用另一条
  物理打包路径。逻辑转置与 Load2D 指令的内部 transpose 位不是同一含义。
- `matmul` 只读显式 L0A/L0B，写显式 L0C，不创建 operand 缓冲或额外搬运。
  当前 M/N/K 为16对齐的合法静态 native tile，输入同为f16或bf16，累加/输出f32。
- `initialize=False` 保留已有 FP32 累加；未初始化的首次读取拒绝。两个短 MAD
  顺序累加时，自动同步保留必要 M barrier，不能假定短 MAD 的隐式完成顺序。
- AST 的 `gemm` 隐式 L0 recipe 已明确拒绝，序列化 AST IR 也不能绕过该检查。
  遗留 `trace_func` 仍保留旧 L1 recipe，等待 R4 迁移收敛。
- constexpr 条件表达式（例如 `N if TRANSPOSE_B else K`）只解析选中的分支；
  设备条件表达式拒绝，不能在 Python 求值。一般结构化控制和运行时首次 clear
  仍需后续层次生命周期分析，未通过展开来提供表面支持。

## 同步前后：三个独立释放点

完整源 IR、物理前后 IR 和 ASC 由 runner 写入外层目录。下列是其实际任务简写：

```text
before:
  load_A/load_B(g), stage0: GM → L1[g%3]           MTE2
  operand_A/B(g),   stage1: L1[g%3] → L0[g%2]     MTE1
  mad(g),          stage2: L0A/B[g%2] → L0C[g%2]  M
  export(g),       stage3: L0C[g%2] → GM          FIX

ready:
  load_A/B(g) → operand_A/B(g)      MTE2→MTE1
  operand_B(g) → mad(g)            MTE1→M，覆盖先前 A 装载
  mad(g) → export(g)               M→FIX

release:
  operand_A/B(g) → load_A/B(g+3)    MTE1→MTE2，释放 L1
  mad(g) → operand_A(g+2)          M→MTE1，释放 L0A/B
  export(g) → mad(g+2)             FIX→M，释放 L0C
```

Set/Wait 只为真实消费者配对，release 在尾部按未来任务是否存在生成；零次
不产生虚构通知，最终只有 actor 返回边界的 drain。L1 release 不等待 MAD
或 Fixpipe，L0 release 不等待整个输出，三个生命周期没有合并成父任务完成。

测试删除 L1 release 后最终checker拒绝 WAR；把 B 的逻辑 producer 换为同形状
A producer也拒绝，不能把显式 L0 的 B 错当成 A 的 native operand。独立字节
oracle 和有限硬件 DAG 覆盖两种 B 方向、零次、一次及各槽复用边界。有限样本
不是任意规模证明，parametric verifier 仍使用已有受审查的 native 语义。

## AscendC 对照中的真实事件资源

[ascendc_cube.asc.in](../benchmarks/references/ascendc/ascendc_cube.asc.in) 使用公开 AscendC
DataCopy/LoadData/Mmad/Fixpipe 和手写事件，未从 PlanIR 自动转换。
固定循环排步、容量和数值契约相同。CANN9.1没有公开 `HardEvent::S_FIX` 枚举，
一次性 Fixpipe 配置交接使用同目标的原生 `set_flag/wait_flag`。

验证曾发现手写对照的 L0B 读写冲突（ACL507015）。根因不是布局或放大输入：
安装的 `asc/impl/basic_api/kernel_tpipe_impl.h:97–111` 显示 TPipe 初始化会占用
并预置 M→MTE1 的ID0/1/2，Destroy在约533–538行回收。手写 L0 credit 重用了
0/1，额外初始 token 使覆盖提前。修正为ID3/4，不改 tile/缓冲/搬运或等待位置。
编译器生成的 ASC 不使用 TPipe，没有这些初始 token。

失败版本源码、共享库和日志保留在外层 `failure-nn-bf16-acc/`。仅调整 Set/Wait
相对顺序不能修复该问题；对应 C API 诊断通过后，源码审查才定位到 TPipe 的
隐藏资源。修复后同卡重新验证大小规模。全程未 reset、未换卡规避错误。
因此旧错误版本即使有通过样本也不用于最终性能对照。

## 固定大例的性能与硬件解释

910B3，20 AIC，全卡检查后使用物理1→逻辑0。默认为20×1024个
32×64乘64×64矩阵，f16输入、f32输出，共41943040个输出元素，GM工作集400 MiB。
非对称输入由可精确表示的二进制分数构成，CPU单独计算17种矩阵再按任务映射
比较全部输出；结果要求逐元素精确相等，另检查输入和输出 guard。

修正后的独立 AscendC 与生成代码各3轮独立 msprof，每轮5 warmup+30 samples。
以下为设备 task 时间（µs），不是 ACL stream interval：

| 轮次 | 生成版本 p50 / p90 | AscendC p50 / p90 |
|---|---:|---:|
| 0 | 456.608 / 457.948 | 454.838 / 456.908 |
| 1 | 454.970 / 457.088 | 456.788 / 458.232 |
| 2 | 456.450 / 465.988 | 455.440 / 457.968 |

各轮p50的中位数456.450 / 455.440 µs，比值1.0022，在本配置下没有超过计划的
5%退化门槛；不能把这一结果外推为完整GEMM或FA无性能损失。参考有 TPipe
初始化/收尾，生成端没有；循环内实际搬运、3/2/2槽、MAD数和事件范围保持一致。

实际工作量为5,368,709,120 FLOPs，GM请求419,430,400字节，强度12.8 FLOP/请求
字节；L1为36864字节/AIC，L0A/B/C分别8192/16384/16384字节。设备时间对应约
11.762 TFLOP/s及918.9 GB/s的GM请求口径，这不是实测HBM带宽。
按20核、1800MHz与每核每周期8192个b16操作的**条件假设**，Cube峰值294.912
TFLOP/s、纯计算理想下界18.204 µs；指令启动、DMA、Scalar和依赖开销未包含。
频率记录有1800/800MHz切换采样，不能以单次快照认定全程恒频。

两方 MTE2 activity 中位数约0.998、MAC约0.120、MTE1约0.157、Fixpipe约0.416–0.419。
这些重叠计数提示检查搬运路径，不能单凭比例断言全部时间卡在HBM；工作集虽大于
192 MiB L2，实际HBM事务与持续带宽仍未测得。本轮固定策略不做无界tile调参。

## 复现与剩余边界

189项前端、Cube、符号循环、归约、prefix、编译及分层GEMM相关回归通过
（58.16秒）；最后新增负例后的相关52项通过（21.99秒）。最终5次冷进程包含
源码捕获与AST、lowering、规划、最终验证、ASC和产物快照，p50为0.2286秒，
max为0.2400秒，peak RSS最高36832 KiB；import/CANN单列。
计时在 `cold-final/`，较早与离线构建重叠的 `cold/` 不作为最终数据。

```bash
.venv/bin/pytest -q compiler/tests/test_ast_cube.py compiler/tests/test_source_language.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.gemm.run --output runs/refactor/ast-cube/nt-f16 --build
npu-smi info
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.gemm.run --output runs/refactor/ast-cube/nt-f16 --physical 1
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/gemm/batched.py --entry batched_matmul --trials 5 --output runs/refactor/ast-cube/cold-final
```

NN/f16另设 `--nn --m 32 --n 48 --k 64`；NT/bf16另设
`--dtype bf16 --m 16 --n 32 --k 48`；NN/bf16短MAD累加增加 `--nn --accumulate`。
每个配置使用不同输出目录，设备运行参数须与其构建完全一致。

3轮各2个版本的 profiling 使用独立 `trial-*/generated|ascendc` 目录：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/refactor/ast-cube/profiles/trial-0/generated --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.gemm.run --output /workspace/xdsl-910b/runs/refactor/ast-cube/nt-f16 --record /workspace/xdsl-910b/runs/refactor/ast-cube/profiles/trial-0/generated/device.json --physical 1 --variant generated --tiles 1024 --warmup 5 --repeats 30' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/refactor/ast-cube/profiles/trial-0/generated --warmup 5 --repeats 30
PYTHONPATH=compiler .venv/bin/python -m benchmarks.gemm.analyze runs/refactor/ast-cube/profiles --output runs/refactor/ast-cube/profiles/comparison.json
```

证据全部在外层 `runs/refactor/ast-cube/`。分析器绑定源码/库/CSV hash、编译
选项、shape、dtype、任务数、grid、数值输出、cache policy与物理设备。
小规模/原始设备计时不替代上述固定大例的profiling。

R2剩余：保留嵌套物理region与独立 query/KV/native 时钟；runtime首次clear、
循环携带状态、真正fractal子矩阵映射与层次摘要；跨通道event资源规划。
这里没有添加空壳嵌套接口，也没有将当前独立矩阵任务当作动态内层归约已完成。
