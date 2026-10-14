# 分层与连续 NT GEMM

`gemm_nt(M,N,K)` 编译 `C=A@Bt.T`，输入 f16 ND、输出 f32 ND。每个 AIC 固定
负责一个 N 列 tile，并由显式 `grid_m` 决定连续处理多少 M 行 tile。所有搬运、
MAD、Fixpipe、缓冲和执行 stage 均进入同一个物理 IR/AutoSync/verifier；没有
额外 kernel、GM 部分和累加、任意 barrier 或数据搬运 fallback。

## 实际政策与时钟

参数为 `tile_m/tile_n`、`l1_k/l0_k`、`l1_slots/l0_slots/c_slots`、`grid_m`、
`cache_b`。M/N/K 必须整除选定 tile，L1 K 必须整除成完整 L0 K slice。固定
资源超限会拒绝，不改变提供的政策。`grid_m` 默认每个 M tile 一个 actor；
更小的合法行 grid 让每个 actor 连续处理多行 tile。B 驻留是显式开关。

设 `P=K/l0_k`、`q=l1_k/l0_k`、`Ng=N/tile_n`，flattened stream 时钟为 g：

```text
output_epoch = g / P
output_row   = (output_epoch*grid_m + core_id/Ng) * tile_m
output_col   = (core_id%Ng) * tile_n
K_slice      = g%P
L1_epoch     = g/q
L1_slot      = L1_epoch%l1_slots
L0_slot      = g%l0_slots
C_slot       = output_epoch%c_slots
```

每个 L1 tile 在 `g%q==0` 载入，服务 q 个 MTE1 reader。开启 B 驻留时，每核
只在 g0 用原生 ND2NZ 载入完整 Bt tile，随后读取其 K 子片段；A 的环继续复用。
每个输出开始时 MAD 的效果为 W、以后为 RW；`mad_cycle` 的 flag 和效果均由
已验证的静态 `g%P` 决定。每个输出只在 `g%P==P-1` 由 Fixpipe 导出一次。

View 保存常量整数 divisor/modulo 与核坐标项，不接受任意 C++ 地址字符串。
嵌套 view 保留分配身份及精确 byte unions。所有 actor 的父边界、对齐和 GM
ownership 都检查；对被写 GM root，还要求各 view 使用相同核平移，从而
单 actor 的局部完成证明可组合。若只有 core1 发生内部 GM alias，不能偷用
core0 的证明；当前明确拒绝这种需分别同步的 actor 程序。

## Ready、最后读取与退休

例子：`tile_m/tile_n=16/16`、`l1_k/l0_k=32/16`、L1双槽、K96。
完整 before/after IR、ASC 与 witness 由 CLI 写入外层输出目录：

```text
before:
  LoadL1(g)    g=0,2,4   → L1[(g/2)%2]
  Operand(g)   g=0…5     R corresponding half of L1; W L0[g%2]
  MAD(g)       g=0…5     R L0; W/RW C
  Export(g)    g=5       R C; W GM

after, relevant L1 path:
  LoadL1(0): ready ordinal0
  Operand(0): wait ready0; Operand(1) shares the real ready fence
  Operand(1): Set read-retired; this is the final read of old L1 slot0
  LoadL1(2): ready ordinal1, into slot1
  LoadL1(4): Wait read-retired from g1, before overwriting slot0
```

该 release 的 distance 是3，从 reader g1 到 overwrite g4，源为 MTE1，不能
等同于 MAD 完成。通知的 stride 是2，硬件 ID 按通知序号 `(g-lower)/2` 轮换，
不是 `g%2`（后者把所有稀疏 producer 压到同一 ID）。删除 release 或将 ready
宽度缩成1时，最终 verifier 与有界交错检查都有负例。

多输出的 C 生命周期同样明确：M16/N16/K32、三个输出、双 C 槽时，Export g1
的读取退休必须保护第三个输出的首次 MAD g4。删除 FIX→M 通道报告 WAR。
新 C 槽的首次 clear 无需读取旧 C，上一输出的 Fixpipe 可以与新 C 的计算交错。
最后 drain 只在 kernel 出口，不能修复这些内部缺失路径。

## 设备发现并修正的短 MAD 问题

最初统一把 M 流水当作完成顺序，纯 IR 与有界模型均通过；M32/N32/K64 的
连续三输出设备却出现 max error=7.09375。移除 B 驻留/连续输出的 M16/N32
单输出累加也失败，说明错误不是简单缓存布局或输出指针偏移。

CATLASS 当前 HEAD `698379e0d4b1b94d12728e7181df2cf83a958e8f`（未修改）在
`include/catlass/gemm/tile/tile_mmad.hpp:71–74` 对
`(m/16)*(n/16)<10` 加 `PIPE_M`。最终模型不再对这种短 M 无条件授予完成边，
从实际 C RAW/WAW 插入局部 M barrier。下例仅在累加 g1/g3/g5 等待 M，独立
C 槽的首次写 g2/g4 不扩大等待范围：

```text
g0 clear C0
g1 barrier M; accumulate C0; later FIX reads C0
g2 clear C1                    # no blanket M fence here
g3 barrier M; accumulate C1
g4 wait FIX release; clear C0  # protects actual reuse
g5 barrier M; accumulate C0
```

最终相同失败配置 max error=0，误差标准、搬运、slot 和算法均未放宽/改变。
`persistent-small` 保留失败输出；`persistent-small-m-completion` 是修正产物。
大 tile 保留既有 M 隐式顺序契约，M128/N128 和 M128/N256 不插入 `PIPE_M`。
这是硬件/native契约的修正，不能由此前 xDSL verifier 或有限枚举推导或替代。

## 已验证的范围

2026-10-13，xDSL0.73/CANN9.1/dav-2201，全部184项测试与ruff通过。
新回归包含不同 L1/L0时钟、1/2/3槽、周期 pairing、实际通知 ordinal、二维 GM
strip ownership、逐输出 clear、驻留 B 容量、删除 C/L1 release、删除短 M 完成
与非法核依赖 alias。有限交错检查 M16/N16/K32、三输出、驻留 B：单 C 槽
13080状态、双 C 槽20887状态，均穷尽且唯一干净终态；不是任意规模证明。

以下最终例均通过 device object/ELF/shared编译及独立 CPU oracle。每个进程在
ACL前检查全部卡，选物理1、映射逻辑0；没有reset/错误后换卡。输入为非对称
二进制分数，f32输出要求零误差；不宣称一般浮点数据位级等价。

| 配置 | 核数/连续输出 | 最大绝对误差 |
|---|---|---:|
| M64/N128/K192，tile32/64，L1K64/L0K32 | 4核，每核一个输出 | 0 |
| M128/N1280/K768，tile128/256，L1K256/L0K64 | 5核，每核一个输出 | 0 |
| M96/N64/K64，tile32/32，双C、驻留B | 2核，每核三个输出；含短 M修正 | 0 |
| M512/N1024/K768，tile128/128，L1K256/L0K64、双C、驻留B | 16核，每核两个输出 | 0 |
| M384/N256/K256，tile128/128，双C、驻留B | 2核，每核三个输出，真实复用C0 | 0 |

性能、编译命令和完整hash保存在 `runs/gemm/`；基线直接调用安装的
`aclnnMatmul`，相同 f16 A、Bt.T strided view、f32 C，cubeMathType=0。
原始基线实现来自安装库，不是对当前上游源码/device ELF等价的宣称。

固定性能比较使用 M512/N1024/K768，`-O2 -fPIC -std=c++20 -mllvm
-cce-aicore-dcpreload-args=false --npu-arch=dav-2201`，设备对象经
`ld.lld -m aicorelinux -Ttext 0 --no-mmap-output-file` 链接。生成源码 SHA256：
`dc88c1e3cb1990df2b4bfc7e567b0992eeaa38b09791fd5470712032d47abf8f`。
CPU输入、源码、设备ELF、共享库、版本与编译选项都保存于外层编译/运行清单。
两边参数bytes的hash一致，基线没有额外转置/Cast任务，trace中每次只有一个
`aclnnMatmul_MatMulV3Common_MatMulV3` AI_CORE任务。

各做一次msprof，5次warmup、随后30次连续调用。取原始`op_summary`内核时间：

| 指标 | 生成16核 | ACLNN20核 |
|---|---:|---:|
| median µs | 7.460 | 10.990 |
| min/max µs | 7.360 / 7.740 | 10.500 / 11.420 |
| 数值最大绝对误差 | 0 | 0 |

生成侧的Cube/MTE1/MTE2/Fixpipe平均活动比例为
0.5567/0.5255/0.6262/0.1495，合计超过1，支持此配置有分层流水重叠。
这不是任意shape、dtype、设备或专家排步的性能无损证明；双方核数与分块政策
不同，连续30次调用也不是30个独立实验。原始数据为
`profile-persistent-{generated,reference}`，`summary.json`保留CSV hash与全部样本。
ACL stream interval包含提交间隙，另存于`device.json/reference.json`，不用于上表。

当前不支持一般尾块、BF16/NN、split-K、动态 group或跨 AIC/AIV。它们仍须
扩展物理/native/ownership契约后进入同一最终验证，不能在 ASC层藏入实现。

```bash
.venv/bin/python compiler/examples/build_legacy.py gemm --m 512 --n 1024 --k 768 --tile-m 128 --tile-n 128 --l1-k 256 --l0-k 64 --grid-m 2 --c-slots 2 --cache-b --output runs/gemm/persistent-512-1024-768 --compile --shared
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/gemm/persistent-512-1024-768 --physical 1
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_reference.py runs/gemm/persistent-512-1024-768 --physical 1
```

采集脚本保持venv解释器，并在每次进程加载ACL前检查全部卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof \
  --output=/workspace/xdsl-910b/runs/gemm/profile-persistent-generated \
  --application='/workspace/xdsl-910b/compiler/examples/profile_device.sh /workspace/xdsl-910b/runs/gemm/persistent-512-1024-768 --physical 1 --repeats 30' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
.venv/bin/python compiler/examples/profile_summary.py runs/gemm/profile-persistent-generated
```

基线采集换用`profile_reference.sh`和新的外层输出目录，其余计时口径一致。
