# 静态 GDN 与递推状态生命周期

`recurrent_gdn(sequences, steps, key_dim=64, value_dim=64, slots=2,
snapshot_slots=2, scale=None)` 已使用同一个 physical IR、AutoSync、最终 verifier
和 ASC 后端实现。每个独立 AIV block 拥有一个序列的私有 FP32 state；输入预取环、
attention 输出环与 snapshot 环都是同步前可见的物理分配。没有通用 extern、
内部 `PIPE_ALL` 或同步阶段新增的搬运。

## 算法与输入契约

输入 Q/K 为 f32 `[sequences,T,64]`，V 为 f32 `[sequences,T,Dv]`，初始 state
为 f32 `[sequences,Dv,64]`。`log_decay` 与 `beta` 各为 f32
`[sequences,max(8,align8(T))]`，padding 是调用者提供的可读内存；只消费前 T
个元素。两个输出分别为 f32 `y[sequences,T,Dv]` 和
`snapshots[sequences,T,Dv,64]`。所有外部基址需满足明确的 disjoint 契约。

对每个 token 按顺序计算：

```text
q_scaled = f32(q * scale)                    # 默认 scale=1/sqrt(64)=1/8
gamma = exp_f32(log_decay)                   # 该序列向量只计算一次
state = f32(state * gamma[t])
delta = f32(f32(v - reduce64(f32(state*k))) * beta[t])
state = vmla(delta_broadcast, k, state)      # 保留原生 multiply-add 指令
y[t] = reduce64(f32(state*q_scaled))
snapshots[t] = bitwise_copy(state)
```

state 始终保留在 UB，初始状态只搬入一次。没有从 snapshot 重载 state、修改
累加顺序、用 Mul+Add 拆开 `vmla`，或通过量化 snapshot 改变 carry。
`brcb8` 与 `broadcast_rows64` 是两个独立物理 Vector task；其间的 RAW 完成由
同步器生成，recipe 内没有隐藏 barrier。snapshot 的 bitwise `vcopy` 保留位模式，
与下游 MTE3 的读退休分开。

实际检查的 ops-transformer HEAD 是
`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，工作树干净；工作区报告中
`98f82c34…` 的固定链接是历史来源。本阶段没有 pull。对应 arch22
`recurrent_gated_delta_rule_arch22.h` 的 `Compute` 使用 resident f32 state、
MatVecMul/ReduceSum、delta/Broadcast/MulAddDst，并独立产生 state snapshot。
安装的 CANN 9.1 `reduce_sum_v220_impl.h` 对 64-key AR 使用 `WholeReduceSum`
(`vcadd`)，`MulAddDst` 与 ASC `asc_fma` 都使用 `vmla`；本实现保留这两个
指令的归约和更新边界。更大的 key 维度需要另行建模其 folding 与 scratch effects，
当前拒绝，不能直接换成另一个归约算法。

这是静态 f32 子集。上游 ACLNN GDN 的 Q/K/V/Beta/attention 接口要求 BF16，
还具有动态 state ID 与接受 token 的语义；本产品没有把该接口登记为已支持，
也没有通过插入 dtype 转换来制造同数据性能基线。

## carry 与 snapshot 的最小反例

三个代次、双 snapshot 槽的最小程序只含一次初始搬入、原地更新、snapshot
和输出。它在 `tests/test_recurrent.py::state_snapshot_example` 中构造；完整
IR 与 witness 保存于外层 `runs/gdn/state-snapshot-proof`。

```text
before:
  load_initial GM -> carry[0]                    # 仅 e0
  e: V update(carry, carry)                     # stage1
     V bitwise_copy(carry, snapshot[e%2])       # stage1
     MTE3 store(snapshot[e%2], result[e])        # stage2

after（列出物理条件）：
  MTE2 load_initial; Set MTE2->V; Wait V
  e: V update
     barrier V                                  # update 写完成 -> snapshot 读
     [e2: Wait MTE3->V release[0]]               # 仅保护 snapshot 槽复用
     V bitwise_copy
     Set V->MTE3 ready[e]
     [下一次 update: barrier V]                  # carry 最后读取退休
     Wait V->MTE3 ready[e]; MTE3 store
     [e0: Set MTE3->V release[0]]
```

各 local Set 异步生效，不 fence 后续源工作。carry 下一次写只需要旧 Vector
snapshot-copy 的读退休；snapshot 槽下一次写才需要旧 MTE3 store 的读退休。
最终图检查了前者的完成路径，同时确认 `store_snapshot[0]` 完成不是
`decay_state[1]` 开始的前提。两个 milestone 没有合并成“整个 token 完成”。
hardware 同 pipe Wait 的作用范围仍是该 Vector queue，不能宣称它只阻塞一个
buffer；生成位置尽量靠近真正消费/复用 task。kernel 返回的 `PIPE_ALL` 仅为
外层 drain。

删除所有 V completion barrier 后，最终 verifier 与独立有界执行器都拒绝该
最小程序；删除初始 writer 后完整 GDN 拒绝读取未定义的 carry。snapshot 槽和
输入槽的数量不会被同步器扩容或缩减。

## 验证与边界

有限交错检查的穷尽范围如下，不能外推为任意规模证明：

| 样本 | 状态数 | 是否穷尽 |
|---|---:|---|
| 三代次最小 carry/snapshot 流水 | 179 | 是 |
| 1 token、Dv8、单输入/snapshot 槽的完整 GDN | 758 | 是 |
| 2 token、Dv8、单输入/snapshot 槽的完整 GDN | 38404 | 是 |
| 5 token、Dv8、双输入/单 snapshot 槽的完整 GDN | 200000 | 否，预算用尽 |

最终静态图仍重算每个实际实例的完整 RAW/WAR/WAW、事件配对和复用，以及每个
block 的 GM 所有权；bounded check 未穷尽不会被标成通过证明。
CPU oracle 独立计算所有 token 的 attention 与 snapshot，使用 libm `fmaf`
实现状态更新。它以双精度求和累加已舍入的 f32 乘积，不假装模拟硬件 vcadd tree；
设备比较固定使用 rtol=5e-5、atol=5e-6。

当前支持 T=0…128、Dv 为8…128的8倍数、Dk64、静态独立序列和有限预算内的计划。
一般 tails、数据相关分支、ragged 序列、动态/间接 state ID、gk、低精度输入或
snapshot、跨核状态 ownership 尚未实现。缺少 disjoint 外部契约、UB 超容量、
event bank 超8 ID 均明确报错。局部 event 分配器现已支持跨 channel 的有限
生命周期打包；T5、Dv8 的三输入槽计划在当前完成契约下仍至少需要9个 ID，
诊断明确列出不可比较的通知集合，不能用改变槽数或任意 barrier 修复。
该诊断不外推为所有排步均硬件不可行，详见 [event-allocation.md](event-allocation.md)。

## 设备证据与复现

2026-10-13，xDSL0.73/CANN9.1/dav-2201，全部运行选择健康空闲的物理910B3卡1、
显式映射逻辑0；每个进程保存全卡健康/进程/HBM/利用率检查，未 reset 或换卡。

| sequences/T/Dv/Dk | 输入/snapshot 槽 | attention 最大绝对误差 | snapshot 最大绝对误差 |
|---|---|---:|---:|
| 2/0/8/64 | 2/2 | 0（空输出） | 0（空输出） |
| 1/1/8/64 | 1/1 | 4.6566e-10 | 3.7253e-9 |
| 2/5/8/64 | 2/1 | 1.8627e-9 | 1.4902e-8 |
| 32/16/64/64 | 2/2 | 3.7253e-9 | 2.2352e-8 |

大实例每个 block 的 UB 使用70016字节，snapshot 输出共8MiB；生成代码保持
resident carry、原生单指令列广播 multiply-add、两个 snapshot 槽和一次 kernel
启动。源码 SHA256 为
`f0243ac4182a4a792889a740b5a237767bbb2ce84b5c799035352bd9547e866e`。
该实例 ACL event stream 间隔为31.568微秒/次，包含宿主派发空隙，不是内核时间。
一次 msprof、5次 warmup 后30次连续调用，确认每次为一个32 block 的
AI_VECTOR_CORE task；设备 task 时间中位数为26.842微秒，min/max为
23.420/43.880微秒。早期样本较慢的原因尚未确定，保留全部样本，没有剔除
离群值或无界调参。Vector/MTE2/MTE3 activity ratio 的中位数为
0.932/0.085/0.234，支持此固定配置的计算/DMA 活动与重叠，不能单独证明
所有阶段最优或任意 shape 的性能。profiler 数值输出也逐元素检查通过。
没有同 dtype 的专家 GDN 性能基线，因此尚未证明达到专家性能目标。

工作区根目录执行：

```bash
.venv/bin/pytest -q compiler/tests/test_recurrent.py
.venv/bin/xdsl-npu gdn --sequences 1 --steps 1 --value-dim 8 --slots 1 --snapshot-slots 1 --check-interleavings all --output runs/gdn/one-token --compile --shared
.venv/bin/xdsl-npu gdn --sequences 2 --steps 5 --value-dim 8 --slots 2 --snapshot-slots 1 --check-interleavings all --output runs/gdn/five-token-single-snapshot --compile --shared
.venv/bin/xdsl-npu gdn --sequences 32 --steps 16 --value-dim 64 --slots 2 --snapshot-slots 2 --output runs/gdn/32-16-64 --compile --shared
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/gdn/32-16-64 --physical 1 --repeats 30
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/gdn/profile-generated \
  --application='/workspace/xdsl-910b/compiler/examples/profile_device.sh /workspace/xdsl-910b/runs/gdn/32-16-64 --physical 1 --repeats 30 --record-output /workspace/xdsl-910b/runs/gdn/profile-generated/device' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
.venv/bin/python compiler/examples/profile_summary.py runs/gdn/profile-generated
```

完整源码、编译参数、Bisheng 版本、源码/ELF/库 hash、输入 hash、两份输出与
各自误差保存在这些外层目录；编译参数与其他产品样例相同，固定为
`-O2 -fPIC -std=c++20 --npu-arch=dav-2201` 和关闭 dcpreload args。
运行数据、编译产物和 profiler 原始文件均不进入产品 Git。
