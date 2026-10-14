# 运行时 K 归约：首代清零、末代导出与同步

本切片在 `f951390` 基础上实现一个保留的 K-panel 流水，不是嵌套父子流水。
[DSL 示例](../examples/ast_reduction_gemm.py)、[独立 AscendC](../examples/ascendc_reduction_gemm.asc.in)
与[验证入口](../examples/run_reduction_gemm.py)均在 examples。输入明确为 panel-major，
编译器不增加 GM packing。`panels=0` 保持输出不变；这不是空 GEMM 输出零的契约。

## 实现与同步

`panels: int64` 是运行时参数；L1 三槽、L0A/B 两槽、L0C 一个累加器。源码核心为：

```python
for k in T.Pipelined(panels, stages=4,
                    issue_order={"cube": ("export", "load", "operands", "compute")}):
    # Stage 0: GM -> L1；Stage 1: L1 -> L0A/B
    with T.Stage(2, name="compute"):
        value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))
    with T.Stage(3, name="export"):
        if k == panels - 1:
            T.copy(value, C[group, :, :])
```

同步前物理结构的简写（实际可解析 IR 由下方命令导出）：

```text
loop k in [0, panels):
  stage 0: MTE2 A/B -> L1[k % 3]
  stage 1: MTE1 L1[k % 3] -> L0A/B[k % 2]
  stage 2: M mad_first(L0A, L0B, L0C)
  boundary last(k):
    stage 3: FIX L0C -> C
```

`npu.boundary` 保留末代区域，表示 `k=panels-1 && panels>0`，不表示 drain。
`mad_first` 首代只写 C，以后读写 C；首次 clear 仅初始化它实际覆盖的 C 区域，
不授权其他 slot 的读取。当前 native 契约的最后读取点仍为该指令可信完成点。

同步后关系与生成指令对应如下：

| 义务 | 生成机制/等待范围 |
|---|---|
| L1 ready | MTE2→MTE1 Set/Wait |
| L1 槽复用 | MTE1→MTE2，释放真正最后一次 L1 读取，不等 MAD/FIX |
| L0 ready | MTE1→M Set/Wait |
| L0 槽复用 | M→MTE1，等待旧操作数读取退休 |
| FP32 累加顺序 | 本例短 MAD 需要的 M 完成约束；首代无前驱 |
| 最终结果 ready | 仅末代 M→FIX Set/Wait，然后一次 FIX |
| 返回 | 排空已派发引擎；零次不产生任务或虚构通知 |

默认配置有6个 native 模板（其中1个末代模板）、7个 channel、124个完成节点族、
205条边族、7个证明域和4345个证明状态。运行时长度不增加这些结构。
完整 `before.xdsl`、`after.xdsl`、ASC 在外层 `runs/refactor/dynamic-k/nt-f16/generated/`。

## 两个容易漏掉的边界问题

1. **末代写不能初始化稳定段读取。** 例如每代读 scratch、只在末代写 scratch，
   若把末代写混进周期 frontier，就会错误地为第一代制造前驱。实现仅在周期
   frontier 的副本上分析末代 effects，分析后丢弃其更新。
2. **最后一条 read 退休不自动代表全部历史 read 退休。** 对全程只读、末代才
   覆写的同一 cell，要求对应 reader family 的累计读取退休前缀。普通有 writer
   窗口的 WAR 仍保留窗口内全部 reader，不能普遍缩成一个“最后 task”。

末代查询保留与 N 的等式关系，但不从0展开到 N−1。修复过的反例是静态
N=1048576 的末代查询：以实际末代作为证明域历史起点会导致 O(N) 状态扩张。
现在仅用常量 N 筛选有效余数，再证明有界平移族。

回归包含缺失末代 ready、错误 Set 激活、终端写遮蔽 WAR、未初始化 slot、
条件 clear 的借用累加值，以及独立有限字节访问/硬件 DAG 检查。有限 N 样本
0/1/2/3/4/7/17 不是任意规模形式化证明；最终 checker 与 planner 仍共享部分分析。

## 验证记录与限制

当前源码五个独立冷进程，完整源码→ASC p50 **0.2423秒**、max **0.2434秒**、
peak RSS **37652 KiB**，import/CANN 单列，未与回归同时计时。
`cold-verified/benchmark.json` 中生成 ASC 与已 profiling 源码 hash 一致：
`de361c0a962d5e4dc4cd38dd2aca2e57efa71dffc60fe43bed3eb46b4f158584`。

CANN9.1/dav-2201 离线/shared 编译及物理卡1→逻辑0设备运行已完成：
NT/f16 `(M,N,BK)=(32,64,64)`、NN/bf16 `(16,32,48)`，20组；同一配置的 binary
运行 panels=0/1/2/4/7/1024/4096。非空输出逐元素精确相等，输入与输出 guard
保持，空域保留 C。oracle 使用有界整数/64产品，保证 FP32 各中间和精确可表示；
没有放宽容差。这些模式覆盖布局及同步，不代表一般随机浮点误差的全面验收。

默认4096-panel输入共960 MiB，有效 K=262144，最终输出160 KiB。3轮独立
msprof，每轮5 warmup+30 samples，设备 p50 中位数为生成代码 **1449.822 µs**、
AscendC **1450.510 µs**，比值 **0.99953**。这是固定配置保真证据，不是完整
GEMM/FA 的竞争力结论。原始数据在 `runs/refactor/dynamic-k/profiles/`。

本例21.475 GFLOPs、1,006,796,800 GM请求字节，对应14.812 TFLOP/s和
694.43 GB/s请求率；后者不是 HBM 带宽。20 AIC×1.8 GHz×8192 ops/cycle 的
条件 Cube 模型为294.912 TFLOP/s，计算项下界72.818 µs。MTE2活动0.999、
MAC活动0.151须结合实际搬运和 timeline 解读，比例不是互斥时间份额。
没有实测 HBM 流量/持续带宽；FIX比率四舍五入为0不表示没有20次实际导出。

尚不支持：父 query/output 循环、多 runtime 轴 row pitch、一般 first/else 分支、
混合核末代协议、跨调用事件摘要、fractal 平面内任意子视图。不能以本例替代
重构 R2 的层次控制或 R3 的真实 online-softmax FA。

## 复现

在工作区根目录执行，产物仅写外层 runs：

```bash
.venv/bin/python compiler/examples/run_reduction_gemm.py --output runs/refactor/dynamic-k/reproduce --build
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_reduction_gemm.py --entry reduction_gemm --trials 5 --output runs/refactor/dynamic-k/reproduce-cold
.venv/bin/pytest -q compiler/tests/test_loop_boundary.py compiler/tests/test_loop_sync.py
```

检查所有卡健康、进程、显存和利用率，选择健康空闲卡后运行；物理1仅是本次
记录的选择，runner 会再次检查，设备错误不 reset 或换卡绕过：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_reduction_gemm.py --output runs/refactor/dynamic-k/reproduce --physical 1 --panels 0 1 2 4 7 1024 4096
```

AscendC 的 TPipe 占用 M_MTE1 IDs0/1/2，手排 L0 credits 使用3/4。
profiling 使用 `profile_reduction_gemm.sh` 包装同一入口；保存3轮生成/参考目录后：

```bash
.venv/bin/python compiler/examples/analyze_cube.py runs/refactor/dynamic-k/profiles --reduction --output runs/refactor/dynamic-k/profiles/comparison.json
```
