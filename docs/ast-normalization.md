# AST 归约、Scalar 捕获与固定排步验证

本例是 FP32 RMS normalization 原语，不含 gamma/bias；不替代已有完整 RMSNorm。
算法在 [ast_normalize.py](../examples/ast_normalize.py)，独立 AscendC 对照在
[ascendc_normalize.asc.in](../examples/ascendc_normalize.asc.in)，均位于 examples。
编译器只增加 reduce、Scalar read、带标量参数的乘法及排步机制，没有识别算子名。

## 数值与存储契约

每个逻辑任务处理一行，严格保持：

```text
square = RN_f32(X * X)
partials = native_reduce_sum(each 64 elements of square)
sum = native_reduce_sum(partials)
mean = RN_f32(sum * RN_f32(1 / D))
adjusted = RN_f32(mean + RN_f32(epsilon))
root = native_sqrt_f32(adjusted)
inverse = native_div_f32(1, root)
Y = RN_f32(X * captured_inverse)
```

`N.reduce_sum(..., scratch=partial, order="r64_then_native_merge")` 明确归约顺序和
用户提供的 scratch。不是把任意归约改为该顺序。完整输出、partials 和 statistics
的 footprint 不相同：D=4096 时分别为16384、256、4字节。统计量 allocation
对齐为32字节，但有效读写保持4字节。显式 input/work 双槽、部分和和统计量各自
保留生命周期。每个 AIV 申请65856字节 UB，未扩大缓冲来满足同步。

参考依据为 ops-nn HEAD `64b41ad63a5073bc711ba03802498bfe8bdb77e5` 的
`norm/rms_norm/op_kernel/rms_norm_merge_n.h:256–265`：其核心顺序也使用
mean、epsilon、Sqrt、Duplicate(1)、Div。这里的 R64 recipe 和无 affine 输出
是本例明确的数值契约；不能把上游完整算子或安装 ACLNN 的性能归到本例。

## Scalar 值不等于异步完成

```python
N.reduce_sum(work, axis=0, scratch=partial,
             out=statistics[:1], order="r64_then_native_merge")
# 显式 mean / epsilon / sqrt / div 后：
inverse = N.load_scalar(statistics[:1])
N.mul(inputs, inverse, out=work)
```

`npu_source.scalar_read` 具有实际访存 effects，降为同步的 Scalar UB read。
异步 Vector writer 完成之前，仍必须生成并验证 V→S ready；SSA use-def 不能
替代它。原生 Scale 的 f32 参数在发射时捕获；它不继续读取 statistics。

实际物理 IR 顺序为 `reduce_sum64 → merge_sum → muls → adds → sqrt → fill →
div → scalar_load → scale_value`。局部 V barrier 和 V→S Set/Wait 保护对应 RAW。
跨任务的 statistics WAR 从 `scalar_load[g]` 指向下一代 writer，而不是从整个
normalization 的 Vector 完成指向 writer。删除 V→S channel 会被最终 checker
拒绝；将 capture 移到 consumer 后面或不同 stage 也会拒绝。

这是最早可复用条件；实际下一 writer 还受自己的计算依赖和 pipe 顺序约束。

当前寄存器捕获只支持同 source scope、同物理 step、capture 先于消费。跨 stage
descriptor 保存仍需明确版本化存储；没有用一个 Python 变量冒充跨代次快照。

## 源顺序与物理 issue order

源码数值顺序仍为 load → compute → store，固定物理顺序由：

```python
N.Pipelined(rows, stages=3,
            issue_order={"vector": ("store", "load", "compute")})
```

指定。物理 step j 先发射 store[j−2]，再 load[j]，最后 compute[j−1]，使旧输出
不会被当前 Scalar Wait 阻挡。地址和 generation 仍随所属任务位移。没有在
循环中增加 PIPE_ALL。错误/缺失/重复 Stage 名及空 actor 排步列表不会被忽略。
同一机制已对双向 CV 的 Cube QK→PV、Vector Output→Compute 排步做 IR 回归；
该 CV 回归没有新增设备性能声明，更不是 online-softmax FA 的迁移证明。

## 前端整数/布尔 SSA

`if/else` 现在可合流已有或两边都定义的整数/布尔值，生成 `scf.if` results/yields。
`N.serial` 的携带标量生成 `scf.for` iter_args/results，零次循环返回初值，上界
只在入口求值一次。显式 `N.unroll` 的 constexpr 更新也会保留，修复了此前把
更新恢复为循环前值的问题。缓冲重绑定不会被当成免费的快照或隐式复制。

这些结构有独立小整数解释测试；**一般 scf 分支/嵌套循环的物理同步 lowering
仍拒绝编译**。当前 normalization 运行路径没有依赖尚未完成的结构化控制后端。

## 实测与硬件解释

Ascend 910B3，物理卡1→逻辑0；每个进程检查全部8张卡健康/进程/HBM/利用率。
默认 D=4096、32个 AIV，每个 binary 的 rows 参数覆盖0/1/2/4/256。大例形状为
8192×4096，共33,554,432个 f32 元素。生成代码与 AscendC 输出 hash 相同；
与独立 CPU 参考最大绝对误差为 `2.384185791015625e-7`，输入和输出 guard 不变。
epsilon 固定为 RN_f32(1e-6)，CPU 对照阈值固定为 rtol=5e-5、atol=5e-6；
设备间输出 hash 一致的检查比这个误差阈值更强，但仍只覆盖实际测试输入。

下表是3个独立 profiling 进程，每个预热5次、计30次的设备 task 时间，单位 µs。
两方相同输入/输出复用策略；不是 ACL stream 间隔或 host 端到端计时。

| 轮次 | 生成代码 p50 / p90 | AscendC p50 / p90 |
|---|---:|---:|
| 0 | 218.314 / 219.464 | 234.316 / 235.204 |
| 1 | 218.206 / 219.144 | 234.034 / 235.204 |
| 2 | 218.456 / 219.124 | 234.764 / 235.604 |

各轮 p50 的中位数比值为0.9317，即该固定配置延迟低约6.8%。本例未观察到
性能回退；这不证明所有算子或所有配置保真，也没有把差异归因于某一条优化。
期间没有搜索 tile/slot 配置。

硬件共有40个 AIV，本例使用32个；每核 UB 192 KiB，申请约33.5%。GM 请求
包括128 MiB读和128 MiB写，工作集256 MiB大于192 MiB L2。单独输入仍可放入
L2，输出缓存策略也影响真实 HBM 流量，因此不能直接把256 MiB当作 HBM 字节。
UB 操作数请求合计1,078,329,344字节，重复源操作数可能共享实际读取。

算术工作量为100,671,488个 add/mul 操作，另有8192个 sqrt 和8192个 div；
Cube FLOPs 为0。按 GM 请求计算的 add/mul intensity 约0.375 op/byte。
生成版本的 GM 请求有效吞吐约1229.6 GB/s，**不是实测 HBM 带宽**。

理论约束应使用 `max(各资源工作量/对应持续上限, 依赖关键路径)`；共用 Vector
资源的指令需要混合指令吞吐模型，不能将其全部代入 Cube TFLOP/s。当前 AIV
频率查询状态为 -8255，持续 HBM/UB 带宽和真实 HBM 流量未测，故不报告伪精确
硬件 roofline。msprof 中生成侧 MTE2 activity 约0.954–0.956、Vector约0.547；
指标会重叠，只提供调查方向，不能单凭比例给出瓶颈因果结论。

完整源码到 ASC 的5次冷进程中位数0.2895秒、最大0.2900秒、RSS约37.2 MiB，
import/CANN 分开。native模板数12，周期2；不会按 rows 产生 Python task 实例。

## 复现

工作区根目录执行；运行前检查全部卡，以下选择物理卡1：

```bash
.venv/bin/python compiler/examples/run_normalize.py --output runs/refactor/ast-normalize-comparison --build
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_normalize.py --output runs/refactor/ast-normalize-comparison --physical 1
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_normalize.py --entry rms_normalize --trials 5 --output runs/refactor/ast-normalize-cold
```

对 generated / ascendc 各采3轮，分别使用独立输出目录及 `--record`：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/refactor/ast-normalize-profiles/trial-0/generated --application='/workspace/xdsl-910b/compiler/examples/profile_normalize.sh --output /workspace/xdsl-910b/runs/refactor/ast-normalize-comparison --record /workspace/xdsl-910b/runs/refactor/ast-normalize-profiles/trial-0/generated/device.json --physical 1 --variant generated --rows 256 --warmup 5 --repeats 30' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
.venv/bin/python compiler/examples/profile_summary.py runs/refactor/ast-normalize-profiles/trial-0/generated --warmup 5 --repeats 30
.venv/bin/python compiler/examples/analyze_normalize.py runs/refactor/ast-normalize-profiles --output runs/refactor/ast-normalize-profiles/comparison.json
```

IR、ASC、共享库、CANN选项/版本、输入输出 hash、硬件快照、CSV和频率记录均
在外层 runs。分析器检查 shape、grid、数值输出、cache policy、编译选项和
源码/库/CSV hash 一致；手写 AscendC 不冒充被本编译器 verifier 证明的 C++。
