# R2a：父输出循环与子K流水的真实闭环

在 `83d7d09` 的边界摘要上接通物理父子区域、GM frame、跨调用完成/事件检查及
ASC。[源码](../examples/ast_hierarchical_gemm.py)、[独立AscendC](../examples/ascendc_hierarchical_gemm.asc.in)、
[运行入口](../examples/run_hierarchical_gemm.py)均在examples。算子调参没有恢复。

## 支持的结构与地址契约

`outputs`、`panels`都是runtime int64。源程序使用一个零起点、unit-step的
`serial(outputs)`，内部保留一个 `Pipelined(panels)`。每个output的首次MAD
clear，随后按K顺序FP32累加，最后一次FIX导出；零panels保持该输出不变。
L1三槽、L0A/B两槽、L0C一个累加器，既未扩槽，也未额外GM packing。

物理IR结构简写如下，完整可解析前后IR由运行入口导出：

```text
kernel: runtime outputs, panels; original GM allocations; private L1/L0/C
scf.for invocation = 0 .. outputs:
  A_frame = npu.frame A, invocation, panels, scale=M*BK*sizeof(dtype)
  B_frame = npu.frame B, invocation, panels, scale=N*BK*sizeof(dtype)
  C_frame = npu.frame C, invocation, 1,      scale=M*N*sizeof(f32)
  npu.stream panels:
    MTE2 load -> L1[k % 3]
    MTE1 load -> L0[k % 2]
    MAD initialize=(k == 0)
    boundary last(k): FIX -> C_frame
```

Frame是别名，地址为 `(core_id*outputs+invocation)*extent*scale`，没有DMA。
最终checker从真实SSA检验frame根、parent坐标、大小和owner分区，并验证子访问
位于frame中。GM声明大小必须等于groups×outputs×frame bytes，直接越过frame
访问原root、错extent/scale/坐标均拒绝。host在uniform乘法之前检查溢出，随后
检查字节范围、对齐及外部区间不重叠。调用者仍负责实际分配满足声明大小。

当前要求GM形状的前导轴是 `[group, invocation, ...]`，帧中至多一个匹配child
bound的动态轴，native内层pitch固定。child上界在整个parent范围保持不变。
一般父body工作/SSA carry、每次调用不同的child长度、任意fractal子矩阵、
Pipeline嵌套Pipeline、多child/CV参与者仍未实现，不能把这个子集称作通用层次流水。

## 同步足够性与等待范围

单独child的局部同步不足以授权重复调用。下面是实际ASC中已有slot-credit
channel的扩展，D分别为L1的3和L0的2：

```text
old Set guard:  g < panels - D
new Set guard:  g < panels - D || invocation + 1 < outputs
old Wait guard: g >= D
new Wait guard: g >= D || invocation > 0
event selector: slot = g % D
```

同一slot在前一调用的最后实际reader产生尾部credit，后一调用的头部writer消费。
N不是D的倍数时，最后reader位于 `N-1-((N-1-slot)%D)`，不能按扁平总次数猜slot。
首调用不等不存在的credit；末调用不发没有消费者的尾部credit；N<D时只涉及
实际使用的slot。L1 credit仍来自MTE1，不扩大到MAD或FIX。

另外增加一个真实 `npu.invocation_channel`，以及定位在native前后的Set/Wait：

```text
after final FIX: if another invocation exists: Set(FIX -> M, ID0)
before first MAD: if invocation > 0:           Wait(FIX -> M, ID0)
```

最小反例是outputs=2、panels=1：若只有内层MAD→FIX ready，第二次MAD可能清零
第一份FIX还在读取的L0C。移除上述release会触发missing invocation WAR。
反之，把它放到下一调用入口并阻挡所有pipe，会扩大等待；实际等待紧贴MAD，
MTE2/MTE1可在各自slot允许时先搬下一份输入。整个kernel仅最后一次PIPE_ALL，
每个child没有whole-pipe drain。

证明分开处理内存和事件：每个byte frontier的RAW/WAR/WAW必须被实际跨调用
协议覆盖；上一调用某ID的最后Wait消费也必须先于下一调用首次Set生效。
carried ID还检查内部事件与边界事件的两处交界。所有ID按真实actor/有向bank
计费，FIX_M与M_FIX是不同bank；TPipe参考预占M_MTE1 IDs0/1/2，手排L0用3/4。

首到末的长距离关系用native退休前缀的一代递推，加两端完成路径组合，不能
把最后一条native自身完成当作全历史退休。父层进展依赖顺序调用的归纳：头部
Wait只依赖此前调用，尾部Set不阻塞源pipe，child内部仍经过原有进展检查。
最终checker重新读取物化IR，不信任planner给出的摘要标志。

这不是任意层次程序的形式化证明。当前只接受可由一个已支持的参数化协议族
覆盖的关系；跨channel条件共享、复杂guard和多actor组合仍会明确拒绝。

## 编译成本与验证

当前默认有6个native模板、124个完成节点族、205条边族；最终检查53个域、
48455个证明状态。runtime outputs/panels不增加这些规模。静态百万×百万边界
同样保留一份native body；测试不运行该巨大设备工作量。

最终5个独立冷进程，完整源码→ASC p50 **0.6889秒**、max **0.7028秒**、peak RSS
**61448 KiB**，import/CANN单列，无并发回归/profiling。数据在
`runs/refactor/invocations/cold-final/`。旧单层K生成ASC逐字不变，新ASC与实际
设备/profile库源码一致：`50b466f6f181980d11ab0cb2f3d89a1f694b6241cc85184f9f921925036110d5`。

`compile(..., timeout_s=60, cancel=predicate)`提供协作式deadline/取消。完成图
松弛、证明域和路径查询、调用边界分析等检查deadline并报告phase/结构计数。
取消不会返回半验证产物，输入IR及后续编译不被污染；CANN时间不在此Python预算内。
进程级cold benchmark另有60秒硬预算。结构预算仍独立存在，不靠延长期限掩盖膨胀。

291项扩大回归通过（76.19秒）；增补纯Vector及错误scope/ID反例后，最终相关
38项复验通过（9.34秒），集合重叠不相加。新的
独立有限硬件DAG按实际指令建立FIFO、退休捕获和Set/Wait关系，再逐字节检查
RAW/WAR/WAW及同ID再次Set；不使用生产parent摘要/完成图作为答案。覆盖空域、
1/2/3次调用与1/2/3/4/7/17个panel，并有删release、错frame、漏Set等负例。
这些有限尺寸检查不冒充任意规模的设备或形式化证明。

## 设备正确性与性能

检查全部8张卡后使用物理1→逻辑0。生成与AscendC两个版本均通过CANN离线/shared
编译和同binary多参数验证。f16 `(M,N,BK)=(32,64,64)` 与bf16 `(16,32,48)`，
20组，分别覆盖outputs=0/1/2/3/17、panels=0/1/2/7/1024及固定16×256大例。
所有非空输出逐元素精确相等，空域保持原始输出，输入/output guard不变。
oracle用有界整数/64产品确保每步FP32和精确可表示，不放宽容差；不据此宣称
一般随机浮点误差或任意dtype/布局均已验收。

固定f16大例：20组×16outputs×256panels，输入960 MiB、输出2.5 MiB。
每variant 3轮独立msprof，每轮5 warmup+30 samples，AI_CORE、20 blocks：

| 实现 | 三轮p50 / p90（µs） |
|---|---|
| 生成ASC | 1452.360 / 1454.728；1452.268 / 1456.828；1452.240 / 1457.108 |
| 独立AscendC | 1451.418 / 1454.732；1452.298 / 1454.728；1451.858 / 1455.708 |

中位数之比 **1.00028**，差异在批间波动范围内。结果支持固定policy的保真，
不表示完整GEMM、FA或ACLNN竞争力达标。该词法嵌套仍有每个child的控制填排；
尚不支持跨query连续预载。源码/选项/库/CSV/hash绑定于 `profiles/comparison.json`。

实际MAD81920、FIX320，共21.475 GFLOPs；GM请求1,009,254,400 bytes，intensity
21.278 FLOP/请求byte。设备约14.787 TFLOP/s、694.95 GB/s请求率；请求率不是
HBM带宽。每AIC L1/L0A/L0B/L0C分别36864/8192/16384/8192 bytes。
20 AIC×1.8 GHz×8192 ops/cycle的条件模型为294.912 TFLOP/s，计算项下界72.818µs。
两实现MTE2活动约0.999、MAC0.150、MTE1 0.196、FIX0.002；生成/参考Scalar约
0.144/0.132。这提示当前配置搬运的重要性，但重叠比例不能单独证明HBM瓶颈，
未知HBM/L2持续带宽保持未知，不能把计算项下界当作可直接实现的加速目标。

## 复现

工作区根目录执行，产物只写外层runs：

```bash
.venv/bin/python compiler/examples/run_hierarchical_gemm.py --output runs/refactor/invocations-reproduce --build
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_hierarchical_gemm.py --entry hierarchical_gemm --trials 5 --output runs/refactor/invocations-reproduce-cold
.venv/bin/pytest -q compiler/tests/test_invocations.py compiler/tests/test_budget.py
```

设备检查后选择健康空闲卡；下例为本次物理1，runner再次检查所有卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_hierarchical_gemm.py --output runs/refactor/invocations-reproduce --physical 1
```

profiling每个variant/批次独立进程，下面替换trial编号和variant运行共6次：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/refactor/invocations-reproduce/profiles/trial-0/generated --application='/workspace/xdsl-910b/compiler/examples/profile_hierarchical_gemm.sh --output /workspace/xdsl-910b/runs/refactor/invocations-reproduce --record /workspace/xdsl-910b/runs/refactor/invocations-reproduce/profiles/trial-0/generated/device.json --physical 1 --variant generated --cases 16:256 --warmup 5 --repeats 30' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
.venv/bin/python compiler/examples/profile_summary.py runs/refactor/invocations-reproduce/profiles/trial-0/generated
.venv/bin/python compiler/examples/analyze_cube.py runs/refactor/invocations-reproduce/profiles --hierarchical --output runs/refactor/invocations-reproduce/profiles/comparison.json
```

父output×K主闭环已完成；R2a剩余的通用循环外工作，以及R2b状态/guard/布局
子块、R2c多层CV和R3/R4真实FA验收继续保留为必需工作，整体重构未完成。
