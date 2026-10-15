# 父循环体的初始化、子流水和异步导出

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

实现保存点 `ed87249`，G0验证于2026-10-15收尾。源码是
[grouped_state_sum](../examples/state/grouped.py)，独立参考是
[AscendC模板](../benchmarks/references/ascendc/ascendc_grouped_state.asc.in)。输入按
`[group, output, tile, B]`组织，不包含隐式packing。

## 语义和实际同步

两个长度outputs/tiles均为runtime，B/GROUPS为constexpr。物理IR保留两层设备
循环和四个native模板。每调用先加载Initial，再按原次序FP32累加，最后导出；
child=0导出Initial，parent=0不写。state和双槽x在所有调用间复用。

插入前IR投影：

```text
invocation p in outputs:
  once r0: MTE2 Initial[group,p] -> state
  stream r1(tiles):
    stage0: MTE2 X[group,p,g] -> x[g%2]
    stage1: V add(state,x[g%2]) -> state
  once r2: MTE3 state -> Output[group,p]
```

插入后同步投影，忽略无效stage实例，实际代码还包含对应条件：

```text
r0: if p>0 Wait(MTE3,MTE2,call0)
    load Initial
    if tiles==0 Set(MTE2,MTE3,empty0)
r1.load[g]: if g>=2 Wait(V,MTE2,(g-2)%2)
            load X[g]; Set(MTE2,V,g%2)
r1.add[g]:  if g>=1 Pipe(V)
            Wait(MTE2,V,g%2); add
            if g==tiles-1 Set(V,MTE3,output0)
            if g+2<tiles Set(V,MTE2,g%2)
r2: if tiles==0 Wait(MTE2,MTE3,empty0)
    else Wait(V,MTE3,output0)
    store state
    if p+1<outputs Set(MTE3,MTE2,call0)
kernel exit: Pipe(ALL)
```

不同有向bank中的ID0独立；这里只用一个跨调用MTE3→MTE2 credit，不seed，首调用
不Wait、末调用不Set。其余两条区域ready分别处理空/非空child。
完整[source/before/after IR和ASC](../../runs/refactor/parent-sequence/g0-current/)
与运行数据在外层，不提交产品Git。

为何足够且未扩大等待：

- 初始化MTE2先于首输入MTE2，首个输入ready覆盖初始化；无需再加独立MTE2→V。
- 输入x的release在实际Vector最后读后，允许预取，不等输出MTE3。
- state的下一调用初始化必须等旧MTE3读退休。换成较早Vector结束会在旧DMA
  仍读state时覆盖；静态checker和独立有限oracle都有该负例。
- 固定源政策要求state初始化在本调用输入load之前，因此跨调用credit也会推迟
  这些load。这是该政策的必要约束，不是任意额外整child drain。
- 同ID重复使用另检查Wait消费→下一Set生效；通知生效不fence后续source-pipe
  native工作。父次数为0/1/2、子为空/短于ring/跨ring都有对应激活路径。

生产分析只用符号相邻调用p/p+1，不展开outputs或tiles。最后writer及所有pending
readers保留真实root，GM frame跨调用独占。后续G1已接通
[无frame只读共享GM](shared-invocation-inputs.md)；跨sibling写同frame、不同child
runtime bound或mixed层次仍不支持。叶入口假设在最终组合重新检查。

## 验证

相关sequence、region、invocation、loop、frontend、Cube、Scalar、prefix、budget、
compilation和performance回归共303项通过，86.89秒，Ruff通过。其中父体21项
包含两child复用scratch、错误实际frame坐标、缺失return、carried改forward、提前
release、Scalar/end语义、通知不能fence后续native及百万常量结构计数。
有限逐字节/事件DAG检查不构成任意规模形式化证明。

generated/独立AscendC的CANN离线和shared构建已成功，ASC未变化，复用已有产物。
旧state_sum、hierarchical_gemm、reduction_gemm的ASC hash也保持原值。
新父体generated ASC SHA256：
`aa6a8304ccae1b6d016066701dab37dc5fae093c8bcd8c19174eaa5a8b1fa4cd`。

小例9组outputs:tiles加各自舍入反例，共20条设备结果误差0。固定大例
B4096/G32/outputs16/tiles128，X=1GiB，Initial和Output各8MiB；全部2097152个
输出与按次序的精确FP32 oracle相等，输入不变、guard完好。每次profiling也执行
完整检查；不只抽样。设备为物理1→逻辑0的910B3，所有卡健康/进程/显存/利用率
在运行前检查；没有reset/换卡重试。

五次冷源码→ASC p50=0.88045s/max=0.88241s，最大RSS=38320KiB；启动/import和
CANN另计。这是该例的成本，不代表完整FA。父体工作量统计现按region报告，
不再从首个once继承错误的aggregate tasks_per_invocation。

## 固定政策性能与硬件解释

每版本三批，顺序generated/AscendC、AscendC/generated、generated/AscendC，
每批warmup5/repeats30。msprof为每launch一个AI_VECTOR_CORE task，32 blocks。

| 版本 | 三批p50（μs） | 三批p90（μs） |
|---|---|---|
| generated | 752.914 / 755.356 / 753.352 | 756.292 / 758.016 / 755.716 |
| 独立AscendC | 757.914 / 753.424 / 756.924 | 759.816 / 756.456 / 758.876 |

批中位数比值generated/reference=0.99528，差异小于1%，批间方向有变化，只说明
该固定政策下未观察到明显退化，不作确定加速或全编译器性能无损结论。
非profile的ACL stream间隔分别755.207/765.066μs，包含host空隙，不能与kernel
时间混为同一测量。输入/输出分配重复使用，无显式cache flush。

精确工作量：268435456次FP32加法，GM请求1090519040字节，UB读写请求
4311744512字节，每AIV峰值49152字节；总工作集为192MiB L2的5.4167倍。
requested GM速率约1447.56GB/s，**不是实测HBM带宽**。此例超过L2但仍可能命中
部分缓存；本次没有额外buffer轮换测量，不能外推全部HBM行为。

profile中generated的MTE2活动率约0.973–0.974，Vector约0.236，Scalar约0.043；
参考分别约0.972–0.973、0.241–0.242、0.052，MTE3都约0.004。结合每任务16KiB
load、依赖的4096个加法和双槽代码，结果与搬运压力较大的政策一致；活动率可
重叠，不能仅凭此断言HBM饱和或量化等待损失。

理论条件下界使用`max(actual_HBM_bytes/B_HBM, UB_request_bytes/B_UB,
268435456/P_vector_add, dependency_path)`。当前HBM持续带宽、UB有效吞吐和
Vector持续频率/峰值未测，不给虚构数值下界。硬件快照HBM频率1600MHz，AIV
频率查询返回不支持；AIC快照800MHz不代表Vector执行频率，也不用于本例峰值。

数据：`runs/refactor/parent-sequence/performance.json`绑定各源码/库、编译命令、
CSV hash、硬件平台及频率记录；`profiles/trial-*`保留全部样本、device和summary。
没有拿该局部结果替代FA或ACLNN竞争性能验收。

## 复现

工作区根目录执行，设备前重新选择健康空闲卡，下例为physical1：

```bash
.venv/bin/python -m pytest -q compiler/tests/sync/test_sequence_invocations.py compiler/tests/sync/test_sequence_sync.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.run_grouped --output runs/refactor/parent-sequence --build
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.state.run_grouped --output runs/refactor/parent-sequence --physical 1 --cases 16:128 --warmup 5 --repeats 30 --skip-rounding-case --record runs/refactor/parent-sequence/device-large.json
```

一个profile的命令如下。generated/ascendc各三批，按上述顺序替换variant和trial，
不要并行运行；profile目录使用新目录，防止混入旧CSV。

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/refactor/parent-sequence/profiles/trial-0/generated --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.state.run_grouped --output /workspace/xdsl-910b/runs/refactor/parent-sequence --physical 1 --cases 16:128 --variant generated --warmup 5 --repeats 30 --skip-rounding-case --record /workspace/xdsl-910b/runs/refactor/parent-sequence/profiles/trial-0/generated/device.json' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/refactor/parent-sequence/profiles/trial-0/generated
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.analyze runs/refactor/parent-sequence/profiles --output runs/refactor/parent-sequence/performance.json
```

wrapper显式使用venv，避免msprof解析Python软链接后丢失环境。最后的分析命令
要求已有六份summary/device；校验相同政策、调用数、输入/输出、编译flags、
源码/库和设备身份，并检查实际Vector工作量。
