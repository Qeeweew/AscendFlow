# 跨父调用共享只读 GM

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

G1在统一源码入口之后增加的受限ownership能力。源码
[shared_state_sum](../examples/state/shared_inputs.py)让Initial/group和X/group/tile
在所有output调用中重读，只有Output按group/output分frame。也有所有group读取
同一份GM输入的回归；未增加copy、buffer或同步。

## 验证合同

此前“每个GM根都必须有frame”过严；直接删除检查又会让未知并发写通过。
现在区别两类引用：

- 已分frame的根：所有native访问必须通过实际FrameOp，校验调用坐标、extent、
  root和独占输出域。禁止绕过frame访问原root，否则会把同一分配误作独立存储。
- 无frame的根：必须有外部初值合同，并且实际每个native effect都是R。W、RW
  均拒绝；`initialized=True`不授权写入，`access=read_write`也不排除实际纯读。
  输入在kernel期间不受外部并发修改仍是调用合同。

group范围现在按实际引用决定：已检查的frame剥离了group/调用坐标，可用局部
frame域；共享输入保留完整group域。边界、嵌套view和native对齐检查都采用这个
区分。不能因为存在父循环就把所有引用的access_groups都改成1。

最小反例：四个group读取各自16个f32，将最终物理view统一平移64字节。
它在group0仍合法，group3越界。回归绕过source parser，直接变异最终IR；
生产checker拒绝，独立逐字节oracle在group3也发现越界，避免只测group0。
还覆盖已初始化共享根被写、未初始化输入、raw/frame混用和所有group共同只读。

当前不接纳无frame写、跨sibling写同frame、动态间接ownership或随父调用变化的
共享地址。共享动态extent仍需符合现有受限uniform/child-bound模型；这不是
一般BNSD多参数地址或任意层次CV已完成。

## 实际代码和同步

B4096/G32的ASC地址片段：

```text
Initial + core_id * 16384
X + tiles * core_id * 16384 + g * 16384
Output + (core_id * outputs + p) * 16384
```

输出仍是检查后的frame，输入无调用轴。两个设备循环、四native模板、state与
双槽输入共49152字节UB保持不变。同步仍为两个child channels、空/非空输出
ready及一个跨调用MTE3→MTE2 credit，没有共享输入的写冲突，也不加无意义release。
同步前后与最小WAR反例见[父体闭环](runtime-sequence-calls.md)。每kernel只有
最终PIPE_ALL，无每调用drain。

旧state_sum/hierarchical_gemm/reduction_gemm/grouped_state_sum和旧独立AscendC
源码hash均不变。新例ASC、before/after IR、CANN命令/产物hash保存在
`runs/refactor/shared-inputs/device/`，源码/模板及runner属于产品Git。

## 正确性、编译和大规模profile

专用12项回归通过（11项主组加全group共享例），另235项相关loop/sequence/
invocation/Cube/prefix/performance回归通过。有限oracle增加真实group坐标和
分配边界，分别检查各group字节历史；它不是一般跨核交错证明。Ruff通过。

五次冷源码→ASC中位数0.8639s，最大0.8662s，RSS38332KiB，import/CANN另计。
完整记录及package/source/ASC hash在`runs/refactor/shared-inputs/cold`。

CANN离线/shared构建成功。physical1→logical0，运行前检查全卡。generated与
独立AscendC各9个小case和舍入反例全部精确正确，输入/guard不变。大例
B4096/G32/outputs16/tiles2048：共享X本身1GiB，Initial512KiB，Output8MiB，
共2097152个输出全部比对；所有profile进程也执行完整检查。

三批交替顺序，每批warmup5/repeats30，32个AI_VECTOR_CORE blocks：

| 版本 | 三批p50（μs） | 三批p90（μs） |
|---|---|---|
| generated | 11284.756 / 11293.986 / 11278.808 | 11298.708 / 11305.004 / 11300.248 |
| 独立AscendC | 11286.864 / 11281.218 / 11295.044 | 11307.348 / 11307.024 / 11313.764 |

批中位数比0.999813，在观测波动内一致，不宣称更快。无profile的ACL stream间隔
分别11289.661/11257.828μs，包含host空隙，另列而非充当kernel时间。

16次调用共有4294967296次FP32加法，GM请求17196646400字节，UB读写请求
68736253952字节。实际工作集1082654720字节，约192MiB L2的5.3776倍。
requested GM速率约1523.88GB/s，不是HBM实测流量/带宽。输入跨调用复用且
无flush；本次没有额外轮换分配试验，不能外推所有缓存条件。

首批generated MTE2/Vector/Scalar活动率约0.987/0.250/0.044，参考约
0.986/0.257/0.054；其余批次数据保留。与大输入搬运和双槽政策一致，但各活动率
可重叠，不能单凭它们断言HBM饱和。理论条件下界按实际HBM流量、UB请求、加法
需求及依赖关键路径分别除以可核实吞吐再取max；当前没有持续HBM/UB/Vector
上界测量，不给虚构数值下界。硬件身份和频率查询限制沿用同卡父体报告。

## 复现

在工作区根目录执行。设备前按全卡健康与占用选卡，以下物理1映射逻辑0：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_shared_invocation_inputs.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/state/shared_inputs.py --entry shared_state_sum --trials 5 --output runs/refactor/shared-inputs/cold-recheck
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.run_grouped --shared-inputs --output runs/refactor/shared-inputs/device --build
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.state.run_grouped --shared-inputs --output runs/refactor/shared-inputs/device --physical 1 --cases 16:2048 --warmup 5 --repeats 30 --skip-rounding-case --record runs/refactor/shared-inputs/device-large.json
```

profiling使用[同一wrapper和流程](runtime-sequence-calls.md#复现)，application增加
`--shared-inputs`、case改`16:2048`，artifact output为`shared-inputs/device`，profile
为`shared-inputs/profiles/trial-N/variant`。各三批依次generated/ascendc、ascendc/
generated、generated/ascendc。每批生成summary，再运行：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.analyze runs/refactor/shared-inputs/profiles --output runs/refactor/shared-inputs/performance.json
```

分析校验政策/输入输出、调用数、源码/库、flags、CSV、设备和Vector工作量。
参考沿用独立手写state模板，只改变两处输入GM基址；输出frame与同步次序不变。
算子优化仍暂停，这里验证的是编译器的共享输入语义和固定政策保真。
