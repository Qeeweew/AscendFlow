# Flash Attention：当前性能与 GM→L1 差距

2026-10-16。当前已完成设备验证的政策为 BNSD、B1/Hq8/Hkv8、Sq=Sk4096、
D128/f16，BQ128/BK1024/CK128/VR8，20组AIC/双AIV。真实公开DSL源码位于
[attention.py](../examples/flash_attention/attention.py)。性能尚未通过相对ACLNN≤1.05的目标。

## 三批设备计时

每个后端交替运行三批，每批5次warmup、30次device task计时；排除host提交间隙。
物理6→逻辑0，每个进程先检查全部卡的健康、占用、显存和利用率。
相同Q/K/V字节、scale、误差合同；每head每query tile采样6行，另检查全输出有限、
输入不变和保护区。复用相同allocation、不flush；64.5MiB工作集小于192MiB L2。

| 后端 | 三批p50，µs | 三批中位数，µs |
|---|---|---:|
| 当前DSL，softmax→update | 606.220 / 605.780 / 605.692 | 605.780 |
| 此前DSL，update→softmax | 822.950 / 822.660 / 821.910 | 822.660 |
| 独立固定政策AscendC | 1191.320 / 1191.220 / 1190.810 | 1191.220 |
| 安装版ACLNN | 512.530 / 512.152 / 510.790 | 512.152 |

改排步使延迟降低26.4%，但仍慢于ACLNN **18.3%**。实际ASC的DMA调用多重集、
MAD调用与旧版一致，未改变算术或缓冲；同步按新顺序重新生成。普通msprof中，
AIV Vector活动380.181µs（旧版380.455），AIC MAC261.279µs（旧版261.689），
支持收益来自重叠改善，而不是算术工作减少。

证据位于外层 `runs/refactor/cv-order/performance.json`、`profiles/`、`asc-audit.json`。
当前ASC SHA256为`e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`，
CANN9.1、dav-2201、-O2；源码/IR/库/选项均保存在`build/`。旧block8和R64证据分别在
`runs/refactor/compile-vector/`、`runs/refactor/query-boundaries/`。

## msprof op 的其他指标

另用MindStudio op profiler 26.1.0，`--aic-metrics=Default --replay-mode=kernel`，
5次应用warmup后采集3次launch。以下冲突比例是每次全AIV中位数再取中位数，
搬运字节与指令是全核总和；三个launch的整数计数完全一致。这些replay数据用于归因，
不替换上面的三批30次普通task计时。

| 指标 | 当前DSL | ACLNN |
|---|---:|---:|
| Vector bank+bank-group冲突比例 | 13.714% | 8.032% |
| 其中bank冲突 | 7.755% | 3.548% |
| 其中bank-group冲突 | 5.959% | 4.484% |
| GM→L1请求 | 776MiB | 544.3125MiB |
| AIC MTE2指令 | 24,852 | 17,508 |
| AIC MTE1指令 | 141,332 | 57,464 |
| AIV MTE2指令 | 32,808 | 20,560 |
| AIC L2读命中率，核中位数 | 97.71% | 95.41% |
| AIV L2读命中率，核中位数 | 100% | 99.29% |
| 全核Cube指令 | 16,384 | 16,384 |

冲突数据支持继续优化UB访问；GM→L1请求多42.6%则直接暴露K/V重复载入。
当前源码预算可精确复现PMU的776MiB：256个query tile，Q只在首KV载入，合计8MiB；
每query的K、V、P各有4个KV×8个CK×32KiB，即各256MiB。
安装版实际选中的`...18324677680_mix_aic`元数据启用了L1 reuse。安装arch22源码把
同一AIC的两个AIV绑定到相邻query；K/V基址不随query变化，Matmul缓存仅在GM基址变化时
失效，完整tile命中后直接取L1，不重发GM读取。

用户指定下一步先解决GM→L1。对应政策是每个通信块含两个BM128原生行块，K/V每个CK
只搬一次，分别供两个QK/PV MAD读；Q/P各自保留。源码请求预算为
`Q8 + K128 + V128 + P256 = 520MiB`。它不是ACLNN的实际tiling反推，也不是已测结果。
候选BQ256/BM128的最新构建/设备状态见[状态页](refactor-progress.md)。

L1→L0的8次Load2D展开可解释141,312条源码搬运与PMU141,332的接近，但指令数多
本身不足以证明它是端到端瓶颈，本轮不把Load3D列为优先优化。

原始op CSV、raw PMU、源码/库hash和命令位于`runs/refactor/cv-order/op-metrics/`，
三次汇总为`repeated-comparison.json`。解释这些指标时须保留以下口径：

- op冲突比例分母是截断到整微秒的task duration×频率，不是Vector活跃周期；
  Default各PMU组分别replay，不可把冲突、pipe或wait时长相加分解18.3%的差距。
- Memory的GB/s列实为GiB/s，KB列按KiB换算；分母是完整task duration。
  GM请求含L2命中，不是HBM流量。
- op Cube FOPS在这两个kernel中等于指令数×8192，比实际矩阵FLOP少512倍。
  普通msprof实际采用PipeUtilizationExct，其MAC时间也不能与op Cube时间直接混比。
- 两份kernel的event ID语义不同，不能用某个wait_id在ACLNN中为0证明它没有CV等待。

## 硬件与源码边界

本卡910B3有20 AIC/40 AIV，采集时1800MHz。按每core每cycle一个完整b16 FMA fractal、
FMA计2 FLOP的条件模型，峰值294.912TFLOP/s；两次矩阵乘工作68.719GFLOP，Cube单独
理想下界233.017µs。当前整体113.44TFLOP/s，为条件峰值38.5%；下界忽略Vector、DMA
和等待，不是延迟预测。

参考ops-transformer HEAD为`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`。
安装版kernel模板家族可由安装元数据确认，但安装header与上游checkout不同，不能宣称同提交。
当前Vec2每次8行、上游D128按64行，以及统计量布局/首KV计算仍有差异；先处理GM→L1
并复测，再决定UB和Vector改动，不做无界参数搜索。
