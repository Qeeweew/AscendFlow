# 新 DSL 行运算与大规模设备验证

最新重构后复测见[当前状态](refactor-progress.md#当前-asc-与性能复测)：物理6上
32768×1024 f32的generated/AscendC/ACLNN三批中位数为210.540/239.590/267.240μs。
本页以下保留原保存点的政策、测量和硬件分析，不能与最新数据混合计算差值。

2026-10-15。同步复杂度问题保存于 `c4a7ced`，见[已知问题](sync-known-issues.md)。
本轮接通packed行归约、maximum和显式scratch行广播，新增
[softmax](../examples/normalization/softmax.py)，复测已有RMS normalization。
算法数学主体在examples，编译器仅实现typed基元。

## 实现与最终 ASC

`reduce_sum/max(..., order="r64_then_packed_merge")` 先写R64部分结果，再merge
到紧凑的 `(R,)` 统计向量。`sub/mul/div(x, row[:,None], out=y, scratch=b)`
生成BRCB和行算术；b只有R×8个f32。统计量的最后reader是BRCB，scratch最后
reader是后续算术。旧统计量被覆盖后再借用的反例会被拒绝。

softmax默认R8/D1024/GROUPS32，输入输出各双缓冲，每AIV UB为131872字节。
11个native模板、四条局部event通道和8个Vector barrier形成：

```text
load → ready → R64 max → packed max → BRCB → sub → input release
                                              → exp → R64 sum → packed sum → BRCB → div
                                                                                   → ready → store
store读取退休 → 下一代相同output slot的sub
```

input release不等待exp/sum/div，output credit不阻挡前面的max归约。runtime
循环保持一个for，只有退出时drain，循环内没有PIPE_ALL、GetValue、V↔S统计
交接或额外GM搬运。独立AscendC采用相同缓冲、数学和同步排步；连续partial
使用一条repeat归约，没有从PlanIR生成参考。

partial行的32字节对齐padding由例子显式分配，再传精确view。D64的padding
路径和D1024的连续路径均通过CANN与设备检查。当前广播为完整连续f32 UB行、
R是8的倍数、D是64的倍数且≤1024，还要满足UB容量。tail、一般广播和完整
nested CV不在本轮支持范围。

## 显式 block8 分层归约

`N.reduce_sum/max(x, axis=1, out=statistic, scratch=workspace,
order="block8_then_packed_merge")` 是独立可选顺序，输出仍为紧凑的 `(R,)`
统计量。`workspace` 必须是显式一维 f32 UB view；每级的列宽为上一级的八分之一，
直到列宽不超过64，再进行一次 packed merge。每级输入必须满足现有 native 的
完整64元素 repeat 合同，结果行必须32字节对齐。不满足整除和对齐条件的宽度
直接拒绝，不隐式padding或切回R64顺序。

workspace 按级依次保存完整连续平面，大小恰为 `R * sum(level_widths)` 个f32。
例如8×1024的级宽为128、16，需要1152个f32：前4096B保存8×128，后512B保存
8×16；最后读512B并写32B统计量。各级读写区域独立，不放宽收缩原地别名约束，
也不隐藏分配、复制或额外GM访问。与原512B partial相比，此例scratch增加4096B。
输入最后reader为第一级，结果producer为最终merge；内部RAW、跨代scratch复用
以及borrowed value检查均经过原有同步与最终验证。

该候选根据CANN9.1/2201的分层BlockReduce实现选择八元素归约；它改变FP32 sum树。
以8×1024为例，两个block8阶段各覆盖128、16次repeat，再接8次whole merge，
不能按ASC调用数宣称更快。通用宽度的Python/IR反例已覆盖sum/max、精确scratch、
输入最后reader、别名和删除必要同步；默认FA已接入并单独完成CANN及设备验证，
具体证据见[Attention性能](attention-performance.md)。本页下面的历史normalization数据
不属于新block8顺序，其他宽度也没有由FA的设备结果取得性能结论。

## 测量结果

Ascend910B3/dav-2201，物理1→逻辑0，每进程检查全部八卡的健康、进程、HBM、
利用率。CANN9.1；选项为 `-O2 -fPIC -std=c++20 -mllvm
-cce-aicore-dcpreload-args=false --npu-arch=dav-2201`。三个variant交替三批，
每批warmup5、测30次。表中为**三批kernel duration中位数的中位数**：

| 算子与shape，f32 | 生成代码 μs | 独立AscendC μs | 安装ACLNN μs |
|---|---:|---:|---:|
| softmax 32768×1024 | 217.914 | 240.054 | 284.696 |
| RMS normalization 8192×4096 | 218.346 | 235.884 | 未比较 |

RMS示例没有gamma/bias，不能冒充完整aclnnRmsNorm。softmax的generated/
AscendC均32 AIV，ACLNN实际40 AIV，具有独立tiling和数值实现。两个固定shape
相对手写参考耗时少9.22%/7.44%；softmax相对安装ACLNN少23.46%。不外推到其他
shape或FA，不宣称硬件最优。

softmax三批p50为217.914/214.084/218.396μs；AscendC为240.054/240.094/239.986；
ACLNN为284.984/278.236/284.696。各批p90、全部样本及频率采样在JSON中。
非profile单进程ACL stream间隔另为211.889/240.195/278.647μs，含host间隙，
不混入上表。最初未合并partial的手写参考测量已弃用。

每个大例检查全部33554432个输出、输入不变和guard。softmax相对float64 oracle
最大绝对误差generated/AscendC为9.31e-10，ACLNN为1.40e-9；行和误差≤1.43e-7。
RMS相对分阶段FP32 oracle误差2.39e-7。softmax的0/1/2/3个runtime tile复用同一
binary通过；D64短例误差≤5.97e-8。输入为可复现有限值序列，不覆盖所有浮点输入。

## 硬件工作量与归因边界

实机20 AIC/40 AIV，UB192KiB/AIV、L2 192MiB。softmax工作集256MiB，是L2的
1.333倍；相同allocation重复测量且没有flush，不能假设全部请求到HBM。
32 AIV使用80%的物理Vector核，UB占用约67.1%。

softmax请求268435456字节GM读写；effects计数的UB操作数请求1355284480字节。
对应1231.84GB/s GM请求速率、6219.35GB/s UB操作数请求速率，均不是实测HBM
或UB物理带宽。计算含33521664次max比较、67076096次sub/add、各33554432个
exp和div，没有Cube工作；exp/div不能当作相同成本的普通FLOP或套用Cube峰值。

三批pipe activity中位数：

| 版本 | Vector | Scalar | MTE2 | MTE3 |
|---|---:|---:|---:|---:|
| generated | .956 | .050 | .899 | .449 |
| AscendC | .963 | .073 | .440 | .166 |
| ACLNN | .597 | .223 | .235 | .089 |

活动比例可以重叠，不能相加成时间或单独证明关键路径。生成/手写版具有相同
源级搬运、缓冲、同步排步，但活动比例明显不同；约9%的差距尚未逐机器指令
归因，不能归功于某项同步优化。当前objdump未提供dav-2201可读反汇编。

理论时间应约束为 `max(GM路径服务时间, Vector混合指令服务时间, 依赖关键路径)`；
同一Vector资源上的max/sub/exp/sum/div需求要合并计费。HBM字节、持续带宽和
可靠AIV持续频率尚未测得，平台raw rate单位未核实，不填写伪造峰值利用率。

参考HEAD：ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，ops-nn
`64b41ad63a5073bc711ba03802498bfe8bdb77e5`，CATLASS
`698379e0d4b1b94d12728e7181df2cf83a958e8f`；未pull。
ops-transformer的`attention/flash_attention_score/op_kernel/arch22/
flash_attention_score_s1s2_bn2gs1.h`约1960行区分首次/update，紧凑/普通统计布局
分别调用SoftmaxFlashV2。这支持Vector基元的需求判断，不证明完整online FA
已接通。ops-nn主softmax_v2当前为arch35，不能套用到910B；安装ACLNN也不
宣称等同上述源码HEAD。

## 验证与复现

前端/归一化58项、AST Cube/编译产物31项、新行运算11项分组通过；同步保存前
另有78项。未跑分钟级全量。generated/AscendC离线和shared编译通过。
softmax五个独立冷进程源码→ASC p50 **0.3016s**、max **0.3022s**，RSS32136KiB；
import/CANN另计，未与测试或profile并发计时。

外层 `runs/dsl-vector-rows/` 的softmax/、rms/含前后IR、ASC、编译选项/hash和
非profile设备记录；profiles/、rms-profiles/含三批数据，performance.json、
rms-performance.json为汇总，cold-softmax/为编译记录。padding通用化后D1024
ASC与被profile源码逐字节相同，SHA256：
`836f61fe58cfe8a2bc8f491c2a1c1f21ead355d1226adaefc92f67840f4122d7`。
原测量DSL保存在softmax/measured-source.py。运行数据不进产品Git。

从工作区根目录执行；设备命令的物理1需按当时全卡检查选择：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_vector_rows.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.normalization.run --operator softmax --output runs/softmax --build
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.normalization.run --operator softmax --output runs/softmax --physical 1 --rows 1024 --variant all --warmup 5 --repeats 30
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/normalization/softmax.py --entry softmax --trials 5 --output runs/cold-softmax
```

profiling使用[通用包装器](../benchmarks/README.md)，替换为normalization模块和
本页参数，各variant独立目录、三批交替；汇总命令：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.normalization.analyze runs/profiles --variants generated ascendc aclnn --output runs/softmax-performance.json
```

下一步是runtime BNSD地址与nested CV的真实QK/PV/Vector stage组合，再迁移
online状态和完整FA。本轮局部softmax结果不外推为FA结果。
