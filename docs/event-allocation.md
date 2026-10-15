# 二值 event 生命周期分配

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

逻辑 channel 的数量不等于物理 flag 的数量。局部 AutoSync 保留原本可验证的
连续 rotating ring；该方案不能放入每个有向 bank 的八 ID 时，改用同一组
真实命令的有限生命周期分配。不改变 task、issue order、stage、buffer、同步
插入位置或等待范围，也不增加任何完成边。

每个通知是 `(channel, source_epoch)`。仅当实际图已有
`Wait(old) -> Set(new).effect` 的路径时，这两个通知才可连续使用一个 ID。
不是词法上“旧循环已结束”，也不是 Set 把后来源任务 fence 住。此关系来自最终
同步命令、source prefix completion 和 destination gates；ID 数值不参与推导。

`events.py` 用最大二分匹配求该偏序的最小链覆盖，每条链对应一个物理 ID。
位集与迭代 DFS 避免递归深度依赖。分配后，独立最终 verifier 重新读取全部真实
ID、效果与命令，检查每个旧 Wait 到下次 Set 的路径；不会信任 matching 或
planner 证书。交错检查也从实际二值 flag 与 notification token 检查错误消费和
重发，不读取 matching 或完成图。测试枚举所有五节点前向 DAG 的传递闭包，以
独立穷举最大不可比较集合核对链数；这只验证分配算法的有限实例，不是硬件证明。

## 可共享的阶段边界

`phased_pipeline(9,1,16)` 在同一个本地 stream 中有九个静态阶段，每阶段一组
不同的 load/scale/store task identity。每个阶段仍执行 `Y=2*X`，input 只有一个
UB 槽，output 两个 UB 槽；不存在阶段 barrier。未同步 IR 的33个最终逻辑
channel 若各自占独立 ID，ready bank 会超8；实际生命周期只需要以下 ID：

| bank | ID集合 |
|---|---|
| MTE2→V ready | `{0}` |
| V→MTE2 input release | `{0}` |
| V→MTE3 output ready | `{0,1}` |
| MTE3→V output release | `{0,1}` |

input 单槽的最小路径如下；before 没有下列同步：

```text
MTE2 load[e]; Set ready[e]
V Wait ready[e]; scale[e]; Set input_release[e]
MTE2 Wait input_release[e]; load[e+1]; Set ready[e+1]
```

`Wait ready[e] -> scale[e].read_retired -> release[e].effect ->
MTE2 Wait release[e] -> load[e+1].write_done -> ready[e+1].effect`
使两个不同 channel 的 ready 安全共 ID0。output 双槽的同类路径距离为2，
不会把相邻两个仍重叠的 output ready 都塞入 ID0。删除或篡改这些资源配对时，
最终检查与独立交错模型都拒绝。

`phased_pipeline(5,3,16)` 跨阶段保留同一个物理 output 环。某些三通知 channel
需要 `[1,0,1]`，该序列不能用旧的 `base+(ordinal%width)` 表达。可选
`npu.channel event_ids` 按真实通知 ordinal 存储其静态 ID，ordinal 为
`(source_generation-lower)/stride`；Wait 使用 `g-distance`，不能用当前 g
冒充产生代次。显式表必须完整、ID为0…7，并具有中性 `base=0,width=1`，
有歧义的组合直接拒绝。常量或仍能表达为 rotating ring 的分配保持旧编码。

ASC 后端将有限选择表缩成分段短周期表达式：先选择区间，再在区间内计算
ordinal 的余数。每个区间尝试1…8的周期，取能精确覆盖的最长前缀；每次扫描
不超过本轮消费的区间长度，发现区间的时间为 O(8N)，N为有限表项数。
区间间使用平衡选择树，区间内最多八项；不重新分配ID或移动同步。例如：

```cpp
// ordinal=(g-lower), IDs=[1,0,1]
asc_sync_notify(PIPE_V, PIPE_MTE3, ((ordinal % 2) < 1 ? 1 : 0));
```

这只是纯 Scalar 整数选择，不建立常量数组或新增 GM/UB 搬运；选择表达式不能
取代 event 消费证明。shape/phase 更多时，条件表达式会增加 Scalar 指令成本，
不能仅根据分配算法宣称没有性能损失。

2026-10-15直接审查生成ASC发现，旧代码只识别整张表的短周期；跨query尾部
更换rotation后，128项表生成127个三元选择。修复后的4K MHA源码从92176字节
降至49032字节，三元选择2044→228。全部1056个显式表项逐项核对；源文件只改
16条本地event选择语句，原生指令、等待位置和IR不变。CANN9.1/dav-2201离线
编译含共享库6.54s通过。这不是Python同步分析加速或设备性能提升的证据；
有限表本身仍随generation数量增长。证据在外层`runs/attention-next/asc-audit/`。

## ASC纯整数表达式接入标准pass

`asc_scalar.py`将地址偏移和本地event编号构造成临时`arith` SSA函数，执行
`canonicalize → cse → canonicalize`，再打印实际交给CANN的ASC。共享结果输出为
局部`const int64_t`/`const bool`，例如`g-0-0`变为`g`，重复`g%2`计算合并。
finite、runtime叶以及使用这些后端的region路径共用此入口。异步操作不进入
这个临时函数；原生指令、同步位置、buffer和已验证物理IR保持原样。

仅使用xDSL0.73.0已有规则，没有新增范围/同余分析、guard合并或event重新分配。
原生指令内部row/chunk循环的索引、外层guard和host输入检查仍由原emitter打印。
该版本没有覆盖的`%1`等恒等式也不额外补齐。所有SSA除数是正整数常量，临时
结果局限在所属task/command作用域，不跨异步任务共享；此机制不改变浮点运算。

25项定向回归通过，包括runtime前端、prefix、局部/CV编译、Load3D及CPU实际执行
生成的整数代码。对同一4K FA物理计划逐项执行event Set/Wait编号，共2112次
比对通过，原生与同步调用顺序保持。ASC输出阶段1.08s，CANN离线编译含共享库
6.49s；这是单次测量，没有编译加速或设备性能提升结论。提取共享变量后源码
49032→52429字节，三元选择228→216；文件变短不是本轮验收目标。

最新ASC及审查数据：外层`runs/attention-next/standard-passes/`。从工作区根复现：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_asc_scalar.py compiler/tests/test_compilation.py compiler/tests/test_load3d.py
.venv/bin/xdsl-npu --input runs/attention-next/mha4k-load3d/after.xdsl --synchronized --output runs/attention-next/standard-passes-reproduce --compile --shared
```

## 真正无法证明八 ID 安全的情况

GDN 的 T5/Dv8/三输入槽计划仍拒绝。根据当前可信完成契约，MTE2→V 有九个
两两无法证明可复用的通知：三种输入 ready 各有三个尚未消费的 epoch。
错误诊断列出这九个实际 channel/epoch，不再只报某个 channel 的 base 超限。
这是“在当前完成关系下至少需要九条链”的证据，不能外推为采用任何其他排步
的硬件不可能。同步器不修改专家排步、缩小槽数、增加任意 barrier 或采用 mutex
来消除这个诊断。单 channel 的非均匀生命周期也会尝试有限分配，而非仅因
rotating width 不合适就提前拒绝。

本机制只处理本地二值事件。混合核 mode-2 counting notification、动态 stream、
分支参与者与全局 phase 有各自的配对/ownership 契约，没有被冒充为此机制支持。

## 分层验证与设备结果

2026-10-14，xDSL0.73/CANN9.1/dav-2201，物理910B3卡1映射逻辑0，每个进程
先保存全部卡健康、进程、HBM与利用率；未 reset、换卡或使用 `/tmp` 锁。

| 实例 | 独立模型穷尽状态 | Native task实例 | UB字节 | 设备最大误差 |
|---|---:|---:|---:|---:|
| 九阶段×一代次，tile16 | 3062 | 27 | 192 | 0 |
| 五阶段×三代次，tile16 | 5516 | 45 | 192 | 0 |

两个样例都离线生成 ELF 与共享库，运行5次 warmup+10次连续测量。这里只做
正确性验证；ACL stream 间隔包含宿主空隙，没有专家性能比较。
已有32序列GDN和768×1024×768混合核GEMM+bias+SiLU的生成源码 hash 仍分别为
`f0243ac4182a4a792889a740b5a237767bbb2ce84b5c799035352bd9547e866e` 与
`08fe73665f5a4ad750f9e8f2b87a699447e7129e3bf5c32a524c39db570662b8`；
分配扩展没有改变这两个原已可编译计划的生成代码。

完整 before/after IR、重用 witness、ASC、编译参数/版本/hash、输入/output数据
在外层 `runs/events/{nine-phases,epoch-selection}`。工作区根目录复现：

```bash
.venv/bin/pytest -q compiler/tests/test_events.py
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build phased --phases 9 --phase-length 1 --columns 16 --check-interleavings all --output runs/events/nine-phases --compile --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build phased --phases 5 --phase-length 3 --columns 16 --check-interleavings all --output runs/events/epoch-selection --compile --shared
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python -m benchmarks.legacy.run runs/events/epoch-selection --physical 1 --repeats 10
```
