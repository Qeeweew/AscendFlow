# 二值 event 生命周期分配

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

ASC 后端将有限选择表缩成小周期/平衡条件表达式。例如：

```cpp
// ordinal=(g-lower), IDs=[1,0,1]
asc_sync_notify(PIPE_V, PIPE_MTE3, ((ordinal % 2) < 1 ? 1 : 0));
```

这只是纯 Scalar 整数选择，不建立常量数组或新增 GM/UB 搬运；选择表达式不能
取代 event 消费证明。shape/phase 更多时，条件表达式会增加 Scalar 指令成本，
不能仅根据分配算法宣称没有性能损失。

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
.venv/bin/xdsl-npu phased --phases 9 --phase-length 1 --columns 16 --check-interleavings all --output runs/events/nine-phases --compile --shared
.venv/bin/xdsl-npu phased --phases 5 --phase-length 3 --columns 16 --check-interleavings all --output runs/events/epoch-selection --compile --shared
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/events/epoch-selection --physical 1 --repeats 10
```
