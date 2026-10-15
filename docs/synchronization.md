# 局部流水线的同步契约

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

`Ref` 保存 storage root、以字节为单位的 offset/stride、常量整数 divisor/modulus、
独立核坐标、dtype/layout 和末块
范围。Subview/recast 不产生新 allocation；布局转换只允许在具有明确效果的 recipe
中发生。分析采用指令的完整 native footprint，不把逻辑有效 shape 当作访存许可。

依赖按算法中的 generation/source order 建立，与 stage issue order 分开。
每个字节区间记录最后 writer 和全部 reader；新写需等旧读退休及旧写完成。
Vector in-place shifted overlap 在单指令内部不合法时直接拒绝，同步不能修复它。
外部指针不同不证明不别名：`disjoint_external=True` 是 caller 保证，root 的多个 view
则由编译器检查共同物理身份。当前 runner 为这些例子分配独立参数。

## 从任务到完成图

每个原生 task 有 start/end；实际效果默认在 recipe 的可信 end 退休。SSA use-def
不建立 end→start。S 同步读取有隐式完成保证；M 不能无条件采用该保证。
经 CATLASS AtlasA2 源码审查与设备反例，少于10个输出 fractal 的短 MAD 在
相同 C 的连续累加前需要真实 M 完成路径。当前从物理依赖插入 `PIPE_M`，
不同 C 槽的首次 clear 不因此被串行化；大 MAD 保留既有隐式顺序契约。
V、MTE 和 FIX 的发射顺序保持为弱边。

Set 的发射与生效也是两个点：Set 在生效前需完成该 source pipe 的前序 native
工作，但它不阻止后续 source 指令在它生效前开始。Wait 消费匹配通知并门控
destination pipe 的后续工作。只有真实 completion 路径能满足物理依赖。
Scalar Wait 会影响全核 dispatch；完成图与交错检查均检查其门控范围。
Fixpipe 的唯一 invariant S 初始化使用一个真实 S→FIX handshake。

这是设备验证中修正的关键前提：曾把 source Set 当作 source fence，导致删除
Vector RAW barrier 后读到未写完的结果。当前完成图与状态机都将 Set 异步生效
单独建模，相关错误计划由负例回归拒绝。

## 双缓冲前后对照

以下是 `vector_pipeline(count=6)` 的可读简写；完整、可解析的 xDSL IR 由 CLI
写入 `before.xdsl` / `after.xdsl`，它们保留原始 view 和 stage。

```text
before:
  load(g)        MTE2: GM[g] -> Input[g%2]       stage 0
  scale(g)       V:    Input[g%2] -> Output[g%2] stage 1
  postprocess(g) V:    Output[g%2] += 1          stage 1
  store(g)       MTE3: Output[g%2] -> GM[g]      stage 2

after, in each task's own logical generation:
  load:
    if g>=2: wait V->MTE2[(g-2)%2]
    DMA
    set MTE2->V[g%2]
  scale:
    wait MTE2->V[g%2]
    if g>=2: wait MTE3->V[(g-2)%2]
    Muls
    if g<4: set V->MTE2[g%2]       # input last read retired
  postprocess:
    barrier V                      # Notify does not replace this RAW fence
    Adds
    set V->MTE3[g%2]
  store:
    wait V->MTE3[g%2]
    DMA
    if g<4: set MTE3->V[g%2]       # output store source retired
```

Ready 连接这代写完成到读取；release 连接旧 reader 到两代后的覆写。
输入在 Scale 最后读取后即可归还，后续 Adds 和 Store 仍可继续；新增 `last_read`
任务后，release 自动移动到该 reader。MTE3 读取 Output 是异步的，SSA 最后一次
使用 Store 的输入以后也不能马上覆盖它。

WAW 通过 Load→Scale→下一 Load 和 Postprocess→Store→下一 Scale 的真实完成
路径覆盖，不额外增加整个 allocation 的等待。每个 channel 只等待指定 pipe 的
前序工作；没有 stream 内的 PIPE_ALL，也没有新增搬运或改变 buffer 环。
kernel 入口 `asc_init` 和出口唯一 `asc_sync_pipe(PIPE_ALL)` 属于 CANN 生命周期 ABI。
出口 drain 不参与任何 stream 内依赖的证明。

## 有限事件与 slot 生命周期

事件 ID 的银行是 `(physical core, source pipe, destination pipe)`，每方向八 ID。
buffer ring 和 event ring 是不同资源。分配前采用逻辑 channel，删除冗余候选必须
重新建立覆盖；event ring 从实际 old-Wait→next-Set 路径确定。
不相交的不同 channel 生命周期也可以共享 ID；rotating ring 放不下时，有限
偏序分配仅选择 ID，不增加完成边或扩大等待。不可证明的八 ID 计划给出实际
通知集合诊断。表编码、最小反例与设备验证见 [event-allocation.md](event-allocation.md)。
本版不实现跨 channel 的主动 ID 共享搜索，但 verifier 可以检查最终真实冲突。
分配失败诊断是当前方案的失败，不宣称硬件上不存在别的合法方案。

release 只对有下一消费者的代次产生 Set，Wait 只在旧代确实存在时激活。
周期 producer 的 bank ID 按实际通知序号 `(g-lower)/stride` 轮换，而不是直接
用稀疏 stream generation modulo 环宽。静态非规则义务分解成精确的区间并集，
不会在空洞中创造 Set/Wait。
因此 startup 无 seed，所有通知都有且仅有一次消费，epilogue 无剩余 credit。
N=0 无 stream 通知；N=1/小于环宽时只产生实际 ready。
最终 verifier 从实际 SyncOp 位置重建配对、完成闭包和每个真实 ID 的消费路径。
`witnesses.json` 分别保存 data/storage 义务和 notification consumption 的具体路径。

## Cube/Fixpipe 的独立生命周期

```text
GM->L1 write_done -> MTE1 operand load
MTE1 L1 read_retired -> next GM->L1 overwrite
MTE1 L0A/B write_done -> MAD
MAD L0A/B read_retired -> next operand overwrite
MAD L0C write_done -> FIX
FIX L0C read_retired -> next MAD clear
```

L1 release 放在 MTE1 最后读取后，不等待 MAD 或 FIX。L0A/B 归还在 MAD 后，L0C
归还在 FIX 后。当前 A/B 的 L1 是不同 root、同一 event bank，统一分配不能拷贝
各组件内部 ID0。MAD 的 clear 与累加模式分别产生 W/RW，不改变累计顺序。

## 关键最小反例

| 被删改的约束 | 错误 | 回归 |
|---|---|---|
| 删除 Input release | Load(g+2) 与 Scale(g) 读取同一 slot 并发 | `test_delete_war_release_rejected_by_two_independent_checkers` |
| 删除 Store release | 新 Vector 写覆盖尚在被 MTE3 读取的 Output | `test_ssa_last_use_does_not_release_async_store_input` |
| source Notify 替代 V barrier | Adds 在 Muls 结果写完前开始 | `test_source_notifications_do_not_fence_later_vector_accesses` |
| ready ring 从 2 缩为 1 | 第二个 Set 在旧 Wait 消费前生效 | `test_binary_event_ring_reduction_needs_reverse_completion` |
| 两个 Cube ready channel 共用 ID | 一方通知可被另一方 Wait 消费 | `test_independent_components_cannot_copy_event_numbers` |
| 部分写后全范围读 | padding 区没有真实 writer | `test_uninitialized_partial_write_is_rejected` |
| 给 view 另起名字 | 不能消除共同 root 的字节冲突 | `test_partial_view_and_recast_preserve_storage_provenance` |

静态 1 AIC/2 AIV 的 GM handoff 已实现：真实 shared SSA allocation、共同语义
epoch、两份 AIV GM read retirement、mode-2 counter 和尾部 credit drain 进入
最终全组图。910B 跨核 Wait 是 controller 全核 dispatch gate，即使公开接口
传入 PIPE_MTE2，也不能把它放进 MTE2 queue 或依赖此前 MTE2 local Wait 完成。
具体协议与反例见 [mixed-ownership.md](mixed-ownership.md)。

动态激活仍需要显式 task 时钟；不能把本地二值事件或反向通知 ack 直接推广到
动态 ownership、不同 participants 或跨组通信。
后续原生 recipe 必须先物化搬运、布局和 scratch，再进入同一效果/同步/验证机制。

静态 GDN 将 FP32 carry 与 snapshot 环分开：carry 的最后 reader 是 Vector
bitwise snapshot-copy，snapshot 槽的最后 reader 是 MTE3 output DMA。
下一轮 carry 写不等待整个 snapshot 输出；只有 snapshot 槽复用才需 MTE3→V
release。最小反例、同步前后 IR、独立交错检查与设备结果见
[recurrent-gdn.md](recurrent-gdn.md)。
