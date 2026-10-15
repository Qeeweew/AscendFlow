# 910B 静态混合核 GM ownership

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

本阶段实现真实 `__mix__(1,2)`：每组一个 AIC 和两个参与的 AIV，AIC 的 Fixpipe
把结果发布到显式有界 GM scratch，两个 AIV 各导入半个 tile，然后继续 Vector
后处理及输出。普通 `autosync`、`verify_plan`、`emit_asc` 入口接受混合核 IR。
源码不依赖参考仓库，也不使用轮询、内部 `PIPE_ALL` 或额外 kernel 修复交接。

`gemm_bias_silu(M,N,K)` 实现 f16 `A@Bt.T`、f32 列 bias、f32 SiLU。
K 累加保持原先 GEMM 顺序，bias、negate、exp、加1、div 各有独立 f32 边界。
`tile_m/tile_n/l1_k/l0_k/l1_slots/l0_slots/grid_m/c_slots/cache_b` 沿用 GEMM 政策；
`gm_slots/ub_slots` 独立指定 GM 环及各 AIV 的私有 UB 环，不会自动扩容。
`mixed_pipeline(count,slots)` 是可运行的最小 Cube→Fixpipe→GM→Vector→GM 示例。

## 逻辑 epoch 与物理派发

混合 op 外部 BufferOp 是共享 GM 的真实 SSA 分配身份，actor 内视图捕获它，
并非用两个名字猜测不别名。外部参数基址需要显式 disjoint 输入契约；低层
`compose` 默认拒绝未声明的契约。产品 factory 声明该契约，runner 以不同 ACL
分配满足它。共享 root 内的别名继续从精确 byte unions 分析。

各 actor 在共同 epoch 内表达逻辑计算/effects 顺序；AIC 的 clock divisor
为 `P=K/l0_k`，AIV divisor 为1。AIC 的输出任务位于 `g%P=P-1`，对应 AIV 的
输出 epoch `g/P`。epoch 不是物理完成 token，stage 和 issue rank 控制实际派发。
上个输出的 Vector/store 在下一次阻塞 ready Wait 前派发，保持已发射工作可继续。

所有写入共享 GM 的视图必须具有相同 group 平移，且不同 group 的 GM 写入与
读写不重叠；否则拒绝，不能把 group0 的内部 alias 证明偷用于其他 group。
raw ASC `asc_get_block_idx()` 已是 mixed group index；AIV 用
`asc_get_sub_block_id()==0/1` 选择两个私有执行域，不能把 raw group index 再除2。

## 自动同步前后与最小反例

三代次、GM 双槽的主要 IR，省略各 actor 内已经单独建模的本地流水：

```text
before:
  shared scratch : GM, uninitialized, slots=2
  AIC  e: FIX(C[e], scratch[e%2])
  AIV0 e: MTE2(scratch[e%2].first_half -> private_UB[e%2]); V; MTE3(Y)
  AIV1 e: MTE2(scratch[e%2].second_half -> private_UB[e%2]); V; MTE3(Y)

after:
  cross_channel gm_ready:  sources=[AIC], destinations=[AIV0,AIV1], id=0, count=3
  cross_channel gm_credit: sources=[AIV0,AIV1], destinations=[AIC], id=1, count=3
  AIC e0/e1: FIX; arrive FIX ready[e]               # 初始 free slots，无 seed
  AIC e2:    wait credit[0]; FIX; arrive FIX ready[2]
  AIC tail:  wait credit[1]; wait credit[2]
  AIV0/1 e:  wait ready[e]; MTE2 import; arrive MTE2 credit[e]; V; MTE3 store
```

实际 ASC 的跨核指令是 `asc_sync_block_arrive(PIPE_FIX,0)`、
`asc_sync_block_wait(PIPE_MTE2,0)`、`asc_sync_block_arrive(PIPE_MTE2,1)`，
AIC 在实际 GM overwrite 前等待1，尾部用 Scalar Wait 消费剩余返回。
最后 `PIPE_ALL` 仅在每个 actor 的 kernel 生命周期边界，不能覆盖内部 hazard。

ready 的 FIX prefix 完成保护 GM RAW/可见性；每个 AIV 的 credit 在 MTE2 prefix
读退休后生效，两个 AIV 的 quorum 才允许同一 GM 槽下一次覆盖。credit 不依赖
Vector、激活函数或最终输出 store 的完成。source arrive 本身不 fence 后续源任务。
同一 read-retirement 路径已经覆盖的本地 FIX WAW barrier 会在全组重检后删除。

910B cross Wait 是 controller dispatch gate，公开接口的 pipe 参数被忽略。
它不属于 MTE2 queue，也不能被此前尚未完成的 MTE2 local Wait 绑住。UB 单槽
反例中，MTE2 的 local release 要等下一物理 step 的旧 store，而 controller
可先消费下个 ready，再派发那个 store；错把 cross Wait 放入 MTE2 会造出死锁。

| 反例 | 最终检查/独立交错检查 |
|---|---|
| 删除整个 ready channel | 缺 GM RAW；AIV 可在 FIX 写完成前开始读取 |
| 删除 credit channel | 缺 GM WAR；AIC e2 可覆盖尚未读退休的 e0 |
| 增加一个更晚的 GM reader，仍在初次 import 后 release | read_retired 路径缺失，旧 release 不足 |
| 只让 AIV0 归还，漏 AIV1 | 参与者/quorum 契约拒绝 |
| 省略尾部 drain | 返回 epoch 没有消费，finite boundary 拒绝 |
| flag_id=16 或两个 channel 共 ID | 不允许截断和错误计数器命名空间 |
| 将 drain 放入 stream，或在零代次内声明非空 cross_sync | 输入结构/激活检查拒绝；条件投影不能删掉未建模的 cross effects |
| GM16槽允许连续16次未消费发布 | 无法证明硬件15次上限，拒绝；不通过改 slot 或插全局 barrier 修复 |

`--output` 的 `before.xdsl/after.xdsl/witnesses.json` 保存完整 IR、真实命令序列
和各段跨核 RAW/WAR 的源、目的、字节范围、milestone，便于直接审查。

## 验证层次与明确边界

局部分析的 `_imports` 只是有条件前提，scratch 的 `initialized` 始终为0。
最终 verifier 从共享 writer 和真实 effects 重算完整物理依赖；既不信任 reason
字符串，也不把局部 import 前提作为已经完成的全组证明。局部二值 events 仍
分别占每 actor 的八 ID 有向 bank。跨核是另外的计数资源；单环示例独占 ID0/1，
多环最多使用八个独立计数 ID，当前不共享跨核计数器；每一
通知 ordinal 必须与每个真实参与者的一次消费相配，最大未消费 backlog ≤15。

独立有限交错检查区分 native start、retirement、延迟到达、二值 flag、FIFO
cross counters、双 AIV quorum 和 controller gates；不读 planner frontier/graph。
`scope=all` 的一代次最小流水穷尽1104状态；一代次 GEMM+bias+SiLU 穷尽5212状态。
两代次完整 kernel 在20万状态预算内尚未穷尽，明确返回 `exhaustive=false`。

显式 `scope=shared` 删除私有工作，保留全部共享访问、真实源
prefix retirement 和全核 cross gates；单环删除 local flags，多环保留 local flags
和 Scalar gates，避免消掉证明环复用必需的往返路径。它不证明私有
bank 或数值计算。三代次双槽穷尽1923状态，17代次双槽穷尽14999状态。
该抽象报告的反例可能来自删除私有同步，不自动等同于具体硬件反例；无反例的
穷尽结果才构成对应共享协议检查。完整全组图仍重检所有真实私有/共享访问。
这些结果针对实际有限样本，不能冒充任意规模的形式化证明，也不能取代机器契约。

当前接受1..4个独立 GM handoff ring，每个环固定为 FIX→两个 AIV MTE2 reader，
或两个 AIV MTE3 writer→AIC MTE2 reader。每个真实参与者每 epoch 一个完整
producer/reader、恒定复用距离、静态 balanced 1:2 collective。逻辑阶段独立于
物理 stage/issue order；Flash Attention 的 QK/softmax/PV/O 分成四个逻辑阶段。
多次读取尚需显式物化为一个经过审查的完整 reader recipe，否则 AutoSync 拒绝，
不能提前 release。所有 ready/credit 提案先完整验证；整条 channel 的通知、等待和
drain 删除后，若全组 RAW/WAR/WAW、event 生命周期、计数资源和收尾仍通过，才保留删除。
Flash Attention 的双向往返路径因此通常仅保留三个 ready；不把这个结果硬编码为规则。

静态 cases 按真实 launch group 无洞分区，三个 actor 选择同一 case；每个 case
单独完成并 drain。最终 verifier 重新检查所有 GM bindings、分支和跨 case ownership。
这用于 causal 实际 KV 前缀及较大静态 grid，不提供动态分支或动态 ownership。

动态 task stream、动态别名、不平衡参与者、跨组通信和一般混合尾块尚未实现。
每 actor 最多4096个实际 task instance；有限检查
的成本也随物理范围及代次数增加。更一般的生命周期扩展必须进入同一全组 proof。

## 设备与固定性能证据

2026-10-13，xDSL0.73/CANN9.1/dav-2201，物理910B3卡1映射逻辑0；每个进程
加载 ACL 前记录所有卡的健康/进程/HBM/利用率。没有设备 reset 或换卡重试。

| 实例 | 执行组织 | 最大绝对误差 |
|---|---|---|
| 零代次 mixed_pipeline | 一组1:2，无计数通知，保留实际 FIX 初始化 | 0 |
| 单代次 GEMM+bias+SiLU 16×16×16 | 一组、单 GM/UB/C 槽，短 MAD | 5.960464477539063e-8 |
| mixed_pipeline，17代次/双槽 | 超过15个总通知，但实际 backlog≤2；所有环反复复用 | 0 |
| mixed_pipeline，5代次/双槽，AIV1 延后导入 | 每代32个真实 Vector 更新后，volatile Scalar 读取门控 AIV1；双参与者信用配对 | 0 |
| GEMM+bias+SiLU 80×32×64 | 2组/5输出，L1三槽、L0双槽、C双槽、GM/UB各单槽；共享协议穷尽271状态 | 2.384185791015625e-7 |
| GEMM+bias+SiLU 768×1024×768 | 16组、每组三输出、双 GM/UB/C 槽、L1K256/L0K64、B 驻留 | 4.76837158203125e-7 |

大实例的生成 ASC 是一个 MIX_AIC task（16 AIC，Mix Block Num32）。同数据 ACLNN
基线直接调用 Matmul(f16→f32)、Add(f32列 bias)、SiLU(f32)，profiler 确认每次分别
一个20核 AI_CORE、39核/40核 AI_VECTOR_CORE task，无另列的转置/Cast task。
双方参数字节 hash 完全一致；ACLNN 与生成代码都通过同一独立 f32 oracle。
生成代码有显式2MiB scratch 环，三输出共3MiB结果；ACLNN另有两个全形状中间结果。
比较的执行组织不同，不能冒充复现了某个专家 fused kernel。

各一次 msprof，5次 warmup+30次连续调用，设备任务时间统计：

| | 中位数 µs | min/max µs |
|---|---:|---:|
| 生成融合 kernel | 12.760 | 12.680 / 12.980 |
| ACLNN 三任务合计 | 26.610 | 25.300 / 27.904 |

此固定配置任务时间约2.09倍；不是独立多次实验或所有 shape 的性能结论。
生成代码 AIC Cube/MTE1/MTE2/FIX activity ratio 中位数为
0.531/0.5045/0.5355/0.144，AIV Vector/MTE2/MTE3 为0.275/0.125/0.115。
这些计数支持此配置的流水活动与局部重叠，不单独证明任意阶段的最优排步。
ACL event stream 间隔含宿主派发空隙，不能当作上述内核时间。生成源码 SHA256：
`08fe73665f5a4ad750f9e8f2b87a699447e7129e3bf5c32a524c39db570662b8`。

产物在外层 `runs/mixed/{zero-epoch,one-epoch,17-epochs,768-1024-768}`、
`uneven-participants`、`profile-{generated,reference}`。最早 `first` 记录了96×64×64、三输出、两组的
初始正确性（2.38e-7）；它使用早期逐行输出 DMA，最终后端已改为原生二维 DMA。
所有运行数据、编译产物和 profiler 原始数据都不纳入产品 Git。
独立环数量的追加设备回归位于 `runs/mixed/asymmetric-rings`，源码 SHA256 为
`cb7685f266effc7e7d1a8880ca659e80337cc436c01c3c1f744cde03ef605976`；
该回归验证短 MAD、单 UB 槽和控制器 Wait 的组合，不用于性能结论。

## 复现

在 `/workspace/xdsl-910b`，先纯 IR，再离线编译；卡1需在运行时仍健康空闲：

```bash
.venv/bin/pytest -q compiler/tests
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build mixed --count 17 --slots 2 --check-interleavings shared --output runs/mixed/17-epochs --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build mixed --count 0 --slots 2 --check-interleavings all --output runs/mixed/zero-epoch --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build mixed --count 5 --slots 2 --aiv1-delay 32 --check-interleavings shared --output runs/mixed/uneven-participants --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build gemm-bias-silu --m 16 --n 16 --k 16 --tile-m 16 --tile-n 16 --l1-k 16 --l0-k 16 --gm-slots 1 --ub-slots 1 --check-interleavings all --output runs/mixed/one-epoch --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build gemm-bias-silu --m 768 --n 1024 --k 768 --tile-m 128 --tile-n 128 --l1-k 256 --l0-k 64 --grid-m 2 --c-slots 2 --cache-b --output runs/mixed/768-1024-768 --shared
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build gemm-bias-silu --m 80 --n 32 --k 64 --tile-m 16 --tile-n 16 --l1-k 32 --l0-k 16 --l1-slots 3 --l0-slots 2 --grid-m 1 --c-slots 2 --gm-slots 1 --ub-slots 1 --check-interleavings shared --output runs/mixed/asymmetric-rings --compile --shared
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python -m benchmarks.legacy.run runs/mixed/768-1024-768 --physical 1 --repeats 30
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python -m benchmarks.legacy.run_reference runs/mixed/768-1024-768 --physical 1 --repeats 30
```

profiler 采用两个不同输出目录，避免覆盖原始数据；第二次换
`benchmarks/profile.sh benchmarks.legacy.run_reference` 和 `profile-reference`，其余口径相同：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof \
  --output=/workspace/xdsl-910b/runs/mixed/profile-generated \
  --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.legacy.run /workspace/xdsl-910b/runs/mixed/768-1024-768 --physical 1 --repeats 30 --record-output /workspace/xdsl-910b/runs/mixed/profile-generated/device' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/mixed/profile-generated
```

Bisheng 固定 `-O2 -fPIC -std=c++20 --npu-arch=dav-2201` 与
`-mllvm -cce-aicore-dcpreload-args=false`；实际命令、编译器版本、源码/库/ELF hash
分别记录在输出的 compile manifest。混合设备启动目前通过 CANN 的 Bisheng
共享库 ABI，离线 ELF 生成不意味着其他手工 ELF 启动路径也已实现。
