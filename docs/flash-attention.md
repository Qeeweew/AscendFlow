# Flash Attention：三环自动同步与大规模设备验证

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

本页记录 **legacy物理构造路径** 的功能与实验，不是新AST DSL的支持列表。
截至 `b98d541`，真实FA尚未完成结构化前端迁移；新的语言/连续流水/性能验收
以[总体设计](design.md)和[重构方案](refactor-plan.md)为准。下文结果保留其原有
源码、数值配置和测量范围，不作为尚未实现的新路径证据。

## 2026-10-15：CV等待与max归约的设备改进

本轮只改examples中的`tiled_legacy.py`排步，并新增通用原生`reduce_max8`访问
合同；完整AST FA仍未接通。显式选项为`vector_update="tile", max_reduce="block"`，
原默认政策保留用于对照。`benchmarks.attention.suite --policy block-max`组合这两个
选项与跨query overlap；测试同时启用Load3D，所有版本使用相同CANN编译选项。

源码对照依据：ops-transformer HEAD `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
arch22 `flash_attention_score_s1s2_bn2gs1.h`第941/947行的完整Vec1→旧PV等待，
第1058/1062行的Vec1八行、D128 Vec2六十四行分块。安装CANN9.1的
`softmax_flashv2_block_reduce_impl.h:27`使用分层BlockReduce max；安装头文件hash
与参考源码hash记录在外层`runs/attention-cv/comparison.json`，不假称安装ACLNN
由这个参考checkout构建。

ASC的关键改变如下，R=每AIV的八个softmax行块：

```text
旧：step16 softmax(tile1,row0) → 等PV(tile0) → update(tile0,row0)
新：step23 softmax(tile1,row7) → 等PV(tile0) → update(tile0,全部64行)
```

新Vec2只在`g % R == R-1`激活，仍位于stage `2*R`；该时刻下一Vec1的最后一组
已先行派发。score输入仍为两个32KiB槽，scale/softmax在槽内原地计算，取消独立
score_work，为独立整块PV输入腾出空间。槽的release从scale最后读移至partial_sum
最后读，删除这个新WAR协议的回归会被最终检查拒绝。跨阶段直接复用score槽给PV
曾产生等待环，已被拒绝，没有删除验证或添加全局barrier来绕过。

max将连续64个值的归约换为两级八值BlockReduce，使用显式4KiB scratch；所得
每64列partial仍交给原merge_max。有限scaled scores合同保持；sum仍为原来的
64列partial及合并，FP32在线更新、PV累加和P/output舍入均未改。MHA每AIV UB
为191040字节，小于192KiB；没有增加GM搬运，AIC排步和L0C分配本轮未改。

实机910B3，20 AIC/40 AIV，物理1→逻辑0。每进程预检全部八卡，warmup5/repeats30，
相同输入与allocation重复使用，表中时间均为msprof设备kernel时间，不含host间隙：

| Shape / dtype | 最终生成代码 μs | ACLNN μs | 时间比 |
|---|---:|---:|---:|
| B1 H8/Hkv8 Sq=Sk4096 D128 f16，三批中位数 | 698.726 | 512.678 | 1.3629 |
| B1 H8/Hkv2 Sq=Sk8192 D64 bf16，单批 | 2262.544 | 1687.266 | 1.3410 |

4K三批为698.692/698.726/699.624；ACLNN为515.532/512.678/511.982。
**仍未达到无明显性能差距的目标**，1.05验收明确失败。单批消融：同版本旧排步
1310.356μs，仅整块Vec2与UB复用906.048μs，再加block max698.692μs。
独立手写AscendC为1188.334μs，但它采用20组和不同Update/Softmax排步，只作为
独立参考，不用于归因某一项生成器变化。

同批Vector活动时间由850.212→831.784→622.480μs，ACLNN325.869μs；活动区间
可以重叠，不能相加为总耗时或作为纯ALU服务时间。剩余方向是WholeReduce sum
的指令/存储组织，以及64组对20 AIC的负载分配。CANN的sum树与当前合同不同，
本轮没有为了追性能直接替换其浮点累加顺序。

4K实际Cube工作68.719GFLOP，条件阵列峰值`20*1.8GHz*8192=294.912TFLOP/s`，
Cube-only下界233.017μs；当前约98.35TFLOP/s，是该条件峰值的33.35%。这不计
Vector、搬运和等待，不能作为整个FA的性能预测。4K GM分配136MiB可装入192MiB
L2；8K为420MiB，超过L2。请求字节率不是实测HBM带宽，未使用未核实的带宽值。

F16小例1/1/1/256/2048/128全输出误差3.0518e-5，block max前后逐字节相同；
同shape uniform输入误差0。BF16小例1/2/1/256/2048/64全输出误差6.1035e-5。
两个大例各比较557056个元素，并检查全部4194304输出finite、所有allocation
guard及Q/K/V不变；大例仍是采样数值验证。FA排步/最后读取/原生footprint回归
6项、超时与性能验收回归4项通过，未重复长全量测试。

整个隔离构建含Python、同步、ASC和CANN：4K block-max31.22s，8K28.72s；
本轮最慢构建31.22s，未触发300s硬停止。Python内部仍分别23.48/21.62s，
同步分析的编译性能问题没有解决。历史AscendC库先因源码hash不匹配被拒绝，
本轮在新目录重建；另一次测量在设备初始化前因占用预检退出，之后全卡检查空闲
才在同卡执行。失败记录保留；没有设备错误、reset或换卡。

最终ASC在`runs/attention-cv/block-large/{mha4k,gqa8k}/flash_attention.asc`；三批
profile、消融、编译和数值记录均在外层`runs/attention-cv/`。复现：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.suite --cases mha4k gqa8k --policy block-max --load3d --output runs/fa-block-max --compile-timeout 300
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run runs/fa-block-max/mha4k --physical 1 --oracle sampled --warmup 5 --repeats 30 --record-output runs/fa-block-max/mha4k/device
```

第二条仅给stream interval；kernel时间需使用`benchmarks/profile.sh`配合msprof，
本轮完整调用参数保留在各profile目录的日志中。

## 此前ASC审查与未通过的性能验收

当前审查对象为`tiled_legacy.py`的BQ128/BK1024/CK128/VR8、每组四个query
tile、跨query overlap和Load3D候选。它仍走物理构造路径，完整新AST FA未完成。
B1/H8/Hkv8/Sq=Sk4096/D128/f16三批设备kernel中位时间为
1316.708/1317.258/1318.048μs；ACLNN为512.820/513.220/514.190μs。
中位数比2.56665，**性能验收失败**。大例为采样数值比对加全输出finite、输入
与allocation guard检查，不能写成全输出数值比对。对照文件在
`runs/attention-next/mha-comparison.json`，原始profile在`mha-profiles/`。

直接读`runs/attention-next/mha4k-load3d/flash_attention.asc`，发现：

- **CV派发粒度不等价**：AIV在step16完成下一score tile的第一组8行softmax
  后，就在第297行等待上一tile的完整PV；下一组softmax要到step17才派发。
  AIC第一次完整PV到自身step24才发布。910B的`asc_sync_block_wait(pipe,id)`
  实际展开为`wait_flag_dev(id)`，是controller gate，不能理解为只阻塞MTE2。
  于是旧PV等待可以挡住下一tile剩余56行的softmax派发。
- **参照实现的等待位置不同**：ops-transformer arch22的`Process`先执行完整
  `ProcessVec1`，随后`WaitBmm2Result`、`IterateBmm2`、`ProcessVec2`。
  `ProcessVec1`内部才是各8行子块的循环；把Vec1/Vec2逐8行交错不等价。
  同文件1058行在S2=1024时也选8行，所以不能把8行本身判定为性能原因。
- **L0C限制重叠**：ASC将qk_accum固定在0、pv_accum在65536；每次QK复用
  同一个64KiB区域，均受上一次FIX完成约束。QK没有L0C双缓冲。增加第二个QK
  槽会与现有PV同时超过128KiB容量，必须先按整块阶段重新安排复用。
- **分组与硬件不匹配**：当前launch为64组，设备20 AIC；ACLNN实际launch为
  20组。各组工作量相等、按波次执行的简化模型下，64组需四波，末波仅4组，
  有效槽位占用64/80。它是静态排步问题，不是已测出的独立耗时贡献。
- **event选择源码膨胀**：有限分配表有分段周期，但旧emitter只认整表周期，
  导致2044个三元选择。已修为精确分段编码，源码92176→49032字节，全部显式
  ID逐项一致，未改变等待。详见[event分配](event-allocation.md)。该修复尚无
  新设备性能数据，不能用旧profile代表新二进制。

参照HEAD为`4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，文件为
`attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h`。
该checkout用于源码对照，安装的ACLNN二进制不是本轮从它重建的。
原ASC的Python规划/最终检查分别25.90/4.60s；源码选择式压缩没有修复这些成本。
下一项应是完整Vec1/PV/Vec2的阶段派发与L0C复用，不再仅调整Load指令或tile参数。

`flash_attention` 将前向 online attention 展开为真实 xDSL physical IR：
一个 AIC 执行 QK/PV，两个 AIV 各处理半个 query tile，三个有界 GM 环分别
交接 scores、operand-dtype P 和 FP32 PV。Q/K/V/O 为连续 BNSD；K/V 保持原布局，
NN 的 V 在原生 L1→L0B Load2D 中转置，没有全形状 GM 转置或额外 kernel。
算法入口不写 Set/Wait，不提供同步证书；与 Vector、GEMM、GDN 共用 AutoSync 和最终 verifier。

## 实际接口和数值契约

```python
from examples.flash_attention.legacy import flash_attention  # PYTHONPATH=compiler
from xdsl_npu import autosync, verify_plan, emit_asc

source = flash_attention(
    1,
    2,
    1,
    128,
    384,
    128,
    dtype="f16",
    block_q=128,
    block_kv=128,
    q_cache="l1",
    reuse_ub=True,
    pipeline=True,
    packed_reductions=True,
)
plan = autosync(source)
verify_plan(plan)
asc = emit_asc(plan)
```

| 变体/政策 | 已实现的范围 |
|---|---|
| Dense MHA/GQA/MQA | 正整数 B/Hq/Hkv，Hq 为 Hkv 的倍数；D64/128；f16/bf16 |
| Prefill | BQ32/64/128、BK64/128；Sq、Sk 整除各自 tile；每组一个 query tile |
| Causal/offset | `j <= query_start+i`；非负静态 offset，`query_start+Sq<=Sk` |
| Packed decode | Sq1，Hq/Hkv 是 BQ32/64 的倍数；相邻 heads 作为 Cube rows，保持同一绝对 query position |
| GM/local slots | GM1..3、local1..2；容量、固定 issue 合法性和 event 资源仍需最终验证 |
| Q cache | `l0`：Q operand 驻留；`l1`：只从 GM 导入一次，Q/P 共用 L0A 环 |
| UB reuse | score/PV 导入共用真实 root；最终 output cast 复用 P 输出 root |
| Lookahead | `pipeline=True` 显式选择 QK[g+1] 先于 PV[g]、softmax[g+1] 先于 O[g]；需要 local 双槽 |
| Packed reductions | 多行 repeat 归约到紧凑 planes，再独立合并；保持 BK128 的两 partial 边界 |

默认 BQ64；Sq1 且 GQA ratio32 时默认 BQ32。默认保留原始简单政策，性能示例明确
启用四项政策。AutoSync 不改变 tile、issue order、stage、算法或 slot 数。
BQ128/D128、Q/P 双槽在独立 L0A 分配时超容量；`q_cache=l1` 的显式复用政策
允许它在64 KiB内运行，不能偷偷削减为单槽。

输入在 operand rounding 后以及 QK 的 FP32 scores 必须有限；初始 maximum 和
mask 的负无穷属于算法。每个**逻辑 BK** 的数值步骤为：

```text
scores = f32(Q @ K.T) * scale_f32
m_new  = max(m, reduce_max(scores))
alpha  = exp(f32(m - m_new))
P_f32  = exp(f32(scores - m_new))
l_new  = f32(f32(alpha * l) + sum(P_f32))
O_new  = f32(f32(alpha * O) + RN_operand(P_f32) @ V)
output = RN_operand(f32(O * f32(1/l)))
```

FP32 denominator 读未舍入 P，PV 读 RN f16/bf16 P；rescale/add 是独立指令。
BK64/BK128 是不同的舍入配置，不把更改 BK 当作保持位级语义的物理优化。
多行归约减少 Scalar/API 发射数量，不改变每64列 partial 再合并的结构。
CPU oracle 使用独立 NumPy online recurrence；native exp、归约和 BLAS 累加存在
FP32误差差异，容差明确记录，不能声称逐位一致。

## 自动同步前后、最后读取和复用

下面是循环 IR 的简化投影；完整可解析语法和逐字节 witness 由 CLI 的
`before.xdsl/after.xdsl/witnesses.json` 输出。

```text
before:                         # 不含同步；逻辑 phase 不等于物理 stage
  phase0 AIC: QK -> FIX(scores[g % B])
  phase1 AIV0/1: MTE2(scores的各半 -> UB); softmax; MTE3(P的各半)
  phase2 AIC: MTE2(P -> L1); MTE1(P,V); PV -> FIX(pv[g % B])
  phase3 AIV0/1: MTE2(pv的各半 -> UB); f32 rescale/add O

after:                          # 该实例最终只需要三条 ready
  scores_ready       id4 : AIC FIX arrive -> 两 AIV controller wait -> import
  probabilities_ready id0: 两 AIV MTE3 arrive -> AIC quorum wait -> load P
  pv_ready           id2 : AIC FIX arrive -> 两 AIV controller wait -> import
  每个 actor 内仍有独立 local Set/Wait 和必要同流水 barrier
  每个 actor 最后 drain 自己已发射的流水；没有循环内 PIPE_ALL
```

分配相同物理 root 的两 AIV 写入区间必须互不重叠，AIC reader 必须覆盖真正的
P footprint。不能用两个名字或逻辑 SSA use-def 推断这些性质。

同步器先按各环 RAW/WAR 提案 ready/credit。最终从全部原生访问重新检查，允许
删除**整条**冗余 channel，含其 Set/Wait/drain。FA 中已有往返 ready 能覆盖旧
reader retirement：发布 P 证明相应 score 已导入并计算；发布 PV 证明旧 P 的
AIC 导入/计算完成；下一轮 controller 路径约束对应 ring 的覆写。具体是否足够
由有限所有 generation 的全组图判定，而不是硬编码“FA 不需要 credit”。

| 最小反例/回归 | 检查的实际问题 |
|---|---|
| 删除三个 ready 中任意一个 | GM RAW/可见性缺失，独立交错检查产生反例 |
| 两代次 GM单槽/local单槽 | scores、P、PV 都真正覆盖旧槽；检查全部 WAR |
| 将 PV 固定 stage 延迟到4 | 往返 dispatch gate 形成环；明确拒绝该固定排步 |
| 将 P import 的 logical phase 改成0 | 读尚未定义的 probabilities；拒绝而不加 barrier 掩盖 |
| score/PV 共 UB、Q/P 共 L0A | 最后读与覆写跨 generation 分析，局部 release 不等于整个输出完成 |
| 改 causal case workspace offset | 检测跨 case GM 写读重叠，分支内 verifier 通过不能代替跨分支 ownership |

两个 AIV 的 MTE2 最后读退休仍早于后面的 Vector/O store；独立的单环示例保留
这个早 release。FA 冗余 channel 的删除可能借用更晚的既有 ready，既不新增更宽
等待，也不把所有读取里程碑合并为“整个计算完成”。

Causal 按每个 query tile 的真实 KV 前缀形成精确静态 cases，完全未来的 tile
不发射。AIC 和两个 AIV 使用同一 group selector；每个 case 有完整局部协议和
drain。较大 dense grid 也可分区，每 case≤1024组，整个静态 launch≤65535组、
cases≤256；不是 persistent 或数据相关动态任务。跨 case 验证保持精确区间的 holes，
逐核检查平移，不把 bounding box 当作访问。

有界交错：一代 shared 样本已穷尽；两代单槽 shared 和一代完整 FA 在20万状态
预算内未穷尽，返回 `exhaustive=false`。不能以已有终态或 verifier 通过冒充任意
规模证明。多环 shared 抽象保留 local events/Scalar gates；机器语义仍依赖受审查的
dav-2201 契约，必须与设备证据结合。

## 已验证的正确性和性能

2026-10-14，CANN9.1/xDSL0.73，物理910B3卡1显式映射逻辑0。每进程 ACL 初始化
前检查所有八卡的健康、进程、HBM、利用率；曾因占用预检终止，没有设备 reset、
换卡绕过错误或设备执行错误。离线编译为
`-O2 -fPIC -std=c++20 --npu-arch=dav-2201 -mllvm -cce-aicore-dcpreload-args=false`。

| 完整数值验证，shape为 B/Hq/Hkv/Sq/Sk/D | 配置 | 最大绝对误差 |
|---|---|---:|
| 1/2/1/64/256/64 | f16 GQA，BQ64/BK128 | 6.1035e-5 |
| 1/4/1/64/384/128 | bf16 MQA，sharp 输入 | 9.7656e-4 |
| 1/1/1/32/192/64 | bf16，单 GM/local 槽 | 0 |
| 1/2/1/128/384/128 | f16 BQ128，四项优化政策 | 3.0518e-5 |
| 1/2/1/128/192/64 | f16 causal offset17，三 cases | 6.1035e-5 |
| 2/64/2/1/256/64 | f16 packed decode，causal offset17 | 7.6294e-6 |

大规模每个 head、每个 query tile 的两个 AIV owner 和末行采样；新版两形状均
数值比较98304元素，全部4194304输出 finite。4K/8K 最大绝对误差分别
1.5259e-5/6.1035e-5。这是采样数值验证，不是全部输出的数值比对。

各一次 msprof，5次 warmup+30次连续调用，设备 task 时间，不含 host/stream空隙。
旧/新/CANN 是独立 profiling trial，同卡、同 seed 和输入参数 hashes；不是30次
独立实验。CANN 直接调用安装的 `aclnnFlashAttentionScore`，dense BNSD、keepProb1、
innerPrecise0、sparseMode0；源码分析依据参考 arch22，但未重建那个 checkout，
不能把安装二进制宣称为指定参考提交的构建。

| 大规模 shape | 初版 µs | 新版 µs (min/max) | CANN µs | 新版相对初版 |
|---|---:|---:|---:|---:|
| B1 H8/Hkv8 Sq=Sk4096 D128 f16 | 2517.620 | 1034.870 (1016.440/1047.380) | 513.828 | 2.43倍 |
| B1 H8/Hkv2 Sq=Sk8192 D64 bf16 | 8989.880 | 3591.080 (3566.872/3669.952) | 1686.072 | 2.50倍 |

初版 BQ64/BK128；新版 BQ128/BK128、Q驻留L1、UB别名复用、lookahead、packed
reductions，双方 GM/local 双槽不变。4K 新版归约 intrinsic 调用65536，初版逐行
归约2097152；概率舍入边界不变。4K K/V GM请求量减半，Q 每个 query group 仅
导入一次；内部 Q operand 从 L1 重载的代价已计入 IR 工作量。没有达到专家性能
目标：新版仍约为 CANN 时间的2.01/2.13倍，且执行组织与 CANN20核内部排步不同。

## 真实硬件与条件吞吐分析

`attention_performance.py` 绑定当前生成 ASC、IR、库、device记录和 profiler CSV
hash，重新提取全部真实 activation 工作量；保存原始硬件 metadata、DCMI 查询及
profiler `FreqParse` 频率样本。910B3 配置20 AIC/40 AIV、HBM64 GiB、L2 192 MiB、
UB192 KiB/L1512 KiB/L0A&B64 KiB/L0C128 KiB。最大/配置 Cube1800 MHz，两个
profiling 的频率样本为1800；空闲快照800 MHz不是运行频率。Vector频率查询失败
返回-8255，明确记录未知。HBM1600 MHz不是 GB/s；未核实的平台 rates 单位保持 raw。

条件 Cube 模型假设每核每周期一个16³ b16 FMA fractal、FMA计2 ops：
`20 * 1.8 GHz * 8192 = 294.912 TFLOP/s`。这是理论阵列假设，未测得持续峰值。

| 最终 IR / 硬件对照 | 4K f16 D128 | 8K bf16 D64 |
|---|---:|---:|
| 实际/有用 Cube GFLOPs | 68.7195 / 68.7195 | 137.4390 / 137.4390 |
| 条件 Cube-only 下界 µs | 233.017 | 466.034 |
| 生成实际 TFLOP/s / 条件峰值比例 | 66.404 / 22.52% | 38.272 / 12.98% |
| 原生 GM 请求总量 | 3.016 GiB | 9.016 GiB |
| 全 GM 分配 / L2 容量 | 112/192 MiB | 148/192 MiB |
| AIC Cube/MTE1/MTE2/FIX activity | .264/.221/.3225/.411 | .165/.1525/.272/.364 |
| AIV Vector/Scalar activity | .786/.784 | .8255/.9375 |

8K 的 scores 元素数是4K的4倍，Cube FLOPs仅2倍，时间增加约3.47倍；结合
Vector高活动、Cube低活动，支持优先检查 softmax、归约、发射和阶段交接成本。
Scalar活动包含控制与等待，不单独证明 Scalar算术是瓶颈；活动比例彼此可重叠，
不能相加当耗时占比。CANN4K Cube/FIX activity .537/.829，对照有更持续的
计算/结果交接，但这些计数也不证明最优排步。

GM请求可命中L2，全部分配小于名义L2容量不保证全部驻留。3.129/2.696 TB/s是
生成的**GM请求速率**，不是HBM带宽实测。没有把 warmed unique I/O当HBM下界。
脚本仅在明确提供 `--hbm-gbps/--l2-gbps` 时计算相应条件带宽情景，不虚构内存峰值。

原始产物在外层 `runs/attention`，不纳入Git。新版源码 SHA256：

```text
4K e8593e3821b56433c6676e7d2bb5a8c8d1b74b0d1a87b356eddaaa53967c3ede
8K 33b85791d3c540900f4cd0cc3b5c94e1d2855fb562b0fc2948127b4fc25500d9
```

## 复现与未完成范围

在工作区执行；先 IR 回归，再离线编译，再选健康空闲卡。下面固定物理1/逻辑0：

```bash
.venv/bin/pip install --no-build-isolation -e 'compiler[validation]'
.venv/bin/pytest -q compiler/tests/test_attention.py compiler/tests/test_performance.py
PYTHONPATH=compiler .venv/bin/python -m examples.flash_attention.legacy --batch 1 --query-heads 2 --kv-heads 1 --query-length 128 --kv-length 192 --head-dim 64 --block-q 32 --block-kv 64 --causal --query-start 17 --shared --output runs/attention/causal-example
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.attention.run runs/attention/causal-example --physical 1
PYTHONPATH=compiler .venv/bin/python -m examples.flash_attention.legacy --batch 1 --query-heads 8 --kv-heads 8 --query-length 4096 --kv-length 4096 --head-dim 128 --block-q 128 --q-cache l1 --reuse-ub --pipeline --packed-reductions --shared --output runs/attention/mha-4k-reproduce

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof \
  --output=/workspace/xdsl-910b/runs/attention/profile-reproduce \
  --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.attention.run /workspace/xdsl-910b/runs/attention/mha-4k-reproduce --physical 1 --oracle sampled --repeats 30 --record-output /workspace/xdsl-910b/runs/attention/profile-reproduce/device' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/attention/profile-reproduce
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.performance runs/attention/mha-4k-reproduce runs/attention/profile-reproduce --physical 1
```

CANN对照另选输出目录，在相同 application 中加 `--backend aclnn`；8K GQA改为
`--query-heads 8 --kv-heads 2 --query-length 8192 --kv-length 8192 --head-dim 64 --dtype bf16`。

目前明确拒绝一般 Q/KV tails、低GQA ratio的 padded decode、动态任务/ownership、
varlen/paged/sparse/persistent、dropout/bias/任意mask，以及容量或事件预算溢出。
最多每actor4096个实际task instance；合法shape也可能因政策资源/有限预算被拒绝。
下一步需将逻辑BK和物理Cube staging独立建模，以保持舍入点并扩大CV通信粒度；
动态或分页扩展必须增加实际metadata generation、间接footprint及ownership契约，
再进入同一依赖/同步验证。当前没有空接口冒充这些能力，也没有复现专家全部排步。
