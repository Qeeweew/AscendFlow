# KDA prefill

实现按计算阶段放在 [KDA 示例目录](../../examples/kda/README.md)，每个模块维护该阶段的
入口和 inline helper。全部 kernel 使用普通 DSL、
xDSL 自动同步和 ASC/CANN；算法与流水政策在 examples，编译器没有 KDA 特判。

## 数值与输入合同

默认合同为 Q/K/V/Output BF16，Gate/Beta/Initial/Final FP32，K=V=128、
H=HV、H 为 8 的正整数倍、BT=64，启用 Q/K L2 归一化，Gate 为非正的 log 衰减值。
state 布局为 `[sequence, head, V, K]`。
每条序列的 chunk 数相同且非零；最后一个 chunk 可以不足 64 tokens，长度可以不同。
`TOKENS: N.i64(minimum=1)` 是运行时值，Starts/Lengths 为不可变 GM i64 元数据。
调用方保证每个 chunk 在对应序列内且 `1 <= valid <= min(64, TOKENS-start)`，
各序列只有最后一个 chunk 可以不满 64 tokens。
当前 runner 拒绝零长度和不同 chunk 数的序列；不由此推导 grouped heads、speculative
state indexing 或其他 dtype 的设备资格。

原始 Triton 的**矩阵乘两侧输入 dtype** 见[输入精度表](kda-prefill-precision.md)。
整块 Gate 前缀、跨块 Gram、块逆合并和历史状态投影使用 IEEE FP32×FP32 Cube；
W/U、状态增量与两个输出乘法使用 BF16×BF16、FP32 累加。
W 先舍入为 BF16，再转为投影所需的 FP32；不能省去中间舍入。
同一 16×16 对角块的 Gram 使用 FP32 元素乘、exp 和归约，不调用 dot。
FP32 carry 与 BF16 snapshot 是不同的值和存储，snapshot 不反馈到递推状态。

## 计算与流水

当前 runner 在同一 stream 上依次发射 `prepare → scan → output` 三个 kernel，
具体 ABI 与顺序见[运行器](../../benchmarks/kda/prefill.py)。kernel 边界建立全设备阶段顺序；
Prepare 内的一个 chunk/head 完整归属一个 mixed group，因此不需要 kernel 内全核同步。
核内 ready/release、GM 交接和有限事件资源由公共同步链建立。

| 阶段 | 计算及归属 |
|---|---|
| prepare | Q/K 归一化、Gate 前缀、对角与跨块 Gram、16→32→64 块逆、RHS/KG/QG/A16、W/U；每个 chunk/head 在同一 AIC 与两个 AIV 内完成 |
| scan | 以 sequence/head 分配任务；chunk 间两阶段预取/递推，历史投影、residual、状态增量和 FP32 carry 更新保持递推顺序 |
| output | 以 chunk/head 并行，双缓冲导入、Cube 计算两项输出乘法、Vector 合并及有效行写回 |

Prepare 在同一任务域中先处理非末块，再处理每序列末块。非末块使用 FP32
下三角矩阵乘计算 Gate 前缀；末块保留 FP32 Vector 顺序累加并清零 padding，
随后用单位矩阵乘发布完整前缀。末块即使有 64 个有效 token，也保持该求和次序。
末块按 16 行预取 Gate 到既有 UB，再逐行累加；不足一片时只搬入有效行，
其余补零，仍执行原有的 64 次顺序 FP32 加法。
Gate 的搬运和发布先于 Q/K/V 搬运与归一化，使 Cube 前缀能与两者重叠。
归一化按每个 AIV 拥有的连续 32 行执行；跨块 Gram 的 Q/K 读取也限制在同一
32 行归属内，因此提前 Gate 不依赖另一个 AIV 尚未完成的归一化。
FP32 carry、BF16 snapshot 和矩阵乘输入精度分别遵守上述数值合同。

所有阶段以显式 slots/at 表达时间缓冲，helper 接收当前输入 Tile 或实际 payload
view。跨块 Gram、块逆和 W/U 的同队列操作数导入使用 CopyGroup 共同 ready，
每份 L1/L0 的 release 仍取自己的最后 reader。融合 Prepare 显式选择原有的
固定 workspace 面，不轮换到额外面；full/tail 继续共享原来的 L1、L0 和私有 UB。
Q/Key 归一化、前缀、Gram 与变换之间的 phase alias 保留实际 root，既不增加
重复的阶段存储，也不扩大各 bank 容量。归一化复用随后构造 Gram 的 FP32
工作面；延迟 W 发布使用另一工作面，不能让仍活跃的 MTE3 reader 阻塞下一任务
的归一化。Gram 的工作面在真实最后 MTE3 reader 退休后才可覆写。

块逆使用 `[[D0,0],[-D1*A10*D0,D1]]` 合并 16→32→64，乘法输入保持 FP32。
16-token 对角块是本示例的分块政策；32 来自两个相邻对角块合并，64 是 chunk 大小。
两个 AIV 各计算两个对角块，沿列打包为本 peer 的 `[16,32]` 块逆；只计算实际拥有的
32 列，保留每列原有 FP32 递推及 16→8→4→2→1 加法树。
跨块部分恰好是以下三个矩形，8/16 行的发布切片分别来自两个 AIV 平分其输出行。

| 合并层级 | chunk 行 | chunk 列 | 每个 AIV 的行数 |
|---|---|---|---:|
| 16→32，前半块 | 16:32 | 0:16 | 8 |
| 16→32，后半块 | 48:64 | 32:48 | 8 |
| 32→64 | 32:64 | 0:32 | 16 |

跨块 Gram 采用 16/32-token 分层矩形；四次 FP32 GEMM 的结果占用同一
`[32,96]` L0C 的不重叠列区间，经一次 FIX 写入 `GramProducts`。
前两个 16 列区间各上下堆叠 K/Q 分数，后两个 32 列区间分别存放大块 K/Q 分数。
该物理打包不引入第四种数学分块。独立 `prefill_gram_cross` 保留各自紧凑的
small/large 结果平面和四次 FIX，Vector 消费紧凑的行分片；融合阶段采用一个
结果平面，以已有的双 AIV 前缀交接获取完整结果。独立阶段保留 Negative→Aqk
发布顺序；融合阶段仍要求 Aqk→Negative，使后续获取涵盖两个 peer 的 Aqk 写入。
对角 Gram 和跨块 Gram 的操作数只依赖同一份已发布的输入与 Gate 前缀，
因此先发布跨块操作数，让 Cube 的跨块乘法与 Vector 的对角计算重叠。
GramLeft 的 0:32 行由 AIV0 发布，32:128 行由 AIV1 发布；GramRight 的
0:16 和 32:64 行由 AIV0 发布，16:32 行由 AIV1 发布。每个 AIV 最后发布自己的
GramLeft。Cube 先整体导入 GramLeft，取得两个真实发布者的完整 MTE3 前缀，
再导入 GramRight 的三片。后续单个 AIV 所写分片的读取消费此前完整获取，
不新增单侧通知。Diagonal、Negative 只发布实际使用的块，均在读取前完整覆写；
Aqk、Ai32 保留整面补零。恒定三角 mask 在 kernel 入口构造一次。

块逆合并先导入最后发布的 32×32 Negative，将其保存在既有 B 侧 L1 的
32:64 行；两次 16×16 合并只使用 0:16 行，因而不会覆盖它。
所有非转置 FP32 B 的物理 leading M 保持 64，避免因临时存放位置改变而反复
切换 fmatrix 配置。该导入仍取得两个 AIV 的完整发布前缀。
中间 GM workspace 按 group 分配，并由实际最后 reader 约束回收。
transform 的两个 AIV 共读末行时，读取已经发布的当前 chunk 的 Gp 输出；
不用下一任务即将复用的 LocalGp 半区承载该跨 peer 读取。末块的末行仍来自
各自 UB 中保留的 FP32 顺序前缀，保持原求和次序。
transform 在已有 UB 中保留当前面板的 FP32 K 与 `exp(Gp)`，分别供两项 K
变换和 QG 使用；每项乘法及 BF16 舍入的顺序保持，避免重复搬入、转换和 exp。
该行变换由同一个 inline helper 表达：Prepare 使用每 AIV 32 行，独立 transform
使用 64 行。独立入口保留按 valid-1 读取的实际末行，不从尾块 padding 推导末值。
Right、KG、QG 的形成提前到块逆获取之前，与 Cube 合并块逆重叠。
A16 仍在取得块逆的完整可见性之后读取 Aqk；每个 AIV 的两个连续 16 行面板
合为一次 32 行导入、BF16 转换和发布，复用既有 32 行缓冲。
两个 AIV 共用 body，通过 runtime SID 区分地址与角色；分析保留各自私有 UB
及完整 quorum 贡献，不把同组不同 AIV 的源码顺序当作完成关系。

跨块 Gram 的行分工与 transform 的连续 32 行分工不同。每个 AIV 先发布 Aqk，
再发布送入 Cube 的 Negative；transform 在获取 Cube 最终块逆结果后读取 Aqk，
保留完整的双 AIV 可见性链。块逆的 FP32 UB 副本跨 transform 保留，之后才舍入
并发布 BF16 Inverse，使 W/U 的首个导入同时获取此前 Right 的发布前缀。

跨块 Gram 的四片左操作数在 L1 中保存为一个 `[128,K]` NZ slab，使用一次真实
GM→L1 DMA 导入，再由四次行切片 Load3D 读取；下一次覆盖等待全部旧 reader 退休。
右操作数保留紧凑转置平面。FP32 Load3D A 与非转置 B 的 strided NZ 源保留
每个 K/8 平面的实际行距，行距按 16×8 个 FP32 对齐且宽度不超过 256；
目标仍是完整紧凑 L0A/B 平面。该能力不放宽 FP32 转置 Load2D 的紧凑源合同。

scan 中每个 chunk 保存更新前的 BF16 snapshot，同时用 FP32 历史状态做投影；
residual 舍入为 BF16 后参与状态增量。carry 在 FP32 中完成衰减与加法，
每个 sequence/head 必须按 chunk 顺序执行。独立的 output 阶段读取各 chunk 的
snapshot、residual 等数据，按 chunk 分配负载，不重新计算递推状态。
W32 的两个 K=64 列片独立搬入双槽 L1，与 KG 共同 ready；每片存储分别在
自己的旧 reader 退休后即可复用，投影保持原有 FP32 分段累加顺序。

output 的两项 FP32 乘积在 L0C 中占据同一 `[BT,2*K]` 平面的两个列区间，
分别保留原有初始化和累加顺序。一次 FIX 将两项结果发布到
`OutputProducts[GROUPS,2,BT,2*K]`，两个 AIV 各连续导入 BT/2 行，再用
strided FP32 `N.add` 写入紧凑求和区并 RN 输出。每个 task 只需一轮完整 CV
交接；尾块按实际有效行裁剪，允许其中一个 AIV 不写输出，但两个 AIV 都完成
workspace 读取和 credit 归还。L0C 仍为 128 KiB，GM workspace 字节数不变，
每个 AIV 的 UB 为 112 KiB。

## 验证边界

运行器用独立 CPU FP32 递推检查整个 pipeline 的 output/state，relative RMSE
门槛均为 `<0.005`，同时记录输入与编译产物身份。同库改变 runtime tokens/尾块
只能使用已编译的 sequence/chunk 域。流水输出通过不等于每个中间矩阵已独立验收，
也不外推其他 dtype 或未列出的元数据合同。
性能使用 `msprof op` 的 kernel Task Duration，保持默认 kernel 重放；
BasicInfo、PipeUtilization 和 L2Cache 分开采集，不能混用各次时间。
AscendFlow 的采集入口在提交每个 phase 后进行 host stream 同步，避免依赖阶段
尚在重放时提交下一阶段；该设置记录为 `phase_host_sync`。耗时只累计设备
kernel，不包含 host 间隔。普通运行仍在同一 stream 连续提交三个 kernel，
数值验收也覆盖不加逐阶段 host 同步的调用。
具体数值与性能结果随每次构建和采集保存，旧版本快照不作为当前代码资格。

## AscendC 基线与复现

主基线为 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)
的 `aclnnChunkKdaFwdV2`。它是 upstream Python `chunk_kda_fwd` 在 BF16、
K=V=128、chunk64 且开启 Q/K 归一化时选择的公开组合入口，按 Prepare、FwdH、
Finalize 发射；与旧版本的单 L0 融合入口需区分。使用自然 log gate、预处理 beta、
FP32 初始/最终状态和 V-first state，不启用额外 gate/beta 激活。
这里的内部 Q/K 归一化开关会强制选择 V2。未启用这些强制开关时，upstream
按 `heads × total_chunks >= 4096` 选择 V2；H32、4096 tokens、chunk64 只有
2048 个 work item。所核对的 vllm-ascend 封装先独立归一化 Q/K，再调用默认
分支，因此该 shape 会选择旧融合入口，并可能包含布局转换。当前 V2 性能
不能代表该 vllm-ascend 完整默认路径；后者尚未测量。
该版本 FwdH 使用 BF16 W 与 BF16 history 做投影，FP32 carry 单独保留；
AscendFlow 仍使用上述 FP32 W×carry 投影。双方共享整算子误差门槛，内部矩阵
输入精度需分别报告，不能用此基线放宽 AscendFlow 的数值合同。

双方以原始 BF16 Q/K/V 为输入，kernel 内归一化计入耗时，完整 output/state
relative RMSE 必须各自小于 0.005。MSTX `kda` 标记覆盖完整调用，统计其中所有
kernel，包括 ACLNN 内部的 transpose、copy 与辅助输出，不按名称删去其成本。

```bash
source ../.venv/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.kda.prefill --build-only \
  --artifacts build/kda/prefill --output build/kda/prefill-device --lengths 4096
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/prefill --output build/kda/prefill-ours
python -m benchmarks.kda.profile_prefill --physical 2 --backend ascendc \
  --reference-artifacts build/kda/reference/opp/vendors/fla_npu_transformer \
  --artifacts build/kda/prefill --output build/kda/prefill-ascendc
python -m benchmarks.kda.compare_prefill \
  --ascendflow build/kda/prefill-ours --ascendc build/kda/prefill-ascendc \
  --output build/kda/prefill-comparison.json
```

完整 pipeline 构建和每次运行分别有单次 300 秒进程组硬上限。运行前检查所有卡的
健康、进程、显存和利用率，每进程显式映射物理卡至逻辑 0；设备错误直接停止。
`--run-lengths` 可复用同一份库改变 tokens 数和尾块，只接受已编译的 sequence/chunk 域。
数值检查先于报告性能；batch=4 可构建 `--lengths 512 512 512 512`，该库可运行
`--run-lengths 449 477 505 512`，输出目录使用新路径。
