# 分页 sparse MLA decode

[gather.py](../../examples/sparse_attention/gather.py) 和
[decode.py](../../examples/sparse_attention/decode.py) 移植 DeepSeek V4.1 推理中的
shared-KV attention 核心：一个请求本次只有一个 query，滑窗 KV 与稀疏压缩 KV
共同归一化，每个 head 有一个 FP32 sink。两份 KV 各自同时充当 K 和 V。
索引器、压缩器、投影和缓存写入由调用方提供。

对应 vllm-ascend 的 `AscendDSAV41Impl._forward_attention` 非 A5 路径：
`torch.ops._C_ascend.npu_sparse_flash_mla`，`layout_q="TND"`、
`layout_kv="PA_BBND"`、`ori_mask_mode=4`、`cmp_mask_mode=3`、
`topk_value_mode=1`。这是该调用的 decode 计算子集，有自己的 workspace 和 launcher，
不提供 vLLM 插件注册或 ACLNN ABI 替换。

## 计算与输入

令请求长度为 L，滑窗实际可见长度为 W，压缩倍率为 R。当前 query 的原始位置
为 L−1。参与计算的行依次来自原始 KV 的 `[L−W,L)` 和稀疏索引前缀中
`0 <= index < floor(L/R)` 的压缩 KV。重复索引保留重复贡献，不做去重。

对选出的各行 x，计算 `score[h,x] = dot(Q[h], KV[x]) / sqrt(D)`：

```text
O[h] = sum_x(exp(score[h,x]) * KV[x])
       / (sum_x(exp(score[h,x])) + exp(Sinks[h]))
```

实际实现减去行最大值。sink 的 value 为零，只贡献一次分母。空集合输出全零。

| 输入 | 存储与合同 |
|---|---|
| Q | `[B,H,D]`，每请求一个 query；与 upstream `[T,H,D]` 在 T=B 时对应 |
| Ori / Cmp | `[physical_blocks,page_size,D]`，BF16 或 FP16；去掉 PA_BBND 的 singleton KV-head 轴，且 K=V |
| OriTable / CmpTable | 连续 INT32 `[B,ceil(MAX_SEQ/PAGE)]`；分别映射逻辑页到各自合法物理页，可共享只读物理页 |
| Lengths | INT32 `[B]`，`0 <= L <= MAX_SEQ`；包含本次 query token |
| WindowLengths | INT32 `[B]`，`0 <= W <= min(WINDOW,L)`；可缩短 replay 的可读后缀 |
| Counts / Indices | INT32 `[B]` / `[B,TOPK]`；`0 <= count <= TOPK`，前 count 项非负且小于 `MAX_SEQ/R`，后缀可填 −1，不读取 |
| Sinks | FP32 `[H]`，每个 head 一个有限值 |

R 支持 1、2；原始页大小 PAGE，压缩页大小 PAGE/R，对应同一数量的原始 token。
标准 decode 的压缩缓存已包含 `floor(L/R)` 个完成项。前缀中的未来项被过滤，
不会读取其页或取得 softmax 权重。上游 `prepare_indexer_indices` 的已整理前缀
可直接使用；Counts 可使用已有 topk lengths，或由设备端统计非负索引生成。
这里不要求索引已排序。

页表、索引、长度在 gather 内不可变，cache 与所有输出/workspace 不别名。
调用方负责输入合同；编译器检查声明和实际访问，launcher 不检查输入值。
数据须有限，且 QK、PV 的 FP32 运算及输出转换不能溢出。允许空请求以 L=W=count=0
表达；B、H、容量及页大小是 constexpr，同一二进制复用不同请求长度和索引。

## 流水与数值

`sparse_mla_decode` 是一个 mixed kernel。每个请求先 gather 所有选中的压缩 KV，
再把 window 与压缩行的 Score 放在同一逻辑矩阵中，只做一次 softmax 和 PV。
它不再分别计算 window/压缩 tile 的 PV 后维护在线 numerator、alpha 和 max carry。
两个 AIV 共用代码，按 runtime SID 处理各自一半 head；每个 head 仍只计一次 sink。

令 `CP=ceil(TOPK/512)*512`、`S=WINDOW+CP`、`W=ceil(S/512)*512`。
压缩 gather 使用双槽 16 行 UB，逐行读取索引和页表；无效槽清零并产生 −∞ Bias，
合法槽 Bias 为 0。重复索引仍逐次参与。窗口继续从分页 GM 直接进入 L1，
没有为合并 softmax 新增窗口 gather。算法不要求索引排序。

QK 先计算压缩行，最后计算 window，将结果分别写入 Score 的 `[WINDOW,S)` 和
`[0,WINDOW)`。window 的最后 FIX 完成前缀经过全部压缩 QK 输入，因此包含双方
AIV 的 gather 发布。softmax **先获取完整 Score，再读取 Bias**。普通 AIV peer GM
访问没有直接交接协议；不能把 Bias DMA 提到 Score acquire 前。

AIV 将 S 列 Score 读入 W 列 UB，额外列填 −∞，仅在 UB 内参与归约；这些列
不增加 KV 读取、MAD 或 GM 的 Score/P 容量。默认每组处理 16 行 softmax。
Bias 直接按行加到 Score，不物化完整 Bias 矩阵。先计算包含 sink 的 FP32 最大值，
再计算概率、FP32 分母和 b16 P。分母保存为每请求的双槽私有 UB 快照。

Cube 的 PV 从 window 开始，随后按原压缩顺序累加全部行，FP32 L0C 只在首次
MAD 清零。D 可被 256 整除时，一次 128×256 L1 搬运供两个相邻 128 列输出，
共用 P 装载，再一次 FIX 写出 256 列。finish 每次处理 8 行，只做 FP32 分母
归一化、转换与最终写出。QK/PV 共用双槽 L1 KV 池、L0A/B 和按输出列切分的 L0C。
L1 的真正最后 reader 是 MTE1，源码不指定事件 ID 或全核 barrier。

每个 worker 持续处理自己拥有的请求，默认至多 20 个 mixed group。发射顺序为
Cube 的 `pv→qk`、Vector 的 `gather→softmax→finish`，阶段偏移分别为 0/1/2。
KV/Bias 使用 `query%3`，Score/P/PV 和分母快照使用 `query%2`，不能共用一个 slot 时钟。

三槽还保护 **两个 AIV 对 Bias 的最后读取**。Cube 先执行 PV(q)，再执行 QK(q+1)；
后者的最终 Score 完成前缀经过读取双方 P(q) 的 PV，因此已经涵盖双方 Bias(q)
读取。每个 AIV 在下一轮 gather 前已获取该 Score。到 q+3 覆盖旧 Bias(q) 时，
这条中转已经发生；两槽在 q+2 覆盖时尚没有这一保证。KV 的 Cube reader 退休
本身不能授权覆盖 Bias。回归用独立有限模型保留提前读取和两槽覆盖的 RAW/WAR
反例，不以普通编译成功代替 peer 访问证明。

调用方分配以下 GM workspace，其中 `G=B*(H/BM)`，实际只使用 resident worker
对应的 group 切片：

| workspace | 形状和 dtype |
| --- | --- |
| KV / Bias | `[G,3,CP,D]` b16 / `[G,3,CP]` FP32 |
| Score / P | `[G,2,BM,S]` FP32 / b16 |
| PV | `[G,2,BM,D]` FP32 |

QK/PV 使用 b16 输入和 FP32 累加。P 在 PV 前显式舍入成输入 dtype；分母使用
未舍入的 FP32 概率，最终输出仍是输入 dtype。统一最大值后舍入 P，与原先
分 tile 舍入再重缩放的实现不位等价。设备 runner 按独立 FP64 公式使用既定
`atol=0.005, rtol=0.02` 检查；这不是任意输入的误差上界或 upstream 位精确资格。
最大值、分母、矩阵累加和最终归一化保持 FP32。Sinks、Q/KV 及实际 FP32 运算仍须满足有限性
与不溢出合同，SCALE 必须为正。

默认验收政策为 B=4、H=64、D=512、WINDOW=128、TOPK=512、S=640、
BM=64、VR=16、BK=512、C=128、PAGE=128、MAX_SEQ=4096。UB 内 W=1024；
更宽的 Score 使用 8 行组和显式 512 列 UB 工作块处理行广播。当前 W 至多 4096，
且所有具体组合仍需通过 192KB UB、L1/L0 和有限事件资源检查；不是所有 D/TOPK
组合均能 resident，不自动切换到另一编译路径。

与官方 CSA 保持相同的选中行、分页和共享 KV 语义，计算组织改为每请求联合
softmax。官方 gather 还在合法时用双行 runtime-stride DMA 并压紧有效行；这里
仍逐行搬运并保留无效槽，不声称逐指令复刻。满有效块仍执行 b16 清零。

算法分块、slot、任务顺序、QK/PV 复用和快照都在 example DSL 中表达。runtime
NZ 前缀覆盖、真实 pitched effects、初始化、ownership 和有限事件分配继续使用
同一编译路径；这项优化没有修改编译器或 event sharing。官方动态 stride、
有效行压紧和 UB metadata Scalar cache 仍未开放，不以扩大硬件容量、全局 barrier
或静默 fallback 绕过拒绝。

## 运行与边界

从仓库根目录执行，先加载 CANN 环境并激活工作区 venv：

```bash
python -m benchmarks.attention.run_sparse --build --output build/sparse-c2
# 检查所有卡后，显式映射所选物理卡；runner 会再次检查健康和进程。
ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 \
  python -m benchmarks.attention.run_sparse --physical 2 --output build/sparse-c2

# C1 特化：构建与运行均传 --ratio 1。
python -m benchmarks.attention.run_sparse --build --ratio 1 --output build/sparse-c1
python -m benchmarks.check_examples --only sparse-mla-decode --output build/sparse-smoke
```

只做 Python/IR 检查时加 `--ir-only`。每个构建共享 300 秒硬期限，失败即停。
运行复用 manifest 中已发布的库，不隐式重新编译。输出保存 policy、源码身份、
DSL/物理 IR、ASC/库 hash、CANN 命令、设备健康、原始输入和逐项误差；输入及
workspace 的尾部哨兵也要保持。默认检查空集合、window-only、混合两类 KV、
compressed-only、重复索引、replay 后缀及极端有限 sink，并复用同一组 workspace。

每次设备资格只属于其记录的 DSL/IR/ASC/库与输入。runner 的 ACL event 包含
host submission gaps；性能对照使用 `benchmarks.attention.profile_sparse` 的 msprof
device task duration。官方参照必须是从 vllm-ascend 原始 host/kernel 构建的 custom OPP，
通过 `--reference-artifacts` 指定；元数据准备在计时外，输入身份逐项匹配。
`--rotations` 分配多组独立缓存和输入，并逐组检查输出；轮换本身不证明冷 HBM，
应结合 `--metrics MemoryAccess` 和 `--metrics L2Cache` 的原始记录分析。

当前不包括 prefill/多 query、LSE、split-KV、多卡、A5 的 FP4/FP8 packed cache，
也不包括 QLI TopK、跨层候选筛选、compressor、RoPE 和模型端到端接入。
INT32 元数据加载与受支配分支范围是通用能力，但不代表任意条件访问的存储分区、
通知配对或初始化合流已经支持；其余形式继续经过原有准入并可能明确拒绝。
