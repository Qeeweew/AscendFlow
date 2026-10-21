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

`sparse_mla_decode` 是一个混合核 kernel：两个 AIV 读取压缩索引和页表，将选中行
经双槽 16 行 UB 流水物化到三槽 GM KV，并发布有效为 0、填充为 −∞ 的 Bias。
同一 head tile 的各个 head 共用这次 gather。滑窗直接从原始分页 KV 搬到 L1。
Cube 执行 QK/PV，两个 AIV 共用代码、按 runtime SID 处理各自一半 head 的
online softmax 和 FP32 输出更新。GM 交接由编译器推导完整双 AIV quorum。

gather、cache 搬运、QK、PV 和 softmax 使用透明 inline helper；调用处以具名实参
列出真实存储、任务坐标和分块参数。helper 不隐藏同步或额外存储，metadata
读取保留在相应计算作用域内，Stage 只表达发射偏移。

示例至多使用 20 个 mixed group，按 owner 轮转处理请求和 head tile。一个 worker
的所有请求共用一条连续 tile 流水，GM/UB 槽按全局 tile 编号轮换，只在整个 worker
结束时收尾。新请求分别在 QK 和 softmax 前重载 Q、sink 和分母；输出累计状态
在该请求首次 update 前清零。softmax 的 FP32 分母按 tile 保存到双槽快照，与
递推 carry 分离，因此旧请求最后一次归一化允许与新请求 softmax 交叠。

Score、P、PV 是每 group 双槽 GM workspace，调用方分别分配
`[B*(H/BM),2,BM,BK]` FP32、同形 b16、`[B*(H/BM),2,BM,D]` FP32。
KV/Bias 分别为 `[B*(H/BM),3,BK,D]` b16 和 `[B*(H/BM),3,BK]` FP32。
QK/PV 共用 KV L1 池、双槽 L0A/B 和一份按输出列分片的 L0C。D 能被 256 整除
时，PV 复用一次 P 装载计算两个相邻 128 列输出，再一次 FIX 写出 256 列；
两个 MAD 使用各自的 L0C 列切片。128×128 分块的 L1 最后读取在 MTE1，双槽
允许其与下一次 MTE2 装载重叠。源码不指定事件或全核 barrier。

行状态 `(max,denominator,numerator)` 以 `(sink,1,0)` 初始化，FP32 递推。
QK/PV 使用 b16 输入与 FP32 累加；P 在 PV 前显式舍入成输入 dtype，分母读取
未舍入的 FP32 概率，最终输出保持输入 dtype。不同合法分块可改变 FP32 求和
次序；设备 runner 使用独立 FP64 公式参考，以 `atol=0.005, rtol=0.02` 检查，
这不是任意输入的误差上界证明或 upstream 位精确资格。

默认验收政策为 B=4、H=64、D=512、WINDOW=128、TOPK=512、S=640、
BM=64、BK=512、C=128、PAGE=128、MAX_SEQ=4096。它是显式示例政策，不由编译器
按模型名称选择。资源超限会被拒绝，不扩大硬件容量。

这是 CSA 计算与流水结构的移植，当前 sparse gather 逐行搬运并保留无效索引的
空槽和 mask；官方实现还使用合法时的双行 DMA 及有效行压紧。因此数值集合相同，
不声称搬运指令与官方逐条一致。
当前压缩 gather 集中在对应 compressed tile。所测官方 arch22 CSA 在 window
任务将 `v0S2DealSize` 设为 0，同样不在该任务执行 compressed gather。
每块 16 行 UB 使用一次 b16 `fill` 完整清零，再由真实有效行搬运覆盖；满有效块
仍执行初始化。它减少对 FP32 零缓冲的读取和转换工作，不减少稀疏 GM 读取量，也不
采用有效行压紧或只清无效行的策略。

算法政策和编译器能力按实际机器工作区分：

| 改动 | 职责及当前边界 |
|---|---|
| tile、worker 映射、slot、预取、QK/PV 复用和递推快照 | 在 example DSL 中声明；实际读写及排步改变后重新推导依赖 |
| 两段 GM 页放入同一 L1 NZ tile | example 选择布局；首段补零、后段 runtime 前缀覆盖经过真实 pitched effects、容量和初始化合同 |
| 官方双行 DMA | 需要 runtime source stride 及两段精确 effects；当前静态行距 view 不能直接表达，尚未支持 |
| 官方有效行压紧 | 政策在 example；底层还需 count 归纳范围、已定义前缀、发布数据/count 对应和退休合同，当前未开放 |
| UB 缓存元数据后 Scalar 读取 | 缓存时机属于 example；当前 index load 只接受只读 GM，不支持把 UB view 直接替换进去 |

稀疏只读输入允许多个 reader 和重复地址。ownership 约束的是哪代物理存储仍被
哪些 reader 使用；覆盖槽位前必须等待所有相关读取退休，不能把两个 AIV 的
私有 UB 或共享 GM 的两份贡献合并。event ID 不由 DSL 指定，资源分配失败不
授权扩大 bank、插入全局 barrier 或自动改变算法排步。

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
