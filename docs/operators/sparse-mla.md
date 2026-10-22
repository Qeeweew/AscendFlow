# 分页 sparse MLA attention

[gather.py](../../examples/sparse_attention/gather.py) 和
[attention.py](../../examples/sparse_attention/attention.py) 移植 DeepSeek V4.1 推理中的
shared-KV attention 核心：一个请求在一次 forward 内可有多个 query，支持因果
prefill、chunked prefill 和推测译码验证阶段。滑窗 KV 与稀疏压缩 KV 对每个 token/head 分别共同归一化，
每个 head 有一个 FP32 sink。两份 KV 各自同时充当 K 和 V。
索引器、压缩器、投影和缓存写入由调用方提供。

对应 vllm-ascend 的 `AscendDSAV41Impl._forward_attention` 非 A5 路径：
`torch.ops._C_ascend.npu_sparse_flash_mla`，`layout_q="TND"`、
`layout_kv="PA_BBND"`、`ori_mask_mode=4`、`cmp_mask_mode=3`、
`topk_value_mode=1`。这是该调用的 C1/C2 计算子集，有自己的 workspace 和 launcher，
不提供 vLLM 插件注册或 ACLNN ABI 替换。

## DeepSeek V4.1 Flash 的 DSpark query 数

[DeepSeek-V4.1-Flash-1017 配置](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/master/config.json)
给出 H=64、KV-head=1、D=512、window=128、index TopK=512，
`dspark_block_size=5`、`num_nextn_predict_layers=3`。
[配套 vllm-ascend 部署示例](https://github.com/vllm-project/vllm-ascend/blob/2581a69bcc16ee6af56e72985a15799bb5f61808/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md)
使用 `{"method":"dspark","num_speculative_tokens":5}`。

该部署的主模型满宽验证包含 **5 个 draft token + 1 个普通 decode token**，
每请求 6 个 query，因此 benchmark 使用 `--query-tokens 6`；B64 对应 T384，
B128 对应 T768。3 是 draft 层数，不是每请求 query 数；draft 在
`sample_from_anchor=True` 时自身使用 5 个 query，与 target 验证的 6 个不同。
动态调度可缩短实际验证长度，调用方应以真实 `cu_seqlens_q` 为准。

这里覆盖主模型的因果 C1/C2 验证 attention；DSpark draft 的非因果 SWA 不在
当前合同内。配置中的 `max_position_embeddings=1048576` 是模型上下文上限，
不是每次 decode 的上下文长度；测量必须另外标明实际上下文。

## 计算与输入

Q/O 使用 `[T,H,D]`，B 是请求数，T 是本次所有请求的 query token 总数。
`cu_seqlens_q[B+1]` 从 0 开始、单调不减且末项为 T；允许空请求。
对请求 b，令 `n=Cu[b+1]-Cu[b]`、`j=token-Cu[b]`，L 为包含本次 token 的
原始缓存长度，C 为压缩缓存完成项数，r 为 residual，R 为压缩倍率：

```text
原始窗口右端（不含） = max(0, L-n+j+1)
压缩索引上界（不含） = max(0, C*R+r-n+j+1) // R
检查的索引 rank 数   = min(TOPK, C, 压缩索引上界)
```

原始窗口取截至本 token 的至多 WINDOW 行；正的 `ori_topk_length[token,0]`
进一步缩短可见后缀，0 表示标准窗口，与官方 band mode 一致。每个 token 有自己的
压缩索引行；rank 范围内的未来压缩项不贡献权重，重复索引保留重复贡献。
标准完整缓存中 `C=L//R`、`r=L%R`，压缩边界等于 `floor((position+1)/R)`。
不能让所有 query 共用请求最终的窗口或压缩边界。

对选出的各行 x，计算 `score[h,x] = dot(Q[h], KV[x]) / sqrt(D)`：

```text
O[h] = sum_x(exp(score[h,x]) * KV[x])
       / (sum_x(exp(score[h,x])) + exp(Sinks[h]))
```

实际实现减去行最大值。sink 的 value 为零，只贡献一次分母。空集合输出全零。

| 输入 | 存储与合同 |
|---|---|
| q / attention_out | `[T,H,D]`，BF16 或 FP16；T 与 B 独立 |
| ori_kv / cmp_kv | `[physical_blocks,page_size,1,D]`；保留 PA_BBND 的 singleton KV-head 轴，且 K=V |
| ori_block_table / cmp_block_table | 连续 INT32 `[B,MAX_SEQ/PAGE]`，映射各自的逻辑页；可共享只读物理页 |
| cu_seqlens_q | INT32 `[B+1]` 的 query token 前缀和 |
| seqused_ori_kv | INT32 `[B]`，`0 <= L <= MAX_SEQ`，包含本次全部 query token |
| seqused_cmp_kv / cmp_residual_kv | INT32 `[B]`，`0 <= C <= MAX_SEQ/R`，`0 <= r < R` |
| ori_topk_length | INT32 `[T,1]`，0 为标准窗口，`1..WINDOW` 为缩短的可见后缀 |
| cmp_sparse_indices | INT32 `[T,1,TOPK]`；非负前缀中的索引 `< MAX_SEQ/R`，其后全部填 −1；无需排序 |
| sinks | FP32 `[H]`，每个 head 一个有限值 |

张量名字与形状对齐官方 `aclnnSparseFlashMla` 的上述参数。布局和 mask 固定为
TND/PA_BBND、ori band4、cmp causal3、logical indices。SCALE/RATIO 等政策在 DSL
中 constexpr 特化；workspace 和 launcher 由 AscendFlow 管理，不冒充 ACLNN ABI。
这里不传自定义 Counts，也不把 910B 官方保留的 `cmp_topk_length` 当作有效项计数。
`ori_topk_length` 是必需的输入；标准窗口调用提供全 0 张量。

R 支持 1、2；原始页大小 PAGE，压缩页大小 PAGE/R。索引张量包含后续 query 可见的
缓存项是合法的；内核逐 token 过滤，不读取不可见项的页表或缓存数据。

页表、索引、长度在 gather 内不可变，cache 与所有输出/workspace 不别名。
调用方负责输入合同；编译器检查声明和实际访问，launcher 不检查输入值。
数据须有限，且所有实际 QK、PV 的 FP32 运算及输出转换不能溢出，包括被 mask 的
已初始化旧行。允许以 L=C=r=0 表达空缓存，或以相邻 Cu 相等表达空请求；
B、T、H、容量及页大小是 constexpr，同一二进制复用不同请求长度和索引。

## 流水与数值

`sparse_mla` 是一个 mixed kernel。每个 query token 先 gather 所有选中的压缩 KV，
再把 window 与压缩行的 Score 放在同一逻辑矩阵中，只做一次 softmax 和 PV。
它不再分别计算 window/压缩 tile 的 PV 后维护在线 numerator、alpha 和 max carry。
两个 AIV 共用代码，按 runtime SID 处理各自一半 head；每个 head 仍只计一次 sink。

令 `CP=ceil(TOPK/512)*512`、`S=WINDOW+CP`。窗口和压缩 Score 使用各自
连续的 `VR×WINDOW`、`VR×CP` UB 矩阵，不再将拼接的 S 列扩展到 512 的倍数。
压缩 gather 默认使用双槽 32 行 UB，索引及请求的压缩页表先批量载入私有 UB，
再由 Scalar 逐行读取；有效槽 Bias 为 0，无效槽 Bias 为 −∞。
行缓冲仅在 kernel 入口清零，无效槽随后保留有限的旧值，并由 Bias 排除权重。
双槽保留旧块发布与新块预取的重叠，复用仍等待旧 MTE3 读取退休。
`ROWS=16` 政策也使用同一条双槽流水；具体组合仍检查真实 UB 容量。
重复索引仍逐次参与。窗口继续从分页 GM 直接进入 L1。
L1 流式槽及常驻 KV 块在 kernel 入口用 `N.fill(...,0)` 初始化，此后仅覆盖真实页片段，
不为每个跨页窗口重新清零。未覆盖行保留已定义的旧值，但不会获得 softmax 权重；
初始化和全部部分写仍经过同一 effects、ownership 和同步检查。算法不要求索引排序。

QK 先计算 window 的 Score `[0,WINDOW)`，与两个 AIV 的压缩 gather 重叠，
随后计算压缩 Score `[WINDOW,S)`。D 可被 256 整除时，每个 L1 panel 一次
搬运 256 维，供按原顺序执行的两次 128 维 L0 MAD 使用；其余完整 128 维
特征块使用 128 维搬运。若完整 Q tile 的 b16 数据不超过 64 KiB，QK 将其
按 `BM×128` 的紧凑平面一次装入 L0A，供全部 window/压缩块复用；否则
继续使用原双槽流式装载。默认 Q 的 L1→L0A 搬运由 320 KiB 降为 64 KiB，
相应 128 维 Q operand 装载由 20 次降为 4 次。L0A 默认使用 64 KiB，未提高硬件容量；
MAD 数量、K 维累加次序及输入精度保持。PV 随后复用前两个 L0A 平面，
覆写仍等待全部真实 MAD reader 退休。
最后一个压缩 FIX 的完成前缀经过全部压缩
QK 输入，包含双方 AIV 的 gather 发布。softmax **先获取完整 Score，再读取 Bias**；
仅获取提前完成的 window Score 不提供这条交接。普通 AIV peer GM 访问没有直接
交接协议；不能把 Bias DMA 提到完整 Score acquire 前。

AIV 先获取压缩 Score 的完整发布前缀，再读取窗口 Score 和 Bias。默认每组处理
16 行 softmax。两份 Score 分别加 Bias、求行最大值，再与同一个 sink 合并为
共同最大值；两份概率均减去该值后计算，两个 FP32 行和相加，sink 只计一次分母。
两份 b16 P 分别写回原 GM 矩阵的窗口列和压缩列，后续 PV 顺序不变。
这不恢复在线 max/numerator 递推，也不分别归一化两个集合。Bias 使用普通
向量广播或逐行加法，不物化完整 Bias 矩阵。分母保存为每 token/head tile 的
双槽私有 UB 快照。

Cube 的 PV 从 window 开始，随后按原压缩顺序累加全部行，FP32 L0C 只在首次
MAD 清零。D 可被 256 整除时，一次 128×256 L1 搬运供两个相邻 128 列输出，
共用 P 装载，再一次 FIX 写出 256 列。finish 每次处理 8 行，只做 FP32 分母
归一化、转换与最终写出。QK/PV 共用双槽 L1 KV 池、L0A/B 和按输出列切分的 L0C。
PV 将前 `min(D,S)` 列 P 存入已经完成 QK 读取的 Q L1 allocation，供所有输出列块
复用；其余 P 继续按 128 列流入双槽。默认 D=512、S=640 时，P 的 GM→L1 搬运
从十次 128 列变为一次 512 列和两次 128 列；没有增加 L1 分配，Q/P 的存储复用
经过普通 MTE1 最后读取与 MTE2 覆写依赖，不能只凭源码阶段退出认定已经退休。
容量允许时，窗口及最前面的压缩块按 128 行、完整 D 维保留在 L1，最多两块，
由 PV 复用；其余压缩块继续从 gather workspace 载入。常驻块也保留原来的
MTE1 最后读取与跨 query 覆盖约束，不按存储中已有数据推断异步完成。
L1 的真正最后 reader 是 MTE1，源码不指定事件 ID 或全核 barrier。

B=1、H=BM、T>=32 且窗口可常驻 L1 时，prefill 将窗口位置 p 存在 L1 第
`p % 128` 行。每个 worker 首次载入真实可见窗口，随后普通完整窗口只补入
新进入的行；默认20个 worker 时每 query 从128行降为20行。页大小和窗口均为128，
因此每次补入最多跨两个页片段。当前或前一 query 缩短过窗口时，重新载入当前
真实可见段，避免把未存储的旧行当作可复用数据。每个 kernel 入口仍初始化常驻
L1，下一 query 的覆盖仍等待旧 PV 的真实 MTE1 reader。这个政策不改变 decode 排布。
窗口 Score/P 和 PV 使用同一个环形行序，mask 按当前窗口两段区间构造；
压缩 TopK 仍逐 query 独立 gather。环形窗口只调整窗口归约及 PV 累加的行顺序，
沿用下述误差合同，输出 dtype、sink 和压缩边界不变。

每个 worker 持续处理自己拥有的 token/head tile，默认至多 20 个 mixed group。
任务从 `[0,T*(H/BM))` 循环分配，不在请求边界清空流水。每个阶段从同一不可变 Cu
做 upper_bound 定位请求；二分查找使用普通动态 `if`，不需要复制页表或预展开为假 batch。发射顺序为
Cube 的 `pv→qk`、Vector 的 `gather→softmax→finish`，阶段偏移分别为 0/1/2。
KV/Bias 使用 `query%3`，Score/P/PV 和分母快照使用 `query%2`，不能共用一个 slot 时钟。

三槽还保护 **两个 AIV 对 Bias 的最后读取**。Cube 先执行 PV(q)，再执行 QK(q+1)；
后者的最终 Score 完成前缀经过读取双方 P(q) 的 PV，因此已经涵盖双方 Bias(q)
读取。每个 AIV 在下一轮 gather 前已获取该 Score。到 q+3 覆盖旧 Bias(q) 时，
这条中转已经发生；两槽在 q+2 覆盖时尚没有这一保证。KV 的 Cube reader 退休
本身不能授权覆盖 Bias。回归用独立有限模型保留提前读取和两槽覆盖的 RAW/WAR
反例，不以普通编译成功代替 peer 访问证明。

调用方分配以下 GM workspace，其中 `G=min(GROUPS,T*(H/BM))`，仅分配 resident worker 的切片：

| workspace | 形状和 dtype |
| --- | --- |
| KV / Bias | `[G,3,CP,D]` b16 / `[G,3,CP]` FP32 |
| Score / P | `[G,2,BM,S]` FP32 / b16 |
| PV | `[G,2,BM,D]` FP32 |

QK/PV 使用 b16 输入和 FP32 累加。P 在 PV 前显式舍入成输入 dtype；分母使用
未舍入的 FP32 概率，最终输出仍是输入 dtype。统一最大值后舍入 P，与原先
分 tile 舍入再重缩放的实现不位等价。设备 runner 按独立 NPU FP32 公式使用既定
`atol=0.005, rtol=0.02` 检查；窗口和压缩分别求 FP32 行和、再相加也遵守这份
误差合同，不承诺与原拼接归约树位等价。这不是任意输入的误差上界或 upstream 位精确资格。
最大值、分母、矩阵累加和最终归一化保持 FP32。Sinks、Q/KV 及实际 FP32 运算仍须满足有限性
与不溢出合同，SCALE 必须为正。

默认政策为 B=4、T=16、H=64、D=512、WINDOW=128、TOPK=512、S=640、
BM=64、VR=16、ROWS=32、BK=512、C=128、PAGE=128、MAX_SEQ=4096。默认
UB 内为窗口 128 列及压缩 512 列；更宽的压缩 Score 使用 8 行组和显式 512 列
UB 工作块处理行广播。当前仍要求 `ceil(S/512)*512 <= 4096`，
且所有具体组合仍需通过 192KB UB、L1/L0 和有限事件资源检查；不是所有 D/TOPK
组合均能 resident，不自动切换到另一编译路径。

与官方 CSA 保持相同的选中行、分页和共享 KV 语义，计算组织改为每 query 联合
softmax。官方 gather 还在合法时用双行 runtime-stride DMA 并压紧有效行；这里
仍逐行搬运并保留无效槽，不声称逐指令复刻；满有效块不再重复清零行缓冲。

算法分块、slot、任务顺序、QK/PV 复用和快照都在 example DSL 中表达。runtime
NZ 前缀覆盖、真实 pitched effects、初始化、ownership 和有限事件分配继续使用
同一编译路径。通用标量分支结果范围和 L1 清零原语不改变 event sharing 政策。官方动态 stride、
有效行压紧仍未开放；UB 整数元数据读取使用普通初始化、Tile 来源及 Scalar
代次捕获合同，不以扩大硬件容量、全局 barrier
或静默 fallback 绕过拒绝。

## 运行与边界

从仓库根目录执行，先加载 CANN 环境并激活工作区 venv：

```bash
python -m benchmarks.attention.run_sparse --build --batch 4 --query-tokens 4 --output build/sparse-c2
# 检查所有卡后，显式映射所选物理卡；runner 会再次检查健康和进程。
ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 \
  python -m benchmarks.attention.run_sparse --physical 2 --output build/sparse-c2

# C1 特化：构建与运行均传 --ratio 1。
python -m benchmarks.attention.run_sparse --build --ratio 1 --output build/sparse-c1
python -m benchmarks.check_examples --only sparse-mla --output build/sparse-smoke

# B1 prefill 使用同一个入口，query 数由 T 指定。
python -m benchmarks.attention.run_sparse --build --batch 1 --query-tokens 8192 \
  --max-seq 131072 --output build/sparse-prefill

# DSpark 主模型满宽验证：每请求 6 query，B64/T384，缓存容量 128K。
python -m benchmarks.attention.run_sparse --build --batch 64 --query-tokens 6 \
  --max-seq 131072 --output build/sparse-dspark-b64
```

只做 Python/IR 检查时加 `--ir-only`。每个构建共享 300 秒硬期限，失败即停。
运行复用 manifest 中已发布的库，不隐式重新编译。输出保存 policy、源码身份、
DSL/物理 IR、ASC/库 hash、CANN 命令、设备健康、原始输入和逐项误差；输入及
workspace 的尾部哨兵也要保持。默认检查空集合、window-only、混合两类 KV、
变长/空请求、索引 padding、重复索引、replay 后缀及极端有限 sink，并复用同一组 workspace。

每次设备资格只属于其记录的 DSL/IR/ASC/库与输入。runner 的 ACL event 包含
host submission gaps；性能对照使用 `benchmarks.attention.profile_sparse` 的 msprof
device task duration。官方参照必须是从 vllm-ascend 原始 host/kernel 构建的 custom OPP，
通过 `--reference-artifacts` 指定；元数据准备在计时外，输入身份逐项匹配。
`--rotations` 分配多组独立缓存和输入，并逐组检查输出；轮换本身不证明冷 HBM，
应结合 `--metrics MemoryAccess` 和 `--metrics L2Cache` 的原始记录分析。
`profile_sparse --batches 1` 由产物读取 query 数：context 等于 query 数时为完整
prefill，更长 context 表示带历史前缀的 chunk。早期 token 的不足窗口和不足 TopK
均按真实有效行计数。`profile_sparse` 只采集性能：记录种子、
输入/缓存 hash、源码与产物身份，不转储大规模输入/输出，也不运行数值参考、
gather 检查或算子输入/输出/workspace 回读检查。完成状态为 `measured`，
不代表新的正确性资格；正确性由独立的 `run_sparse` 入口检查。

`run_sparse` 按输入元数据独立选行，每128个 query 在 NPU 上执行 FP32 QK、
包含一个 sink 的 softmax 和 FP32 PV，禁用 HF32，不使用算子自身的 gather workspace。
设备数值校验需要工作区的 `torch` 和 `torch_npu`，编译和 CPU/IR 回归不依赖它们。
大 query 的输入准备及采集可用 `--application-timeout` 设置测量进程
期限；它不改变每个算子构建的300秒硬上限，也不改变 kernel 计时范围。

prefill 当前优化限于 B1 的常驻窗口复用，不合并不同 query 的 TopK。
当前不包括独立 SWA、非因果 draft/DSpark mask、LSE、split-KV、多卡、A5 的 FP4/FP8 packed cache，
多 token 资格仅属于因果验证阶段，不包括接受/拒绝后的缓存回滚。
也不包括 QLI TopK、跨层候选筛选、compressor、RoPE 和模型端到端接入。
INT32 元数据加载与受支配分支范围是通用能力，但不代表任意条件访问的存储分区、
通知配对或初始化合流已经支持；其余形式继续经过原有准入并可能明确拒绝。

`--query-lens 0 2 0 14` 可替代 `--query-tokens`，以同一 B4/T16 二进制运行不同
的 Cu。`--reference-artifacts <custom_transformer>` 同时检查官方同输入输出。
性能工具读取产物中的 B/T，按整个 forward 记录 msprof 耗时；T=B 的历史记录
不能作为推测译码性能证据。
