# KDA prefill

实现位于 [prefill.py](../../examples/kda/prefill.py)、
[prefill_prepare.py](../../examples/kda/prefill_prepare.py)、
[prefill_gram.py](../../examples/kda/prefill_gram.py)、
[prefill_solve.py](../../examples/kda/prefill_solve.py) 和
[prefill_output.py](../../examples/kda/prefill_output.py)。全部 kernel 使用普通 DSL、
xDSL 自动同步和 ASC/CANN；算法与流水政策在 examples，编译器没有 KDA 特判。

## 数值与输入合同

默认合同为 Q/K/V/Output FP16，Gate/Beta/Initial/Final FP32，K=V=128、
H=HV、H 为 8 的正整数倍、BT=64，启用 Q/K L2 归一化，Gate 为非正的 log 衰减值。
state 布局为 `[sequence, head, V, K]`。
每条序列的 chunk 数相同且非零；最后一个 chunk 可以不足 64 tokens，长度可以不同。
`TOKENS: N.int64(minimum=1)` 是运行时值，Starts/Lengths 为不可变 GM i64 元数据。
调用方保证每个 chunk 在对应序列内且 `1 <= valid <= min(64, TOKENS-start)`，
各序列只有最后一个 chunk 可以不满 64 tokens。
当前 runner 拒绝零长度和不同 chunk 数的序列；不由此推导 grouped heads、speculative
state indexing 或 BF16 的设备资格。

原始 Triton 的**矩阵乘两侧输入 dtype** 见[输入精度表](kda-prefill-precision.md)。
整块 Gate 前缀、跨块 Gram、块逆合并和历史状态投影使用 IEEE FP32×FP32 Cube；
W/U、状态增量与两个输出乘法使用 FP16×FP16、FP32 累加。
W 先舍入为 FP16，再转为投影所需的 FP32；不能省去中间舍入。
同一 16×16 对角块的 Gram 使用 FP32 元素乘、exp 和归约，不调用 dot。
FP32 carry 与 FP16 snapshot 是不同的值和存储，snapshot 不反馈到递推状态。

## 计算与流水

当前 runner 顺序发射十个 phase specialization，具体 ABI 与顺序见
[运行器](../../benchmarks/kda/prefill.py)。同一 stream 保证 kernel 间依赖，
kernel 内的 ready/release、GM 交接和事件资源由公共同步链建立。

| 阶段 | 计算及流水 |
|---|---|
| prepare_full | 非末块以 FP32 下三角矩阵乘计算 Gate 前缀；三阶段双槽流水，Cube 前缀与 Vector 归一化、beta 提取重叠 |
| prepare | 每序列末块使用运行时 valid；三阶段 token 流水与 FP32 Vector 顺序前缀，清零 padding |
| gram | 四个 16×16 对角块在 UB 计算；四块逆按 `[16,64]` 合批，复用不变 packed mask |
| gram_cross | chunk/head 任务在核内循环；外层 prepare/compute/store 三阶段复用两槽 GM workspace，Cube 子流水双缓冲导入并做 IEEE FP32 dot |
| solve_merge16 / solve_merge32 | 同一 kernel 的两个 constexpr 特化，FP32 Cube 将 16×16 对角逆合为 32×32、64×64，三阶段双槽流水 |
| transform | Vector 构造 beta*K*expG、beta*V、KG、QG 和 A16，保留各自 FP16 舍入边界 |
| wu | FP16 Ai @ RHS 生成 W/U；三阶段双槽流水，W 经 FP16 GM 槽后转存 FP32 |
| scan | 以 sequence/head 分配任务；chunk 间两阶段预取/递推，历史投影、residual、状态增量和 FP32 carry 更新保持递推顺序 |
| output | 以 chunk/head 并行，双缓冲导入、Cube 计算两项输出乘法、Vector 合并及有效行写回 |

块逆使用 `[[D0,0],[-D1*A10*D0,D1]]` 合并 16→32→64，乘法输入保持 FP32。
对角逆的四个矩阵沿列合并，使乘法及加法树填满 64 个 FP32 lane；行依赖仍在核内顺序执行。
跨块 Gram 采用 16/32-token 分层矩形，运行器将一个 cohort 的任务数设为
`PACK=gcd(CHUNKS*H,16)`。每组两个 workspace 槽由实际最后 reader 约束回收，
不再由 host 按 512-task slab 重复发射 Vector/Cube kernel。
Cube/Vector 通过 GM workspace 交接；每个 mixed kernel 的两个 AIV 共用 body，
通过 runtime SID 区分地址和角色，分析保留各自私有 UB 与 quorum 贡献。

gram_cross 每项的四片左操作数在 L1 中保存为一个 `[128,K]` NZ slab，使用
一次真实 GM→L1 DMA 导入，再由四次行切片 Load3D 读取。slab 下一次覆盖等
全部旧 reader 退休。右操作数保留紧凑转置平面；每项导入从七次减少为四次，
L1 容量不变。FP32 Load3D A 的 strided NZ 源保留每个 K/8 平面的实际行距，
行距按 16×8 个 FP32 对齐且宽度不超过 256；目标仍是完整紧凑 L0A 平面。
该能力不放宽 FP32 转置 Load2D 的紧凑源合同，也不改变矩阵乘输入精度。

scan 中每个 chunk 保存更新前的 FP16 snapshot，同时用 FP32 历史状态做投影；
residual 舍入为 FP16 后参与状态增量。carry 在 FP32 中完成衰减与加法，
每个 sequence/head 必须按 chunk 顺序执行。独立的 output 阶段读取各 chunk 的
snapshot、residual 等数据，按 chunk 分配负载，不重新计算递推状态。

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
具体数值与性能结果随每次构建和采集保存，旧版本快照不作为当前代码资格。

## 复现

使用工作区独立 venv，从产品仓库执行：

```bash
source ../.venv/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.kda.prefill --build-only --physical 2 \
  --artifacts build/kda/prefill --output build/kda/prefill-device --lengths 4096
python -m benchmarks.kda.profile_prefill --physical 2 \
  --artifacts build/kda/prefill --output build/kda/prefill-ours
python -m benchmarks.kda.profile_prefill --physical 2 --backend triton \
  --artifacts build/kda/prefill --output build/kda/prefill-triton
python -m benchmarks.kda.compare_prefill \
  --ascendflow build/kda/prefill-ours --triton build/kda/prefill-triton \
  --official-release /workspace/xdsl-910b/materials/triton-official/release.json \
  --output build/kda/prefill-comparison.json
```

完整 pipeline 构建和每次运行分别有单次 300 秒进程组硬上限。运行前检查所有卡的
健康、进程、显存和利用率，每进程显式映射物理卡至逻辑 0；设备错误直接停止。
`--run-lengths` 可复用同一份库改变 tokens 数和尾块，只接受已编译的 sequence/chunk 域。
数值检查先于报告性能；stream interval 另有标记，不用它代替 kernel 时间。
batch=4 构建时将 `--lengths` 改为 `512 512 512 512`；该库可用
`--run-lengths 449 477 505 512` 运行尾块配置，输出目录使用新路径。
