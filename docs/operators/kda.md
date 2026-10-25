# KDA decode

入口为 [decode.py](../../examples/kda/decode.py)，算术 helper 为
[ops.py](../../examples/kda/ops.py)。这是一条普通 DSL → xDSL → ASC → CANN 链路，
没有手写 kernel、事件、全局 barrier 或编译器中的 KDA 特判。

## 输入与数值合同

默认示例为 128 条序列，每条至多 1 token，H=HV=32、K=V=128，
Q/Key/Value/Output 为 BF16，Gate/Beta/State 为 FP32，开启 Q/K L2 归一化。
状态采用 `[slot, head, V, K]`，原地更新。不同活跃序列拥有互不重叠的正状态槽；
槽 0 为 inactive sentinel，禁止覆盖。数值验收检查完整输出、完整状态和槽 0，
output/state relative RMSE 门槛均为 `<0.005`。

`TOKENS: N.i64` 是运行时维度。Cu/Indices 是只读 GM int64 tensor；
Cu 单调，区间满足 `0 <= bos <= eos <= TOKENS` 且 `eos-bos <= 1`。
Cu 的 `N.load_index` 范围限定实际 token 时钟，用于生成通知的整数算术；
Indices 使用完整 int64 范围，不为间接地址额外填写范围提示。范围声明是调用方
输入合同，不生成 clamp；正状态槽及访问合法性仍由调用方保证。元数据在 S pipe
有序读取。recurrent 与 decode 共用 `delta_step`，保持相同的 FP32 递推和归约次序；
recurrent 的外积复用 product scratch，不再另存一份广播后的 K 矩阵。

其他 constexpr/dtype 组合需要重新满足 UB 容量及 native 几何合同，并获得自己的设备资格。
prefill 的算法与精度见 [KDA prefill](kda-prefill.md)。[recurrent.py](../../examples/kda/recurrent.py)
保留 token 顺序、FP32 carry 与独立快照的实现；不能由 decode 的检查推导其 speculative 等选项的资格。

## 多 token recurrent

每个 Vector block 拥有一个 sequence/head，完整 FP32 carry 在 UB 中跨 token 保留。
默认 BV=64，只对 state update 做 V 轴分块；Q/K 归一化、gate exp 和 beta 每个
token/head 只计算一次。不同 V 分块不再各自重复加载和归一化 Q/K。

流水时钟为该序列内的 token ordinal，三阶段依次为双槽输入预取、FP32 更新与
快照生成、整 head 输出及快照写回。独立的单份快照缓冲允许 MTE3 写回与下一
token 的 carry 更新重叠；编译器根据快照的最后 reader 约束其复用。
默认 H=HV=32、K=V=128、BV=64、两输入槽、BF16 输入/输出与 FP32 状态合计
使用 177216 B UB，不扩大硬件容量。其他 shape/dtype 组合仍须通过真实容量检查。

每个 token 更新后的状态按 `INPLACE` 写入 `Indices` 指定的正槽或 `Snapshots[token]`；
快照转换为 `STATE_DTYPE`，下一 token 继续使用未经该转换的 FP32 carry。
`SPECULATIVE` 从 `Accepted[seq]-1` 选择初始状态，正初始槽才激活 indexed 序列。
空序列和 inactive 序列不写输出或快照，非正快照目的槽不写状态；其他活跃
token 仍按原顺序更新 carry。Cu 必须满足单调及实际 tensor/索引宽度合同。

`benchmarks.kda.profile --lengths ...` 可以采集编译后的 recurrent 产物，长度总和、
序列数和索引宽度须与产物一致。例如 4 条各 32 tokens 的配置使用
`--lengths 32 32 32 32`，kernel 时间以 profiler 的 `Task Duration` 为准。

## 计算与流水

归一化后，令 `qbar=scale*q`、`d=exp(g)`：

```text
D[v,k] = State[v,k] * d[k]
p[v]   = sum_k D[v,k] * k[k]
r[v]   = beta * (Value[v] - p[v])
S[v,k] = D[v,k] + r[v] * k[k]
O[v]   = sum_k S[v,k] * qbar[k]
```

两次主归约使用显式 `block8_then_packed_merge`：K=128 时每行先 128→16，再 16→1。
乘、加分别保留，不将低精度快照反馈到 FP32 状态。
外积用 `N.mul(k[None,:], r[:,None], scratch=...)`，直接读取列向量和 BRCB 行值，
省去先复制整份 K 矩阵的中间读写。

默认 1024 个 Vector launch blocks，每个 block 拥有一个 sequence 与循环分配的 4 个 heads。
流水时钟是 head ordinal，三阶段为完整 head import、BV=64 的 FP32 分块更新、完整 head export。
Q/K 归一化、gate exp 和 beta 每个 head 只做一次。两个状态槽加单份计算 scratch 共占
177216 B（173.0625 KiB）UB，低于 910B 单 AIV 的 192 KiB。
编译器生成输入 ready、输出 ready、真实最后 reader 的 release 及槽位回收。

recurrent/decode 的输入与输出分别以显式 slots 和 `at(item)` 选择，保留原来
的独立容量。一次输入导入用 CopyGroup 共同 ready，包含实际的 Q/K、Gate、
Value 及 decode 的 State 读取；不合并 DMA 或放大存储 release。
归一化、decay、delta update 和转换结果传递对应 Tile；recurrent 的完整 FP32
carry 继续原地递推，导出的快照来自独立存储及其真实 writer。

每 token/head 的有用算术为 `7*K*V + 7*K` FLOPs，另有 K 次 exp 和两次 rsqrt。
默认完整 launch 为 473432064 FLOPs。有效 GM 流量约 518.016 MiB，其中 state 读写为 512 MiB；
PMU 请求流量、UB 流量与 HBM 实流量是不同计数口径。

## AscendC 对照方法

主基线为 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)
的未修改 `RecurrentKda`，也是 vllm-ascend 使用的 AscendC recurrent 算子。
Q/K/V/Output 统一为 BF16，Gate/Beta/State 保持 FP32，开启 kernel 内 Q/K
归一化，使用相同的间接槽、输入字节和原地更新合同。参考公开接口只支持 BF16
Q/K/V，不能与历史 FP16 Triton 数字混作同一对比。

参考通过 `aclnnRecurrentKda` 调用，输出和完整状态先通过独立 CPU 递推检查。
OPP 的构建记录包含上游提交、源码、编译命令与 runtime 文件 hash；比较时复核
产物、输入、平台和 PMU 模式。Triton 入口仅供原始精度审计和历史诊断。

参考源码单独保留，上游源码不做补丁。`benchmarks.kda.build_reference` 使用上游
`build.sh --jit` 生成 host/源码包，再调用 CANN `asc_opc` 编译本次 dtype 的动态
shape 签名，生成标准 kernel 配置。单次 300 秒期限覆盖整个参考构建。
`--dependencies` 指向 CANN 第三方安装材料（json/makeself/pkg）；这些只供参考构建。

```bash
python -m benchmarks.kda.build_reference --checkout /path/to/flash-linear-attention-npu \
  --dependencies /path/to/cann-third-party --output build/kda/reference
```

```bash
source ../.venv/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.kda.profile --artifact build/kda/decode \
  --output build/kda/ours --physical 2
python -m benchmarks.kda.profile --artifact build/kda/decode \
  --output build/kda/ascendc --physical 2 --backend ascendc \
  --reference-artifacts build/kda/reference/opp/vendors/fla_npu_transformer
python -m benchmarks.kda.compare --ascendflow build/kda/ours \
  --ascendc build/kda/ascendc --output build/kda/comparison.json
```

recurrent 使用相同工具并传入 `--lengths 32 32 32 32`。batch 矩阵通过
`benchmarks.kda.matrix --reference-artifacts <vendor>` 使用同一 AscendC 基线。
所有 runner 都先检查全卡状态、显式映射物理卡到逻辑 0，并使用 300 秒进程组
硬期限。性能只用 msprof 的 kernel Task Duration；MSTX `kda` 范围包含实际调用，
不含输入准备与数值检查。stream interval 另行标记，不代替 kernel 时间。
