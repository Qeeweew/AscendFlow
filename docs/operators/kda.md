# KDA decode

入口为 [decode.py](../../examples/kda/decode.py)，算术 helper 为
[ops.py](../../examples/kda/ops.py)。这是一条普通 DSL → xDSL → ASC → CANN 链路，
没有手写 kernel、事件、全局 barrier 或编译器中的 KDA 特判。

## 输入与数值合同

默认示例为 128 条序列，每条至多 1 token，H=HV=32、K=V=128，
Q/Key/Value/Output 为 FP16，Gate/Beta/State 为 FP32，开启 Q/K L2 归一化。
状态采用 `[slot, head, V, K]`，原地更新。不同活跃序列拥有互不重叠的正状态槽；
槽 0 为 inactive sentinel，禁止覆盖。数值验收检查完整输出、完整状态和槽 0，
output/state relative RMSE 门槛均为 `<0.005`。

`TOKENS: N.int64` 是运行时维度。Cu/Indices 是只读 GM int64 tensor；
Cu 单调，区间满足 `0 <= bos <= eos <= TOKENS` 且 `eos-bos <= 1`。
`N.load_index` 的 inclusive 范围是调用方输入合同，不是 clamp。
元数据在 S pipe 有序读取；没有引入新的同步协议。

其他 constexpr/dtype 组合需要重新满足 UB 容量及 native 几何合同，并获得自己的设备资格。
prefill 的算法与精度见 [KDA prefill](kda-prefill.md)。[recurrent.py](../../examples/kda/recurrent.py)
保留 token 顺序、FP32 carry 与独立快照的实现；不能由 decode 的检查推导其 speculative 等选项的资格。

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

每 token/head 的有用算术为 `7*K*V + 7*K` FLOPs，另有 K 次 exp 和两次 rsqrt。
默认完整 launch 为 473432064 FLOPs。有效 GM 流量约 518.016 MiB，其中 state 读写为 512 MiB；
PMU 请求流量、UB 流量与 HBM 实流量是不同计数口径。

## Triton 对照方法

参考为 vllm-ascend `2581a69bcc16ee6af56e72985a15799bb5f61808` 的未修改
`fused_recurrent_kda`，保留 BV=8、num_warps=1、num_stages=3。
比较工具核对[官方 Triton-Ascend 3.6.0](https://github.com/triton-lang/triton-ascend/releases/tag/v3.6.0)
wheel、参考源码、输入、平台和产物身份，不能将本地补丁版作为未修改的参考。
双方采用相同 shape/dtype、输入和原地状态合同，先检查数值，再比较 kernel Task Duration。

batch=1、2、4、8、16、32、64 的 BasicInfo 矩阵使用相同的
HEAD_GROUPS=8、BV=64、两槽三阶段政策；H=HV=32、K=V=128，每序列一 token。
结果、输入 hash、数值记录、原始 CSV 和逐点比较写入指定输出目录：

```bash
python -m benchmarks.kda.matrix --output build/kda/matrix --physical 2 \
  --official-release /workspace/xdsl-910b/materials/triton-official/release.json
```

单点比较从产品仓库执行，使用独立工作区 venv。安装的 Triton wheel 必须与官方 Release 记录匹配。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
from ascendflow.build import build_file
build_file('examples/kda/decode.py', 'build/kda/decode', entry='decode_kda',
           diagnostics=True, shared=True, timeout_s=300)
PY
python -m benchmarks.kda.profile --artifact build/kda/decode --output build/kda/ours --physical 2
python -m benchmarks.kda.profile --artifact build/kda/decode --output build/kda/theirs --physical 2 --backend triton
python -m benchmarks.kda.compare --ascendflow build/kda/ours --triton build/kda/theirs \
  --official-release /workspace/xdsl-910b/materials/triton-official/release.json \
  --output build/kda/comparison.json
```

每个 runner 先检查全卡健康、进程、显存和利用率；每个构建或运行有 300 秒硬上限。
`run.py` 的 stream interval 含 host 提交间隙，性能比较只读取 profiler 的 kernel 时间。
