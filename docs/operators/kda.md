# KDA decode

入口为 [decode.py](../../examples/kda/decode.py)，算术 helper 为
[ops.py](../../examples/kda/ops.py)。这是一条普通 DSL → xDSL → ASC → CANN 链路，
没有手写 kernel、事件、全局 barrier 或编译器中的 KDA 特判。

## 合同与已验证范围

2026-10-18 的设备配置为：128 条序列，每条 1 token，H=HV=32，K=V=128，
Q/Key/Value/Output 为 FP16，Gate/Beta/State 为 FP32，开启 Q/K L2 归一化。
状态采用 `[slot, head, V, K]`，原地更新。不同序列拥有互不重叠的正状态槽，
槽 0 为 inactive sentinel，禁止覆盖。当前设备证据覆盖完整输出、完整状态和槽 0。

`TOKENS: N.int64` 是运行时维度。Cu/Indices 是只读 GM int64 tensor；
Cu 单调，区间满足 `0 <= bos <= eos <= TOKENS` 且 `eos-bos <= 1`。
`N.load_index` 的 inclusive 范围是调用方输入合同，不是 clamp。
元数据在 S pipe 有序读取；没有引入新的同步协议。

其他 constexpr/dtype 组合需要重新满足 UB 容量及 native 几何合同，并获得自己的设备资格。
本页不声明 chunk/prefill 已实现或验证。[recurrent.py](../../examples/kda/recurrent.py)
保留早期 token 顺序、FP32 carry 与独立快照的源码；其 speculative 等选项未获得完整设备资格。

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

## 与原始 Triton 实现的设备比较

参考为 vllm-ascend `2581a69bcc16ee6af56e72985a15799bb5f61808` 的未修改
`fused_recurrent_kda`，保留 BV=8、num_warps=1、num_stages=3。
使用 [triton-lang/triton-ascend v3.6.0](https://github.com/triton-lang/triton-ascend/releases/tag/v3.6.0)
官方 wheel。下载索引为仓库 README 指定的 Ascend 索引，其 SHA256 与 GitHub Release 完全一致：
`d47369703417e0cbb9ee97795a6eaf0fa42996cfaf037eb777cba01d558868c8`。
已安装 runtime 源码与该 wheel 相同，无本地补丁。

设备为物理 2、逻辑 0，Ascend910B3，CANN 9.1.0，1800 MHz；Torch 2.7.1+cpu、
torch_npu 2.7.1.post10。两边使用相同 shape、dtype、seed=42 的逐字节相同输入、原地状态合同和 oracle。
各采集一次 `msprof op --aic-metrics=Default --launch-count=1 --warm-up=0`，时间为 Task Duration。

| 实现 | kernel 时间 | Output relative RMSE | State relative RMSE |
|---|---:|---:|---:|
| AscendFlow | 418.500 μs | 0.00122164 | 0.000164236 |
| Triton-Ascend 3.6.0 | 2072.980 μs | 0.00000849911 | 0.0000000362726 |

本次加速比 4.953×；两边通过 relative RMSE < 0.005 的完整数值验收。
这是一个 batch=128 配置的单次 profiling 结果，不外推其他 shape 或统计置信区间。
此前同规模逐 head 填排的 AscendFlow 为 508.780 μs；连续 head 流水改善到上述结果。

证据位于 Git 忽略的 `build/kda/`：

- `decode-final-b128/`：source/physical IR、ASC、CANN binary/library、选项及 SHA256。
- `op-final-b128-01/`、`op-triton36-b128-01/`：全量数值记录、原始 CSV、计数汇总；后者包含 Triton IR、npubin 及编译 metadata。
- `comparison-b128.json`：验证输入、设备、官方 wheel、参考 checkout 与产物身份后生成的比较。

## 复现

从产品仓库执行，使用独立工作区 venv。安装的 Triton wheel 必须与官方 Release 记录匹配。

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
