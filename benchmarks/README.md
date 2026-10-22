# 示例验证与测量工具

日常只需要两个入口，完整操作见[测试指南](../docs/guide/testing.md)：

```bash
python -m benchmarks.check_examples --list
python -m benchmarks.check_examples --output build/examples-cann
# 检查全部卡并显式映射后，增加 --physical 进行数值验证。
python -m benchmarks.profile_defaults --output build/kernel-times --physical 2
python -m benchmarks.profile_defaults --causal --require-no-slower --output build/causal-times --physical 2
```

`check_examples` 检查统一清单内的公开示例；`profile_defaults` 对 Flash Attention 和 GEMM normal
固定的 10 个 shape（4 个等长非 causal FA、6 个 GEMM），分别采集 AscendFlow 与 ACLNN，
各一次默认 msprof，计算时间、加速比和 TFLOPS。FA 两种模式都只按 1K/2K/4K/8K 正方形验收。
`--causal` 单独采集这四个等长 causal FA shape，使用 example 中三组显式分块；
`--require-no-slower` 在任一 shape 慢于 ACLNN 时返回失败，数值通过不代表性能通过。
输出目录须是新目录。运行环境、卡选择和
输入合同见[快速上手](../docs/guide/quickstart.md)，每次运行的结果和产物保存在指定输出目录。

| 工具 | 用途 |
|---|---|
| `check_examples.py` | 统一示例清单，CANN 构建、设备数值检查、失败即停 |
| `profile_defaults.py` | FA/GEMM 多 shape 的 ACLNN 对照、kernel time、加速比与 TFLOPS |
| `vector/`、`normalization/`、`gemm/`、`cv/`、`state/`、`attention/` | 单个示例 runner；参数用 `python -m <模块> --help` 查看 |
| `vector/run_masks.py` | 通用mask示例：`--build`离线共享构建；`--artifact/--library`只校验已发布ASC/共享库/manifest，不重新编译；`--physical`按独立逻辑oracle运行 |
| `vector/run_mask_rows.py` | 小行/列mask示例：单UB tile、`cols<hi`/`cols!=lo`/`rows>=cols-q` 三输出；同样`--build`或只校验复用`--library`，记录逐输出位精确证据 |
| `profile.sh`、`profile_summary.py`、`artifacts.py` | msprof 应用包装、设备 task 汇总和产物身份检查 |
| `kda/run.py`、`kda/profile.py`、`kda/compare.py`、`kda/matrix.py` | KDA decode 全量数值验收、BasicInfo、batch 1–64 同输入/设备/官方 Triton wheel 比较；见[decode](../docs/operators/kda.md) |
| `kda/prefill.py`、`kda/profile_prefill.py`、`kda/compare_prefill.py` | KDA prefill 多 kernel 完整构建、独立 CPU 递推数值验收、BasicInfo 总时间及原始 Triton 对照；见[prefill](../docs/operators/kda-prefill.md) |
| `kda/audit_prefill.py`、`gemm/run_fp32.py` | 原始 dot 两侧输入精度离线审计、FP32 Cube 随机输入设备检查 |
| `attention/run_sparse.py` | V4.1 CSA 单 kernel decode/prefill、INT32 页表/索引、NPU FP32 数值参考与输入/哨兵检查；见[合同](../docs/operators/sparse-mla.md) |
| `attention/profile_sparse.py` | 普通 msprof 采集 CSA 与官方 vllm-ascend kernel；支持独立缓存轮换、显式 L2 驱逐、batch 和上下文矩阵 |
| `validation/` | 独立数值参考、输入数据和 dtype 编解码 |
| `references/` | 日常 ACLNN 适配器与独立 AscendC 参考 |

Sparse 冷缓存测量通过 `--cache-evict-artifacts <目录>` 启用。驱逐产物复用已有 FP32 递推
DSL 样例，可在产品仓库内构建：

```bash
python - <<'PY'
import json
from pathlib import Path
from ascendflow.build import build_file
output = Path("build/l2-evict")
parameters = {"B": 8192, "GROUPS": 40}
build_file("examples/state/recurrence.py", output, entry="state_sum",
           parameters=parameters, shared=True, timeout_s=300)
(output / "eviction-policy.json").write_text(
    json.dumps({"parameters": parameters, "tiles": 819}) + "\n")
PY
```

每次 attention launch 前在同一 stream 读取独立的 1023.75 MiB 输入，将 checksum 保留在 UB，
最后仅写每核一个 tile（合计 1.25 MiB），避免大输出的脏 L2 尾部干扰后续 HBM 访问。
读取容量须至少为实卡 L2 的四倍；`device.json` 记录产物身份、容量和扫描校验。汇总检查驱逐与 attention
任务交替且不重叠，仅报告 attention 的 device task duration，排除驱逐、metadata 和 host gaps。
使用普通 `msprof`，不使用 `msprof op`；可选 `--metrics L2Cache` 核对缓存计数。
这表示 kernel 入口前的缓存驱逐，kernel 内部正常的 KV/L2 复用仍会发生；轮换输入本身不代表冷缓存。

## 专项诊断

以下工具保留给明确的编译或性能调查，输出均写入 Git 忽略的 `build/`：

- `compile.py`：冷进程 source→ASC 计时；单次 probe 与至少五次资格分开，CANN/RSS 另列。
- `attention/run_policy.py`：独立手写 AscendC 政策对照。
- `gemm/compare_*`、`attention/compare.py`：历史三批严格比较流程，要求各自记录 schema。
- `*/analyze*.py`、`work.py`、`regions.py`：结构清单、算法工作量与条件速率分析。

`source-work-v1` 的静态模板数不代表运行期执行数，算法 FLOPs 不代表 native FLOPs；
运行期请求数尚不可得时保持 `unavailable`。profile 与实际 DSL、IR、ASC、库、选项、
输入和设备绑定，不能将参考路径的通过当成当前 DSL 通过。验证边界见
[开发验证方法](../docs/development/validation.md)。
