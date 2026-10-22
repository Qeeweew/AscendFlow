# KDA 示例

[recurrent.py](recurrent.py) 每个任务持有完整 head 的 FP32 carry，按 token 做输入预取、
分块更新和独立快照写回的三阶段流水；[decode.py](decode.py)
为每序列至多一个 token 的 head 流水。两者共用 [ops.py](ops.py) 的归一化和
delta update，状态槽、快照与数值条件见 [KDA 合同](../../docs/operators/kda.md)。

Prefill 按下面的顺序发射，同一模块保存该阶段的算法和 inline helper：

| 模块 | 入口 / 阶段 |
|---|---|
| [prefill_prepare.py](prefill_prepare.py) | `prefill_prepare_full` 整块前缀；`prefill_prepare` 尾块与 padding |
| [prefill_gram.py](prefill_gram.py) | `prefill_gram` 对角 Gram 与块逆 |
| [prefill_gram_cross.py](prefill_gram_cross.py) | `prefill_gram_cross` 跨块 Gram |
| [prefill_solve.py](prefill_solve.py) | `prefill_solve_merge` 的 16/32 两个特化 |
| [prefill_transform.py](prefill_transform.py) | `prefill_transform` 形成 RHS、KG、QG 与 A16 |
| [prefill_wu.py](prefill_wu.py) | `prefill_wu` 计算 W/U 并保留 W 的低精度舍入 |
| [prefill_scan.py](prefill_scan.py) | `prefill_scan` 历史投影、snapshot 与 FP32 carry 递推 |
| [prefill_output.py](prefill_output.py) | `prefill_output` 两项输出合并与有效行写回 |

完整输入、精度及同步合同见 [KDA prefill](../../docs/operators/kda-prefill.md)。
构建与设备 oracle 统一由 `python -m benchmarks.check_examples --only kda-prefill`
执行，参数和输出路径见[示例入口](../README.md)。
