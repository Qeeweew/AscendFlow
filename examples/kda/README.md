# KDA 示例

[recurrent.py](recurrent.py) 每个任务持有完整 head 的 FP32 carry，按 token 做输入预取、
分块更新和独立快照写回的三阶段流水；[decode.py](decode.py)
为每序列至多一个 token 的 head 流水。两者共用 [ops.py](ops.py) 的归一化和
delta update，状态槽、快照与数值条件见 [KDA 合同](../../docs/operators/kda.md)。

Prefill 发射三个 kernel：

| 模块 | 入口 / 阶段 |
|---|---|
| [prefill_prepare.py](prefill_prepare.py) | `prefill_prepare`：每个 chunk/head 在同一 mixed group 内完成归一化、前缀、Gram/块逆、RHS 与 W/U |
| [prefill_scan.py](prefill_scan.py) | `prefill_scan`：历史投影、snapshot 与 FP32 carry 递推 |
| [prefill_output.py](prefill_output.py) | `prefill_output`：两项输出合并与有效行写回 |

Prepare 的对角 Gram、跨块操作数、块逆和 W/U 分别复用
[prefill_gram.py](prefill_gram.py)、[prefill_gram_cross.py](prefill_gram_cross.py)、
[prefill_solve.py](prefill_solve.py) 和 [prefill_wu.py](prefill_wu.py) 中的 inline helper。
[prefill_transform.py](prefill_transform.py) 的行变换同时用于 Prepare 的每 AIV 32 行
和独立 transform 的整块 64 行；FP32 K 与 exp(G) 在该阶段复用，舍入边界保持。
Prepare 入口集中声明存储、阶段别名和 Dataflow 排步；非末块与末块先后进入同一个
任务域。不同 kernel 间由 stream 建立全设备阶段顺序，kernel 内使用同一 group
的普通自动同步。

完整输入、精度及同步合同见 [KDA prefill](../../docs/operators/kda-prefill.md)。
构建与设备 oracle 统一由 `python -m benchmarks.check_examples --only kda-prefill`
执行，参数和输出路径见[示例入口](../README.md)。

独立 fragment 可用以下命令构建；完成全卡检查和显式映射后增加
`--physical <物理卡号>` 检查设备结果。

```bash
python -m benchmarks.check_examples --only kda-gram kda-gram-cross \
  kda-solve16 kda-solve32 kda-transform kda-wu --output <目录>
```

这些检查使用完整小 cohort、
37/64 行尾块、环复用和三次调用，对照独立 CPU 方程；不作为三阶段性能结果。
