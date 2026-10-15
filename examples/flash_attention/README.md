# Flash Attention

**此目录目前只有 legacy 实现。新 AST 的嵌套/连续 CV FA 尚未接通。**

- [legacy.py](legacy.py)：旧 physical builder 的 online forward FA。
- [tiled_legacy.py](tiled_legacy.py)：通信块、Cube native 块和 Vector 行块独立分块。
- [历史数值合同、变体与性能](../../docs/flash-attention.md)：支持范围针对该旧路径。
- [新 DSL 目标](../../docs/attention-target.md)：共用数学主体的 nested/continuous 规格。

执行和性能工具在 `benchmarks.attention`；手写 AscendC 在
`benchmarks/references/ascendc/ascendc_attention.asc.in`，ACLNN 适配器在
`benchmarks/references/attention_aclnn.py`。手写参考可运行不代表新 AST FA 已支持。

下一部分先迁移完整固定政策 FA，再逐项验收 f16/bf16、MHA/GQA/MQA、
prefill/decode、causal/非方形/tails；varlen/paged/sparse/persistent 需要额外
ownership/descriptor 合同。不会靠目录名或空接口宣称变体已完成。

新DSL的packed行归约、maximum、带scratch的行广播已经接通，并用于
[softmax示例](../normalization/softmax.py)。它验证FA所需的Vector基元；
完整online softmax的状态、QK/PV及nested CV仍待接通。

当前已测的性能政策为`tiled_legacy.py`的`--vector-update tile --max-reduce block
--overlap-queries --query-tiles-per-group 4 --load3d`（4K MHA）；8K GQA用每组两个
query tile。它将完整下一Vec1放在旧PV等待之前，Vec2处理整个owner的行，max使用
分层BlockReduce，sum累加与概率舍入保持原样。大例结果及限制见[设备对照](../../docs/flash-attention.md)：
4K三批698.726μs对ACLNN512.678μs，8K单批2262.544μs对1687.266μs，仍有明显差距。

编译与运行数据写在外层runs；完整构建的300秒硬上限通过suite入口执行：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.suite --cases mha4k gqa8k --policy block-max --load3d --output runs/fa-block-max --compile-timeout 300
```
