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
