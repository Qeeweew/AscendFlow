# 可运行示例

算法、分块、缓存和排步政策放在 `examples/`；编译器走同一条 DSL→IR→ASC→CANN
路径。示例文件只保留 DSL 定义；构建统一使用编译器 CLI，数值与性能检查放在
`benchmarks/`。先读[快速上手](../docs/guide/quickstart.md)，所有示例的统一检查命令为：

```bash
python -m benchmarks.check_examples --list
python -m benchmarks.check_examples --output build/examples-cann
# 完成全卡检查与显式映射后，再加 --physical <物理卡号> 检查设备结果。
```

所有示例的私有 GM scratch 均使用 `N.alloc_workspace`，runtime 自动分配，各 allocation
起点按 512 B 对齐。固定 worker、tile 与 slot 政策后，容量不随任务数无限增长；教学
attention 也由固定 worker 依次处理 head。跨 kernel 的中间结果保持显式 tensor，其容量
仍随数据规模变化。设备核数查询、容量和生命周期见 [workspace 合同](../docs/guide/workspace.md)。

| 检查名称 | 源码 / DSL 入口 | 主要展示内容 |
|---|---|---|
| `scale` | [vector/scale.py](vector/scale.py) · `scale` | GM→UB→Vector→GM 双缓冲 |
| `grouped-scale` | [vector/grouped.py](vector/grouped.py) · `grouped_scale` | runtime BNSD、循环分组、空 owner |
| `rings` | [vector/independent_rings.py](vector/independent_rings.py) · `independent_rings` | 输入/输出独立的 5/7 槽环 |
| `masks` | [vector/masks.py](vector/masks.py) · `masks` | interval XOR 64-lane periodic + runtime origin、masked更新、compare/where；runner按已发布shape运行 |
| `mask-rows` | [vector/mask_rows.py](vector/mask_rows.py) · `mask_rows` | 单 tile 行/列 mask：`cols<hi` 前缀、`cols!=lo` 补集、`rows>=cols-q` 行前缀；三个独立拷贝输出 |
| `rms` | [normalization/rms_norm.py](normalization/rms_norm.py) · `rms_normalize` | 归约、Scalar 最后读取 |
| `softmax` | [normalization/softmax.py](normalization/softmax.py) · `softmax` | 行统计、广播 scratch、Vector 流水 |
| `batched-gemm` | [gemm/batched.py](gemm/batched.py) · `batched_matmul` | 多级 L1/L0/MAD/FIX 环 |
| `reduction-gemm` | [gemm/reduction.py](gemm/reduction.py) · `reduction_gemm` | runtime K 归约 |
| `hierarchical-gemm` | [gemm/hierarchical.py](gemm/hierarchical.py) · `hierarchical_gemm` | 输出块循环嵌套 K 流水 |
| `dense-gemm` | [gemm/dense.py](gemm/dense.py) · `dense_gemm` | 连续 NT、FP16 输入/FP32 输出 |
| `gemm` | [gemm/matmul.py](gemm/matmul.py) · `gemm` | 连续 NN、FP16/BF16、静态尾块 |
| `flash-attention` | [flash_attention/attention.py](flash_attention/attention.py) · `attention_target` | QK/softmax/PV/update 多级 CV 流水 |
| `simple-attention` | [flash_attention/simple.py](flash_attention/simple.py) · `simple_attention` | 整块 non-causal attention 教学示例，D128、四阶段与两槽 workspace |
| `sparse-mla` | [sparse_attention/gather.py](sparse_attention/gather.py) / [attention.py](sparse_attention/attention.py) | V4.1 decode/prefill、分页 shared-KV、稀疏 TopK、sink 与双 AIV 流水 |
| `scalar-store` | [vector/scalar_store.py](vector/scalar_store.py) · `scalar_store` | Scalar capture/store 与精确 FP32 检查 |
| `kda-recurrent` | [kda/recurrent.py](kda/recurrent.py) · `recurrent_kda` | token 顺序递推、FP32 carry 与独立快照 |
| `kda-decode` | [kda/decode.py](kda/decode.py) · `decode_kda` | 单 token 解码、三阶段 head 流水 |
| `kda-prefill` | [KDA prefill](../docs/operators/kda-prefill.md) · prepare / scan / output 三个入口 | chunk 预处理、Gram/块逆、递推及输出 |

[gemm/schedule.py](gemm/schedule.py) 是 GEMM 的 inline L2 政策 helper，随 GEMM 验证。
`independent_rings` 用于测试互质环的组合，不是推荐的性能参数。

单个示例可用 CLI 构建，例如：

```bash
python -m ascendflow --input examples/vector/scale.py --entry scale --shared --timeout 300 --output build/scale
```

设备运行器可按[工具目录](../benchmarks/README.md)查找。普通 GEMM 的尺寸、dtype、
对齐与数值条件见[GEMM 合同](../docs/operators/gemm.md)；dense NT 有单独的
[整块合同](../docs/operators/dense-gemm.md)。FA 的 runtime Sq/Sk、workspace、
数值边界与限制见[FA 合同](../docs/operators/flash-attention.md)。
V4.1 的计算入口、缓存与索引条件、数值和接入边界见
[Sparse MLA 合同](../docs/operators/sparse-mla.md)。

统一清单包含 KDA 的小规模数值检查；大规模构建和 Triton 对照使用专用 runner：
[decode](../docs/operators/kda.md) 提供 batch 1–64 的 Triton 对照方法；
[prefill](../docs/operators/kda-prefill.md) 包括 IEEE FP32 Cube、块逆及混合流水。

逐项编译/运行证据保存在 `check_examples` 的输出目录。默认 Flash Attention 与
GEMM normal 的多 shape ACLNN 对照及 TFLOPS 采集见[测试指南](../docs/guide/testing.md)。
动态 `valid=` prefix 地址等未支持形式保留在拒绝测试中，不作为 runnable example；
runner 接受某个参数不代表该变体已经设备验证。

示例遵守当前[动态访问与标量合同](../docs/guide/dsl.md)：`load_index` 默认保留源
整数 dtype 的完整有符号范围，普通间接地址不需要逐个填写 `minimum/maximum`。
实际访问合法、metadata 单调及状态槽互斥等仍由调用方保证；决定通知时钟的循环
metadata 可以声明必要的有限范围。静态非法原式、物理容量和同步资源仍由编译器检查。

整数元数据也可用普通 `copy` 批量预取到 UB，再用 `load_index` 读取；DMA 的对齐、
初始化、ready 和最后 reader 退休合同保持不变。具体索引、页表缓存与双槽 gather 流水见
[sparse attention](sparse_attention/attention.py) 和 [gather](sparse_attention/gather.py)。
其他算子按实际复用选择缓存，KDA 的 GM int64 ABI 保持原有合同。
