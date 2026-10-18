# KDA prefill 的矩阵乘输入精度

参考为 vllm-ascend `2581a69bcc16ee6af56e72985a15799bb5f61808`，
Triton-Ascend 官方 3.6.0，目标 Ascend910B3。下表的两列是 **dot 两侧输入**，
以 Q/K/V 为 FP16、gate/beta/initial state 为 FP32 为例。

| 阶段 | 左输入 | 右输入 | 源码原因 |
|---|---|---|---|
| Gate chunk cumsum | FP32 | FP32 | FP32 三角掩码与显式转 FP32 的 gate 做 dot；`allow_tf32=False` |
| KKT / QKT 的跨 16×16 块 | FP32 | FP32 | K/Q 与 FP32 `exp2(g-g_anchor)` 相乘后成为 FP32；不是直接对原始 FP16 K/Q 做 dot |
| 三角逆矩阵的块间合并 | FP32 | FP32 | 加载 A 后显式转 FP32；`DOT_PRECISION="ieee"` |
| U = Ai @ cast(beta*V) | FP16 | FP16 | Ai 在 solve 出口转 k.dtype；右输入在 dot 前转 V.dtype |
| W = Ai @ cast(beta*K*exp2(g)) | FP16 | FP16 | Ai 与右输入均为 k.dtype |
| 历史状态投影 W @ Hᵀ | FP32 | FP32 | 加载 W 后显式转 FP32，H 是 FP32 carry；`input_precision="ieee"` |
| 状态增量 KGᵀ @ residual | FP16 | FP16 | KG 已存成 k.dtype，residual 在 dot 前转 k.dtype |
| 输出 QG @ snapshotᵀ | FP16 | FP16 | Q*scale 与再乘 exp2(g) 各有一次转 Q.dtype；snapshot 是 k.dtype |
| 输出 Aqk @ Vnew | FP16 | FP16 | FP32 Aqk 在 dot 前转 Vnew.dtype；`allow_tf32=False` |

同一 16×16 对角块内的 KKT/QKT 使用 FP32 元素乘、exp2 和 `tl.sum`，没有 dot。
输入改为 BF16 时表中的 FP16 随对应 tensor 改成 BF16，FP32 项保持不变。

Triton 3.6.0 的 `default_dot_input_precision` 为 `ieee`。对未修改的
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda` 做离线编译后，ttadapter 中
两次状态投影均保留 `linalg.matmul {input_precision = "ieee"}` 的
`tensor<64x64xf32>` 两侧输入。证据为 `build/kda/prefill-precision/` 中的
audit.json、ttir、ttadapter、CANN 编译产物及版本/源码 hash。
这证明输入类型与 lowering 前的精度合同，不把该中间 IR 当作最终指令审计。

源码位置：参考目录 `vllm_ascend/ops/triton/kda/` 中的
`cumsum.py:150`、`kda.py:679`、`kda.py:969`、`kda.py:1186`、
`solve_tril.py:286`、`chunk_delta_h.py:161`、`chunk_delta_h.py:234`。
FP32 dot 不能仅用一次 FP16/BF16 cast 后的 Cube 乘法替代。
