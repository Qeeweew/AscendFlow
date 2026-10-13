# 首轮源码依据

2026-10-13 在新工作区读取 AGENTS、三份工作区设计/审计报告、910B compiler/
InsertSync 文档、FA 910B 文档和 materials manifest。开始时 compiler 不存在，
外层 src/tests/examples 只有 .gitkeep，已保留；未创建外层 Git。

| 参考仓库 | 实际 HEAD，开始时工作区干净 |
|---|---|
| ops-transformer | `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e` |
| CATLASS | `698379e0d4b1b94d12728e7181df2cf83a958e8f` |
| ops-nn | `64b41ad63a5073bc711ba03802498bfe8bdb77e5` |
| TileLang snapshot | `6ba65d204c937fcb1c43b1a22b32979506595655` |
| xDSL 0.73.0 snapshot | `aa84c3f4a80ac14a3278d69e58b3db80e9a05b18` |
| TVM TIRx snapshot | `d984d10c48c403d061ec36e544ab474a1819bc9f` |

ops-transformer 与报告的固定 `98f82c34...` 不同；历史固定链接不是当前 checkout。
本轮没有 pull 或修改参考仓库，也没有访问原 TileLang 工作区/容器。

直接核对的关键点：

| 来源（相对于工作区 reference） | 对实现的约束 |
|---|---|
| `tilelang/src/ascend/transform/insert_sync.cc`, `HasImplicitSamePipeOrder` | 最初 S/M 顺序设计输入；短 MAD 的无条件 M 假设已被后续设备反例修正；V/MTE/FIX 不能直接当完成边 |
| `tilelang/docs/compiler_internals/ascend_local_events.md` | 二值 flag、八 ID bank、旧 Wait→下一 Set、跨核 Wait 的 dispatch 语义 |
| `ops-transformer/moe/moe_finalize_routing/op_kernel/moe_finalize_routing_fp_db.h`, 325–357 | Add 读取输入后 release，后续 Scale/输出计算仍继续；V RAW barrier 必须保留 |
| `catlass/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp`, 278–375 | L1/MTE1、L0/MAD、L0C/FIX 的独立最后读取和 release |
| `catlass/include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp`, 32–62 | zN→zZ 使用 `ifTranspose=false`；用非对称设备输入修正了首个 emitter |
| `ops-transformer/posembedding/rotary_position_embedding/op_kernel/rotate_interleaved_split_bsn_pad.h` | Gather offsets 与原地 scratch 具有真实 alias；同 SSA 名称不足以描述访问 |
| `ops-nn/norm/rms_norm/op_kernel/rms_norm_merge_n.h` | Vector/reduce/scratch 与 gamma 生命周期，不能围绕 GEMM 单独设计效果 |
| `ops-transformer/attention/recurrent_gated_delta_rule/op_kernel/arch22/recurrent_gated_delta_rule_arch22.h`, 265–300 | 多次原地状态读写及 reduction；carry 与 snapshot 数值边界分开 |
| `ops-transformer/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h` | 稀疏/paged DMA、L1 multiple readers；真实任务代次和 metadata 不能由词法循环猜测 |
| `ops-transformer/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_post.h`, 225–254 | 完整行 Abs/ReduceMax 再产生 scale 和 INT8；禁止移动量化舍入边界 |
| 工作区 FA 910B 文档 | query/state/task/notification 时钟不同；BK 与 P 舍入点属于算法 |

ASC API 以安装的 `/usr/local/Ascend/cann-9.1.0/asc` 为准，直接检查了 2201 copy、
MAD、Fixpipe、sync 和 Vector count 的实现。普通 GM/UB DMA 长度除以 32，尾部
不会自动复制，因此 v0.1 明确拒绝非 32 字节的 DMA；没有悄悄 overread 或 padding。
Muls/Adds 的公开名字是 `asc_mul_scalar` / `asc_add_scalar`。
Fixpipe wrapper 不设置 ND_PARA，产品在明确的一次初始化契约中设置它。

原生操作的 source read retirement 与 destination write done 默认同在其可信
完成点；这里没有推断指令内部的更早读取完成。Set 发射/生效/后续 source work
必须分开，设备反例已经说明单靠静态图中一个错误的“fence”前提会误判。

后续实际 HEAD 仍为上述记录，参考工作树未改变。CATLASS
`include/catlass/gemm/tile/tile_mmad.hpp:71–74` 对
`(m/16)*(n/16)<10` 显式加 M barrier。连续小 tile 设备反例暴露了此前统一
采用 M completion 的问题；最终生成器从真实 C RAW/WAW 推导局部 M 完成，
保留其他 C 槽的交错。具体数值反例和修正产物见 `hierarchical-gemm.md`。

已有设计建议按能力逐步验证。跨核 GM handoff、动态 segmented task、GDN state
ownership、MoE global phase、communication completion 都尚未实现；其审计结论
只用于检验接口和拒绝边界，没有登记为空接口或宣称已支持。
