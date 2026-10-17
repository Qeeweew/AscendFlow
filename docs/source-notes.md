# 源码依据与硬件合同边界

2026-10-17 再次核对 CATLASS、ops-transformer、ops-nn 的 HEAD 和工作树，均与下表一致且无修改，没有 pull。本轮已执行当前编译器的 CANN 离线构建，未执行设备试验；产物见状态页。以下仓库不是编译器构建依赖；报告里的历史固定链接可能与checkout不同。

| 参考 | 实际HEAD |
|---|---|
| TileLang snapshot | `6ba65d204c937fcb1c43b1a22b32979506595655` |
| xDSL | `aa84c3f4a80ac14a3278d69e58b3db80e9a05b18` |
| ops-transformer | `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e` |
| CATLASS | `698379e0d4b1b94d12728e7181df2cf83a958e8f` |
| ops-nn | `64b41ad63a5073bc711ba03802498bfe8bdb77e5` |

本轮检查这些仓库没有已跟踪文件修改；未清理或覆盖其未跟踪内容。目标为dav-2201，实际接口以安装的CANN9.1.0为准，不能套用3510实现。

## 必须保留的来源约束

下表是已有源码研究的保留项，不声称本轮重新完成硬件证明。路径相对于工作区reference或CANN安装目录。

| 来源 | 约束 |
|---|---|
| `tilelang/docs/compiler_internals/ascend_910b.md`、`ascend_910b_insert_sync.md`、`ascend_local_events.md` | issue/完成/通知消费分开；借鉴合同，不迁移整套闭包与重试 |
| `catlass/include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp` | 最后L1→L0读可释放L1，MAD/FIX各有独立生命周期 |
| `catlass/include/catlass/gemm/tile/tile_mmad.hpp` | 小tile显式M barrier反例，不能无条件认为同M流水已完成 |
| `ops-transformer/moe/moe_finalize_routing/op_kernel/moe_finalize_routing_fp_db.h` | 输入Add读后释放，后续Scale/输出仍执行；不要等待整个算法块 |
| `ops-transformer/posembedding/rotary_position_embedding/op_kernel/rotate_interleaved_split_bsn_pad.h` | gather offset、原地scratch及alias必须可见 |
| `ops-nn/norm/rms_norm/op_kernel/rms_norm_merge_n.h` | reduction、scratch、Scalar读与gamma生命周期，不能只围绕GEMM建模 |
| `ops-transformer/attention/recurrent_gated_delta_rule/op_kernel/arch22/recurrent_gated_delta_rule_arch22.h` | 递推carry、snapshot、原地更新与数值边界独立 |
| `ops-transformer/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h` | 稀疏/paged metadata与多reader；任务owner不能由词法位置猜测 |
| `ops-transformer/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_post.h` | 全行scale/reduce及量化边界，不为统一表示移动舍入 |
| `ops-transformer/attention/flash_attention_score/` | 核间通信tile与核内tile不同；安装ACLNN版本不等于上游checkout |

## dav-2201的可信前提

- 各pipe的隐式顺序、native end、累计退休以及Set捕获使用真实原语合同；不能从SSA或源码顺序推断。
- 本地二值flag按bank分配有限ID，消费先于复用；AscendC TPipe本身可能预置/占用event，手写参考与裸ASC分别记录。
- 安装 CANN 的 `compiler/asc/impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_sync_block_arrive_impl.h` 中，`GetfftsConfig` 使用 mode=2 和 `flag_id & 0xf`，因此只分配 0..15，禁止 overflow alias。
- mode2跨核通知、双AIV贡献及GM可见性属于具体硬件协议。`asc_sync_block_wait`在2201调用`wait_flag_dev`，阻塞控制器后续派发，不能按某一MTE queue wait处理。
- GM→UB/UB→GM的burst/gap单位、L1→L0转置和Fixpipe配置以2201头文件为准；原生recipe精确effects必须涵盖内部展开。
- ready、reader release、event credit、最终drain不是同一种里程碑。仅在原语有明确合同的情况下使用内部早读完成；默认不猜指令内部退休时间。
- kernel初始化、控制寄存器状态及返回drain保留已有必要合同；局部循环边界不引入PIPE_ALL。

模型和测试共用这些硬件前提，不能反过来把图验证当成它们的独立证明。新原语或新设备行为仍需源码、CANN和必要设备证据。

## 理论性能的证据要求

平台配置、实际芯片核心数、频率和数据类型决定理论算力条件；字节流量分清GM请求、L2命中和HBM实流量。配置中的未知rate单位和HBM MHz不能直接当GB/s；不同profiler replay组的时间不能相加分解延迟。

当前重构不运行性能调优。恢复后的AscendC/ACLNN、msprof和理论下界口径统一见[验证](validation.md)，不在每份算子文档复制一套指标定义。
