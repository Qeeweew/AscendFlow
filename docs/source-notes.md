# 首轮源码依据

后续 R1 AST normalization 的新增依据：安装9.1.0的
`asc/impl/c_api/instr_impl/npu_arch_2201/vector_compute_impl/asc_repeat_reduce_sum_impl.h`
以 count mode 调用 `vcadd`；AscendC `ReduceRepeat<ReduceType::SUM>` 在2201路径
调用 WholeReduceSumImpl。`kernel_tensor_impl.h:135–195` 的 f32 GetValue 为
Scalar 读取。独立参考以公开 AscendC API 和手排 flags 编写，没有从 PlanIR
复制生成代码。ops-nn HEAD 仍为下表版本，`rms_norm_merge_n.h:256–265` 保持
Sqrt/Duplicate/Div 数值顺序。具体条件、profiling 和局限见
[AST normalization](ast-normalization.md)。

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

已有设计建议按能力逐步验证。动态 segmented task、动态 GDN state ownership、MoE
global phase、communication completion 都尚未实现；其审计结论
只用于检验接口和拒绝边界，没有登记为空接口或宣称已支持。

## 静态混合核的实际契约

本阶段再次确认上述 ops-transformer/CATLASS HEAD 与干净工作树，没有 pull。
安装的 9.1.0 2201 头文件是 emitter 的实际契约来源：

| 来源 | 已采用并检验的行为 |
|---|---|
| `asc/impl/basic_api/dav_c220/kernel_operator_sys_var_impl.h`, `GetBlockIdxImpl` | raw `get_block_idx()` 是 group index；AscendC 的复合 AIV index 才是 `group*2+subblock`，不能把 raw ASC index 再除2 |
| `asc/include/c_api/sync/sync.h` 和 2201 `asc_sync_block_arrive_impl.h` | arrive 固定 mode2，flag 编码4位；wait 是 `wait_flag_dev(id)`，忽略 pipe 参数 |
| `tilelang/docs/compiler_internals/ascend_local_events.md` | cross Wait 阻止后续所有 pipe dispatch，已发射的工作可以继续；并非 MTE2 queue Wait |
| `catlass/include/catlass/arch/cross_core_sync.hpp` | 连续未消费通知最多15；反向计数 ack 与真实 GM read retirement 不相同；本实现独占0/1，只允许0…7 |
| `ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul_quant_mixcore.h`, 170–178、210–215、244–248 | AIC FIX 发布、两 AIV MTE2 返回、下一 scratch overwrite 前 wait，以及最终剩余 credit drain |
| 2201 `vector_datamove_impl/asc_copy_{gm2ub,ub2gm}_impl.h` | 二维原生 DMA 的 `n_burst,len_burst,src_gap,dst_gap`；len/gap 单位32字节，方向不能照抄3510头文件 |

`__mix__(1,2)` 通过 Bisheng 的共享宿主启动路径调用，产物引用
`aclrtGetHardwareSyncAddr`。未手工覆盖 FFTS base；本 ABI 已由实际多组、槽复用
及多次 kernel 调用验证。仅离线生成混合 ELF 不表示已有其他 ELF 启动 ABI 支持。
通知从同一源 pipe FIFO 发布，捕获此前 source completion；它不阻止后续源任务。
mode2 ready 的 FIX 完成/GM 可见性和双 AIV 返回的参与者契约，在上述上游算法
和安装接口基础上经设备样本验证。模型与交错检查仍共享这些可信机器前提，
不能反过来把图验证或有限枚举当成硬件契约的独立证明。

## 静态 GDN 的指令边界

本阶段 ops-transformer 实际 HEAD 为 `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
工作树干净，未 pull。`recurrent_gated_delta_rule_arch22.h` 的 carry、snapshot 与
MulAddDst 是本阶段的算法输入；动态接受 token/state ID 没有被替换成静态能力。
安装的 `asc/impl/adv_api/detail/reduce/reduce_sum/reduce_sum_v220_impl.h`
对 R64 使用 WholeReduceSum(vcadd)，避免套用大 R 的 folding 算法。
`kernel_operator_vec_binary_impl.h` 的 MulAddDst 与
`npu_arch_2201/vector_compute_impl/asc_fma_impl.h` 均生成 vmla。
`broadcast_v220_impl.h` 的 BRCB→Copy 保持为两个 task；ASC `asc_brcb` 与
`asc_copy` 直接对应 vbrcb 与 vcopy，其 RAW/barrier 和物理 scratch 已显式建模。
f32 bitwise snapshot 使用 vcopy，保留源位模式，没有以加0代替复制。
当前只实现静态 f32、R64 的递推子集，具体接口、拒绝范围与设备证据见
[recurrent-gdn.md](recurrent-gdn.md)。

## Flash Attention 与硬件性能分析

2026-10-14 延续上述参考 HEAD，没有 pull；安装的 CANN9.1头文件仍是最终
native contract 来源。没有使用原 TileLang 的构建、虚拟环境、缓存或容器。

| 已核对来源 | 产品实现/修正 |
|---|---|
| `tilelang/examples/ascend/flash_attention/910b.md`、`910b.py`、`persistent_910b.py` | FP32 online carry、未舍入 denominator、P operand RN；逻辑BK属于算法。query/state/native时钟独立，不迁移整个TileLang pass体系 |
| `ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md`, 35–54、182–207 | A2交接经过GM/L2；CV通信基本块和核内计算块不同。当前产品BK128仍未复现该源码的BK1024通信组织，明确保留性能差距 |
| 2201 `cube_datamove_impl/asc_copy_l12l0b_2d_impl.h` | classic wrapper把transpose固定false。NN V使用审查后的 `load_cbuf_to_cb(..., true, inc)`；精确分块映射及f16/bf16矩形设备验证，不增加GM转置 |
| `asc/include/c_api/cube_datamove/cube_datamove.h` 的 `asc_copy_l12l0b_transpose` | 这个相似名字的接口属于3510路径，不能套用到2201。2201另有不同footprint的 `*_trans` 指令，本实现没有混用 |
| 2201 `vector_compute_impl/{asc_repeat_reduce_sum_impl.h,asc_repeat_reduce_max_impl.h,asc_brcb_impl.h,vconv}` | 普通异步cast、repeat-stride归约和行广播均展开为原生recipe；没有使用 `_sync` 变体隐藏同步 |
| `/usr/local/Ascend/cann-9.1.0/aarch64-linux/data/platform_config/Ascend910B3.ini` | 20 AIC/40 AIV、192 MiB L2及各bank容量；原始memory rates单位未核实，不冒充GB/s |
| `/usr/local/dcmi/dcmi_interface_api.h`, `dcmi_get_device_frequency` | 枚举6/7/9/12查询HBM、当前/最大AIC、当前AIV。未知AIV状态-8255保留；HBM MHz不能直接推导带宽 |
| msprof `op_summary*.csv`、device `FreqParse` | 设备task时间、重叠的pipe activity比例和运行频率样本，与源码/IR/库/CSV hash绑定；不能仅凭ratio判定因果瓶颈 |

Flash Attention 的双向三环共享同一物理访问/生命周期分析；冗余credit删除后
重检全部跨核RAW/WAR/WAW、local event、计数器及最终drain。逻辑phase描述算法
依赖，stage/issue order描述实际派发，二者不能替代异步完成关系。

同卡大规模4K f16 MHA和8K bf16 GQA已经离线编译、采样数值验证及msprof。
条件Cube峰值模型明确声明8192 ops/core/cycle和1800 MHz假设；没有得到HBM/L2
持续带宽实测，也没有宣称达到专家流水性能。具体范围、同步反例和数字见
[Flash Attention](flash-attention.md)。全产品回归288项通过；有界枚举未穷尽的
FA样本仍如实保留 `exhaustive=false`，没有提高预算来暗示证明。
