# 当前执行顺序：先解决 Attention 的 GM→L1 重复读取

2026-10-16。最新用户明确优先GM→L1，不以L1→L0指令数判定瓶颈。
已验证状态见[状态页](refactor-progress.md)，原始指标和口径见[性能页](attention-performance.md)。
算子政策只写在examples；不恢复旧task DSL，不增加算子名、shape或ASC字符串特判。

## 1. 使显式K/V复用通过同一编译路径

唯一example的BQ256/BM128政策在一个KV epoch内处理相邻两个native Q行块，K/V在L1
只加载一次、被两次MAD读取。矩阵原生尺寸仍128；Q/P独立，双AIV各处理128行。
softmax原地使用score_in双槽，UB省32KiB以容纳更多O状态。Sq/Sk仍是runtime参数，
当前政策明确要求Sq是256倍数；不引入odd-pair的隐式padding或fallback。

必须核对三类实际完成关系：

1. score_in的下一次MTE2覆盖等最后cast读取，不能沿用旧Muls后的release。
2. qa/pa在两个part间覆盖等旧MAD最后读；kb/vb下一chunk覆盖等第二个MAD最后读。
3. 下一QK覆盖两个共享L0C plane前，两个PV FIX的读取均已完成。

前端/物理lowering已通过；序列边界的重复endpoint工作已通过共享完整semantic pattern修复，
仍逐物理root/byte区间实例化义务。key保留part、固定count、完整CountDomain、
Access/ReadSpan及definitions；孔洞不合并，工作预算继续计实际输出。
当前卡点是固定child投影对1042个端点使用平方预估，先构造完整关系表、再做传递约简。
改为从已有origin覆盖位集直接生成闭包等价的稀疏图；输入/输出port独立，同实际里程碑
的别名仍保留双向可达，admission不能反向进入旧prefix。不提高阈值、不跨构建缓存。

通过后一次构建必须完成实际IR终检、ASC、CANN/shared，总期限300秒；到时立即停止整个
候选序列。审查最终ASC的K/V载入次数、真实release位置、buffer和drain。

## 2. 用设备证明GM请求和性能

先短域全输出，再4K/8K采样，输入/全输出有限/保护区均检查。同一新binary支持多个长度。
`msprof op Memory`全核GM→L1目标为源码预算520MiB（当前776MiB），不能只看L2命中率；
这不是对ACLNN544.3125MiB实际tiling的推定。

复测当前BQ128、BQ256、独立AscendC、ACLNN，三批各5次warmup+30次device task计时，
绑定真实DSL/IR/ASC/库/选项、输入与设备。工作组数/缓存政策固定，不做参数穷举。
性能超过ACLNN5%仍未通过。请求下降但时间不改善时，保留反证并重新看critical path。

随后再按PMU证据处理UB bank冲突和Vector政策；L1→L0的Load3D单独排后，不能将
减少指令条数替代减少GM读取。统计量布局、Vec2行块及首KV省略计算均须保持数值合同。
FP32 sum顺序可变，P/O舍入和递推状态仍保持。

## 3. 继续削减编译算法成本

默认FA源码→ASC141.2秒，当前主要是planning和final verifier；两者各自独立重建事实。
不以C++重写、跨构建缓存或提高预算掩盖重复工作。

- 本轮先修复序列transfer按字节片段重复组合。
- 符号传播已跳过完全不相交origin的包含扫描；重叠路径仍保留原判断。
- 下一独立优化是同一flow/node/point/完整domain的coverage查询按origin mask合批读取，
  将q次扫描k条事实改为一次扫描及结果分发；不能将EVEN/bit1与ODD/bit2合成ALL/两bit。
- 稳定后再做冷进程多次计时；常用kernel目标p50≤1秒/max≤2秒，FA≤5秒/≤10秒。

## 4. 变体与通用能力

bf16 GQA已CANN构建，设备资格待GM→L1主线之后继续；MQA/其他dtype逐项验证。
continuous须先解决runtime-cycle首末和跨query denominator ownership；causal/tails/decode
须真实有效域与DMA范围；varlen/paged/sparse/persistent须metadata epoch及动态ownership。
不因为runner接受参数或前端能表达就声明支持。

通用event借用仅支持已证明的固定child空档及distance1单ID parent；保留实际IR verifier。
纯整数标准pass已接入，其他ASC冗余按公共规则继续处理。算法抽象继续用GEMM、normalization、
RoPE/MoE、动态GMM/FFN和GDN检验，不能为统一模型改变有序状态更新。

多Agent按根AGENTS一文件一写入者；主Agent集成并统一排队设备、独占完整构建计时。
测试只跑受影响层，常规pytest不重复长编译FA；环境/运行数据不进产品Git，不打包或推送。
