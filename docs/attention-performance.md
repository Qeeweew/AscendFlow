# 新 DSL FA：block8归约的设备结果与剩余差距

2026-10-16。默认 dense f16 已贯通源码→IR→ASC→CANN→设备，但**性能验收失败**。
本页记录当前基线，不把独立参考或旧FA结果算成新编译器成绩。

## 测量合同

BNSD，B1/H8/Hkv8/Sq=Sk4096/D128/f16，BQ128/BK1024/CK128，VR8，20组。
物理6映射逻辑0；每个进程先检查所有卡的健康、进程、HBM和利用率。
旧R64、新block8、ACLNN和独立AscendC交替运行三批，每批5次warmup、30次计时。
下表是msprof的device task时间，
排除host提交间隙。所有后端使用相同Q/K/V字节、scale和数值阈值；每head每query tile
抽样6行对照CPU oracle，并检查全输出有限、输入不变和全部保护区。该大例不是全输出oracle。

复用相同allocation、不flush。实际IO+workspace约64.5MiB，小于本卡192MiB L2；
不能因为序列长度4096就称为HBM带宽测试。小例的全输出证据另见[状态页](refactor-progress.md)。

| 后端 | 三批p50，µs | 结论 |
|---|---|---|
| 当前DSL block8 | 826.132 / 825.440 / 824.870 | 相对ACLNN为1.612倍，仍未通过≤1.05目标 |
| 此前DSL R64库，本轮重测 | 1141.090 / 1143.700 / 1144.388 | block8降低27.8%延迟 |
| 固定政策独立AscendC | 1194.272 / 1196.048 / 1194.250 | 当前生成代码约为其0.691倍；不代表达到专家最佳政策 |
| 安装版ACLNN | 512.210 / 511.990 / 513.140 | `aclnnFlashAttentionScore`，BNSD、keepProb1、innerPrecise0、sparseMode0 |

证据：外层`runs/refactor/compile-vector/performance.json`和`profiles/trial-{0,1,2}/`，
包含实际ASC/库hash、编译选项、输入hash、CSV及CSV hash、硬件配置/频率快照。
CANN为9.1，dav-2201，-O2。当前generated ASC SHA为
`be2ecac39b1182f48f443367611c8db18af61f66265a574dd1162a4015a58389`；
独立AscendC SHA为`497af23387f1a44196e314d9e6e50882dcd9eac8ad80ebdfe7fceaf9651e03fa`。
ACLNN记录安装libopapi身份；源码checkout不等于该二进制的构建提交。

本轮无设备错误或reset。历史R64首次测试及一次健康preflight拒绝保留在
`runs/refactor/query-boundaries/`，不混入当前三批结果。8K同一binary采样正确性通过；
其约3167µs ACL stream间隔只是探索性记录，不构成8K三批device task性能结论。

本次唯一算子变化是通用typed `block8_then_packed_merge`顺序：8×1024先到8×128，
再到8×16，最后packed merge。CANN9.1的2201 `softmax_flashv2_block_reduce_impl.h`
同样使用分层block归约，本实现使用独立scratch平面，未照搬其收缩原地写法。
显式scratch从512B增至4608B，每AIV总UB由178304B增至182400B。
最终ASC的搬运、MAD及完整event语句（包含条件）与旧版逐字节相同；
FP32归约树改变，P/O舍入、状态更新、CV排步及GM请求不变。捕获地址`/1`也已化简。

## 硬件指标与源码定位

各批pipe活动中位数再取中位数，单位µs。这些活动可以重叠，**不能相加为kernel时长**。

| 指标 | block8 | R64重测 | 固定AscendC | ACLNN |
|---|---:|---:|---:|---:|
| AIC MAC | 261.689 | 261.689 | 261.689 | 247.573 |
| AIV Vector | 380.455 | 698.106 | 770.473 | 325.871 |
| AIV Scalar | 147.866 | 209.877 | 295.876 | 181.621 |
| AIC MTE2 | 322.294 | 318.734 | 315.075 | 248.382 |
| AIV MTE2 | 172.241 | 165.548 | 178.118 | 171.847 |

Vector活动下降约45.5%，而Cube活动不变，与这次变换的范围一致。剩余整体延迟差距
仍约313µs，不能把各pipe差值相加归因，也不能仅凭Scalar活动下降声称指令数减少。
下一步需要检查CV时间线、ready等待与输出更新排步；当前没有证据支持继续调整GEMM块。

本卡Ascend910B3有20 AIC/40 AIV、最高AIC频率1800MHz。按“每core每cycle完成
一个16×16×16 b16 FMA fractal、FMA计2 FLOP”的**条件模型**，Cube峰值294.912TFLOP/s。
此shape的QK+PV工作为68.719GFLOP，对应Cube单独理想下界233.017µs；生成代码整体
约83.25TFLOP/s、条件峰值的28.2%。该下界忽略Vector、DMA、启动与等待，不能当延迟预测。
频率快照不是运行期持续频率；平台配置中的未标明单位带宽值不当作实测GB/s。

源码计算的GM请求总量约2.299GB；其请求速率约2785GB/s包含L2命中，不能称为HBM流量。
因此下一步优先减少Vector工作/等待并改善CV重叠，再通过带宽计数判断是否有内存瓶颈。

上游依据是ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
`attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h`：

- 1058附近的Vec2行块、1967附近的SoftmaxFlashV2、2344附近的Div：D128输出更新按64行，
  历史统计量每行8份；当前显式block8/packed merge/BRCB，输出按8行及reciprocal→mul。
- 855附近的Process：Vec1先于Vec2且taskId跨query连续；当前update先于softmax、query为serial。
  当前query派发尾部没有PIPE_ALL，不能把它称为硬件drain。
- 411附近的f16 ND Score/P地址复用、2254附近的O中间态GM写回：当前三个workspace独立、O驻UB。
  上游Matmul内部同步不能仅凭API名称认为与当前显式ready/release逐条等价。

下一步按[计划](refactor-plan.md)一次验证一个通用Vector配方或明确example政策。
FP32 sum顺序允许改变；P舍入、状态更新与最终输出数值合同仍保留。不按FA名称、shape或ASC字符串特判。
