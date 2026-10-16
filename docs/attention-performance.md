# 新 DSL FA：首个设备基线与差距

2026-10-16。默认 dense f16 已贯通源码→IR→ASC→CANN→设备，但**性能验收失败**。
本页记录当前基线，不把独立参考或旧FA结果算成新编译器成绩。

## 测量合同

BNSD，B1/H8/Hkv8/Sq=Sk4096/D128/f16，BQ128/BK1024/CK128，VR8，20组。
物理6映射逻辑0；每个进程先检查所有卡的健康、进程、HBM和利用率。
三后端交替运行三批，每批5次warmup、30次计时。下表是msprof的device task时间，
排除host提交间隙。所有后端使用相同Q/K/V字节、scale和数值阈值；每head每query tile
抽样6行对照CPU oracle，并检查全输出有限、输入不变和全部保护区。该大例不是全输出oracle。

复用相同allocation、不flush。实际IO+workspace约64.5MiB，小于本卡192MiB L2；
不能因为序列长度4096就称为HBM带宽测试。小例的全输出证据另见[状态页](refactor-progress.md)。

| 后端 | 三批p50，µs | 结论 |
|---|---|---|
| 新DSL生成代码 | 1144.150 / 1140.392 / 1139.072 | 相对ACLNN为2.230倍，未通过≤1.05目标 |
| 固定政策独立AscendC | 1190.350 / 1190.400 / 1190.490 | 当前生成代码约为其0.958倍；不代表达到专家最佳政策 |
| 安装版ACLNN | 510.890 / 512.190 / 511.370 | `aclnnFlashAttentionScore`，BNSD、keepProb1、innerPrecise0、sparseMode0 |

证据：外层`runs/refactor/query-boundaries/performance.json`和`profiles/trial-{0,1,2}/`，
包含实际ASC/库hash、编译选项、输入hash、CSV及CSV hash、硬件配置/频率快照。
CANN为9.1，dav-2201，-O2。generated ASC SHA为
`ae3e169c7ebf1a34f8c1f736b3d673b0258198eabdeeebffe93ffb560a596f12`；
独立AscendC SHA为`497af23387f1a44196e314d9e6e50882dcd9eac8ad80ebdfe7fceaf9651e03fa`。
ACLNN记录安装libopapi身份；源码checkout不等于该二进制的构建提交。

一个profiling进程在preflight发现物理6尚有进程/利用率，未运行kernel便退出；
恢复空闲后在同卡重新检查再继续。拒绝日志保留，没有reset或换卡绕过错误。

## 硬件指标与源码定位

各批pipe活动中位数再取中位数，单位µs。这些活动可以重叠，**不能相加为kernel时长**。

| 指标 | generated | 固定AscendC | ACLNN |
|---|---:|---:|---:|
| AIC MAC | 261.689 | 261.689 | 247.543 |
| AIV Vector | 698.100 | 770.473 | 325.870 |
| AIV Scalar | 209.861 | 295.329 | 181.832 |
| AIC MTE2 | 316.577 | 312.861 | 248.185 |
| AIV MTE2 | 164.570 | 177.064 | 172.042 |

Vector活动是明显差距；也须用时间线检查活动之间的空档和CV重叠，不能从单列直接
断定唯一瓶颈。Cube本身活动相近，当前没有证据支持继续调整GEMM块来解决2.23倍差距。

本卡Ascend910B3有20 AIC/40 AIV、最高AIC频率1800MHz。按“每core每cycle完成
一个16×16×16 b16 FMA fractal、FMA计2 FLOP”的**条件模型**，Cube峰值294.912TFLOP/s。
此shape的QK+PV工作为68.719GFLOP，对应Cube单独理想下界233.017µs；生成代码整体
约60.26TFLOP/s、条件峰值的20.4%。该下界忽略Vector、DMA、启动与等待，不能当延迟预测。
频率快照不是运行期持续频率；平台配置中的未标明单位带宽值不当作实测GB/s。

源码计算的GM请求总量约2.299GB；其请求速率约2016GB/s包含L2命中，不能称为HBM流量。
因此下一步优先减少Vector工作/等待并改善CV重叠，再通过带宽计数判断是否有内存瓶颈。

上游依据是ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
`attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h`：

- 1058附近的Vec2行块、1967附近的SoftmaxFlashV2、2344附近的Div：D128输出更新按64行，
  历史统计量每行8份；当前显式R64/packed merge/BRCB，输出按8行及reciprocal→mul。
- 855附近的Process：Vec1先于Vec2且taskId跨query连续；当前update先于softmax、query为serial。
  当前query派发尾部没有PIPE_ALL，不能把它称为硬件drain。
- 411附近的f16 ND Score/P地址复用、2254附近的O中间态GM写回：当前三个workspace独立、O驻UB。
  上游Matmul内部同步不能仅凭API名称认为与当前显式ready/release逐条等价。

下一步按[计划](refactor-plan.md)一次验证一个通用Vector配方或明确example政策。
FP32 sum顺序允许改变；P舍入、状态更新与最终输出数值合同仍保留。不按FA名称、shape或ASC字符串特判。
