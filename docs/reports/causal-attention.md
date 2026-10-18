# Causal Flash Attention 验收

2026-10-18。实现和输入条件见[算子合同](../operators/flash-attention.md)。
源码由当前 AscendFlow DSL 经唯一同步链生成 ASC/CANN，没有手写 kernel 替代。

## 比较合同

Ascend910B3，物理卡 2 → 逻辑卡 0，CANN 9.1.0，dav-2201，`-O2`。
B1/Hq8/Hkv8/D128，BNSD，FP16 输入输出，无 dropout，FP32 scale，左上 `j<=i`。
对照本机 `aclnnFlashAttentionScore`，keepProb=1、innerPrecise=0、sparseMode=2，
使用 2048² 压缩 bool mask。ops-transformer arch22 源码用于算法研究；安装库的
实际 hash 单独记录，不将研究源码版本冒充安装二进制版本。

双方使用相同输入 hash、采样行、容差和物理卡。每个 shape/backend 各一次 msprof，
预热 1 次、统计 3 次，取实际设备 task 时长中位数；多 task 合计，排除预热。
进程内复用 allocation、不 flush 缓存，host stream interval 不作为 kernel time。
有效 FLOPs 为 `4*B*Hq*D*sum_i min(i+1,Sk)`，不计算未来 key 或 softmax 的 FLOPs。

## 设备与性能记录

最新成对 1K/2K/4K/8K 等长采集位于 `build/causal/acceptance-square/`；性能验收仍以
其最终 `results.json` 为准，不能用较早的 tiling probe 宣称全部达到 ACLNN。

中等分块的完整输出验证使用 BK512/VR16，Sq:Sk 为 256:1024、768:3072、
2048:1024，standard/uniform/sharp 三种输入。九项均通过；standard 最大绝对误差
0.000244140625，uniform 为 0.00006103515625，sharp 为 0.0009765625，均满足
逐元素 rtol=0.002、atol=0.0002。完整输出有限、输入不变及 allocation guard 同时检查。
记录在 `build/causal/wider-mid-validation/`，对应产物在 `build/causal/wider-mid/`。

采样数值检查不称全输出 oracle，当前配置的通过不外推其他 head/dtype、任意 mask、
尾块或 KV cache offset。单次 profile 的接近 1 差异是本次观测，不是任意设备/输入的
普遍性能保证，也不等于历史三批严格验收。

## 复现

检查全卡健康、进程、显存和利用率后，显式映射健康空闲卡到逻辑 0：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1
python -m benchmarks.profile_defaults --causal --require-no-slower --output build/causal-comparison --physical 2
```

三组政策各用单次 300 秒进程组硬期限，覆盖 Python 构造、分析、ASC 和 CANN。
总表区分数值状态和性能状态；任一 shape 慢于 ACLNN 时严格命令返回非零。
每个产物保存 DSL/IR/ASC/库与编译 manifest，每个 profile 保存 CSV、设备、输入和库身份。
构建失败、超时或设备错误立即停止，不继续后续候选或换卡绕过。
