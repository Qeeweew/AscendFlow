# Flash Attention：nested 示例与验证范围

正常算子入口是 [examples/flash_attention/attention.py](../examples/flash_attention/attention.py)
中的 `@N.prim_func attention_target`。数学主体只有这一份；
[测试 fixture](../tests/fixtures/attention_target.py) 的 nested 入口直接引用该函数。
函数名保留为 `attention_target`，生成的 ASC 符号与已验证产物 ABI 保持一致。

该 nested 程序已通过 DSL、source IR、物理 lowering、自动同步、实际 IR 终检、
ASC、CANN 9.1 / `dav-2201` 构建和下表设备数值检查。这里没有使用手排参考
替代生成代码，也没有绕过公共同步检查。

## 已验证配置

当前记录为 dense、无 mask 的 forward，BNSD、f16 输入/输出，
`B=1, HQ=HKV=8, D=128`，`BQ=128, BK=1024, CK=128, VR=8, GROUPS=20`。
`Sq`、`Sk` 是 runtime 参数，要求正数且分别整除 BQ、BK；同一 binary 覆盖下表。
空 owner 不派发任务。每组一个 AIC、两个 AIV，query tile 按组循环分配。

| Sq | Sk | 最大绝对误差 |
|---:|---:|---:|
| 128 | 1024 | 3.0517578125e-5 |
| 384 | 3072 | 1.52587890625e-5 |
| 768 | 1024 | 3.0517578125e-5 |
| 512 | 2048 | 3.0517578125e-5 |

四点数值检查均通过 `rtol=0.002, atol=0.0002`，同时检查输入不变及所有分配的
保护区。记录位于工作区 `runs/refactor/query-boundaries/fa-build/device/correctness.json`；
相邻目录保留 source/物理 IR、ASC、验证报告及 CANN 产物。
这些记录不构成设备性能达标证据，也不据此扩展其他长度或 constexpr 变体的验证范围。

迁入 example 后，仅重新运行 frontend/lowering 与保存产物比较：source IR 和
`fa-build/before.xdsl` 均逐字节相同。物理 IR 为 169421 字节，SHA256：
`969513f3a632a8a0aa612468dd1d94dd95d73bf91cc1ebd7a94f8b35d04a3b66`。
迁移没有重新规划或重新运行 CANN。

## 数学与流水政策

外层保留 `serial query → Pipelined kv`，内层保留真实 child pipeline，三个 clock
各自独立。AIC 在同一步先派发新 QK、再派发旧 PV；AIV 先派发旧 output update、
再派发新 softmax。它们是显式源码政策，异步访问、最后读取和 event 复用仍由
公共自动同步及最终实际 IR 检查证明。query 内不插入全局 barrier，最终 actor
返回才结束整个 lifetime。

每个逻辑 BK 的 QK 输出为 FP32，经过 scale、行最大值与在线 max/sum 更新。
denominator 使用未舍入的 p32；P 显式 RN 到 f16，再按 CK128 顺序累计 PV。
O 在 FP32 中独立 rescale/add，最后执行 `1/l`、乘 O、RN 到 f16。
现有例子采用 R64 partial 后 packed merge，广播 scratch 明确可见。这里不要求
FP32 归约与 BLAS 位级相同，也没有将 softmax 缩小到每个 CK。

QK child 的实际派发顺序为 load/MAD/operand/FIX，PV child 为
load/MAD/operand；后者循环外只做一次 FIX。这保持现有 L0 operand 槽位与
已验证执行政策，不以增加 buffer 或重排算法代替同步。

## 显式存储与调用合同

Q、K、V 按 BNSD 直接读取，不需要预转置或 GM packing。调用方除 Q/K/V/O 外，
还须提供三个互相独立、可写且未初始化的 GM workspace：

| 参数 | shape | dtype | 默认每组字节数 |
|---|---|---|---:|
| Score | `(GROUPS, 2, BQ, BK)` | f32 | 1048576 |
| P | `(GROUPS, 2, BQ, BK)` | f16 | 524288 |
| PV | `(GROUPS, 2, BQ, D)` | f32 | 131072 |

workspace 只随工作组数与固定 tile 增长，不是完整 Sq×Sk 中间矩阵。对未初始化
workspace 的读取必须来自真实 writer。L0C 为两个 128×128 FP32 平面，QK/PV
真实共用；每个 AIV 的 m/l/O 常驻其自己的 UB，alpha 双槽。物理访问和双 AIV
贡献、跨 query 内存关系、binary event 与 cross-counter lifetime 均进入最终验证。

## 限制与测试设计变体

设备数值验证目前只按上表配置声明。bf16、GQA/MQA、其他 B/H/GROUPS 或 tile
specialization 没有由这些记录获得设备支持标记。causal、mask、非整 tile 的 tail
和空序列不属于本例当前合同。

continuous 仍是**不支持的测试设计 policy**：fixture 从 example 的同一 AST 主体
派生 `Pipelined task → query=task//kv_tiles, kv=task%kv_tiles`，没有另一份数学主体。
它的 runtime-cycle first/last 谓词不能当作实际 KV parent 的 first/last，lowering
会明确拒绝。有限激活表只用于源码派发检查，不是生产执行器或硬件证明。

本例不声称逐条复刻 ops-transformer 的 arch22 Matmul 实现；其 Process 中 Vec1
先于 Vec2，与本例的 update-before-softmax 不同。独立手排参考与历史性能分析见
[attention-reference.md](attention-reference.md)，不能代替本例的生成代码记录。

## 构建

从工作区根目录使用通用 CLI，读取正常 example 函数：

```bash
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli \
  --input compiler/examples/flash_attention/attention.py \
  --entry attention_target --output runs/flash-attention/nested --timeout 300
```

该命令生成 source/物理 IR、同步计划、验证报告与 ASC。追加 `--compile --shared`
执行 CANN 离线构建并生成 shared launcher；它不会运行设备。设备运行前仍须检查
所有卡健康与占用，并显式选择健康空闲物理卡映射到逻辑 0。
