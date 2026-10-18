# Flash Attention 示例合同

源码：[examples/flash_attention/attention.py](../../examples/flash_attention/attention.py)，入口 `attention_target`。算法与固定排步属于example，不属于编译器内部。编译和设备验证使用[统一验收口径](../development/validation.md)。

## 当前默认配置

- dense、无mask forward，BNSD，f16输入输出，B1/Hq8/Hkv8/D128。
- BQ256/BM128/BK1024/CK128/VR8，20组AIC/双AIV。
- Sq/Sk为runtime参数，满足`Sq>0且Sq%BQ==0`、`Sk>0且Sk%BK==0`；空owner不发任务。
- 每组串行处理分配到的Q块，每个Q块遍历KV块，内部Cube/Vector再分块。这里的Q块不是编译器应特殊识别的控制类型。

| 显式workspace | shape | dtype |
|---|---|---|
| Score | `(GROUPS, 2, BQ, BK)` | f32 |
| P | `(GROUPS, 2, BQ, BK)` | 输入dtype |
| PV | `(GROUPS, 2, BQ, D)` | f32 |

Q/K/V/O与workspace的真实指针区间遵守外部disjoint合同；局部L1/L0/UB以及上述GM环仍按真实访问同步，不能因变量名不同认为无复用。

QK产生FP32 score，online softmax保留FP32 m/l；denominator使用未舍入p32，P显式RN后进入PV。O在FP32 rescale/add，最后归一化并RN输出。FP32求和树允许按误差合同调整；其余算法/舍入边界不自动获得修改授权。

默认AIC先新QK后旧PV、AIV先新softmax后旧update。每个K/V分块服务两个BM128行片，减少重复的GM→L1读取。两个片各占一份L0C；QK子流水先导出旧片，再按load/MAD/operand发射，PV子流水为load/MAD/operand并在外部FIX。编译器保持这些政策、早释放、缓冲复用与事件共享。

两个AIV分别处理128行，仍共用一份runtime SID body。其GM分片对应不同的Cube操作；通用通知合并只在相同激活域和坐标下、Cube发射次序可证明时，选择较晚的累计capture或较早的acquire。两个AIV的真实贡献、原始storage lifetime和有限配对检查均保留。

L1为320KiB，L0A/B各64KiB、L0C128KiB，每AIV UB187264B。Score/P/PV的GM环分别为40/20/5MiB。相对BQ128，workspace增大，K/V每次装入服务的行数翻倍；这项权衡在example声明，编译器没有按FA或shape特判。

输出状态的各行独立rescale合并为一次整片广播和乘法，再逐块加PV。除法的常数1在每个AIV入口填充一次；最后归一化复用广播scratch的前8行。FP32乘、加及除法的舍入边界保持不变。

QK/PV共用L0B的两个32KiB槽，按各自chunk代次选择槽；总容量仍为64KiB。
QK可在MAD读取一槽时装入下一槽，跨QK/PV交接及每槽真实最后reader由同一同步链建立。
默认配对BM片的L0A也在相同64KiB容量中分为两槽，按part选择；QK/PV共用它们。
装入第二片无需等第一片MAD读完同一槽；跨阶段覆盖仍等待对应槽的真实最后reader。
softmax中独立的alpha Exp与矩阵减法按完成依赖排步；P的独立FP16转换提前到
denominator归约之前，归约继续读取原始FP32 work，不读取转换后的P。
这些选择不改变逐行递推、FP32归约树或输出dtype。

最终 O 只在最后一个 KV 块输出。对应行流水采用 import/accumulate 两个 stage，
归一化、RN cast 和 GM 写回在 accumulate 末尾的同一个 LAST 条件内执行；
Vector→MTE3 ready 和输出 UB slot 的最后读取仍由通用同步建立。

## Causal 模式

同一入口用 `CAUSAL=True` 特化，runner 使用 `--causal`。语义为左上对齐
`j <= i`，对角线保留，query 起点为 0；不是带 KV cache offset 的右下对齐。
当 Sq>Sk 时，后面的 query 行可以读取全部 Sk 个 key；Sq<Sk 时跳过未来 KV 块。
输入输出、FP32 online 状态、P/O 的 RN 边界和数值容差与 dense 保持一致。

`causal_policy(Sq, Sk)` 是 example 中显式的 host 分块政策：

| `min(Sq, Sk)` | BK | VR |
|---|---:|---:|
| <2048 | 256 | 16 |
| 2048..8191 | 512 | 16 |
| ≥8192 | 1024 | 8 |

BQ256/BM128/CK128、20 组不变。每份产物内 Sq/Sk 仍为 runtime，要求
`Sq>0, Sq%256==0, Sk>=2*BK, Sk%1024==0`。显式指定其他参数不取得同等资格。
编译器没有按 attention 或 shape 选择政策。

每个 query 块只遍历 `min(Sk/BK, max(2, ceildiv(query_row+BQ, BK)))` 个 KV 块。
保留最少两次迭代使每个父代次都访问两个交接槽；必要时多算一个全 mask KV 块，
不改变输出。去掉该下限会需要父循环条件激活 family 的闭合，当前仍明确拒绝。
PV 的子循环再按实际可见列裁剪，最少保留两个 CK 槽；QK/softmax 保持完整 BK。
读取前缀的物理 footprint 不变，GM 槽由实际最后 reader 释放。

score 缩放后、reduce_max 前调用通用 `N.fill_upper` 将 `j>i` 的 FP32 lane
填为负无穷。全可见块不触碰向量 mask，全不可见块整片填充，部分可见块按行设置 mask；
有填充的路径恢复普通全 mask 状态。它仍是需要初始化的读写操作，不是隐藏 mask workspace。

任务按 head 优先分配，根据有效 K 前缀和计算量已饱和的 query 尾段统一循环平移，
分散长短任务。所有原始轴 bounds、owner 和双 AIV quorum 检查保留。
最后归一化/写回位于最后 KV 的 update 末尾，与逐行 accumulate 同一 stage；
使用一个输出 UB 槽，为 VR16 留出容量，复用等待实际 MTE3 最后读取。

编译器新增的是通用三角更新、clamped 整数范围、原始轴网格坐标及有界 reader family
支持；没有 attention 专属 pass。设备和性能资格见[causal 验收](../reports/causal-attention.md)。

## 资格边界

数值容差为 rtol=0.002、atol=0.0002。当前日常检查对 256:1024 和 768:3072
进行全输出 oracle；causal 与非 causal 的性能验收都只覆盖 1K/2K/4K/8K 等长正方形，保持默认政策，
每个 shape 的生成代码与 ACLNN 各采集一次默认 msprof。预热和执行次数见[测试指南](../guide/testing.md)，实际结果分别见
[示例验证](../reports/examples.md)与[性能报告](../reports/performance.md)。FA 的 TFLOPS 只计 QK/PV 两次矩阵乘法的算法 FLOPs。

这些记录只覆盖对应配置和产物；BQ128 及其他政策的历史记录不代替当前默认配置。
连续 fence 合并的规则见[同步优化边界](../development/sync-optimizations.md)。

runner能接受某些dtype/head/tile参数不代表其全部通过。causal 的资格单独记录；任意外部 mask、tail、空序列、varlen、paged/sparse、persistent及continuous多时钟状态不在上述资格内。后续变体应通过通用控制/访问合同表达，不创建attention专属pass。

## 复现入口

从本仓库根目录、已激活的 Python 环境运行：

```bash
python -m ascendflow --input examples/flash_attention/attention.py --entry attention_target --compile --timeout 300 --output build/attention/cann
python -m benchmarks.attention.run --output build/attention/generated --build
python -m benchmarks.attention.run --output build/attention/causal --causal --build
```

独立手写AscendC/ACLNN、设备运行和profile见[benchmarks](../../benchmarks/README.md)。手写参考通过不等于本DSL通过。每次构建总期限≤300s；设备运行需全卡检查、显式物理映射，失败立即停止。
