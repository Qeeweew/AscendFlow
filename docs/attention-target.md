# Flash Attention 示例合同

源码：[examples/flash_attention/attention.py](../examples/flash_attention/attention.py)，入口 `attention_target`。算法与固定排步属于example，不属于编译器内部。编译和设备验证使用[统一验收口径](validation.md)。

## 当前默认配置

- dense、无mask forward，BNSD，f16输入输出，B1/Hq8/Hkv8/D128。
- BQ128/BM128/BK1024/CK128/VR8，20组AIC/双AIV。
- Sq/Sk为runtime参数，满足`Sq>0且Sq%BQ==0`、`Sk>0且Sk%BK==0`；空owner不发任务。
- 每组串行处理分配到的Q块，每个Q块遍历KV块，内部Cube/Vector再分块。这里的Q块不是编译器应特殊识别的控制类型。

| 显式workspace | shape | dtype |
|---|---|---|
| Score | `(GROUPS, 2, BQ, BK)` | f32 |
| P | `(GROUPS, 2, BQ, BK)` | 输入dtype |
| PV | `(GROUPS, 2, BQ, D)` | f32 |

Q/K/V/O与workspace的真实指针区间遵守外部disjoint合同；局部L1/L0/UB以及上述GM环仍按真实访问同步，不能因变量名不同认为无复用。

QK产生FP32 score，online softmax保留FP32 m/l；denominator使用未舍入p32，P显式RN后进入PV。O在FP32 rescale/add，最后归一化并RN输出。FP32求和树允许按误差合同调整；其余算法/舍入边界不自动获得修改授权。

默认AIC先新QK后旧PV、AIV先新softmax后旧update。QK子流水为load/MAD/operand/FIX，PV子流水为load/MAD/operand并在外部FIX。编译器保持这些政策、早释放、缓冲复用与事件共享。

## 资格边界

数值验收配置包括 Sq:Sk 为128:1024、384:3072、768:1024、512:2048等，
误差阈值为rtol=0.002、atol=0.0002；4K/8K用于稳态性能比较。
每次结论绑定本次生成的ASC和库、输入与设备，采样与全输出验证分别标明。

runner能接受某些dtype/head/tile参数不代表其全部通过。causal、mask、tail、空序列、varlen、paged/sparse、persistent及continuous多时钟状态不在默认例资格内。后续变体应通过通用控制/访问合同表达，不创建attention专属pass。

## 复现入口

从本仓库根目录、已激活的 Python 环境运行：

```bash
python -m ascendflow --input examples/flash_attention/attention.py --entry attention_target --compile --timeout 300 --output build/attention/cann
python -m benchmarks.attention.run --output build/attention/generated --build
```

独立手写AscendC/ACLNN、设备运行和profile见[benchmarks](../benchmarks/README.md)。手写参考通过不等于本DSL通过。每次构建总期限≤300s；设备运行需全卡检查、显式物理映射，失败立即停止。
