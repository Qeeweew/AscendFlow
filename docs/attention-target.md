# Flash Attention 示例合同

源码：[examples/flash_attention/attention.py](../examples/flash_attention/attention.py)，入口 `attention_target`。算法与固定排步属于example，不属于编译器内部。**当前编译性能未达标；新的统一循环重构尚未实现。**

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

默认AIC先新QK后旧PV、AIV先新softmax后旧update。QK子流水为load/MAD/operand/FIX，PV子流水为load/MAD/operand并在外部FIX。编译器重构应保留这些政策、早释放、缓冲复用与事件共享；本轮不调参。

## 资格边界

旧实现曾对默认政策完成CANN和设备数值验证：Sq:Sk为128:1024、384:3072、768:1024、512:2048，阈值rtol=0.002、atol=0.0002；4K/8K另有采样记录。这些是对应旧产物的历史资格，不是当前未提交求解器实验的新设备结果。

历史4K同卡msprof摘要为生成代码605.780µs、ACLNN512.152µs，差18.3%，仍未达标。旧库、部分profiler原始数据和旧目录已经清理，需要重建后才能再次运行；历史报告与当前状态入口见[状态页](refactor-progress.md)。

runner能接受某些dtype/head/tile参数不代表其全部通过。causal、mask、tail、空序列、varlen、paged/sparse、persistent及continuous多时钟状态不在默认例资格内。后续变体应通过通用控制/访问合同表达，不创建attention专属pass。

## 复现入口

从工作区根目录运行；以下命令说明接口，本轮文档整理未执行：

```bash
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --compile --timeout 300 --output runs/structured-loops/cann-fa
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run --output runs/structured-loops/fa --build
```

独立手写AscendC/ACLNN、设备运行和profile见[benchmarks](../benchmarks/README.md)。手写参考通过不等于本DSL通过。每次构建总期限≤300s；设备运行需全卡检查、显式物理映射，失败立即停止。
