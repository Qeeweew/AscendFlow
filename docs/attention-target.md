# P0：真实 FA 的结构化源码合同

代码基线 `fafeea3` 后新增。入口为
[attention_target.py](../examples/attention_target.py)，数学主体为
[源模板](../examples/design/attention_target.py.in)。这是**待接通的设计fixture**，
没有新增可运行FA支持，不调用legacy builder或绕过同步验证。

## 固定输入和执行政策

初始合同：BNSD、f16/bf16、正Sq/Sk、Sq整除BQ、Sk整除BK、HQ整除HKV，
D=CK=128、BQ128、BK1024、VR8。scale固定为FP32的1/sqrt(128)。首先实现
无mask的forward；尾部/causal/空序列另立合同。工作组默认20，每组round-robin
处理扁平query tile；组内1AIC/2AIV共享task域，不要求各组任务数相同。

query内嵌套和跨query连续流使用**完全同一组stage源码**，区别只有设备控制：

```text
nested:     serial query → Pipelined kv
continuous: Pipelined task → query=task//kv_tiles, kv=task%kv_tiles
```

所有输入长度都是runtime参数，目标工厂不接受它们，也不按长度生成Python任务。
有限激活表工具只用于审查，限制64个逻辑task；它不是生产编译器或硬件模拟器。

固定policy：AIC qk0→pv1；AIV update2→softmax1。同一actor本地step中先派发
新QK再到旧P gate；先派发旧Update再到新Score gate。连续流的query边界例如：

```text
AIC: QK(query1,kv0) 先于 PV(query0,last)
AIV: Update(query0,last) 先于 Softmax(query1,kv0)
```

这使单份m/l/O在所选源顺序下可复用：旧query最终归一化先于新query重置l；
真实异步读退休仍必须由编译器证明。不能把这一源码顺序当成完成，也不能
在改为softmax-before-update时继续无条件沿用相同状态分配。
Q缓存可在旧PV期间服务新query，因为PV读P而非Q；覆盖Q前仍要等其最后MAD reader。

ops-transformer HEAD `4bbfb3b` 的 arch22
`flash_attention_score_s1s2_bn2gs1.h:920–958` 可见 Vec1 在 Vec2 前。
本fixture是明确的候选专家policy，不冒称该CANN模板的逐条复刻；Matmul库内部
派发还需独立AscendC和实际代码核对。A2通信块/行块/native块依据其FA设计文档。

## 数值和物理存储

每逻辑BK：QK以FP32输出；scale、row max、max历史、exp得到alpha和p32；
使用p32做denominator；P显式RN到operand dtype；PV按CK128顺序累计；O独立
rescale/add。末块先FP32 `1/l`，再乘O，最后RN输出。不替换成rsqrt、融合FMA
或逐native块softmax；P舍入不越过逻辑BK。

行归约固定R64 partial后packed merge；广播使用显式8列BRCB scratch。
这几项generic typed语法还需实现，并映射到已有受审查native操作；不能让宏
隐藏临时分配或引入整行broadcast搬运。CPU oracle可复用现有online recurrence，
但native累计/exp误差需要独立容差，不能当位级仿真。

默认源码分配（每个AIC/每个AIV分别计费）：

| 存储 | 组织 |
|---|---|
| L1 | Q cache32KiB；K/P/V各双槽64KiB；总224KiB |
| L0A/B | Q/P和K/V分别各32KiB，总各64KiB |
| L0C | 两个128×128 FP32平面，共128KiB，QK/PV真实共用 |
| UB | 两槽Score输入、P输出、PV输入、输出cast；一份score work/m/l/O；alpha双槽；归约与广播scratch |
| GM | 每group双槽Score/P/PV，共1MiB+512KiB+128KiB |

UB静态总量由测试直接读取源码分配表达式计算，必须≤192KiB/每AIV，不能将两个
AIV容量相加。这个检查仅证明声明容量，不证明布局、异步复用或代码性能。
三个GM工作区声明为未初始化的可写分配；后续读必须来自实际writer，不预授权入口值。

QK内层保留4stage load/operand/MAD/FIX，实际issue_order为load/MAD/operand/FIX；
PV保留3stage累加子流水，issue_order为load/MAD/operand，循环外只做一次FIX。
旧fixture遗漏内层order，单槽L0 operand会在旧MAD派发前被下一代覆盖；已修正并
增加负例，没有扩大buffer。Vector按8行保留load/compute/store子流水。K/V按BNSD直接读，
没有预转置、GM packing或全形状Score/P矩阵。

## 缺口映射与实际拒绝

| 缺口 | 依赖/工作包 |
|---|---|
| 普通region前后copy/fill/FIX、真实初始化及异步出口 | P1顺序区域组合 |
| K/V跨query/head/group只读共享；交接scratch未初始化读检查 | P1真实effects/provenance，P2参与者 |
| 每group任务数、runtime extent/stride、非连续2D片段 | P1/P2地址/owner证明；不能强制panel-major |
| Pipelined内嵌Pipelined及内外两个generation | P2层次CV |
| first/last合流、NEG_INF fill、UB bitwise copy | P1边界/P2原语 |
| maximum、packed reduce、带显式scratch的row broadcast | P2通用typed lowering |
| runtime商余和延迟query/head/kv descriptor | P3a先验证关系/规模，P3b接完整FA |
| GM完整贡献、双AIV quorum、各MTE2最后读与event寿命 | P2/P3同步/资源及最终检查 |
| 大规模独立AscendC/ACLNN/profiling对照 | P0固定reference、P2/P3设备验收 |

当前nested的首个诊断是`unsupported index product`，continuous是动态除数
不在受支持索引子集中。测试要求明确失败，不把fixture解析成功当作FA编译通过。
解决一个诊断后还要检查整个表，不应通过缩小任务域或隐藏调用跳过其他缺口。

**P0独立参考已落地**：[AscendC参考及证据](attention-reference.md)覆盖手排控制、
小例完整数值、8K大例采样/profile及条件硬件下界；没有新AST FA的CANN或设备
结果，参考性能也未达到最终目标。P1已接通根部local顺序区域
的同步、代码生成和设备例，但父体组合及本fixture所需mixed层次仍缺，见
[区域组合](region-sequence.md)。最新实施次序以[重构方案](refactor-plan.md)为准。

复现源码和有限派发表（不启动设备，也不宣称编译FA）：

```bash
.venv/bin/python compiler/examples/attention_target.py --output runs/refactor/attention-target
.venv/bin/python -m pytest -q compiler/tests/test_attention_target.py
```
