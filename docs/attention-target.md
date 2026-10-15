# 待实现目标：真实 FA 的结构化源码合同

源码规格最初在 `fafeea3` 后建立；目录整理不改变其数学主体。测试入口为
[attention_target.py](../tests/fixtures/attention_target.py)，数学主体为
[源模板](../docs/design/attention_target.py.in)。这是**待接通的设计fixture**，
没有新增可运行FA支持，不调用legacy builder或绕过同步验证。

## 固定输入和执行政策

初始合同：BNSD、f16/bf16、正Sq/Sk、Sq整除BQ、Sk整除BK、HQ为HKV的整数倍，
D=CK=128、BQ128、BK1024、VR8。scale固定为FP32的1/sqrt(128)。首先实现
无mask的forward；尾部/causal/空序列另立合同。工作组默认20，每组round-robin
处理扁平query tile；组内1AIC/2AIV共享task域，不要求各组任务数相同。

query内嵌套和跨query连续流使用**完全同一组stage源码**，区别只有设备控制：

```text
nested:     serial query → Pipelined kv
continuous: Pipelined task → query=task//kv_tiles, kv=task%kv_tiles
```

所有输入长度都是runtime参数，目标工厂不接受它们，也不按长度生成Python任务。
`Sq/Sk` 已使用 `N.int64(minimum=1, multiple_of=BQ/BK)` 声明，通用输入合同
已接入IR和launcher；索引SSA与`fill(-inf)`已能表达两版完整FA的source IR，
物理lowering仍被nested CV阻塞。buffer声明移到共同Kernel
scope，避免从Cube/Vector region逸出未导出的SSA。
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

当前设计模板使用R64 partial后packed merge；广播使用显式8列BRCB scratch。
这些 generic typed 基元已接入 AST softmax，完整 FA 仍被地址与 nested CV 合同
阻塞。按用户最新要求，FP32 sum 归约次序允许改变，不要求位级相同；这里的
固定树只是现有 fixture 配方，不是新 DSL 的限制。scratch/effects 仍须可见，
CPU oracle 使用明确误差限，不能当作 native 指令位级仿真。

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

以下按合同分类，执行顺序以[重整计划](refactor-plan.md)为准。

| 当前合同/缺口 | 下一部分位置 |
|---|---|
| local ordinary work/init/child/use | 已有受限子集与符号完成关系；一般mixed组合仍缺失 |
| Q/K/V共享、runtime BNSD地址、group任务数 | 局部cyclic BNSD搬运已接通；共享KV和mixed分组组合仍待证明 |
| AST嵌套pipeline、内外generation、共享L0C | 一般nested CV与父子完成关系 |
| first/last、初始化与空路径 | 控制/激活域合同；不能只支持最后一代后缀 |
| maximum/reduce/broadcast | 已有AST基元；完整FA集成仍待完成 |
| runtime商余、跨query延迟descriptor | 前端SSA已支持商余；跨query完成/状态时钟仍未接通，不能按K枚举 |
| 双AIV贡献、MTE2最后读取、通知/可见性/资源 | 公共同步、event分配与最终实际IR检查 |
| 大规模AscendC/ACLNN/profile | 完整DSL链验收；已有参考不等于编译器FA结果 |

普通f32 `fill(±inf)`已贯通源/物理合同和ASC，CANN与设备位模式回归通过。
原始两版源码现在可完整通过source IR验证，未使用有限常量替换负无穷。
物理lowering仍明确拒绝：nested因mixed父serial，continuous因nested pipeline。

下一步需要把祖先clock/Stage/guard作为独立执行上下文传递。例如Score导出
的child Stage3仅移动chunk，不能改变父task；O的地址不含kv，仍受父kv的
last guard约束；PV读取常量地址`c0[0]`仍须等待子循环最后MAD。不能从地址
变量集合或region yield推断这些完成关系。固定child循环也必须保留，不以
静态界为理由展开native模板。完整FA仍未通过物理同步或设备验收。
解决一个诊断后还要检查整个表，不应通过缩小任务域或隐藏调用跳过其他缺口。

**独立参考已有记录**：[AscendC参考及证据](attention-reference.md)覆盖手排控制、
小例完整数值、8K大例采样/profile及条件硬件下界；没有新AST FA的CANN或设备
结果，参考性能也未达到最终目标。局部顺序区域、共享只读GM和stationary可写frame已有子集；
FA所需的完整BNSD地址、mixed层次与连续task关系仍未接通。当前边界见
[状态页](refactor-progress.md)，本源码不是可运行example。

复现源码和有限派发表（不启动设备，也不宣称编译FA）：

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.fixtures.attention_target --output runs/refactor/attention-target
.venv/bin/python -m pytest -q compiler/tests/test_attention_target.py
```
