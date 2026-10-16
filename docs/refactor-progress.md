# 当前实现与待解决问题

2026-10-16。默认 dense f16 Flash Attention 已从公开DSL贯通到CANN和设备；
softmax-first排步改善了性能，但编译耗时及相对ACLNN性能仍未达标。
唯一数学主体在[examples/flash_attention/attention.py](../examples/flash_attention/attention.py)。
没有恢复旧task DSL、隐藏算子实现或兼容builder。

## 已验证的纵向路径

路径为 `prim_func → source xDSL → physical IR → autosync → 实际IR独立终检 → ASC → CANN`。
query、runtime KV、固定child三个时钟保留；Sq/Sk不展开。BQ128默认政策有76个native模板、
12个call，20组AIC/双AIV。Vector现在先softmax再update；在dav-2201，跨核
`asc_sync_block_wait`调用`wait_flag_dev`，旧顺序会让旧PV等待阻挡新softmax派发。
改前后物理IR除AIV call issue order外相同，DMA调用多重集/MAD调用相同。

二值event新增一种受限共享：只有bank容量不足时，distance1/单ID的parent通知才可借用
固定child事件的空档。它必须证明旧child Wait→parent Set、parent Wait→新child Set，
条件child中间缺席还须证明FIRST/LAST锚点。每个物理child ID最多一个借用者，配置event
和任意稀疏激活不在此合同内。实际IR终检重新提取所有端点/ID；串行query另外检查全部
mixed parent/child旧新配对。最小反例与范围见[event分配](event-allocation.md)。

当前默认FA有4,372个query内存端点检查、195个资源检查：parent binary30、child113、
mixed44、跨核counter6、query event1、入口配置1。仅补一条跨query协议：

```text
同步前：最后PV FIX读取共享L0C → 下query首QK MAD覆盖，尚无完成证明
同步后：最后PV FIX后 Set(FIX,M,7)；下query首QK MAD前 Wait(FIX,M,7)
Set只在存在下query时发出；Wait只在q>=1时消费；ID7为常量
```

PIPE_ALL仅有3处actor返回；没有query drain、额外copy或新增buffer来换正确性。
有限模型交叉检查短N与破坏反例，不宣称任意程序的形式化证明。

通用constexpr assert已接入AST前端，Python `-O`也不删除其检查；runtime SSA条件拒绝。
纯整数canonicalize/CSE与公共Expr恒等式渲染已接通。符号完成事实对完全不相交origin
跳过包含扫描，不扩大guard/domain、不添加完成边。它未解决总体编译成本。

本阶段此前定向回归322项通过：symbolic59、frontend50、runner/math105、event91、
attention入口17；新增配对Q政策测试另记，不能把尚未完成的端到端构建算成通过。
普通pytest不反复编译完整FA。

## 已有构建与设备证据

最新完成设备资格的产物在外层`runs/refactor/cv-order/build/`，
ASC SHA256 `e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`。
CANN9.1/dav-2201/-O2；源码、IR、同步/验证报告、库及全部编译选项均保留。

| 验证 | 结果 |
|---|---|
| 默认BQ128源码→ASC | 141.225秒，planning79.685、final58.583；旧版128.963秒，仍严重超目标 |
| 同次完整构建含CANN/shared | 147.301秒；统一300秒硬期限 |
| 小规模全输出 | Sq:Sk 128:1024、384:3072、768:1024、512:2048，误差≤3.052e-5 |
| 大规模采样 | 4K/8K，同一binary，误差≤1.526e-5；全输出有限、输入/保护区通过 |
| 三批4K设备task中位时间 | 当前605.780µs，旧排步822.660，独立AscendC1191.220，ACLNN512.152 |
| 性能资格 | 比旧排步降26.4%，仍慢于ACLNN18.3%，未通过≤5%目标 |
| bf16 GQA Hq8/Hkv2 | 同一DSL政策CANN/shared构建148.596秒通过；尚未设备验证 |

runner现在按显式dtype/head合同核对实际source/physical IR与launcher ABI，拒绝用f16库
冒充bf16或GQA。原runner六政策工具资格不能替代设备资格。各进程设备前检查全部卡；
本轮物理6→逻辑0，无reset/换卡规避错误。

## 正在推进：GM→L1复用

用户指定先解决GM→L1，不把Load2D指令数多直接当作性能瓶颈。
`msprof op`三次计数均显示当前776MiB、ACLNN544.3125MiB；详细计数与指标口径见
[性能分析](attention-performance.md)。

同一example新增BQ256/BM128政策：每通信块包含两个独立native128行块，共用一次K/V
L1载入；两AIV各处理128行，Sq要求整除256，pair不跨head。Q/P保留显式独立plane。
softmax在原双缓冲输入槽上计算，省下独立32KiB scratch以容纳输出状态；L1=320KiB，
每AIV UB=183424B，L0容量不变。源码预算GM→L1降为520MiB，尚不能称设备收益。

前端/物理lowering已通过；首次完整构建22.697秒在序列边界分析被静态工作预算拒绝。
诊断为8,968个物理interval重复组合1,296种完整访问pattern，共120,156个输入cell。
现已共享相同transfer，保留逐字节义务和原预算，37项关系/有限历史回归通过。
第二次构建22.184秒推进到child投影，1042个实际端点加10个port的平方预估超过1M，
仍在CANN之前拒绝，未超时。下一步直接生成闭包等价的稀疏投影关系，不先建立完整关系表。
配对政策结构/访问回归27项、runner158项通过；它们不替代完整同步验证。
候选证据保存在`runs/refactor/l1-reuse/`。

## 仍未解决

- 完整FA编译仍需约141秒，目标p50≤5秒/max≤10秒。不存在按Sq/Sk展开；成本来自静态
  物理片段/条件事实重复处理。每个算子构建仍受统一300秒硬期限，超时停止候选序列。
- BQ256复用、bf16/GQA/MQA、causal/tails/decode等资格须逐项取得；目前设备资格只覆盖
  默认dense f16完整tile及独立Sq/Sk矩形输入。paged/sparse/persistent未完成。
- continuous仍拒绝runtime-cycle首末；softmax-first直接跨query还会提前重置单槽`l`，
  必须先解决状态代次，不能仅放开语法或依赖SSA use-def。
- 新query event目前只支持local跨pipe。动态ownership、任意稀疏激活及自动排步未完成。
- 公共结构化ABI元数据仍需改善。加载器目前检查实际launcher名/参数/IR/源和库hash。

复现默认已验证政策（不会安装或打包）：

```bash
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --output runs/fa-reproduce --compile --shared --timeout 300
```

阶段执行顺序见[计划](refactor-plan.md)，设备命令见[benchmarks](../benchmarks/README.md)。
