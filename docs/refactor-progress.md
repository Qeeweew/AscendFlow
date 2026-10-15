# 当前状态与问题清单

2026-10-15，runtime索引已保存于 `97b3c5a`，数值合同、ASC可读性与当前softmax
性能复测已保存于 `f21b57e`；随后接入词法作用域和固定子循环的完成投影。
之前文档与停止的block-sum
实验diff已保存到外层 `runs/refactor/dsl-implementation/starting.patch`。
本页区分本轮实现、历史性能和仍未接通的能力。

**新 DSL 的完整 Flash Attention 仍不能编译。** 旧FA、公开physical builder与
tracer入口已删除；没有绕行内部构造器的兼容example。runtime BNSD局部流水已
接通，下一主线是一般nested CV，[计划](refactor-plan.md)规定其退出条件。

## 本次推进：真实作用域与固定 child 边界

- AST lowering已实际使用一次收集的`Scope/Context`，保留原始clock SSA、
  祖先Stage、guard和actor；每个源效果只收集一次。现有Vector、Cube、serial、
  grouped、单层mixed五类程序的物理IR与改动前一致。AST不再进入旧的展开walker。
- 固定child完成投影已接入现有**local sequence**的规划及最终检查。
  `child_completion.py`复用真实native graph，只暴露指定端点与actor/pipe的
  issue、queue、retire、implicit端口。调用退出只传递对应流水状态，不产生child-done。
  完整single-child invocation已有跨调用协议时，仍走原完整检查，不冒充独立叶。
- 最小Cube反例保留5个native模板：`FIX(7).end → 下一调用 MAD(0).begin`
  本来不可达；从真实`FIX(7).prefix`连接后可达，误用`MAD(7).prefix`仍不可达。
  同时覆盖Set效果与读退休不同、Wait消费、零次child、多actor同UID和端点alias。
- 固定GM列切片不再被“每代必须落在宽度等于stride的连续frame内”误拒绝。
  以地址商/余数区间证明不同代次与owner不重叠；不将child次数放进storage period。
  父级需要的精确frontier按真实固定K扫描行区间，保留每个列块的writer/reader，
  不把最后一个列块writer当作整张矩阵writer。
- `sequence_sync`优先尝试同一consumer真正需要的较晚producer prefix；覆盖仍由
  完成关系证明，直接选出所需同步，不先插入再删除。新增跨owner预算在区间
  物化前计算groups×generation×rows，避免只计core0后产生巨量owner区间。

真实DSL反例在 [pitched_children.py](../tests/fixtures/pitched_children.py)，属于
编译器边界测试，不是FA example。其源程序先用4次循环逐列块写Y，再用1次循环
整体读取Y。同步前有16个跨child byte-cell RAW义务；同步后只有1个边界channel：

```text
r0: MTE3 store(Y[:, j*8:(j+1)*8]), j=0..3
    j=3: Set(MTE3 -> MTE2, ID=2)   // 捕获实际producer prefix
r1: k=0: Wait(MTE3 -> MTE2, ID=2)
    MTE2 load(Y[:, :])
```

两个child循环、5个native模板和双缓冲保留；只有kernel return处最终drain。
删除这对边界同步，最终verifier和独立有限byte/event模型均拒绝；少写一个列块
也不能通过初始化检查。Cube列切片另保持8次循环、5个模板、真实GM row pitch。

| 验证 | 本次结果 |
|---|---|
| Python源码→ASC，五冷进程p50/max | pitched children 0.3132/0.3141秒；Cube列切片0.2834/0.2900秒；层次GEMM 0.5869/0.5895秒 |
| Python→ASC→CANN及共享库，单次总期限300秒 | 两个新fixture分别5.609/5.573秒完成 |
| 设备正确性，物理6→逻辑0 | 所有卡健康/进程/显存/利用率已检查；两个fixture各10次，512/4096个输出精确一致，input/guard不变 |
| 完成投影规模 | count=8和2^40+3的同一leaf有相同native模板、节点、边与投影关系数量 |
| 固定Score访问摘要 | 128行×8列块：1024 cells、7179 visits、2个native模板，storage period=1 |

最后一项摘要成本受静态tile的K和实际访问片段影响，**不是**关于任意K的常数
成本声明；其有65536片段和累计visit预算，运行时N不进入这个扫描入口。
投影事实传播和端点关系提取也有成本，已分别计入
`sequence_child_projection_statistics`，不能只报告父图缩小后的节点数。
端点/端口两两关系提取为二次候选数；在建图前检查100万候选预算，并在覆盖
提取中响应编译取消，不依赖300秒子进程硬截止来掩盖结构膨胀。
本次小设备检查没有吞吐量/ACLNN/profile结论；已有softmax性能不能外推FA。

证据在外层`runs/refactor/nested-child/`：`build/`包含同步前后IR、最终ASC、
CANN选项/hash；`device/result.json`记录正确性；`cold-*-final`记录最终独立进程计时。
最终ASC hash与设备所用共享库的manifest一致。可从工作区根目录复现构建：

```bash
PYTHONPATH=compiler .venv/bin/python - <<'PY'
from tests.fixtures.pitched_children import pitched_children, cube_columns
from xdsl_npu.build import build_module
for name, source in (("pitched_children", pitched_children), ("cube_columns", cube_columns)):
    build_module(source, f"runs/refactor/nested-child/build/{name}", shared=True, timeout_s=300)
PY
```

**未完成：**父级Pipelined中重叠的child调用、父代次地址捕获、跨child的双AIV
贡献与资源寿命、独立query/kv运行时钟及first/last路径。当前nested FA仍明确
拒绝，不能将本次local sequence接线称为nested CV或FA接入完成。

追加完成：

- 普通f32 `fill(±inf)`在source、有限与symbolic physical入口使用同一合同；
  其他scalar仍要求有限值，NaN、有限值溢出和伪造dtype仍拒绝。
- 两版原始FA数学主体都能生成并验证source IR；nested mixed/CV仍拒绝，未迁入
  executable examples。祖先clock、Stage、guard与物理读退休是下一步的组合边界。
- 相邻且同Stage的命令共用generation/激活guard和纯整数CSE作用域；async调用、
  内层协议guard与同步IR不变。softmax ASC由200行降至108行，g定义只剩3处。
- cyclic工作量计数按至多两类owner聚合，覆盖空owner，独立于runtime长度；
  GM流量仅为指令请求，禁止当作HBM实测。参数/派生extent/循环步数溢出会拒绝。

已保存的索引闭环：

- source view改为原始轴SSA origin/width/axes，保留shape与动态radix来源，
  Stage间的纯算术有合法支配关系。先逐轴证明范围，再化简字节地址。
- minimum/multiple进入独立事实环境；运行时商余和整除恒等式不按长度展开。
  多项式有单式项数与累计工作预算，runtime radix不进入存储周期LCM。
- `examples/vector/grouped.py`通过公共DSL/同步路径。cyclic组数不要求整除
  total；空owner正确保留。物理verifier从实际total/blocks/所有GM读写重验ownership。
- 原始source division先于化简校验，避免`bad-bad`或精确除法消元掩盖负dividend。
  动态父view进一步缩窄明确拒绝，不能只保留最后范围后丢弃父约束。
- 数值运算仍在examples；该保存点越过索引失败点后遇到的`fill(-inf)`限制已在
  后续补齐。一般nested mixed与控制合同仍未完成。

详细合同、同步前后和命令见 [runtime索引](runtime-indexing.md)。

上一保存点已实施：

- 删除 `xdsl_npu.Kernel`、`xdsl_npu.builder`、`trace_func`、隐式`versions=`/
  L0 `gemm`、旧examples/runners。`_physical_builder`只供lowering和内部测试。
- 接入 `T.int64(minimum=1, multiple_of=B)`；constexpr可后置声明。合同贯通
  source/physical IR和local/sequence/mixed launcher，非法值在地址计算和派发前
  拒绝。分析保守使用非负域；没有宣称任意参数约束或BNSD已支持。
- 跨source region逸出的allocation明确诊断；FA设计模板改为共同scope声明。
- NativeCompletion共享一份拓扑，各query domain保留独立激活和事实，新增真实
  结构计数。初版没有提速，定位后去掉重复激活绑定和包装调用；最终收益有限，见下方实测。
- 通用构建工具覆盖输入构造→Python→ASC→CANN总期限，失败不发布成功产物；
  现有AST/独立参考runner统一调用。编译缓存没有代替分析。
- 关键机器反例迁到内部fixture；旧FA整算子策略测试改为小型CV、跨owner状态
  WAR和独立数学测试。编译器内旧FA专用`mask_causal/mask_prefix`recipe已删除，
  后续mask应由通用typed运算和example有效域表达。

## 路径与实际能力

| 路径 | 已有能力 | 明确边界 |
|---|---|---|
| AST DSL → source IR → physical IR → AutoSync → ASC | runtime 局部流水、Vector/归约、显式 Cube、受限 serial parent/child、部分局部 CV、BNSD cyclic局部寻址 | 一般 nested mixed/CV、跨query多时钟与完整 FA 未接通 |
| 内部physical IR测试 | 保留机器访问、事件、ownership与退休反例 | 没有旧用户DSL；有限fixture不证明runtime FA支持 |
| FA 设计 fixture | 原始完整数学主体可生成source IR，包含负无穷初始化；静态容量与有限派发表检查 | 物理编译仍拒绝，不是 executable example |
| 独立 AscendC / 安装 ACLNN | 数值、排步与性能参照 | 不证明编译器或新 DSL 支持同一算子 |

AST 的两 runtime 轴、只读共享、serial 层次已有受限实例，不是全部从零开始。
同样不能由BNSD局部搬运推断一般多时钟 CV 或间接访问已支持。
公共入口在 [compilation.py](../src/xdsl_npu/compilation.py)，
目标失败回归在 [test_attention_target.py](../tests/test_attention_target.py)，
旧FA suite已删除，新FA设备runner尚未实现。

## 待处理的不止四项

| ID | 实际问题与源码依据 | 处理与验收方向 |
|---|---|---|
| API | 旧公开task/tracer及FA入口已删除，AST接口保留 | 不再维护兼容层；内部物理fixture不成为用户入口 |
| INPUT | minimum/multiple合同已贯通IR/launcher；多参数关系与head映射仍未完成 | 继续完善可验证地址/owner前提，保留溢出和空工作语义 |
| INDEX | 原始轴SSA与受限runtime商余已贯通局部流水；runtime row pitch/多时钟动态stride仍拒绝 | 延伸同一范围/ownership合同，不把合法长度放入编译键 |
| CV | `region_lowering.py` 拒绝一般 mixed sequence 和 AST nested pipeline；条件 lowering 仅有受限末代后缀 | 保留父子 actor/clock/generation、first/last/零次路径；完整 QK/softmax/PV/update 纵向闭环 |
| MEMORY | cyclic局部不等长/空group已证明；共享 Q/K/V、间接地址及跨 child 复用仍不足 | 真实范围、有效域、最后读取和 owner 分区；buffer 分配/布局/可见性与同步一同验证 |
| SYNC | runtime 叶已直接生成，局部mixed已有双AIV quorum；legacy仍有候选试删。一般nested CV中的贡献组合和事件共享未完成 | 共用硬件完成语义，分开物理访问与事件寿命；最终从实际 IR 检查，不能旁路或加任意 barrier |
| EFFECT | 细粒度 native 已能在最后 reader 后早 release，但主要使用 task end；SPR/入口协议另有特殊处理 | 明确每个 native 的读退休、写完成与配置资源合同；不能让高层 recipe 隐藏多个里程碑 |
| EVENT | 已有有限表分段周期压缩和标准整数 pass；有限 ID 表仍随 generation 增长，runtime 跨 channel/region 共享不足 | 从符号寿命直接分配/表达 ID；化简不得改变配对、代次或等待位置 |
| COMPILE | 查询域拓扑已共享；条件记录扫描、激活绑定、父层重复工作仍需测量 | 拒绝用结构改善代替真实耗时；事实成本和父层计数继续跟踪 |
| LOWER | 旧FA专用mask recipe已删；一般mask/间接访问typed合同仍缺失 | head mapping/online状态留在examples；native effects/scratch完整可见 |
| TEST | 目标失败测试与端到端支持易混淆；runtime 部分测试复用生产访问分析，本地独立模型尚不覆盖一般 Cube/CV | 成功闭环、故意破坏 IR、独立有限模型分层；保留反例而非旧 API；不放宽判定以获得通过 |
| PERF | 新 DSL FA 无性能结果；旧参考的分组/排步不总相同，部分新候选只有一批 | 同路径 source/IR/ASC/binary 与 AscendC/ACLNN/profile 对照；固定大例和硬件瓶颈分析 |
| PRODUCT | 当前入口已统一总deadline和只读产物快照；父层成本及全部ABI合同仍待完善 | 一份当前状态、一份计划；同源码/库/输入证据，运行数据不进产品Git |

## 同步与复杂度：纠正过时结论

当前 `region_completion.py` 已使用 `SymbolicCompletion`，不再复制旧首尾指令
窗口；本轮又共享了局部查询域拓扑，仍有按调用的family和各域条件事实。不能按旧计划
“先取消窗口”重复做已经完成的工作。runtime 直接生成与 legacy 有限搜索也是
两条不同路径，不能笼统宣称“全编译器已没有试删”。

[同步问题清单](sync-known-issues.md)区分已处理的拓扑/零计数与剩余问题。以下为历史保存点数据：
既有单次内部计时为 GEMM 0.414s、frame 0.262s、局部 CV 3.172s；
不是五冷进程结论。局部 CV 仍有 1197 family、1947 arc、68957 transfers；
`proof_domain_states=0` 是当时指标失效；本轮已换成真实completion_statistics。
证据在外层 `runs/refactor/symbolic-frontier/domains/`。

旧 FA 的几十秒成本包含 finite generation/ownership/event 分配，不能用于判断
新 DSL runtime FA 的成本，因为后者还不能编译；也不能把它一概归因于 Python。
全链300秒硬停止已从旧suite提取至 `xdsl_npu.build` 并用于当前构建入口。

另核实 Load3D 入口直接生成 S→MTE1 ID0 的 Set/Wait，却未列入 `event_ids`
报告。纯 Python 编译 `batched_matmul.specialize(TRANSPOSE_B=False)` 并启用
`load3d=True`，ASC 中两者各一次，资源表缺该 bank。当前 AIC native 闭集没有
普通 Scalar task，尚未据此发现实际冲突；它是资源报告和后续组合合同的缺口，
不能描述成已发现设备错误。依据为 `codegen.py` 的入口配置与 `loop_sync.py`
的 channel 资源汇总；后续应检查报告覆盖所有实际同步，包括入口与收尾。

## 现有性能证据的归属

| 路径与例子 | 已有证据 | 可得结论 |
|---|---|---|
| AST softmax，32768×1024 f32 | 三批 generated 217.914μs / AscendC 240.054μs / ACLNN 284.696μs，全输出检查 | 该 Vector 实例已有完整链，不能外推 FA |
| AST RMS primitive，8192×4096 f32 | generated 218.346μs / AscendC 235.884μs | 仅对应所记录原语与政策，不是完整 RMSNorm 库 |
| 已提交 legacy FA，4K MHA f16 | 三批 generated 698.726μs / ACLNN 512.678μs | 仍慢约36%，新 DSL 无此成绩 |
| 已提交 legacy FA，8K GQA bf16 | 单批 2262.544μs / ACLNN 1687.266μs | 证据不足以作三批验收，且仍有差距 |
| 未提交 legacy block-sum 实验，4K MHA | 单批 603.274μs / ACLNN 512.690μs；小例数值检查 | 已停止，未完成性能验收，不作为后续主线 |

详细口径分别见 [Vector](vector-row-performance.md)、[legacy FA](flash-attention.md)。
FP32 求和归约顺序按用户要求允许调整，不再设位级相同门槛；精度、数值误差、
概率舍入和递推状态仍有明确合同。不能用更宽容差掩盖地址或同步错误。

## 当前 ASC 与性能复测

同Stage合并前后的softmax **after.xdsl逐字节相同**；生成ASC由8258降到6590字节，
设备aibin由3272降到3224字节。纯整数优化没有移动native/sync调用，也没有改
buffer容量、搬运或浮点计算配方。独立C++/UBSan轨迹执行覆盖N=0..7的实际
guard与事件ID；相关codegen/归约/边界/invocation回归91项通过。

物理6→逻辑0，32768×1024 f32，三批generated/AscendC/ACLNN交替运行，
每批5次warmup、30次测量。下表是kernel duration中位数的中位数，排除host间隙：

| 实现 | 三批p50，μs | 中位数，μs |
|---|---|---:|
| 当前DSL生成 | 210.450 / 210.688 / 210.540 | 210.540 |
| 独立AscendC，同政策 | 239.732 / 239.590 / 239.560 | 239.590 |
| 安装ACLNN | 267.240 / 264.980 / 267.740 | 267.240 |

该shape生成版耗时低于ACLNN约21.2%，低于固定AscendC约12.1%；不能外推FA或
其他shape。第三批生成版p90为225.160μs，尾部有波动，原始样本保留。
每进程检查全部33554432输出、输入与guard；最大绝对误差生成/AscendC为
9.31e-10，ACLNN为1.40e-9。generated和AscendC均32 AIV，ACLNN使用自己的40核政策。

硬件为910B3，20 AIC/40 AIV、L2 192MiB。工作集256MiB；生成版使用80%的AIV，
UB131872字节/AIV。Vector/Scalar/MTE2/MTE3活动比例的三批中位数分别为
`.980/.050/.4155/.184`，ACLNN为`.646/.241/.1755/.094`。比例可重叠，不能相加
成总延迟或据此单独证明关键路径；后续应优先调查Vector混合指令工作。

当前IR请求GM 268435456字节、UB操作数1355284480字节，包含33521664次max、
67076096次sub/add及各33554432个exp/div，没有Cube工作。条件下界仍是各资源
服务时间及依赖路径的最大值；未经核实的AIV频率、exp/div吞吐和HBM流量不能
替换成Cube峰值。约1274.99GB/s是GM请求/实测kernel时间，不是实测HBM带宽。

五冷进程DSL→ASC p50/max **0.4026/0.4074s**，import与CANN另计。
最终generated/AscendC全链构建 **5.791/13.000s**，300秒期限始终有效。
本轮没有同场profile合并前的生成版，不把历史成绩差值归功于本次化简。
证据：外层 `runs/refactor/softmax-current/`，`build-final/`为被测ASC/IR/库，
`profiles/`为九次完整测量，`performance.json`绑定源码/库/CSV/hash与硬件，
`cold-final/`为编译测量。命令模式见 [Vector性能](vector-row-performance.md)。

普通fill正负无穷全链构建5.466s；物理6上10次重复，128个输出的IEEE位模式
严格等于正/负无穷，guard完好。证据在 `runs/refactor/fill-constants/`。
fill及真实FA source IR/拒绝边界定向69项通过；cyclic accounting新增19项和
既有22项通过。未重复运行分钟级全量套件。

## Runtime 索引闭环的验证

源码/IR迁移后完整回归 **999 passed / 209.04s**。随后审查发现并修复负unsigned
dividend被消元和动态父view缩窄丢约束两处漏洞；对应source/索引/轴合同最终
定向 **63 passed / 2.53s**，事件ID的实际C++/UBSan定向 **4 passed / 1.15s**。
没有为每个补丁重跑完整套件。Ruff与diff检查通过。

最终grouped构建全链 **5.776s**，CANN `-O2 -std=c++20 --npu-arch=dav-2201`。
五个独立冷进程，从读取DSL到ASC，不含import/CANN：

| 示例 | p50 | max |
|---|---:|---:|
| grouped BNSD | 0.2737s | 0.2751s |
| hierarchical GEMM | 0.6145s | 0.6203s |

GEMM比上一保存点的0.5617s增加约9.4%；本轮增加了source轴和division验证，
但未做同场旧版A/B，不能把差值全部归因于某个pass，也不宣称全面编译提速。
两者都不按runtime长度展开；grouped为3个native模板、局部存储周期2，动态radix
不增加周期。等差event ID直接用`base+step*(ordinal%width)`，默认例为`g%2`；
非等差表保持选择，分配/配对/guard不变。ASC更短不代表机器指令或延迟必然减少：
本次等差化简前后aibin为2840/2920字节，未据此宣称性能提升。

检查全部8卡后选健康空闲physical6→logical0。同一个库运行S=16/48/64/4096/
131072，全部输出精确正确，输入与guard完好；覆盖14个空owner和不等长分组。
最大shape **[2,3,131072,64] f32**，50331648元素、每张量192MiB。
5次warmup、30次重复的stream interval为 **508.205μs/launch**，包含host提交
间隙，尚无本例的kernel-only/AscendC/ACLNN对比，不能用它验收FA性能目标。

证据在外层 `runs/refactor/runtime-index/`：`grouped-final/`保存source/before/
after IR与ASC、CANN命令/库hash；`device/device.json`保存全卡快照与所有输入；
`cold-grouped/`、`cold-gemm/`记录源码和编译器树hash、分阶段耗时。
完整测试日志在 `runs/refactor/dsl-implementation/runtime-index-tests.log`。
可复现命令见 [runtime索引文档](runtime-indexing.md)。

## 上一保存点的验证与复现

集成回归 **894 passed / 196.95s**，只运行一次完整套件；Ruff与diff检查通过。
Vector与层次GEMM分别完成CANN离线/shared构建，总耗时5.46/5.94秒，均在300秒
总期限内。源码/命令/库hash与原始日志保留在外层 `runs/refactor/dsl-implementation/`。

五个独立冷进程，源码/输入IR→ASC，不含import与CANN：

| 输入 | 改动前p50 / max | 最终p50 / max | 生成ASC |
|---|---|---|---|
| AST hierarchical GEMM | 0.5824 / 0.5874s | 0.5617 / 0.5632s | SHA256相同 |
| segmented CV physical IR | 3.9251 / 3.9576s | 3.8145 / 4.0647s | SHA256相同 |

中位数改善约3.6%/2.8%，CV最大值仍有波动，不能称编译性能问题已解决。
CV输入是内部物理IR，不是新DSL完整FA。首版共享拓扑回退到4.083s后，通过
profile去掉重复激活绑定、约27万次包装调用和重复条件key访问；同步统计和
ASC均保持一致。该改动没有消除各域事实存储及条件包含扫描的复杂度。
对照见 `compile-comparison.json`；旧IR为适配新必填合同显式写minimum=0后，
双方使用同一份输入，不增加生产兼容层。

全卡健康/进程/HBM/利用率检查后，physical1→logical0，同一个Vector库验证
0/1/2/4096/1048576个tile。最大输入256MiB，67108864输出逐元素精确正确，
输入和guard完好，负长度在派发前拒绝。该运行是正确性验证；记录的stream
interval包含host间隙，没有新的算子kernel性能或FA性能结论。

```bash
PYTHONPATH=compiler .venv/bin/xdsl-npu --input compiler/examples/vector/scale.py --entry scale --output runs/reproduce/scale --shared --timeout 300
PYTHONPATH=compiler .venv/bin/xdsl-npu --input compiler/examples/gemm/hierarchical.py --entry hierarchical_gemm --output runs/reproduce/gemm --shared --timeout 300
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/gemm/hierarchical.py --entry hierarchical_gemm --trials 5 --output runs/reproduce/cold-gemm
```

该保存点之后已实现本页开头的runtime索引闭环；完整FA及其变体、一般nested CV、
跨region事件共享、设备性能对齐仍未完成。

## 工作区和本次交付边界

根目录无Git，产品变更只在compiler仓库；旧block-sum实验与本轮前所有未提交
修改已有starting.patch，未丢弃用户工作。旧FA策略代码随入口删除；通用
`reduce_sum8`物理访问合同保留，不据此宣称新DSL已有block-sum FA。

三名 `gpt-6.1-sol / max` subagent 分别实施语言/合同、构建期限、拓扑与反例迁移；
文件写入分工明确，主Agent负责集成与冷计时。参考checkout未更新：
ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
CATLASS `698379e0d4b1b94d12728e7181df2cf83a958e8f`，
ops-nn `64b41ad63a5073bc711ba03802498bfe8bdb77e5`。
参考checkout与已安装ACLNN的构建来源不能混同。
