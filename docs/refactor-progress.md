# 当前状态与问题清单

2026-10-15，runtime索引已保存于 `97b3c5a`，数值合同、ASC可读性与当前softmax
性能复测已保存于 `f21b57e`；随后接入词法作用域和固定子循环的完成投影。
之前文档与停止的block-sum
实验diff已保存到外层 `runs/refactor/dsl-implementation/starting.patch`。
本页区分本轮实现、历史性能和仍未接通的能力。

**新 DSL 的完整 Flash Attention 仍不能编译。** 旧FA、公开physical builder与
tracer入口已删除；没有绕行内部构造器的兼容example。runtime BNSD局部流水与
一层固定child的嵌套CV已通过CANN与设备正确性；stationary私有L0C的跨调用复用
现也完成设备闭环。parent native首末执行已接通；本轮将原nested FA完整降到
保留query/KV/child时钟的物理IR。**跨query内存与event寿命尚未证明，完整FA
仍不能生成可执行ASC**，[计划](refactor-plan.md)规定其退出条件。

## 当前：query 地址证明、完成关系组合与编译开销

单 query 的访问分析现直接读取真实 Capture/native effects，进入原有 parent
规划与检查。按同一物理 root 的全部 actor/首末分支分类：只读且已初始化的 GM、
cyclic query 独占输出、全局 epoch 环，以及原位复用的存储。分析中的地址归一化
不改实际 Capture、GM ABI、私有容量或布局；环保留真实
`(query*kv_tiles)%slots` 相位。当前仅接纳 cyclic query，普通 uniform query
在这一步明确拒绝。Q/K/V 的只读性来自全 root 实际 effects，不能由参数名推断。

两个真实 IR 反例已固定：Score reader 改为 `(epoch+query)%2` 虽然仍能通过
物理范围提取，但相位证明拒绝；把末次 O 写入重绑定到 K 后，全 root 扫描发现
隐藏写入，不再允许只读归一化。地址 oracle 检查奇偶 KV、空 owner、不等长 owner
及 native 行片段，不能把它当作同步证明。

内部 `_plan_scope_parent` / `_verify_scope_parent` 与普通公开编译共用同一实现。
小型 mixed recurrence 已通过单 query 的实际协议生成与复核；返回报告明确标记
`cross_query_verified=false`。autosync、专家物理验证及 ASC 门禁仍保留。
外层完成关系适配器复用原来的硬件拓扑和符号求解器，按静态份数连接各 actor/pipe
实际出口到入口；保留 inactive call 的透明 prefix，绝不把 child exit 当完成。
它只查询完成关系，尚未生成跨 query 协议或验证外层事件配对和进展。

两槽环在 KV=1 时隔两个 query 才复用同一槽，不能只看相邻 query；静态三份
拓扑的测试覆盖该距离，生产算法尚需从真实槽周期及未触及字节的 identity transfer
推导边界。只在首次 KV 写入的 qa 还需保留所有 KV/chunk reader span；末次 O
store 的 state 读取和旧 Wait 消费也各有独立义务。缺少投影的真实端点会报错，
不会因没有节点而声称已经完成。

真实原 FA 的单 query 规划和独立复核现已通过。首次构建约76秒触发原2M
传播工作量上限。一次诊断复现记录：4236节点/6285边、575候选、534 origins、
254次传播、2000180次边传递、约1988万次事实比较；新增边 replay 只有1774次。
主因是逐候选加入少量 origin 后重复传播，不是运行长度展开。候选、child event
和不同bank的最小width现批量登记，逐边选择、case覆盖与最终验证保留；没有提高
预算。fixed child另将逐source/target重复扫描改为每个真实target一次求origin
mask，保留完整relations及不同Site身份。无需actor重命名时直接复用projection。

最终同一原FA源码到单query实际IR规划累计52.59秒，独立复核后累计84.74秒；
watchdog覆盖的进程总时间86.50秒。保持72个native模板、12个call、11个父协议，
最终图4303节点/6503边，复核623168次传播。该阶段优化前后生成的parent-only IR
SHA256完全相同。这里只是单次构建证据，仍远未达到完整FA的5秒目标，没有ASC。
证据位于外层 `runs/refactor/query-sync/parent-final/`；这些 parent-only IR
不能用于设备执行。较早失败及逐flow诊断也保留，不用成功的小例覆盖真实失败。

最终相关集成回归 **597项通过，111.22秒**，包含实际IR破坏、固定child/条件
关系差分、独立有限query执行图、地址别名与事件复用。此有界检查不等于任意规模
形式化证明。五个已有设备证据的fixture重新生成ASC，均与原设备源逐字节一致：
ub_copy、native_guarded_recurrence、nested_copy、nested_cv、shared_l0c_qk_pv。
没有重新运行设备，也没有给FA借用这些成绩。Ruff/格式/diff检查通过。

在工作区根目录可复现本层检查：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_scope_query_access.py compiler/tests/test_scope_query_completion.py compiler/tests/test_scope_query_parent.py
```

尚无本阶段完整 FA ASC、CANN、设备正确性或性能结果。下一步完成外层
memory/event 边界与实际同步，
之后才迁入可运行 examples 和进行大规模 AscendC/ACLNN/profile 对比。

## 已保存阶段 2fcceee：真实 FA 的三个时钟与 UB 原样复制

原[FA主体](design/attention_target.py.in)未改。现在source→physical路径保留
20组AIC/双AIV、12个call、72个native模板及原始buffer、数学运算和排步。
query为实际`scf.For`，KV仍为runtime Stream，四类固定child保持自己的时钟。
各actor的query域必须一致，支持cyclic不等长和空owner；`(query*kv_tiles+kv)%2`
保留真实跨query槽相位。没有使用旧Frame假设、展开query或插入query drain。
物理IR提取检查clock、原root、地址范围和alignment；这不替代同步/ownership证明。

地址修复均按通用语义处理：展开的`tile*core+tile*groups*query`从已验证joint
domain恢复相关区间；捕获分离继续用多项式证明等价，但保留原始商余表达式，
避免把`x%2`展开成`x-2*(x/2)`后丢失范围和可读性。若源程序没有显式SSA tile
依赖，跳过该专用检查的冗余RAW构造；物理抽取与最终内存/同步验证仍保留。

真实FA还暴露了`N.copy(next_m,m[row])`未接UB→UB的问题，现已增加通用原生
路径。CANN 9.1的dav_c220接口明确标记`begin_pipe(V)`，2201实现直接调用
`copy_ubuf_to_ubuf`，size单位是byte。编译器建模为V的R/W异步访问，使用
`asc_copy_ub2ub`；拒绝重叠/不可证明别名、非32字节块、prefix/tail及错误owner。
没有用浮点运算模拟复制，也没有额外GM staging。

独立fixture [ub_copy](../tests/fixtures/ub_copy.py)的同步前是GM→UB→UB→GM三段；
同步后保留3个native模板、原来两份64字节UB，生成以下4组协议：

```text
MTE2 load(g) → V copy(g)       ready
V copy(g)   → MTE3 store(g)   ready
V copy(g)   → MTE2 load(g+2)  source-slot release
MTE3 store(g) → V copy(g+2)   destination-slot release
```

删除每个实际ready/release，最终verifier和独立byte/DAG模型均拒绝。新route
24项回归通过；query/地址/既有pipeline等集成回归406项通过，52.49秒。query
fixture另以真实source SSA解释检查顺序数值和byte footprint，未将其冒充同步模型。
UB copy整链构建5.450秒（统一300秒期限，CANN 9.1/dav-2201/-O2）；全卡状态
检查后物理6→逻辑0，同一binary对N=0/1/2/3/7/4096各运行10次，输出、输入和
保护区逐字节一致，包括±0、inf及qNaN/sNaN payload。没有FA吞吐量结论。

五个独立冷进程，源码读取/构造→最终检查→ASC（import/CANN另计）：

| 输入 | p50 / max，秒 |
|---|---|
| ub_copy | 0.2107 / 0.2166 |
| native_guarded_recurrence | 1.9829 / 2.0053 |
| nested_copy | 1.3523 / 1.3862 |
| nested_cv | 1.6733 / 1.6772 |
| shared_l0c_qk_pv | 3.2898 / 3.2978 |

不能由局部省掉分析宣称整体提速；四个既有输入仍未达到常用kernel p50≤1秒
目标，其ASC与上一阶段设备版本逐字节相同。UB copy的ASC/设备来源hash一致。
全部证据在`runs/refactor/scoped-queries/`：`physical/attention/`保存原source和
物理IR，明确记录同步拒绝、无ASC；`ub-copy/`保存同步前后IR、ASC、CANN和设备
结果；`cold-final/`与`validation.json`记录冷测、代码身份和hash核对。

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.support.run_ub_copy --output runs/refactor/scoped-queries/ub-copy --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 PYTHONPATH=compiler .venv/bin/python -m tests.support.run_ub_copy --output runs/refactor/scoped-queries/ub-copy --physical 6
.venv/bin/python -m pytest -q compiler/tests/test_scope_serial_integration.py compiler/tests/test_ub_copy.py
```

设备命令执行前须确认物理6仍健康空闲。下一步在真实capture上处理query tile、
共享只读head映射与全局epoch环，再证明query尾全部reader/退休/Wait消费到
下query初始化和Set的关系。当前autosync、专家physical验证及emitter均保留门禁，
不会漏掉外层循环而返回成功。continuous仍拒绝runtime-cycle首末谓词。

## 已保存阶段 4070d31：保留子流水中的原生首末条件

公开DSL fixture [native_guarded_recurrence](../tests/fixtures/guarded.py)现在贯通
源码→同步前后IR→ASC→CANN→设备。它在首次parent初始化state/factor，固定的
两行子流水每轮执行`state=(state+X)*0.5`，仅末次parent在原compute/store stage
转换f16并输出。N=0不读输入也不写输出；N是运行时参数。

`ScopeGuardOp`同时保留parent和child的真实SSA坐标。分析分别检查N=1、first、
middle、last四个互斥域，再合并相同的event配置；实际IR和ASC仍只有9个native
模板、2个call，不复制四份设备程序，不枚举N。每个case独立完成约简，不能先
合并互斥case的边再做SCC。依赖从实际激活访问重建，不是把无条件依赖加个if。
首次写入→全域reader和全域reader→末次覆盖使用带首末锚点的区间覆盖查询。

同步前只有首轮fill、每轮load/add/brcb/mul、末轮cast/store。同步后关键关系为：

| 真实访问义务 | 生成同步及等待范围 |
|---|---|
| 首次state/factor初始化→后续使用 | 首个add前`PIPE_V`栅栏，之后由已证明的流水关系传递；不每轮重置 |
| 本行GM→UB完成→add读取 | `MTE2→V` Set/Wait，ID为`row%2` |
| add最后读取x行→下一parent覆盖该行 | add后`V→MTE2` Set，下一parent相同行load前Wait；两行分别使用常量ID 0/1 |
| 末轮cast完成→GM store | 仅`parent_g==N-1`执行`V→MTE3` Set/Wait，ID为`row%2` |

x的release不等待后续mul或GM输出；没有新增buffer、copy或child drain。UB仍为
原来的10528字节，4个父协议含2个局部栅栏及2个release channel。循环内无
`PIPE_ALL`，仅kernel返回处最终drain。完整对照在外层
`runs/refactor/native-guards/build/{source,before,after}.xdsl`及
`native_guarded_recurrence.asc`，不是只检查source IR就宣称同步正确。

同时修复共享符号求解器的一个错误：处理singleton时修改了外层分支的左端点，
污染后续affine分支，使只在N=2成立的首→尾移位路径被误判在N=1也成立。
独立BFS反例固定该错误；实际IR verifier还新增检查，父Set/Wait/barrier的
声明激活域必须落在其实际native存在域内，扩大或删除first-only条件会被拒绝。

fixed child的local row访问现在接受统一正stride的有限frame，逐项检查行孔洞、
frame内范围及最终allocation边界。runtime无界local stride、同root不同drift、
跨frame以及越界均明确拒绝；没有把不同view当作不同storage。

| 验证 | 当前结果 |
|---|---|
| scope/事件/符号关系、实际IR破坏与独立有限模型 | 666项通过，84.54秒；fixed local frame另8项通过 |
| 独立实际IR数值解释 | N=0/1/2/3/7，f32递推后f16输出精确一致；零次输出sentinel不变 |
| Python→ASC→CANN/shared，总期限300秒 | 6.640秒；CANN 9.1.0、dav-2201、`-O2 -std=c++20` |
| 设备，同一binary | 全卡健康/进程/显存/利用率检查后，物理6→逻辑0；N=0/1/2/3/7/31/257，各10次，输出/输入/保护区全通过 |

有限模型与设备点检不是任意规模形式化证明，本例也不是FA或吞吐量基准。
父完成图438节点/728边、30个内存义务，条件事实不按runtime N实例化。
四case仍有重复的叶分析与投影成本，未宣称编译目标已经达成。最终源码五个
独立冷进程计时（源码读取/构造至最终验证与ASC；import/CANN另计）如下：

| 输入 | p50 / max | RSS峰值 |
|---|---|---|
| native_guarded_recurrence | 1.9528 / 1.9667秒 | 41716 KiB |
| nested_copy | 1.3382 / 1.3411秒 | 39868 KiB |
| nested_cv | 1.6672 / 1.6690秒 | 41240 KiB |
| shared_l0c_qk_pv | 3.2294 / 3.2371秒 | 48608 KiB |

三个已有fixture的ASC均与上一阶段设备验证版本逐字节相同；本例冷测ASC也与
设备所用CANN manifest一致，核对结果在`validation.json`。未达到常用kernel
p50≤1秒目标，没有FA、ACLNN或profiling性能结论。

证据在`runs/refactor/native-guards/`的`regression.log`、`cold-final/`、
`build/`和`build/device/result.json`。设备所用ASC SHA256为
`05006bac6fa2afff4f4ec45826006c63ac771bb46ec56018b4fe8c083aca5ccd`。
以下命令从工作区根目录执行；运行前须再次确认物理6仍健康空闲：

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.support.run_native_guards --output runs/refactor/native-guards/build --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 PYTHONPATH=compiler .venv/bin/python -m tests.support.run_native_guards --output runs/refactor/native-guards/build --physical 6
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/tests/fixtures/guarded.py --entry native_guarded_recurrence --output runs/refactor/native-guards/cold-reproduce
```

**边界：**只支持一个实际parent时钟上的first/last、直接固定child与stationary
条件root。runtime周期谓词、条件parent ring、更深mixed query时钟、缺少跨parent
消费证明时的条件child event宽度调整仍拒绝。nested FA实际报mixed父serial限制；
continuous实际报`runtime-cycle predicates are unsupported`。下一步按原FA主体
接通serial query→runtime KV→fixed child及跨query最后reader/event消费，
不将`task%kv_tiles`冒充整个parent的首末条件。

## 已保存阶段 d83635c：保留逻辑结果的共享 L0C 生命周期

`tests/fixtures/nested.py:shared_l0c_qk_pv`现已走通公开DSL→实际IR验证→ASC→CANN→
设备。它是编译器反例，不是完整FA或性能基准：QK、PV使用独立L1/L0A/L0B，
复用原来的两个16×16 FP32 L0C槽（2048字节），10个native模板、3个call。
没有增加buffer、搬运或child drain，运行长度`n`仍是参数。

修复前，逻辑frontier要求`PV(t).FIX → QK(t+1).MAD`，与显式排步中先发
`QK(t+1)`再发`PV(t)`形成假环。现在先保存每个读的逻辑writer与RW累加链，再
按真实Call stage/issue次序安排独立定义的存储占用。GM仍保持逻辑顺序和最终值。
父层只投影实际跨调用义务的完成端点，不把所有内部定义都投影成父层状态。

自动生成的c0复用关系如下；箭头均为真实FIX退休到MAD开始，不是child退出：

| 激活条件 | 同步前后对应关系 |
|---|---|
| N=1 | QK(0).FIX6 → PV(0).MAD0 |
| N≥2，启动 | QK(0).FIX6 → QK(1).MAD0 |
| 0≤k≤N−2 | QK(k+1).FIX6 → PV(k).MAD0 |
| 0≤k≤N−3 | PV导出(k).FIX → QK(k+2).MAD0 |
| N≥2，排空 | PV导出(N−2).FIX → PV(N−1).MAD0 |

`PV(k).MAD7 → PV导出(k).FIX`的RAW保留；c1另外在FIX7退休后复用。
ASC使用实际parent generation和count条件，负代次差也由同一个来源epoch配对
Set/Wait；该例最终event ID均为常量。15个父channel中6个带条件，FIX→M bank
恰好使用8个ID。条件channel暂独占pool，不套用普通prefix channel的共享规则。
检查器从实际`ScopeChannelOp.activation`重建条件；缩窄启动条件、错位排空条件、
改错signed distance均被拒绝。event消费和内存依赖共用带count条件的完成查询。

独立有限模型直接解释实际IR的地址、native顺序及物理event ID，固定逻辑RAW/RW
版本后检查存储占用；覆盖N=0/1/2/3/7与漏启动/排空、漏Wait、错ID及错误producer。
它不是任意N的证明。设备使用物理6→逻辑0，同一二进制在这些N上各运行10次，
Y、最终Score槽、输入及保护区精确一致。运行前全卡健康/进程/显存/利用率快照
随结果保存。没有FA、ACLNN吞吐量或理论峰值达成结论。

编译成本来自固定child的byte切分、静态stage跨度内的占用族，以及完成图上的
条件查询；不枚举runtime N。裁剪无关端点、批量传播后，再将存储alias周期投影
到实际协议周期；不为双槽GM把整张完成图加倍。该投影证明更强的完整epoch区间，
仍检查首末条件，不能借用仅启动时成立的路径。父图由初次集成的1532节点/2498边
降至447节点/757边，89个去重查询、153个静态case、无runtime实例图。

相关回归457项通过；最后周期投影改动另通过55项针对性检查，固定child/event
summary及依赖frontier的131项回归也通过。最终编译器源码的
五个独立冷进程结果如下（含源码读取、前端、最终验证、快照与ASC，不含import/CANN）：

| 输入 | p50 / max | RSS峰值 |
|---|---|---|
| shared_l0c_qk_pv | 3.2273 / 3.2364秒 | 48520 KiB |
| nested_copy | 1.3640 / 1.3683秒 | 39772 KiB |
| nested_cv | 1.6559 / 1.6841秒 | 41164 KiB |

初次共享L0C集成单次观察约14秒；上述正式冷测仍未达到p50≤1秒目标。原copy/CV
ASC与上一设备验证版本逐字节相同；共享L0C最终ASC与本次设备binary来源相同，
SHA256为`f1452ba445d24c05c91afa19cbff39c3b9a19257dff1912d3deba0def0e65068`。
完整阶段、源码tree hash与冷测在`runs/refactor/shared-l0c/cold-final/`；设备及
CANN证据在同级`build/`。该次完整构建12.920秒，使用CANN 9.1.0、dav-2201、
`-O2 -std=c++20`，准确命令见compile manifest。没有把缓存取结果时间计为编译时间。

复现（工作区根目录）：

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --entry shared_l0c_qk_pv --cases 0 1 2 3 7 --output runs/refactor/shared-l0c/build --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --entry shared_l0c_qk_pv --cases 0 1 2 3 7 --output runs/refactor/shared-l0c/build --physical 6
```

`build/source.xdsl`、`before.xdsl`、`after.xdsl`及`shared_l0c_qk_pv.asc`分别保留
源码IR、同步前后IR和设备所用代码。构建使用一个覆盖Python与CANN的300秒期限。
本次定义重排仅支持stationary、actor-private、明确初始化的独立定义与只读导出；
incoming RW保留逻辑recurrence，不能当作fresh定义。周期capture、更深子循环、
native条件执行和跨独立runtime时钟尚未由本次能力接通；不存在静默重排。

## 已保存阶段 f91ac07：跨父调用的事件宽度与安全池共享

此前明确拒绝的`tests/fixtures/nested.py:nested_copy`现已通过公开DSL、CANN及
设备正确性。它保留两个固定child、4个native模板、4个父channel、原来的128字节
UB和全部搬运/同步位置；只改变已证明合法的event ID分配。

- child的固定Set/Wait边界改用通知ordinal直接算first Set/last Wait，每个实际
  使用的ID一条记录，不枚举child长度或整个period×width域。runtime与carried
  invocation仍使用其原有合同。
- 先检查现有ID的跨父调用寿命；只有失败时，才批量检查同一leaf图上的1..8种
  硬件宽度，并为候选端点构建一次父投影，不逐候选重编译。两个ready child
  都需要width=4，原来的2或3不足；增加ID不改变任何完成关系。
- 若各parent channel独占pool，反向bank需要child的4个ID加parent的6个，共10。
  两组ready/release通过实际消费路径共享width=2的pool后，parent只用4个ID，
  反向bank恰好8个；正向bank也恰好8个。没有增加buffer、barrier或搬运。

其中一组共享关系如下，`g`是父代次，子位置来自实际native前后：

```text
produce(g).store(j=2): Set c0, ID = 4 + g%2
consume(g).load(j=0): Wait c0
consume(g).store(j=2): Set c2, ID = 4 + g%2  // 仅g+2<N
produce(g+2).load(j=0): Wait c2
produce(g+2).store(j=2): 下一次Set c0
```

检查器从实际ID tuple重建每个channel的自复用证明及共享成员之间的Wait→Set
证明。仅支持同bank、同period/residues、完整相同tuple的共享；部分重叠、tuple
顺序不同、child/config冲突均拒绝。每个channel的自复用检查仍必需，因为尾部
有些成员不再发送，不能用共享关系的真空覆盖代替收尾检查。
分配采用有明确成本预算的确定性合并，不宣称全局最优，也不以ID相同制造顺序。

独立有限模型直接解释实际Call/Capture及父子Set/Wait，覆盖N=0/1/2/3/7的字节
依赖和真实物理ID消息配对；46项模型测试还覆盖旧width=2的event覆盖、错位
共享ID、漏Wait、RAW/WAR/WAW与死锁。模型不导入生产完成图或分配器，不是任意N
的形式化证明。设备使用物理6→逻辑0，一份binary在N=0/1/2/3/7/31/257各10次，
Y/O与输入精确一致，guard不变；全卡状态已记录（物理2当时有其他进程）。
源码→CANN/shared总构建6.671秒，受同一个300秒期限约束。

最终相关回归435项通过（74.53秒）；五个独立冷进程的源码→ASC结果为：

| 输入 | p50 / max | RSS峰值 |
|---|---|---|
| nested_copy | 1.3343 / 1.3421秒 | 39756 KiB |
| 原nested_cv | 1.6814 / 1.7103秒 | 40956 KiB |

原CV上阶段为1.6563/1.6663秒；本批不能据此宣称编译提速。两例均未达到常用
内核p50≤1秒目标。冷计时在`cold-copy/`与`cold-cv/`，每个进程的最终compiler
tree hash一致，ASC hash与设备所用CANN manifest一致。copy的ASC SHA256为
`12207bdcbd46233172fa7d2a50f0192ea7e38bfbeac8eda8a1dadeef52d753e0`。

证据在`runs/refactor/scoped-events/`，设备记录为
`build/nested_copy/device/result.json`，ASC及同步前后IR同目录。复现：

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --entry nested_copy --output runs/refactor/scoped-events/build/nested_copy --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --entry nested_copy --output runs/refactor/scoped-events/build/nested_copy --physical 6
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/tests/fixtures/nested.py --entry nested_copy --output runs/refactor/scoped-events/cold-copy
```

这仍是编译器正确性闭环，没有FA或ACLNN吞吐量结论。原`nested_cv`的ASC与
`4e23713`设备验证版本逐字节一致。单通知child每次调用都重启ordinal时，扩大
inner宽度仍不能解决所有跨父复用；缺少消费路径的情况继续拒绝。

## 已保存阶段 4e23713：保留实际父子流水与跨核协议

公开`compile(prim_func)`已接入实际`CallOp`与`CaptureOp`，外层`Pipelined(n)`
保持runtime参数，固定child保留独立时钟与native模板。capture只是原allocation
的地址别名，不分配新buffer、不搬运数据，也不提供完成语义。
当前接纳一个根Pipelined及其直接固定child；同层普通工作作为Once call保留。
跨层动态child、first/last分支和多个runtime轴仍明确拒绝。

`tests/fixtures/nested.py:nested_cv`是两组AIC/双AIV的编译器回归：Cube child
分两列块计算Gram矩阵并FIX导出；Vector child分行读取、乘二并写出。
11个native模板和3个call保留，父GM环为双缓冲。父同步直接定位到真实inner
native与child generation；不在child退出插入drain。同步前后的主要区别为：

```text
before: FIX Score[t%2]    / 两个AIV各自MTE2读取Score[t%2]
after:  最后FIX → ready Set(0) → 两个AIV各自第一次MTE2前Wait(0)
        两个AIV各自最后MTE2 → release Set(1) → FIX(t+2)前Wait(1)
```

ready覆盖全部已写列块，release等待两个真实reader的MTE2退休；无需等待其后
Vector计算或输出搬运。L1/L0A/L0B/L0C/UB的跨父调用复用另外从实际最后reader
生成本地event。最终verifier从实际IR重建地址、端点、quorum及event消费。
同一native侧的多个父同步必须符合声明顺序并紧邻该native，不能跨越child同步。
已有反例证明交换local Wait与cross Wait会破坏旧Wait→下一Set，不能只核对配对。

父图只压缩已证明的child内部关系；不压缩父协议环。SCC后的局部DAG用拓扑序
与bitset直接约简，未对真实同步做“试删再验证”。初始CV父图的7,551条边约简至
554条（补全FIFO出口前的结构测量）；补齐FIFO与跳过同步的透明前缀后为415节点、
663边，仍为11个native模板、12个父channel和2个跨核counter。实际指令未因关系
约简删除，不使用带cProfile的耗时代替冷计时。

初始化检查另发现并修正实际IR漏洞：child内部残留`core_stride/core_terms`会在
blocks=1的分析视图中丢失，父摘要可能把另一核读到的未写区域误认成已写。
现在明确要求祖先核坐标/count平移全部进入Capture；真实child残留分量会拒绝。
错误根、span、对齐、双clock、隐藏effect及count+stage溢出均有mutation回归。

复杂度边界：runtime父长度N不枚举。固定child长度C仍参与访问footprint扫描，
readonly family可保留C个端点；交叉访问切片的最坏成本可为二次，不能宣称与C
无关。`parent_access_visits`记录child片段、child摘要、父周期片段与cut成本，
累计上限100万；父存储周期来自Capture的modulo，不把child长度当作周期。
完成投影的端点两两关系也单独计数并限预算，约简没有取消其提取成本。

| 当前证据 | 结果 |
|---|---|
| 受影响的Python/IR、符号关系、独立有限图及代码生成回归 | 477项通过，91.97秒；不是单算子编译时间 |
| 五个独立冷进程，源码读取/构造→最终检查→ASC | p50 1.6563秒 / max 1.6663秒；import另计，RSS峰值40 MiB |
| Python→ASC→CANN/shared，总期限300秒 | 6.476秒；其中CANN 4.774秒；`-O2`、`dav-2201` |
| 同一binary设备正确性 | 物理6→逻辑0；全卡状态检查后，n=0/1/2/3/7/31/257，各10次；全部输出、最终两个槽精确一致，input/guard不变 |

证据位于外层`runs/refactor/nested-cv/`：`regression.log`、`cold-release/benchmark.json`、
`build/source.xdsl`、`before.xdsl`、`after.xdsl`、`nested_cv.asc`及CANN manifest；
`build/device/result.json`记录设备、shape、重复数、源码与库hash及结果。
ASC SHA256为`fa18b42dd67aa464e090a3e4fe0665dfb2f79da458cf026d5ee76df0e3192bea`。
该小回归没有吞吐量、ACLNN或profiling结论，也未达到常用内核p50≤1秒的目标。
复现命令从工作区根目录执行，设备命令仅在物理6仍健康空闲时使用：

```bash
PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --output runs/refactor/nested-cv/build --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 PYTHONPATH=compiler .venv/bin/python -m tests.support.run_nested_cv --output runs/refactor/nested-cv/build --physical 6
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/tests/fixtures/nested.py --entry nested_cv --output runs/refactor/nested-cv/cold-release
```

**该阶段记录的两个具体反例：**

- `nested_copy`的child本地event只按inner generation轮转；跨parent调用仍可能
  在上次Wait前re-Set；已在本页当前阶段通过宽度与事件池证明解决。
- `shared_l0c_qk_pv`保留共享L0C，QK/PV操作数独立。现有logical frontier要求
  `PV(t).FIX → QK(t+1).MAD`，与显式`QK(t+1) → PV(t)`派发相冲突，当前报cycle。
  后续先固定逻辑RAW及RW recurrence，再对独立结果的物理生命周期推导复用；
  不按physical issue重绑所有read。协议需要独立的端点代次与激活域，处理负逻辑
  代次差和启动/排空。这个反例不是FA已接通的证据。

## 已保存阶段 a788cbc：真实作用域与固定 child 边界

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

**该阶段未完成：**父级Pipelined中重叠的child调用、父代次地址捕获、跨child的双AIV
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
| AST DSL → source IR → physical IR → AutoSync → ASC | runtime 局部流水、Vector/归约、显式 Cube、受限 serial parent/child、固定直接child的nested CV、BNSD cyclic局部寻址 | 分支/多层或动态child、独立生命周期重排、跨query多时钟与完整 FA 未接通 |
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
| CV | 原nested FA的query/KV/fixed-child已到物理IR，parent first/last保留；跨query同步仍拒绝 | 从真实capture和最后reader/event消费继续扩展；完整 QK/softmax/PV/update 纵向闭环 |
| MEMORY | cyclic局部不等长/空group已证明；共享 Q/K/V、间接地址及跨 child 复用仍不足 | 真实范围、有效域、最后读取和 owner 分区；buffer 分配/布局/可见性与同步一同验证 |
| SYNC | runtime叶与直接固定child已有双AIV quorum；跨parent事件宽度/池共享、stationary私有独立定义复用及首末访问域已接通；跨query消费与条件ring仍不足 | 共用硬件完成语义，分开物理访问与事件寿命；最终从实际 IR 检查，不能旁路或加任意 barrier |
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
