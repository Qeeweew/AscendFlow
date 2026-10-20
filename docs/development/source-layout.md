# AscendFlow 源码布局和唯一编译链

规则及事实归属见 [generation-rules.md](generation-rules.md)，验证方法见 [validation.md](validation.md)。

## 产品与包边界

产品名 **AscendFlow**；安装项目、Python 包与命令均为 `ascendflow`。
xDSL 0.73.0 是实现所依赖的 IR 框架。以下目录和源码路径均相对于本仓库根目录。

```text
AscendFlow/
  pyproject.toml          # ascendflow 包映射到 src；统一 CLI
  src/                   # 直接存放产品代码
    __init__.py          # ascendflow 公共 API
    cli.py               # 命令行入口
    frontend/            # TileLang 风格 DSL
    dialects/            # 基于 xDSL 的 source/physical IR
    analysis/            # 内存、依赖、同步与资源事实
    passes/              # lowering、生成和物化
    backend/             # ASC 与 CANN
    runtime/             # 构建期限和进程组
  examples/              # 算法及政策
  tests/                 # 语义与集成回归
  benchmarks/            # 编译及后续设备测量
  docs/                  # 已落实的设计、合同与使用指南
```

`src/` 直接放置代码，没有额外的 `src/ascendflow/`。项目配置将逻辑包
`ascendflow` 映射到 `src`，子包仍使用 `ascendflow.analysis` 等导入名，
不把 `analysis`、`runtime` 等通用名字暴露为顶层包。独立 venv 的开发安装
直接指向当前 `src`；CLI、测试和 benchmark 子进程使用同一份源码。
`ascendflow` 与 `python -m ascendflow` 都调用 `cli.main`。profiling 使用
`ASCENDFLOW_PYTHON`；构建组期限使用 `ASCENDFLOW_BUILD_*` 环境变量。
不提供旧包或旧环境变量的兼容入口。

## 编译链与职责

```text
compile / Compilation
  synchronize_structure
    lower_native_structure                # DSL→source、原式检查、native lowering
    collect_native_memory
    bounds → admission → dispatch
    access_order → dependencies             # 可能冲突顺序、region 队列端口和局部 packet
    generate_notifications
      bind_notifications                 # 覆盖、真实位置及原始 ownership edges
      EventDomains                       # 有序原式定义性、坐标往返、配对
      establish_queue_order              # 逐层 physical-step / phase 准入
      construct_cell_channels            # 物理 lifetime credit 归纳
      HandbackRule                       # 无稳定 cell 的有限/ring 生命周期
    allocate_structured_events           # 消费生命周期，建立 bank/ID/共享新约束
    materialize_notifications
    ScalarFacts                         # 复用原始控制和合法索引范围
    executable projection               # 独立终端模块；映射 native、geometry 与 SSA 范围
    lower_execution                     # actor 本地时钟与存在域→arith/scf
    simplify_scalars → simplify_conditions
  backend.structured → ASC
build_file                               # 同一链路 + CANN，单次进程组硬期限
```

`NotificationProgram` 由生成器私有构造；coverage-only selection 不能传给分配器。
`generated-lifetimes` 是报告来源标签，不能作为外部验证凭据。公共 compile
只接受 source，外部 physical snapshot 不进入正常编译链。

| 文件/范围 | 职责与输出 |
|---|---|
| `frontend/`、`dialects/source.py`、`language.py` | TileLang 风格 AST、typed source、逻辑 Tile 和显式 buffer |
| `passes/lower_pipeline.py`、`lower_native.py`、`native_recipes.py`、`native_addresses.py` | 参数化控制及 native recipe；尚未建立异步完成 |
| `analysis/structured_control.py`、`structured_memory.py`、`native.py` | 原始有序算术、真实地址/effects、绑定真实循环的条件读取；消费共享目标描述 |
| `target/native.py` 的 `NativeSpec` | 一份不可变的 native opcode、参数、operand geometry/effects、packet、read reset 与完成规则；逻辑分析和终端共享，不含地址/代次 |
| `analysis/smt.py`、`integer_facts.py`、`guard_cover.py` | 原式数学关系与条件覆盖共用 Z3 查询；定义性和机器范围独立，输入环境与完整前提限定缓存 |
| `analysis/structured_bounds.py`、`structured_admission.py`、`structured_host.py` | bounds、外部别名、实际存储实例、初始化、借用 Tile 及整数范围前提；group/peer 独立性属于源程序合同 |
| `analysis/structured_access.py`、`structured_regions.py` | 按真实地址周期建立可选精确 cell 与可能别名的 AliasRegion；保留实际 native 访问，不将 may-write 当完整定义 |
| `analysis/access_order.py`、`storage_packets.py`、`storage_word.py`、`structured_dependencies.py` | 可能冲突的 RAW/WAR/WAW、条件访问的并集及完整入口可用域、单次激活证明、共同队列端口、局部完整 packet 的实际复用 |
| `analysis/local_schedule.py`、`structured_dispatch.py` | 按迭代域和 actor family 建立本地 stage 顺序与 horizon；派发消费该顺序并绑定 native、控制入口/出口的位置 |
| `passes/notification_binding.py`、`analysis/structured_protocols.py` | 复用已经必要的 capture/acquire、建立完整 quorum；保留自身 origin，不将新增 coverage 当作 ownership，不按 Stage 合并独立完成边界 |
| `passes/generate_notifications.py` | 唯一通知生成入口，组合规则并输出完整生命周期 |
| `analysis/event_domains.py`、`event_windows.py`、`event_families.py` | 配对及按需的精确成员、FIRST/LAST、非空投影；稳定 cell 不要求 dense rank |
| `analysis/queue_order.py` | 每层 actor clock-origin 差约束、零距离 DAG 与正距离前进；child 仅导出 open ports 和 may-block cuts |
| `analysis/cell_channels.py` | 相同完整物理 ownership 集合和 role 的非重复 credit；原始值/存储 lifetime 归纳 |
| `analysis/notification_lifetimes.py` | 有限使用或明确 queue handback，连接实际首末端口，无任意路径搜索 |
| `analysis/event_sharing.py`、`structured_resources.py`、`event_contracts.py` | 有限 bank、配置/drain 台账；固定成员的兼容关系只建立一次，候选 ID 分配复用真实 handback 或单发布者 FIFO，并检查完整 counter 积压 |
| `passes/materialize_structure.py` | 插入确定动作、共享同位置短路 guard 前缀；不改变 native 或 issue order |
| `passes/executable.py` | 终端模块、native 对应、共享 NativeSpec 与最终 SSA 范围；不携带逻辑控制、地址证明或同步求解入口 |
| `passes/lower_execution.py` | 将 count/stage 表示降为实际 arith/scf；按 family 保持 issue order，移动原 native，不复制双 AIV body |
| `analysis/scalar_uses.py` | 终端 lowering 和输出共用的逐 family SSA 使用关系 |
| `passes/simplify_scalars.py` | 复用源循环联合范围；共享仿射视图、倍数事实、支配 CSE、域内化简和安全外提 |
| `passes/simplify_conditions.py` | 用实际loop/祖先/分支前提化简条件并集，共享连续条件块，合并互斥事件通知及event ID；同一完成前缀的连续 fence 按幂等规则合并，保留重叠 Set/Wait 贡献 |
| `backend/structured.py`、`backend/cann.py`、`build.py`、`runtime/process.py` | 打印实际 IR；CANN 离线编译与 300 秒组期限 |
| `tests/models/structured_events.py` | 独立有限队列/二值 flag/quorum/字节版本检查，可对照 pre-sync 快照 |

旧 `event_progress`、`event_paths`、`event_lifetimes`、`progress_*` 和
`structured_event_sharing` 已被删除，测试不保留旧 solver。历史源码通过 Git
追溯，不另建 legacy 或转发别名。纯整数查询仍负责真实前提，不能以删除所有
`prove` 调用代替重构。

局部 WAR/WAW 通知前缀保留 `StorageCell.physical_key` 对应的 ownership 集合，
使循环内及 FIRST/LAST 边界继续属于同一 credit。RAW 的覆盖合并与跨参与者
quorum 保持各自合同。原式数学查询与最终 SSA 条件查询共用 SMT 服务，
但各自的求值定义域和机器范围检查仍独立保留。

算法和政策在 `examples/`；测量及独立 AscendC/ACLNN 在 `benchmarks/`。

前端 `@T.inline` 只做保留词法作用域和源位置的 AST 内联；`T.tile_ranges`
只拆 constexpr 整块/尾块区间。两者都直接形成原有 source IR，继续同一条
native、bounds、ownership、同步和资源分配链，不是第二编译路径。

标准连续 b16 `C=A@B` 与尾块的合同见 [标准 GEMM](../operators/gemm.md)。
类型转换、补零及 strided NZ 几何经过相同 native/effects、bounds、ownership
和同步链；封闭 packet 的内部顺序不提供外部异步完成保证。
L0C NZ 允许保留全部 M 行的、按 16 列对齐的列切片；它仍是紧凑物理平面，
可由独立 MAD 定义并由一次完整 FIX 读取。部分 M 行不能套用此合同。
二维 FP32 `N.add` 支持按 32 B 对齐的 UB 行距，每行包含完整的 64-lane 分组，
至多 255 行、repeat stride 为 1..255 个 32 B 块。lowering 选择完整 normal-mask
repeat packet，访问和初始化检查保留真实行间空洞，结果可写入紧凑的 UB 行。
UB 的跨步矩形 `N.copy` 使用有完整 mask 设置的 Vector bitwise copy：每行
32–256 字节、至多 255 行，行距与起点按 32 字节对齐，repeat stride 不超过
255 个 32 字节块。它保留真实行与间隙，源/目标必须证明不重叠；未知别名不准入。
这与连续 UB copy 使用同一 source/native/同步链，没有标量逐元素搬运。
复现命令将运行产物和诊断写入已忽略的 `build/`，文档不依赖仓库外的实验目录。

## 逻辑迭代与本地排步的最小表示

source 的 `Dataflow` 和单 actor `Pipelined` 统一形成 `IterationOp`，
它只保存逻辑次数、坐标及政策；Stage 在展开 inline 后绑定实际 actor 和最近迭代域。
不保存全 actor horizon，不重复保存父 stage。同步分析使用
`PipelineLoopOp(count, issue_order)` 和
`PipelineStageOp(offset)`；stage 只有一个 index 块参数 `g`。
`g=j-offset` 且 `0<=g<count` 是完整存在域。源程序的普通 serial 循环
仍保留自己的 lower/upper/step；pipeline 不重复保存这些字段，也没有 phase 表。
各 family 最大偏移由其实际 stage 推导，没有声明的 stage 数或共同 horizon。

派发时建立的 `LocalSchedule` 只引用该逻辑循环内实际参与的 actor stage，
保存本地发射顺序并推导本地 horizon；它不建立完成或事件消费证据。
物化后的 owned clone 显式映射该排步，终端 lowering 不重新选择 issue order。

`lower_execution` 在同步与事件分配之后，把该表示降为 `arith/scf`。
每个 actor family 生成自己的 clock loop，使用自己的 horizon 和安全移位范围；
同一物理 IR 保留 Cube/Vector 的发射政策，
两个 AIV 共用一个 Vector body。只有纯 SSA 和控制壳按 family 投影，
native 和保留的通知移动原对象；地址 operands 显式映射到新的 SSA。
配置等有 effects 的操作必须明确处理，未知操作拒绝降低。

次数、时钟、偏移与存在条件都是可化简的 SSA。常规时钟为 `[0,N+S)`；
若合法 N 可能使 `N+S` 溢出，使用 `[-S,N)` 和相应的 stage 坐标平移。
选择在编译期完成，不生成运行时分派。平移坐标在合法 guard 内求值，
不会为未激活的 stage 提前计算可能溢出的 ordinal。零次和短循环遵守同一规则。

MAD 首轮/周期 clear 所需的 ordinal 是显式 index operand，
不靠 ASC 后端查找已经替换的 stage。`npu.loop_id` 和 stage offset
保留控制对应，独立有限模型从实际 guard、地址和指令检查执行。

## 标量与条件的共同事实

可执行标量使用 xDSL SSA，`scalar.Expr` 是原始整数分析的视图。
`symbols.Symbol` 绑定实际 SSA/control 身份与明确种类，名字只用于诊断；
metadata/carry 捕获 group、SID 与祖先 loop ordinal，并随同时坐标替换特化。
`analysis.smt` 在一次编译内按完整不可变输入快照共享查询服务，不跨构建保存。
原式索引的全局/词法域非负性也询问该服务，完整保留实际 coordinate bounds；
不再用手写余数符号界、min/max 候选或递归坐标消元作第二套 sign 决策。
原式定义性先检查；数学化简不能掩盖非法 unsigned 求值或原始溢出。
`ScalarFacts` 复用已建立的控制域、联合 owner/iteration 范围和合法索引前提。
整数与谓词 pass 共用以真实 SSA 为原子的仿射视图，并保留非线性、除法和余数节点。
作用域分别记录区间、直接关系、倍数和已知谓词；通知域把实际 SSA 范围与
合法 modulo 关系送入同一 SMT 服务，使用数学布尔组合决定可达性、蕴含与覆盖。
保留紧凑 guard 的语法构造，删除旧差分闭包、余数组合搜索和覆盖减法。
区间查询按 SSA DAG 与完整前提复用；显式范围更新、父域变化或源范围放宽
使相关 scope 的区间/affine 视图失效。新增未被查询的 SSA 定义不清空既有结果。
同一 scope 树共享输入变更时钟；没有变更时，查询不再逐层检查祖先。
时钟变化只触发真实 parent/source revision 检查，不直接清空无关兄弟域的缓存；
源范围放宽和已观察父范围变化仍会使依赖结果失效，不授权复用旧前提。

默认 FA 是联合范围的一个例子：`T=HQ*query_tiles`，默认 `B=1`，
活跃迭代满足 `query < ceildiv(max(T-group,0),GROUPS)`，因此
`0<=group+query*GROUPS<T`，`batch=(group+query*GROUPS)//T` 化为0。
实现消费通用循环关系，不识别 FA 名称或特定 shape。

嵌套时，父 stage 提供 `0<=gp<Np`，子域叠加 `0<=gc<Nc(gp)`。
内层分支只细化当前 child 域，退出后恢复父域；兄弟 stage 使用各自的
`j-offset` 绑定，不枚举各层三阶段组合。

## 公共条件与剩余条件

通知先合并执行域，再在公共 guard 内构造 event 的剩余选择条件。例如：

```cpp
if (g == 7) {
  notify((a || b) ? 6 : 7);
}
```

内部选择式不再重复已经成立的 `g==7`。只有互斥分支的同类事件通知可以共享
一次指令，重叠分支保留原 Set/Wait 次数与次序。同 pipe、同参与者、同完成
前缀的连续 fence 可合并重叠条件；native、循环、其他通知及未知 effects
阻止跨边界合并。连续指令共用相同 SSA guard，协议来源全部保留。
结构边界先判断一次，局部域内的区间、余数、条件规范形和并集查询按不可变
输入复用；这些表仅属于本次 packet，不跨编译保存事实。

条件变换新建或移动的标量加入受影响集合。最终清理只对这些定义重新检查
合法外提，并传播域内常量、CSE 和 DCE；正常链路只调用一次全模块 arithmetic
canonicalization，不反复运行整个标量/条件 pass。

移动定义时，其旧域中的范围事实失效，必须改为目标求值域的范围。
局部非负范围不能授权在空循环或未进入分支上提前做 unsigned 除法、余数或
可能溢出的计算。未知优化事实保留原式，不跳过 admission 或同步检查。

## ASC 输出

`StructuredSynchronization` 保留的 structure/admission/dependencies/resources 是逻辑
诊断，指向没有被终端重写的同步后逻辑模块；`module` 指向独立的
`executable.module`。投影时显式建立 native 与 SSA 对应，复用已准入的几何和
范围，不重跑内存分析。`ExecutablePlan` 仅检查终端 native 对应及逻辑控制已消除，
保存只读的最终 SSA 范围。它本身不是外部输入的同步资格；公共 compile 仍只接受 source。
每个 `NativeExecution` 共享原分析建立的不可变 `NativeSpec`，交接时检查 opcode、
静态参数及地址 operand 数量一致。后端直接消费其中的几何和 packet；地址、
scalar SSA 与 scalar literal 仍取实际最终 IR，不在分析里保留另一份 literal。
ASC emitter 只接收该终端数据，不再读取 `NativeMemoryFacts`、`ControlFacts`、
`MemoryTemplate` 或原始地址表达式。

后端打印最终 `scf.for/scf.if` 和实际地址 SSA，不构造循环次数、stage 偏移
或阶段状态机。表达式内联按运算代价和使用次数选择；共享的复杂谓词保留
SSA 名称，`i1` 输出为 `bool`。只有已知 operands 非负且除数为正时才省略
unsigned 除法/余数的显式转换。普通缓存的 launcher 只有 kernel 调用和
`return 0`；显式 GM cache bypass 先查询目标 runtime 的同一 allocation 地址
别名，错误直接返回。它不改变物理 root 或在 kernel 中增加同步。
