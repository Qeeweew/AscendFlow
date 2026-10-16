# 结构化循环与统一自动同步设计

2026-10-16。**这是替换现有算法的目标设计；M1部分基础组件已落地，完整主线尚未替换。** 重构前实验保存在bbc1458，当前实现事实见[状态](refactor-progress.md)，实施顺序见[计划](refactor-plan.md)。

## 1. 要替换什么

尚未替换的同步与生成主线按 finite / loop / sequence / invocation / scope / query 分派。`scope_lowering.py` 特别要求外层 serial 只包含一个 KV pipeline，child 必须是直接固定长度循环；原`ir.py`（现已拆入dialects/physical.py）的层次专用Channel/Sync尚未删除，`scope_query_completion.py`仍复制父流水图。

这些限制来自实现，不是 dav-2201 的要求。FA 的 Q 块、GEMM 的输出块、MoE 的任务只是不同算法的外层迭代，不应决定编译器选哪个求解器。普通顺序、嵌套循环、局部多级流水、CV 排步必须组合相同的语义。

现有全来源条件传播不是重构后的默认算法。它按来源、位置、循环区间、长度条件保存事实；增边和新查询会触发重放，区间合并又扫描已有事实。小图仍可能产生百万次传播。换包名、压缩节点、缓存求交结果都不能替代算法变更。

## 2. 源语言和 IR 边界

用户继续用 TileLang 风格 AST DSL 表达 tensor、物理空间、views、serial/Pipelined、Stage 和 Cube/Vector 域。stage offset、buffer slot 数量、event width 分开。普通循环无论常量界还是 runtime 界都保留；只有显式 unroll 授权展开。旧 `cube.task/vector.task` 不恢复。

保留三种可打印产物：source xDSL、未同步 physical xDSL、同步后 physical xDSL。两套 dialect 足够，不为每个 pass 再造一棵程序树。分析结果只引用本次编译持有的 IR 与不可变描述。

- Source IR 保留结构化控制、逻辑值、显式布局/缓冲和排步政策。
- Physical IR 保留相同控制层次，操作已经具备完整 native effects、engine、owner、地址和插入位置。普通循环和 pipeline 使用同一循环语义，后者额外携带固定排步。
- 同步后 IR 在真实位置增加有类型的通知、等待、必要的同流水 fence 和明确的生命周期收尾。不存在 QueryChannel、InvocationChannel 这样的算法/层次专用类别；机制按硬件能力区分。

xDSL 是控制、SSA、region、rewriter、verifier 和 pass 基础设施。通用优化只处理确定纯的标量子图；异步 native、SPR、通知和 ownership 都具有 effects。

## 3. 循环统一为 prologue / loop / epilogue

控制上的循环处理可以直接、统一。对 N 次逻辑迭代，Stage s 在本 actor 的物理 step j 执行逻辑迭代 `g=j-s`，存在条件为 `0<=g<N`。令 S 为最大 stage offset，最小 offset 已归一为零。

例如 `stages=3` 使用偏移0/1/2时，S=2；S是填排深度，不是buffer数或DSL的stage数量。

```text
N = 0：没有循环体 native 及其配对通知，边界状态透明传递
0 < N < S：一个短片段 j ∈ [0, N+S)，保留各 stage 的存在条件
N >= S 且 N > 0：
    prologue：j ∈ [0, S)
    loop：    j ∈ [S, N)
    epilogue：j ∈ [N, N+S)
```

S=0 就是普通循环。短循环不得机械叠加相互重合的 prologue 和 epilogue。N=S时loop零次重复为identity，短片段不能借用尚未建立的稳态假设。上述是执行域分解；允许保留小循环，不要求复制每个 step，更不枚举 N。FIRST/LAST 在 N=1 同时成立，保持原词法次序和真实次数。

S只划分stage的执行域，不意味着同步在S步后已无首尾条件。例如S=0的普通双缓冲仍有距离2的release，首两代不等待旧reader，末两代不通知不存在的下一次覆盖。此类guard由carried依赖和event合同导出，可保留在loop中，不强行增加与N相关的展开。零次循环继承的token以及外围资源自身的合法seed/cleanup，仍按各自生命周期处理。

分析使用一份参数化循环体；首尾按这些域代入。两端外部连接和循环回边由同一组合规则产生：

```text
前置节点 → entry → prologue → loop 的参数化重复 → epilogue → exit → 后置节点
                                      ↖ carried state ↙
```

这张图只表示控制结构。entry/exit 是一组带类型的状态端口，不是“全部工作完成”的两个节点。epilogue 指排空已安排的 stage；它不自动等待所有执行引擎退休。

例如循环已派发 MTE3 从 UB 读出，随后 Vector 覆写 UB：exit 连接必须保留 MTE3 源读取的退休条件。若后续仅使用另一块 UB，则不需要等待它。外部消费者通过相应端口连接，不从控制退出推导完成。

## 4. 两端到底传递哪些状态

每个顺序 region / 循环具有同类边界合同，字段只为真实外部依赖建立：

| 状态 | 保留内容 | 不能推出 |
|---|---|---|
| 控制和坐标 | 当前循环坐标、存在域、父坐标绑定、stage/issue order | actor 同 step 同时完成 |
| 存储 | root、精确范围、版本、slot/generation、未退休 reader 前沿 | 不同名字不别名；写入名字更新意味着旧读已结束 |
| 完成信息 | 各 actor/engine 合法 issue、admission、retirement/capture 前缀 | 一个 native 的 end 等于整条流水完成 |
| 通知状态 | 逻辑 event 的产生、消费、序号、未完成 lease、参与者 | flag 被消费等于任何未被捕获的内存访问完成 |

输入未被该 region 改变的事实直接透传；执行部分贡献新事实；未执行分支不凭空产生贡献。无写循环中的多个旧 reader 只有被可信退休前缀覆盖后才能压缩。

边界端口引用实际内部 milestone 和插入位置。比如某个输入在 child 的早期读完，释放通知仍可以留在该读取之后；不能因为导出 exit 合同就把 Set 移到 child 尾部。每个物理 slot 最后一次使用可保留为符号表达式：余数 r 的末代为 `N-1-((N-1-r) mod P)`，并以该代非负为存在条件。

嵌套循环按后序生成一次合同；父循环绑定 child 的地址、坐标和外部状态。父层只组合真正跨边界的内容，不复制 child 的内部完成图、不把所有内部访问变成全局 origin。多层嵌套沿用同一过程，没有“外层 query”分支。

例如 `前置W → 循环中R(i) → 后置覆盖W'`：入口需求引用前置W的ready，循环出口保留相关R的累计退休前缀，W'只取得这个release。N=0时R不存在，入口状态直接传到W'；不能丢失前置W与W'本来存在的WAW。这些连接是依赖和状态传递，不要求在控制入口/出口各插一个Wait。

这不意味着固定两个标量端口足以描述任意循环：宽度来自实际存储/流水/通知边界。它必须被计量，不能把无界内部状态藏进 summary。

M1的表示合同必须具体到：键由真实端口、actor/engine、里程碑类型、owner和必要的slot/guard分区组成；值表达输入状态的透传或已建立的参数化前缀，包含内部消费者的入口要求。禁止以历史查询来源或完整路径作为键。条件只来自输入控制、实际依赖距离和资源序号，不允许在路径传播中任意生成条件组合。归一化、同键比较/合流必须幂等；重复/重排外部查询不能扩大已建立状态。

固定政策下，边界键数和每键允许的表达式条款上界必须能从控制、effects和依赖推导，并分别证明组合/回边保持；不能把实际膨胀后的任意数量命名为B/G就声称复杂度受控。上述闭合规则及其适用域仍是M1必须完成的算法交付，当前设计不宣称已得到普遍的有限前沿算法。

## 5. 同步生成：物理依赖 + 完成前沿

### 5.1 一次建立访问与依赖

native lowering 完成 layout、subview、recast、scratch、SPR effects。依赖分析按 root 和精确访问描述处理：

- read 绑定正确的 reaching definition，形成 RAW，并加入未退休 readers；
- write 对旧定义形成 WAW，对相关 readers 形成 WAR，再更新定义；
- slot 复用形成带代次距离的依赖，owner/guard 约束两端的真实存在；
- 连续/规则 strided 区间保留描述，必要相交处分割，不枚举每个字节或完整迭代历史。

逻辑值依赖与物理复用关系分别保留。只保存最后 writer 不能为历史 read 重新绑定定义。未知动态 alias/owner 要有可验证合同，否则诊断拒绝。

### 5.2 消费已有完成事实，为缺口构造协议

沿固定 issue 顺序遍历，每个消费位置持有已经取得的完成前沿。对其依赖直接查询相同 clock/owner/guard 下的退休或就绪信息：

```text
for consumer in 固定排步:
    needs = 本位置的真实 RAW/WAR/WAW 要求
    missing = needs 中未被已取得完成前沿覆盖的部分
    在合法 producer capture / consumer admission 位置构造协议
    按协议的硬件合同更新受影响流水的完成前沿
```

完成前沿以硬件允许的累计前缀为基本单位，不以每个访问点分配一个全局来源位。可比较前缀使用序号/代次关系；不可比较的 owner、guard 或退休种类分别保留。不把 `covers()` 实现成隐藏的全源闭包。

Wait 合并它实际取得的事实；Set 捕获源流水当时合法的退休信息，不能自我反馈成后续操作的退休。已由同一路径携带的事实直接省去新同步，不做“插入所有同步再逐个删除并重检”。分支合流要求每条实际路径成立：互斥路径的事实不能无条件做最大值。

### 5.3 循环回边建立有限的关系

双缓冲的必要关系可以直接保留为模板：

```text
ready:   write(i) → read(i)
release: last_read(i) → write(i+2)
```

对循环，先由访问/slot 确定 carried 义务，再构造首轮条件和稳态协议；在边界状态上建立“进入满足合同 → 一步保持合同 → 退出导出合同”的归纳关系。普通执行不模拟 N 次，不靠复制多个完整循环来猜成立。

这一步还需要实现并论证：前缀的代次表达、受限 guard 合流、跨嵌套循环的边界状态闭合。不能用一句“求不动点”隐藏不受控的迭代。若闭合需要新增条款，每条必须对应真实边界/依赖，统计新增原因；无法在支持的符号域表达就明确拒绝。具体比较、收敛规则和成本反例必须在第一阶段定下来，再迁移 FA。

## 6. 局部流水与 CV 共用关系，不混用机制

Scalar、Vector、MTE1/2/3、Cube、Fixpipe 的原生隐式顺序以 dav-2201 合同为准。完成前沿不默认所有同 pipe 都 FIFO 完成；短 MAD、Set 的后续发射等既有硬件反例必须保留。

CV 仍用同一物理依赖和循环端口：FIX 写 GM 的 ready、两个 AIV 的相关 MTE2 reader release、跨核可见性和通知参与者分别记录。两路 AIV 的必需贡献是 AND；同一事实的替代路径才是 OR。cross Wait 会阻塞控制器后续派发，因此等待位置同时影响进展与性能。

资源机制有不同实现：本地二值 event、跨核 counter/quorum、必要的同流水 fence。统一算法不等于把它们强行降成一个 event 规则。

## 7. event 与进展

同步生成产出逻辑通知和 lease：源 capture、目标 acquire、存在条件、ordinal、consumer 集合、上次消费到下次 Set 的关系。资源分配消费这些关系，处理硬件 bank、ID、width、配置占用和父子共享；不因 ID 改名重建内存依赖。

一个 lease 可以跨 prologue/loop/epilogue 和相邻外层迭代存续。这三段不是各自 seed/drain 的调用。startup 只预置语义上成立的 credit；尾端不为不存在的循环体消费者发通知，或在明确生命周期边界配对收尾。外围资源自身的seed/cleanup由其生命周期决定，不简单以N>0作为条件。kernel 返回处的 drain 与一般循环出口严格分开。

不能假设有数据顺序就有 event 空闲，也不能假设配对正确就无死锁。阻塞进展在固定排步及新增协议约束上建立一次；使用已有 witness，必要的约束求解单列成本。一般资源冲突不保证线性，不能用容量常数 8 掩盖两两共享搜索，也不保证任意合法政策都能由启发式分配成功。

## 8. 复杂度约束

设 T 为 native 模板，A 为访问描述，I 为真实相交记录，H 为依赖，B 为边界条款，C 为逻辑通知，J 为资源冲突，P 为实际耦合周期，G 为条件条款。N 是运行长度。

| 阶段 | 必须可见的工作量 |
|---|---|
| 前端、native lowering、PLE 构造 | AST/T/表达式 DAG；不含 N 次循环展开 |
| 连续区间索引 | `O(A log A + I)`；复杂布局求交另计 |
| 循环摘要组合 | child 内部分析一次 + 实际边界绑定/条款合并；不复制内部图 |
| 同步生成 | H、前沿比较/更新及 P/G 条款；记录最大宽度和总比较数 |
| 资源/进展 | C、J、实际约束/候选访问数及求解成本 |
| 输出 | 实际 IR/ASC 大小，不把求解放进 emitter |

固定政策的 N 改变不得使结构和工作量随 N 增长。固定 tile、stage、slot、真实布局相交和嵌套边界增长仍须计量。独立 ring 不取全局 LCM；真实耦合周期不能假装不存在。任意条件/间接访问不承诺线性；复杂输入须显式限制或给出具体拒绝，不加任意 barrier。

算法验收先看工作量，再看冷编译：默认 FA source→ASC 五个独立进程 p50≤5s、max≤10s。不放宽阈值、不改算子政策、不用跨编译 cache。lowering/emission/snapshot 也纳入总时长，不能只优化同步后宣称达标。

## 9. 不变量、验证与边界

每个变换有明确前提、建立项、保持项和失效项，详见[合同](compilation-invariants.md)。normal 编译消费已保持事实；debug 和外部同步 IR 才独立重建检查。验证实现遵守规则与每次编译反复证明同一事实不是一回事。

第一轮必须覆盖现有真实 DSL 的 Vector、normalization、GEMM、状态算子和 dense FA；保留早 release、buffer reuse、event 借用和固定 CV 排步。动态间接 ownership、任意布尔 guard、continuous 多时钟状态和 paged/persistent attention 仍需各自明确合同，不能靠命名一个接口宣布支持。

循环和包结构重构之外，标量化简、错误诊断、测试独立性、产物溯源和硬件性能验收也在[计划](refactor-plan.md)中。当前不恢复算子优化。
