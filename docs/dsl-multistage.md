# DSL规格：TileLang风格的结构化多阶段程序

2026-10-14修订，对照实现`be957e8`。本文的“目标”是待实现合同，“可运行”只用于
已经接通的子集。架构见[design](design.md)，工作次序见[refactor-plan](refactor-plan.md)。

## 1. API与语法边界

使用`import xdsl_npu.language as T`，设备函数不作为Python执行或tracing。
前端解析受限AST，产生xDSL控制流；宿主Python只选择显式constexpr配置。

| 语法 | 语义 |
|---|---|
| `@T.prim_func` | 捕获源码、解析AST、保留source map |
| `T.Tensor(shape, dtype, access=...)` | 有类型的外部引用；shape可引用后声明的runtime标量 |
| `n: T.int64` / `B: T.constexpr` | 设备值 / 特化常量，调用时已知n也不自动特化 |
| `T.Kernel(groups, kind=...) as group` | launch和group域；mixed固定1 AIC+2 AIV |
| `T.Cube()` / `T.Vector() as lane` | actor作用域；lane为组内AIV编号 |
| `T.serial(...)` | 保留的设备顺序控制，迭代边界不隐含硬件drain |
| `T.Pipelined(N, stages=S, issue_order=...)` | N个逻辑任务与最多S个stage位移；不展开N |
| `T.Stage(s, name=...)` | 最近pipeline的代次偏移与排步标签 |
| `T.unroll(...)` | 显式constexpr展开，受结构预算限制 |
| `alloc_ub/l1/l0a/l0b/l0c` | 片上分配，容量与layout显式；view保持同一root |
| `copy/matmul/reduce/cast/...` | typed原语；具体支持见已有例子与诊断 |
| `@T.macro`（目标） | 受限AST内联；宏位于examples，不是同步或ASC逃生接口 |

保留现有`stages`拼写，不再增加`num_stages`同义参数；它只限制`0<=s<S`，不
分配S份buffer。首版要求显式Stage；默认每actor按源码的stage顺序派发，
`issue_order`则完整指定该actor的直接stage标签。标签只在所属pipeline内解析，
不可跨层引用child stage。源码语义规定读写版本，派发政策不能改变数值依赖。

前端应在源码位置报告未知参数、非法作用域、漏/重复标签和不支持控制流。
不执行任意Python注解、闭包或调用；不以退回tracing解决解析失败。当前并非
表中所有语法组合已能lower，实际矩阵见末节。

## 2. 可运行的局部流水

下面是当前AST子集的完整示例；`tiles`为runtime，B为constexpr。Tensor的
`(tiles,B)`表达需要多少存储，launcher检查范围/溢出/别名，分配真实性由调用合同保证。

```python
from __future__ import annotations
import xdsl_npu.language as T
from xdsl_npu import compile

@T.prim_func(disjoint_external=True)
def scale(X: T.Tensor((tiles, B), T.f32),
          Y: T.Tensor((tiles, B), T.f32, access="write"),
          tiles: T.int64, B: T.constexpr = 64):
    with T.Kernel(kind="vector"):
        x = T.alloc_ub((2, B), T.f32)
        y = T.alloc_ub((2, B), T.f32)
        for t in T.Pipelined(tiles, stages=3):
            with T.Stage(0, name="load"):
                T.copy(X[t, :], x[t % 2, :])
            with T.Stage(1, name="compute"), T.Vector():
                T.mul(x[t % 2, :], 2.0, out=y[t % 2, :])
            with T.Stage(2, name="store"):
                T.copy(y[t % 2, :], Y[t, :])

compiled = compile(scale)
```

同一binary用于tiles=0/1/百万，不能通过host循环生成百万份操作。一般元素尾部
需要显式valid；已有对齐prefix见[ast_prefix](../examples/ast_prefix.py)，不授权
越界读取或扩大外部allocation。实际prelude→pipeline→epilogue见
[ast_state_sum](../examples/ast_state_sum.py)：加载Initial、逐代FP32累加、导出；
零次循环仍执行加载与导出。

## 3. 类型、存储和数值合同

前端区分constexpr、uniform runtime scalar、actor/代次相关scalar、storage ref
和借用tile result。运行时标量计算使用SSA；动态shape/地址不是不可分析字符串。
无数据相关side effect的表达式可重算，但只能使用该逻辑代次捕获的输入。
异步Scalar metadata有自己的读写及捕获寿命，不能读控制器当前下一任务的值。

- GM extent/stride允许受支持的runtime表达式；片上capacity和layout为constexpr。
  有符号范围、乘法溢出、除零、alignment和有效域分别检查。
- `x[t%2]`是明确的物理复用。`state[q%2]`、`Score[t%2]`、`L1[k%3]`使用不同
  时钟，编译器不因同为2槽便视作同一generation。目标代次身份包含region路径、
  parent调用坐标与local iv，不能仅用变量名字匹配。
- slice、转置和recast保留allocation provenance。logical valid与native实际
  读取padding/fractal分开；缺初始化或容量则拒绝。
- `v=T.copy(...)`借用该次写的版本，覆盖后再读取旧v拒绝；需要快照应明确copy。
  `out=`原地操作记R/W。当前跨region借用值仍拒绝，mutable ref组合已部分支持。
- 外部`access=read`声明必须与effects一致。`disjoint_external=True`通过直接
  指针区间检查约束别名，不解决间接索引唯一性；只读跨owner重读可以合法。
- UB由各AIV私有，不能跨actor传local handle。GM scratch显式传入并分group/lane；
  自动workspace分配不作为首轮语法可用的前提，也不隐式建立全尺寸中间矩阵。

精确Cube入口保留`T.matmul(a_l0, b_l0, accum=c_l0c, initialize=first)`。
算法层的GEMM/softmax由examples中的宏组合原语。宏接受实际refs、scratch和
constexpr政策；只内联静态源码结构，普通循环保留，内部名字卫生化，来源链和
宏传递内容参与诊断/cache key。禁止递归宏、任意host调用、裸同步和opaque ASC。
宏系统尚未实现，现阶段完整低层源码仍是可审查的依据。

FP32 carry、低精度snapshot、P cast和输出cast拥有不同版本与寿命。reduce树、
exp实现、round/saturate、FMA或分开的Mul/Add，以及div或reciprocal+mul必须固定。
宏展开及同步变换不能凭数学恒等式重结合数值运算。

## 4. CV多阶段：同一任务、不同actor、独立进度

以下为**目标语法骨架**，省略的宏均需在examples展开为实际原语，不可直接作为
可运行FA发布。完整待接通源码见[FA目标](attention-target.md)。

```python
for kv in T.Pipelined(
    kv_tiles, stages=3,
    issue_order={"cube": ("qk", "pv"),
                 "vector": ("update", "softmax")},
):
    with T.Stage(0, name="qk"), T.Cube():
        qk_export(kv, Q, K, Score, qk_scratch)
    with T.Stage(1, name="softmax"), T.Vector() as lane:
        softmax_rows(kv, lane, Score, P, m, l, alpha, vec_scratch)
    with T.Stage(1, name="pv"), T.Cube():
        pv_export(kv, P, V, PV, pv_scratch)
    with T.Stage(2, name="update"), T.Vector() as lane:
        update_rows(kv, lane, PV, alpha, l, o, O, out_scratch)
```

源码kv始终指逻辑任务。actor本地step为j，Stage s绑定`kv=j-s`；所有地址、
guard、scalar、first/last和子循环调用坐标一起移动。两处Stage1是不同区域，
没有共同完成点；两个AIV也不是一个UB空间。

| 本地step（N=3） | AIC派发 | 每AIV派发 |
|---|---|---|
| 0 | QK[0] | 无 |
| 1 | QK[1]，PV[0] | Softmax[0] |
| 2 | QK[2]，PV[1] | Update[0]，Softmax[1] |
| 3 | PV[2] | Update[1]，Softmax[2] |
| 4 | 无 | Update[2] |

这是派发表，AIC/AIV的step j并不同步到达。实际依赖包括QK全部必要FIX写→
Score对应区域可读、两AIV的P写→PV可读、PV写→Update；每个GM槽的释放则来自
实际最后MTE2 reader。完整tile需要多个贡献时不能第一份贡献就ready，但也不
应把独立范围的consumer都扩大到无关范围的完成点。

cross-core Wait是后续派发的gate。所选policy要求新QK先于旧P gate，旧Update
先于新Score gate；若固定顺序死锁或bank不够，诊断，不静默改序。此policy是
待验证的专家选择，不是ops-transformer所有模板的统一顺序；当前arch22
`Process`可见Vec1在Vec2前，其Matmul内部派发还需结合代码与profile核对。

## 5. Stage内部的子流水

通信块BQ128×BK1024、Cube128子块、Vector8行块是不同层级。一个CV Stage可以
包含普通工作、serial和自己的Pipelined；child Stage只改变child iv。下面为
目标片段，父kv从其stage捕获，chunk是另一个代次：

```python
with T.Stage(0, name="qk"), T.Cube():
    if kv == 0:
        T.copy(Q[query, :, :], q1)
        T.copy(q1, qa)
    for chunk in T.Pipelined(
        BK // CK, stages=4,
        issue_order={"cube": ("load_k", "native_qk", "operand_k", "export")},
    ):
        with T.Stage(0, name="load_k"):
            T.copy(K[kv, chunk, :, :], k1[chunk % 2, :, :])
        with T.Stage(1, name="operand_k"):
            T.copy(k1[chunk % 2, :, :], kb, transpose=True)
        with T.Stage(2, name="native_qk"):
            T.matmul(qa, kb, accum=c0[chunk % 2, :, :], initialize=True)
        with T.Stage(3, name="export"):
            T.copy(c0[chunk % 2, :, :], Score[slot, :, chunk, :])
```

此处地址为分块示意，真实BNSD例必须使用保持原layout的views，不能靠隐式输入
packing成立。child退出不drain，不自动把父PV插入child循环。若需要这样的
穿插，要明确改变执行政策；不得为追求重叠暗中改变固定源码组织。

这里的L0B只有一槽，所以每step必须先派发旧块MAD，再装下一块operand；
否则Stage1的新写先于Stage2的旧读，同步会形成等待环。原目标fixture遗漏了
这项内层policy，现已显式修正。增加第二份L0B会超过该配置的分配预算，不能
用扩buffer隐瞒问题。PV的单槽L0A/L0B有相同约束。

边界必须保留真实首读/末写/未退休reader和未消费event。`init; child; use`在
child=0时让use读取init；多reader的cache覆写等全部reader而非最后派发者。
父子同bank不能因为作用域不同就都分配ID0。一般嵌套CV尚未实现。

## 6. 规则跨query连续流

目标保留一个设备task循环，显式区别于每query一个kv循环：

```python
if K > 0:
    for task in T.Pipelined(queries * K, stages=3,
                           issue_order={"cube": ("qk", "pv"),
                                        "vector": ("update", "softmax")}):
        query = task // K
        kv = task % K
        first = kv == 0
        last = kv == K - 1
        # 同上四个Stage，均使用本任务捕获的query/kv/first/last。
```

K在流入口捕获且uniform，保留`task=query*K+kv, 0<=kv<K`关系；编译器不枚举
K个余数，也不按queries×K展开。K=0的输出语义由算法明确给出，除法不执行。
每actor必须对空任务达成一致的通知激活，不能一侧skip而另一侧Wait。

task槽、query状态和chunk槽独立。在所选update-before-softmax政策下，旧query
最后Update先于新query重置m/l；单份状态能否复用仍需实际读退休证明。换政策
后不能沿用这一假设。Q cache覆写只等其QK最后reader，不因旧PV尚未结束就等
整个query；反之也不能仅凭QK调用已返回就覆盖。

本子集只接纳uniform K和受限规则解码；动态分段、页表、稀疏索引、persistent
工作队列需要另外的descriptor/owner合同。商余语法能够解析不等于已经支持。

## 7. FA数学与变体

真实例必须实现以下在线更新，不能用乘法/cast代替softmax：

```text
m' = max(m, row_max(scores))
alpha = exp(m - m')
p32 = exp(scores - m')
l' = f32(alpha*l) + row_sum(p32)
o' = f32(alpha*o) + RN_operand(p32) @ V
output = 指定的归一化recipe(o_final,l_final)，再显式输出cast
```

初始m=-inf、l/o=0；p32归约在P cast之前。全mask可能产生-inf-(-inf)，必须有
明确identity/输出合同；不能只mask最终store。当前目标fixture先限定正长度无mask。

BQ128/BK1024/D128、VR8检验通信与native/UB独立粒度；Q cache、m/l/O、alpha、
Score/P/PV和L1/L0环分别检查寿命。64×1024 f32不能作为一个AIV的整片UB工作区。
CV workspace随group和固定slots有界，不允许随完整Sq×Sk分配Score。

变体分批：dense MHA/GQA/MQA、f16/bf16、prefill/decode；再加causal、非方形绝对
位置、tails、空序列/全mask；随后varlen/paged/sparse/动态persistent与split-KV。
训练backward/dropout/RNG另立合同。每个配置有明确支持矩阵，不静默调用CANN。

## 8. 同步反例与诊断

局部三阶段流水的物理IR投影及插入后的必要关系：

```text
前：MTE2 X[t]→x[t%2] ; V x[t%2]→y[t%2] ; MTE3 y[t%2]→Y[t]
后：MTE2.write_done[t] → V.read[t]              # ready
    V.read_retired[t] → MTE2.write[t+2]         # input release/WAR
    V.write_done[t] → MTE3.read[t]              # ready
    MTE3.read_retired[t] → V.write[t+2]         # output release/WAR
```

这不是完整事件程序，ID复用、guard、零域和最终收尾还要验证。漏input release
会覆盖在读x；把它推到MTE3结束会损失预取。实际含零域的前后IR与同步序列见
[region-sequence](region-sequence.md)，单流见[synchronization](synchronization.md)。

诊断分三类：已证非法（冲突路径/容量/初始化反例）；不支持（具体view/control/
ownership缺口）；分析未决（phase、结构计数、预算和最小定位）。均包含source
位置、双方任务/代次及真实root/字节范围。失败不改变输入IR或留下可执行缓存。

## 9. 实际支持矩阵

| 能力 | 当前状态 |
|---|---|
| AST单流水、对齐prefix、归约/Scalar、显式L0/runtime K | 声明子集已贯通，有设备证据 |
| serial父×单local child | 严格frame子集已贯通 |
| 根部普通工作/多个local stream顺序组合 | 已贯通；单共同bound、stationary共享存储和实际只读GM |
| serial父体一般前后工作/多child、跨region借用值 | 未接通/拒绝 |
| Pipelined内嵌Pipelined、mixed层次组合 | 未接通/拒绝 |
| uniform K商余的连续query流 | 待实现 |
| AST宏、新AST真实FA | 待实现；fixture和legacy设备记录不计入 |
| 一般分支、间接动态ownership、paged/sparse | 无完整合同，拒绝 |

支持标记必须绑定实际编译入口、最终ASC及输入合同，不能把source verifier通过
作为整个编译路径或任意规模同步证明。
