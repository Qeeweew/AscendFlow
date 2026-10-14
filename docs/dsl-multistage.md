# DSL 规格：逻辑程序、执行者与多级流水

2026-10-14，基线 `b98d541`。本文区分现有语法与待实现语义；目标片段不是当前
可运行 API 承诺。[总体设计](design.md)规定取舍，[重构方案](refactor-plan.md)
规定何时进入支持范围。使用 `import xdsl_npu.language as T`，无需修改包名。

## 1. 用户应能从源码读出什么

一份源码应能回答：每个逻辑任务计算什么、由哪个执行者处理、使用什么存储、
哪些代次可以重叠、固定派发次序是什么。用户无需编写 event ID、初始 credit、
反向 ack、`t-1/t-2` 填排状态机或手动最终收尾。

沿用 TileLang 风格的 scope，不追求复刻其全部 API：

| 写法 | 固定语义 |
|---|---|
| `@T.prim_func` | 捕获并解析受限 Python AST；不执行设备函数体 |
| `T.Tensor(shape, dtype, access=...)` | 外部引用、动态形状与读写合同；布局明确 |
| `T.Kernel(groups, kind="mixed") as group` | launch/group 域；mixed 为1 AIC+2 AIV |
| `T.Cube()` / `T.Vector() as lane` | AIC / AIV 执行域；不是只包含 MAD / V 指令 |
| `T.serial(...)` | 设备控制顺序，不承诺各次循环的异步工作全部完成 |
| `T.Pipelined(N, stages=S, issue_order=...)` | 保留 N 个逻辑任务的结构化循环及固定排步 |
| `T.Stage(s, name="...")` | 最近 pipeline 的代次位移；标签供排步/诊断引用 |
| `T.unroll(...)` | 显式、受预算限制的 constexpr 展开 |
| `alloc_ub/l1/l0a/l0b/l0c`、slice/view | 固定容量、明确布局及别名；槽位轴由用户选择 |
| `copy/matmul/cast/reduce_*/...` | typed 原语；scratch、数值与有效范围可见 |

保留现有 `stages` 拼写，不为模仿 TileLang 增加第二个同义参数。S只界定
`0<=s<S`，不决定缓冲份数。首版要求显式 Stage，默认每 actor 源码派发顺序；
可选 `issue_order` 必须完整列出该 actor 的直接 stage 标签。没有默认自动搜索。

## 2. 最小可运行的局部流水

下面使用现有 AST 子集，`tiles` 是 runtime int64；实际调用者提供 `tiles*B`
个元素。例子只演示一个 AIV，吞吐例另有多核划分。

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

`tiles=0/1/百万` 使用同一份代码。原始元素长度写为 runtime n、
`ceildiv(n,B)` 与显式 valid；当前实现仅支持已有对齐 prefix 子集，见
[ast_prefix](../examples/ast_prefix.py)。不自动扩大外部 allocation 或越界补齐。

## 3. Python、值和存储的边界

前端区分 constexpr、uniform runtime scalar、随 actor/任务代次变化的 scalar、
storage ref 和借用 tile result。形状可引用后声明的标量。注解不 eval 任意 Python，
闭包只捕获受限不可变常量；不支持的调用给出源码位置和诊断，无 tracing fallback。

`for` 保留为 region；device `if` 保留为 guard；constexpr `if` 可以裁剪。
标量变量合流/循环携带使用 SSA yield，初期物理 lowering 只接通明确列出的子集。
`Pipelined` 中任意标量递推不能因为 Python 语法能写就被推断为合法。
数据相关 while、异常、生成器和隐式 host side effect 先拒绝。

分配与 alias 规则：

- 片上 capacity/layout 是 constexpr；GM extent/stride 可以包含受支持的 runtime
  表达式。uniform 参数乘法要先检溢出；不是任意非线性地址证明器。
- 物理 slot 是显式轴，例如 `x[t%2]`。query state、通信槽、L1/L0 子环可以分别
  用 query、task、chunk 选择；必要时明确二维 slot，不能自动猜版本所属循环。
- view 不复制数据。转置 view、跨 dtype alias、带孔洞/strided 区域保留共同 root
  和精确 footprint。native 读取 padding/fractal 时必须已有真实初始化与容量。
- `v=T.copy(...)` 的 v 借用该写的版本；覆盖后再读旧 v 必须拒绝，不隐藏 snapshot。
  `out=` 原地更新是显式 R/W，不因更换变量名变为独立存储。
- `disjoint_external=True` 的直接区间由 launcher 每次检查。分配大小仍是调用者
  合同；间接索引不能靠区间不重叠声明解决唯一性。只读共享也要检查真实 effects。
- 每 AIV 的 UB 独立，Cube 的 local handle 不可给 Vector 直接读。GM scratch
  是显式参数并按 group/lane 分区，不隐式增加跨核搬运或全尺寸中间矩阵。

精确指令入口保留 `T.matmul(a_l0, b_l0, accum=c_l0c, initialize=first)`。
常用组合使用 examples 中拟议的 `@T.macro` 受限 AST 宏；参数是上述值类别，
调用者提供 scratch/layout，展开一次并保留普通循环。宏递归/任意 Python 调用
不支持；宏体和传递捕获进入编译键。宏尚未实现，当前使用显式原语。

FP32 carry、低精度 snapshot、P cast 和最终输出 cast 是不同版本/寿命。
reduce 树、exp/reciprocal 实现、FMA 与分开的 Mul/Add 都是数值合同。
编译器不能凭数学恒等式改变舍入；如果宏换算法，必须作为新的配置验证。

## 4. CV multi-stage 的准确含义

目标语法片段如下，宏是例子侧的工作缩写；每个宏必须接受实际输入、输出和
scratch，展开为普通 typed 操作。本片段尚未通过新路径全链路验证。

```python
for kv in T.Pipelined(
    kv_tiles, stages=3,
    issue_order={"cube": ("qk", "pv"),
                 "vector": ("update", "softmax")},
):
    with T.Stage(0, name="qk"), T.Cube():
        qk_export(kv, out=Score[kv % 2, :, :], scratch=qk_scratch)
    with T.Stage(1, name="softmax"), T.Vector() as lane:
        softmax_rows(kv, lane, Score, P, m, l, alpha, scratch=vec_scratch)
    with T.Stage(1, name="pv"), T.Cube():
        pv_export(kv, P, V, out=PV[kv % 2, :, :], scratch=pv_scratch)
    with T.Stage(2, name="update"), T.Vector() as lane:
        update_rows(kv, lane, PV, alpha, o, scratch=out_scratch)
```

源码中 kv 始终是逻辑任务；编译器在各 actor 的物理 step j 执行 Stage s 时绑定
`kv=j-s`，所有地址、guard、scalar capture、first/last 随这个绑定一起移动。
AIC 和两个 AIV 各自推进 j，没有共同墙钟或隐式 barrier。

| 本地 step | AIC 发射次序 | 每 AIV 发射次序 |
|---|---|---|
| 0 | QK[0] | 无 |
| 1 | QK[1]，PV[0] | Softmax[0] |
| 2 | QK[2]，PV[1] | Update[0]，Softmax[1] |
| 3（N=3） | PV[2] | Update[1]，Softmax[2] |
| 4（N=3） | 无 | Update[2] |

这是激活/派发表，不是完成时间线。两处 Stage1 是独立 region。真正的同步
来自访问与 ownership；cross-core Wait 会阻挡后续派发，因此等待位置是性能
合同的一部分：上述 policy 要求新 QK 先于旧 P gate，旧 Update 先于新 Score gate。

这是可选择的专家 policy，**不是 ops-transformer 所有模板的统一顺序**。
其当前 arch22 `Process` 源码包含 Vec1 先于 Vec2；异步 Matmul 内部还需结合最终
指令判断。若固定 policy 的依赖/资源不能成立，诊断具体环路或资源冲突；不能
悄悄换顺序。也不把两个 actor 同编号 stage 视为共同完成点。

## 5. 内层流水与父子边界

一个 CV 阶段可包含自己的 `serial/Pipelined` 和原语。例如 QK 用多个原生 N/K
块，Vector 用8行工作块处理完整通信块。内层循环必须保留为设备控制；stage
只作用于最近 pipeline 的 index，不把父子迭代数乘成 Python task 列表。

```python
# 目标片段：外层 stage 内的独立 DMA/MAD 子流水。
# a1/b1/a0/b0/c0 已由调用者分配；tile/布局及完整块条件已检查。
for k in T.Pipelined(panels, stages=3):
    with T.Stage(0, name="load"):
        av = T.copy(A[group, output, k, :, :], a1[k % 3, :, :])
        bv = T.copy(B[group, output, k, :, :], b1[k % 3, :, :])
    with T.Stage(1, name="operands"):
        aa = T.copy(av, a0[k % 2, :, :])
        bb = T.copy(bv, b0[k % 2, :, :], transpose=True)
    with T.Stage(2, name="compute"):
        T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))
# 后续 FIX 等待 c0 的真实末次写；下次 L1 覆盖只等相应 MTE1 reader。
```

当前可运行的有限父子形状另见 [ast_hierarchical_gemm](../examples/ast_hierarchical_gemm.py)。
上面片段在外层 Pipelined 中的组合、一般外部 FIX 还未接通。

父到达 child 时派发 child 的控制循环，再继续下一个父 region。子流水填排结束
只意味着延迟任务已派发，不意味着硬件 drain。父不能把 child 等价成一个完成点。
child 上界每次入口捕获，代次为 `(scope,parent_invocation,local_iv)`。

| 边界 | 必需行为 |
|---|---|
| `init; child; use` | init 是真实 writer；child=0 时 use 消费入口状态 |
| child只读同一cache多次 | 父覆写前覆盖所有未退休reader，不只最后派发那一个 |
| child末次写/导出 | 不初始化此前读取；first=last 的一次情形正确 |
| child后有异步工作 | reader/writer/event继续存活，不能重seed未消费ID |
| 两个child共享scratch | 对同一真实字节形成依赖；bank/ID按组合后的寿命分配 |

先接通规则串联和有限嵌套，不自动把父 sibling 工作穿插进 child。要这种穿插，
源码需明确不同的连续流或 stage 划分。

## 6. 规则跨 query 连续流属于本轮语言合同

逐 query 嵌套和连续 CV 流是两种明确政策，不能声称前者自动达到后者性能。
以下是待实现的规则解码语法；无需新 task 类：

```python
# 目标：K 在整个流入口捕获；K==0 的行为在此流之外处理。
if K > 0:
    for task in T.Pipelined(queries * K, stages=3,
                           issue_order={"cube": ("qk", "pv"),
                                        "vector": ("update", "softmax")}):
        query = task // K
        kv = task % K
        first = kv == 0
        last = kv == K - 1
        # 与上一节相同的四个 Stage；各自使用该逻辑task的query/kv/first/last。
        # first初始化该query状态，last发布输出；旧query尾部与新query头部可重叠。
```

编译器识别受限同一 divisor 的商余关系，保留 q/k/任务域，不枚举 K 个 residue。
参数乘法、除零、长度上界和地址范围在 launcher 验证；异步 Scalar metadata
需捕获并版本化，不能在执行延迟 stage 时读取当前 query 的描述符。

state 的 `query % QSlots` 与交接的 `task % Slots` 不是同一时钟；Q cache 下次
加载前等所有 QK reader，输出 state 下次初始化前等最后 update/export。
不能每逢 first 就重置仍在途的共享事件。K=1、K小于流水深度、K与slots互素、
query=0/1/多次是必要反例；零 K 不凭空产生可读取的最后结果。

本模式只支持规则 uniform K。变长段、页表、稀疏任务、持久工作队列不是该
语法自动获得的能力，另需 metadata/owner/epoch 合同。

## 7. 真实 FA 对语法的约束

完整例子必须在 examples 展开下面的数学及存储组织，不能以乘法/cast 代替 softmax：

```text
m' = max(m, row_max(scores))
alpha = exp(m - m')
p32 = exp(scores - m')
l' = f32(alpha*l) + row_sum(p32)
o' = f32(alpha*o) + RN_operand(p32) @ V
output = 选定数值recipe(o_final, l_final)，再显式RN到输出dtype
```

初始 m=-inf、l/o=0；p32 的归约先于 P cast。最终 recipe 明确 div 或
reciprocal+mul；宏不能任意替换。每个逻辑 BK 的归约、累计和舍入边界固定。

| 生命周期 | 必须能单独表达 |
|---|---|
| Q cache | query初始化至最后一个QK读取，与KV scratch不同 |
| m/l/O | query的FP32递推状态；适当物理复用，不能跟着chunk清零 |
| alpha | softmax产生、延迟update消费；slot及代次独立检查 |
| Score/P/PV | 三个GM交接域；真实完整贡献ready、全部MTE2读取退休后可复用 |
| L1/L0/UB工作块 | 子流水独立slots，local容量和event寿命逐一检查 |

BQ128、逻辑BK1024、D128为表达能力目标之一，Cube使用较小native块，Vector
每次处理8行。每AIV可拥有64行状态、以8次行迭代处理通信块；不能把64×1024
f32整片放进192 KiB UB。双槽 Score/P/PV 的GM空间分别约1 MiB/512 KiB/128 KiB
每组，仍需计算各local cache/temporary的实际峰值。

Score完整ready必须覆盖所有FIX贡献，P完整ready覆盖两AIV实际行片；GM credit
取实际MTE2读取退休，不等整个softmax/update结束。若已有其他完成路径覆盖了
某release，可以凭具体证明删除；不能把“FA通常往返”写为硬编码规则。

基础变体为 dense MHA/GQA/MQA、f16/bf16、prefill/decode、causal和tails。
头映射、non-square causal对齐、绝对位置、空序列/全mask的输出必须逐项定义。
初版可以明确拒绝未定义情形；不能靠mask最终store掩盖上游NaN或通知不配对。
varlen/paged/sparse/dynamic persistent分阶段加入；backward/dropout/RNG另立范围。

## 8. 同步示例与拒绝诊断

双槽Vector程序的同步前投影：

```text
stage0 MTE2: X[t] -> x[t%2]
stage1 V:    x[t%2] -> y[t%2]
stage2 MTE3: y[t%2] -> Y[t]
```

同步后所需关系（尚未分配实际ID，不冒充完整生成ASC）：

```text
MTE2.write_done[t] -> V.read[t]            # x ready
V.read_retired[t]  -> MTE2.write[t+2]       # x WAR/release
V.write_done[t]    -> MTE3.read[t]          # y ready
MTE3.read_retired[t] -> V.write[t+2]        # y WAR/release
```

只对真实未来consumer发通知，或采用完整验证的seed/drain协议；二者不能混搭。
去掉第二条会覆盖仍在读的x；把第二条延后到MTE3结束会丢失本可重叠的预取。
事件自身还要证明旧Wait先于同ID新Set生效。实际前后IR及ASC见
[同步规则](synchronization.md)和[运行时prefix](runtime-prefix.md)。

新诊断应带源码位置、root/字节范围、双方任务/代次和原因，例如：
“child为0时state未初始化”“query1覆盖仍被query0读取的Q槽”
“P只发布lane0贡献”“同bank旧Wait尚未消费”“商余关系不在当前可证明子集”。
超预算是分析未决，缺少协议是支持缺口；它们不等价于程序必然不安全。

## 9. 实际支持矩阵

| 能力 | AST/IR | 物理/同步 | 新路径设备证据 |
|---|---|---|---|
| 单流水Vector/对齐prefix/归约Scalar | 有 | 声明子集有 | 有 |
| 显式L0、runtime K、末代FIX | 有 | 声明子集有 | 有 |
| serial父×单local child | 有 | 严格GM frame子集有 | 有 |
| 一般prelude/epilogue/两个child | 部分结构可表达 | 无 | 无 |
| Pipelined内嵌Pipelined、mixed层次组合 | 可保留部分结构 | 拒绝 | 无 |
| runtime商余的规则连续流 | 待扩展 | 无 | 无 |
| examples受限AST宏、真实FA | 待实现/迁移 | 无完整新路径 | legacy证据不计入 |
| varlen/paged/sparse/动态ownership | 无完整合同 | 拒绝 | 无 |

实现达到哪一层就只登记哪一层。合法源码不等于已支持lowering；硬件证据绑定
最终ASC/hash、配置及输入合同，不能从有限样本推导任意程序正确性。
