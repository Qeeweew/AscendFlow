# DSL 前端、结构化循环与 CV 多阶段语义

2026-10-14 修订。本文是待实现规格，代码块展示目标语法，不能作为当前可运行
API 使用。现有 tracing/单循环原型的能力和设备记录见[重构记录](refactor-progress.md)。
架构约束见[总体设计](design.md)，实施顺序见[重构计划](refactor-plan.md)。

## 1. 采用 TileLang 风格作用域，明确 Python 与设备的边界

```python
from __future__ import annotations
import xdsl_npu.language as N

@N.prim_func(disjoint_external=True)
def scale(
    X: N.Tensor((n,), N.f32),
    Y: N.Tensor((n,), N.f32, access="write"),
    n: N.int64,
    B: N.constexpr = 256,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((2, B), N.f32)
        y = N.alloc_ub((2, B), N.f32)
        for t in N.Pipelined(N.ceildiv(n, B), stages=3):
            slot = t % 2
            valid = N.min(B, n - t * B)
            with N.Stage(0, name="load"):
                N.copy(X[t * B:(t + 1) * B], x[slot, :], valid=valid)
            with N.Stage(1, name="compute"), N.Vector():
                N.mul(x[slot, :], 2.0, out=y[slot, :], valid=valid)
            with N.Stage(2, name="store"):
                N.copy(y[slot, :], Y[t * B:(t + 1) * B], valid=valid)
```

该例用一个 AIV 演示语义，大规模性能例另按核划分工作。`n` 是运行时元素长度，
`B` 是显式特化 tile。`n=0` 时没有读写或虚构的 stage task。初始实现可要求
`n % 8 == 0`、`B % 8 == 0`，使 f32 DMA 有效长度满足32字节契约；未实现精确
非对齐尾搬运时诊断拒绝。不能为了让例子运行而静默越界读取或填充用户分配。

`valid` 限制实际访问与 Vector mask，名义 tile 可跨逻辑尾部，实际访问不能
跨越有效范围。若某操作需要 padding，用户显式指定值，lowering 展示相应
写入/effects；不存在默认隐藏补零。tail 和 align 条件进入输入/运行时启动检查。

### 前端执行规则

- 装饰器取得 AST、源码位置和受限环境；函数体不通过 Python 迭代器执行。
- 先解析 scalar/constexpr 参数，再解析张量形状，支持形状引用后声明的 `n`。
  future annotations 中的字符串只经过 DSL 解析器，不能执行任意注解表达式。
- `for`、设备 `if/else`、局部标量赋值、切片、具名 DSL 调用和注册宏构成首版
  子集。动态 `while`、异常、生成器、任意 Python 对象调用先明确拒绝。
- constexpr 条件可选择静态分支；runtime 条件形成 `scf.if`，不能被 Python
  `bool()` 求值。标量合流/循环携带使用 SSA yield，buffer 是可变存储。
- host 工厂在函数外选择 tile/layout/stages。闭包只捕获有限不可变常量；宏
  参数与展开 provenance 进入 IR 和缓存键。宏不得隐式创建同步或更改数值边界。

`disjoint_external=True` 是全部独立外部 tensor 分配的字节范围不重叠契约，
host 包装器必须能依据指针/长度检查；同一 root 的显式 views 仍允许别名。
不能检查的间接地址需要更具体契约或拒绝。运行时长度还需非负、地址与索引
算术不溢出等约束；不得把注解当作未经验证的安全假设。

## 2. 最小语法与存储语义

| 写法 | 语义 |
|---|---|
| `N.serial(start, stop, step)` | 设备顺序循环，常量边界也保留；首版 step 为正编译期常量 |
| `N.Pipelined(stop, stages=S, ...)` | 设备逻辑任务循环；省略 start/step 为0/1；固定 stage 排步 |
| `N.unroll(...)` | 用户明确请求 constexpr 展开，超过模板预算报错 |
| `N.Stage(k, name=...)` | 所属 pipeline 中逻辑任务的派发位移，不是完成边 |
| `N.Cube()` / `N.Vector()` | AIC / AIV 执行域；混合核中 Vector 可返回 lane 0/1 |
| `N.alloc_ub/l1/l0a/l0b/l0c(...)` | 固定形状、dtype、layout、执行域的物理存储 |
| `N.copy / matmul / cast / reduce_* / ...` | typed 原语，明确输入/输出、布局和舍入参数 |
| `buf[iv % slots, ...]` | 显式物理 slot，代次由真实写入与所属逻辑任务决定 |

外部 tensor 形状可动态，片上容量与 tile shape 首版固定。内置布局用实际字节
映射定义；转换只能由明确操作完成，不能因为 view 改名而多出搬运。
切片、转置视图、跨 dtype alias 都要保留 allocation identity 与精确 footprint。
重复使用一个 Python 名字是绑定操作，不产生 buffer 副本。

混合核局部 allocation 在对应 `Cube/Vector` 域内声明，handle 可以被同 actor
后续 region 引用；每个 AIV 的 UB 是各自私有存储，不因同名产生跨核别名。
GM scratch 作为显式参数提供并按 group/row 分区，local handle 跨 actor 使用
被拒绝。`with` 里的 lane/group 符号是受域约束的设备坐标，不是 host 整数。

`N.matmul(a=a_l0, b=b_l0, accum=c_l0c, initialize=first)` 显式表达操作数与累加
存储，shape/layout 决定合法 native recipe。`reduce_sum` 的归约树、`exp` 的
近似实现及 NaN/Inf 行为属于版本化原语契约；省略策略不允许后端任意换算法。
首次 FA 基线固定同一归约 recipe。Mul/Add 不自动缩并为 FMA，显式 cast 的舍入
边界不能被融合越过；若需要不同数值策略，使用显式 policy 并纳入特化键。

逻辑 tile value 不能保证原存储内容永远存在：若借用 view 对应的版本已覆盖，
后续读取被拒绝。需要保留旧值就显式分配存储并复制。跨 stage 的 scalar 若
来自内存，需在正确任务发射时捕获并携带；不能按物理 step 再读。编译器若需要
保存 descriptor，应在计划中显示其存储与生命周期，不能隐藏数据搬运成本。

当前 `versions=2` 糖衣默认关联最近 pipeline 的规则不保留为核心语义。未来
保留兼容糖衣也必须无歧义地降到显式 slot，歧义时拒绝，不能猜测 query/KV/chunk。

### 独立嵌套循环

```python
for kv in N.Pipelined(N.ceildiv(Sk, BK), stages=3):
    with N.Stage(0, name="qk"), N.Cube():
        for chunk in N.Pipelined(N.ceildiv(D, DK), stages=2):
            with N.Stage(0, name="load_k"):
                N.copy(Q[:, chunk * DK:(chunk + 1) * DK],
                       q_l1[chunk % 2, :, :])
                N.copy(K[kv * BK:(kv + 1) * BK,
                         chunk * DK:(chunk + 1) * DK],
                       k_l1[chunk % 2, :, :])
            with N.Stage(1, name="mad"):
                # 示例宏展开为显式 L1→L0 copy 和 matmul；scratch 由调用者提供。
                qk_chunk(q_l1[chunk % 2, :, :], k_l1[chunk % 2, :, :],
                         a_l0, b_l0, score_l0c,
                         chunk=chunk, initialize=(chunk == 0))
```

这是完整块索引片段，要求 D 可被 DK 整除；宏和固定大小的 Q/K 双槽 storage
定义在示例侧，不分配运行时大小的片上 Q cache。内层循环既不在 Python 展开，
也不把外层 CV step 乘成统一微步时钟。运行时 `initialize` 可形成 native 条件
或结构化分支，必须保持首次 clear 和其余顺序累加；不能修改 FP32 累计顺序。

这个语法不保证任意 D 都有低成本可行排步。动态尾部、容量或分析域不受支持时
给具体诊断。独立环可写为 `state[query % QSlots]`、`score[kv % KVSlots]`、
`k_l1[chunk % L1Slots]`；需要外内两层同时存活时显式用两个槽位轴。

## 3. CV multi-stage 的精确定义

同一 pipeline 中所有 stage 源码引用同一个逻辑 `kv`。stage k 在物理 step j
处理 `kv=j-k`，只有该逻辑任务在域内才激活。地址、slot、guard、first/last
及 descriptor 一起位移。编译器产生 prologue/steady state/epilogue；用户不写
`t-1/t-2` 和 `tasks+2`。零次循环不产生伪通知，边界只实例化实际存在的工作。

词法顺序定义同任务的数值/effect 顺序；loop-carried 依赖维持算法递推顺序。
物理 issue 可通过每 actor 的 stage 标签排列显式约束：

```python
for kv in N.Pipelined(
    N.ceildiv(Sk, BK), stages=3,
    issue_order={"cube": ("qk", "pv"),
                 "vector": ("update_o", "softmax")},
):
    with N.Stage(0, name="qk"), N.Cube():
        qk_export(kv)
    with N.Stage(1, name="softmax"), N.Vector() as lane:
        softmax_and_publish(kv, lane)
    with N.Stage(1, name="pv"), N.Cube():
        pv_export(kv)
    with N.Stage(2, name="update_o"), N.Vector() as lane:
        update_output(kv, lane)
```

此片段只说明排步，真实 softmax 展开见下节。默认同 actor 使用源码顺序；标签
按嵌套 region 路径限定。首版显式 order 要列全该 actor 的直接 stage regions，
缺失/重复/跨作用域标签拒绝。内层 issue 顺序由内层作用域指定。

| 同一物理 step j | AIC 发射顺序 | 每个 AIV 发射顺序 |
|---|---|---|
| 稳定段 | QK[j] → acquire P[j−1] → PV[j−1] | acquire PV[j−2] → UpdateO[j−2] → acquire Score[j−1] → Softmax[j−1] |

箭头是控制器派发顺序，完成边仍由访问分析生成。AIC 在可能阻塞的旧 P Wait
之前发射新 QK；AIV 先处理旧输出，避免新 Score gate 阻挡它。两处 Stage1 是
独立 region，不能合成共同完成点。这是可表达的专家策略，不承诺所有 tile/
slot/guard 都可行：planner 必须结合跨核 gate、依赖和资源检查死锁/复用。

软件 stage 数、buffer slots、query state 数、native segment 数、event 环宽度
分别定义。完整 Score ready 需覆盖该任务所有实际 FIX 贡献；相应 credit 只
等待真正读取该 GM 槽的两 AIV MTE2，不能拖到 softmax 或最终 O 完成。

## 4. 用真实 Flash Attention 检验语言

所有宏、算法和变体放 examples；编译器没有 `flash_attention` 内建分支。
首个迁移目标是固定 query tile、运行时 KV 长度的 dense attention，保留三个
GM 交接环和明确的 FP32 状态。GQA/MQA、causal/tails 另有语义与验证门槛。

### 存储与数值契约

| 存储 | 所属/生命周期 |
|---|---|
| `q_l1` | AIC 的 query cache，循环前加载，全部 QK 最后读取后可复用 |
| `Score[2,BQ,BK]`、`P[2,BQ,BK]`、`PV[2,BQ,D]` | group-private GM scratch，KV 槽复用；两个 AIV 操作不相交行片 |
| `m[2,R]`、`l[R]`、`o[R,D]` | 每 AIV 的 query state，R 为该 lane 行数；不能绑定到 native chunk |
| `alpha[2,R]` | Stage1 产生，延迟的 Stage2 消费，按 KV 槽保存 |
| score/probability/row reduction/temporary UB | 由明确布局和容量约束分配；只在最后读取后复用 |
| L1/L0A/L0B/L0C scratch | native 子流水自己的 slot 和 read-retired，不共享一个整体 release |

以下为最小完整数学主体的目标语法片段；scratch 和 GM 参数按上表由示例分配，
`qk_export/pv_export` 是接受显式 scratch 的示例宏，此处缩写实参。初始值为
`m[0,:]=-inf`、`l=0`、`o=0`。首次版本要求 `Sk>0`、完整 tile、每行存在有效
key，BQ 可均分给两 AIV，所有显式 scratch 满足容量/对齐；不把全 mask/空
attention 的行为交给偶然的 `-inf - -inf`。BQ/BK/D 的具体选择固定于示例策略。

```python
for kv in N.Pipelined(
    N.ceildiv(Sk, BK), stages=3,
    issue_order={"cube": ("qk", "pv"),
                 "vector": ("update_o", "softmax")},
):
    s = kv % 2
    old = kv % 2
    new = (kv + 1) % 2
    with N.Stage(0, name="qk"), N.Cube():
        qk_export(kv, out=Score[s, :, :])
    with N.Stage(1, name="softmax"), N.Vector() as lane:
        rows = N.row_partition(BQ, lane, parts=2)
        N.copy(Score[s, rows, :], scores)
        N.mul(scores, scale, out=scores)
        N.reduce_max(scores, axis=1, out=row_max)
        N.maximum(m[old, :], row_max, out=m[new, :])
        N.sub(m[old, :], m[new, :], out=alpha[s, :])
        N.exp(alpha[s, :], out=alpha[s, :])
        N.sub(scores, m[new, :, None], out=scores)
        N.exp(scores, out=scores)
        N.reduce_sum(scores, axis=1, out=row_sum)
        N.mul(l, alpha[s, :], out=l)
        N.add(l, row_sum, out=l)
        N.cast(scores, N.f16, rounding="rn", out=p_local)
        N.copy(p_local, P[s, rows, :])
    with N.Stage(1, name="pv"), N.Cube():
        pv_export(kv, P[s, :, :], out=PV[s, :, :])
    with N.Stage(2, name="update_o"), N.Vector() as lane:
        rows = N.row_partition(BQ, lane, parts=2)
        N.copy(PV[s, rows, :], pv_local)
        N.mul(o, alpha[s, :, None], out=o)
        N.add(o, pv_local, out=o)

# 每 lane 最后一次 l 与 o 更新完成后，输出各自的 query 行片。
with N.Vector() as lane:
    N.div(o, l[:, None], out=normalized)
    N.cast(normalized, output_dtype, rounding="rn", out=output_local)
    N.copy(output_local, O[N.row_partition(BQ, lane, parts=2), :])
```

`m` 交替保存前后状态，不为消除覆盖问题增加统计量副本。每次 KV 的数学为：

```text
scores = scale * Q @ K^T
m_new = max(m_old, row_max(scores))
alpha = exp(m_old - m_new)
p32 = exp(scores - m_new)
l_new = alpha * l_old + row_sum(p32)
o_new = alpha * o_old + RN_operand(p32) @ V
O = RN_output(o_final / l_final)
```

denominator 使用未舍入的 p32。P 舍入边界属于逻辑通信 BK；把 BK 拆成多个
native Cube 块不得改成每块独立 softmax、改累计顺序或重新舍入中间 O。
bf16 变体显式修改 operand cast 和对应 matmul 类型，不由后端猜测。

`qk_export` 展开为 K 的 GM→L1、Q/K 的 L1→L0、显式初始化/累加 matmul、
FIX 到 Score；`pv_export` 展开 P/V 搬运、PV matmul 和 FIX。两者均接受调用者
声明的 scratch/layout/tile 参数，不能隐藏额外全局搬运或手工 event。
较大 BK 下 AIV 可按少量行分段，Cube 可按 native 矩阵块分段；这需要嵌套 region
和精确摘要，不能为了让代码片段放得下而强制所有高性能配置使用小 BK。

这个片段用于确定语义，不宣称其配置已达到高性能。第一份真实实现必须同时
给出可编译完整示例、独立 oracle、同契约 AscendC 基线和固定配置的生成 ASC。

## 5. 分支、尾部和动态任务的边界

- 分支的 effects 保留 guard。互斥分支不人为建立冲突；合流后的 read 必须在
  所有可达分支有有效版本。通知的生产、消费和参与者 guard 必须一致或可证明
  配对，不能让一个 AIV 跳过另一个仍等待的通知。
- 尾部谓词属于逻辑任务，随 stage 移动。causal 无效位置在 reduce/exp 前按
  数值契约处理；全 mask 行需单独定义，不能仅屏蔽最终 store。
- 零次 KV 在通用语言中合法；FA 首版可拒绝 `Sk=0`，后续必须选择并测试明确
  输出契约。子循环结果在零次情况下未定义时，使用点给出诊断。
- runtime sizes 和静态 owner 的规则循环先实现；数据依赖 task stream、页表/
  稀疏索引和变长任务随后增加 metadata 捕获与输入契约，不能混称“支持动态”。
- 将来 persistent query 流沿同一 region/ownership 表示扩展。语言不规定每个
  query 必须清空所有 pipe；实现未支持跨 query 连续排步时明确声明子集。

## 6. 一个最小同步反例及应有输出

双槽流水中 `load[t+2]` 覆盖 `x[t%2]`。同步前结构片段为：

```text
pipeline t in [0,N), stages=3:
  stage 0: MTE2 load X[t] -> x[t%2]
  stage 1: V    mul x[t%2] -> y[t%2]
  stage 2: MTE3 store y[t%2] -> Y[t]
```

所需义务及抽象同步后片段（表示完成关系，不是完整 ASC 或已分配的 event ID）：

```text
load.write_done[t]   -> ready_x[t] -> mul.dispatch[t]       # RAW
mul.read_retired[t]  -> free_x[t]  -> load.dispatch[t+2]    # WAR
mul.write_done[t]    -> ready_y[t] -> store.dispatch[t]     # RAW
store.read_retired[t]-> free_y[t] -> mul.dispatch[t+2]     # WAR
exit: 等待剩余输出写完成，并收齐实际未消费的 event/credit
```

跨 pipe 实例通常选择 MTE2→V ready、V→MTE2 release、V→MTE3 ready、
MTE3→V release；最终是否可复用/删去 channel 由真实完成路径和 ID 生命周期
决定。prologue 不等待不存在的 `free[t-2]`，末尾不留下无消费者的 Set。
若去掉 x 的 WAR，快 MTE2 可覆盖仍被 Vector 读取的槽；若等到 store 完成才
释放 x，则扩大等待范围，抹掉合法预取重叠。最后读取的意义就在这个区别。

每类关键问题都按此格式交付实际 before/after IR、ASC 序列和反例，包括：
partial view WAW、同槽多 reader、内层最后一次读取、提前发布部分 Score、
旧 P Wait 阻挡新 QK、错 generation、分支漏通知、零次循环和最终 drain。
示意关系通过并不代替最终 ASC、CANN 编译和设备证据。
