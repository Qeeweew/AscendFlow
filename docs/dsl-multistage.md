# DSL 规格：结构化控制与多级流水

2026-10-15，代码基线 `ed87249`。这是语言合同，不是全量支持声明。
可运行例、目标片段分别标注；总体取舍见[设计](design.md)，接通顺序见
[重构方案](refactor-plan.md)。

## 1. 采用的语法及理由

参考 TileLang 的 tensor、tile、循环和 stage 表达，使用受限 Python AST。
`prim_func` 捕获源码，不执行设备函数体。普通循环生成设备控制；宿主 Python
只配置 constexpr。用户能直接从源码读出数据流、物理存储和流水排步。

| 语法 | 合同 |
|---|---|
| `@T.prim_func` | 设备函数、源码位置；不退回 tracing |
| `T.Tensor(shape, dtype, access=...)` | 外部引用，shape 可使用 runtime 参数 |
| `n: T.int64` / `B: T.constexpr` | 设备值 / 特化常量，不按实参值猜测 |
| `T.Kernel(groups, kind=...) as group` | launch 域；mixed 每组1 AIC+2 AIV |
| `T.Cube()` / `T.Vector() as lane` | actor 作用域，lane 是组内 AIV 编号 |
| `T.serial(N)` | 保留的顺序控制，无隐式 drain |
| `T.Pipelined(N, stages=S, issue_order=...)` | N 个逻辑任务，显式偏移范围和 actor 派发政策 |
| `T.Stage(s, name=...)` | 最近 pipeline 的代次偏移，0≤s<S |
| `alloc_ub/l1/l0a/l0b/l0c` | 显式空间、容量、layout 和 slots |
| `copy/matmul/reduce/cast/...` | typed 原语，所有 native effects 可见 |
| `T.unroll(...)` | 显式静态展开，受结构预算约束 |
| `@T.macro`（待实现） | 受限 AST 内联，内部普通循环保留 |

保留当前 `stages` 参数，不增加同义拼写。它不等于 phase 数，不自动分配 S 份
存储，也不承诺并发性能。首轮全部 Stage 显式；后续自动政策仍生成同一模型。
不恢复 `cube.task/vector.task` 或要求用户手写 ready/release。

actor 决定核，原语决定 engine；Cube 作用域也能包含 DMA/Fixpipe。常见 vector
源代码可省略能唯一推断的 actor，mixed 有歧义则要求显式作用域。

## 2. 可运行局部例：双缓冲

以下属于现有 AST 子集，tiles 为 runtime，B 为 constexpr：

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

同一 binary 接收 tiles=0/1/百万。shape 参与地址范围和 launch 合同检查，不生成
tiles 份 Python 对象。一般尾部要明确 valid/padding；已有对齐 prefix 见
[ast_prefix](../examples/ast_prefix.py)，不能默认外部 allocation 有额外 padding。

真实初始化、循环、导出见 [state_sum](../examples/ast_state_sum.py)；两个 runtime
轴见 [grouped_state_sum](../examples/ast_grouped_state_sum.py)：父循环内初始化，
子流水累加，最后导出。child=0 仍执行初始化和导出；parent=0 不执行 body。
当前父体可写GM要求严格 frame，实际只读输入可跨调用共享，见
[共享输入合同](shared-invocation-inputs.md)；一般多参数/间接访问仍未支持。

## 3. Stage 的精确定义

源码 iv 是逻辑任务坐标。在一个 actor 的本地派发 step j，Stage s 使用
`iv=j-s`，仅在 `0<=j-s<N` 时激活。编译器位移其整个表达式环境：地址、view、
first/last、descriptor、子循环捕获值一起移动，不只移动数据指针。

同一 pipeline 的多个 Stage 可有相同 s。它们是独立区域，没有共同完成点；
不同 actor 的 j 不同步。prologue/steady/epilogue 从激活域生成，N=0 无任务也
不能留下无匹配 Wait/Set。最终 drain 发生在真正 kernel 生命周期边界。

默认各 actor 按源码直接 Stage 顺序派发；`issue_order` 完整指定该 actor 的
Stage 标签，必须唯一且不遗漏。标签只在当前 pipeline 解析。政策改变派发位置，
不改变原源码的数学版本和累计顺序。任何违背依赖、容量或产生等待环的政策
给出诊断，不静默重排或扩 buffer。

## 4. CV multi-stage 的源码表达

以下为**目标片段**，四个函数代表待实现的 examples 宏，不能作为已可运行 FA。
完整低层 fixture 见 [attention_target](attention-target.md)。

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

| 本地 step，N=3 | AIC 派发 | 每个 AIV 派发 |
|---|---|---|
| 0 | QK[0] | 无 |
| 1 | QK[1]，PV[0] | Softmax[0] |
| 2 | QK[2]，PV[1] | Update[0]，Softmax[1] |
| 3 | PV[2] | Update[1]，Softmax[2] |
| 4 | 无 | Update[2] |

这是派发表，不是墙钟时间线。真实顺序另由访问产生：QK 的 FIX 写完成才能读
Score；两个 AIV 的 P 分片都写完，PV 才能读取完整范围；PV 写完成后 Update
读取。Score/P/PV 的槽分别在对应所有 MTE2 reader 退休后释放。

两 AIV 的行分片是实际不重叠 views，UB 私有，不能把两个192KiB合成384KiB。
GM workspace 显式按 group/slot 分配，初始未定义，读必须来自真实 writer；
不隐式分配随 Sq×Sk 增长的完整 Score/P。

当前政策使新 QK 先于旧 P gate、旧 Update 先于新 Score gate。过早 cross-core
Wait 会阻挡整个 actor 后续派发。此顺序是候选政策，不冒称 ops-transformer
所有模板均如此；当前 arch22 Process 可见 Vec1 在 Vec2 前，实际库内行为还需
代码和 profile 对照。语法必须能表达两种政策，状态分配和合法性分别验证。

## 5. Stage 内部仍是普通多级程序

通信 BK1024、Cube CK128、Vector VR8 独立，不要求整个通信块进入 UB。目标片段：

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

此处地址是分块示意，实际 BNSD fixture 使用保持原 layout 的 views，不预打包输入。
父 kv 在 Stage 入口捕获，child chunk 有自己的 pipeline clock；固定 BK/CK 的
循环也保留在设备。单槽 kb 要先派发旧 MAD，再装下一块 operand，不能依 Stage
数字排序。当前 fixture 已修正这一点，不靠加 buffer 解决。

child 填排退出不等于全部异步完成，不额外 drain。共享 L0C 的两个 child 必须
组合 FIX→下一写的 WAR 和事件寿命。普通嵌套不自动把父 PV 插入 QK child；若
专家政策要求这种穿插，必须明确表达，不能隐含变换。

## 6. 连续 task、状态和输入合同

跨 query 的目标控制形态：

```python
if K > 0:
    for task in T.Pipelined(queries * K, stages=3,
                           issue_order={"cube": ("qk", "pv"),
                                        "vector": ("update", "softmax")}):
        query = task // K
        kv = task % K
        first = kv == 0
        last = kv == K - 1
        # 相同四个 Stage；使用该逻辑 task 的 query/kv/first/last。
```

此处纯 descriptor 表达式属于逻辑任务，在每个 Stage 的正确代次求值或捕获。
有副作用的 metadata 读取不能当普通纯表达式复制。首批 K 在流入口 uniform 且
不变；保留商余关系，不枚举 K 个余数。K=0 的输出由算法显式定义，不能省略后
宣称支持空序列。

`state[query%2]`、`Score[task%2]`、`L1[chunk%3]` 有不同 owner/generation；同为
两槽不意味着同一时钟。固定 update-before-softmax 下能否复用单份 m/l/O 仍由
真实 reader 证明；换政策不能沿用假设。Q cache 覆写等其最后 QK reader，不
扩大到整个 query 的 PV/Update 完成。

类型/别名规则：

- GM extent/stride 是受支持的 runtime SSA，片上 capacity/layout 为 constexpr。
  边界、乘法溢出、除零、alignment、padding 和有效域分别检查。
- view/recast 保留 root；不同 tensor 名不证明 disjoint。外部 disjoint 合同在
  launch 检查直接指针区间，间接索引唯一性需要另有可验证合同。
- `access=read` 必须与真实 effects 一致；initialized 只说明初值，不说明永不写。
  共享只读 GM 要覆盖真实 group/调用域，不能从单 frame 假设推导。
- 借用 tile 绑定写版本，覆盖后旧借用不可读；跨 region 借用在版本验证接通前
  拒绝。需要 snapshot 明确 copy，低精度 snapshot 不替代 FP32 carry。
- 任意 Python、递归宏、裸 ASC、手写 flags 和未知 native effects 不作为源语言逃生口。

## 7. 数值与 FA 变体

真实 FA 必须实现 online softmax，FP32 m/l/O，p32 先做 denominator，再显式
舍入 P；O 的 rescale/add 次序、PV 累计顺序、reduce 树、exp、最终 div 或
reciprocal+mul 均固定。逻辑 BK 改变舍入边界，不能当作纯同步优化。

每变体有独立数值和参与者合同：f16/bf16、MHA/GQA/MQA、prefill/decode，随后
causal/非方形绝对位置/tails/空序列/全 mask，再到 varlen/paged/sparse/persistent。
全 mask 的 -inf-(-inf) 必须定义；split-KV 增加全局合并语义，backward/dropout/RNG
另立合同。新语法不自动获得 legacy 变体支持。

## 8. 同步前后与诊断

局部例的物理访问及必要同步关系：

```text
前：MTE2 X[g]→x[g%2] ; V x[g%2]→y[g%2] ; MTE3 y[g%2]→Y[g]
后：MTE2.write_done[g] → V.read_begin[g]       # ready
    V.read_retired[g] → MTE2.write_begin[g+2]  # input release
    V.write_done[g] → MTE3.read_begin[g]       # ready
    MTE3.read_retired[g] → V.write_begin[g+2]  # output release
```

删 input release 可覆盖在读 x；将其推到输出 MTE3 会扩大等待范围。上述是语义
关系，不是已分配 ID 的完整程序，配对/零域/event 复用/drain 还要检查。真实
同步前后 IR 与序列见 [synchronization](synchronization.md) 和
[region-sequence](region-sequence.md)。每个新增问题交付同样的最小反例。

错误分三类：已证非法、不支持、分析未决。报告 source 位置、双方任务/代次、
真实 root/字节范围、相关 phase/计数。预算耗尽不称程序非法；失败不改变政策，
不生成可运行 fallback，不写入成功缓存。

当前可运行边界是单 local 流、受限父体和顺序组合；一般层次 mixed、连续商余
流、宏、完整 AST FA 仍待实现。支持状态统一查[设计的保存点表](design.md#7-实现边界与来源)，
不能把目标片段、source verifier 通过或有限 legacy 路径当作已贯通能力。
