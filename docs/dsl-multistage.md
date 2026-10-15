# DSL：运行时循环与多级流水

2026-10-15。本文区分**可运行子集**与**下一部分的语言合同**。
实际后端支持见[状态页](refactor-progress.md)，统一同步设计见[设计](design.md)。

## 前端选择

采用 TileLang 风格的 tensor、显式存储、循环和作用域，受限 Python AST 进入
xDSL；不执行设备函数体来 tracing 运行次数，不使用 `cube.task/vector.task`。

| 语法 | 意义 |
|---|---|
| `@T.prim_func` | 捕获设备源码及位置，不执行函数体 |
| `T.Tensor(...), n: T.int64` | 外部引用及运行时参数；长度不是 constexpr |
| `B: T.constexpr` | tile 容量、layout/dtype、固定政策等显式特化值 |
| `T.Kernel(groups, kind=...)` | launch 域；mixed 每组一个 AIC 和两个 AIV |
| `T.Cube()`、`T.Vector() as lane` | actor 作用域；engine 由原生操作决定 |
| `T.serial(N)` | 保留循环与源顺序，退出没有隐式 drain |
| `T.Pipelined(N, stages=S, issue_order=...)` | 逻辑任务域、stage 偏移范围和每 actor 派发政策 |
| `T.Stage(s, name=...)` | 最近 pipeline 的逻辑任务偏移，不是完成边界 |
| `alloc_ub/l1/l0a/l0b/l0c`、views | 显式物理空间、容量、layout、slot、别名 |
| `copy/matmul/reduce/...` | typed 操作；实际读写和 scratch 必须可见 |

语法被前端识别不表示所有组合均可编译。AST 嵌套 pipeline、一般 mixed 层次、
runtime 商余时钟及宏尚未贯通。显式 unroll 受预算限制；普通常量界不授权展开。

## 可运行：三阶段双缓冲

完整入口在 [scale.py](../examples/vector/scale.py)，下面与其计算和存储政策相同：

```python
from __future__ import annotations
from xdsl_npu import language as T

@T.prim_func(disjoint_external=True)
def scale(X: T.Tensor((tiles, B), T.f32),
          Y: T.Tensor((tiles, B), T.f32, access="write"),
          tiles: T.int64, B: T.constexpr = 64):
    with T.Kernel(1, kind="vector"):
        x = T.alloc_ub((2, B), T.f32)
        y = T.alloc_ub((2, B), T.f32)
        for t in T.Pipelined(tiles, stages=3):
            with T.Stage(0, name="load"):
                T.copy(X[t, :], x[t % 2, :])
            with T.Stage(1, name="compute"), T.Vector():
                T.mul(x[t % 2, :], 2.0, out=y[t % 2, :])
            with T.Stage(2, name="store"):
                T.copy(y[t % 2, :], Y[t, :])
```

同一个编译产物接收不同 tiles。双缓冲是显式两槽；stages=3 不分配三份 buffer。
ready 和 input/output release 由实际访问推导。输入最后 Vector reader 退休就
可以重载该槽，输出槽则等待 MTE3 读取退休。

[state](../examples/state/) 保留初始化、递推和导出；[prefix](../examples/vector/prefix.py)
展示两个 runtime 长度的对齐有效前缀。它们不代表一般动态别名或任意尾部已支持。

## Stage 与派发顺序

本地派发 step j 上 Stage s 使用逻辑坐标 `t=j-s`，仅 `0<=t<N` 时激活。
编译器位移完整环境：view、地址、guard、first/last、descriptor 及 child 捕获值。
纯 descriptor 表达式可在对应代次求值；带 effects 的 metadata 读取不能随意复制。

默认每个 actor 按源码直接 Stage 的顺序派发。`issue_order` 用当前 pipeline 中
唯一且完整的标签指定另一个顺序，不按 Stage 数字排序。policy 不改变逻辑数据
版本或数值累计顺序；违背依赖或造成等待环时给诊断，不偷偷改排步。

不同 actor 的 step 不是同一个墙钟，两个相同偏移 Stage 也没有共同完成点。
prologue/steady/epilogue 是激活域的边界；N=0 必须正确配对或不产生通知。

## 目标：CV 的四个阶段

下列为下一部分合同，函数名代表**待接通的 examples 数学块**，不是现有 opaque
算子或已实现宏。完整展开数学见 [FA 源码规格](design/attention_target.py.in)。

```python
for kv in T.Pipelined(
    kv_tiles, stages=3,
    issue_order={"cube": ("qk", "pv"), "vector": ("update", "softmax")},
):
    with T.Stage(0, name="qk"), T.Cube():
        qk_export(kv, Q, K, Score, qk_scratch)
    with T.Stage(1, name="softmax"), T.Vector() as lane:
        online_softmax(kv, lane, Score, P, m, l, alpha, vec_scratch)
    with T.Stage(1, name="pv"), T.Cube():
        pv_export(kv, P, V, PV, pv_scratch)
    with T.Stage(2, name="update"), T.Vector() as lane:
        update_output(kv, lane, PV, alpha, l, o, O, out_scratch)
```

| 本地 step，三个逻辑任务 | AIC 派发 | 每个 AIV 派发 |
|---|---|---|
| 0 | QK[0] | 无 |
| 1 | QK[1]，PV[0] | Softmax[0] |
| 2 | QK[2]，PV[1] | Update[0]，Softmax[1] |
| 3 | PV[2] | Update[1]，Softmax[2] |
| 4 | 无 | Update[2] |

这是派发顺序，实际重叠由硬件和同步决定。Score 必须在对应 FIX 写完成后读取；
PV 等待两个 AIV 的完整 P 贡献；三个 GM 环分别等待自己的最后 reader 释放。
不能用单个 task-done 替代这些里程碑，不能把通知 ack 当作存储 release。

CV 必须能够表达不同专家政策。当前 update-before-softmax 政策与所审查的
ops-transformer arch22 Vec1-before-Vec2 不同；每种政策的状态分配都需重新验证。

## Stage 内的流水与跨 query 连续流

QK/PV 内仍有 `T.Pipelined(BK//CK, ...)`，Vector 内仍有行块 pipeline。通信
BK1024、native CK128、Vector VR8 独立选择，不能要求整个通信块都进入 UB。
父 kv 作为捕获值进入 child；child 的 chunk/row 坐标和 slot 独立。

单槽 L0 要先派发旧 MAD，再装下一块 operand，所以内层 order 是
load/MAD/operand(/FIX)。child 控制退出不等异步退休；共享 L0C 的 FIX→下一写
必须跨 child 组合。普通嵌套保持词法派发，不暗中将其他父 Stage 插入 child。

nested 和 continuous 共用数学主体，后者目标控制为：

```python
if K > 0:
    for task in T.Pipelined(queries * K, stages=3, issue_order=policy):
        query = task // K
        kv = task % K
        # 同一组四个 Stage，使用各自逻辑 task 的 descriptor。
```

query 状态、task 交接环、child 原生环的 generation 不可混同。新 query QK
可先于旧 query PV；Q cache 只等其实际最后 QK reader，不扩大到整个 query。
这类 runtime 参数乘积和商余关系尚未支持，不能靠枚举 K、隐式展开或改成
constexpr 实现。空 K 的输出由算法显式定义；当前 FA 目标限定正 Sk。

## 借用、数值和拒绝边界

view/recast 保留同一个 allocation 身份；不同名字不证明不别名。借用值不是
快照，覆盖后再读旧借用会拒绝。外部 disjoint 合同要检查实际指针区间；
initialized 只说明初值，不能授权永远只读或跨 owner 覆写。

共享 readonly、fresh frame、ring slot 是不同事实。只有经过实际访问和范围
验证的不相交 frame 才能省跨调用存储冲突；GM/local ring 不自动 fresh。
未知动态别名/ownership 要有可验证合同或明确拒绝，不以 barrier 掩盖。

FA 保持 FP32 m/l/O、p32 denominator、P RN、CK 累计、rescale/add 和最终归一化
边界。改变逻辑 BK、reduce 树或融合运算属于算法/数值变化，不能当同步优化。
宏未来仅做透明 AST 内联，scratch/effects 可见，内部普通循环仍保留。

报错要区分非法、不支持和预算未决，并带源位置、两端代次、字节范围与分析阶段。
新 AST 不静默 fallback 到 legacy、裸 ASC 或整算子 CANN 调用。
