# AST 运行时长度与精确尾部前缀

这是已接通的 R1 子集，不是通用分支/嵌套同步。源码在
[ast_prefix.py](../examples/ast_prefix.py)，独立字节访问与负例在
[test_prefix.py](../tests/test_prefix.py)。

## 输入及执行语义

`scale_prefix(X, Y, n, m)` 处理 `min(n,m)` 个 f32 元素，保留 Y 的其余元素。
n、m 都是运行时原始元素长度；仅 B=64 为 constexpr。输入和输出按各自实际
长度分配，没有向 tile 边界补齐。循环体保持一份，设备上计算：

```python
length = N.min(n, m)
for t in N.Pipelined(N.ceildiv(length, B), stages=3):
    valid = N.min(B, length - t * B)
    # load / Vector mul / store 均明确传入 valid=valid。
```

当前要求 `min(n,m) % 8 == 0`，使 f32 DMA 的真实字节数满足32字节约束；
例如 n=65、m=8 合法，n=7、m=64 被拒绝。该子集只支持一个独立 AIV owner、
连续 GM stream 和等宽对齐槽位。没有静默扩大搬运、补零或回退全局 barrier。

host launcher 检查负长度、派发步数及 extent 乘法溢出、DMA 有效长度对齐、
非空指针/基址对齐、指针加法溢出和声明的外部区间不重叠。调用者仍须提供
符合声明长度的实际分配；C ABI 无法凭裸指针验证 allocator 的真实分配大小。

## IR 和同步

实际 xDSL 中保留两个 runtime 参数、`arith.minsi` 和 `arith.ceildivui`。
物理 `npu.stream` 的 limit 引用该 SSA 上界，GM buffer 的 extent 则分别引用
n 和 m。以下为省略类型/常量定义的结构摘录：

```text
%length = arith.minsi %n, %m
%count  = arith.ceildivui %length, %tile
X: GM extent_count=%n, count_scale=4
Y: GM extent_count=%m, count_scale=4
stream limit=%count, generation=%g:
  load:    X[g*256 : +valid*4] -> inputs[g%2]
  compute: inputs[g%2] -> outputs[g%2], valid elements
  store:   outputs[g%2] -> Y[g*256 : +valid*4]
```

物理 view 通过 SSA `extent_count=%length` 和 `prefix_elements=64` 表达真实
访问 `[address(g), address(g)+4*min(64,length-g*64))`。其最大 tile 为256字节，
但不能把这个最大范围当成尾块已经写满的证据。

同步后仍为原双缓冲四条通道，每 bank 使用 IDs 0/1：

| 通道 | 完成条件与等待位置 |
|---|---|
| MTE2→V ready | 当前 load 写完成后，当前 Vector 才能读 |
| V→MTE2 release | Vector 最后读取 inputs[g%2] 后，load[g+2] 才能覆盖 |
| V→MTE3 ready | 当前 Vector 写 outputs 完成后，当前 store 才能读 |
| MTE3→V release | store 最后读取 outputs[g%2] 后，compute[g+2] 才能覆盖 |

原 full-tile AST 示例与 prefix 示例的最终 ASC 同步行逐条一致；仅 native 长度
变成真实有效前缀。x 的释放不等待输出 store；prologue 不等不存在的旧代次，
末尾不产生没有后续消费者的 release。循环内无 `PIPE_ALL`，返回前保留 drain。

## 为什么当前证明足够，哪里会拒绝

每个 clipped allocation 的全部访问须使用同一个逻辑长度和前缀、相同 dtype/
cell 宽度及对齐的整个 cell 起点。对于合法循环，所有真实任务的前缀非空，
只有最后一个 generation 可以不足一块，且之后没有新的 generation。
因此删去最终任务从未访问的后缀不会改变 cell 间的 RAW/WAR/WAW 关系或初始化
义务。已有周期 frontier/完成图可以在这个经过检查的条件下证明同步。

这个条件不是“对写范围任意取包围盒”。最小反例是 clipped load 只写8个元素，
后续 Vector 按64个元素读取：会被 `full and clipped accesses` 检查拒绝。
不同 prefix、错长度绑定、32字节错位 alias、普通非对齐尾部、跨 owner prefix
及未知 native footprint 也会拒绝，不用扩大等待掩盖无法证明的访问。

测试独立解释实际 SSA 长度和字节地址，枚举初始化、RAW/WAR/WAW，再对照实际
通知/等待/派发硬件 DAG；它没有调用生产依赖分析来生成期望冲突。测试范围为
14组小规模长度。符号验证仍依赖上述索引域和已审查的 native 契约，有限测试
不构成任意程序/硬件的形式化证明。

uniform 标量 DAG 的每个节点只生成一次，避免嵌套 min/max 拼接三目表达式时
复制源码。当前表达式证明有4096项/128层预算；复杂未知表达式返回诊断。
native 模板使用独立 `Templates` 表示，已移除假定 `Program(count=0)` 的容器。

## 重现与证据

工作区根目录执行：

```bash
.venv/bin/python compiler/examples/run_prefix.py --shared --output runs/refactor/ast-prefix-final
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_prefix.py --entry scale_prefix --trials 5 --output runs/refactor/ast-prefix-cold
```

检查全部卡后，显式选择健康空闲物理卡；此次为卡1映射逻辑0：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_prefix.py --output runs/refactor/ast-prefix-device --library runs/refactor/ast-prefix-final/scale_prefix.so --physical 1
```

同一 binary 的14组测试，最大16,777,224个处理元素，每组5次，误差0，输出
未处理后缀和 guard 不变；6种非法调用在 host 返回 -1。CANN9.1/dav-2201
离线及 shared 编译成功。运行数据、before/after IR、最终 ASC、工具链选项和
hash 均在上述外层目录。此次不记录设备吞吐率，不据此宣称算子性能达标。
