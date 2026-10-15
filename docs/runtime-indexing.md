# Runtime 索引、原始轴与 cyclic 局部流水

可运行源码：[grouped.py](../examples/vector/grouped.py)。输入是
`X/Y[B,H,S,D]`，`S` 是正值且整除 `ROWS` 的运行时参数。每组处理
`flat=group+task*GROUPS`，通过商余取回 batch/head/row，使用两槽 UB 做
GM→Vector→GM。不要求任务总数整除组数。

## 保留什么，证明什么

source View 保留原始 allocation、每轴 SSA origin/static width、保留轴映射
和可选 valid prefix。lowering 先检查各轴范围，再转换成字节地址。仅证明
扁平地址在分配内不够：非法 `head=H` 可能恰好落到下一 batch。

参数事实独立于表达式身份，来自实际 `SizeOp/RuntimeSizeOp` 的 minimum 和
multiple。比如 minimum=1/multiple=16 的最小合法值是16，故 `S//16` 为正。
每条源 SSA unsigned division 都在化简前检查；即使 `bad-bad` 或 `bad*0`
取消了结果，负 dividend 或不确定的 divisor 仍不能进入已支持域。

地址证明使用受预算约束的稀疏整数多项式与 Euclidean 恒等式，例如：

```text
((flat // (H*tiles))*H + (flat // tiles)%H)*tiles + flat%tiles = flat
(S // ROWS)*ROWS = S             前提：S % ROWS == 0
```

这不是任意整数求解器。单个正规式最多256项，每次证明另有累计工作预算；
未知关系拒绝。runtime radix 不进入存储周期的 LCM，也不展开运行次数。
静态局部槽 `task%2` 保留为物理周期项。

## Cyclic 任务域

源 count 为 `ceildiv(max(0,total-group),GROUPS)`；物理 `npu.cyclic_count`
只保存 total SSA，groups 取自实际 Kernel。最终 verifier 独立检查：

- 第 g 代属于 `group+g*GROUPS`，local count 只包含真实任务；空 owner 无任务。
- 每个 GM root 的 `total*tile_bytes` 与实际分配大小相同，包括只读输入。
- 所有该 root 的别名共享相同 tile；代次步长为 `GROUPS*tile_bytes`，owner
  平移为 `group*tile_bytes`，每个 native row/子范围都在 tile 内。

因此不同 owner 的 frame 不重叠；局部 buffer 仍需正常的 RAW/WAR/WAW 同步。
这是对限定 affine 域的推导，有限枚举 oracle 另作回归，不冒充一般形式证明。

目前只接通 standalone local actor 的 cyclic stream。与 mixed、serial invocation、
sequence、prefix 的组合明确拒绝；任意动态 alias、间接 gather、runtime row pitch、
动态 task stride 及动态父 view 的进一步缩窄也拒绝。静态父 view 切片与动态
父 view 保留原样的切片可用；当前 IR 不保留额外父范围义务，不能丢掉后默许。

## 同步前后与生成代码

默认 B2/H3/ROWS16/D64/GROUPS20，每 tile 4096字节。source.xdsl 保留商余各轴；
before.xdsl 的 GM view 已化成 `offset=g*81920+core_id*4096`，UB 为
`(g%2)*4096`。after.xdsl 在相同三个 native task 间生成：

| 边 | 源代次→目标代次 | 保护的事实 |
|---|---|---|
| MTE2→V | g→g | UB 输入 ready |
| V→MTE3 | g→g | UB 输出 ready |
| V→MTE2 | g→g+2 | 输入槽的最后 reader 退休，下一次可写 |
| MTE3→V | g→g+2 | 输出槽搬出完成，下一次可写 |

后两条只为存在的后续消费者发布通知。无额外搬运或中间 PIPE_ALL，只有原有
最终 lifetime drain。地址/同步化简不改变 task、通知资源或等待位置。
对于任意等差ID序列，发射器直接生成`base+step*(ordinal%width)`；默认例的
ready使用`g%2`，release的Wait使用`(g-2)%2`。非等差映射保留准确选择。

生成的 owner count 只计算一次：

```cpp
const int64_t npu_remaining = total - core_id;
const int64_t npu_owned = npu_remaining > 0 ? npu_remaining : 0;
const int64_t n = npu_owned / 20 + (npu_owned % 20 != 0);
```

## 复现

从工作区根目录运行，构建的300秒总期限涵盖Python和CANN：

```bash
PYTHONPATH=compiler .venv/bin/xdsl-npu --input compiler/examples/vector/grouped.py --entry grouped_scale --output runs/reproduce/grouped --shared --timeout 300
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/grouped.py --entry grouped_scale --trials 5 --output runs/reproduce/grouped-cold
```

构建目录包含 `source.xdsl`、`before.xdsl`、`after.xdsl` 和 `grouped_scale.asc`。
全卡检查后选健康空闲物理卡，以物理6为例（执行时需重新检查）：

```bash
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 .venv/bin/python -m benchmarks.vector.run --example grouped --output runs/reproduce/grouped-device --library runs/reproduce/grouped/grouped_scale.so --physical 6 --counts 16 48 64 4096 131072 --repeats 30
```

最大例 B2/H3/S131072/D64/f32，每张量192MiB。runner复用同一二进制，检查全部
输出、输入不变与guard；计时为ACL event stream interval，包含host提交间隙。
它不代表FA或ACLNN性能验收。实际结果见[状态页](refactor-progress.md)。
