# xDSL NPU compiler

基于 xDSL 0.73.0 的独立 Python 编译器，目标 Ascend 910B / dav-2201，直接输出
ASC 并调用 CANN 9.1。源码不依赖 reference、TileLang、TVM 或 MLIR 构建。

2026-10-14对照保存点 `be957e8` 重写了[总体设计](docs/design.md)、
[DSL 前端与 CV 多阶段规格](docs/dsl-multistage.md)和[重构计划](docs/refactor-plan.md)。
新计划分别验收语言表达、编译成本和最终机器执行；规则跨query连续CV流水
进入本轮范围，动态任务队列另行扩展。独立FA参考已补齐；下一步收敛区域lowering
入口，并提前检验连续流的query/task代次与编译规模，再接通层次CV和真实FA。
受限 AST 已接通单循环 Vector、运行时原始
长度与精确对齐尾部；
一般多层CV同步和真实FA迁移尚未完成。新 `prim_func` 不执行设备函数体，遗留
构造明确使用 `trace_func`。实现边界和历史证据见[重构记录](docs/refactor-progress.md)。

新的[FA目标源码](docs/attention-target.md)已将嵌套/连续两种组织、真实online
数学及声明存储固定在examples，仍是明确拒绝编译的设计fixture。
[独立AscendC参考](docs/attention-reference.md)已完成小例数值、8K大例采样/profile
及硬件下界分析；同时发现并修正fixture的单槽L0派发顺序。参考不代表新AST FA
已经可运行，首批性能也不是最终竞争目标。
[顺序区域组合](docs/region-sequence.md)已接通根部单actor的普通工作、多个流水、
真实初始化与异步出口协议。[state_sum](examples/ast_state_sum.py)有1 GiB设备例
及独立AscendC/profile证据，冷源码→ASC p50约0.185秒。父serial体一般组合、
跨region事件ID复用和mixed层次仍未接通，不能由此推断FA性能已达标。

两个独立长度参数、`ceildiv` 循环和有效前缀的可运行例子见
[ast_prefix.py](examples/ast_prefix.py)；实际同步、拒绝条件和设备命令见
[运行时尾部](docs/runtime-prefix.md)。

归约、Scalar 最后读取及专家派发顺序已在
[normalization 示例](examples/ast_normalize.py)接通；其独立 AscendC 对照、
8192×4096设备 profiling 和硬件工作量分析见[验证记录](docs/ast-normalization.md)。
前端已有整数/布尔 SSA 合流与循环携带值，一般结构化控制的物理同步仍待实现。

R2 已将存储地址周期与同步执行条件周期分开，避免无关 buffer 周期相乘后
复制整个完成图。[互素槽位示例与证明边界](docs/factored-periods.md)包含最终
ASC 对照、有限逐字节反例、冷编译和大长度设备验证；嵌套流水仍待接通。

[显式 Cube 示例](examples/ast_cube.py)已接通 L1/L0A/L0B/L0C 独立槽位、
`copy → matmul → Fixpipe`，支持已验证的完整 native tile NN/NT 与 f16/bf16。
400 MiB固定配置的独立 AscendC/profiling、硬件工作量和支持边界见
[Cube 验证记录](docs/ast-cube.md)。[运行时 K 归约](docs/runtime-reduction.md)进一步
接通首代清零和末代一次 Fixpipe，960 MiB固定例已对照 AscendC/profiling；
该例本身是单层流；父子流水与跨调用生命周期见下方闭环。

R2a的[父输出×子K程序](examples/ast_hierarchical_gemm.py)已接通物理父子区域、
GM frame、跨调用同步及最终检查；960 MiB固定例通过AscendC/profiling对照，
完整源码冷编译p50约0.689秒。支持边界、前后IR/ASC、设备证据和复现命令见
[运行时层次流水](docs/runtime-hierarchy.md)。一般多层CV及真实FA迁移仍未完成。

当前源语言用 `Pipelined / Stage / Cube / Vector` 表达核内和 CV 多阶段程序。
源程序只写一个逻辑任务 `t`；编译器位移访问及 descriptor，生成填排、收尾和
实际同步。软件 stage、buffer versions 和硬件 event 资源各自独立。

```python
from __future__ import annotations
from xdsl_npu import language as N
from xdsl_npu import compile

@N.prim_func(disjoint_external=True)
def scale(X: N.Tensor((tiles, B), N.f32),
          Y: N.Tensor((tiles, B), N.f32, access="write"),
          tiles: N.int64, B: N.constexpr = 64):
    with N.Kernel(kind="vector"):
        x = N.alloc_ub((2, B), N.f32)
        y = N.alloc_ub((2, B), N.f32)
        for t in N.Pipelined(tiles, stages=3):
            with N.Stage(0, name="load"):
                N.copy(X[t, :], x[t % 2, :])
            with N.Stage(1, name="compute"), N.Vector():
                N.mul(x[t % 2, :], 2.0, out=y[t % 2, :])
            with N.Stage(2, name="store"):
                N.copy(y[t % 2, :], Y[t, :])

compiled = compile(scale)
print(compiled.asc)
print(compiled.report)
```

遗留 CV 示例直接写 Stage0/Cube、Stage1/Vector、Stage1/Cube、Stage2/Vector。
两处 Stage1 是独立 region；Cube region 内仍有 MTE2、MTE1、Cube、Fixpipe 的
各自最后读取和复用约束。静态内层预取拥有独立 clock，不把整个 CV task 改成
一个扁平微步索引。具体语义、分段例子和设备结果见
[重构记录](docs/refactor-progress.md)；目标语义见[multi-stage DSL](docs/dsl-multistage.md)。

实现路径：typed source → xDSL source regions → NPU physical IR → AutoSync →
最终实际 IR 检查 → ASC → CANN。逻辑 tile 是物理 buffer 的借用值，不是隐式快照，
SSA use-def 不承诺异步完成。覆盖后的旧 tile 会拒绝；不会加入隐藏搬运。
`compile()` 返回不可变 ASC、IR 字符串和验证摘要，重复消费避免完整重检。
外部可变 physical/PlanIR 仍重新验证；checker 与 planner 共享部分分析，不能称
完全独立证明。底层物理 builder 当前保留给专家及反例。

编译器包只包含 DSL、IR、同步、后端和通用工具。RMSNorm、RoPE、Softmax、
SwiGLU、GEMM、GDN、Flash Attention 及其 oracle、AscendC/ACLNN 对照均在
`examples/`。这些旧算法仍在迁移源语法；其物理构造入口为
`examples/build_legacy.py`，不能当作新 DSL 已覆盖全部算子。算子调参仍暂停，
Python 编译性能和目录重构的进度见 [重构记录](docs/refactor-progress.md)。
源码、回归、示例和产品文档属于独立产品 Git；环境和运行数据在外层工作区。

## 重现首个闭环

在 `/workspace/xdsl-910b` 执行。AST 前端的向量闭环和包含前端的五次冷编译计时：

```bash
.venv/bin/python compiler/examples/ast_vector.py --output runs/refactor/ast-vector --shared
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_vector.py --entry scale --trials 5 --output runs/refactor/ast-vector-cold
```

设备运行前先检查全部卡；下例显式选择物理卡1，runner 再次检查：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/runtime_vector.py --frontend ast --output runs/refactor/ast-vector-device --library runs/refactor/ast-vector/scale.so --physical 1 --counts 0 1 2 4 4096 65536 262144
```

以下是显式 legacy 构造的局部和双向 CV 闭环：

```bash
.venv/bin/python compiler/examples/multistage.py vector --count 4 --output runs/refactor/vector --shared
.venv/bin/python compiler/examples/multistage.py segmented-cv --count 4 --segments 3 --groups 2 --output runs/refactor/segmented-cv --shared
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_multistage.py runs/refactor/segmented-cv --physical 1
```

`source.xdsl`、`before.xdsl`、`after.xdsl` 和 `schedule.json` 对照源 region 与实际
派发；所有新 QK segment 在旧 P-ready Wait 之前发射。分段 CV 是同步和数值例子，
不是 FA 或吞吐率 benchmark。遗留物理反例仍可重现：

```bash
.venv/bin/pip install --no-build-isolation -e compiler
.venv/bin/pytest -q compiler/tests
.venv/bin/ruff check compiler

.venv/bin/python compiler/examples/build_legacy.py vector --count 6 --output runs/stage1/vector --compile --shared
.venv/bin/python compiler/examples/build_legacy.py last-read --count 5 --tail 24 --output runs/stage1/last-read-tail --compile --shared
.venv/bin/python compiler/examples/build_legacy.py cube --count 5 --output runs/stage1/cube --compile --shared
```

每个输出目录包含 `before.xdsl`、`after.xdsl`、验证摘要、编译阶段计时、ASC、
设备 ELF、可选 host launch 库，以及精确编译命令、版本和文件 SHA256。
`xdsl-npu --input ... --output ...` 接受 typed source 或未同步物理 IR；
`--synchronized` 检查专家 PlanIR，`--explain` 显式导出逐访问 witnesses。
当前静态有限验证预算是每 actor 4096 个 task instance，超出后拒绝，不运行未经检查的计划。

设备运行需先选择健康空闲卡。runner 在每个进程加载 ACL 前重新检查全部卡的
健康、进程、HBM 和利用率；以下映射显式选择物理卡 0、逻辑卡 0：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/stage1/vector --physical 0
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/stage1/last-read-tail --physical 0 --late-reader
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/stage1/cube --physical 0
```

没有设备 reset、换卡重试或 `/tmp` 锁。设备错误会终止运行。runner 还检查库的
SHA256 和生成源码是否与当前 PlanIR/backend 相符，防止执行失效产物。

固定手排参考来自产品内的 `examples/manual_vector.asc.in`，只用一个固定配置对照：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/compare_vector.py runs/stage1/comparison --count 32 --tile 128 --physical 0 --run
```

它单独编译手排代码并记录双方源码、选项、shape、dtype、设备和计时口径。
手排 ASC 是外部参考，不能冒充经 PlanIR verifier 验证的任意 C++。

## 当前支持边界

下表主要记录遗留有限物理路径，不能直接套用到当前 AST/符号循环路径。
AST 路径支持多个非负 uniform 参数及 min/max/ceildiv 上界；对齐尾部目前限于
单独 AIV 和共同有效前缀。单actor末代effect后缀已接通；一般分支与嵌套循环
保留为IR；目前接通单local actor的serial父循环×一个pipeline子循环，以及
根部单actor普通工作与流水的顺序组合。父体一般组合和mixed层次仍未接通；
legacy 静态内层仍有展开。两条路径的准确边界见[重构记录](docs/refactor-progress.md)。

| 能力 | 实际支持 |
|---|---|
| 执行域 | 独立 AIV/AIC grid；或每组 1 AIC+2 AIV 的静态 GM handoff；各组 GM ownership 不重叠 |
| 控制 | 静态有限单 stream、连续/周期 task 激活、once cache；固定 stage 的填排；零次执行；按 group 精确分区的静态 mixed cases |
| 存储 | UB/L1/L0A/L0B/L0C 容量和对齐；GM 输入契约；嵌套子视图和跨 dtype 字节别名 |
| 代次 | 常量整数 divisor/modulo 地址与二维核坐标；多缓冲；完整 RAW/WAR/WAW 和多 reader |
| Vector | f32 Add/Sub/Mul/Div/Exp/Rsqrt、列 bias 广播、标量和逐行计算；静态末块，DMA 长度为 32 字节倍数 |
| 归约 | 64 元素 Sum/Max partial、每行合并、Sqrt/Div 和精确统计量 footprint |
| 状态 | 64-key 静态 GDN；独立 FP32 carry、snapshot 环及每 token 两份输出；原生 vmla、vcadd 与 bitwise vcopy |
| Scalar | f32 UB word 读取，显式 SSA 值、同步读取和发射时捕获；支持 Scalar Wait 的全核 dispatch 门控 |
| Cube | f16/bf16 矩形 MAD、逐输出 f32 clear/跨 K 累加、NZ→ND Fixpipe；短 MAD 的实际 M 完成约束 |
| 搬运 | GM↔UB（含原生二维 DMA）、GM→L1、原生 GM ND→L1 NZ、完整 tile 的 L1→L0A/B；NN B 经原生转置 Load2D 进入 L0B |
| 同步 | 本地二值事件、必要同流水 barrier；1..4 个 GM 环、两个交接方向、mode-2 计数、双 AIV quorum；逐义务验证后删除冗余 credit |
| Event分配 | 每有向bank八ID；跨channel共享已证明不相交的生命周期；有限epoch映射与资源压力诊断 |
| 保护 | executing ops 的 xDSL effects 防止普通 DCE/CSE 删除异步工作 |

遗留有限路径不支持动态 task count；数据相关分支/带洞激活、间接 gather/scatter、任意 native extern、
动态或不平衡跨核 ownership、跨 mixed group 交接、全局 phase 和通信会被拒绝。
没有直接 L0C→UB 路径、事件溢出的 mutex fallback，或静默缩小 tile/扩大 buffer。
融合 FFN、varlen/paged/sparse/persistent attention、间接 routing 和动态 grouped matmul 尚未实现。GEMM
当前要求选定 tile 整除静态 M/N/K，算子入口输入 f16、输出 f32；未支持一般尾块。
算子契约见 [算子入口](docs/operators.md)，分层/连续 GEMM 与设备反例见
[GEMM 实现和验证](docs/hierarchical-gemm.md)。
实际混合核协议、自动同步前后、设备证据和复现命令见
[混合核 GM ownership](docs/mixed-ownership.md)。
递推 carry、snapshot 的独立最后读取及当前 event packing 限制见
[静态 GDN](docs/recurrent-gdn.md)。
局部 channel 生命周期共享及输入代次选择见 [event 分配](docs/event-allocation.md)。

legacy Flash Attention 的数值契约、三环同步、变体和大规模性能见
[Flash Attention](docs/flash-attention.md)。4K MHA/8K GQA 新生成代码相对初版分别
快2.43/2.50倍，仍约为安装的 CANN 内核时间的2.01/2.13倍；没有宣称性能达到专家目标。

同步生成使用有限图提案和最终重检；有界交错检查独立探索 task 开始、退休、Set
发射、生效和 Wait 消费。两者共享受审查的 native effect 表，不能当作独立硬件
语义来源，也不能证明任意规模或任意动态程序。数值和性能仍需设备验证。

参见 [同步规则与反例](docs/synchronization.md)、[源码依据](docs/source-notes.md)
和 [阶段验证记录](docs/validation-stage1.md)。阶段1对应 Git `6a54f43`，该记录
保留当时的验证范围。后续基础能力见 [物理执行扩展](docs/physical-extensions.md)。

## 编译和运行算子

在工作区执行；运行前依据全部卡的健康和占用选择物理卡，下面使用物理卡1、逻辑0：

```bash
.venv/bin/python compiler/examples/build_legacy.py rms-norm --rows 256 --columns 1024 --blocks 8 --output runs/operators/rms --compile --shared
.venv/bin/python compiler/examples/build_legacy.py rope --rows 4096 --columns 128 --blocks 32 --output runs/operators/rope --compile --shared
.venv/bin/python compiler/examples/build_legacy.py softmax --rows 256 --columns 1024 --blocks 8 --output runs/operators/softmax --compile --shared
.venv/bin/python compiler/examples/build_legacy.py swiglu --rows 256 --columns 1024 --blocks 8 --output runs/operators/swiglu --compile --shared
.venv/bin/python compiler/examples/build_legacy.py batched-matmul --batch 8 --m 64 --n 64 --k 256 --k-block 64 --output runs/operators/matmul --compile --shared
.venv/bin/python compiler/examples/build_legacy.py gemm --m 512 --n 1024 --k 768 --tile-m 128 --tile-n 128 --l1-k 256 --l0-k 64 --grid-m 2 --c-slots 2 --cache-b --output runs/gemm/persistent-512-1024-768 --compile --shared
.venv/bin/python compiler/examples/build_legacy.py mixed --count 17 --slots 2 --check-interleavings shared --output runs/mixed/17-epochs --shared
.venv/bin/python compiler/examples/build_legacy.py gemm-bias-silu --m 768 --n 1024 --k 768 --tile-m 128 --tile-n 128 --l1-k 256 --l0-k 64 --grid-m 2 --c-slots 2 --cache-b --output runs/mixed/768-1024-768 --shared
.venv/bin/python compiler/examples/build_legacy.py gdn --sequences 32 --steps 16 --value-dim 64 --slots 2 --snapshot-slots 2 --output runs/gdn/32-16-64 --compile --shared

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/operators/matmul --physical 1
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_reference.py runs/operators/rms --physical 1
```

`run_device.py` 对上述任一目录执行独立数值 oracle，并保存数据、源码和产物 hash。
`run_reference.py` 直接执行安装的 ACLNN RMSNorm/Softmax/Matmul，或 Matmul+Add+SiLU，作为独立比较基线。
[算子验证记录](docs/validation-operators.md) 区分设备正确性、ACL stream 间隔和
msprof 内核时间，记录当前性能证据及其限制。

Flash Attention 示例及完整自动同步前后 IR：

```bash
.venv/bin/python compiler/examples/flash_attention.py --batch 1 --query-heads 2 --kv-heads 1 --query-length 64 --kv-length 256 --head-dim 64 --shared --output runs/attention/example
.venv/bin/python compiler/examples/flash_attention.py --batch 1 --query-heads 8 --kv-heads 8 --query-length 4096 --kv-length 4096 --head-dim 128 --block-q 128 --q-cache l1 --reuse-ub --pipeline --packed-reductions --shared --output runs/attention/mha-4k-d128
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python compiler/examples/run_attention.py runs/attention/example --physical 1
```

大规模使用明确的 `--oracle sampled`；逐 head 和 query tile 采样数值，并对全部输出
检查 finite。`attention_performance.py` 将最终 IR 的原生工作量、真实硬件信息和
msprof 活动计数绑定到源码/库/CSV hash；条件吞吐上界不替代设备时间或 HBM 实测。
