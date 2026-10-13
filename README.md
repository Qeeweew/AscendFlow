# xDSL NPU compiler

基于 xDSL 0.73.0 的独立 Python 编译器，目标 Ascend 910B / dav-2201，直接输出
ASC 并调用 CANN 9.1。源码不依赖 reference、TileLang、TVM 或 MLIR 构建。

当前可编译运行固定 shape 的 RMSNorm、rotate-half RoPE、稳定 Softmax、SwiGLU
和带 K 块累加的矩形 batched NT matmul。所有算子展开为同一个物理 IR，并经过
局部 AutoSync、最终 verifier 和 ASC 后端。独立核 grid、Scalar 捕获、strided
访问和多级 buffer 复用均由共享模型检查。
`Kernel` builder 明确表达原生计算、搬运、物理布局、子视图、slot 和固定 stage。
专家指定的 issue 顺序、stage 和 buffer 不会被同步器重排或扩大。
源码、测试和文档属于本产品 Git；环境和运行产物放在外层工作区。

```python
from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.examples import vector_pipeline

physical_ir = vector_pipeline(count=6, tile=128, slots=2)
plan_ir = autosync(physical_ir)
print(verify_plan(plan_ir))
print(emit_asc(plan_ir))
```

实现路径：builder → xDSL NPU physical IR → 物理访问依赖 → 本地同步 PlanIR →
独立最终验证 → ASC → CANN。task 的 SSA 结果是身份，不能作为异步完成证明。
buffer/view 保留分配身份和字节范围；任务的效果来自封闭的原生 recipe 表。
生成的同步保留 RAW/WAR/WAW 原因、物理范围、代次和最终完成路径。

## 重现首个闭环

在 `/workspace/xdsl-910b` 执行：

```bash
.venv/bin/pip install --no-build-isolation -e compiler
.venv/bin/pytest -q compiler/tests
.venv/bin/ruff check compiler

.venv/bin/xdsl-npu vector --count 6 --output runs/stage1/vector --compile --shared
.venv/bin/xdsl-npu last-read --count 5 --tail 24 --output runs/stage1/last-read-tail --compile --shared
.venv/bin/xdsl-npu cube --count 5 --output runs/stage1/cube --compile --shared
```

每个输出目录包含 `before.xdsl`、`after.xdsl`、`witnesses.json`、验证摘要、ASC、
设备 ELF、可选 host launch 库，以及精确编译命令、版本和文件 SHA256。
`xdsl-npu ir --input ... --output ...` 也接受未同步的文本 xDSL IR。
当前静态有限验证预算是 4096 个 task instance，超出后拒绝，不运行未经检查的计划。

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

| 能力 | 实际支持 |
|---|---|
| 执行域 | AIV 或 AIC 的独立核 grid；检查 GM 写入/读取的跨 owner 不重叠 |
| 控制 | 静态有限单 stream、连续 task 激活区间、once cache；固定 stage 的填排；零次执行 |
| 存储 | UB/L1/L0A/L0B/L0C 容量和对齐；GM 输入契约；嵌套子视图和跨 dtype 字节别名 |
| 代次 | 显式线性或 modulo 地址；多缓冲；完整 RAW/WAR/WAW 和多 reader |
| Vector | f32 Add/Sub/Mul/Div/Exp/Rsqrt、标量和逐行计算；静态末块，DMA 长度为 32 字节倍数 |
| 归约 | 64 元素 Sum/Max partial、每行合并、Sqrt/Div 和精确统计量 footprint |
| Scalar | f32 UB word 读取，显式 SSA 值、同步读取和发射时捕获；支持 Scalar Wait 的全核 dispatch 门控 |
| Cube | 完整 f16 矩形 MAD，f32 clear/跨 K 块累加，最终 NZ→ND Fixpipe；受实际 bank 容量约束 |
| 搬运 | GM→UB、UB→GM、GM→L1、原生 GM ND→L1 NZ、完整 tile 的 L1→L0A/B |
| 同步 | 本地二值事件、同流水必要 barrier；每有向 bank 八 ID；配对和复用验证 |
| 保护 | executing ops 的 xDSL effects 防止普通 DCE/CSE 删除异步工作 |

不支持的动态 task count、数据相关分支/带洞激活、间接 gather/scatter、任意 native extern、
跨 AIC/AIV 交接、跨核 ownership、全局 phase 和通信会被拒绝。
没有直接 L0C→UB 路径、事件溢出的 mutex fallback，或静默缩小 tile/扩大 buffer。
普通大矩阵的多级 M/N 分块、融合 FFN、完整 attention、间接 routing 和动态 grouped
matmul 尚未实现。batched NT matmul 的每个矩阵由一个 AIC 执行，不能把它当作
任意规模的高性能 GEMM。算子 shape、数值和存储契约见 [算子入口](docs/operators.md)。

同步生成使用有限图提案和最终重检；有界交错检查独立探索 task 开始、退休、Set
发射、生效和 Wait 消费。两者共享受审查的 native effect 表，不能当作独立硬件
语义来源，也不能证明任意规模或任意动态程序。数值和性能仍需设备验证。

参见 [同步规则与反例](docs/synchronization.md)、[源码依据](docs/source-notes.md)
和 [阶段验证记录](docs/validation-stage1.md)。阶段1对应 Git `6a54f43`，该记录
保留当时的验证范围。后续基础能力见 [物理执行扩展](docs/physical-extensions.md)。

## 编译和运行算子

在工作区执行；运行前依据全部卡的健康和占用选择物理卡，下面使用物理卡1、逻辑0：

```bash
.venv/bin/xdsl-npu rms-norm --rows 256 --columns 1024 --blocks 8 --output runs/operators/rms --compile --shared
.venv/bin/xdsl-npu rope --rows 4096 --columns 128 --blocks 32 --output runs/operators/rope --compile --shared
.venv/bin/xdsl-npu softmax --rows 256 --columns 1024 --blocks 8 --output runs/operators/softmax --compile --shared
.venv/bin/xdsl-npu swiglu --rows 256 --columns 1024 --blocks 8 --output runs/operators/swiglu --compile --shared
.venv/bin/xdsl-npu batched-matmul --batch 8 --m 64 --n 64 --k 256 --k-block 64 --output runs/operators/matmul --compile --shared

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/operators/matmul --physical 1
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_reference.py runs/operators/rms --physical 1
```

`run_device.py` 对上述任一目录执行独立数值 oracle，并保存数据、源码和产物 hash。
`run_reference.py` 直接执行安装的 ACLNN RMSNorm/Softmax，作为独立比较基线。
[算子验证记录](docs/validation-operators.md) 区分设备正确性、ACL stream 间隔和
msprof 内核时间，记录当前性能证据及其限制。
