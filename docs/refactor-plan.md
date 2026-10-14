# 下一阶段：DSL、编译耗时与产品边界重构

2026-10-14。原算子优化目标保持暂停。`06fce4d` 保存重构前代码；这份计划不代表
新的 DSL 或编译性能改进已实现。本阶段不增加算子变体、不继续调算子排步。

## 已确认的问题

1. `Kernel.task(uid, native, *refs, params=..., stage=..., issue_order=...,
   logical_phase=..., period=..., phase_width=...)` 同时承担指令选择、物理访问、
   算法顺序和派发。字符串、位置参数和多个整数时钟使算子难读，也容易把语义
   阶段与执行阶段弄混。给这个接口增加几个缩写不能解决分层问题。
2. 依赖分析对每个访问扫描 root 的全部切分区间；完成图给每个 Set/barrier
   连上该 pipe 的全部历史 task retirement。候选同步删除、宽度选择再反复
   重建同样的图。`build_module`、ASC 输出、witness 导出有重复验证和提取。
   这些成本存在明显的非线性增长，需以 CPU profile 确定修复优先级。
3. FA 已移到 examples，但 `kernels.py`、`recurrent.py`、对应数值 oracle、
   CANN 算子适配器和 CLI 分支仍在包内。编译器核心仍包含具体算法知识。
4. 当前优化证据主要是生成 ASC、参考 AscendC 源码和安装 ACLNN 的 profiling。
   缺少可运行、数值契约明确的 AscendC 对照闭环。参考源码组织相近不能证明
   与安装二进制采用同样排步，也不能替代设备性能证据。

一次 CPU profile 已完成：输入为已保存的 dense 小例 IR，B1/H1/Sq128/Sk2048/D128，
单 mixed group、950 个实际 task。原始结果在产品仓库外的
`runs/refactor/cpu-baseline.log` 和 `.prof`；输入 SHA256 为
`08656eaa89ab69f71748ed9b5ee108dbb89f9891334d55884dc90306df4062e8`。

| 带 cProfile 的阶段 wall time | 秒 |
|---|---:|
| AutoSync | 100.03 |
| 再调用公共 verify | 4.95 |
| 公共 ASC emission（含内部重检） | 5.58 |
| witness 与 JSON 导出 | 9.53 |

调用统计约8292万次，局部完成图构造320次、图闭包361次、依赖分析12次、全组
依赖分析6次、全组验证41次；`Graph.edge` 被调用约2807万次。区间全扫描和
图闭包各占约25.9/25.7秒 self time。这是带 profiler 的单次测量，不是正常编译
计时，也不包含 CANN、NPU 执行或 Python 导入。它已足以确定重复分析、历史边
和区间扫描是首要改造点；下一阶段另建不带 profiler 的固定基线。

## DSL 方向

首版采用受限、类型明确的 Python 构造接口，直接形成 xDSL；不先实现通用 Python
AST 编译器。语义操作、存储和执行排步分别表达，收敛为三类对象：

- `Tile/Value` 表达计算结果、shape、dtype 与明确的数值操作顺序。
- `Buffer/View/Ring` 表达 memory space、布局、别名、有效范围、slot 和所有权。
- `Pipeline/Schedule` 用具名操作组和数据 tile 表达预取、执行顺序及驻留政策。
  native segment、通知 ordinal 和具体 event ID 由 lowering/同步器产生。

拟议的最小语法，尚未实现：

```python
with npu.kernel("scale", grid=grid) as k:
    inputs = npu.buffer((T,), dtype=npu.f32, memory=npu.UB, slots=2)
    outputs = npu.buffer((T,), dtype=npu.f32, memory=npu.UB, slots=2)
    with npu.pipeline(npu.tiles(X, size=T), prefetch=1) as tile:
        x = npu.load(X[tile], into=inputs.slot(tile))
        y = npu.mul(x, 2.0, out=outputs.slot(tile))
        npu.store(Y[tile], y)
```

Cube 使用具名 `matmul(a=..., b=..., accum=..., initialize=...)`、具名 transfer
和布局/operand placement，替代 `params=(m,n,k,...)`。所有操作由类型与命名参数
检查 shape、布局、内存空间和 dtype。精度转换、P 舍入、FP32 rescale/add 和递推
状态更新保持显式，不由所谓融合或默认政策改变数值边界。

专家通过独立 Schedule 指定真实操作组的相对排步和缓存，而不是在每条计算
调用里混入 `stage`、`logical_phase` 和任意 `issue_order` 数字。Schedule 必须
降低为可打印的物理派发表，供审查和生成 ASC；不能用它宣称操作已经完成。

默认和专家入口均进入同一物理 effect、last-read、ownership、event 和 drain
验证。SSA value 的就绪不能代替 native retirement；不同逻辑值不能代替物理
不别名。无法证明的动态访问仍明确拒绝，首轮不扩大当前动态控制支持范围。

先用双缓冲 Vector 和一个 Cube/Fixpipe 程序验证接口表达力，再迁移现有算子。
物理 TaskIR 可以作为后端表示保留，但不继续作为算子的公共源语言。

## Python 编译性能改造

先建立固定 CPU benchmark，分开记录 source construction、lowering、访问分析、
同步规划、最终验证、ASC emission 和可选诊断导出。记录 cold compile、重复编译、
peak RSS、实际 task/interval/graph 数量及候选次数；CANN/Bisheng 耗时独立统计。
正常计时与带 profiler 的计时必须区分。

按 profile 处理下列问题，不先换语言或引入新的构建栈：

1. 用 immutable 分析对象规划候选，不在每次试探中擦除/重建 xDSL。IR、访问和
   ownership 不变时复用对应分析；排步、存储、effects 或激活变化必须失效。
2. 精确区间索引代替每次扫描全部 cuts；只访问重叠区间，保留 holes 和真实
   native footprint。绝不以 bounding box 或忽略别名换取速度。
3. 用线性 completion-prefix 节点压缩重复历史边。prefix 只汇总此前退休，
   不增加后续源 task 的完成约束，不能重新引入“Notify 是 source fence”的错误。
4. event 宽度/ID 选择复用与 ID 无关的完成关系；冗余同步删除按受影响义务检查，
   避免逐候选全程序重建。最终独立 checker 仍从最终物理 IR 检查全部义务。
5. `compile` 返回拥有已冻结 PlanIR 的编译产物，供 ASC 和诊断消费。同一不可变
   产物避免重复完整验证；接收外部或可变 IR 的入口仍重新验证。不能凭一个
   可伪造的 `verified=True` 或历史 witness 放行。
6. 普通 compile 只生成代码和必要摘要；逐字节 witness、完整图和有界交错是
   显式诊断选项。关闭诊断不关闭同步正确性检查，也不改变同步序列。

暂定的可用性验收目标：在同一开发机上，常用静态内核的纯编译阶段不超过1秒，
4K/8K FA 不超过5秒，未修改 IR 的复用不超过100毫秒。它们是待实现目标，需用
固定 benchmark 的多次中位数和尾延迟验收；不能把 cProfile 时间当作正常计时。
还要检查规模翻倍的增长与峰值内存，不只对单个 shape 优化。达不到时记录具体
热点并调整实现，不通过缩减验证规模或增加同步来满足计时门槛。

## 目录与验证边界

- `src/xdsl_npu/` 只保留 DSL、xDSL IR/lowering、dav-2201 契约、物理分析、同步
  规划/验证、ASC 后端和通用 CANN toolchain。
- 所有具体算子与算法放 `examples/`，包括 GEMM、normalization、RoPE、GDN、
  FA 及变体。数值 oracle、算子专属 ACLNN 适配器、AscendC 对照和 benchmark
  随示例组织；共享运行辅助代码集中，避免每个算子复制 runner。
- 编译器 CLI 接收 IR/编译配置，示例自己提供 shape/dtype/算法选项；编译器
  核心不按算子名选择分支。
- 最小 hazard 程序作为测试 fixture 使用。编译器本体可安装、导入和测试通用
  编译能力，且不需要 examples、NumPy 或研究 checkout。

后续恢复算子优化时，每次必须同时保存和审查：

1. DSL、最终物理 IR 和 ASC：搬运、缓冲、预取、Wait 的实际阻塞范围。
2. 可运行 AscendC 对照：源码版本、kernel/tiling、算法/数值契约和编译选项。
   同契约手排基线用于比较编译开销；安装 ACLNN 作为额外产品基线。契约不同
   的实现要明确区分，不宣称复现同一个 kernel。
3. 大规模同卡设备正确性与 profiler：多次独立 trial，区分 device task 时间、
   stream 间隔、冷/热缓存和连续重复采样。检查所有卡，显式物理卡→逻辑0。
4. 硬件模型：实际 Cube 工作量、Vector 工作、GM 请求与 cache footprint、
   条件峰值/带宽假设。请求量不能当作 HBM 实测，活动比例不能当作因果证明。

## 实施顺序与交付门槛

| 阶段 | 交付 | 进入下一阶段前必须满足 |
|---|---|---|
| 0. 保存与基线 | 重构前 Git checkpoint；CPU profile；固定复现集 | 记录已验证/未验证范围，所有运行数据在产品 Git 外 |
| 1. 编译管线 | immutable analysis/产物、区间和完成图改造、按阶段计时 | 旧负例继续拒绝；有限差分/交错检查；生成排步与同步范围不退化；编译时间验收 |
| 2. 最小 DSL | typed 计算/存储/独立 Schedule；Vector 与 Cube 两个示例 | 不写 native 字符串、位置参数 tuple、event/credit；能生成可审查物理 IR 与 ASC |
| 3. 全部算子迁移 | examples 中的算法、oracle、AscendC/ACLNN runner | 包内无算子 dispatch；完整回归；代表程序 ASC 离线编译及必要设备正确性 |
| 4. 恢复性能工作 | 同契约 AscendC 与设备 profiling 的成套比较 | DSL 与编译耗时已可用，结果可溯源；本轮不执行此阶段 |

每阶段为可回退的小提交。保留原同步反例、机器契约和必要的独立最终验证，
重构时以当前保存点做差分参考，不能把本轮未验证排步当作设备正确性的基准。
