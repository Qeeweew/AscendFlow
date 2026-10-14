# DSL 的 multi-stage 源程序与完成语义

2026-10-14，按用户要求修正重构方向。以下语法是正在实现的目标，不是当前支持
声明。参考为只读 TileLang snapshot `6ba65d204c937fcb1c43b1a22b32979506595655`，
尤其 `examples/ascend/flash_attention/persistent_910b.py:224–371`。借鉴其作用域
和任务位移语义，前端和 lowering 使用 xDSL，不依赖 TileLang/TVM 构建。

## 同一流水模型覆盖两种并行

核内 GM→UB→Vector→GM 可以表达成 load/compute/store 三阶段，也可以在同一
阶段中暴露搬运和计算。数据依赖由逻辑值表达；DMA ready、最后读取、slot 复用
和实际完成仍由 native effects 决定。

CV 多阶段必须直接出现在源程序结构中。Attention 的示意如下，省略算法宏的
具体数学和 buffer 声明；这些宏属于 examples，并必须展开为普通 typed 操作：

```python
with N.Kernel(groups, kind="mixed"):
    # Score/P/PV 是 group-private task ring，各有明确 versions。
    # Q cache 和 query 的 m/l/O state 有独立生命周期。
    for t in N.Pipelined(tasks, stages=3, order="source"):
        query, kv = decode_task(t)
        with N.Stage(0, name="qk"), N.Cube():
            qk_and_export(query, kv, Score[t])
        with N.Stage(1, name="softmax"), N.Vector():
            online_softmax(query, kv, Score[t], P[t])
        with N.Stage(1, name="pv"), N.Cube():
            pv_and_export(kv, P[t], PV[t])
        with N.Stage(2, name="update_o"), N.Vector():
            update_output(query, kv, PV[t])
```

源码中的 `t` 表示同一逻辑任务，编译器将每个 stage 的访问和 descriptor 一起
延后。稳定段的派发意图为：

| 物理 outer step j | AIC | AIV0/1 |
|---|---|---|
| Stage0 | QK[j]，发布完整 Score[j] | |
| Stage1 | 导入 P[j−1]，PV[j−1] | Softmax[j−1]，发布 P[j−1] |
| Stage2 | | 导入 PV[j−2]，UpdateO[j−2] |

这是排步意图，不能作为异步完成保证。AIC 必须先发射新的 QK，才进入旧 P 的
controller Wait；否则会阻止 QK 与 Vector 的重叠。AIV 也必须把已有输出工作
放在可能阻塞的新输入 gate 的合适位置。固定 issue 导致环时拒绝编译，不能
通过随意改 stage、补全局 barrier 或扩 buffer 静默修复。

软件 stage 不等于硬件 pipe：一次 Cube region 可包含 MTE2、MTE1、Cube 与 FIX，
Vector region 可包含 MTE2、Vector、MTE3 和 Scalar。region 内仍有独立的 ready、
read-retired 和 buffer reuse；给 region 一个 stage 不把它视作整体完成。

## 必须分开表示的数量

| 数量 | 语义 |
|---|---|
| `Stage(k)` | 本 pipeline 的 logical task 延迟/派发阶段 |
| `Pipelined(..., stages=S)` | 本 pipeline 的执行 stage horizon；不是 buffer 数 |
| `versions=B` | 某个 task-ring 的物理存储版本预算 |
| query/state slots | 算法状态的 owner 生命周期，可能跨多个 KV task |
| native segment/grain | 核内 Cube/Vector 的真实工作划分，独立于 CV 交互块 |
| event ring | 编译器根据消费及复用路径证明的硬件通知资源 |

不照搬 TileLang 将 `num_stages` 同时用作版本上限的语义。普通和专家代码使用
同一 region IR：专家固定 stage/order/versions，后续自动排步在同一表示上选择
合法配置。初版只实现明确固定排步；未实现的自动排步模式必须给出诊断。

嵌套核内 `Pipelined` 有自己的迭代域：例如外层一个 BK1024 的 CV task，内部
Cube 执行八个 N128/K128 segment，Vector 执行多组8行。外层 handoff 要覆盖
全部真实贡献；只发布第一个 segment 或先归还尚有 reader 的 slot 都不合法。
这些 native 操作必须按 region 层次降低，不能用一个全局扁平微步时钟改变
“完整下一块 QK 在旧 PV Wait 之前发射”的源意图。有限静态内循环可先展开为
同一 outer task 的 typed native 操作；较大或动态嵌套不能偷偷省略检查。

## 向 xDSL 与物理执行 IR 降低

源 IR 显式保留 pipeline、stage、执行域、typed 操作、task descriptor 和 Buffer/View。
逻辑依赖顺序沿源 task/region 建立；实际 issue 从 pipeline/stage/order 降低。
不同 region 可处于同一 Stage，不能靠 stage 数字排序改变数值定义顺序。

Lowering 在 GM/片上布局、实际 native footprint、owner、task generation、SPR
effects 和指令完成点确定后，再生成同步 PlanIR。SSA use-def 只提供逻辑依赖，
不提供 DMA/MAD/FIX 的完成边。外部 disjoint 契约、两 AIV 的真实 slice/quorum、
Counter pairing/backlog 和 terminal drain 沿用同一最终 verifier。

第一次实现保留静态范围，不用未实现的 persistent/动态接口冒充支持。连续
query 流必须表达 state reset 与末次读退休；first/last-KV predicate 随所属 task
移动，不能按物理 step 重新计算。Scalar 从内存读出的动态 descriptor 不能当作
可自由重算的纯表达式；需要明确捕获/存储和 ownership 契约。

## 实现与验收

1. 局部三阶段 Vector 和 Cube/Fixpipe 使用普通 typed copy/math/matmul。
2. 双向 CV：Stage0/Cube、Stage1/Vector、Stage1/Cube、Stage2/Vector，共同 task
   clock，两个 AIV 真实参与；生成完整可解析的 stage→physical issue 映射。
3. 零/一/两次迭代、填排、中间段、收尾、task-ring 复用和跨 query state 边界。
4. 至少一个多 segment CV task 与独立核内预取；对早 ready、早 credit、删除
   channel、错误 descriptor 代次和不可行固定 gate 顺序给出反例。
5. 最终 ASC 检查完整 QK 发射相对 P Wait 的位置，并离线编译、设备验证。
   pipeline 声明或 verifier 通过不能代替这些证据。

编译性能改造继续执行，但要对这个 stage/region 模型建立固定 benchmark；不能
通过将 multi-stage 全部串行化来降低编译成本，也不先恢复算子调参。

## 本轮已经接通的路径

源语法和 region IR 现已实现于 `language.py` / `source_ir.py`；
`lowering.py` 降到现有物理执行 IR，再进入统一 AutoSync。可运行算法示例在
`examples/multistage.py`，没有放入编译器内部。这些是同步示例，不是 FA。

- `vector_source`：双缓冲 load/compute/store；零次循环保持空派发和最终 drain。
- `cv_source`：QK、Vector 舍入、PV、输出；两个 Stage1 保持独立逻辑 phase。
- `segmented_cv_source`：每个 CV task 有三个 QK/P 贡献；内层 K、P/V 双槽预取，
  PV 仅首 chunk 初始化，其余按源顺序 FP32 累计。外层仍是三个 stage。
- Stage 的返回 tile 与 typed view、数值 producer 分开。借用 tile 被原 buffer
  覆盖后再读会拒绝；不会增加隐藏复制。每个输入单独检查，允许同 root 的
  不相交子视图分别来自不同 writer。
- 从内层导出的最后一个结果保留 producer descriptor 和 slot；不能按 consumer
  的外层任务重新选择版本。零次内层循环不能导出有效 tile。
- 逻辑轴 bounds、yield/result 类型和执行域归属都有负例；不忽略未知 effects。

2026-10-14 的分段 CV 验证：2 groups、4 tasks、3 segments、f16 输入、f32 输出；
CANN 9.1 / dav-2201 离线和 shared 编译成功。全部卡健康且无进程后，物理卡1
映射逻辑0执行10次，独立 CPU 计算最大绝对误差为0。数值契约是
`sum_chunk RN_f16(2 * QK_chunk^T) @ V_chunk`，没有 softmax。
源码、IR、ASC、选项、设备和输入哈希在产品 Git 外的
`/workspace/xdsl-910b/runs/refactor/segmented-cv/`。

最终 ASC 中全部新 QK segment 在旧 `probabilities_ready` Wait 前发射；
内部预取先于上一个 segment 的 Cube 指令。同步验证另检查每个 QK export 都先于
Score-ready，人工将 ready 移到第一个 segment 后会拒绝。设备正确性和这些顺序
检查不是吞吐率证据，本轮未进行算子性能调参。

当前边界：静态完整 native tile、一个连续外层流、静态内层展开、source order。
`initialize=(chunk == 0)` 支持静态内层；依赖运行时 outer task 的初始化仍明确
拒绝。动态任务/ownership、运行时条件 region、query-state clock 和自动排步还未
接通；现有 FA 示例还需要迁移到此源语法，不能把 CV 同步小例当作迁移完成。
