# Load3D 与 FP32 Cube 寄存器状态

寄存器状态属于实际 actor，不是 tensor 存储，也没有字节地址。`NativeSpec.registers`
记录 setter 的写入值和 reader 所需值；`RegisterCell(register, actor)` 将这些 effects
交给共同 access-order、通知、生命周期和事件分配链。当前这些状态只属于 AIC。

| 状态 | 实际 setter | reader |
|---|---|---|
| FMATRIX_A/B | `set_fmatrix_a/b`，宽度低 16 位、高度为 1 | 对应 Load3D A/B，宽度取实际 L1 allocation 的 leading rows |
| L3D_RPT | `set_load3d` 包中的 repeat 配置 `1<<16` | Load3D A/B |
| L3D_PADDING、L3D_SIZE | 同一包中的零 padding/size 配置 | Load3D A/B |
| FP32_MODE | `set_fp32`，IEEE 模式 | FP32 输入的 MAD |

这些 setter 都是具有真实 effects 的 `TaskOp`。后端只输出 setter；必要的 S→MTE1、
S→M ready 和 MTE1→S 旧读取退休由共同同步生成器确定，使用普通事件资源台账。
S→S 保持 Scalar 的实际有序发射，不产生 `PIPE_S` barrier。

## 定义与放置

Load3D recipe 不要求全 kernel 只有一个宽度。某个 FMATRIX 只有一种所需值时，
lowering 在入口建立该值；存在多个所需值时，在各实际 reader 前设置对应 FMATRIX。
RPT/padding/size 和 IEEE 模式保留各自的状态身份，不能因共同初始化就合成一个配置资源。
入口 setter 位于实际 group、runtime 参数和纯标量前缀之后，生成的激活条件才能使用这些 SSA。

`analysis/registers.py` 为每个源 block 建立一次输入值要求和有限的输出值集合。
顺序组合替换输入，分支合流保留所有可能值，可能零次的循环保留进入状态；
可重复 body 必须保持自身的输入要求。它不按 runtime 次数展开，也不重复遍历
child 来求整图固定点。读取没有匹配的支配配置或遇到已知错误值时明确拒绝。

这份数值定义事实不证明配置已经对 reader 可见。共同依赖分析分别保留：

- setter 写入后，reader 才能读取配置的 RAW；
- 旧 Load3D reader 退休后，新的 setter 才能覆盖配置的 WAR；
- setter 之间必要的 WAW。

旧 reader 使用实际 MTE1 指令退休；不猜测配置在发射时已锁存。状态没有进入
tensor alias、私有 packet 重排或物理 cell credit；不以存储释放替代配置退休。
Fixpipe 的现有 invariant 配置 packet 保持其独立合同。

## 验证与支持边界

`tests/sync/test_register_lifetimes.py` 使用真实 DSL 检查不同宽度顺序组合、分支与循环
重入，核对错误/缺失配置，并删除真实退休握手验证寄存器 WAR 反例。
独立终端模型同时检查实际寄存器值和 RAW/WAR/WAW，不以 tensor 字节别名模拟 SPR。
ASC/CANN 和设备数值分别提供对应产物的资格。

当前状态值为静态 native recipe 要求的常量，不提供任意动态 SPR 写入或外部 clobber。
条件相关性无法由有限值合流保留时不会据此获得新的配置资格。
