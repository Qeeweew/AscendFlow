# 复杂算子所需的物理执行扩展

当前新增内容均进入同一 AutoSync 和最终 verifier，不绕过产品同步契约。

`ViewOp` 可描述逐行 footprint：每行有效字节数、行数和 row stride。依赖和状态机
使用实际区间的并集，父 view 的空洞不授权子 view 访问；DMA/count-mode recipe
要求连续，逐行原生 recipe 则生成明确的每行指令。局部 bank 仍是 core 私有。

`TaskOp` 的静态 `[lower,upper)` 激活区间支持 once cache 与连续切片。
实际 task instance 决定 generation 配对，未激活的 source 不会凭空产生通知。
这是静态受限控制，不支持数据相关 guard 或任意动态 task stream。

issue rank 与语义 source order 独立：依赖按逻辑代次和 source order 建立，发射
按 physical step、固定 issue rank、原始序号排序。可把上一代 Store 排到当前
Scalar wait 之前，保留已就绪输出的独立发射；非法计划仍拒绝，不改专家顺序。

Scalar f32 UB load 有真实四字节读取效果，其 SSA 值与普通异步 task 身份不同。
读取同步返回，后续 Vector 指令按值取得参数；ASC 使用 volatile UB load 保持
访问位置。当前只允许 provider/consumer 在同一 physical step 且 provider 提前
发射，跨 stage 的延迟 descriptor 需要另行建模，不能偷用上一代变量。

Scalar Wait 在模型中阻止全核的后续 dispatch。已经发射的 native task 和异步
Set 生效仍能继续。图与独立状态机都检查这一规则，不能把 Scalar Wait 当作
普通 destination-pipe wait，也不能把所有 SSA use-def 当成异步完成。

独立 grid 使用实际 block index 为 GM view 加 core stride。每个 actor 的局部
容量和 event bank 独立，最终 verifier 另检查不同 actor 的 GM W/R 和 W/W 区间。
共享只读参数合法；需要交接或全局 phase 的重叠会拒绝。没有宣称实现 AIC/AIV
mixed handoff。有限图/状态机验证一个 actor，静态 disjointness 检查组合条件。

2026-10-13：85 项回归通过，包括上述扩展的正例与删除 Scalar ready、非法
descriptor 延迟、strided 空洞读取、跨 owner 写重叠等负例。
Scalar→Vector 双缓冲、count=5、四核 grid 已通过 CANN 编译和物理卡1（逻辑0）
执行；160个 f32 输出与 oracle 精确相同。此例仍是基础语义验证，不代表完整
normalization/RoPE/GEMM 等复杂算子已经交付。

```bash
.venv/bin/python compiler/examples/build_legacy.py scalar --count 5 --blocks 4 --output runs/stage2/scalar-grid --compile --shared
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/stage2/scalar-grid --physical 1
```

产物、健康快照、编译选项和数值结果保存在外层 runs/stage2，未提交产品 Git。

上述85项记录属于当时的阶段。后续实际扩展加入了静态周期 task、常量整数
divisor/modulo 和二维核坐标，已用于分层/连续 GEMM；不等同于支持数据相关
动态 stream。短 Cube 的旧 M completion 假设也已由设备反例修正，详情见
[分层 GEMM](hierarchical-gemm.md)。
