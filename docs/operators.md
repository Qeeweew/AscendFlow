# 固定 shape 算子入口

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

这些 Python 工厂生成真实的 `npu.buffer/view/task/stream`，并通过产品的共享
AutoSync、最终 verifier 和 ASC 后端。工厂中没有 event ID；ASC 后端也不增加
未进入效果分析的搬运或 scratch。`from xdsl_npu import rms_norm, rope, softmax,
swiglu, batched_matmul_nt, gemm_nt, gemm_bias_silu, recurrent_gdn` 是可安装包的公开入口。

| 入口 | 参数/输出 | 首轮 shape 契约 |
|---|---|---|
| `rms_norm(R,D)` | f32 X、Gamma → f32 Y | D 为64的倍数，64…4096 |
| `rope(R,D)` | f32 X、Cos、Sin → f32 Y | rotate-half；D 为16的倍数，16…4096；Cos/Sin 为 R×D/2 |
| `softmax(R,D)` | f32 X → f32 Y | 最后一维；D 为64的倍数，64…4096 |
| `swiglu(R,D)` | f32 Gate、Up → f32 Y | 两个独立输入；D 为8的倍数，8…4096 |
| `batched_matmul_nt(B,M,N,K)` | f16 A[B,M,K]、Bt[B,N,K] → f32 C[B,M,N] | 每个矩阵一个 AIC；M/N/K-block 为16的倍数，16…256；K 为 K-block 的正整数倍 |
| `gemm_nt(M,N,K)` | f16 A[M,K]、Bt[N,K] → f32 C[M,N] | 二维输出 grid；静态 M/N/K 整除选定 tile；独立 L1/L0 K 与槽；可连续输出/驻留 B |
| `gemm_bias_silu(M,N,K)` | f16 A/Bt、f32 bias → f32 SiLU(A@Bt.T+bias) | 静态1 AIC:2 AIV；显式 GM/私有 UB 环；支持范围见 [混合核](mixed-ownership.md) |
| `recurrent_gdn(sequences,steps)` | f32 Q/K/V、g/beta、初始 state → attention 与逐 token snapshot | 静态序列、Dk64、Dv8…128；resident carry；完整契约见 [GDN](recurrent-gdn.md) |

Vector 行数静态非负，须整除 `blocks`，每核行数须整除固定 `row_block`。
未指定行粒度时，在8/4/2/1中选能容纳完整分配的最大合法值；这是明确的默认
资源政策，没有设备调参。指定不合法粒度/槽数/容量会给出诊断，不悄悄缩小。
本地 UB 为192 KiB；L1/L0A/L0B/L0C 另受实际容量检查。

所有 GM 参数分别分配、至少32字节对齐，且互不别名；执行期间输入由调用者
保持稳定。独立核 GM 写区间由 verifier 检查。输入不支持 NaN/Inf；数值测试
没有证明完整 f32 范围，包括所有 subnormal、溢出及特殊舍入行为。有限静态
task instance 总数每 actor 最多4096，超出拒绝；不会执行未经检查的计划。

## 浮点边界与 buffer 生命周期

RMSNorm 用 `Mul(X,X) → Sum64 partial → row Sum → mean → epsilon → Sqrt → Div(1,root)`，
再按行捕获 inverse、`Mul(X,inverse) → Mul(Gamma)`。每个算术节点有独立 f32
边界；不会融合成改变累计顺序的表达式。原生归约的硬件树仍由其指令定义，
CPU oracle 是容差检查，不冒充位级模拟。epsilon 必须在 f32 舍入后为正且有限。
首次近似 `vrsqrt` 未通过原有容差，因此改用安装接口支持的 Sqrt/Div 路径；
源码依据是 ops-nn 910B 的 `rms_norm_merge_n.h`，没有放宽容差。

Gamma 每核只载入一次；input/work 各一个多槽分配，work 先放 squares、再放
normalized output。partials/statistics 单独复用。每行 Scalar load 同步返回，
Vector 按值捕获 inverse；统计槽可在 Vector 执行结束前退休。input 的最后
reader 是 normalize，后续 Gamma 只读取 work。上一代 Store 排在当前 Scalar
wait 之前，保持已经就绪的 Store 发射。

Softmax 使用 `Max → Sub(max) → Exp → Sum → reciprocal → Mul`，保留 reciprocal
与最后 Mul 的 f32 边界。Max/Sum 复用相同 partials/statistics。输入最后读在
Sub；后面的 Exp、Sum、normalize 都不再访问输入。当前排步中的必要 V→S
等待已门控下一步 DMA 发射，verifier 因而可以省去重复的 input release event。
这是已知 Scalar Wait 契约产生的完成路径，没有把 SSA use-def 当作完成。

SwiGLU 为 `round(round(G / round(1 + exp(-G))) * U)`。denominator 与输出共享
work；Gate 在 Div 后归还，Up 在 Mul 后归还。默认公式不改成另一种 Sigmoid
近似或舍入次序。输入幅度过大、指数上溢等数值情形不在本轮验证证据内。

Matmul 直接用原生 ND2NZ 把 strided GM K slice 放入 L1；A 的 Load2D 按完整
fractal 重排到 ZZ，Bt 按 NT 契约进入 L0B。源/目的字节并集包括精确的 GM
行片段，不扩大成整段矩形包围盒。一个 L0C carry 经首次 clear、后续 RW MAD，
最后一次 Fixpipe 导出。没有每个 K slice 各自清零，也没有先写 GM 再累加。
L1 在 Load2D 最后读取后归还，L0A/B 在 MAD 后归还；L0C 最后由 FIX 读取。

## RoPE 的最小 WAR 反例与生成结果

`rope(3,64,row_block=1,slots=2)` 的 Cos/Sin 共享真实 `trig_slots` 分配：

```text
cos_low  [V]    R X.low, Trig ; W Y.low
cos_high [V]    R X.high,Trig ; W Y.high
load_sin [MTE2] W Trig
sin_high [V]    R X.high,Trig ; W Temp
```

如果只等待 Cos DMA ready，`load_sin` 可以在两个 Cos reader 退休前覆写 Trig。
两个 tensor 名称不解除别名。这是同一 generation 的 WAR，不是循环计数错误。
未同步 xDSL IR 只有这些 task；同步后的实际 IR 增加通道与以下 task-local 节点
（省略未改变的 task/view 和其他通道）：

```mlir
"npu.sync"(%generation) {action = "set", channel = "c1_cos_high_load_sin_d0", pipe = "V", lower = #builtin.int<0>, upper = #builtin.int<3>, distance = #builtin.int<0>} : (index) -> ()
"npu.sync"(%generation) {action = "wait", channel = "c1_cos_high_load_sin_d0", pipe = "MTE2", lower = #builtin.int<0>, upper = #builtin.int<3>, distance = #builtin.int<0>} : (index) -> ()
```

该通道的实际 reason 为
`WAR:trig_slots[0:128]:read_retired; WAR:trig_slots[128:256]:read_retired`。
对每个 active generation，生成的 ASC 序列为：

```cpp
// Cos 的两个 Vector reader 均已发射。
asc_sync_notify(PIPE_V, PIPE_MTE2, 0);
asc_sync_wait(PIPE_V, PIPE_MTE2, 0);
asc_copy_gm2ub(/* Sin → 同一个 Trig slot */);
asc_sync_notify(PIPE_MTE2, PIPE_V, 2);
asc_sync_wait(PIPE_MTE2, PIPE_V, 2);
// Sin reader 随后发射。
```

第一个 Set 的生效等待此前两次 V 读取退休，Wait 只门控 MTE2。第二对保证 Sin
写完后才可读。Set 不阻止后续 V 指令，相关 Vector RAW 另保留必要 `PIPE_V`
barrier。后续 `sin_low` 是 X 的最后 reader，其 release 放在 `rotate_high` 前，
后者只访问 Y/Temp。内核唯一 `PIPE_ALL` 在出口，是 lifetime drain。

回归删除这对 WAR Set/Wait 后，最终 verifier 报 `missing WAR`，独立交错检查
找到覆盖冲突；两者均没有使用出口 drain 修复内部错误。完整插入前/后 IR 与
witness 可复现为：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.legacy.build rope --rows 3 --columns 64 --row-block 1 --output runs/operators/rope-war
```

`before.xdsl`、`after.xdsl`、`witnesses.json` 和生成 ASC 保存在该外层目录。

## 明确的后续边界

这些工厂不支持动态 task count、带洞/数据相关 guard、间接 gather/scatter、
动态 group descriptors、可变跨核 ownership、一般跨组 mixed 协议、全局归约 phase、
稀疏/paged attention、GDN 间接状态和通信。当前未提供假空接口声称支持它们。
普通/连续 GEMM 的已实现政策见 [分层 GEMM](hierarchical-gemm.md)。融合
FFN/attention、NN 输入格式、BF16 与一般尾块尚未完成。
扩展时需新增经审查的物理访问、owner/activation 与完成契约，并进入同一个
最终 verifier；不能在 ASC 文本层隐藏工作，或凭事件数字绕过检查。
