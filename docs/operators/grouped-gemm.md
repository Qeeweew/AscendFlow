# Grouped GEMM

入口为 [DSL](../../examples/moe/grouped_gemm.py)，Tensor 封装为
[GroupedGemm](../../examples/moe/torch.py)。使用普通 AscendFlow 编译链，目标 910B / dav-2201。

## 调用合同

```python
op = GroupedGemm(runtime, "build/gmm/grouped_gemm.so")
y = op(x, weight, group_list)
op(x, weight, group_list, out=y)
```

| 参数 | shape / 存储 | 语义 |
|---|---|---|
| `x` | 连续 ND `[C,K]`，FP16/BF16 | 已按本地专家排好的输入 |
| `weight` | `[E,K,N]`，编译时固定 ND 或 FRACTAL_NZ | 本 rank 的本地权重，与 x 同 dtype |
| `group_list` | 连续 ND device int64 `[E]` | **固定 type=0，累积 exclusive 结束位置，无前导零** |
| `out` | 连续 ND `[C,N]`，与 x 同 dtype | 可选，必须与所有输入不重叠 |

要求 `0 <= group_list[0] <= ... <= group_list[E-1] <= C`，`E,K,N > 0`。
专家 e 的区间是 `[0,group_list[0])` 或 `[group_list[e-1],group_list[e])`。
kernel 直接读取原始结束位置；专家循环保留上一次结束值作为下一次起点，
零尾部直接读取最后一个原始结束位置。空专家不计算；不做 cumsum，不生成 offsets，不接收 counts。
没有 bias、路由权重、激活或通信参数。输入在执行期间只读。

每专家计算 `out[begin:end] = x[begin:end] @ weight[e]`。FP32 累加后在 FIXPIPE
舍入为输入 dtype；`out[group_list[-1]:C]` 为零。C 可以为零；同一产物支持改变
C 和 group_list 内容。调用不读回 metadata，不同步 host，使用 Torch 当前 NPU stream。
runtime 由调用方长期持有；存储和 stream 生命周期遵守 [runtime 合同](../guide/runtime.md)。

权重 layout 由 `WEIGHT_LAYOUT="linear"` 或 `"NZ"` 固定。NZ 的最后两轴必须是 16
的倍数，且 Tensor 已经为 format 29。权重加载时可在算子外执行
`torch_npu.npu_format_cast(weight,29)`。算子和封装都不转换权重或 group_list，
不回退其他实现。ND 支持 K/N 尾块。NZ 按真实列块、源 leading K 直接 GM→L1 DMA，
不使用 ND2NZ。

编译时固定 E/K/N、dtype、layout、BM/BN/BK/CK 和 worker 政策，C 是 runtime 参数。
分块须满足实际 L1/L0 容量及 native 限制。所有核按输出 tile 序号循环分工，
标量 tile 序号只用于分核，不改变 group_list。K 使用两槽流水，L0 槽随真实
panel/chunk 代次轮换。完整 K/N 面板维持跨输出 tile 的流水，K/N 尾块使用
精确搬运与输出几何。只保留单个 L0C 平面；不再提供静态专家分核、行列复用分块、
多 L0C 槽或逐 tile 关闭流水的实验开关。资源不满足时报错。
相同 kernel 随后处理输出零尾部，没有额外清零 kernel 或全局 barrier。

`WEIGHT_CACHE="normal"` 继承普通权重缓存，`"bypass"` 对全部权重读取绕过 L2；
`"auto"` 直接根据当前专家的 M tile 数选择：仅一个 M tile 时绕过权重 L2，
其余专家保留普通缓存。这与 CANN 的 `blockDimM == 1` 政策一致。选择使用真实
device metadata，既不读回 host，也不转换权重。全部绕过 L2 会严重拖慢大 M，
不能用小 batch 的结果选择全局 bypass。底层是通用的
`copy(...,cache_bypass=predicate)`，公开调用接口不增加参数。

`SWIZZLE=1` 保持行优先 tile 序号。大 M 可显式选择方形窗口，例如 `SWIZZLE=8`：
M tile 数大于 5 时使用对角分核，M/N 尾窗口按真实高度和宽度分配。
该政策只改变 tile 的访问顺序，适用于
全核分工和跨 tile 流水；不改变 metadata、ownership 或同步合同。

默认启用 L0C 细粒度 MAD/FIX unitflag 握手。N 是 BN 的倍数时，kernel 直接检查当前专家
行数是否是 BM 的倍数：完整专家使用中间 MAD=2、末轮 MAD=3、FIX=3，省掉
tile 内 M↔FIX 整块事件；其余专家及 N 尾块使用源码中的 runtime 行前缀路径。
不转换分组数据，不在 host 读回 metadata。编译器协议资格见
[unitflag](../development/unit-flags.md)，收益按实际产物与 shape 验收。

## EP、TP 和接入边界

三种模式使用同一接口，不传 rank 或通信组：

| 模式 | 本地输入/权重 | 输出 |
|---|---|---|
| EP | 本地专家子集；x 只含这些专家的 token；group_list 从隐含 0 开始 | 本地专家输出，顺序不变 |
| TP | 本 rank 的 intermediate 分片 | GMM2 返回 `[C,H]` 的局部贡献，由调用方归约 |
| EP+TP | 同时选择专家子集和 intermediate 分片 | 本地专家的 TP 局部贡献 |

两次调用分别是 GMM1：`[C,H] @ [E,H,2J] -> [C,2J]`，外部执行激活，
GMM2：`[C,J] @ [E,J,H] -> [C,H]`。GMM1 在激活前已舍入。
W13 列顺序是 `[gate,up]`。TP 区间 `[j0,j1)` 必须同时取 gate、up 的对应列，
拼为 `[gate_local,up_local]`；W2 取对应 K 区间。不能直接连续切开整个 2J 轴。

vllm-ascend 可在权重加载时准备 `[E,K,N]` 与 NZ 存储，初始化时加载特化产物，
在现有 `apply_gmm1/apply_gmm2` 的本地矩阵计算位置调用上述接口。
路由、重排、EP dispatch/combine、TP 归约仍由现有调用方完成，不做通信与计算融合。
调用方若提供其他分组格式，须在算子外准备 type=0 输入。
接口没有框架依赖；独立算子测试不安装或运行 vllm-ascend。

核对的 vllm-ascend 提交为 `2581a69bcc16ee6af56e72985a15799bb5f61808`。
未量化路径在 `ops/fused_moe/routed_experts.py` 的两个 hook 调用原始 grouped_matmul。
权重按 `need_trans` 得到逻辑 `[E,K,N]` 后，在加载阶段一次性 contiguous/打包并保存，
hook 中的矩阵计算替换为：

```python
# 初始化时建立 af_gmm1/af_gmm2，并保存准备好的本地权重 af_w13/af_w2。
# apply_gmm1 中：
assert mlp_compute_input.group_list_type == 0
gate_up_out = layer.af_gmm1(
    mlp_compute_input.hidden_states, layer.af_w13, mlp_compute_input.group_list)
# apply_gmm2 中，hidden_states 已由现有调用链完成激活和 topk_scales：
hidden_states = layer.af_gmm2(hidden_states, layer.af_w2, mlp_compute_input.group_list)
```

这是无 bias 的本地 b16 GMM 接口映射，权重分片沿用调用方已完成的 EP/TP 划分。
不在 hook 内临时转置、打包或转换 group_list。现有 activation、topk_scales 和通信
调用链继续使用其输出。本仓库提供可调用封装和该映射，未修改或部署参考框架。

## 验证与性能验收

```bash
../.venv/bin/python -m benchmarks.moe.run --build --output build/gmm \
  --experts 4 --k 2048 --n 4096 --layout NZ --rows 1
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 \
  ../.venv/bin/python -m benchmarks.moe.run --artifact build/gmm --output build/gmm/decode \
  --rows 1 --profile --warmup 10 --repeats 30 --trials 3 --require-performance
```

构建入口只接收 shape、dtype 和 layout 等输入信息。按构建时的输入容量选择
BM16（容量不超过 16E）或 BM128，对应 SWIZZLE1/8；统一使用 BN256/BK256/CK64、
auto 权重缓存和 unitflag，不提供分块、缓存或 unitflag 的命令行调参开关。
prefill 产物可用 `--rows 128` 构建。同一产物仍支持 runtime C；封装加载一个
预编译产物，不在调用时自动切换产物。

运行前检查全部卡并选择空闲健康卡。`--ends` 接受原始累积结束位置，例如
`--ends 0,0,0,0 1,1,19,20`，`--padding` 增加输出容量。
有限模型检查实际访问、初始化、同步和输出覆盖；设备检查 CPU FP64 参考、最终舍入、
guard，以及原始 `torch_npu.npu_grouped_matmul`（split_item=2、group_type=0、group_list_type=0）。

性能验收允许生成版本每轮中位数不超过原始版本的 1.01 倍。
`--require-performance` 对三轮及以上采样强制该条件，不以跨轮平均掩盖超限。
性能采用原始 `task_time.csv` 的设备 task duration，包含每次调用的全部任务，
不以 stream interval 代替。预热后交替后端顺序，保存 trial、产物身份和原始 CSV。
权重准备与输入创建不计入 kernel 时间。原始 Torch GMM 自行分配输出，生成版本
使用固定 out，因此输出地址不同；两者输入与权重地址相同，复用且不刷新缓存。

已实现 ND/NZ、FP16/BF16、runtime 行前缀及独立 EP/TP/EP+TP 组合检查。
简化版本采用全核分工、单个 L0C 平面和两槽 K 流水，移除了静态专家分核、
行列复用及多累加器的实验开关。全量 CPU 回归 2962 项通过；最终源码的
41 项算子模型及文档/入口检查共 44 项也已通过。设备结果以当前源码对应的产物为准。

当前性能验收采用 1% 容差，覆盖 decode 以及每专家 128/512/2048 行的
BF16 GMM1、FP16 GMM2 和 E8 较大权重。分块是编译政策，不自动读取或转换
分组数据：decode 使用 BM16、SWIZZLE1，后三档使用 BM128、SWIZZLE8。
两者均使用 BN256/BK256/CK64、auto 权重缓存和相同的 unitflag 资格规则。

以下为三轮中位数的中位数，单位 μs；验收实际逐轮检查。后三档全部满足
1% 容差，最慢的单轮相对差距为 E8 大档的 0.831%。

| dtype / E / K / N | 每专家 128 行：生成 / 原始 | 512 行 | 2048 行 |
|---|---:|---:|---:|
| BF16 / 4 / 2048 / 4096 | 42.55 / 42.72 | 127.04 / 127.13 | 488.78 / 492.32 |
| FP16 / 4 / 2048 / 2048 | 23.06 / 25.13 | 70.92 / 70.93 | 247.74 / 247.91 |
| BF16 / 8 / 2048 / 7168 | 131.49 / 132.62 | 527.73 / 530.79 | 1793.08 / 1778.62 |

对应 `stable-{gmm1-bf16,gmm2-f16,e8-bf16}` 产物。E8 的 ASC、device binary、
共享库与 `carry-cache-window-e8` 逐字节相同，因此复用其三轮设备证据，
在新产物的 `acceptance.json` 中记录同一性与 1% 容差复核。

`unitflag-ab/m16-n256-on` 的均匀每专家 1 行、16 行及单活跃专家 1 行分别为
28.83/30.42、28.88/29.22、8.66/11.65 μs，三轮均通过。此分块与上述大档
分开编译，调用方按容量选择已准备的产物，不在算子内读取 host 分组信息。
当前源码的 EP、TP、EP+TP 组合检查在 `final-parallel-policy/validation/device.json`。

小 batch 的固定地址对照使用 E4/K2048/N4096、每专家 16 行，三轮各 50 次；
四个 kernel 共用同一份 x、weight、group_list 和 out：

| BM/BN/BK/CK | unitflag 关闭 / μs | 开启 / μs |
|---|---:|---:|
| 16/128/128/128 | 30.74 | 30.65 |
| 16/256/256/64 | 28.82 | 28.79 |

开关本身基本持平。此前将两种分块间的回退归因于 unitflag 是错误的。
`msprof op` 确认两种分块的 Cube 活跃时间均约 6.087 μs；BN256/BK256
使每核 MTE2 指令从 206.8 降为 53.2，整卡 GM→L1 请求从 72 MiB 降为 68 MiB
（权重仍为 64 MiB，A 从 8 MiB 降为 4 MiB）。差异在搬运量和指令数，
不能通过关闭 unitflag 解决，也不将 GM→L1 请求量当成 HBM 实际流量。
固定地址对照及四份串行 PMU 证据位于 `build/grouped-gemm/unitflag-ab/`。

大 M 的同输入 `msprof op` 对照确认，E4 大档开启 unitflag 后 Cube 活跃率由
91.08% 升到 95.37%，原始为 95.34%，计算与搬运量不变。PMU 采集不替代
三轮交替 task duration 验收。原始 CSV、源码/IR/ASC/库身份、编译选项与设备
记录保留在 `build/grouped-gemm/`；未执行 vllm-ascend 端到端测试。
