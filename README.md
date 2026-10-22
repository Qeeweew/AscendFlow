# AscendFlow

**用 Python 表达计算与流水，让编译器生成同步。**

AscendFlow 是基于 **xDSL**、面向 **Ascend 910B / dav-2201** 的算子编译器。
开发者明确计算、数据搬运、物理 buffer 和阶段排步，编译器从真实访问与存储生命周期
推导同步，生成 ASC，再由 CANN 编译为设备 kernel。

[快速上手](docs/guide/quickstart.md) · [DSL](docs/guide/dsl.md) ·
[算子示例](examples/README.md) · [编译器设计](docs/development/README.md)

![AscendFlow：从 Python DSL、真实访问与生命周期分析，到自动同步和 Ascend 910B 执行；下方从 attention 的 stage 偏移、buffer 双槽与访问依赖推导填充、稳态发射和排空](docs/assets/ascendflow-design.svg)

## 设计理念

- **作者掌握算法与排步**：用 Python DSL 表达分块、搬运、buffer 和 Cube/Vector 阶段。
- **编译器推导同步**：从真实读写、逻辑代次和最后 reader 推导 ready/release，管理有限事件资源。
- **优化遵守硬件与精度合同**：检查 UB/L1/L0 容量与数据生命周期，生成可检查的 IR、ASC 和设备 kernel。

## LLM 算子性能

**Ascend 910B3（20 AIC / 40 AIV）· CANN 9.1.0 · 2026-10-22**。
参照为 vllm-ascend `2581a69`；加速比 = 参照耗时 / AscendFlow 耗时。

### DSv4.1 Sparse Attention

C2（压缩倍率2）、H64、KV-head1、D512、BF16、window128、TopK512。参照为官方 AscendC。
Prefill 使用 B1；decode 为 **DSpark target 因果验证，每请求 6 query（5 draft + 1 decode）**，
上下文128K。1K=1024 tokens。

| 配置 | AscendFlow (ms) | TFLOPS | 官方 AscendC (ms) | TFLOPS | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill · Q8K / 上下文32K | 6.847 | 100.4 | 7.853 | 87.5 | **1.15×** |
| Prefill · Q32K / 上下文32K | 27.225 | 99.7 | 31.399 | 86.4 | **1.15×** |
| Decode · B64 / Q384 | 0.410 | 78.5 | 0.440 | 73.3 | **1.07×** |
| Decode · B128 / Q768 | 0.810 | 79.5 | 0.830 | 77.6 | **1.02×** |

TFLOPS 按有效 QK+PV FLOPs / 完整 kernel 耗时计算，FMA=2；耗时包含 gather、softmax 和写回。
计算范围与精度见 [Sparse MLA](docs/operators/sparse-mla.md)。

### KDA

H=HV=32、K=V=128，FP16 Q/K/V 与输出、FP32 carry。参照为 vllm-ascend Triton。

| 配置 | AscendFlow (ms) | Triton (ms) | 加速比 |
| --- | ---: | ---: | ---: |
| Decode · 128 条序列 × 1 token | 0.427 | 2.093 | **4.90×** |
| Recurrent · 4 条序列 × 32 tokens | 0.484 | 1.269 | **2.62×** |
| Prefill · 1 条序列 × 4096 tokens，chunk64 | 3.734 | 10.540 | **2.82×** |

实现与使用说明见 [KDA 示例](examples/kda/README.md)。

**测量方法**：双方同卡同输入，用 msprof 统计完整算子的设备 kernel 耗时，重复采样取中位数。
Sparse 每次调用前驱逐 L2，KDA 使用 kernel replay；不计 host 提交和输入准备，均通过独立数值验证。
复测入口见 [测试与 profiling](docs/guide/testing.md)。

## 一段 DSL，一条流水

用整块 **non-causal attention** 说明 stage、buffer 与稳态排步：
`O = softmax(Q @ K.T / sqrt(128)) @ V`。
固定 **head dimension=128、K/V 长度=128、query tile=64**，query 长度为 64 的正整数倍。
Q/K/V、P 与输出为 FP16，分数和 softmax 统计为 FP32。

数学阶段分别写成 `@N.inline` helper，例如 QK 的搬运、矩阵乘和发布：

```python
@N.inline
def qk(query, key, scores, buffers):
    q1, q0, score0 = buffers
    N.copy(query, q1)
    N.copy(q1, q0)
    N.matmul(q0, key, accum=score0, initialize=True)
    N.copy(score0, scores)
```

`softmax`、`pv`、`store` 同样封装各自的计算。它们在编译时展开，实际读写保持可见。
每个 head 使用 1 AIC + 2 AIV，两个 AIV 各处理 32 行；K/V 预载后常驻 L0。
`Score`、`P`、`PV` 各有两个 GM 槽，`*_buffers` 是各 helper 的 L1/L0/UB 缓冲区。
完整声明见[示例源码](examples/flash_attention/simple.py)，主体只表达四个阶段的排步：

```python
for tile in N.Dataflow(S // 64):
    q = tile * 64
    slot = tile % 2

    with N.Cube(), N.Stage(0, name="qk"):
        qk(Q[head, q : q + 64, :], k0, Score[head, slot, :, :], qk_buffers)

    with N.Vector() as lane, N.Stage(0, name="softmax"):
        r = lane * 32
        softmax(
            Score[head, slot, r : r + 32, :],
            P[head, slot, r : r + 32, :],
            softmax_buffers,
        )

    with N.Cube(), N.Stage(1, name="pv"):
        pv(P[head, slot, :, :], v0, PV[head, slot, :, :], pv_buffers)

    with N.Vector() as lane, N.Stage(1, name="store"):
        r = lane * 32
        store(PV[head, slot, r : r + 32, :], O[head, q + r : q + r + 32, :], store_buffers)
```

**Stage 决定发射哪一代，buffer 决定复用哪一块存储。** 对每个 actor 自己的本地步 `j`，
偏移为 `s` 的阶段执行逻辑 tile `t=j-s`，仅在 `0<=t<tiles` 时激活。
因此填充后，Cube 每步按顺序发射 `QK(j)`、`PV(j-1)`，Vector 发射
`softmax(j)`、`store(j-1)`；末尾再排空剩余的 PV 和写回。两侧本地步独立推进，
相同列号不表示同一物理时刻。

**同步决定这些发射何时能安全前进。** 每个 tile 使用 `slot=t%2`：生产者写完后，
消费者才可读取；同槽的 `t+2` 必须等待 `t` 的最后 reader 退休。
例如 Score 的两个 AIV 都完成 GM→UB 读取后，该槽即可交给下一代 QK，softmax 仍可继续。
Stage 偏移与两槽声明由作者选择，编译器从真实访问生成 ready/release 和首尾处理，
在实际交接和存储复用处生成等待；stage 数字不声明完成，也不分配 buffer。

上图依次展示这些声明、访问依赖，以及四个 tile 的填充、稳态发射和排空。
这是发射关系示意；实际重叠与吞吐取决于指令耗时和等待。
示例通过独立 FP32 数值检查；长上下文的 KV 分块与在线 softmax 见
[Flash Attention](examples/flash_attention/attention.py)。

## 开始使用

需要 **Python 3.11+、xDSL 0.73.0**。从仓库根目录执行，以下使用工作区独立环境：

```bash
python3 -m venv ../.venv  # 已有工作区环境可跳过
source ../.venv/bin/activate
python -m pip install -e '.[dev,validation]'
python -m ascendflow --input examples/flash_attention/simple.py --entry simple_attention --output build/quickstart/attention
```

这会生成 source IR、同步前后 IR 与 ASC。生成共享库时加载 CANN 环境并增加 `--shared`：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m ascendflow --input examples/flash_attention/simple.py --entry simple_attention --shared --output build/quickstart/attention-shared
```

构建命令不运行设备，每个算子构建的 Python→IR→ASC→CANN 共用 300 秒硬期限。
设备运行前先检查全卡状态并显式映射空闲物理卡，完整步骤见
[快速上手](docs/guide/quickstart.md)。编译无需构建 MLIR、TVM 或 TileLang；
设备数值参考使用工作区安装的 PyTorch / torch_npu。

## 开发与导航

| 目录 | 内容 |
| --- | --- |
| [src/](src/) | `ascendflow` 包：DSL、xDSL IR、分析、自动同步与 ASC/CANN 后端 |
| [examples/](examples/README.md) | 算法、分块、缓存与排步政策；另含 GEMM、Flash Attention、归一化等示例 |
| [benchmarks/](benchmarks/README.md) | 编译、设备数值检查、性能采集与参照对比 |
| [tests/](tests/README.md) | Python/IR 回归、硬件反例与独立有限执行模型 |
| [docs/](docs/README.md) | 当前设计、使用指南、算子合同与支持边界 |

```bash
python -m benchmarks.check_tests --jobs 32
python -m benchmarks.check_examples --list
```

自动同步规则与贡献入口见 [编译器开发](docs/development/README.md)。
安装包和命令名均为 `ascendflow`，源码直接位于 `src/`。
本机构建与测量产物默认写入 Git 忽略的 `build/`；参考源码仅用于研究和性能对照。
