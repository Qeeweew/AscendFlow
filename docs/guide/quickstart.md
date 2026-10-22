# 快速上手

AscendFlow 编译单个 NPU 算子。用户写受限 Python DSL，明确计算、物理 buffer、
执行域和流水政策；编译器生成 ready/release、Set/Wait 及事件资源计划。
目标是 Ascend 910B / dav-2201，输出 ASC，由安装的 CANN 9.1.0 离线编译。

## 1. 环境

需要 Python 3.11+。以下命令从 `AscendFlow/` 仓库根目录执行。当前工作区已有
独立 `../.venv`；新环境可先执行 `python3 -m venv ../.venv`。

```bash
source ../.venv/bin/activate
python -m pip install -e '.[validation]'
python -m ascendflow --help
```

生成 IR/ASC 只需要 Python 和 xDSL；数值检查使用 NumPy。CANN 编译、运行与
profiling 使用本机 `/usr/local/Ascend/cann-9.1.0`，需要时加载环境：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
command -v msprof
command -v npu-smi
```

示例和 benchmark 从仓库根目录导入，开发安装的 `ascendflow` 指向本仓库 `src/`。
无需构建 TileLang、TVM 或 MLIR。

## 2. 编译第一个 kernel

[scale.py](../../examples/vector/scale.py) 对 FP32 输入逐元素乘 2。`B=64` 是编译时
 tile 容量，`tiles` 是运行时 tile 数，同一二进制可以接收不同 `tiles`。

```bash
python -m ascendflow --input examples/vector/scale.py --entry scale --parameters '{"B":64}' --output build/quickstart/scale --timeout 300
```

每个输出目录属于一个构建。成功时包含：

| 文件 | 用途 |
|---|---|
| `source.xdsl` | 捕获后的 source IR |
| `before.xdsl`、`after.xdsl` | 同步前快照、最终物理 IR |
| `scale.asc` | 生成的 ASC 与 C launcher |
| `verification.json`、`compilation.json` | 输入/生成报告与阶段计时 |
| `build-result.json`、`build.log` | 总构建状态、耗时与诊断 |

增加 `--compile` 编译设备对象与二进制；增加 `--shared` 同时生成可调用的共享库：

```bash
python -m ascendflow --input examples/vector/scale.py --entry scale --shared --output build/quickstart/scale-shared --timeout 300
```

额外产物为 `.o`、`.aibin`、`.so` 和 `scale.compile.json`；后者记录目标、CANN 命令、
选项及 ASC/库 hash。以上命令均不运行设备。每次构建共享一个 300 秒硬期限，
包含 Python 构造、分析、ASC 和 CANN；失败看 `build.log`，不要继续运行失败产物。

## 3. 在设备上检查结果

先执行 `npu-smi info`，查看**所有卡**的健康、进程、显存和利用率，选择健康空闲卡。
下面假设本次选择物理卡 2，每个进程显式将它映射到逻辑卡 0；应按实际状态修改。
设备错误立即停止，不 reset 或换卡绕过错误。

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1
python -m benchmarks.vector.run --output build/quickstart/scale-shared --library build/quickstart/scale-shared/scale.so --physical 2 --counts 0 1 7 --repeats 1
```

runner 校验 ASC/库身份，检查全部输出、输入不变和 guard，将结果写到
`build/quickstart/scale-shared/device.json`。`--library` 会另建 `source-check/` 核对
当前源码，不覆盖原库。记录中的 ACL stream interval 包含 host 提交间隙；
kernel time 用[msprof 流程](testing.md)获得。

## 4. 理解并修改 DSL

下面是 scale 的主要结构，完整可运行定义见[源码](../../examples/vector/scale.py)：

```python
from ascendflow import language as T

# x/y 是两槽 UB buffer；X/Y 是 GM tensor，t 是运行时代次。
for t in T.Pipelined(tiles):
    with T.Stage(0, name="load"):
        T.copy(X[t, :], x[t % 2, :])
    with T.Vector(), T.Stage(1, name="compute"):
        T.mul(x[t % 2, :], 2.0, out=y[t % 2, :])
    with T.Stage(2, name="store"):
        T.copy(y[t % 2, :], Y[t, :])
```

用户选择两槽和三个 stage；编译器负责跨 pipe 的就绪与最后 reader 释放。
stage 偏移、buffer 槽数和事件宽度是不同概念。混合 kernel 用 `T.Cube()` 和
`T.Vector() as sid` 表达角色；两个 AIV 共用一份代码，按 runtime SID 选择地址。

Python API 可用于检查 IR 和生成 ASC：

```python
from ascendflow import compile
from examples.vector.scale import scale

artifact = compile(scale.specialize(B=64))
print(artifact.asc)
```

直接 `compile()` 使用进程内协作预算；带硬期限的文件构建和 CANN 交付使用上面的 CLI。
只能特化 `constexpr`，不能把 runtime `tiles` 作为 `--parameters`。
完整语法、输入声明与限制见[DSL](dsl.md)。

## 5. 下一步与排错

- 普通 `C=A@B` 从 [GEMM](../operators/gemm.md) 开始；完整 CV 算子见 [Flash Attention](../operators/flash-attention.md)。
- [示例目录](../../examples/README.md)提供每个算法的入口；[测试指南](testing.md)提供统一检查命令。
- `ModuleNotFoundError`：确认已激活独立环境、完成开发安装，并在仓库根目录运行示例。
- CANN 编译失败：确认 SDK 路径及环境，查看对应 `build.log` 和 `build-result.json`；
  `.compile.json` 仅在构建成功时保留。
- 编译拒绝 bounds、对齐或同步合同：根据诊断修改源程序或输入声明；当前支持边界见 [DSL](dsl.md)。

调用方必须满足声明的 shape、对齐、disjoint、整数范围及倍数条件。launcher 只转发
stream、设备指针和标量，不检查这些输入；不要提交负尺寸、越界指针或重叠 buffer
来测试 host 拒绝。自建 launcher 集成按生成 `.asc` 中的 `launch_<entry>` 命名参数绑定
ABI，使用设备地址并保持 allocation 存活直到 stream 完成；设备示例见
[vector runner](../../benchmarks/vector/run.py)和[FA runner](../../benchmarks/attention/run.py)。
