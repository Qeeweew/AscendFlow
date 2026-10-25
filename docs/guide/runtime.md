# Tensor 调用与 host launch

`TorchRuntime` 是可选的 Torch NPU 适配器。导入 `ascendflow` 不导入 Torch；构造 runtime
或运行设备示例才需要 `torch` / `torch_npu`。运行前按[快速上手](quickstart.md)检查全部
卡，并将所选物理卡显式映射为逻辑 NPU 0。

## 编译并调用

```python
from ascendflow import TorchRuntime
from examples.flash_attention.simple import simple_attention

# Q/K/V/O 是符合该特化 shape、dtype 的连续 NPU tensor，O 由调用方提供。
with TorchRuntime() as runtime:
    kernel = runtime.compile(simple_attention, "build/simple-tensor", H=4, S=256)
    kernel(Q, K, V, O)
```

`compile` 先查询映射设备的 AIC/AIV 数，通过 `bind_device` 特化普通 DSL，再走同一条
有 300 秒总期限的 `build_module` 链。mixed/cube 返回 AIC 数，vector 返回 AIV 数。
资源和 constexpr 不在 launch 时改变，不在依赖分析后重选 group 数。

复用已构建的库时直接绑定：

```python
with TorchRuntime() as runtime:
    kernel = runtime.kernel("build/attention/attention_target.so", "attention_target")
    kernel(Q=Q, K=K, V=V, O=O)
```

普通 runtime 长度（如 Sq/Sk）由外部 tensor 的对应动态轴取得；多个 tensor 必须给出
相同值。无法从单个轴确定的参数必须按源码名称显式传入，例如 `kernel(..., active=7)`。
动态轴乘积只校验已绑定参数，不求解未知因子。constexpr 仍在编译时指定。

## 编译器发布签名

共享库发布 `kernel_metadata_<entry>()`，返回版本化 JSON；同一份内容也在
`verification.json.abi` 中。它描述真实 C ABI 顺序的外部 tensor、dtype、shape/动态轴、
layout、访问权限、runtime 参数约束、launch groups、资源及 workspace 标志。
签名从保留的 source 合同和最终参数次序建立，不重新求解内存或同步事实。

runtime 读取库内描述。调用方不用维护 ORDERS、手动取 `data_ptr()` 或解析 ASC 文本。
缺少 metadata 的旧产物需要重新构建，不猜测签名。相同路径的已加载库若发生变化会拒绝
再次绑定；新构建使用新输出目录，防止动态链接器静默复用旧代码。
底层 `launch_<entry>` 和 `ACL.kernel` 仍提供显式指针接口；完整容量与 512 B 对齐合同
见 [GM workspace](workspace.md)。Tensor 适配器不改变这个 C ABI。

需要精确控制输入字节和尾部 guard 的设备测试使用已有 `ACL.kernel`：

```python
kernel = runtime.kernel(library, "scale", parameters=(tiles,))
kernel(input_pointer, output_pointer)
```

标量在绑定时固定，stream 由 ACL runtime 持有，tensor 个数和 workspace 从库内查询。
Vector、归一化、GEMM、attention 和 KDA 的生成 kernel 均走这两个 runtime 入口。
`ACL.elapsed_us(start, stop)` 只封装事件计时和毫秒到微秒的换算，不代表 kernel time。

## 校验、复用与 stream

默认每次 tensor 调用检查名称、dtype、rank、完整 shape、连续存储、映射设备及
runtime 标量合同。源码要求外部 disjoint 时还检查 tensor 字节区间。
linear ABI 仅接受 ND 存储；外部 NZ ABI 要求实际 FRACTAL_NZ format 29。
runtime 不把连续的逻辑 shape 当成打包布局证明，也不在绑定或执行时做格式转换。
不执行隐式 cast、contiguous copy、clamp 或填充，也不检查 metadata 内容或逐次间接
访问的动态边界；这些仍遵守[内存合同](../development/memory-contracts.md)。

对于参数 tensor 固定的重复执行，可以只校验一次：

```python
invoke = kernel.bind(Q=Q, K=K, V=V, O=O)
invoke()
invoke()
```

绑定后 tensor 必须保持原有 shape、dtype、设备和存储地址，内容可按算法合同更新。
每次调用仍取得 Torch 当前 NPU stream。workspace 在该 stream 和 runtime 参数组合的
首次调用分配，之后复用；不同 stream 或参数组合拥有不同 arena。tensor 及 arena
通过实际 consumer stream 的 `record_stream` 保持异步生命周期。跨 stream 的输入
ready 顺序仍由调用方建立，runtime 不插入等待或 barrier。

重复调用通过工作区 Torch NPU 的 `_npu_getCurrentRawStreamNoWait` 查询当前句柄；
只在首次遇到该 stream 时获取 Python stream 对象。`bind` 为固定存储在每个实际使用的
stream 登记一次生命周期，不反复构造 stream 对象，也不把第一次的 stream 固定下来。

普通 launch 不同步设备。`close()` / `with` 退出时先退休 runtime 使用过的 stream，
再释放 arena。未关闭的 runtime 保留已使用的尺寸/stream binding；不要为每次 launch
新建 runtime。输出保持显式参数：`access="write"` 允许部分写，不能据此自动分配一个
未初始化输出。跨 kernel 中间结果仍由算法 host 代码编排。

## 验证与性能口径

CPU 检查直接编译执行生成的 C++ metadata/容量查询和 launcher 参数绑定，覆盖动态轴
冲突、非法 dtype/shape/layout/device、disjoint、尺寸变化、多个 stream、关闭后的调用
和产物身份变化。KDA pipeline 与独立 fragments 从同一签名绑定 tensor。

runner 简化后的 25 组 example 均完成设备数值检查。事件分配改为独立有限求解上下文后，
27 个特化中 23 份 ASC 与改动前一致，复用相同编译选项和编译器下的设备证据；其余
GEMM、KDA prepare、gram-cross 和 Sparse MLA 重新完成 CANN 编译与设备数值检查。
同一 typed source 仍走公共 source 编译链，不接受 physical 快照作为编译资格。

变化项的性能对照共用外部 tensor 和 workspace 地址，每组预热 20 次、测量 50 次，
交替顺序执行三组批次，并覆盖两组实际地址。使用原始 CSV 的 device task duration；
prepare 两组中位数分别为 37.55→36.87 us、37.96→37.01 us，其余变化项差异在 ±0.2%
内。不同 GM 地址的首轮对照不作为性能验收。所测特化与地址下未观察到明显设备回退，
不能外推全部 shape、稳定加速或业务 E2E。完整身份、复用条件与原始 CSV 索引见
`build/example-code-simplification/validation-summary.json`。

独立 host probe 通过动态 0/1/79 行、双 stream 与六个独立 arena 的设备全输出检查。
相同空任务 probe 的五组交错采样中，固定 tensor 调用的 host 提交时间中位数从初版
stream 对象查询的约 54.5 us 降至 raw stream 查询的 11.1 us，同批低级指针入口约
8.3 us。三种路径在同一进程交错执行，每批预热 20 次并提交 500 次。这是 host 提交
计时，不能用作 kernel time；probe 不代表完整业务 E2E，记录在
`build/host-runtime-device/host-submit/`。
