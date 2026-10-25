# 设备核数与 GM workspace

`N.alloc_workspace` 声明入口未初始化的内部 GM tensor，支持静态 shape、runtime
长度参数和显式 slots。workspace 不出现在 DSL 的 tensor 参数中。每个 allocation
的起点自动按 512 B 对齐；不改变逻辑 shape、行距或每槽 stride，尾部 padding
不能作为 tensor 元素访问。

## 核数在依赖分析前绑定

`ACL.resources` 通过 CANN 9.1 `aclrtGetDeviceInfo` 查询映射的逻辑卡 0 的
Cube/Vector 核数。`N.get_core_num("cube")` 和 `"mixed"` 返回 AIC 数量，
`"vector"` 返回 AIV 数量，与 CANN 接口一致；mixed 的 1 AIC + 2 AIV 参与关系
是独立的执行合同，不用于折算查询结果。
它是 host 设备事实，在构造 source IR 之前解析为 constexpr，不是 kernel 内的 load。
算法选择实际 launch group 数，物理核数和逻辑任务数不互相替代。

```python
@N.prim_func
def example(Out: N.Tensor((TASKS, 64), N.f32, access="write"), TASKS: N.constexpr = 128):
    groups = N.min(TASKS, N.get_core_num("vector"))
    with N.Kernel(groups) as group:
        assert TASKS > 0
        work = N.alloc_workspace((groups, 2, 64), N.f32)
        row = N.alloc_ub((64,), N.f32)
        for t in N.serial(N.ceildiv(N.max(0, TASKS - group), groups)):
            N.fill(row, 2.0)
            N.copy(row, work[group, t % 2, :])
            N.copy(work[group, t % 2, :], row)
            N.copy(row, Out[group + t * groups, :])
```

`shape` 表示完整逻辑容量，上例显式包含 group 轴和两槽，不隐式再乘核数。
mixed group 内的共享 GM 不因有两个 AIV 再乘二；SID 的区域由索引表达。
跨 kernel 的中间值仍作为显式 tensor 传递，其生命周期超过一次 kernel 调用。

完成全卡健康检查并设置物理卡到逻辑 0 的映射后，host 可使用：

```python
from ascendflow.build import build_module
from ascendflow.device import ACL

runtime = ACL()
try:
    bound = example.bind_device(runtime.resources)
    build_module(bound, "build/example", shared=True)  # 300 秒总构建期限
    kernel = runtime.kernel("build/example/example.so", "example")
    kernel(output_pointer)  # 只传外部 tensor，自动分配并复用 workspace
    runtime.synchronize()
finally:
    runtime.close()
```

离线编译不初始化设备或猜核数：显式将
`DeviceResources(cube_cores=..., vector_cores=...)` 传入 `bind_device`，或通过 CLI
`--device-resources` JSON / `build_file(..., device_resources=...)` 提供目标资源。
配置改变后重新特化和编译；运行时拒绝将绑定了不同资源配置的产物用于当前设备。

直接传 Torch tensor 时使用产品 [TorchRuntime](runtime.md)：它从库内签名推导动态轴
和实参顺序，自动使用当前 NPU stream，调用方无需维护指针表。

## 分配与生命周期

runtime 参数在绑定 callable 时按源码签名顺序指定 `parameters=(...)`，同一 binding
内固定。运行时从生成的 `kernel_config_<entry>` 获取 tensor/标量数量及资源要求，
再调用 `workspace_size_<entry>(uint64_t* bytes, ...runtime scalars)` 计算容量。
该查询检查动态维度乘法、512 B 对齐和总和的 int64 溢出；空动态轴占零字节。

每个 binding 分配一块对齐的 GM arena，各 allocation 的大小向上取整至 512 B
再顺序打包。静态总量及动态轴的字节系数记录在 `verification.json.workspace`。
同一 stream 上的调用复用它；每次调用的入口仍不承诺已有值，不能读取前次调用的残留。
不同尺寸或并发 stream 使用独立 binding，arena 由 runtime 保持到 stream 退休后释放。
重复 launch 不申请显存，也不添加 device 同步。

底层 C launcher 在外部 tensor 后、runtime 标量前接收一个合并的 workspace 指针。
直接使用 C ABI 的调用方负责按查询结果提供足够容量、512 B 对齐、与外部 tensor
分离且覆盖全部异步访问生命周期的存储。`ACL.kernel` 自动完成这些绑定工作。

## 依赖分析边界

source allocation 以 `access="local"` 表示内部 GM，physical Buffer 保留
`external=0, initialized=0`。每个 workspace 的 root、原轴、slot selector、参与者、
真实读写和最后 reader 继续进入原有 admission、ownership 与同步链。
不同 workspace 不依靠名字分离；arena 区间由同一布局规则保证实际不重叠。
padding 不扩大 effects 或初始化范围，普通外部 tensor 的 disjoint 合同保持原义。

对齐和打包仅改变 root 的最终基址，不改写 root 内地址、执行顺序或生命周期，
因此在同步后输出时物化。设备资源则在 source 构造前冻结；不会在已经建立依赖事实后
更换 group 数，也不会用 workspace 分配隐式引入跨 group 同步。

## 示例容量

所有公开示例的私有 GM scratch 已迁入 `alloc_workspace`；输入、输出、常量和跨 kernel
中间结果仍在显式 tensor ABI 中。示例用同一 worker 数建立 launch 与 workspace 分片，
超过 worker 上限的任务通过循环复用，复用前保留实际最后 reader 退休检查。

| 示例 | 私有 GM scratch 的容量依据 |
|---|---|
| Flash Attention | GROUPS、BQ/BK/D 与两槽；默认 65 MiB，独立于 Sq/Sk 和 batch/head 任务数 |
| 教学 attention | `min(H,GROUPS)` 个 worker、固定 64×128 tile 与两槽；H 超过 GROUPS 后容量不再增加 |
| Sparse MLA | `min(GROUPS,T*(H/BM))`、TOPK/S、D 与三槽 gather/两槽计算；固定选中容量时不随 context 无界增长 |
| KDA Prepare / Output | GROUPS、固定 tile 和 slots；私有 scratch 不按 chunk/head 分配 |
| KDA Scan | `4*GROUPS*(2*K*K+BT*K)` 字节；默认 20 组共 3.125 MiB |
| KDA Gram-cross / Solve / WU | GROUPS、PACK/WIDTH/tile 和 slots；只保留实际输入输出为参数 |

其余公开示例没有私有 GM scratch。改变 worker、tile 或选中容量政策仍会改变 workspace。
KDA 的 Gp/KG/QG/A16/U/W32、Snapshot/Residual 等跨 kernel 数据仍随任务数增长，
这次迁移不使整个 pipeline 的总显存恒定。具体合同见
[KDA prefill](../operators/kda-prefill.md) 和 [Sparse MLA](../operators/sparse-mla.md)。

## 验证范围

CPU 回归包含：核数绑定后与手写相同常量的 ASC/依赖一致，多 group、双 AIV、
空任务、跨槽和多次调用的有限事件检查，读前未初始化拒绝，以及实际 C++ 容量查询
的动态轴、非整齐字节数、空尺寸和溢出检查。

2026-10-25 在 910B3、CANN 9.1.0 上查询到 20 AIC / 40 AIV。三份产物通过 CANN
编译；动态 workspace 与按 launch group 分片的两个 probe 共七种尺寸通过设备全输出、
workspace 全字节、padding 和尾哨兵检查。教学 attention 已改为内部 workspace。
本次证据保存在 `build/workspace-device/`，其 `identity.json` 关联当前源码和 ASC，
`validation/device.json` 关联 probe 的库、选项及尺寸，`comparison.json` 关联基线与
新版本相同输入、输出和原始 task CSV。

教学 attention 单批 5 次预热、30 次 launch 的 device task 中位数为基线 11.59 us、
新版本 11.62 us；Python→ASC 编译分别为 1.247 s、1.163 s。双方各一次 profiling，
不作为稳定性能结论，也不使用 stream interval 代替 kernel time。

示例迁移另完成全部 26 个公开入口的 27 种特化 CANN 构建（25 个验收组），以及
受影响 attention、Sparse MLA、KDA pipeline 和六种 fragment 的设备数值检查。
教学 attention 额外以 H6/GROUPS2 检查同一 worker 连续处理多个 head。CPU 回归包含
固定 worker 下扩大任务规模的容量比较、runner 外部 ABI 核验和 scan 跨序列复用的有限
事件检查。构建、数值和容量身份保存在 `build/example-workspaces-*`。

本次同输入的单批 device task 对照（各 5 次预热、30 次 launch）为 FA 中位数
35.04→32.87 us、scan 27.268→26.8 us；scan 批内波动较大。Python→ASC 单次时间
分别为 FA 11.794→11.731 s、scan 2.843→2.634 s。这些采样不作为稳定性能结论；
原始 task CSV、输入输出 hash、ASC/库及 CANN 选项保存在
`build/example-workspaces-performance/`。
