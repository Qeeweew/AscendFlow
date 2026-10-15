# 阶段 1 验证记录

> 本页保留对应保存点的合同与验证记录，复现路径已随目录清理更新。最新支持状态见[状态页](refactor-progress.md)，后续工作以[当前计划](refactor-plan.md)为准。

2026-10-13，xDSL 0.73.0、CANN 9.1.0、dav-2201、910B3。
此记录只覆盖局部同步闭环；高完成度的复杂算子编译/运行仍是后续目标。

## Python 和 IR

`pytest -q compiler/tests`：73 passed。`ruff check compiler` 通过。
IRDL/parser round-trip、有限依赖覆盖、slot/WAR/WAW、部分写、嵌套 view、跨 dtype
alias、容量、event 配对、资源溢出、普通 DCE 和负例变异均有回归。
count=0/1/2/3/5/8 × slots=1/2/3，覆盖 Vector 和 Cube。

独立交错状态机将 native start/retirement、Set issue/effect、Wait consumption
分开，对双缓冲小规模计划穷尽得到：

| count | Vector 状态数 | Cube 状态数 |
|---|---:|---:|
| 0 | 1 | 1 |
| 1 | 18 | 58 |
| 2 | 140 | 1152 |
| 3 | 973 | 34604 |

全部有唯一干净终态。budget 耗尽时返回 `exhaustive=False`，不报告证明成功。
静态完成图检查的是每次编译的所有有限实例，仍依赖 native/machine 契约；上述
枚举不是对任意规模、动态 task 或任意控制流的形式化证明。

## CANN 和设备

下列最终 ASC 均完成独立 device object、可执行 ELF 链接和 host launch 库编译。
同一批最终设备验证使用健康空闲的物理卡 1，进程映射逻辑卡 0；每次加载 ACL
之前保存全部八卡健康、进程、HBM 和利用率。卡 0/2/4/6 后来有其他进程负载，
preflight 拒绝占用卡上的运行。没有 reset 或设备错误后的换卡/重试。

| 程序 | 参数 | 数值验证 |
|---|---|---|
| Vector | 6×128 f32、双缓冲、stage 0/1/2 | 768 元素，max error 0 |
| 多 reader + 末块 | 5 tiles，前四块128、最后24 f32 | 536 元素，max error 0 |
| Cube/Fixpipe | 5×16³ f16 NT，f32 累加/输出，双缓冲 | 1280 元素，max error 0 |
| Vector 边界 | count=0 和 1 | 无挂起，非空128元素 max error 0 |
| Cube 边界 | count=0 和 1 | 无挂起，非空256元素 max error 0 |

输入是可精确表示的有限二进制数；Cube 使用非对称 A/B，避免 identity 隐藏转置
错误。不宣称已验证 Inf/NaN、subnormal、BF16 或其他数值范围。

初版 L1→L0A transpose 选择错误被非对称设备输入发现，已按 AtlasA2 zN→zZ 源码
修正。尝试以 source Notify 删除 V RAW barrier 也被设备发现错误；当前两个
同步模型都已修正并增加该计划的负例。最终结果只采用修正后的代码。

## 固定专家计划与性能

比较 shape=32×128，f32，Input/Output 各两槽，stage=0/1/1/2，单 AIV，物理卡1。
`benchmarks/references/ascendc/manual_vector.asc.in` 是独立手排参考，和生成源码分别编译。
最终**整个设备 ELF 字节相同**：

```text
398eebea42cf8613865a1bb6f541c09c6c72702b841eed211c94bf5684652ce7
```

选项为 `-O2 -fPIC -std=c++20 -mllvm -cce-aicore-dcpreload-args=false
--npu-arch=dav-2201`；device-only 后经 `ld.lld -m aicorelinux -Ttext 0
--no-mmap-output-file` 链接。host 库另外以 `-shared` 编译。
编译器版本：2026-07-30 clang 15.0.5。

两方各做一次标准 msprof 采集，均为5次 warmup和30次计时；取 `op_summary.csv`
中 `AI_VECTOR_CORE / vector_pipeline` 的最后30条 kernel duration，不含 host 分配
和复制。未改变频率/功率策略，也没有调参。

| 指标 | AutoSync | 固定手排 |
|---|---:|---:|
| mean kernel duration (µs) | 5.0356 | 5.0413 |
| median (µs) | 4.970 | 5.002 |
| min/max (µs) | 3.784 / 6.400 | 3.800 / 6.440 |
| mean MTE2 ratio | 0.5441 | 0.5475 |
| mean MTE3 ratio | 0.5923 | 0.5901 |
| mean Vector ratio | 0.1790 | 0.1788 |

同一单核的三条 pipeline ratio 合计超过1，设备计数支持此例中存在流水重叠。
两方代码相同，均值差0.0057 µs不解释为性能提升；同一次采集里的30次调用也
不是30个独立对照组。这只验证该固定小例的代码与设备行为，没有证明生产
GEMM、MHA 或任意程序性能无损。

普通 runner 的 ACL event stream interval 包含提交间隙及可能的 profiler 开销，
单独存于 `device.json`，不与上表 kernel duration 混用。

## 产物与复现

最终运行数据在工作区外层 `runs/stage1/`，不提交产品 Git。
各目录保存 before/after IR、完成 witness、生成 ASC、编译日志和清单、健康快照、
设备输出及结果；对照目录另有两份独立源码、profile原始数据/CSV和对照摘要。
复现命令见 README；选择当时健康空闲的物理卡并显式映射逻辑0。

msprof 会 canonicalize Python symlink，直接传 venv Python 可能丢失 venv prefix。
实际成功采集使用产品脚本，保持所选解释器路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 \
XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof \
  --output=/workspace/xdsl-910b/runs/stage1/comparison/generated-profile-valid \
  --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.legacy.run /workspace/xdsl-910b/runs/stage1/comparison/generated --physical 1 --repeats 30' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
```

手排采集换成 `manual` 目录并加 `--reference`，其余选项相同。
