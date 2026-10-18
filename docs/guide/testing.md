# 测试与 profiling

所有命令从仓库根目录、激活独立环境后执行。当前日常流程为：Python/IR 回归 →
示例 CANN 编译与设备数值检查 → 两个关键用例各采集一次 msprof。

## Python/IR 回归

```bash
python -m pip install -e '.[validation,dev]'
python -m pytest -q
python -m ruff check src examples benchmarks tests
python -m ruff format --check src examples benchmarks tests
```

测试职责见[tests](../../tests/README.md)。这些检查不初始化 NPU，也不替代 CANN/设备检查。

## 所有示例编译与运行

列出全部可运行入口，或只做离线编译：

```bash
python -m benchmarks.check_examples --list
python -m benchmarks.check_examples --output build/examples-cann
```

设备检查前先 `npu-smi info` 查看所有卡，按健康、进程、显存和利用率选择空闲卡。
以下假设物理卡 2 可用；每个进程显式映射逻辑 0，runner 还会复核卡状态。

```bash
export ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1
python -m benchmarks.check_examples --output build/examples-device --physical 2
```

这会对清单中的 **16 个公开示例**逐个构建 CANN 共享库、运行小规模全输出 oracle，
检查输入与 guard，包含各自适用的空、单次、多次和跨槽复用情况。GEMM 使用
129×257×259 尾块；dense NT 使用 256×512×256；FA 使用 256:1024、768:3072。
其他参数见[清单实现](../../benchmarks/check_examples.py)，结果见[示例验证](../reports/examples.md)。

每次构建覆盖 Python→IR→ASC→CANN 的总硬期限为 300 秒。失败或超时立即停止，
清理该构建进程组，保存日志和失败记录。设备检查也有独立的 300 秒上限。
`results.json` 汇总命令、状态、源码 hash 和设备记录路径；原始文件留在各示例目录。
输出目录必须不存在，以免旧记录混入新结果。定位后可用 `--only nested-cv` 等名称
在新目录重跑相关项；该记录只代表选中的项。

未支持的动态 `valid=` prefix 地址仅保留在[拒绝测试](../../tests/test_prefix.py)，
不作为可运行示例。`schedule.py` 是被 GEMM 使用的 inline helper，随 GEMM 检查。

## 两个关键用例：一次默认 msprof

性能只记录当前 AscendFlow 生成 kernel，固定两项：

| 用例 | 配置 |
|---|---|
| Flash Attention | Sq=Sk=4096，B1/Hq8/Hkv8/D128，FP16，默认 BQ256/BK1024 政策 |
| GEMM normal | M=N=K=4096，连续 NN，FP16 输入输出，正态数据，默认缓存政策（此尺寸为 normal） |

在完成卡状态检查和映射后：

```bash
python -m benchmarks.profile_defaults --output build/kernel-times --physical 2
```

脚本构建两份产物，然后**每项只调用一次 `msprof --output=... <application>`**。
不覆盖 msprof 的采集参数，不加 PMU replay 或其他指标组，不运行 ACLNN/AscendC 对照，
不做 shape/dtype 搜索。原始 CSV、数值记录和 kernel time 汇总位于
`<用例>/profile/`，总表为 `results.json`。

msprof 的“一次采集”和 kernel 的执行次数分开：沿用运行器默认次数，FA 预热 1 次、
执行 3 次，GEMM 预热 5 次、执行 30 次。按固定顺序排除预热，kernel time 为剩余
设备 task 时长的中位数，单位 µs；同一次采集的重复执行不称独立试验。
多个设备 task 构成一次 launch 时相加。数值使用 sampled oracle，完整输出有限性、
输入和 guard 另检查；采样不能称全输出数值验证。

[当前性能报告](../reports/performance.md)只列两行和复现、证据位置。ACL stream
interval 含 host 提交间隙，不作为 kernel time。旧的三批对照、性能门槛、PMU 与
roofline 工具保留作专项诊断，日常报告不要求它们；它们的历史结果通过 Git 查阅。
