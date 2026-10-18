# 测试与 profiling

所有命令从仓库根目录、激活独立环境后执行。当前日常流程为：Python/IR 回归 →
示例 CANN 编译与设备数值检查 → Flash Attention / GEMM 多 shape 的 AscendFlow、ACLNN 各采集一次 msprof。

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

## FA / GEMM 多 shape：ACLNN 对照与 TFLOPS

性能只覆盖 Flash Attention 与标准连续 GEMM，两者均为 FP16，固定默认分块与排步：

| 算子 | 默认 shape 矩阵 |
|---|---|
| Flash Attention | Sq=Sk=1024、2048、4096、8192；B1/Hq8/Hkv8/D128；causal 与非 causal 都只按正方形验收 |
| GEMM normal | M:N:K=1024³、2048³、4096³、8192³、2048:8192:4096、8192:2048:4096；连续 NN，正态输入 |

每个 shape 在同一卡上先采集 AscendFlow，再采集安装版 ACLNN；每个后端只调用
一次默认参数 msprof，不覆盖采集选项。标准 GEMM 的 A/B/C 同为 FP16，ACLNN
使用 `aclnnMatmul`、`cubeMathType=0`；FA 使用 `aclnnFlashAttentionScore`，
保持相同布局、head、scale 和非 causal 合同。未加入其他 example 的性能测量。

在完成全卡状态检查和物理卡映射后：

```bash
python -m benchmarks.profile_defaults --list
python -m benchmarks.profile_defaults --output build/kernel-comparison --physical 2
python -m benchmarks.profile_defaults --causal --require-no-slower --output build/causal-comparison --physical 2
```

默认采集非 causal FA 与 GEMM，共 10 个 shape、20 次 msprof。FA 的 Sq/Sk 为 runtime，四个 shape 复用一次
CANN 构建；GEMM 为 constexpr，每个 shape 单独构建，仍有覆盖全部构建阶段的
300 秒硬期限。错误立即停止，不继续后续 shape。输出目录须是新目录。
需要检查单项时可用 `--only gemm-normal-4096x4096x4096`，该记录仅覆盖选中项。

`--causal` 单独采集 1K/2K/4K/8K 四个等长左上因果 FA；双方均为 `j<=i`，
ACLNN 使用 `sparseMode=2`。三组显式 BK/VR 政策见[算子合同](../operators/flash-attention.md)，
每份产物内长度仍为 runtime。`--require-no-slower` 要求每项加速比≥1，否则总表为
`performance_failed` 并返回非零；数值状态和性能状态分别保留。有效 FLOPs 只计可见 key 对。

原始记录位于 `<shape>/<generated|aclnn>/`；构建在 `artifacts/`，总表为
`results.json`，可直接查看的比较表为 `tables.md`。汇总会检查成对的实际 shape、
dtype、输入 hash、采样行、容差、物理卡和平台身份，并校验库和原始 CSV。
时间、加速比和 TFLOPS 均从本次原始记录计算。

msprof 的“一次采集”和 kernel 执行次数分开：沿用运行器默认次数，FA 预热 1 次、
执行 3 次，GEMM 预热 5 次、执行 30 次，两个后端相同。排除预热后取设备 task 时长
中位数；一次 launch 含多个设备 task 时相加。ACL stream interval 含 host 提交间隙，
不作为 kernel time。采样 oracle、完整输出有限性、输入不变和 guard 分别检查，
采样不称全输出验证。

[性能报告](../reports/performance.md)列出 shape、两个实现的时间、加速比和算法
TFLOPS，并说明 FLOPs 公式。加速比为 `ACLNN 时间 / AscendFlow 时间`，大于 1 表示
本次 AscendFlow 更快。每个后端只有一次采集，不据此宣称通过历史三批严格门槛。
AscendC、PMU 和 roofline 工具保留作专项诊断。
