# 编译、正确性与性能验证

从 `/workspace/xdsl-910b` 执行，模块入口使用 `PYTHONPATH=compiler`。这些工具
不进入 compiler wheel；生成 ASC、库、输入、日志和 profiling 数据写入外层 runs。
算子源码在 examples；数值参考共用 `benchmarks.validation`，无 ACL 初始化副作用。

2026-10-15：旧 task DSL、legacy runner 和旧 FA suite 已删除。当前AST与独立
参考构建入口使用 `xdsl_npu.build`，输入构造、分析、ASC和CANN共用≤300秒总期限。
每个variant是单个算子构建；超时失败立即抛出，不能继续下一候选。构建结果先写
临时目录，通过后发布；失败使成功manifest和二进制失效。复用已有库时，当前源码
检查写入独立 `source-check/`，不覆盖待测库。默认dense f16 FA现已接通，当前性能与编译成本仍未达标，见产品状态页。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `compile.py`、`regions.py` | 冷进程源码→ASC 计时与静态结构压力输入 |
| `vector/`、`gemm/`、`normalization/`、`state/` | 对应 AST 示例的固定政策验证与分析 |
| `attention/` | 新DSL FA及固定政策AscendC/ACLNN验证、分析与批次资格检查 |
| `references/ascendc/` | 独立手写 ASC 模板，不能由生成 PlanIR 反向生成 |
| `references/` | ASC 模板读取、固定 FA 参考参数与 ACLNN 适配器 |
| `validation/` | 确定性输入和 CPU 数值参考，供测试与 benchmark 共用 |
| `profile.sh`、`profile_summary.py` | 通用 msprof 启动及数据汇总 |
| `artifacts.py` | 统一校验 profile 对应的源码/库 hash 和旧相对路径 |

每个算子的参数、工作量和误差合同保留在其 runner。当前只合并重复启动、产物
校验和共享输入/oracle，不引入通用算子注册框架；仍有重复设备执行代码待收敛。

## 新DSL Flash Attention

唯一数学主体在`examples/flash_attention/attention.py`。runner目前接受dense
B1/Hq8/D128/groups20、Hkv=8/2/1、dtype=f16/bf16、query tile=128/256的显式政策。
不可变合同统一驱动实际specialization、七个buffer的shape/字节数、数值参考、
ACLNN及独立AscendC；加载产物时核对实际source/physical IR与指定政策。
这只是工具支持范围，真实构建、数值及性能资格以[状态页](../docs/refactor-progress.md)为准。
`--build`才编译，复用产物时按实际launcher ABI绑定参数并检查源/库hash。
ACLNN记录其独立库身份，不能用默认库测试另一个dtype/head政策。
`--query-tile 256`选择同一example的BQ256/BM128 K/V复用政策，Sq须整除256，
Score/P/PV workspace随BQ改变。独立AscendC仍是BQ128模板，只作相同数学合同的参考；
runner明确记录这一区别，不把两种排步称为相同。

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run --output runs/fa --build
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run_policy --output runs/fa-reference --batch 1 --heads 8 --kv-heads 8 --groups 20 --dtype f16 --build
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run --output runs/fa --physical 6 --backend generated --cases 4096:4096 --scope sampled --warmup 5 --repeats 30
```

物理6仅为示例，运行前重新选择健康空闲卡。将`--backend`改为`aclnn`运行安装版
基线；改为`ascendc`并加`--reference-output runs/fa-reference`运行独立手写参考。
默认四个短域做全输出oracle；大例采样时每head每query tile检查6行并标注sampled。
runner报告ACL stream间隔，包含host提交间隙；性能判定需msprof的device task时间。
通用`profile.sh benchmarks.attention.run ...`和`profile_summary`均可复用，三批交替
结果与硬件解释见[首个设备基线](../docs/attention-performance.md)。旧`attention/compare.py`
保留历史记录schema；当前三后端profile原始证据及聚合脚本位于该基线列出的外层runs。

## 无设备验证

新增AST softmax复用normalization入口：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.normalization.run --operator softmax --output runs/softmax --build
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.normalization.run --operator softmax --output runs/softmax --physical 1 --rows 1024 --variant all --warmup 5 --repeats 30
```

第二条须先检查所有卡并选择空闲健康卡；物理1仅示例。shape为32768×1024 f32，
`--rows` 是每group行数；`--variant all` 包含独立AscendC和安装版ACLNN。
zero/短循环使用 `--variant both --rows 0 8 16 24`，同一binary无需重新编译。
三批profile可用下文通用包装器，并以 `benchmarks.normalization.analyze ...
--variants generated ascendc aclnn` 汇总。完整证据见[Vector算子验证](../docs/vector-row-performance.md)。

冷编译使用独立工具；每次trial新建Python进程，不复用编译或分析cache：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/scale.py --entry scale --parameters '{"B":64}' --mode both --trials 5 --timeout 300 --metadata '{"policy":"fixed"}' --output runs/compile-cold
```

`--entry`必须指向捕获的AST `prim_func`；省略时读取序列化xDSL。constexpr用
`--parameters` JSON对象显式指定（数组转为tuple），runtime长度不作为特化参数。
`--mode normal/deep/both`分别调用普通编译、额外独立终检或两组对照，报告比较ASC身份。
默认五次冷进程；`--trials 1`明确记为probe，不能作为五次性能资格。每trial的
`--timeout`须在(0,300]秒内，整个worker及其子进程组共用硬期限；超时或失败立即
落盘并停止后续trial、mode与input。正常或错误退出也清理残留子进程。

`benchmark.json`记录当前run、状态、输入/entry/参数/metadata、模式和完成次数。
独立run目录保存各trial的JSON与日志，每组只在首trial保存一份实际源码、source/物理IR、
ASC和报告；重复运行不混用旧产物。每trial记录source/package/ASC SHA、RSS、内部phase，
同组身份变化立即失败。工具不运行CANN或设备。

`source_to_asc_s`从首次读取源码到compile返回，包含输入构造、AST、全部分析、ASC和
内部IR/report snapshot；import、身份计算和artifact磁盘写入另列。`process_wall_s`
覆盖完整worker生命周期，含这些额外工作及启动开销，不能将写文件前后挪动分析来
降低总耗时。冷进程不代表清空OS文件cache。

2026-10-16编译器重构的固定源码/政策对照结果见
[当前状态](../docs/refactor-progress.md#本轮编译耗时)。FA基线1次、当前5次，小算子两版
均5次；比较相同的编译器内部计时口径，同时列出完整源码到ASC和进程时间。
本轮没有运行设备性能调优，编译加速与ACLNN性能差距是两项独立结果。

以下为独立的离线构建示例：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.run_grouped --frame-snapshot --output runs/cleanup/frame --build
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run_policy --heads 1 --kv-heads 1 --groups 1 --output runs/cleanup/attention-reference --build
```

第一条编译生成代码与手写参考；第二条只编译独立 AscendC FA，不能计为编译器
FA 成功。

## 设备与大规模性能

先检查所有卡健康、进程、显存和利用率，选择空闲健康卡。下例物理卡1仅作
映射示例，运行当时须重新选择；runner 在 ACL 使用前再次检查。

```bash
PYTHONPATH=compiler ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.state.run_grouped --frame-snapshot --output runs/cleanup/frame --physical 1 --cases 16:128 --warmup 5 --repeats 30 --skip-rounding-case
```

B4096/G32/16:128 的 X 为1GiB。FA 计划固定4K/8K主例，矩阵与比较规则见
[计划](../docs/refactor-plan.md)。设备错误立即停止，不 reset、不换卡规避错误。

统一 profiling 包装器的第一个参数为模块名，保持原进程 cwd，自动加入产品根
导入路径。它保留 venv Python 路径，避免 msprof 将符号链接解析到系统解释器。

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/cleanup/profiles/trial-0/generated --application='/workspace/xdsl-910b/compiler/benchmarks/profile.sh benchmarks.state.run_grouped --frame-snapshot --output /workspace/xdsl-910b/runs/cleanup/frame --physical 1 --variant generated --cases 16:128 --warmup 5 --repeats 30 --skip-rounding-case --record /workspace/xdsl-910b/runs/cleanup/profiles/trial-0/generated/device.json' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/cleanup/profiles/trial-0/generated --warmup 5 --repeats 30
```

generated/ascendc 交替做三批；完成后：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.state.analyze runs/cleanup/profiles --output runs/cleanup/performance.json
```

旧 manifest 若记录相对编译路径，显式指定当时编译 cwd：`--build-cwd compiler`
或 `--build-cwd /workspace/xdsl-910b`，不能猜测路径。新编译 manifest 为绝对路径。
旧记录的 hash/测量不因文件迁移而重写；新复现产物有新的源码/runner 身份。

先审查最终 ASC 的搬运、缓冲、累计顺序和 wait/release，再看 timeline/内核时间。
固定政策 AscendC、开源专家实现/ACLNN、硬件条件下界分别报告。记录完整源码/
库/编译选项/shape/dtype/设备/计时口径；GM 请求速率不是测得 HBM 带宽。
