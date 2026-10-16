# 编译、正确性与性能工具

从 `/workspace/xdsl-910b` 使用已有 `.venv`，模块入口加 `PYTHONPATH=compiler`。源码在examples；产物、日志、profile和运行数据写外层runs，不纳入产品Git。**算子优化暂停，本轮文档整理未执行本页命令。**

工具参数支持不等于对应算子已取得资格。当前失败指标与历史证据只见[状态页](../docs/refactor-progress.md)；完整测试/性能口径见[validation.md](../docs/validation.md)。

## 目录职责

| 路径 | 职责 |
|---|---|
| compile.py、regions.py | 冷进程源码→ASC、结构成本输入 |
| vector/、gemm/、normalization/、state/ | 对应真实DSL示例的验证 |
| attention/ | DSL FA、固定政策AscendC与ACLNN的构建/验证/profile |
| references/ascendc/、references/ | 独立手写ASC、CANN/ACLNN适配；不从生成PlanIR反向构造参考 |
| validation/ | 无设备初始化副作用的数值参考和确定性输入 |
| profile.sh、profile_summary.py、artifacts.py | msprof包装、汇总及源码/库身份校验 |

每个variant的输入构造、Python、ASC、CANN共用≤300s总期限；超时或失败停止后续候选，清理整个所属进程组。不能拿较短阶段预算代替整个build上限。历史已删除库需重建，不能直接运行旧hash记录。

## 无设备：冷编译与CANN

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/scale.py --entry scale --parameters '{"B":64}' --mode normal --trials 5 --timeout 300 --output runs/structured-loops/scale-cold
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/flash_attention/attention.py --entry attention_target --mode normal --trials 1 --timeout 300 --output runs/structured-loops/fa-probe
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --compile --timeout 300 --output runs/structured-loops/cann-fa
```

`--parameters`只传显式constexpr；runtime Sq/Sk不作为特化参数。normal/deep/both区分普通编译与显式独立检查，`--trials 1`是probe，不是五次资格。source→ASC包含源码构造、全部分析/化简、输出和内部快照，import、进程总时长、CANN及RSS单列。禁止跨编译cache美化数据。

每次独立目录记录输入、entry、参数、编译器/源码/ASC身份、版本、选项、阶段计时和完成状态。失败即终止后续trial。默认FA目标p50≤5s、max≤10s；达到资格前不反复跑五次FA。

## FA三方对照

数学和workspace合同见[attention-target.md](../docs/attention-target.md)。当前工具能接受部分head/dtype/tile政策，但默认例的设备证据不能自动扩展到这些参数。独立AscendC参考是BQ128模板；BQ256等政策与它不具有相同排步，比较必须标注。

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run --output runs/structured-loops/fa --build
PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run_policy --output runs/structured-loops/fa-reference --batch 1 --heads 8 --kv-heads 8 --groups 20 --dtype f16 --build
```

第一条构建生成代码，第二条构建独立手写参考，互不替代。ACLNN另有独立库身份。恢复设备资格后对比generated、ascendc、aclnn并结合ops-transformer源码，不只与较慢手写参考比较。

## 设备运行与profiling（需要先检查卡状态）

先检查所有卡健康、进程、显存和利用率，选择健康空闲物理卡；每进程显式映射逻辑0。以下物理6只演示参数，不能照抄为固定选卡。错误立即停止，不用/tmp锁、不reset、不换卡规避错误。

```bash
ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=compiler .venv/bin/python -m benchmarks.attention.run --output runs/structured-loops/fa --physical 6 --backend generated --cases 4096:4096 8192:8192 --scope sampled --warmup 5 --repeats 30
```

`--backend aclnn`使用安装版对照；`--backend ascendc --reference-output runs/structured-loops/fa-reference`使用手写参考。sampled当前按每head每Q块取6行，必须如实标注；不能替代短域全输出检查、输入不变和guard检查。ACL stream interval包含host提交间隙，性能结论应同时使用msprof设备task时间。

profiling统一通过`benchmarks/profile.sh <模块名> <参数...>`调用，必须设置`XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python`，否则包装器会拒绝启动；物理映射等环境变量同样传给被profile进程。通用汇总入口为：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.profile_summary runs/structured-loops/profiles/trial-0/generated --warmup 5 --repeats 30
```

原始profile必须与该源码/ASC/库和设备绑定，不能混用旧路径。主例使用4K/8K等稳态规模，固定政策，三后端交替有限批次；不无界调参。

先审查最终ASC的搬运、槽位、early release、CV派发与等待，再用可用的pipe/访存/冲突指标解释差距。不同PMU replay组不可相加分解延迟；GM请求包含L2命中，不等于HBM实流量。理论算力/带宽下界写明核心数、频率、dtype、FLOP/字节和假设，不能从设计宣称性能无损。
