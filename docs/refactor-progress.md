# 当前重构状态

本页是实现与证据入口，算法合同见[编译不变量](compilation-invariants.md)，剩余工作见[重构计划](refactor-plan.md)。主干结构和旧实现清退已提交于`e5920d4`及之前；其后的整数事实、端点绑定与定义性修复已完成。**整体目标尚未完成：默认FA仍被参数相关事件域的表示限制阻塞。** 算子调优未恢复，没有本轮设备结果。

## 当前唯一编译链路

```text
DSL/source → 保留控制与地址SSA的native IR
  → 物理访问/准入/ownership → region依赖 → 逻辑通知
  → 精确事件域 → 进展/生命周期/资源分配
  → 实际同步IR → 通用纯整数化简 → ASC → CANN
```

- 已删除被替代的planner、scope/query/loop/sequence求解器、旧物理协议和过渡解码器；没有兼容wrapper或tests中的历史实现，历史由Git追溯。
- 公共compile只接受DSL/source；后端只打印最终IR的真实控制、地址、原语参数和通知，不另用计划重建执行。
- 算法和分块政策留在examples。两个AIV共用一份body，通过runtime sid区分；参与者的私有存储、贡献与ownership分别分析。
- 父region只消费child接口。实际开放通知与queue/controller阻塞接口不代表native完成或buffer退休；跨代次关系由拥有实际时钟的祖先组合。
- 整数推导由一个`IntegerFacts`环境持有完整有序前提；端点绑定只建立一次。Expr保存结构hash/确定性排序信息，替换保留共享节点和不变子树。不跨编译缓存结果，不按runtime长度展开。
- 区间推导保留因子/clamp，代数规约负责等价比较；unsigned除法和余数保留定义性前提。选中FIRST/LAST的非空前提随canonical绑定保存，早于选中坐标的求值；不对上一父代次二次平移。
- 基准工具读取当前产物。`source-work-v1`区分静态模板与显式算法量；尚未建立的运行期native/搬运请求计数明确不可得，不能据此报告设备吞吐/带宽。

## 编译与验证结果

`e5920d4`上的嵌套CV完整构建曾在300秒被终止，见[原构建状态](../../runs/refactor-no-legacy/nested-cv/build-result.json)。随后30秒cProfile诊断停在progress、尚未进入lease/allocator：8个native、19条通知却构造了约53万个Expr，执行约23万次polynomial调用。中断链为`_local_arcs → _impossible → _prove → normalize → expression → repr`，证据见[diagnostics](../../runs/refactor-no-legacy/diagnostics/)。采样耗时不是正常编译时间，各累计时间也不能相加。

完成上述职责收拢后，当前默认配置的独立新进程单次记录为：

| 示例 | source→ASC | 证据 |
|---|---:|---|
| Scale，B64 | 0.76s | [benchmark.json](../../runs/refactor-no-legacy/compile-scale-final/benchmark.json) |
| 批量GEMM，M32/N64/K64，20 groups | 2.30s | [benchmark.json](../../runs/refactor-no-legacy/compile-gemm-final/benchmark.json) |
| Softmax，R8/D1024，32 groups | 1.16s | [benchmark.json](../../runs/refactor-no-legacy/compile-softmax-final/benchmark.json) |
| RMSNorm，D4096，32 groups | 1.21s | [benchmark.json](../../runs/refactor-no-legacy/compile-rms-final/benchmark.json) |

包含源码读取、构造、全部分析、ASC和IR快照；import与进程墙钟另列。运行长度仍为参数。这些是单次probe，不是五次p50资格；没有对应旧基线时不声称加速倍数。一份较早Scale记录可能与测试重叠，已标为不使用其计时，表中只使用隔离后的final记录。

嵌套CV当前通过公共CLI与CANN：编译器内部9.17s、CANN2.43s、总构建11.81s，见[构建状态](../../runs/refactor-no-legacy/nested-cv-final/build-result.json)。实际有8个native模板、40条依赖、19条通知，进展图80点/135边；8个native均只打印一次。可直接查看[最终ASC](../../runs/refactor-no-legacy/nested-cv-final/nested_cv.asc)、[同步前IR](../../runs/refactor-no-legacy/nested-cv-final/before.xdsl)和[同步后IR](../../runs/refactor-no-legacy/nested-cv-final/after.xdsl)。

该最终IR/ASC与前一合格产物逐字节相同；独立最终IR事件/访问模型已覆盖n=0/1/2、groups=0/1及双AIV，见[有限模型记录](../../runs/refactor-no-legacy/nested-cv-current/finite-model.log)。不重复检查相同产物，也不将这些有限实例称为任意N证明或设备资格。

本轮集成组首次173项通过，9项因手工fixture仍按旧字段位置构造而失败；按当前绑定合同迁移后，对应两文件17项重跑全部通过。没有增加兼容分支或删除反例。表达式/index组另有52项通过、2项host构建检查未选。当前有真实IR事件/字节模型、条件进展、配对/资源破坏、定义性和DAG工作量回归；没有运行全量suite。全局Ruff和diff检查通过。

Vector、Cube/Fixpipe和嵌套CV的CANN证据不等于设备数值/性能结果。默认FA尚无当前成功ASC/CANN产物。通用标量pass已接入；event ID中仍有多余min/max和取模，没有宣称可读性或设备性能已经达标。

## 默认FA的实际阻塞

默认FA有44个native、473条依赖、213条通知。通用整数规则及选中坐标条件顺序修复后，已越过此前的无符号算术定义性错误。最新拒绝为：

```text
loop-varying guard lacks an exact interval/residue membership proof
```

证据见[fa-binding-diagnostic](../../runs/refactor-no-legacy/fa-binding-diagnostic/)。具体待证明表达式和有序前提已加入错误诊断。失败退出时间不算编译性能结果；一次较早失败诊断还可能与1.55秒测试短暂重叠，不作为独占计时。

最小通用反例是`m > (p*m)%2`：子长度m=1时父激活只取偶数，m>=2时每代激活。当前单窗口模型没有这种参数分区能力。必须保持真实activation、配对、slot复用与条件下的生命周期；不能调大最小输入、丢条件、关闭检查或加barrier换取通过。该明确拒绝已有小回归，不作为支持能力。

## 下一部分与复现

下一部分实现参数相关事件域及其消费接口，按[计划第2节](refactor-plan.md#2-当前重点把事件合同补完整)先验收数学/通用DSL反例，再恢复默认FA完整资格。没有新域模型前不重复FA构建。随后再做独立进程编译资格和设备验证；算子调优继续留到这些工作之后。

工作区根目录命令：

```bash
.venv/bin/python -m xdsl_npu.cli --input compiler/tests/fixtures/nested.py --entry nested_cv --compile --timeout 300 --output runs/reproduce/nested-cv
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/vector/scale.py --entry scale --mode normal --trials 1 --timeout 60 --output runs/reproduce/scale
```

源码、回归和产品文档仅提交compiler Git；外层数据不提交，不推送或发布。
