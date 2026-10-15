# 当前状态与问题清单

2026-10-15，基于 `041996f4da1f` 推进第一批实现。之前文档与停止的block-sum
实验diff已保存到外层 `runs/refactor/dsl-implementation/starting.patch`。
本页区分本轮实现、历史性能和仍未接通的能力。

**新 DSL 的完整 Flash Attention 仍不能编译。** 旧FA、公开physical builder与
tracer入口已删除；没有绕行内部构造器的兼容example。下一主线是runtime索引/
descriptor和一般nested CV，[计划](refactor-plan.md)规定其退出条件。

本轮已实施：

- 删除 `xdsl_npu.Kernel`、`xdsl_npu.builder`、`trace_func`、隐式`versions=`/
  L0 `gemm`、旧examples/runners。`_physical_builder`只供lowering和内部测试。
- 接入 `T.int64(minimum=1, multiple_of=B)`；constexpr可后置声明。合同贯通
  source/physical IR和local/sequence/mixed launcher，非法值在地址计算和派发前
  拒绝。分析保守使用非负域；没有宣称任意参数约束或BNSD已支持。
- 跨source region逸出的allocation明确诊断；FA设计模板改为共同scope声明。
- NativeCompletion共享一份拓扑，各query domain保留独立激活和事实，新增真实
  结构计数。初版没有提速，定位后去掉重复激活绑定和包装调用；最终收益有限，见下方实测。
- 通用构建工具覆盖输入构造→Python→ASC→CANN总期限，失败不发布成功产物；
  现有AST/独立参考runner统一调用。编译缓存没有代替分析。
- 关键机器反例迁到内部fixture；旧FA整算子策略测试改为小型CV、跨owner状态
  WAR和独立数学测试。编译器内旧FA专用`mask_causal/mask_prefix`recipe已删除，
  后续mask应由通用typed运算和example有效域表达。

## 路径与实际能力

| 路径 | 已有能力 | 明确边界 |
|---|---|---|
| AST DSL → source IR → physical IR → AutoSync → ASC | runtime 局部流水、Vector/归约、显式 Cube、受限 serial parent/child、部分局部 CV | 一般 nested mixed/CV、runtime BNSD/商余 descriptor、完整 FA 未接通 |
| 内部physical IR测试 | 保留机器访问、事件、ownership与退休反例 | 没有旧用户DSL；有限fixture不证明runtime FA支持 |
| FA 设计 fixture | 真实数学主体、静态容量与有限派发表检查 | 测试目前要求编译失败，不是 executable example |
| 独立 AscendC / 安装 ACLNN | 数值、排步与性能参照 | 不证明编译器或新 DSL 支持同一算子 |

AST 的两 runtime 轴、只读共享、serial 层次已有受限实例，不是全部从零开始。
同样不能由这些子集推断一般 BNSD、多时钟 CV 或间接访问已支持。
公共入口在 [compilation.py](../src/xdsl_npu/compilation.py)，
目标失败回归在 [test_attention_target.py](../tests/test_attention_target.py)，
旧FA suite已删除，新FA设备runner尚未实现。

## 待处理的不止四项

| ID | 实际问题与源码依据 | 处理与验收方向 |
|---|---|---|
| API | 旧公开task/tracer及FA入口已删除，AST接口保留 | 不再维护兼容层；内部物理fixture不成为用户入口 |
| INPUT | minimum/multiple合同已贯通IR/launcher；多参数关系与head映射仍未完成 | 继续完善可验证地址/owner前提，保留溢出和空工作语义 |
| INDEX | `language.py` 的乘积、div/mod 与 `lowering_context.py` 的动态维度仅支持特定模式 | 结构化 shape/stride/descriptor 与范围证明，合法长度不进编译键，不靠放宽字符串生成 |
| CV | `region_lowering.py` 拒绝一般 mixed sequence 和 AST nested pipeline；条件 lowering 仅有受限末代后缀 | 保留父子 actor/clock/generation、first/last/零次路径；完整 QK/softmax/PV/update 纵向闭环 |
| MEMORY | 已有精确 view/root/局部 ownership；不等长 group、共享 Q/K/V、间接地址及跨 child 复用仍不足 | 真实范围、有效域、最后读取和 owner 分区；buffer 分配/布局/可见性与同步一同验证 |
| SYNC | runtime 叶已直接生成，局部mixed已有双AIV quorum；legacy仍有候选试删。一般nested CV中的贡献组合和事件共享未完成 | 共用硬件完成语义，分开物理访问与事件寿命；最终从实际 IR 检查，不能旁路或加任意 barrier |
| EFFECT | 细粒度 native 已能在最后 reader 后早 release，但主要使用 task end；SPR/入口协议另有特殊处理 | 明确每个 native 的读退休、写完成与配置资源合同；不能让高层 recipe 隐藏多个里程碑 |
| EVENT | 已有有限表分段周期压缩和标准整数 pass；有限 ID 表仍随 generation 增长，runtime 跨 channel/region 共享不足 | 从符号寿命直接分配/表达 ID；化简不得改变配对、代次或等待位置 |
| COMPILE | 查询域拓扑已共享；条件记录扫描、激活绑定、父层重复工作仍需测量 | 拒绝用结构改善代替真实耗时；事实成本和父层计数继续跟踪 |
| LOWER | 旧FA专用mask recipe已删；一般mask/间接访问typed合同仍缺失 | head mapping/online状态留在examples；native effects/scratch完整可见 |
| TEST | 目标失败测试与端到端支持易混淆；runtime 部分测试复用生产访问分析，本地独立模型尚不覆盖一般 Cube/CV | 成功闭环、故意破坏 IR、独立有限模型分层；保留反例而非旧 API；不放宽判定以获得通过 |
| PERF | 新 DSL FA 无性能结果；旧参考的分组/排步不总相同，部分新候选只有一批 | 同路径 source/IR/ASC/binary 与 AscendC/ACLNN/profile 对照；固定大例和硬件瓶颈分析 |
| PRODUCT | 当前入口已统一总deadline和只读产物快照；父层成本及全部ABI合同仍待完善 | 一份当前状态、一份计划；同源码/库/输入证据，运行数据不进产品Git |

## 同步与复杂度：纠正过时结论

当前 `region_completion.py` 已使用 `SymbolicCompletion`，不再复制旧首尾指令
窗口；本轮又共享了局部查询域拓扑，仍有按调用的family和各域条件事实。不能按旧计划
“先取消窗口”重复做已经完成的工作。runtime 直接生成与 legacy 有限搜索也是
两条不同路径，不能笼统宣称“全编译器已没有试删”。

[同步问题清单](sync-known-issues.md)区分已处理的拓扑/零计数与剩余问题。以下为历史保存点数据：
既有单次内部计时为 GEMM 0.414s、frame 0.262s、局部 CV 3.172s；
不是五冷进程结论。局部 CV 仍有 1197 family、1947 arc、68957 transfers；
`proof_domain_states=0` 是当时指标失效；本轮已换成真实completion_statistics。
证据在外层 `runs/refactor/symbolic-frontier/domains/`。

旧 FA 的几十秒成本包含 finite generation/ownership/event 分配，不能用于判断
新 DSL runtime FA 的成本，因为后者还不能编译；也不能把它一概归因于 Python。
全链300秒硬停止已从旧suite提取至 `xdsl_npu.build` 并用于当前构建入口。

另核实 Load3D 入口直接生成 S→MTE1 ID0 的 Set/Wait，却未列入 `event_ids`
报告。纯 Python 编译 `batched_matmul.specialize(TRANSPOSE_B=False)` 并启用
`load3d=True`，ASC 中两者各一次，资源表缺该 bank。当前 AIC native 闭集没有
普通 Scalar task，尚未据此发现实际冲突；它是资源报告和后续组合合同的缺口，
不能描述成已发现设备错误。依据为 `codegen.py` 的入口配置与 `loop_sync.py`
的 channel 资源汇总；后续应检查报告覆盖所有实际同步，包括入口与收尾。

## 现有性能证据的归属

| 路径与例子 | 已有证据 | 可得结论 |
|---|---|---|
| AST softmax，32768×1024 f32 | 三批 generated 217.914μs / AscendC 240.054μs / ACLNN 284.696μs，全输出检查 | 该 Vector 实例已有完整链，不能外推 FA |
| AST RMS primitive，8192×4096 f32 | generated 218.346μs / AscendC 235.884μs | 仅对应所记录原语与政策，不是完整 RMSNorm 库 |
| 已提交 legacy FA，4K MHA f16 | 三批 generated 698.726μs / ACLNN 512.678μs | 仍慢约36%，新 DSL 无此成绩 |
| 已提交 legacy FA，8K GQA bf16 | 单批 2262.544μs / ACLNN 1687.266μs | 证据不足以作三批验收，且仍有差距 |
| 未提交 legacy block-sum 实验，4K MHA | 单批 603.274μs / ACLNN 512.690μs；小例数值检查 | 已停止，未完成性能验收，不作为后续主线 |

详细口径分别见 [Vector](vector-row-performance.md)、[legacy FA](flash-attention.md)。
FP32 求和归约顺序按用户要求允许调整，不再设位级相同门槛；精度、数值误差、
概率舍入和递推状态仍有明确合同。不能用更宽容差掩盖地址或同步错误。

## 本轮验证与复现

集成回归 **894 passed / 196.95s**，只运行一次完整套件；Ruff与diff检查通过。
Vector与层次GEMM分别完成CANN离线/shared构建，总耗时5.46/5.94秒，均在300秒
总期限内。源码/命令/库hash与原始日志保留在外层 `runs/refactor/dsl-implementation/`。

五个独立冷进程，源码/输入IR→ASC，不含import与CANN：

| 输入 | 改动前p50 / max | 最终p50 / max | 生成ASC |
|---|---|---|---|
| AST hierarchical GEMM | 0.5824 / 0.5874s | 0.5617 / 0.5632s | SHA256相同 |
| segmented CV physical IR | 3.9251 / 3.9576s | 3.8145 / 4.0647s | SHA256相同 |

中位数改善约3.6%/2.8%，CV最大值仍有波动，不能称编译性能问题已解决。
CV输入是内部物理IR，不是新DSL完整FA。首版共享拓扑回退到4.083s后，通过
profile去掉重复激活绑定、约27万次包装调用和重复条件key访问；同步统计和
ASC均保持一致。该改动没有消除各域事实存储及条件包含扫描的复杂度。
对照见 `compile-comparison.json`；旧IR为适配新必填合同显式写minimum=0后，
双方使用同一份输入，不增加生产兼容层。

全卡健康/进程/HBM/利用率检查后，physical1→logical0，同一个Vector库验证
0/1/2/4096/1048576个tile。最大输入256MiB，67108864输出逐元素精确正确，
输入和guard完好，负长度在派发前拒绝。该运行是正确性验证；记录的stream
interval包含host间隙，没有新的算子kernel性能或FA性能结论。

```bash
PYTHONPATH=compiler .venv/bin/xdsl-npu --input compiler/examples/vector/scale.py --entry scale --output runs/reproduce/scale --shared --timeout 300
PYTHONPATH=compiler .venv/bin/xdsl-npu --input compiler/examples/gemm/hierarchical.py --entry hierarchical_gemm --output runs/reproduce/gemm --shared --timeout 300
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/gemm/hierarchical.py --entry hierarchical_gemm --trials 5 --output runs/reproduce/cold-gemm
```

下一项是runtime uniform索引因子与task/query descriptor，随后一般nested CV；
完整FA及其变体、跨region事件共享、设备性能对齐仍未完成。

## 工作区和本次交付边界

根目录无Git，产品变更只在compiler仓库；旧block-sum实验与本轮前所有未提交
修改已有starting.patch，未丢弃用户工作。旧FA策略代码随入口删除；通用
`reduce_sum8`物理访问合同保留，不据此宣称新DSL已有block-sum FA。

三名 `gpt-6.1-sol / max` subagent 分别实施语言/合同、构建期限、拓扑与反例迁移；
文件写入分工明确，主Agent负责集成与冷计时。参考checkout未更新：
ops-transformer `4bbfb3b2bccceb2a1b80828d04eee915169a2b2e`，
CATLASS `698379e0d4b1b94d12728e7181df2cf83a958e8f`，
ops-nn `64b41ad63a5073bc711ba03802498bfe8bdb77e5`。
参考checkout与已安装ACLNN的构建来源不能混同。
