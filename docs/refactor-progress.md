# 当前状态：编译流程重构与冷编译测量

2026-10-16，实施基线 `216b711`（编译器实现与 `fdff8c5` 相同）。本轮修改编译器、
回归与计时工具；FA/其他算子的示例算法和排步政策没有变化。算子性能调优仍暂停。

入口：[设计](design.md)、[不变量](compilation-invariants.md)、[计划及剩余项](refactor-plan.md)。
下面区分本轮可用改动与尚未达到的复杂度目标，不把 normal/deep 分离称作完整线性算法。

## 已落地的编译流程

公开 `prim_func → source xDSL → physical IR → autosync → ASC → CANN` 路径中：

- 私有 `PhysicalFacts` 在一次编译内交付实际scope/loop/sequence和依赖，不跨编译共享结果。
  source统一scope collection，删除`_legacy_work`及旧finite source walker。
- leaf/sequence/invocation/parent保留已建立的完成事实；精确capture/acquire构造直接
  建立其保证，物化按实际端点/guard/clock/ID核对，不独立重建整个证明。
- invocation的child寿命、parent池与counter结果直接交给query；ID置换保持memory
  事实，分配重试复用同一资源结论。无空闲bank不再求解不可能容纳的私有池。
- query新增通知追加到同一topology/flow，保留已成立的关系；未完整覆盖的请求保留
  原cases，新增边传播后重读。width=1复用原图，仅静态证明窗口不足时重建。
- query摘要显式投影所需输出，省去不用的sequence cells及无关来源的物理义务对象；
  保留完整reader/domain推导，默认通用摘要仍返回完整结果。
- 完成前沿只在相同完整guard下合并连续范围；增量传播不重建第二套pending闭包。
  进展排名更新新增/受影响约束；不可变数学域在一次编译内共享，顶层编译互相隔离。
- 合格stationary固定循环使用紧凑摘要，保留真实first acquire/tail retirement prefix。
  需要历史定义、漂移/特殊footprint和query长ReadSpan的路径仍保留精确固定表示。
- 普通结构化编译消费构造结果。`compile(..., debug_verify=True)`显式从最终IR独立
  检查；外部同步IR通过`compile(..., synchronize=False)`或`verify_plan`独立验证。
  不接受`verified=True`或序列化证明标记。低层legacy finite/mixed/dispatch入口仍有原检查。

没有新增全局barrier、搬运、buffer、算子名/shape特判或预算放宽。输入的Sq/Sk仍是
runtime参数，默认FA仍有76个native模板、12个call、20组AIC/双AIV，query/KV循环保留。

## 本轮编译耗时

测量机器为Kunpeng-920，Python3.11.15、xDSL0.73.0。当前版本每例5个独立冷进程；
小算子基线同样5次，FA基线为本轮冻结旧源码的1次构建。没有跨进程编译缓存。
下表使用双方共同的`Compilation.timings.total_s`口径：AST构造、全部分析、ASC和
内部IR/report快照；Python文件读取/prim_func载入、import与完整进程时间另列。

| 算子（固定example默认政策） | 基线中位数 s | 当前中位数 s | 下降 |
|---|---:|---:|---:|
| dense f16 FA，BQ128/BK1024/CK128/VR8，D128/Hq8/Hkv8 | 117.350（单次） | 53.000 | 54.8% |
| hierarchical GEMM，M32/N64/BK64 | 0.567 | 0.492 | 13.3% |
| softmax，R8/D1024 | 0.395 | 0.339 | 14.3% |
| RMSNorm，D4096 | 0.395 | 0.339 | 14.0% |
| Vector scale，B64 | 0.189 | 0.174 | 8.0% |

包含文件读取/构造的当前源码→ASC p50/max：FA **53.114/54.309s**、GEMM
0.526/0.532s、softmax 0.369/0.414s、RMSNorm 0.369/0.372s、scale 0.212/0.214s。
FA完整冷进程p50/max为54.488/55.803s，峰值RSS222260KiB（约217MiB）。
当前五次FA源码→ASC为52.273、52.351、53.234、53.114、54.309s。

FA规划从67.211s降到50.005s（各阶段中位数）；旧独立终检47.060s移出normal。
其余主要阶段为lowering1.600s、emission0.787s、snapshot0.465s。收益既包括去掉重复
终检，也包括约25.6%的规划降幅；仍未达到FA≤5s目标，不能把55%下降说成复杂度已解决。
小算子改善约8–14%，不夸大成与FA相同的降幅。

本轮还消除了query摘要中26711个无消费者Cell和44871个随后丢弃的Obligation构造，
保留49271个实际物理义务及其4372个唯一内存检查。query追加通知复用同一图；最终仍有
14945个query节点和84268个条件事实，表明下一步应压缩边界表示，而非继续放宽预算。

冷测工具每trial使用独立Python进程，记录读取/构造、AST、全部分析、ASC、IR/report
快照、import、完整进程时间和峰值RSS；300秒进程组硬期限，失败即停止后续trial。
数据写入外层`runs/compiler-refactor/`，不纳入产品Git。

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile \
  --input compiler/examples/flash_attention/attention.py --entry attention_target \
  --mode normal --trials 5 --timeout 300 --output runs/compiler-refactor/cold-fa
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile \
  --input compiler/examples/flash_attention/attention.py --entry attention_target \
  --mode deep --trials 1 --timeout 300 --output runs/compiler-refactor/cold-fa-deep
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli \
  --input compiler/examples/flash_attention/attention.py --entry attention_target \
  --compile --timeout 300 --output runs/compiler-refactor/cann-fa
```

Vector/GEMM/softmax/RMSNorm用同一工具分别指定其example和entry。runtime长度不作为
constexpr计时参数；4K与8K使用同一编译产物，不按每个长度重编。

## 回归和输出证据

2831项CPU回归已分段覆盖并通过：一次广泛回归后修复失败，修复只重跑对应层；
最后query增量改动另通过174项相关集成，摘要投影新增15项并通过55项相关回归。
Ruff和diff检查通过。覆盖normal不重建独立证明、deep输出一致、实际IR破坏、event
碰撞/guard孔洞、最后读取、零次/短循环、跨query资源、独立有限字节与指令模型。
有界检查不冒充任意N形式化证明。

FA独立deep检查单次101.348s，其中额外独立检查46.837s，ASC与normal完全一致。
五类算子的全部冷测ASC也均与冻结基线逐字节相同。原始计时/源码/编译器身份及汇总
保存在`runs/compiler-refactor/cold-*`、`comparison.json`；CPU回归日志在同目录。

CANN9.1/dav-2201离线构建通过，选项为`-O2 -fPIC -std=c++20`
和`-mllvm -cce-aicore-dcpreload-args=false`。完整文件→Python→ASC→CANN耗时：
FA54.145s、GEMM2.824s、scale2.121s；其中CANN分别2.615/2.251/1.888s。
每个构建共用300秒期限，全部正常退出。产物位于`runs/compiler-refactor/cann-{fa,hierarchical,scale}`；
包含实际命令、ASC/二进制SHA和编译器版本。本轮未运行设备，不产生新的吞吐性能结论。

默认FA ASC SHA256为
`e6c0e1775b61306e1fdc82311706832a4cce6c2dafd16a6570c7cee7826a9e87`。
同一源码、constexpr政策下逐字节相同，保留3个actor返回处PIPE_ALL，没有query drain。
这证明本轮该配置的生成代码不变，不能将编译加速称作设备算子加速。

## 剩余复杂度与能力边界

- FA≤5s p50/≤10s max仍是目标；目前不能宣称达到。默认FA的条件传播仍有大量
  parent节点、origin、guard facts，query也仍使用有限静态窗口复制。
- source循环不按runtime N展开，但固定child/footprint行片段/条件域仍可能放大分析。
  同域frontier合并和输出投影不等于已经实现完整参数化region transfer。
- child→parent→query仍分阶段分配。当前复用资源结论，尚未完成一次联合资源求解。
  一般guard antichain、真实偏序共享与访问相交不能承诺线性复杂度。
- 旧`cube.task/vector.task`和公开builder/tracer已删除，不恢复兼容。一般动态ownership、
  continuous跨query状态、任意稀疏/分页/persistent task stream仍不支持。
- 默认dense f16之外，bf16/GQA曾有CANN资格，但没有完整设备资格；causal/tails/decode、
  MQA、paged/sparse/persistent等变体不能列为本轮已完成。

## 历史设备结果及暂停候选

最近完整设备证据保留在外层`runs/refactor/cv-order/`，CANN9.1/dav-2201/-O2。
历史BQ128小规模全输出误差≤3.052e-5；4K/8K同一binary采样误差≤1.526e-5，
全输出有限、输入和保护区通过。4K三批device task p50中位数605.780µs，ACLNN512.152µs，
仍慢18.3%，未达≤5%目标。此为历史结果，本轮不复测或优化设备算子。

BQ256/BM128 K/V复用候选仍暂停，尚无成功完整ASC/shared binary或设备结果。
`runs/refactor/l1-reuse/`保留五次拒绝：22.697/22.184/59.493/89.502/99.837秒，
分别涉及sequence endpoint、fixed child endpoint、parent guarded、child lifetime、
parent borrowing预算。未放宽预算重跑；请求预算520MiB不是设备收益。

外层runs清理保留了近期源码/ASC、CSV/性能摘要和失败诊断，旧二进制、缓存、profiler
trace已删除；历史目录不是可直接运行的完整build。详见外层`runs/README.md`。
