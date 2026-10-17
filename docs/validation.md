# 验证与编译、设备性能验收

本页定义验收口径，不表示生成合同或normal路径已完成。当前算法开发已停止交接，结果与缺口见[状态](refactor-progress.md)，新方向见[计划](refactor-plan.md)。本次交接不执行算子构建或设备测试；下述命令供后续实现后使用。

## 1. 分层验证

| 层 | 验证对象 | 能说明什么 |
|---|---|---|
| 输入/IR | 类型、source合同、effects、别名、owner、结构 | 当前输入满足分析前提；不是同步正确性证明 |
| 变换规则 | PLE实例映射、scope绑定、前沿更新、协议与资源合同 | 检查实现遵守已论证规则；目标normal消费保持的保证，当前后置分析尚待替换 |
| 语义反例 | 旧read覆盖、WAR/WAW、短MAD、少一个AIV、错event ID、错误guard | 必须生成所需同步或给具体诊断；不能只测试成功例 |
| 有限执行 | 小N、slot和分支下的实际指令/字节/event交错 | 有界范围正确；不外推任意N，不独立证明硬件前提 |
| 真实DSL集成 | examples→source/physical IR→ASC | 公共接口接通；不是CANN/设备资格 |
| CANN与设备 | 实际ASC、二进制、数值和profiling | 对该产物、配置与设备的实测资格 |

区分三种现有“模型对照”：

1. 从实际IR解释原生指令、字节访问和硬件event的独立实现，检查生成计划；仍依赖声明的硬件合同。
2. 读取生产native topology再独立BFS的有限差分，只独立检查求解器；不能发现双方共享的错误拓扑前提。
3. full/compact共享转移函数的表示差分，只验证表示等价；不能称独立语义oracle。

当前保留 `tests/models/structured_events.py` 的实际IR有限模型；新增progress图/BFS测试主要检查停止时的后置分析实验，不作为下一代生成算法必须保留的实现接口。`tests/support/structured.py` 是生产输入构造 helper，不能算独立 oracle。不能把测试总数当形式化证明。

## 2. 最小反例矩阵

| 结构 | 关键要求 |
|---|---|
| 双缓冲load/compute/store | input release在最后Vector读；output release在MTE3读；后续无关工作不扩大等待 |
| loop后覆写UB | 控制退出不证明旧MTE3读完；不依赖统一loop drain |
| 子循环早释放L1 | 最后MTE1读取后可重填；MAD/FIX可继续 |
| shared L0C | FIX最后读与下一MAD写精确连接；短MAD不能假设全完成 |
| 多reader/无writer循环 | 所有旧reader由可信前缀覆盖；不能只取一个native.end |
| N=0/1、N<stage深度 | 无虚构Set/Wait，FIRST/LAST重合不重不漏，透明边不产生新完成 |
| 两相邻循环、嵌套带前后操作 | 同一边界组合，常量N也不隐式展开；不是只接受FA的结构 |
| guard孔洞/branch | 缺席分支保留输入状态；Set/Wait存在域及event ordinal一致 |
| 双AIV与CV | required quorum齐全，GM可见性真实，控制器Wait不形成派发死锁 |
| event跨段/跨调用复用 | 旧Wait消费先于新Set；配置占用、borrow、seed/drain保留 |
| 动态别名/ownership | 可验证输入合同或具体拒绝，不以barrier隐藏未知 |

## 3. 测试组织和时间纪律

目标分组：frontend、dialects/effects、analysis/access、loops/PLE、sync、resources、backend、integration、models、cost。迁移现有反例，不为每个旧helper新建一套镜像测试。

先跑受影响的小组；同一不可变示例仅编译一次供多个破坏测试使用，修改IR各取独立副本。结构成本用独立新输入，不用fixture/cache掩盖开销。全量回归在稳定集成后做一次，失败后只重跑相关组。

任何单个算子构建≤300秒，覆盖Python构造、分析、ASC、CANN及子进程。超时终止整个所属进程组，保存阶段信息，停止后续候选。测试的外层预算不能用来反复自动重试超过限制的构建。

## 4. 编译复杂度与冷计时

先记录阶段工作量：AST/native数量、访问描述/相交、依赖、边界条款、guard/周期分量、最大前沿、比较/更新、lease冲突/候选、输出大小、实际重复遍历。每项明确统计口径；不能只比较最后图节点或cache命中率。

- runtime N不特化。固定政策下大N不增加静态工作；常量界也采用同一规则。
- 模板/slot/嵌套深度分别增长，排除benchmark同时改变多个维度。
- 默认FA只做一个probe定位；接近门槛后再做五次冷进程资格，不每轮重复FA。
- source→ASC包含源码读取/构造、所有分析/化简/输出和内部快照。import、进程总时间、CANN、RSS另列，不能移动计时边界美化数字。
- 五次p50≤5s、max≤10s；GEMM/scale/softmax/RMSNorm同时报告绝对值和是否回退。未达标就明确失败。

复现当前工具（从工作区根目录运行，本轮未执行）：

```bash
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/flash_attention/attention.py --entry attention_target --mode normal --trials 1 --timeout 300 --output runs/structured-loops/fa-probe
PYTHONPATH=compiler .venv/bin/python -m benchmarks.compile --input compiler/examples/flash_attention/attention.py --entry attention_target --mode normal --trials 5 --timeout 300 --output runs/structured-loops/fa-cold
```

冻结源码、编译器Git加未提交diff身份、constexpr政策、版本、ASC SHA、命令及原始JSON。单次probe和带采样诊断不作为五次p50；当前normal仍含progress/lease等后置分析，必须记录这一事实；不能用owned-construction标签声称它们已被移除。normal与额外结构验证的成本分开。无需清空OS文件cache，不允许跨编译结果缓存。

## 5. ASC与设备性能

先直接审查最终ASC：实际load/store数量与范围、L1/L0/UB容量与slot、MAD/FIX组织、早release、CV派发、event映射和返回drain。PLE或变量命名变化可不逐字节相同，但须保留实际执行组织；不能多加buffer/barrier换编译速度。

CPU合同和CANN离线通过后，设备运行前检查所有卡的健康、进程、显存与利用率，选择健康空闲卡；每进程指定物理卡并映射逻辑0。不用/tmp锁，不reset或换卡绕过错误。

FA设备主例为4K/8K等能反映稳态的大规模；短、零、尾块另作正确性边界。明确full与sampled，不把6行采样称全输出验证。其他算子选择足以饱和设备的规模，固定参数，不无界搜索。

对照必须包含：

- 相同数学合同下的独立手写AscendC；政策不同时注明，而非声称相同排步。
- 安装版ACLNN，并结合对应ops-transformer源码分析组织差距。
- msprof设备task时间与可用硬件指标；host stream计时单列，不能只看利用率百分比猜瓶颈。
- 实际shape/dtype/核心数/运行频率下的FLOP、GM/L2搬运量和条件下界。带宽单位、有效吞吐、cache命中未知时明确标出，不能把请求流量当实测HBM带宽。

同时绑定源码、IR、ASC、二进制、编译选项、运行设备及频率、warmup/repeats和profile原始数据。历史产物删除后，保存报告可作历史线索，不能宣称仍可直接复现或获得当前资格。
