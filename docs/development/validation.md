# 编译器验证方法

本页定义 CPU/IR、ASC、CANN 离线编译以及设备数值和性能的验收口径。生成规则使用唯一 normal 链路，各层结果分别记录。

## 1. 分层验证

| 层 | 验证对象 | 能说明什么 |
|---|---|---|
| 输入/IR | 类型、source合同、effects、别名、owner、结构 | 当前输入满足分析前提；不是同步正确性证明 |
| 变换规则 | PLE实例映射、scope绑定、前沿更新、协议与资源合同 | 检查实现遵守已论证规则；normal 消费生成 credit 和保持的事实 |
| 语义反例 | 旧read覆盖、WAR/WAW、短MAD、少一个AIV、错event ID、错误guard | 必须生成所需同步或给具体诊断；不能只测试成功例 |
| 有限执行 | 小N、slot和分支下的实际指令/字节/event交错 | 有界范围正确；不外推任意N，不独立证明硬件前提 |
| 真实DSL集成 | examples→source/physical IR→ASC | 公共接口接通；不是CANN/设备资格 |
| CANN与设备 | 实际ASC、二进制、数值和profiling | 对该产物、配置与设备的实测资格 |

独立性的区别如下，后两类不能替代当前实际 IR 模型：

1. 从实际IR解释原生指令、字节访问和硬件event的独立实现，检查生成计划；仍依赖声明的硬件合同。
2. 读取生产native topology再独立BFS的有限差分，只独立检查求解器；不能发现双方共享的错误拓扑前提。
3. full/compact共享转移函数的表示差分，只验证表示等价；不能称独立语义oracle。

当前保留 `tests/models/structured_events.py` 的实际 IR 有限模型及独立 pre-sync 对照，旧 progress 图/BFS/Path 的表示测试已删除。`tests/support/structured.py` 是生产输入构造 helper，不能算独立 oracle。不能把测试总数当形式化证明。

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
| 动态别名 | 保留源码内存顺序与全部真实reader；不以barrier隐藏未知，不把may-write当初始化 |
| 并行与write_disjoint合同 | 独立有限模型发现具体输入下的重复写与跨group竞争；normal不重新证明算法全局唯一性 |

## 3. 测试组织和时间纪律

目标分组：frontend、dialects/effects、analysis/access、loops/PLE、sync、resources、backend、integration、models、cost。迁移现有反例，不为每个旧helper新建一套镜像测试。

先跑受影响的小组；同一不可变示例仅编译一次供多个破坏测试使用，修改IR各取独立副本。结构成本用独立新输入，不用fixture/cache掩盖开销。全量回归在稳定集成后做一次，失败后只重跑相关组。

任何单个算子构建≤300秒，覆盖Python构造、分析、ASC、CANN及子进程。超时终止整个所属进程组，保存阶段信息，停止后续候选。测试的外层预算不能用来反复自动重试超过限制的构建。

## 4. 专项编译复杂度与冷计时

先记录阶段工作量：AST/native数量、访问描述/相交、依赖、边界条款、guard/周期分量、最大前沿、比较/更新、lease冲突/候选、输出大小、实际重复遍历。每项明确统计口径；不能只比较最后图节点或cache命中率。

- runtime N不特化。固定政策下大N不增加静态工作；常量界也采用同一规则。
- 模板/slot/嵌套深度分别增长，排除benchmark同时改变多个维度。
- 默认FA只做一个probe定位；接近门槛后再做五次冷进程资格，不每轮重复FA。
- source→ASC包含源码读取/构造、所有分析/化简/输出和内部快照。import、进程总时间、CANN、RSS另列，不能移动计时边界美化数字。
- 五次p50≤5s、max≤10s；GEMM/scale/softmax/RMSNorm同时报告绝对值和是否回退。未达标就明确失败。

复现冷编译工具（从本仓库根目录、已激活的 Python 环境运行）：

```bash
python -m benchmarks.compile --input examples/flash_attention/attention.py --entry attention_target --mode normal --trials 1 --timeout 300 --output build/compile/fa-probe
python -m benchmarks.compile --input examples/flash_attention/attention.py --entry attention_target --mode normal --trials 5 --timeout 300 --output build/compile/fa-cold
```

冻结源码、编译器Git加未提交diff身份、constexpr政策、版本、ASC SHA、命令及原始JSON。单次probe和带采样诊断不作为五次p50；当前 normal 使用生成规则的域、phase 和 credit 前提；不再执行旧 progress/path solver。`generated-lifetimes` 仅描述来源，不能作为独立证明。normal与额外结构验证的成本分开。无需清空OS文件cache，不允许跨编译结果缓存。

## 5. CANN、设备与日常性能记录

先审查最终 ASC 的 load/store 范围、L1/L0/UB 容量与槽位、MAD/FIX、早 release、
CV 派发、事件映射和返回 drain。不能增加全局 barrier 或静默 fallback 换取通过。
CPU/IR 通过、CANN 编译通过和设备数值通过各自记录。

日常验证采用[测试指南](../guide/testing.md)的统一入口：16 个公开示例编译/运行，
仅 Flash Attention 和 GEMM normal 的固定 shape 矩阵，AscendFlow 与 ACLNN 每组各采集一次默认 msprof。记录实际 DSL/IR/ASC/库、
编译选项、输入、设备、预热/执行次数和原始 CSV；kernel time 采用设备 task 时长。
full 与 sampled 分开，ACL stream interval 不当作 kernel time。

运行前检查所有卡的健康、进程、显存与利用率，选择空闲健康卡，每进程显式映射物理
卡到逻辑 0。设备错误立即停止，不 reset、不换卡规避。输入声明由调用方满足，薄
launcher 不执行 shape/alias/对齐检查；非法输入不得发到设备测试 host 拒绝。

安装版 ACLNN 用于日常多 shape 对照；独立 AscendC、PMU 与理论下界用于专项诊断。比较需
保持数学、dtype/layout、输入和设备一致；参考政策不同时说明差异。PMU replay 组
不能相加分解时间，GM 请求不等于 HBM 实流量。日常 kernel time 报告不宣称完成
历史三批门槛、性能无损证明或冷编译资格。历史结果通过 Git 查阅。
