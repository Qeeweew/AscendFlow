# 重构的验证记录

2026-10-14 的[总体设计](design.md)、[DSL 规格](dsl-multistage.md)和
[重构计划](refactor-plan.md)已重写；它们规定后续目标，不能作为实现证明。
本次文档修订没有改运行时代码、运行编译回归或启动设备实验。算子调参暂停。

## 修订时的实现快照

修订开始时产品 HEAD 为 `7c5d0a9`，其后有未提交的 `loop_*`、前端/lowering/codegen、
benchmark 和回归修改。下面符号循环记录来自该未提交原型，不代表 HEAD 单独
包含这些能力，也不表示最新工作树已经完整复验。

| 能力 | 实际状态 |
|---|---|
| TileLang 风格作用域 | tracing 原型已能构造 typed source IR；新受限 AST 前端尚未实现 |
| 运行时循环 | 单个连续外层循环，至多一个 uniform runtime size；完整 native tile |
| 嵌套/索引 | 静态内层展开上限64，动态嵌套拒绝；runtime 轴映射受限 |
| slot/周期 | `versions` 隐式关联；全局 LCM 上限840及模板预算，尚未因子化 |
| 同步/验证 | 周期 frontier、guarded completion 图、整数势函数；最终重读 IR，但共享分析 |
| 资源 | 符号路径尚无完整跨 channel 生命周期共享，不能套用有限路径的支持声明 |
| 算法 | 算法/oracle/算子适配器已在 examples；FA 仍是遗留物理构造 |
| CV 示例 | QK→乘法/cast→PV，含分段例；没有 online softmax，不是 FA |
| 编译缓存 | 不可变产物可重复读取；不等于已完成按源码键的真正编译缓存 |

符号分析的新记录位于外层 `runs/refactor/loop-complexity/`：三个例子的 ASC
离线/shared 编译成功。Vector 同一 binary 已运行0/1/2/4/4096/65536 tiles，
最大8,388,608个 f32 元素，误差0且 guard 检查通过；CV 同一 binary 已运行
0/1/2/4/4096 tasks，误差0。**最新符号分段 CV 只有离线编译证据，未做设备验证。**
历史静态分段 CV 的设备结果见下节，不能移用为新路径的证据。

| 较早三次冷进程中位数，N=4096 / 1048576 | 编译秒数 | peak RSS 约值 |
|---|---|---|
| Vector | 0.0666 / 0.0665 | 27 MiB |
| CV | 1.603 / 1.599 | 154 MiB |
| 分段 CV | 3.346 / 3.373 | 217 MiB |

这些计时从已有 source IR 开始，排除 Python import、源码 tracing/AST 构造和
CANN；不满足新版完整前端计时口径。随后线性 prefix 图将分段图约35k边降到
3894边、2394节点，生成 ASC 保持不变；`linear/*/compilation.json` 的单次探索
结果分别为0.0808 / 1.328 / 2.592秒，不是正式多次验收或真实 FA 编译性能。

`regressions.log` 有线性改造前360项通过记录；后续局部测试、失败修复及图搜索
微调不等于当前工作树全回归已通过。最终搜索顺序微调需在 R0 重新验证。本轮
不把旧测试次数、设备 correctness 或 ACL stream interval 当作吞吐率达标证据。

## 历史静态分段 CV 设备证据

2026-10-14：2 groups、4 tasks、3 segments，f16 输入/f32 输出；CANN 9.1 /
dav-2201 离线/shared 编译成功。检查全部卡后物理卡1映射逻辑0，执行10次，
独立 CPU 参考最大绝对误差0。数值契约是
`sum_chunk RN_f16(2 * QK_chunk^T) @ V_chunk`，没有 softmax。
源码、IR、ASC、编译选项、健康记录和 hash 位于外层 `runs/refactor/segmented-cv/`。

该历史 ASC 中全部新 QK segment 在旧 `probabilities_ready` Wait 前发射，
内部预取先于上一个 segment 的 Cube；提前至首个 export 后发布完整 Score
的负例被拒绝。这验证特定排步/数值，没有测吞吐率，也未证明新版 AST/符号嵌套。

## 以下为早期重构记录

旧实现保存在 `06fce4d`，原计划保存在 `9b40e2f`。下列“下一步”“尚未完成”
描述对应当时阶段；当前边界以上述快照为准，实施顺序以新版计划为准。

初始固定 FA 输入 B1/H1/Sq128/Sk2048/D128、950个 task 的 cProfile 记录为：
AutoSync 100.03秒、再验证4.95秒、含重检的 emission 5.58秒、witness/JSON
9.53秒。约8292万次调用，完成图构造320次、闭包361次、`Graph.edge` 约2807万次。
输入 SHA256 为 `08656eaa89ab69f71748ed9b5ee108dbb89f9891334d55884dc90306df4062e8`，
数据在外层 `runs/refactor/cpu-baseline.log` / `.prof`。这是带 profiler 的单次
定位，不能与上表的无 profiler 时间直接计算加速比。

## 同步分析和 Python 开销

物理访问只枚举自身的精确切分区间。完成图用线性前缀表示此前 native retirement；
Notify effect 不成为后续 source task 的完成约束。候选规划复用完成 DAG，以具体
路径和受影响义务重查被删除的边；最后的独立 checker 仍从实际最终 IR 提取
全部 effects、ownership、event 生命周期和 drain，不读取规划路径缓存。

新增有限随机 DAG 删除差分、native 图规模、前缀捕获反例；本轮局部/物理扩展/
source/规划相关135个回归通过，CV和分段旧回归45个通过。范围是有限回归和
有界执行模型，不是任意规模形式化证明。

新 Compilation 接口拥有独立输入，返回不可变代码、IR字符串和检查摘要。
普通输出复用最后一次完整检查的解析结果；外部可变 IR 的公共 emission 和
verification 仍重新检查。返回的可编辑 plan 是新对象，不携带缓存证明状态。
逐访问 witnesses 是显式诊断选项，普通编译不导出它。

旧固定输入的规划耗时已由原带 profiler 的100秒级降到正常运行约2–14秒；这
两个口径不能直接计算加速比。输出格式化从重复验证中分离后，探索测量的
ASC formatting 为20–64毫秒。常用内核1秒、4K/8K FA5秒的验收仍未完成，
尤其8K旧物理流规划仍超时，需要继续依据新的 CPU profile 修正热点。
正式多次 CPU benchmark 和峰值RSS将单独记录；当前不得宣称编译性能已达标。

## 源语言闭环

静态三阶段 Vector、双向 CV、多 segment CV 已接通。28个 source 回归通过，
覆盖零/一/两/四次循环、内外 clock、最后结果的 slot、覆盖后的 borrowed tile、
同 root 的独立输入视图、提前 ready 和逻辑轴越界。
分段 CV 的历史 CANN 和设备证据见本页上方。它是正确性验证，没有计时结论。

算法/oracle/算子适配器已迁出核心，通用 CLI 和冷进程 benchmark 已建立。
精确区间 frontier 改为索引表并合并同源同目标的相邻区间；不跨孔洞合并。
198个 frontier/source/物理/CV 回归通过，另有逐字节有限差分验证。

用户指出：主编译路径不能按输入长度展开。现有有限代次规划仅保留为遗留回归，
不再通过缓存把它当作最终循环编译模型。下一步 source/physical IR 保留运行时
SSA 上界；以周期访问 frontier、带代次位移的完成路径及整数势函数证明循环。
有限实例化仅用于测试。

仍待完成：符号循环同步；全部算子迁新 DSL；核心独立安装测试；
AscendC可运行对照及 profiler/hardware 比较流程。
这些是未完成工作，当前提交不能作为整个重构完成的声明。
