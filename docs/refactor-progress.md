# 重构的验证记录

本阶段停止算子调参。旧实现保存在 `06fce4d`，原计划保存在 `9b40e2f`。
本轮调整后的源语言使用 Pipelined / Stage / Cube / Vector，详见
[dsl-multistage.md](dsl-multistage.md)。以下记录只覆盖实际检查。

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
分段 CV 的 CANN 和设备证据见 multi-stage 文档。它是正确性验证，没有计时结论。

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
