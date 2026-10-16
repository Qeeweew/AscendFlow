# 编译不变量与变换合同

这是新重构必须满足的合同，**不表示替代实现已完成**。现状只在[状态页](refactor-progress.md)登记，算法见[设计](design.md)。

## 1. 事实只由一个阶段建立

每个变换说明：输入前提、建立的保证、保持的保证、可能失效的保证。规则的正确性论证和实现回归做在开发阶段；每次编译只建立该输入必需的前提和新约束。

normal 不反复重建整个同步图，也不依赖外部可伪造的 verified 属性。输入由一次编译私有持有；外部物理 IR 和修改后的产物重新进入验证入口。不要先建设通用 proof DAG 或任意可变 IR 缓存框架。

| 不变量 | 唯一建立职责 | 下游如何使用 |
|---|---|---|
| 输入合法 | 前端/admission：类型、runtime合同、坐标、constexpr | launcher 检查需要运行时成立的前提；后续不重新解析源码 |
| effects 完整 | native lowering + 目标原语合同 | 访问包含 scratch、padding、SPR、读写及退休点 |
| alias/owner | 物理访问分析 | root、view、范围、owner 与 slot/generation 不能丢失 |
| 值与依赖 | 依赖分析 | read 的定义、RAW/WAR/WAW 与全部相关旧 reader |
| 排步固定 | pipeline lowering | actor-local step、logical generation、stage、issue order |
| 循环等价 | PLE 构造与边界组合 | 实例和词法次序不重不漏；零次 identity；回边正确 |
| 完成与可见性 | typed 协议构造 | 只消费已取得的 ready/release/capture 事实 |
| 通知资源合法 | 资源分配 | Set/Wait 配对、consume→reuse、bank/ID/width、quorum |
| 可进展及生命周期 | 协议/资源的进展合同 | 不等待未来无法派发的工作；seed/drain 与真实端点匹配 |
| 数值及性能政策 | 所有受控变换保持 | 无额外搬运/buffer/barrier，精度与固定排步不暗改 |
| 输出对应 | 物化和 emitter | 生成位置、guard、参与者、ID 与已确定计划一致 |

IRDL verifier 负责结构和类型，不能替代异步正确性。有限模型只覆盖指定规模和硬件假设，不能称任意 N 的形式化证明。

用户已明确不要求重构后的ASC文本、loop形态或event编号与旧版本相同。合法循环改写、有效域化简和冗余同步省略均可直接实施；验收看实际访问/数值合同、buffer复用、派发与重叠以及编译/设备证据。抽取原语时的逐字对照只是该局部迁移的检查手段，不能成为后续控制变换的限制。

## 2. PLE 与循环组合的证明义务

1. 对每个 stage，存在且只存在一次物理 step 执行每个合法逻辑代次。坐标替换作用于地址、guard、slot、metadata 和所有捕获值。
2. 长、短、零次路径互斥且覆盖输入域；FIRST/LAST 重合时不复制指令、不改变词法顺序。
3. entry→exit 透明传递不产生新的完成事实；region 控制退出不等于内部异步退休。
4. prologue建立已激活stage的合法进入状态；含首尾guard的一步转移保持carried合同。stage填充结束不代表距离大于stage深度的release已经无条件生效。N=S时主体零次重复是identity；短片段不能借用未建立的归纳假设。epilogue只执行真实剩余stage，导出精确未退休访问和通知状态。
5. sequence 的右项从左项真实出口接收状态，loop 回边使用同样的接线规则；子循环不因为被嵌套就换一种语义。
6. 同一 ring 的末代/首代和 event ordinal 使用各自真实时钟。owner、guard、slot 相关性不能通过分别取模后拼错。

正确性论证写在变换旁；normal 只验证实际适用前提并构造结果，不针对每个 N 再跑归纳检查。

## 3. 完成与同步的关键限制

- source SSA use-def 不证明异步完成；issue/queue/native end/累计退休/通知消费不同。
- 只在目标原语允许的累计前缀内比较进度；多个流水或参与者不可仅取一个最大序号。
- Set 采集合法 source capture 的事实；Set 自身的效果不使后续源操作自动退休。
- Wait 只在其真实 admission 后传递已消费通知的事实。CV controller Wait 的派发阻塞单独建模。
- release 捕获目标存储的全部相关最后 reader；GM import 完成可以释放 GM，即使 Vector 仍处理 UB。L1→L0 最后读取可以释放 L1，即使 MAD/FIX 未完成。
- 分支合流只承诺实际路径均成立的事实；缺席路径继承输入，不能拼接互斥路径的证据。
- 两个 AIV 的必需贡献必须齐全；OR 可达路径不能替代 AND quorum。
- 省略尚未插入的冗余同步直接使用已有覆盖。删除已存在同步须有不依赖它的替代 witness，并更新所涉 lease；不试删后全图重验。

## 4. 哪些变换使哪些事实失效

| 变换 | 保持 | 需要更新/重新建立 |
|---|---|---|
| 合法纯整数化简、同值地址表达替换 | 数值、地址与访问、同步关系 | 表达式类型/溢出/除法前提，新的支配位置 |
| 保序的 PLE 控制分解 | 逻辑执行实例、native effects、算法顺序 | 坐标和存在条件、边界/回边绑定 |
| fresh frame / subview / layout 改写 | 只有显式证明保持的项 | 若改变物理范围，更新所涉访问、依赖与资源合同 |
| stage/issue order/参与者变化 | 与调度无关的类型和数学定义 | 涉及的完成、进展、通知、生命周期 |
| 一一对应 ID 重命名 | 访问、逻辑通知、完成拓扑 | bank 合法性、碰撞、保留资源和实际 ID 映射 |
| event width/ordinal/共享成员变化 | 不变的 native effects | 相关配对、消费、lease冲突、初始/尾部状态和进展 |
| 新增 capture/acquire 协议 | 既有正向完成事实及未变访问 | 新协议自身合法性、资源与进展，受影响前沿 |
| 删除/移动 native 或改变 guard | 不能假定旧完成仍成立 | 受影响访问、路径贡献、配对及边界 |
| 输出 ASC | 已确定的执行计划 | 只做按合同物化；emitter 不重新选择同步 |

参数合法域缩小与删除操作的某些激活不是同一件事。后者可能移除必要的 Set 或贡献者，不能直接复用原保证。

## 5. event 状态与收尾

同一物理本地 flag 再 Set 前，前次 Wait 必须已消费。slot 可复用不自动证明 flag 可复用；flag 已消费也只证明它捕获的完成关系。跨核 counter 的容量、贡献者和可见性使用独立硬件合同。

逻辑通知跨 PLE 和外层迭代保持同一账本。不能给每段独立 seed/drain；不能为不存在的循环体消费者产生通知。N=0透明传递已有资源状态，外围资源自身的合法seed/cleanup或继承token收尾仍按各自生命周期执行。父子 event 借用保留区间、端点和共享约束，不默认改成独占 ID。

配置 API 自己可能占有 event，例如 AscendC TPipe 的隐式资源。手写参考与原生 ASC 的资源合同分别记录，不能当作无 effects 的分配助手。

## 6. 验证放在哪里

| 层 | 目的 |
|---|---|
| admission / normal | 输入前提一次、依赖一次、必要的新协议/资源/进展一次，消费受控变换结果 |
| 外部同步 IR | 从实际输入重新建立事实，不信任随文件提供的证明标记 |
| debug/deep | 独立检查最终 IR，定位错误变换或遗漏失效；耗时单列 |
| 回归与有限模型 | 检查变换规则和实现，破坏 IR 后发现错误，覆盖短循环/slot/分支等反例 |
| CANN / 设备 | 实际 ASC/硬件契约、数值及性能，不能由 CPU 模型替代 |

现有基于生产 native topology 的有限 BFS 是求解器差分，不是独立硬件语义 oracle。测试必须标出复用了哪些输入事实，见[验证计划](validation.md)。
