# DSL：运行时循环、物理存储与多级流水

本文说明现有公开语法和重构需保持的语言语义。**语法被识别、通过lowering、设备验证是不同资格。** 当前限制见下表，实际证据见[状态](refactor-progress.md)。

## 1. 用户负责表达什么

采用受限Python AST，形似TileLang；不在Python里执行设备循环来tracing迭代次数。用户表达计算、布局、缓冲、执行域和必要排步；同步、event ID和credit由编译器生成。

| 语法 | 合同 |
|---|---|
| `@T.prim_func` | 捕获源码、参数及位置 |
| `T.Tensor` / `T.int64` | 外部tensor与runtime标量；长度不是constexpr |
| `T.int64(minimum=1, multiple_of=B)` | runtime输入合同，launcher检查；不变成编译期常量 |
| `T.constexpr` | tile、dtype/layout、固定政策等显式特化参数 |
| `assert` | 只接受constexpr布尔条件；不是运行时事实假设，即使Python -O也执行编译检查 |
| `T.Kernel(groups, kind=...)` | launch域；mixed每组1 AIC/2 AIV |
| `T.Cube()` / `T.Vector() as lane` | actor域，具体engine由原语确定 |
| `T.serial(N)` | 源顺序循环，退出无隐式drain |
| `T.Pipelined(N, stages=S, issue_order=...)` | 有限stage政策；保留runtime循环 |
| `T.Stage(s, name=...)` | 当前pipeline的逻辑偏移，不是完成边界 |
| `alloc_ub/l1/l0a/l0b/l0c`、views | 显式物理容量、布局、slot与别名 |
| `copy/matmul/reduce/...` | typed操作，scratch和真实读写可见 |

stage数、buffer slot数、event width互不等同。`cube.task/vector.task`已经移除且不恢复兼容。内部physical Task IR不属于旧公开语法。

## 2. 双缓冲示例

完整当前源码：[scale.py](../examples/vector/scale.py)。核心结构如下：

```python
for t in T.Pipelined(tiles, stages=3):
    with T.Stage(0, name="load"):
        T.copy(X[t, :], x[t % 2, :])
    with T.Stage(1, name="compute"), T.Vector():
        T.mul(x[t % 2, :], 2.0, out=y[t % 2, :])
    with T.Stage(2, name="store"):
        T.copy(y[t % 2, :], Y[t, :])
```

同一产物接收不同tiles。MTE2→Vector和Vector→MTE3提供ready，Vector→下一次MTE2提供输入slot release，MTE3→下一次Vector提供输出slot release。输入的release不必等输出store完成。

pipeline在actor-local step j执行 `t=j-s`，仅在合法域内激活。issue_order决定同actor的派发次序，不按stage编号强行排序；它不构成跨actor完成关系。用户不给Set/Wait、初始token或event ID。

## 3. CV与嵌套语义

FA的真实数学主体在[attention.py](../examples/flash_attention/attention.py)。四个stage是QK、softmax、PV、output update；内部还有Cube分块和Vector行块pipeline。用户政策可令AIC先新QK后旧PV、AIV先新softmax后旧update。

不同stage可以共用物理存储，但各自逻辑代次、slot和状态必须准确。两个AIV对P的贡献和GM reader释放由编译器根据实际访问建立；不能把一个actor完成当成两者都完成。

数学块名称不是编译器内置attention API，也不是目前已有的通用函数宏。任意Python函数调用仍被拒绝。透明宏如后续加入，必须保留source位置、类型、scratch/effects、作用域和内部循环，不靠opaque Python回调隐藏原语；这不是本轮已经支持的能力。

serial、pipeline、嵌套及循环前后操作使用统一控制与region接口，语义见[PLE与边界组合](design.md)。各实际示例的当前资格见[状态页](refactor-progress.md)；设计示意本身不构成可运行API。

## 4. 当前识别与lowering边界

| 构造 | 当前情况 | 重构要求 |
|---|---|---|
| runtime局部pipeline、显式buffer、固定stage | 已有examples和旧资格 | 保持政策和原语语义，取消结构专用同步分派 |
| 普通常量界 | 默认保留循环；部分固定child分析仍展开访问 | 普通循环不自动unroll，分析与N无关 |
| serial中的标量carry | 前端可构造scf.for，相关lowering拒绝 | 在统一admission明确诊断；需要完整carry合同后才开放 |
| 外层serial + KV pipeline + fixed child | 现有FA路径，结构限制严格 | 改成普通递归组合，可有循环前后真实工作 |
| runtime child及更深任意嵌套 | 未普遍支持 | 逐合同接入，不设FA/query例外 |
| FIRST/LAST/受限边界guard | 已有子集，组合限制存在 | 空/短/重合/孔洞的统一存在域 |
| 数据相关分支、间接alias/owner、任意多时钟连续任务 | 未普遍支持 | 明确可验证输入合同或具体拒绝 |
| opaque Python函数/任意宏 | 不支持 | 透明AST复用另列实现，不以增加原语名单替代 |

重构把语法/admission与lowering能力对应起来；不能先接受一种构造，再以不相关的“需要KV/query”诊断拒绝。

## 5. 地址、alias与数值合同

source view保留root、原始轴、SSA坐标、宽度和父view约束。先检查每个轴，再转字节地址：扁平地址未越界不能授权`head=H`落入下一batch。subview/recast不创建独立allocation，不同名字不证明不别名。

runtime输入的minimum/multiple支持除数正性和整除推导；非法的原始除法不能因为后续乘0或相减化简而逃过admission。地址化简须说明数学整数、设备i64、溢出、除法和guard支配条件。保留现有host整数溢出检查。

cyclic任务按 `flat=group+task*GROUPS` 绑定owner；空owner没有任务。动态frame、prefix、mixed及父view组合按当前实现分别限制，不从单个grouped例推断全部支持。未知alias不可用任意barrier掩盖。

FA允许FP32 sum改变归约树，但P/O舍入、在线m/l/O更新及其他状态算子的数值边界仍须保持。同步pass不改变算法政策。`[:, None]`的广播意图不能偷偷分配整张tensor，现有Vector recipe的显式scratch读写必须保留。

## 6. 可读输出与诊断

纯标量的canonicalize/CSE应成为显式pass，effects操作不进入通用DCE/CSE。event ID的常量、取模和连续映射由等价变换化简，ASC输出有稳定命名与清晰的PLE；禁止regex改最终字符串或按算子shape特判。

错误需区分非法输入、当前不支持、资源不足、分析预算未决，并给阶段、源位置、相关操作/范围/代次/参与者。当前后端多处仍只有字符串CompileError，这是待重构项，不是已经完成的诊断系统。
