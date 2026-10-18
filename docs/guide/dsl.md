# DSL：运行时循环、物理存储与多级流水

本文说明现有公开语法和语言语义。**语法被识别、通过lowering、设备验证是不同资格。** 当前限制见下表，验证方法见[validation.md](../development/validation.md)。

## 1. 用户负责表达什么

采用受限Python AST，形似TileLang；不在Python里执行设备循环来tracing迭代次数。用户表达计算、布局、缓冲、执行域和必要排步；同步、event ID和credit由编译器生成。

| 语法 | 合同 |
|---|---|
| `@T.prim_func` | 捕获源码、参数及位置 |
| `@T.inline` | 词法作用域的透明 AST 内联；不执行 Python helper |
| `T.Tensor` / `T.int64` | 外部tensor与runtime标量；长度不是constexpr |
| `T.int64(minimum=1, multiple_of=B)` | 调用方满足的runtime输入合同；不变成编译期常量，不生成launcher检查 |
| `T.constexpr` | tile、dtype/layout、固定政策等显式特化参数 |
| `assert` | 只接受constexpr布尔条件；不是运行时事实假设，即使Python -O也执行编译检查 |
| `T.Kernel(groups, kind=...)` | launch域；mixed每组1 AIC/2 AIV |
| `T.Cube()` / `T.Vector() as lane` | actor域，具体engine由原语确定 |
| `T.serial(N)` | 源顺序循环，退出无隐式drain |
| `T.tile_ranges(E, B)` | constexpr 整块/尾块区间 `(base, count, valid)`，最多两份 body；省略空区间 |
| `T.Pipelined(N, stages=S, issue_order=...)` | 有限stage政策；保留runtime循环 |
| `T.Stage(s, name=...)` | 当前pipeline的逻辑偏移，不是完成边界 |
| `alloc_ub/l1/l0a/l0b/l0c`、views | 显式物理容量、布局、slot与别名 |
| `copy/matmul/reduce/...` | typed操作，scratch和真实读写可见 |

stage数、buffer slot数、event width互不等同。`cube.task/vector.task`已经移除且不恢复兼容。内部physical Task IR不属于旧公开语法。

## 2. 双缓冲示例

完整当前源码：[scale.py](../../examples/vector/scale.py)。核心结构如下：

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

只在归约最后一块写回的算子，在 compute stage 末尾保留 LAST 条件并写回，
不为这一次写回单设 export stage。MAD→FIX、Vector→MTE3 及输出存储下一次
覆盖前的最后读取，仍由实际 effects 推导同步。stage 不是硬件 pipe 或完成边界。

## 3. CV与嵌套语义

FA的真实数学主体在[attention.py](../../examples/flash_attention/attention.py)。四个stage是QK、softmax、PV、output update；内部还有Cube分块和Vector行块pipeline。用户政策可令AIC先新QK后旧PV、AIV先新softmax后旧update。

不同stage可以共用物理存储，但各自逻辑代次、slot和状态必须准确。两个AIV对P的贡献和GM reader释放由编译器根据实际访问建立；不能把一个actor完成当成两者都完成。

数学块名称不是编译器内置 attention API。`@T.inline` 已支持透明 DSL helper，
参数接收标量、view 或 tuple，允许一个末尾 return；没有提前 return、变长参数或
Python 回调。helper 捕获自身的不可变常量和其他 DSL helper，局部名字不会读写
调用方的局部变量。内部原语、view、循环及源位置在正常 AST 链路中展开，访问、
scratch/effects、原式定义性、ownership 和同步合同不变。内联共用 10000 条展开
预算，调用深度最多 16；任意 Python 函数调用仍拒绝。显式文本可用
`T.from_source(..., functions={"helper": captured_helper})` 绑定捕获后的 helper。

`T.tile_ranges(259, 128)` 依次绑定 `(0, 2, 128)` 和 `(256, 1, 3)`。
用户再对 count 写 `T.serial` 或 `T.Pipelined`，设备循环仍保留，未按每个 tile 展开。
extent/width 必须是 constexpr 整数，extent 非负、width 为正；只拆分静态有效宽度，
没有声明输入天然在界内或绕过 native 对齐要求。

serial、pipeline、嵌套及循环前后操作使用统一控制与region接口，语义见[循环与边界组合](../development/design.md)。示例源码列于[examples](../../examples/README.md)；设计示意本身不构成可运行API。

## 4. 当前识别与lowering边界

| 构造 | 当前情况 | 语义要求 |
|---|---|---|
| runtime局部pipeline、显式buffer、固定stage | 已实现，使用统一编译链 | 保持政策和原语语义，使用通用同步规则 |
| 普通常量界 | 保留循环；只分析受物理容量约束的 slot 周期 | 普通循环不自动unroll，分析与N无关 |
| serial中的标量carry | 保留 SCF carry；影响地址/循环界而无范围合同时拒绝 | 在统一admission明确诊断；需要完整carry合同后才开放 |
| 外层serial + KV pipeline + fixed child | 默认 FA 使用递归生成规则；不设 query/KV 分派 | 普通递归组合，保留循环前后真实工作 |
| runtime child及更深任意嵌套 | 未普遍支持 | 逐合同接入，不设FA/query例外 |
| FIRST/LAST/受限边界guard | 已有子集，组合限制存在 | 空/短/重合/孔洞的统一存在域 |
| 数据相关分支、间接alias/owner、任意多时钟连续任务 | 未普遍支持 | 明确可验证输入合同或具体拒绝 |
| prefix native 地址、循环内动态 allocation | 前端可表达，native lowering 明确拒绝 | 需要独立访问/生命周期合同后开放 |
| 透明 `@T.inline` helper / 静态尾块区间 | 已实现 | 同一 AST、native lowering、访问检查和同步链；不引入设备 call |
| opaque Python函数/任意宏 | 不支持 | 不以 host 执行隐藏原语或 effects |

语法/admission与lowering能力应对应；不能先接受一种构造，再以不相关的“需要KV/query”诊断拒绝。

## 5. 地址、alias与数值合同

source view保留root、原始轴、SSA坐标、宽度和父view约束。先检查每个轴，再转字节地址：扁平地址未越界不能授权`head=H`落入下一batch。subview/recast不创建独立allocation，不同名字不证明不别名。

runtime输入的minimum/multiple支持除数正性和整除推导；非法的原始除法不能因为后续乘0或相减化简而逃过admission。地址化简须说明数学整数、设备i64、溢出、除法和guard支配条件。调用方满足合法 int64 索引求值前提，launcher 不检查运行时溢出。

cyclic任务按 `flat=group+task*GROUPS` 绑定owner；空owner没有任务。动态 frame、valid prefix、mixed 及父 view 组合按当前实现分别限制，不从单个grouped例推断全部支持。未知alias不可用任意barrier掩盖。

FA允许FP32 sum改变归约树，但P/O舍入、在线m/l/O更新及其他状态算子的数值边界仍须保持。同步pass不改变算法政策。`[:, None]`的广播意图不能偷偷分配整张tensor，现有Vector recipe的显式scratch读写必须保留。

## 6. 可读输出与诊断

纯标量的canonicalize/CSE应成为显式pass，effects操作不进入通用DCE/CSE。event ID的常量、取模和连续映射由等价变换化简，ASC输出有稳定命名与清晰的PLE；禁止regex改最终字符串或按算子shape特判。

错误需区分非法输入、当前不支持、资源不足、分析预算未决，并给阶段、源位置、相关操作/范围/代次/参与者。当前后端多处仍只有字符串CompileError，这是待重构项，不是已经完成的诊断系统。
