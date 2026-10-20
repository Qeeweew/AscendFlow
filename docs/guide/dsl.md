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
| `T.i64` | GM tensor 的 int64 元素类型；与 runtime 参数声明 `T.int64` 职责不同 |
| `T.load_index(view, minimum=..., maximum=...)` | 只读 GM 单个 int64 的 S 读取，返回 index SSA；inclusive 值合同不生成 clamp |
| `T.load_scalar(view)` | f32 UB 或浮点 GM 单元素的 S 读取，返回 f32 SSA |
| `T.store_scalar(value, out=view)` | 将已读取的 f32 SSA 写入单个 f32 UB 元素，保留真实 4 B S 写入效果 |
| `T.broadcast(v[None,:], out=...)` / `v[:,None]` | 显式 f32 行/列展开；行展开需要 `(rows,8)` scratch |
| `T.mul(k[None,:], v[:,None], out=..., scratch=...)` | 直接外积，保留列向量和 BRCB 行值的真实访问，不隐式复制完整列矩阵 |
| `T.constexpr` | tile、dtype/layout、固定政策等显式特化参数 |
| `assert` | 只接受constexpr布尔条件；不是运行时事实假设，即使Python -O也执行编译检查 |
| `T.Kernel(groups, kind=...)` | launch域；mixed每组1 AIC/2 AIV |
| `T.Cube()` / `T.Vector() as lane` | actor域，具体engine由原语确定 |
| `T.serial(N)` | 源顺序循环，退出无隐式drain |
| `T.tile_ranges(E, B)` | constexpr 整块/尾块区间 `(base, count, valid)`，最多两份 body；省略空区间 |
| `T.Dataflow(count, issue_order=...)` | 共享逻辑迭代域；允许 C/V 代码交替 |
| `T.Pipelined(count, issue_order=...)` | 单 actor 迭代与本地流水；降低到同一逻辑域表示 |
| `T.Stage(offset, name=...)` | 最近迭代域内当前 actor 的发射偏移，不是完成边界 |
| `alloc_ub/l1/l0a/l0b/l0c`、views | 显式物理容量、布局、slot与别名 |
| `copy/matmul/reduce/...` | typed操作，scratch和真实读写可见 |
| `copy(..., valid_rows=r)` | 二维矩阵的 runtime 有效行前缀；支持零填充 GM→L1 和部分 UB→GM |
| `alloc_mask(shape)` | 显式 UB packed predicate 存储，64 lane 一个 32 B 块 |
| `arange(n)`、比较、`& \| ~` | 精确整数逻辑 lane 坐标与带shape谓词，支持singleton轴广播 |
| `compare(x, threshold, kind=..., out=...)` | f32逐lane比较写入packed predicate；结果绑定该次实际读写 |
| `materialize/mask_and/mask_or/mask_not(..., out=...)` | 显式物化与packed位布尔组合；mask_and/mask_or/mask_not的out需独立存储 |
| `where(p, x, y, out=...)` | 两路均已定义的完整选择，不短路非法load |
| `fill/mul/add/sub/div/maximum/exp/sqrt/rsqrt(..., mask=p)` | 显式谓词更新；mask=False lane保留目标旧值 |

本地 stage 偏移、buffer slot 数、event width 互不等同。没有 `stages=` 声明。`cube.task/vector.task`已经移除且不恢复兼容。内部physical Task IR不属于旧公开语法。

## 2. 双缓冲示例

完整当前源码：[scale.py](../../examples/vector/scale.py)。核心结构如下：

```python
for t in T.Pipelined(tiles):
    with T.Stage(0, name="load"):
        T.copy(X[t, :], x[t % 2, :])
    with T.Vector(), T.Stage(1, name="compute"):
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

mixed kernel 先进入 actor，再声明 Stage；一处 Stage 不能包住两种 actor。
单 actor Kernel 或外层 Cube/Vector scope 提供隐式 actor。Stage 绑定最近的
Dataflow/Pipelined；不能跨 serial、分支或另一个 Stage 绑定外层迭代。

```python
for task in T.Dataflow(count):
    with T.Cube(), T.Stage(0, name="load"):
        import_inputs(...)
    with T.Cube(), T.Stage(1, name="products"):
        compute_products(...)
    with T.Vector() as lane, T.Stage(0, name="import"):
        import_products(...)
    with T.Vector(), T.Stage(1, name="merge"):
        merge_products(...)
```

上例是排步示意；完整输入及 helper 见
[KDA output](../../examples/kda/prefill_output.py)。同一个 task 绑定同代数据，
Cube@1 → Vector@0 由真实 workspace 访问推导同步，不比较裸 offset 大小。
每个 actor 有自己的 clock 和本地 horizon；两个 AIV 仍共用一份模板。

`Dataflow.issue_order` 是 `{"cube": ("load", "products"), ...}` 映射，
`Pipelined.issue_order` 是当前 actor 的名字 tuple。默认保持该 actor 的源码顺序，
不按 offset 排序。显式政策须完整覆盖该 actor 的实际块；未命名或隐式 offset 0
块不能通过猜测生成名字重排。名字在 actor 和迭代域内解析，两侧或父子域可同名。
偏移为 0..32 的 constexpr，count 保留 int64 范围、定义性与 uniformity 检查。
Dataflow 不声明迭代独立：状态、归约、最后 reader 和跨代复用仍按实际访问保持。
纯任务坐标可以写在 Stage 外；S 读取等 effects 保留实际定义位置。


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
| runtime child及更深任意嵌套 | 已有 clamped 正长度、bounded read prefix 子集，未普遍支持 | 保留实际 child 次数及 LAST；父代次内条件激活仍可能明确拒绝 |
| FIRST/LAST/受限边界guard | 已有子集，组合限制存在 | 空/短/重合/孔洞的统一存在域 |
| 可能重叠的符号地址、错位 ring 访问 | 按源码内存顺序处理 | 不虚构精确 producer 或完整覆盖；仍须满足现有控制与事件规则 |
| 数据相关分支、间接alias/owner、任意多时钟连续任务 | 未普遍支持 | 明确可验证输入合同或具体拒绝 |
| 二维 `copy(valid_rows=...)` | 零填充 GM→L1、部分 UB→GM 已实现 | 静态容量、真实行数、补零写入及 DMA 几何分别检查 |
| 一维 `valid=` prefix、循环内动态 allocation | 前端可表达，native lowering 明确拒绝 | 需要独立访问/生命周期合同后开放 |
| 透明 `@T.inline` helper / 静态尾块区间 | 已实现 | 同一 AST、native lowering、访问检查和同步链；不引入设备 call |
| opaque Python函数/任意宏 | 不支持 | 不以 host 执行隐藏原语或 effects |

语法/admission与lowering能力应对应；不能先接受一种构造，再以不相关的“需要KV/query”诊断拒绝。

## 5. 地址、alias与数值合同

source view保留root、原始轴、SSA坐标、宽度和父view约束。先检查每个轴，再转字节地址：扁平地址未越界不能授权`head=H`落入下一batch。subview/recast不创建独立allocation，不同名字不证明不别名。

普通 Kernel 的 group 独立性属于源程序并行合同，编译器不再证明任意任务映射
的全局唯一性。Vector peers 共用 body、私有 UB，普通内存访问不能依赖未声明
的跨 peer 通信；AIC/AIV 共享 GM 的双参与者交接仍自动同步。
Tensor 默认 `access="read"`，入口已初始化且禁止写；`write` 表示入口未初始化，
可以读取本 kernel 已定义区域；`read_write` 允许更新已有数据。
外部输出可声明 `N.Tensor(shape,dtype,access="write",write_disjoint=True)`，
承诺一次 launch 内所有实际写入互不重叠，包括 group/SID/迭代。它不授权输入
别名、scratch 复用或 reader 提前退休。声明由算法和调用方满足，当前只对没有
内部 reader 的输出消除写间依赖。详见[内存合同](../development/memory-contracts.md)。

runtime输入的minimum/multiple支持除数正性和整除推导；非法的原始除法不能因为后续乘0或相减化简而逃过admission。地址化简须说明数学整数、设备i64、溢出、除法和guard支配条件。调用方满足合法 int64 索引求值前提，launcher 不检查运行时溢出。

二维矩阵尾块用静态 tile 容量与实际有效行数分别表达。例如 `length` 为 runtime
长度，`base` 是当前块起点，`x/y` 是固定128行的本地 tile：

```python
rows = N.min(128, N.max(0, length - base))
N.copy(X[base:base + 128, :], x, valid_rows=rows, pad_value=0)
# 使用完整、已经补零的 L1 tile 计算；y 为已定义的 UB 输出。
N.copy(y, Y[base:base + 128, :], valid_rows=rows)
```

`valid_rows` 只选择二维 view 第一轴的连续前缀，不改变物理计算 tile 的形状。
行数须证明在 `[0, 静态行容量]` 内，真实非空访问仍检查原始 tensor 各轴。
GM→L1 支持 f16/bf16，必须显式 `pad_value=0`：整个 NZ 目标有定义，缺失的
行和扩展列为零；0行只填零、不读取 GM，行列均完整的 tile 不发额外填零。
UB→GM 不接受 `pad_value`，只读取/写出有效行；0行不发 DMA，目标剩余字节
保持原值。这一路仍要求普通 DMA 的32 B行宽、地址和步长合同，不能据此
支持任意列尾块。转置搬运、其他内存路由及一维 `valid=` 不取得该支持。
补零 packet 的内部 MTE2 fence 保持 fill→ND2NZ 顺序，外部 ready 和最后 reader
仍由同一自动同步链生成。输入不需要额外 host padding 或 packing kernel。

`load_index` 的值范围可引用先前支配该读取的 index SSA；范围只对实际读取后的词法域有效，
不是全局 host 参数，也不证明不同元数据取值互不相等。元数据在 kernel 内必须不可变。
S 读取使用既有有序执行语义；条件 reader、异步 producer 和最后 reader 的完成仍由公共同步规则处理。
S store→S load 按已证明的发射顺序执行，跨循环端口也不生成硬件不支持的
`PIPE_S` barrier。Scalar 写入 UB 后交给 Vector/MTE3 仍需实际的跨管线发布；
不能用 Scalar 读的同步返回合同替代写入完成。

`N.matmul` 的 L0A/L0B 输入支持 FP16、BF16 和 FP32，累加器为 FP32。
FP32 路径设置 IEEE 模式并建立 S→M ready，不先将操作数转成 b16/HF32。
FP32 L1 NZ 每列组为 8 个元素；FP32 L0C 仍使用 16 列 fractal。
当前 Load3D 配置在 kernel 入口保持不变，不支持一个 kernel 中任意切换 resident width。
GM→L1、L1→L0 的实际 tile、转置和 FIX 写回仍满足各自 native 几何及访问合同。
`mul/sub/div` 的行广播支持 8 的倍数且小于 64 的窄行，以及既有完整 64 列块；
窄行按精确列数设置 Normal mask，行起点保持 32 B 对齐，scratch 仍为 `(rows,8)`。

`i < min(a,b)` 保留对两边的范围限制；`max` 的已知下限和余数的数学区间不豁免
原始 unsigned 除法的定义性。原始轴 tile 的 Euclidean digit 排列及统一循环平移可
作为可选 streaming 分离坐标，仍检查实际轴、字节几何和当前作用域内的 digit 范围。
无法恢复分离坐标时保留可能别名的源码内存顺序，不将未知当作无冲突。

谓词是通用能力，不是三角或attention特化。`N.arange(n)` 产生逻辑 lane 坐标
（精确整数，不分配索引表）；比较与 `& | ~` 组合出带shape谓词，比较和布尔组合按
NumPy尾部对齐规则广播singleton轴。区间、窗口、尾部、周期（`%`/`//`，要求
不变量lane表达式）和布尔组合都用同一形式；坐标差/和用有界128位精确整数求值，
极端int64起点不溢出、不shift-by-64。没有为三角、带状或任何shape增加识别分支。

`mask=`的语义是更新集合：谓词为真的lane参与计算，其余lane保留目标旧值。因此
第一版masked update要求目标view已完整初始化；部分写不构成初始化，仍由ordinary
initialization coverage检查。坐标谓词直接使用Normal repeat bitmap，不需要存储；
`alloc_mask`显式分配packed数据谓词存储（每64个f32 lane一个32 B块，bit lane存放，
不是每lane一字节）。`compare`是一次真实读写，结果绑定该次写入，不会在其源tile
更新后延迟重算。`where`要求显式materialized谓词且两路输入均已定义，提供完整输出，
不把无效地址变成合法load，不做短路。

坐标谓词在目标每64 lane计算局部bitmap：全0块跳过native（绝不发出空mask），连续
全1块合并repeat，部分块设置自己的bitmap；不同行/块不共用一张tile bitmap。
数据谓词使用packed compare、packet局部CMPMASK与`vsel`，并在`set_cmpmask`后保留
已审计的局部`PIPE_V`。`count_end`只恢复Normal模式，不恢复mask位值；每个packet
自己配置所需状态，不消费incoming mask。packed谓词块是每64个f32 lane一个32 B块：
只有低8字节是逻辑lane位，其余24字节是padding；padding字节有定义但不被当作额外lane
解释。padding由单独的`mask_zero` native定义：它是一个真实的count-mode Vector
`Dup(0)`，覆盖声明中的整个u8块，并按ordinary RAW/WAR/WAW与compare形成required
same-V RAW/WAW，由通用同步生成实际的`PIPE_V` fence，而不是emitter在task内插入
scalar清零或全局barrier。`compare_*`保持audited mask-free Vector packet，只发出
`vcmpv/vcmpvs`调用；其目的view声明为保守的整块preserve update（`RW`）：`vcmpv`
只写低8字节lane word并保留另24字节padding，`R`表示保留旧padding并让“整块已定义”
由真实的前置`mask_zero` writer提供，不代表硬件读取padding，也不让单独一次compare
凭空产生完整初始化事实。因此未初始化packed块上的standalone compare会被ordinary
read admission拒绝。`mask_not`是真实的count-mode Vector `vnot`（`asc_not`），其
packet与实际调用序列一致，输出整块（包括反转后的padding）都有定义。因此
`mask_and/mask_or`、CMPMASK加载和谓词存储复用看到的都是完整定义的块；这些引擎、
effects和字节几何全部在同步分析前可见，不是emitter隐藏的memory effect。
coordinate/affine或lane-scan word recipe、以及“连续全1块最多255 repeat合并”的政策
在lowering选择并记录到native task事实中，emission只重放该选择，不在分析后另选实现。
对affine形式，emission在同一份program DAG上为每个value节点发出当前runtime标量
原子与编译期per-axis lane系数决定的精确有符号128位区间，再按(全真,全假)保守三元
组合：运行时全真直接走全1 bitmap的repeat运行，全假不发任何element调用，未能证明
（或lane系数相关）时进入同一段逐块重放。这不是整程序证明，也不保存运行时数组；
`//`/`%`等非affine节点不产生该运行时区间，lane64 recipe保持原逐lane重放。
单元lane系数±1的word由有符号128位阈值区间一次构造，先把阈值钳到可移位的范围内
再做int64转换，其余常数系数保留二分搜索回退；两种路径都做同一个精确128位求值。
若被admit的谓词是单个比较（可含shape-only BROADCAST）且差异的最后一个轴lane系数为
±1、最后轴为整64，lowering还会选择`mask_iteration=rows`：每个逻辑行用当前runtime
标量原子与其余轴仿射项算出精确128位阈值，把选中列表达为裁剪后的prefix/suffix（`eq`
为单点、`ne`为补点两段），每段最多一个leading partial bitmap、≤255 repeat的full运行
和一个trailing partial bitmap，同一64-lane块只写一次；`-1`系数按取负+flip比较种类
处理。布尔组合、非unit/periodic谓词保留`chunks`重放，不建覆盖证明状态。
`where`/masked data update的`scratch`必须是显式同shape f32 UB，且不能与目标同一
allocation（包括同一buffer的另一view），否则完整scratch写会破坏保留的false lane。
`copy`/DMA与reduction不因同名`mask=`自动获得同一语义；masked reduction、DMA mask和
masked cast/row-broadcast当前明确拒绝。masked谓词存储和scratch都是真实UB内存，
参与普通RAW/WAR/WAW、初始化覆盖、最后reader和事件资源检查；不使用全局barrier或
静默fallback。

坐标整数合同：比较值用有符号128位精确求值；lane系数的绝对值不超过2^61，从而
两个系数差不超过2^63且`int64`常量与取负都有定义；`//`/`%`只接受无scalar原子的
lane表达式和`1..2^63-1`的正int64除数，emission使用floor语义helper，与Python
`//`/`%`一致；INT64_MIN常量按`(-9223372036854775807LL - 1)`发出。比较两侧的
精确值幅度和受2^126限制，所有中间差、128位加乘和回退搜索都在该范围内，超出时
在source/admission明确拒绝而不是截断或发出非法字面量。极端int64常量起点（包括
INT64_MIN）与runtime非负int64 origin都保持精确。

完整示例见[masks.py](../../examples/vector/masks.py)（interval、相邻64 lane不同的periodic、
两者的XOR与非平凡runtime origin `(cols+q)>rows`、全/空/部分、`q=INT64_MAX`、
更新/未更新旧值保留、count op穿插、compare/where/not/or/and、源覆盖后继续使用早期
compare），设备oracle是独立NumPy/Python逻辑；runner只复用带manifest/source快照
校验的已发布产物，`--build`才调用编译器与CANN。

cyclic任务按 `flat=group+task*GROUPS` 绑定owner；空owner没有任务。动态 frame、valid prefix、mixed 及父 view 组合按当前实现分别限制，不从单个grouped例推断全部支持。未知alias不可用任意barrier掩盖。

FA允许FP32 sum改变归约树，但P/O舍入、在线m/l/O更新及其他状态算子的数值边界仍须保持。同步pass不改变算法政策。`[:, None]`的广播意图不能偷偷分配整张tensor，现有Vector recipe的显式scratch读写必须保留。

## 6. 可读输出与诊断

纯标量的canonicalize/CSE应成为显式pass，effects操作不进入通用DCE/CSE。event ID的常量、取模和连续映射由等价变换化简，ASC输出有稳定命名与清晰的PLE；禁止regex改最终字符串或按算子shape特判。

错误需区分非法输入、当前不支持、资源不足、分析预算未决，并给阶段、源位置、相关操作/范围/代次/参与者。当前后端多处仍只有字符串CompileError，这是待重构项，不是已经完成的诊断系统。
