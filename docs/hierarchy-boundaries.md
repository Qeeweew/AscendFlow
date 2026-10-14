# R2a：父子源码与调用边界分析

这是R2a的实现进展，尚不是完整父子流水闭环。新增
[hierarchical_gemm](../examples/ast_hierarchical_gemm.py)保留runtime `outputs`父serial
和runtime `panels`子Pipelined，使用真实的首清末导计算主体。当前能生成结构化
源IR；physical lowering仍明确拒绝父scf.for，不能编译设备代码或声称已支持嵌套。

## 源码地址不再只有一个动态轴

Tensor可声明多个独立runtime extent；片上tile仍为静态形状。外部A的坐标为
`[group, output, panel, M, BK]`，原生tile地址精确为：

```text
(((group * outputs) + output) * panels + panel) * (M * BK * sizeof(dtype))
```

这是地址表达式，父子控制和generation依然分别存在。步长中的uniform乘积
在program参数scope形成共享arith.muli DAG，A/B共用outputs×panels。
source view保留坐标、SSA步长和allocation root，不按长度生成Python实例。
默认示例有59个source operations，与两个runtime值无关。

动态轴要求对应零起点、unit-step的serial/pipeline坐标；错上界、跨轴使用、
非uniform步长因子拒绝。native tile内部动态pitch尚未接通，需先选中动态轴。
多维extent的物理ABI、乘法溢出检查和父GM ownership还要随下一步lowering实现；
当前不能仅凭source verifier就允许launch。参数化乘积最多8个动态因子。

## 精确内存frontier，而非整个child的done

[region_summary.py](../src/xdsl_npu/region_summary.py)对明确选择、跨调用复用的
stationary物理storage导出每个精确byte cell的：

- 首次写之前的reader集合，以及首次writer；
- 最后writer，以及它之后仍需退休的reader集合；
- 条件所属的child长度域和相对首/末代位置。

这些是内存义务，未证明硬件完成路径。旧调用的最后writer到下一调用首次写前
的readers产生RAW；到首次writer产生WAW；旧pending readers到首次writer产生WAR。
所有位置相对于child自己的调用，不能当作parent坐标。

默认K子流水由此导出的真实复用关系包括：

```text
previous L1 slot's MTE1 read_retired -> next GM-to-L1 write
previous L0 slot's MAD read_retired  -> next L1-to-L0 write
previous final FIX read_retired     -> next first MAD clear
```

不同root/slot分别处理，L1不被迫等到FIX。非连续视图的holes不会被补成访问；
同一小区域同时被Scalar和MTE3读取时，两者均保留，不能只取词法最后reader。
若一个cell全程无周期writer，reader保留首到末的等差族，要求整个族退休。
末代writer只改变出口，不能替稳态第一读提供初始化。

当前摘要限定单local actor、stationary且无prefix clipping的根。GM stream frame、
长度相关base、动态row pitch、mixed participants明确拒绝，等待父ownership及
多参与者组合。调用者必须显式选择分析根，不能把局部摘要当作整个程序的证明。
重复调用共用同一immutable child bound；每个parent使用不同长度需要另一个域模型。

## 长度域及复杂度

对storage周期P，有限分类为：N=1..P−1，以及P个 `N>=P && N%P=r` 域；N=0
是没有访问的identity transfer。周期writer的出口只需最后一个周期的访问字；
无周期writer时保留完整reader族。静态child长度仅保留真实可达域，禁止重新绑定。

设每个cell在一个周期内有E个实际访问，边界构造按 `O(P*E + P*log(E+1))`
计费：旋转非空访问字，并用二分定位，避免为每个余数再枚举P个空generation。
不枚举N或outputs×panels。5/7槽仍分别采用5、7周期，不形成35周期摘要。
这只是边界步骤的成本，不是整个编译器线性复杂度声明。

边界访问预算100万，在分配domain前检查；物理访问分析另增加100万fragment
预算，在生成row×residue集合前检查。仅计view数量不足以拦住4095行×840周期
的膨胀，回归确认该例在调用Ref.segments前诊断退出。预算超限不改变tile或同步。

## Event边界与memory边界分别保留

从实际Set/Wait读取endpoint、guard、bank和ID，保留：

```text
old invocation's last Wait consumption for ID i
    -> new invocation's first Set effect for ID i
```

普通channel有guard周期p、每周期k次激活、w个ID，其ID时钟周期为
`p*w/gcd(k,w)`。每个ID首次Set和末次Wait用start/end相对位置及精确激活下界
表示；末代channel用末代等式。零次调用无通知，也没有虚构的初始credit。
Wait消费不代表其后native consumer读完buffer，不能用这个摘要替代WAR。

默认子流水有43个cell/domain记录、70个跨调用memory义务、45个event复用域；
它们不是最终同步指令数。父层仍需证明覆盖、消除冗余并物化同步，不能照单
插入所有候选，也不能只因child单独验证成功就复用其ID。

## 验证与复现

独立测试直接执行小域的逐字节地址与R/W前沿，不使用生产access_cells或
loop_dependencies生成答案，覆盖0/1/2/3/4/5/6/7/11/17等长度、多reader、
holes、末代写及两次调用的RAW/WAR/WAW。event轮换另与N=0..79的直接Set/Wait
序列对照；百万级边界仅算参数位置。有限对照不冒充任意N形式化证明。

最终相关回归256项通过（67.39秒），ruff及diff检查通过。单层K子程序5个独立
冷进程，完整源码→ASC p50=0.2376秒、max=0.2472秒、peak RSS37716 KiB；这是
既有路径的回归测量，**不是父流水编译性能**。最终ASC仍与此前已profiling的
单层K源码hash一致，没有新增设备实验或父流水性能结论。外层记录在
`runs/refactor/hierarchy-verified-regressions.log`、`hierarchy-boundaries/cold-child-final/`。

在工作区根目录：

```bash
.venv/bin/pytest -q compiler/tests/test_hierarchy.py
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'compiler/examples')
from ast_hierarchical_gemm import hierarchical_gemm
from xdsl_npu.ir import dump
print(dump(hierarchical_gemm.to_ir()))
PY
```

下一步将源程序真正降为保留的物理父子region，验证每个GM frame的范围/owner，
并把上述memory与event义务接入父层完成图、同步物化和最终检查。其后才进行
父流水ASC/CANN、同binary双参数设备验证及固定AscendC profiling；R2a仍未验收。
