# P1：顺序区域的精确访问组合

本切片已从[精确访问组合](../src/xdsl_npu/region_sequence.py)接通根部单actor的
普通工作→流水→普通工作，以及共享存储的多个兄弟流水。实现见
[sequence_lowering](../src/xdsl_npu/sequence_lowering.py)、
[sequence_sync](../src/xdsl_npu/sequence_sync.py)与
[sequence_codegen](../src/xdsl_npu/sequence_codegen.py)。这是受限的根部组合，
不是任意层次region闭包；serial父体和mixed层次组合仍未接通。

## 数据来自哪些实际事实

叶流水仍由`region_summary.summarize_reused_storage`提取每个byte cell的入口
reads、首/末writer、出口reader families和count域。现在还保留原分配的SSA身份；
跨region匹配按实际allocation，不能只比较打印的buffer名字。

`Part`标识一次region出现，`Endpoint`保留该region、原native任务、首尾相对
generation和自己的固定count（若有）。不同调用即使复用同一个task UID也不混同。
初始化这样的固定一次工作与runtime child可以组合；固定百万次child也不会
按次数遍历，ReadSpan仍是参数化范围。

当前接受：stationary精确字节范围、一个共同runtime bound和固定count的
叶区域。实际effects证明只读的漂移GM允许共享；跨region漂移写、不同runtime
bound及mixed摘要继续拒绝。没有
推广为任意深度区域闭包，也没有把用户自带摘要注册为可信契约。生产checker
从最终物理IR重建所有输入，不能信任planner的已验证标记。

共同bound必须引用同一个实际SSA值。两个分别捕获的参数即使都打印为`n`、
表达式文本相同，也不能推断相等；这与allocation身份的检查分别进行。

## 顺序组合规则

对每个实际allocation独立扫描区间边界，保留孔洞。对真正相交的cell按count域
组合，互不相交allocation不联合周期。前缀A之后接B：

```text
A.last_writer → B.reads_before_write 产生 RAW
A.last_writer → B.first_writer       产生 WAW
A.pending_readers → B.first_writer   产生 WAR
```

B无writer时，A的writer与B的所有reader继续存活；B有writer时更新出口writer与
pending readers，但覆盖前已产生的WAR不会删除。没有入口writer的读取保留为
开放初始化义务，不能凭借后面的write或initialized标记消去。

例如真实输入状态`init; child(read_v, read_s, read_dma); overwrite`，覆盖前必须
保护V、Scalar和MTE3的全部read families。某个最后派发reader本身并不代表此前
其他读取已退休；只有后续完成证明可以用覆盖完整reader集合的prefix替换它们。

child=0是恒等转换：前后的工作依然存在；`init; empty; use`依赖真实init，
`empty; use`仍有未初始化入口。末代writer不能初始化此前读取，也不在零次时执行。

## 复杂度和验证边界

只对参与同一byte cell的静态count分类取LCM，另保留短count域。构造区间索引、
候选domain、endpoint、义务和输出pieces都受结构预算约束；内部poll支持取消。
固定count在自己的长度求值，不因其很大而创建很多domain。不同runtime bound
当前明确拒绝，未引入二元参数域的笛卡尔枚举。

这是有上限的精确传递实现，不承诺线性复杂度：区间/domain相交及输出义务本身
可能二次增长。预算超限给出具体phase，不能回退有限展开/串行化。源码输入在
失败/取消后不被修改，既有编译流程仍使用原final checker。

测试独立按有限generation和逐字节历史扫描，比较输出frontier和跨region
RAW/WAR/WAW集合，不调用生产frontier/sweep推导期望。覆盖0/1/跨ring、多个
reader、物理孔洞、同名不同分配、两个child、末代writer，以及随机区域组合。
百万固定child检查表示规模，不对百万次枚举；有限样本不冒充任意N形式化证明。

```bash
.venv/bin/python -m pytest -q compiler/tests/test_region_sequence.py compiler/tests/test_hierarchy.py compiler/tests/test_invocations.py
```

## 实际同步前后与设备闭环

[state_sum](../examples/ast_state_sum.py)有真实Initial加载、双槽输入和FP32
逐代累加，最后导出；N=0输出Initial。插入前的物理IR投影为：

```text
once r0: MTE2 Initial -> state
stream r1(N): stage0 MTE2 X[t] -> x[t%2]
              stage1 V add(state, x[t%2]) -> state
once r2: MTE3 state -> Output
```

同步后（投影省略实际槽位ID表达式）：

```text
r0: load Initial; if N==0 Set(MTE2,MTE3)
r1.load[t]: if t>=2 Wait(V,MTE2, t-2); load X[t]; Set(MTE2,V,t)
r1.add[t]:  if t>=1 Pipe(V); Wait(MTE2,V,t); add
            if t==N-1 Set(V,MTE3)
            if t<N-2 Set(V,MTE2,t)
r2: if N==0 Wait(MTE2,MTE3) else Wait(V,MTE3); store
kernel exit: Pipe(ALL)
```

N>0时第一个X的MTE2通知覆盖更早的Initial加载，无需另一个入口event。
input槽释放取Vector读取退休，不等输出搬运；state递推的V等待有实际RAW依据。
普通region退出不drain。N=0使用独立匹配的MTE2→MTE3路径，不能等待不存在的add。

叶规划使用条件入口假设，最终sequence checker从真实前驱写及物化Set/Wait
独立解除这些假设；不能把内部`live_in`当作公开的免检初始化声明。
跨region事件暂用互不相交ID，可能拒绝本可通过寿命复用的组合；boundary消除
目前使用一个bridge加叶路径，不宣称最少同步。固定大count以覆盖它的静态边界
族证明，避免把首尾绝对距离变为百万个查询状态。

已有记录：相关316项回归通过，最后位置检查改动后24项sequence测试通过，
两次测试范围重叠。独立有限指令/逐字节oracle覆盖N=0/1/2/3/4/7/17及破坏协议
的负例。CANN编译、设备小例和X=1 GiB大例通过；独立AscendC对照见
[验证记录](refactor-progress.md)。这些证据不覆盖未接通的父体组合或新AST FA。

工作区根目录运行：

```bash
.venv/bin/python compiler/examples/run_state_sum.py --output runs/refactor/state-sum --build
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/ast_state_sum.py --entry state_sum --trials 5 --output runs/refactor/state-sum/cold-review
```

设备命令在全部卡健康/空闲检查后选择物理卡；已有证据使用物理1→逻辑0：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_state_sum.py --output runs/refactor/state-sum --physical 1
```

仍缺父serial体内前后工作、跨调用共享只读GM的统一组合、一般frame转换、
跨region借用数值/Scalar捕获、mixed层次与event寿命复用；P1整体未完成。
