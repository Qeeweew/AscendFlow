# P1：顺序区域的精确访问组合

本切片实现[region_sequence.py](../src/xdsl_npu/region_sequence.py)的访问分析，
**尚未接通一般prelude/child/epilogue的同步规划与ASC生成**。它产生待解决义务，
不产生完成证书或whole-child barrier；没有新增设备能力或性能结论。

## 数据来自哪些实际事实

叶流水仍由`region_summary.summarize_reused_storage`提取每个byte cell的入口
reads、首/末writer、出口reader families和count域。现在还保留原分配的SSA身份；
跨region匹配按实际allocation，不能只比较打印的buffer名字。

`Part`标识一次region出现，`Endpoint`保留该region、原native任务、首尾相对
generation和自己的固定count（若有）。不同调用即使复用同一个task UID也不混同。
初始化这样的固定一次工作与runtime child可以组合；固定百万次child也不会
按次数遍历，ReadSpan仍是参数化范围。

当前接受：stationary精确字节范围、一个共同runtime bound和任意固定count的
叶区域。来自不同runtime bound的区域、frame漂移和mixed摘要继续拒绝；没有
推广为任意深度区域闭包，也没有把用户自带摘要注册为可信契约。生产checker
后续须从最终物理IR重建所有输入，不能信任planner的已验证标记。

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

下一步将普通物理工作接到此合同，导出并验证完成/事件边界，使这些开放义务
在最终IR上得到真实路径。只读共享GM、源lowering、一般顺序region的codegen
和设备闭环仍未完成；不能仅凭本切片宣布P1通过。
