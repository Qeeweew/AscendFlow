# R2：分离存储周期和同步协议周期

本切片修改现有单层符号流水的分析，不改变 DSL 排步、存储分配或原生指令。
[示例](../examples/independent_rings.py) 使用显式5槽输入、7槽输出，计算
`Y[t,:] = 2 * X[t,:]`；这组互素容量用于验证编译复杂度，不是算子调优建议。

## 原来的问题与新的边界

旧实现把所有 view 的周期取全局 LCM，然后把整个完成图复制这么多份。
5槽和7槽导致35份图；29槽和31槽先因899超过840的预算拒绝，尚未到达真正
的事件资源检查。运行时长度 N 虽未展开，静态地址周期仍会让编译很慢。

现在分开处理三种数量：

- 每个物理分配的地址周期 `P_s`：该分配上的所有别名共同分析，不能按 tensor
  名字拆开。私有存储包含 actor 身份，共享 GM 保持同一分配身份。
- 实际 Set/Wait/pipe barrier 的执行条件周期 `Q`：先合并同端点的各 byte cell
  义务，约简执行条件，再构建完成图。每代都执行的协议周期为1。
- event ID 环宽 `W`：直接证明旧 Wait 在第 W 次后续 Set 前完成，不把 ID 的
  轮转作为完成图的展开周期。buffer 槽数与 W 仍分别检查。

同一分配上不同周期的别名仍需联合分析；周期超过840明确诊断。多个真实条件
协议的周期目前仍会取 LCM，尚未实现按协议连通分量分图。这是移除“所有地址
周期的全局 LCM”的第一步，不是 R2 所有独立时钟和分层摘要已经完成。

## 自动同步前后与等待范围

下面是实际物理 IR 的简写；CLI 会保存完整 `before.xdsl`、`after.xdsl` 和 ASC。

```text
before:
  load(g),    stage=0: MTE2  R X[g]       W inputs[g%5]
  compute(g), stage=1: V     R inputs[g%5] W outputs[g%7]
  store(g),   stage=2: MTE3   R outputs[g%7] W Y[g]

after, 每个真实 generation g:
  before load(g):    wait V→MTE2(g-5),   only if g>=5
  after load(g):     set  MTE2→V(g)
  before compute(g): wait MTE2→V(g)
                    wait MTE3→V(g-7),   only if g>=7
  after compute(g):  set  V→MTE2(g),     only if g+5<N
                    set  V→MTE3(g)
  before store(g):   wait V→MTE3(g)
  after store(g):    set  MTE3→V(g),     only if g+7<N
  after loop:       final actor drain
```

输入 WAR 来自 compute 的最后读取，允许下一次 load 覆盖输入时，旧 store
仍可继续。输出 WAR 来自 store 的最后读取。没有把输入 release 推迟到 store，
没有新增搬运，也没有在循环中插入 PIPE_ALL。四个 directed bank 分别使用
5/5/7/7个 ID，事件配对和再次 Set 的寿命另外验证。

最小资源反例是同样排步配29/31槽：存储依赖分析可以通过，但当前完成模型
允许超过8个 ready 事件积压，无法证明8个 ID 能安全复用。编译器报
`no legal binary event reuse width <=8`，不缩小用户槽数、不添加串行化等待。
这不表示所有使用29/31槽的程序都不可实现；排步或协议规划改变需要独立验证。

## 不同周期之间的证明

某个存储冲突可能只在 `g % P_s ∈ R` 时成立，实际同步图却按 Q 重复。
不能只对 R 中的一个代表代次查询图。实现用整数同余求交，求出每个相交的
Q余数及其第一个真实 generation（同时满足依赖距离与非负边界）。证明从
该位置起的所有图平移覆盖冲突，因此包括 prologue、steady state 和 epilogue。
同一端点/距离/协议余数只保留最早的义务，避免每个 slot 复制同样的证明域。

测试包含一个反例：把5槽程序的 ready 协议改为只在偶数代 Set/Wait。地址
周期和协议周期互不整除，最终 verifier 仍必须拒绝奇数代的缺失 RAW。
另有穷举小周期的同余求交对照、独立逐字节冲突 oracle 与有限硬件 DAG 对照。
有限实例测试不是任意规模的形式化证明；最终 checker 仍复用生产分析原语。

## 复杂度与预算

令 `A_s` 为访问分配 s 的静态 access 数，`C` 为实际命令数，`H` 为精确 byte
cell 义务数。地址枚举从 `LCM(P_s) * sum(A_s)` 改为 `sum(P_s * A_s)`；完成图
规模取决于 `Q * C`，不再取决于所有存储周期的乘积。区间切分、每个 cell 的
reader frontier、路径域大小仍另有成本，不能据此宣称整个编译器线性复杂度。

报告包含 `storage_periods`、`native_accesses`、`storage_residue_accesses`、
`protocol_period`、依赖/图规模和最终检查的 `proof_*` 计数。
`storage_period` 单一字段已移除，避免把协议周期误报成存储周期。
当前预算分别限制单个耦合周期840、存储余数/view总量65536、协议余数/命令
总量65536，以及每份完成图缓存的证明状态1000000。证明域先计数再分配；
超预算诊断给出实际周期、图节点、first/upper gap与请求/已有状态数。
这些结构预算还不是完整的墙钟超时或可取消搜索；冷编译脚本另有60秒单进程
超时。嵌套流水、一般分支、协议分量分图和跨通道 ID 寿命共享仍待实现。

## 复现

工作区根目录执行，产物留在外层 runs；设备命令前检查全部卡并选择健康空闲卡。

```bash
.venv/bin/pytest -q compiler/tests/test_periodic.py compiler/tests/test_loop_sync.py
.venv/bin/python compiler/examples/independent_rings.py --output runs/refactor/factored-periods/device-build --shared
.venv/bin/python compiler/benchmarks/compile.py --input compiler/examples/independent_rings.py --entry independent_rings --trials 5 --output runs/refactor/factored-periods/cold-after
npu-smi info
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/independent_rings.py --output runs/refactor/factored-periods/device --library runs/refactor/factored-periods/device-build/independent_rings.so --physical 1
```

设备 runner 使用同一 binary 测试0/1/2/4/5/6/7/8/14/35/65536/262144个
64元素 tile。最后一项为16777216个 f32元素。它记录的 ACL stream interval
包含 host 提交间隙，不能用作算子吞吐率或相对 AscendC 的性能结论。
本次改动的性能指标是完整源码到 ASC 的编译时间；设备执行组织通过最终
ASC 对照，并继承匹配源码 hash 的已有基线证据。

## 已验证结果

对比 `7ace2dc` 的产品源码快照与本次实现，使用同一份 Python 示例定义，
两边各5次独立冷进程，import/CANN 分开计时；最终批次没有同时运行其他编译
或回归任务。外层记录位于 `runs/refactor/factored-periods/`。

| 指标，5/7槽示例 | 修改前 | 修改后 |
|---|---:|---:|
| 完整源码→ASC p50 | 3.9671秒 | 0.1726秒 |
| 完整源码→ASC max | 3.9765秒 | 0.1886秒 |
| peak RSS中位数 | 310524 KiB | 38100 KiB |
| 规划阶段p50 | 2.2542秒 | 0.0740秒 |
| 最终检查p50 | 1.5151秒 | 0.0150秒 |
| 完成图节点/边 | 2275 / 3640 | 65 / 104 |
| 物理依赖模板 | 36 | 36 |
| coverage / event寿命查询 | 210 / 140 | 6 / 4 |

最终图5个证明域，总计2958个状态、4213条边；这些是最终 verifier 的计数，
不包括规划候选图上的全部查询。完整编译约快23倍，仅为此固定结构的测量。
证据见 `cold-before-final/benchmark.json` 和 `cold-after-final/benchmark.json`；
较早 `cold-before/` 是与其他验证重叠的探索批次，不用于上述数字。

五种配置逐字相同的 ASC 对照记录在 `asc-comparison.json`：普通Vector、
normalization、CV、分段CV和互素槽位例。前四项完成图节点分别从
130/338/872/2394减到65/169/436/1197，没有改变缓冲、事件或原生调用。
normalization 的 ASC hash 仍为
`9d4184683723cbaad79e559c2ffdae3148b1321e19a9b9f5145d5cf981beb590`，
可与上一切片的同契约 AscendC/profiling 记录绑定；本次没有宣称设备内核更快。

新例通过 CANN9.1/dav-2201离线和shared编译。全部8张卡健康、利用率0且无进程
后，使用物理1→逻辑0；同一binary的12组长度、每组1次warmup和5次重复通过，
最大16777216元素，误差0，输入和输出guard保持。设备记录在 `device/device.json`；
生成源码SHA256为
`99eea63c9bacc97809f7c4c2e9e4646b32efcee4b98ec8638f478b9aacdd5283`。
编译选项、ELF/shared hash在 `device-build/independent_rings.compile.json`。

241项前端、符号循环、prefix、归约、CV、编译、物理扩展及算子相关回归通过
（57.76秒）。随后缓存协议周期读取并增加报告计数，最终相关54项通过
（22.23秒）；两批计数有重叠。全目录ruff和Git diff检查通过。
