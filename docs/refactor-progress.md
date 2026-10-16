# 当前实现与待解决问题

2026-10-16。**默认 dense f16 Flash Attention 已从公开 DSL 贯通到 CANN 和设备数值验证；编译耗时及性能仍未达标。**
算子唯一数学主体在 [examples/flash_attention/attention.py](../examples/flash_attention/attention.py)，
没有恢复旧 task DSL、手写 builder 或隐藏算子实现。此前记录见
[历史快照](history/refactor-progress-before-query-20261016.md)。

## 本阶段实际完成

公开路径是 `prim_func → source xDSL → physical IR → autosync → 独立最终验证 → ASC → CANN`。
真实 query、KV、固定 child 三个时钟均保留；Sq/Sk 是 runtime 参数。默认 FA
保留20组AIC/双AIV、12个call，Sq/Sk不展开。最新版本为76个native模板：
显式block8分层归约比此前R64版本增加4个模板、每AIV增加4096B scratch（总182400B）。
最终ASC的19条搬运、MAD和完整event语句（含条件）与R64基线逐字节一致；
没有改变CV派发、P/O舍入位置或递推更新顺序。

跨 query 访问分析保留原allocation、byte cell、首写/末写、完整reader span及slot相位。
N=0是无访问identity，不能凭空初始化；KV=1的双槽可在query距离2才复用。
共享L0C采用已经过逻辑RAW配对的物理占用边界，两个plane各自保留最后FIX读取。
相同effect序列共享一次transfer计算，但不跨孔洞合并地址，也不删除byte义务。

QueryChannel/QuerySync保存query距离、KV CountDomain/Position及真实三条SSA时钟。
完成图从实际同步IR重建，沿用局部流水的native transition；Set effect不能进入
native retirement前缀。旧父/子event按物理bank/ID检查，包含共享池不同channel及条件case。
现有跨核counter要求已验证query内≤15，并证明旧query最后Wait到新query首次Set的反馈；
找不到该充分条件就拒绝。新的query channel目前只支持同actor、不同pipe的local event。

当前FA的49,271条byte义务归并为4,372个端点检查，仍只需补上一条跨query协议：

```text
同步前：PV(q,last KV).FIX读取L0C → QK(q+1,first KV).MAD覆盖同一plane，未证明
同步后：最后PV FIX之后 Set(FIX,M,6)；下query首次QK MAD之前 Wait(FIX,M,6)
Set：M > 1 && q < M-1；Wait：q >= 1；固定child位置均为0
```

Set仅在最后KV发出，Wait仅在下query首KV消费。最终检查还证明旧Wait先于该ID再次Set；
没有query drain、额外buffer或copy。最终ASC中的ID是常量6，PIPE_ALL仅在actor返回处。
最后验证覆盖4,372个内存检查及129个资源检查：父binary8、child binary113、
跨核counter边界6、新query event复用1、入口configuration1。新增local waits只依赖更早query，
与原parent进展和实际native插入位置一起构成进展检查；不按运行时query数展开。
独立有限事件图覆盖短循环和破坏反例，不宣称这是任意程序的形式化证明。

本轮定向回归 **321项通过**，包括真实IR破坏、query资源/生成、归约footprint/alias、
旧Vector路径及FA入口。最后query集成批次119项/62.55秒，未重复全量测试。
普通测试不编译完整FA；下面的单次构建单独验证实际纵向路径。

规划交还已建立的access/leaf/parent结果，去掉进入query层前的完整重复验证；
最终verifier仍从实际IR重新提取。child event宽度适配后重建实际资源，不能使用候选池代替。
新query event先合批检查单ID及内存关系，只有失败才扩大静态窗口一次，仍选最小可行宽度。
两份小型query parent IR和两份nonquery ASC与改前逐字节一致。

## 真实构建与设备结果

最新证据目录：`runs/refactor/compile-vector/`（外层，不进产品Git）；
此前R64基线保留在`runs/refactor/query-boundaries/`。

| 验证 | 结果 |
|---|---|
| 完整FA源码→ASC | 128.96秒；planning75.14秒，final verifier50.95秒；此前169.10秒 |
| 同次源码→ASC→CANN/shared | 135.57秒；此前176.43秒；统一300秒硬期限；CANN9.1、dav-2201、-O2 |
| 同一binary，小规模全输出对照 | Sq:Sk =128:1024、384:3072、768:1024、512:2048；每例4次launch |
| 数值/保护 | f16、B1/H8/Hkv8/D128，max abs error≤3.0518e-5；输入及保护区完整 |
| 设备 | 全卡健康/进程/HBM/利用率检查后，物理6→逻辑0；无reset或换卡 |
| 大规模数值 | 4096×4096及8192×8192，同一binary；每head每query tile采样6行，全输出有限值、输入/保护区检查通过 |
| 额外数据分布 | 512×2048全输出；uniform误差0，sharp最大绝对误差2.4414e-4，均通过rtol=0.002/atol=0.0002 |

源/IR/ASC/库及完整选项在 `block8-build/`；ASC SHA256：
`be2ecac39b1182f48f443367611c8db18af61f66265a574dd1162a4015a58389`。
当前runner按实际launcher绑定`Q,K,Score,P,V,PV,O,Sq,Sk`。
物理launcher参数顺序可能不同于DSL函数形式参数，不能让runner猜测；统一ABI元数据仍需改善。

最小纯Vector跨query示例源码→ASC单次约1.06秒；它不能替代FA编译耗时。
该例按等价端点及完整CountDomain合并短循环条件后只生成两条释放协议、两个常量ID。
当前完整FA边界分析为4,519个byte cells、669个唯一transfer序列；没有展开输入长度。

## 明确限制及下一步

- **编译耗时严重超目标**：完整FA仍需129秒Python时间，目标p50≤5秒/max≤10秒。
  规划前重复parent验证已去掉，条件事实传播仍占较大成本。final parent阶段仍有约1024万次
  条件记录比较；这包含不相交origin mask的比较，不等同于1024万次guard判断。
  下一步先减少无关记录扫描，保留最终独立检查，不提高预算或靠缓存掩盖复杂度。
- **性能仍未达标**：最新4K三批device task中位时间为generated825.44µs、
  独立AscendC1194.27µs、ACLNN512.21µs。相对此轮重测R64的1143.70µs降低27.8%，
  但仍是ACLNN的1.612倍。Vector活动降至380.45µs（此前698.11µs），ACLNN约325.87µs。
  详见[设备结果](attention-performance.md)，下一步必须结合CV重叠继续归因。
- 当前设备验证仅覆盖默认dense f16政策。bf16/GQA/MQA/cross/tails/causal等不能因前端可表达就宣称支持。
  continuous的runtime-cycle首末条件仍明确拒绝；paged/sparse/persistent及动态ownership未完成。
- 新query channel暂不新增跨AIC/AIV通知、同pipe barrier或共享event池。
  小型mixed recurrence目前会因AIC MTE1→MTE2没有空闲ID而拒绝；不加buffer或全局barrier掩盖资源冲突。
- 自动排步尚未实现。现有专家Stage/issue policy经过相同访问、同步、event和最终验证。
- 通用纯整数canonicalize/CSE已在现有路径中使用；公共Expr渲染现已消除capture的`/1`，
  保留原证明DAG、unsigned合法域及溢出检查。其他表达式冗余继续按公共规则处理。

可复现构建从工作区根目录执行（不会安装或打包）：

```bash
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --output runs/fa-reproduce --shared --timeout 300
.venv/bin/python -m pytest -q compiler/tests/test_scope_query_planning.py compiler/tests/test_scope_query_protocols.py compiler/tests/test_scope_query_resources.py compiler/tests/test_scope_query_transitions.py compiler/tests/test_scope_query_codegen.py
```

设备复现使用`python -m benchmarks.attention.run`，三个后端均已真实运行；命令见[工具说明](../benchmarks/README.md)。
性能/变体后续顺序见[执行计划](refactor-plan.md)。
