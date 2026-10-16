# 当前实现与待解决问题

2026-10-16。**默认 dense f16 Flash Attention 已从公开 DSL 贯通到 CANN 和设备数值验证；编译耗时及性能仍未达标。**
算子唯一数学主体在 [examples/flash_attention/attention.py](../examples/flash_attention/attention.py)，
没有恢复旧 task DSL、手写 builder 或隐藏算子实现。此前记录见
[历史快照](history/refactor-progress-before-query-20261016.md)。

## 本阶段实际完成

公开路径是 `prim_func → source xDSL → physical IR → autosync → 独立最终验证 → ASC → CANN`。
真实 query、KV、固定 child 三个时钟均保留；Sq/Sk 是 runtime 参数。默认 FA
保留20组AIC/双AIV、12个call、72个native模板，以及原buffer、数学主体与派发顺序。
从设计模板迁入 example 后，source IR 与同步前 physical IR 均与设备构建逐字节一致。

跨 query 访问分析保留原allocation、byte cell、首写/末写、完整reader span及slot相位。
N=0是无访问identity，不能凭空初始化；KV=1的双槽可在query距离2才复用。
共享L0C采用已经过逻辑RAW配对的物理占用边界，两个plane各自保留最后FIX读取。
相同effect序列共享一次transfer计算，但不跨孔洞合并地址，也不删除byte义务。

QueryChannel/QuerySync保存query距离、KV CountDomain/Position及真实三条SSA时钟。
完成图从实际同步IR重建，沿用局部流水的native transition；Set effect不能进入
native retirement前缀。旧父/子event按物理bank/ID检查，包含共享池不同channel及条件case。
现有跨核counter要求已验证query内≤15，并证明旧query最后Wait到新query首次Set的反馈；
找不到该充分条件就拒绝。新的query channel目前只支持同actor、不同pipe的local event。

完整FA的49,175条byte义务归并为4,360个端点检查。现有流水覆盖4,359条，唯一缺失为：

```text
同步前：PV(q,last KV).FIX读取L0C → QK(q+1,first KV).MAD覆盖同一plane，未证明
同步后：最后PV FIX之后 Set(FIX,M,6)；下query首次QK MAD之前 Wait(FIX,M,6)
Set：M > 1 && q < M-1；Wait：q >= 1；固定child位置均为0
```

Set仅在最后KV发出，Wait仅在下query首KV消费。最终检查还证明旧Wait先于该ID再次Set；
没有query drain、额外buffer或copy。最终ASC中的ID是常量6，PIPE_ALL仅在actor返回处。
最后验证覆盖4,360个内存检查及129个资源检查：父binary8、child binary113、
跨核counter边界6、新query event复用1、入口configuration1。新增local waits只依赖更早query，
与原parent进展和实际native插入位置一起构成进展检查；不按运行时query数展开。
独立有限事件图覆盖短循环和破坏反例，不宣称这是任意程序的形式化证明。

相关集成回归 **536项通过，211.85秒**；runner最终metadata/ABI回归另44项通过。
新query有限图、实际IR破坏、条件/slot/资源和nonquery生成路径均在本层覆盖。

## 真实构建与设备结果

证据目录：`runs/refactor/query-boundaries/`（外层，不进产品Git）。

| 验证 | 结果 |
|---|---|
| 完整FA源码→ASC | 169.10秒；planning111.71秒，final verifier54.51秒 |
| 同次源码→ASC→CANN/shared | 176.43秒；统一300秒硬期限；CANN9.1、dav-2201、-O2 |
| 同一binary，小规模全输出对照 | Sq:Sk =128:1024、384:3072、768:1024、512:2048；每例4次launch |
| 数值/保护 | f16、B1/H8/Hkv8/D128，max abs error≤3.0518e-5；输入及保护区完整 |
| 设备 | 全卡健康/进程/HBM/利用率检查后，物理6→逻辑0；无reset或换卡 |
| 大规模数值 | 4096×4096及8192×8192，同一binary；每head每query tile采样6行，全输出有限值、输入/保护区检查通过 |

源/IR/ASC/库及完整选项在 `fa-build/`；ASC SHA256：
`ae3e169c7ebf1a34f8c1f736b3d673b0258198eabdeeebffe93ffb560a596f12`。
首次临时runner沿用独立参考ABI，被host launcher拒绝，kernel未派发；修正为实际
`Q,K,Score,P,V,PV,O,Sq,Sk`后运行。该拒绝单独保存在`device/host-rejection.json`。
物理launcher参数顺序可能不同于DSL函数形式参数，不能让runner猜测；统一ABI元数据仍需改善。

最小纯Vector跨query示例源码→ASC单次约1.06秒；它不能替代FA编译耗时。
该例按等价端点及完整CountDomain合并短循环条件后只生成两条释放协议、两个常量ID。
完整FA边界分析为4,503个byte cells、667个唯一transfer序列；没有展开输入长度。

## 明确限制及下一步

- **编译耗时严重超目标**：完整FA仍需169秒Python时间，目标p50≤5秒/max≤10秒。
  目前规划再次建立parent proof，最终verifier再独立重建；条件事实传播与投影成本仍高。
  下一步基于本次阶段计时削减规划重复分析，并保留最终独立检查，不提高预算或靠缓存掩盖复杂度。
- **性能未达标**：4K初始ACL stream间隔generated约1143µs、ACLNN约516µs，差距约2.22倍。
  该口径含host间隙；三批device task p50为generated约1140µs、独立AscendC约1190µs、ACLNN约511µs；
  相对ACLNN为2.230倍，明确失败。Vector活动约698/326µs，详见[设备基线](attention-performance.md)。
- 当前设备验证仅覆盖默认dense f16政策。bf16/GQA/MQA/cross/tails/causal等不能因前端可表达就宣称支持。
  continuous的runtime-cycle首末条件仍明确拒绝；paged/sparse/persistent及动态ownership未完成。
- 新query channel暂不新增跨AIC/AIV通知、同pipe barrier或共享event池。
  小型mixed recurrence目前会因AIC MTE1→MTE2没有空闲ID而拒绝；不加buffer或全局barrier掩盖资源冲突。
- 自动排步尚未实现。现有专家Stage/issue policy经过相同访问、同步、event和最终验证。
- 通用纯整数canonicalize/CSE已在现有路径中使用；ASC capture表达式仍有`/1`等冗余，
  需要在公共表达式层处理，不能用FA字符串替换或忽略异步effects的移动消除。

可复现构建从工作区根目录执行（不会安装或打包）：

```bash
PYTHONPATH=compiler .venv/bin/python -m xdsl_npu.cli --input compiler/examples/flash_attention/attention.py --entry attention_target --output runs/fa-reproduce --shared --timeout 300
.venv/bin/python -m pytest -q compiler/tests/test_scope_query_planning.py compiler/tests/test_scope_query_protocols.py compiler/tests/test_scope_query_resources.py compiler/tests/test_scope_query_transitions.py compiler/tests/test_scope_query_codegen.py
```

设备复现使用`python -m benchmarks.attention.run`，三个后端均已真实运行；命令见[工具说明](../benchmarks/README.md)。
性能/变体后续顺序见[执行计划](refactor-plan.md)。
