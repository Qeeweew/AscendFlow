# 内存合同与可能别名重构：首批验收

2026-10-18。首批验收使用唯一 DSL → native/effects → 依赖 → 通知与资源 → ASC
链路。产品 HEAD 为 e29d5ea，改动尚未提交。首批源码指纹、构建和测量关联见
[机器记录](memory-refactor.json)（按仓库约定本地生成，不入 Git）；语义说明见
[内存合同](../development/memory-contracts.md)。

## 已实现

- 删除全局 group/SID 地址分离准入和 OwnerProof，保留真实 StorageInstance、
  外部 pointer alias、初始化、借用 Tile、原式定义性与 native alias 限制。
- 区分精确 StorageCell 和 AliasRegion。未知区域保留源码读写顺序，不伪造
  数值 producer、完整 KILL、初始化、private packet 或稳定 cell credit。
- Tensor 增加 write_disjoint 源合同；只对没有内部 reader 的外部输出消除写间
  依赖。保留实际 native effects、输入 reader 退休及最终输出完成。
- read/read_write/write 权限随 source lowering 保留到原生访问；实际写入只读
  tensor 会被拒绝。FA 的最终 O 声明 write_disjoint，Score/P/PV 仍普通复用。

本地 TileLang 910B 快照 6ba65d20 含历史修改。其 MayConflict 对未知区域保守
建立依赖、assume_no_conflict 表达源前提，是这次职责划分的参考；未引入 TVM、
TileLang 或 MLIR 构建依赖，也未复制其递归闭包求解器。

## CPU、IR 与新访问模式

全量测试为 1861 passed、1 failed，413.39 秒。唯一失败是“五槽嵌套流水应当
拒绝”的旧断言；该用例已改为公共 compile 和有限字节/event 检查。最终相关
18 项重跑全部通过，包括新增的双 AIV AliasRegion quorum 反例。合计覆盖
当前 1863 个不同用例；没有把旧的失败日志写成单次全绿。

覆盖动态符号别名、错位 ring 的前代值、重叠相邻循环、无 reader 的 WAW、
空/单次/短循环、输入/输出合同和两个 AIV 的实际贡献。删除必要 WAW 通知或
一个 AIV release 贡献时，独立模型仍拒绝。Ruff check、format 与 diff check
通过。有限模型是反例检查，不外推为任意控制或硬件语义的完整证明。

动态别名 vector DSL 用标准 CANN 链编译；CANN 阶段 4.757 秒，总构建受 300 秒
进程组上限保护。物理卡 2 → 逻辑卡 0 上验证 30 个参数组合，n 为 0/1/2/5/17，
包含同址与异址。全部输出逐元素 FP32 精确相等，输入未改写、前后 guard 完整。
源码、ASC、共享库与驱动脚本指纹保存在机器记录及 build/refactor-access 下。

## Causal FA 性能

只验收等长 1K/2K/4K/8K，B1、Hq=Hkv=8、D128、FP16，左上 causal j<=i。
物理卡 2，Ascend910B3，CANN 9.1.0，dav-2201，-O2。每个 shape/backend 一次
默认 msprof，预热 1 次、测量 3 次，取设备 task 时间中位数。

| Sq=Sk | AscendFlow µs | 安装版 ACLNN µs | 加速比 |
|---|---:|---:|---:|
| 1024 | 44.740 | 53.620 | 1.198× |
| 2048 | 108.684 | 116.820 | 1.075× |
| 4096 | 307.584 | 347.004 | 1.128× |
| 8192 | 1058.780 | 1072.520 | 1.013× |

四例均通过 require-no-slower、配对 sampled 数值、输入、guard 和全输出有限性
检查。BK256/VR16、BK512/VR16、BK1024/VR8 的完整构建分别为
45.537、55.746、54.796 秒，均包含 Python、ASC 和 CANN。
三套当前 ASC 与重构前验收产物逐字节相同；当前测量仍使用本次重新构建的库。
性能只适用于上述条件，8K 的本次余量约 1.3%，不宣称其他 kernel 无性能损失。

## 后续边界

本批未重写通用父代次条件激活、动态空 child 的 family closure 或跨 group
phase/collective。现有 FIRST/LAST、queue_order、cell credit、handback、有限
bank 和 quorum 检查保留。可能别名的保守顺序会影响可用 overlap；不兼容的
固定排步与超出支持范围的控制仍须具体诊断。

## 同日后续：循环实现去重

本次后续只改 logical_valueflow：普通循环与中间区间共用一份递推规则，首尾
条件在本次 memory snapshot 内分类并复用。文件从 790 行减至 677 行，没有
新增激活区间、周期切分、求解模块、DSL 语法或支持范围。通用父代次条件激活
及动态空 child 的模型重构仍未完成；实现去重不能算作这些能力已经交付。

全量回归单次 **1863 passed，423.54 秒**；Ruff check、format 与 diff check
通过。三套 causal FA 的完整 Python/ASC/CANN 构建分别为
42.564 / 52.266 / 52.704 秒，各自受单次 300 秒硬上限约束；ASC、设备二进制
和共享库均与首批验收产物逐字节相同。本轮未重新运行设备或测量性能。
本次源码指纹、构建及比较记录位于
[后续记录](../../build/refactor-simplify/result.json)，完整测试日志保存在同目录。
