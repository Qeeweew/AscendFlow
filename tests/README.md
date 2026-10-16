# 测试入口与迁移原则

从工作区 `/workspace/xdsl-910b` 使用 `.venv`。本轮只更新文档，没有运行编译或测试；历史测试总数不代表停止的未提交工作树全部通过。

重构目标分组为frontend、dialects/effects、analysis/access、loops/PLE、sync、resources、backend、integration、models、cost。目前多数文件仍平铺，目录迁移按[计划](../docs/refactor-plan.md)推进，不在文档中宣称已经完成。

## 当前有价值的入口

| 范围 | 现有路径/文件 | 保留的合同 |
|---|---|---|
| 语言/API | test_ast_api.py、test_runtime_contract.py | 旧task入口不存在，runtime合同贯通 |
| 变换所有权 | test_compilation.py、test_scope_planning_reuse.py | normal不重复独立证明；旧snapshot不能授权变更 |
| 同步与反例 | sync/test_direct_synthesis.py、sync/test_plan_validation.py | ready/release、实际IR破坏、等待不能扩大 |
| 物理访问 | test_scope_access.py、test_ownership_footprints.py | 精确字节/alias/owner，区间压缩不丢孔洞 |
| 资源与CV | test_scope_event_pools.py、test_scope_event_device_model.py、test_cv_protocols.py | ID/lease/参与者/派发进展 |
| 有限模型 | models/ | 有界的指令、字节和通知行为；需标明独立性 |
| 成本 | sync/test_symbolic_cost.py、test_compact_fixed_summary.py | 当前算法实际工作计数，新算法需改测边界条款与比较数 |
| 构建期限 | test_compile_watchdog.py | 整个构建进程组期限、失败清理和产物身份 |
| 数值 | test_attention_math.py、benchmarks.validation | 数学参考，不代替生成代码设备资格 |

停止的实验包含尚未提交的新测试；文件存在不意味着已全部通过。旧硬件反例保留，固定旧节点数、缓存命中或内部模块名的断言要按新语义改写。

## 验证独立性

从实际IR解释访问/指令的模型，与读取生产native topology后独立BFS的差分不是同一等级。`models/finite_query_native.py`复用生产native topology，只能独立检查求解器路径；full/compact共享转移函数的对照主要检查表示。它们都不能冒充任意N或硬件合同的独立证明。

最小反例矩阵和各层口径统一见[validation.md](../docs/validation.md)，变换保持见[不变量](../docs/compilation-invariants.md)。

## 运行纪律

```bash
.venv/bin/python -m pytest -q compiler/tests/test_ast_api.py compiler/tests/test_runtime_contract.py
.venv/bin/python -m pytest -q compiler/tests/sync --durations=10
```

根据变更选择相关小组，稳定集成后才运行一次必要的全量回归。默认`pytest compiler/tests`仍覆盖现有回归；不默认跳过旧失败或预算超限。修改IR的测试各取独立副本，共用不可变编译产物；成本输入和冷编译不复用cache。

单个算子构建≤300s，失败停止后续候选并清理所属子进程。设备runner在tests/support中只负责小规模资格，不当大算子性能证据；设备运行规则见[benchmark入口](../benchmarks/README.md)。
