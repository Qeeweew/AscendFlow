# 源码职责与包结构重构

2026-10-16，结构审查基线为 `7ddaa54` 加停止的实验，已整体保存至 `bbc1458`。**以下为目标结构；M1已经迁移dialects/frontend/target及进程控制，剩余planner和后端仍待替换。** 实际状态见[状态页](refactor-progress.md)，不以搬文件数量代表完成。

## 1. 当前问题

审查基线核心包有93个Python模块、35,848行，全部平铺；36个模块以scope_开头。`model.py` 1461行、`scope_sync.py` 1383行、`loop_model.py` 1381行、`language.py` 1128行、`lowering.py` 1088行。下面的依赖问题记录基线，不将已迁移部分继续称为当前未改动。

问题是职责重叠而不只是行数。model 同时包含错误类型、物理对象、原语合同、提取和依赖，79个模块依赖它；scope_sync 同时构图、查询、插入、分配、检查和编排。各层持有不同的“已完成事实”，形成适配和重复查询。

静态审查排除TYPE_CHECKING后，初始化阶段没有依赖环；计入函数内runtime imports则有10组依赖闭环。延迟import避免了初始化冲突，但没有解决职责反向依赖：

| 当前依赖 | 要解除的关系 |
|---|---|
| scope_query_completion / scope_resources → scope_sync.ParentCompletion | 分析/资源模块不能依赖顶层planner类来定义输入 |
| child_completion → loop_planning.DirectCompletion | 原生milestone/插入cut不能归属于协议选择器 |
| scope_child_transfer ↔ scope_completion | 公共身份类型下沉，新的组合不沿用两套互调适配器 |
| codegen → sync，loop/scope_codegen → codegen | 公开检查入口、计划、通用打印分别归属 |
| frontend ↔ language | 公开语法、AST解析、内部builder分开 |
| ir.parse → source_ir → ir | 共享traits/types与dialect注册IO分开 |
| build ↔ toolchain | 进程组、deadline从高层构建编排下沉 |

这些是当前代码事实；不能将它们称作已发生的循环导入崩溃。

## 2. 目标目录

不引入插件框架或大型pass管理框架。使用xDSL已有基础设施，顶层只保留薄入口，子包按职责建立：

```text
src/xdsl_npu/
  __init__.py          # compile / Compilation / language 等明确公开入口
  compilation.py      # 一条阶段编排、产物与计时；不含同步算法
  cli.py              # 参数、输入与产物输出
  language.py         # 薄语法入口，内部实现进入 frontend
  diagnostics.py      # 错误类别、源位置、阶段、最小witness
  frontend/
    syntax.py         # Tensor/constexpr/serial/Pipelined/Stage 等声明
    parser.py         # AST捕获、名称/作用域、源码位置
    builders.py       # 内部source IR构造；不依赖sync/backend
    contracts.py      # 源语言能力/admission及runtime输入合同
  dialects/
    source.py         # 逻辑操作、布局、排步与控制
    physical.py       # native、统一loop和typed同步，去掉query专有op
    types.py          # 必需共享types/traits/effects
    io.py             # dialect注册、parse/print
  target/
    ascend910b.py     # engine、milestone、event bank、可见性/阻塞合同
    native.py         # 原语合法性、物理effects、数值与原生表示合同
  analysis/
    iteration.py      # 真实SSA循环域、控制句柄，供admission和PLE共用
    access.py         # root/view/范围、ownership、reaching definitions
    dependencies.py   # RAW/WAR/WAW与slot carried依赖
    frontier.py       # 有类型完成前沿与覆盖/合流，不是全源闭包
    loops.py          # region边界、坐标绑定与循环归纳合同
    resources.py      # event lease、冲突、配对与进展事实
  passes/
    lower_native.py   # 显式recipe、地址/布局、完整effects
    lower_pipeline.py # 固定排步 → 统一PLE；递归处理所有循环
    insert_sync.py    # 消费依赖和前沿，构造逻辑通知
    assign_events.py  # bank/ID/width/共享选择，不重新推导内存关系
    materialize.py    # 已接受同步计划写回真实IR
    simplify_scalar.py# 纯标量canonicalize/CSE与映射化简
  backend/
    asc.py            # 已确定指令与控制结构的输出
    expressions.py    # 已规范表达式的命名和打印
    cann.py           # dav-2201离线编译，不包含设备初始化
  runtime/
    process.py        # 有界子进程、整个进程组期限与清理
    launch.py         # ABI、运行时合同、设备launch
    device.py         # 卡状态及显式物理映射
```

这是一张职责地图，不要求立即创建每个空文件；原语recipe多到需要分文件时按copy/cube/vector分组，不按FA或query分组。公共language名字可保留，它是稳定产品入口，不为旧task语法提供兼容。

独立检查作为明确的验证入口消费实际IR，可放在analysis下独立模块；有限执行模型留tests/models。benchmark、profiling、数值oracle和手写参考留benchmarks，算子算法与政策留examples。runtime不负责产品报告和调参。

## 3. 依赖方向与数据所有权

```text
compilation / CLI → frontend、passes、backend、runtime
frontend          → dialects、diagnostics
passes            → dialects、target、analysis
analysis          → dialects、target、diagnostics
backend           → dialects、target；只消费完成的计划
runtime           → diagnostics / process；不导入编译分析
```

dialect的结构类型不导入planner；目标原语合同可读取physical类型，但physical类型不能反向导入代码生成。分析结果是边界合同、依赖或lease的数据，不携带可回调顶层planner的能力。源码位置和错误类别是底层公共数据。

编排持有一份私有IR；pass返回其明确建立的分析结果。只有会改变相关前提的变换才使结果失效。不可用跨阶段的任意Module哈希缓存代替所有权，也不建立独立可变程序树。

## 4. 现有模块的去留

| 现有部分 | 处理 |
|---|---|
| frontend.py / language.py / _physical_builder.py | 拆公开语法、AST parser与内部builder；_physical_builder仍被lowering使用，不能直接扔进tests |
| ir.py / source_ir.py | source/physical/shared/IO；淘汰Query/Scope/Invocation/Region专用同步op，按机制保留必要属性 |
| model.py / loop_model.py | 分离诊断、目标原语、访问提取、依赖；不把整个大文件搬到analysis/model.py |
| lowering.py / region_lowering.py / scope_lowering.py / lowering_context.py | 合并结构化递归与地址绑定，按原语recipe拆；取消serial/KV/fixed-child结构识别分派 |
| loop_planning / scope_sync / sequence_sync / invocation | 新的同步构造入口替代，迁移有价值的硬件合同和反例，旧planner按阶段删除 |
| scope_query_* / guarded与anchored适配图 / child投影 | 不迁成新的query子包；统一循环边界与carry接替，生产路径退出旧传播模型 |
| events / scope_event_* / scope_query_resources等 | 统一逻辑lease输入、按bank分解的资源分配与进展；不保留child→parent→query三层资源重证 |
| scalar / indexing / asc_scalar | 纯算术合同及通用pass在前，backend只打印；避免两套表达式等价系统各自改语义 |
| codegen / loop_codegen / scope_codegen / sequence_codegen | 统一控制输出与native格式；公开外部IR验证在编排入口，不从emitter回调planner |
| build / toolchain / device | deadline和进程组下沉；离线编译与设备launch分离 |

旧完成求解器若暂留作对照，必须通过显式debug/test入口使用，不能作为未支持结构的静默fallback。迁移期可有明确实验入口；每个阶段结束列出还剩哪条旧生产调用及删除门槛。

## 5. 原语、lowering与输出的界限

目前一个操作在frontend名单、language builder、source verifier、lowering、model.native_contract和codegen.native_line分别登记，容易漏effects或改变语义。重构明确一份原语合同：合法类型/布局、完整读写/配置、数值边界、完成里程碑和对应native表示。前端检查语法，lowering选择透明recipe，target给硬件事实，emitter打印；无需建立复杂注册框架。

现有Load2D在emitter内部展开循环。重构必须把其完整读写与实际指令工作量放到native表示或显式recipe中；不能让同步只看到一个较小的假访问范围。无需按每条静态指令建全局完成来源，recipe可以用已论证的参数化效果。

## 6. 迁移纪律

先抽底层合同和共享身份，再替换控制/同步，再统一输出，最后删除旧内部路径。每次移动与测试导入变更同一补丁，避免长期转发兼容层。普通循环和kernel不增加算子名/shape特判。

通过静态依赖审查确认没有analysis→passes/backend或backend→sync规划反向依赖；通过真实示例确认公共路径已切换。文件少了、名字变了或测试只验证旧模块不存在，都不足以验收。
