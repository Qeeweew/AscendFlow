# 示例验证与测量工具

日常只需要两个入口，完整操作见[测试指南](../docs/guide/testing.md)：

```bash
python -m benchmarks.check_examples --list
python -m benchmarks.check_examples --output build/examples-cann
# 检查全部卡并显式映射后，增加 --physical 进行数值验证。
python -m benchmarks.profile_defaults --output build/kernel-times --physical 2
```

`check_examples` 检查所有公开示例；`profile_defaults` 仅采集默认 4K Flash Attention
和 GEMM normal，各一次默认 msprof。输出目录须是新目录。运行环境、卡选择和
输入合同见[快速上手](../docs/guide/quickstart.md)，结果见[示例验证](../docs/reports/examples.md)
和[性能报告](../docs/reports/performance.md)。

| 工具 | 用途 |
|---|---|
| `check_examples.py` | 统一示例清单，CANN 构建、设备数值检查、失败即停 |
| `profile_defaults.py` | 两个关键用例的一次采集与 kernel time |
| `vector/`、`normalization/`、`gemm/`、`cv/`、`state/`、`attention/` | 单个示例 runner；参数用 `python -m <模块> --help` 查看 |
| `profile.sh`、`profile_summary.py`、`artifacts.py` | msprof 应用包装、设备 task 汇总和产物身份检查 |
| `validation/` | 独立数值参考、输入数据和 dtype 编解码 |

## 专项诊断

以下工具保留给明确的编译或性能调查，输出均写入 Git 忽略的 `build/`：

- `compile.py`：冷进程 source→ASC 计时；单次 probe 与至少五次资格分开，CANN/RSS 另列。
- `references/`、`attention/run_policy.py`：独立 AscendC 与安装版 ACLNN 对照。
- `gemm/compare_*`、`attention/compare.py`：历史三批严格比较流程，要求各自记录 schema。
- `*/analyze*.py`、`work.py`、`regions.py`：结构清单、算法工作量与条件速率分析。

`source-work-v1` 的静态模板数不代表运行期执行数，算法 FLOPs 不代表 native FLOPs；
运行期请求数尚不可得时保持 `unavailable`。profile 与实际 DSL、IR、ASC、库、选项、
输入和设备绑定，不能将参考路径的通过当成当前 DSL 通过。验证边界见
[开发验证方法](../docs/development/validation.md)。
