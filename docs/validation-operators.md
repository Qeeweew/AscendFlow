# 算子闭环验证记录

2026-10-13，独立产品仓库，xDSL 0.73.0、CANN 9.1.0、dav-2201、物理卡1
910B3、每进程映射逻辑0。外层原始产物位于 `runs/operators/`，不提交产品 Git。
此阶段新增实际算子编译与执行；完整复杂算子编译器目标仍未完成。

## 同步与输入检查

153 项 pytest 回归通过，ruff 通过。新增回归覆盖归约的精确 partial/statistic
字节范围、Cos/Sin 同代次 WAR、不同输入的独立 release、缓存的一次激活、
跨 K 块 L0C carry、首次 clear/最终 export、固定 tile 容量诊断、每核 DMA
地址对齐，以及 epsilon 舍入成零/无穷的拒绝。

独立交错检查的以下固定小例全部穷尽，均只有一个干净终态。Scalar Wait 门控
后续全核 dispatch；Set 发射与生效分开；检查没有把末尾 drain 当内部完成边。

| 程序（双槽） | 单 actor 实例配置 | 状态数 |
|---|---|---:|
| RMSNorm | R3/D64、row-block1 | 2607 |
| RoPE | R3/D64、row-block1 | 2636 |
| Softmax | R3/D64、row-block1 | 2492 |
| SwiGLU | R3/D64、row-block1 | 12276 |
| Matmul | M32/N64/K96、K-block32 | 10105 |

删除 RoPE 的 Trig WAR 通道后，最终 verifier 与交错检查均发现错误；把第一
次 MAD 改为 accumulate 则报 undefined accumulator。不能把这些有限枚举
当作任意规模/动态控制的形式化证明。多个独立 actor 另由 GM ownership 检查
组合，不在状态机里宣称探索全部跨核交错。

## 离线编译与设备正确性

下列最终程序均通过设备 object、ELF 链接和 host `.so` 编译，并与独立 CPU
oracle 比较。所有进程在 ACL 初始化前检查全部八卡的健康、占用、HBM 和
利用率；卡0/2/4/6有其他负载，选择健康空闲的物理卡1。没有 reset、换卡或
设备错误后的重试。

| 算子 | 实际 shape / 固定政策 | 最大绝对误差 |
|---|---|---:|
| RMSNorm | 16×128、2 cores、row-block2、双槽 | 3.58e-7 |
| RMSNorm | 256×1024、8 cores、row-block8、双槽 | 4.77e-7 |
| RMSNorm | 4096×4096、32 cores、row-block2、双槽 | 4.77e-7 |
| RMSNorm 数值边界 | 16×128、零行/近零/幅度1e4/混合幅度，Gamma含负数和零 | 0 |
| RMSNorm 空 stream | 0×64 | 无挂起，0输出元素 |
| rotate-half RoPE | 16×128、2 cores、row-block2、双槽 | 0 |
| rotate-half RoPE | 4096×128、32 cores、row-block8、双槽 | 0 |
| Softmax | 16×128、2 cores、row-block2、双槽；含常量行与±1000行偏移 | 3.73e-9 |
| Softmax | 256×1024、8 cores、row-block8、双槽；同样包含稳定性输入 | 4.66e-10 |
| Softmax | 256×1024、32 cores、row-block8、双槽 | 4.66e-10 |
| Softmax | 4096×4096、32 cores、row-block2、双槽 | 1.16e-10 |
| SwiGLU | 256×1024、8 cores、row-block8、双槽 | 3.81e-6 |
| Matmul NT | B2/M32/N64/K96、K-block32、双槽 | 0 |
| Matmul NT | B8/M64/N64/K256、K-block64、双槽 | 0 |
| Matmul NT | B2/M64/N32/K160、K-block32、双槽；M不同于K-block | 0 |
| Matmul NT 单 K 块 | B1/M16/N32/K16、K-block16 | 0 |

RMSNorm/SwiGLU 采用 rtol=5e-5、atol=5e-6；Softmax 采用 rtol=5e-5、atol=2e-7。
RoPE 与 Matmul 这批输入要求零误差。Matmul 输入为非对称、可精确表示的二进制
分数，因此没有据此宣称任意 f16/f32 输入位级正确。没有放宽数值标准来接受
初版 `vrsqrt` 误差；旧失败产物保留在外层 `rms-small`，最终改用 Sqrt/Div。

原始目录包含 before/after IR、访问/代次 witness、验证报告、ASC、编译选项、
源码/ELF/共享库 SHA256、设备输出和健康快照。当前 runner 另记录每个参数的
SHA256，并检查运行库及当前生成源码与编译清单匹配；不会执行失效产物。

## 生成组织与固定性能比较

先审查生成代码：RMSNorm 每核只搬运一次 Gamma；Input/Work各保留两个 slot；
归约/statistics独立复用；最后一次 X reader 在 Gamma 前退休；上一代 Store
在当前 Scalar Wait 前发射。内核 stream 中没有 PIPE_ALL、额外 GM staging 或
隐式缓冲扩大。Matmul 各 K 块保留 MTE2/Load2D/MAD 的三级排步，L1 release
来自 MTE1、L0 release 来自 MAD，最终只有一次 Fixpipe 写出。

安装 ACLNN 是独立比较基线，不是编译器 fallback。RMSNorm 的公共源码审查
使用 `reference/ops-nn` HEAD `64b41ad63a5073bc711ba03802498bfe8bdb77e5` 的
`norm/rms_norm/op_kernel/rms_norm_merge_n.h:249–269`：Mul、ReduceSum、Muls、
Adds、Sqrt、Duplicate、Div。该参考 checkout 干净、未 pull。它与安装的
9.1库并非已证明逐字对应的同一 device 源码；具体基线是版本固定的安装库，
由产品 `cann_reference.py` 直接调用，保存库 hash、workspace和 profiler Op Name。

生成侧选项为 `-O2 -fPIC -std=c++20 -mllvm
-cce-aicore-dcpreload-args=false --npu-arch=dav-2201`；设备链接为
`ld.lld -m aicorelinux -Ttext 0 --no-mmap-output-file`，host库另用 `-shared`。
完整选项和 clang 版本存于每个 `.compile.json`，未无界调参。

双方各做一次 msprof，5次 warmup、随后30次计时，以下是设备 task duration
中位数，排除 host分配/复制和提交间隙。连续30次调用不视为30个独立试验，
也没有宣称冷缓存、随机输入或所有 shape 的性能。

| RMSNorm f32 shape | 生成核数/行粒度 | 生成中位数 µs | ACLNN核数 | ACLNN中位数 µs |
|---|---|---:|---:|---:|
| 256×1024 | 8 / 8 | 6.240 | 37 | 7.770 |
| 4096×4096 | 32 / 2 | 64.614 | 40 | 64.202 |

ACLNN 同时输出 Y 和辅助 Rstd，生成侧只输出 Y。两边 launch政策/输出契约
不同，因此这些数字不是专家排步等价性或全面性能无损证明。小 shape 的固定
生成政策较快，大 shape 与该安装基线接近；尚需更多算子、shape与等价政策
比较。首轮 Vector 手排设备 ELF 字节相同的证据仍限于原固定小例。

生成 RMSNorm 的 V/MTE2/MTE3 平均活动比例分别为：小 shape
0.8062/0.1935/0.2084，大 shape 0.9441/0.2661/0.2852。三者合计超过1，计数器
支持这些固定配置存在流水重叠；它们不代替逐指令时间线或一般性能证明。

对应原始目录为 `profile-rms-{256-1024,4096-4096}-{generated,reference}`。
`summary.json` 保留原始 CSV hash和全部30条时间。两组生成源码 SHA256为：

```text
256×1024:   53e4741a2396c19f701c36853d9d58f5ba3cc7e5691f1a0865fb37c768d61daf
4096×4096: 268a9cc9cd31c4eb41610782f1e52ca93f3725c4bcc825df1d037892743e9530
```

普通 runner 的 ACL stream interval 含提交间隙，另存 `device.json`；不能与
上述内核时间混用。

Softmax 256×1024 的首个固定8核政策，中位数8.082 µs；同 shape 安装的 ACLNN
Softmax 为37核、5.708 µs，两者均只输出 f32 Y。生成侧每核处理32行，基线约7行；
V平均活动时间分别6.1932/1.4890 µs，与核分配差异相符，不能把全部差异归因于
Scalar等待。唯一额外32核配置的中位数为4.040 µs（3.860…4.380），同样通过
数值检查；它支持核分配政策是该差距的重要因素。这时每核只有一次 tile
generation，不能从此配置宣称跨 generation 的流水重叠。原8核政策和双槽分配
均保留，未自动替换用户排步或无界调参。原始目录分别为
`profile-softmax-256-1024-{generated,reference,32cores}`。

32核配置可显式重现：

```bash
.venv/bin/xdsl-npu softmax --rows 256 --columns 1024 --blocks 32 --output runs/operators/softmax-256-1024-32cores --compile --shared
```

## 复现

从工作区根目录执行 README 的编译命令。RMSNorm 数值边界可对已编译目录运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 .venv/bin/python compiler/examples/run_device.py runs/operators/rms-small-sqrt-div --physical 1 --data-case numerical-edges --record-output runs/operators/rms-numerical-edges
```

msprof 通过真实 shell 脚本保持 venv解释器路径。固定采集命令如下，采集前
runner仍检查全部卡；参考侧把 `profile_device.sh` 换成 `profile_reference.sh`：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 \
XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python \
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof \
  --output=/workspace/xdsl-910b/runs/operators/profile-rms-256-1024-generated \
  --application='/workspace/xdsl-910b/compiler/examples/profile_device.sh /workspace/xdsl-910b/runs/operators/rms-256-1024-f32 --physical 1 --repeats 30' \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
.venv/bin/python compiler/examples/profile_summary.py runs/operators/profile-rms-256-1024-generated
```

进一步扩展的主要缺口：动态 task/间接 alias与 ownership，跨 AIC/AIV交接，
普通 GEMM 的 M/N层级复用及融合 FFN/attention。当前验证不包含这些能力。
