# P0独立AscendC FA参考

这是[目标源码](attention-target.md)的独立手排参考，文件为
[ascendc_attention.asc.in](../examples/ascendc_attention.asc.in)，构建/运行入口为
[run_attention_policy.py](../examples/run_attention_policy.py)。不调用编译器FA
builder、AutoSync或codegen，不把生成ASC改名作为参考。新AST FA仍未接通。

## 固定政策与本次发现

BNSD，D128/BQ128/BK1024/CK128/VR8，b16输入和输出，FP32在线m/l/O，p32先归约
再舍入为P，PV按CK128顺序累计，最后reciprocal+mul。B/Hq/Hkv/groups/dtype是
编译常量，Sq/Sk是runtime。同一binary跨长度；GM workspace只随groups和双槽
增长。支持完整tile的dense MHA/GQA/MQA地址关系；实际设备验证配置见下表，
causal、tail和动态任务不在本参考范围。

本次发现旧fixture的内层排步不可行：L0B只有一槽，若同step先发operand[g+1]
再发MAD[g]，会覆写旧输入；插入release等待则形成等待环。PV单槽L0A/B同理。
目标源码与参考现明确固定：

```text
QK child: load_k[step] → MAD[step-2] → operand_k[step-1] → FIX[step-3]
PV child: load_pv[step] → MAD[step-2] → operand_pv[step-1]
```

逻辑源语句仍是load/operand/MAD/export，通过显式issue_order规定实际派发。
没有增加buffer或偷偷改变政策；原默认顺序作为负例保留。外层仍为AIC新QK后
旧PV、AIV旧Update后新Softmax，query边界保持连续，不每query重新seed/drain。

与ops-transformer arch22的差异：此参考采用update-before-softmax政策，而当前
`Process`源码有Vec1-before-Vec2；采用显式native原语，没有复刻高级Matmul库内部
调度或其全部cache策略。它用于隔离未来编译器在此固定政策上的差距，不能
代替安装ACLNN的竞争基线，也不声称当前政策已达到专家性能。

## 物理资源与手排协议

| 资源 | 明确占用 |
|---|---|
| 每AIC L1 | 224 KiB，Q32 KiB，K/P/V各双槽64 KiB |
| 每AIC L0A/B | 各64 KiB，Q/P与K/V各单独32 KiB |
| 每AIC L0C | 128 KiB，QK双平面，PV复用平面0 |
| 每AIV UB | 178304字节，独立于另一AIV |
| 每group GM | 双槽Score/P/PV，1 MiB+512 KiB+128 KiB |
| cross-core | mode2 IDs0…5，Score/P/PV各ready/free一对计数channel |
| M_MTE1 | TPipe预置ID0…2保持不动；Q、K、PV operands使用3/4/5 |
| MTE2_MTE1与MTE1_MTE2 | K0/1、P2/3、V4/5、Q6 |

每个local Set/Wait的bank独立，两个AIV也拥有各自bank。MTE1读退休归还L1，
MAD退休归还L0 operand，FIX退休归还L0C。GM release紧接最后实际MTE2读：
Score在最后行import后发布，P在最后chunk的P加载后、V加载之前发布，PV在
最后行import后发布。不是用整个softmax/PV/update结束替代最后读取。

cross-ready捕获全部必要FIX或两AIV的MTE3贡献。只有`t+2<total`才发送复用
credit，接收方只在`t>=2`等待；零任务不发送或等待。所有剩余local credit在
kernel出口消费，再做每actor最终PIPE_ALL；没有子循环或query边界PIPE_ALL。
最大counter backlog需遵循硬件契约，有限检查中的峰值≤2，不冒充任意N证明。

`TPipe`预置事件依据安装CANN9.1；参考使用AscendC的原生接口，仅S→FIX配置
握手因HardEvent枚举缺失而直接调用set_flag/wait_flag。最大值状态通过int32
Adds(0)保持位模式，避免浮点加零改变数值位；未来生成指令是否相同另行审查。

## 验证证据

纯Python/IR与参考控制相关75项回归通过（11.03秒），Ruff通过。测试从真实
C++源码替换测试头文件后在host执行控制分支，记录实际Set/Wait/通知和pipe调用；
每个有限case使用16个随机合法FIFO退休顺序。覆盖空owner、多个query、K=1/2/3、
双AIV贡献、binary复用和尾部消费；漏一个AIV贡献/本地ready必须失败。
另从目标AST提取inner issue_order，旧默认顺序在2块输入时暴露单槽覆盖。
这不是任意规模证明，也不模拟浮点或完整物理访问，不能替代编译器最终checker。

CANN9.1/dav-2201离线和shared编译通过。设备运行均先查全卡，物理1→逻辑0，
没有设备错误/reset/换卡。每次完整数值检查同时检查所有输入不变及所有GM
分配的guard。NumPy oracle固定BK1024与P舍入；BLAS累计及exp/reduce误差与native
不同，不声称位级仿真。容差f16为rtol0.002/atol0.0002，bf16为0.01/0.001。

| 配置 | 检查范围 | 最大绝对误差 |
|---|---|---|
| f16 B1/H1/Hkv1/G1 | Sq0/128/256/384/896，Sk1024/3072/7168；非笛卡尔组合 | ≤3.05176e-5 |
| f16均匀score | Sq128/Sk1024，全输出 | 0 |
| bf16 B1/H8/Hkv2/G20 | Sq128/384/896，Sk1024/3072/7168；全输出 | ≤2.44141e-4 |
| 同bf16，Q/K放大3倍 | Sq128/Sk1024与Sq384/Sk3072；全输出 | ≤0.00390625，在声明相对/绝对容差内 |
| bf16 B1/H32/Hkv8/G20，Sq=Sk8192 | 33,554,432输出全部finite；每head每query块抽6行 | ≤1.22070e-4，采样数值证据 |

运行数据在外层`runs/refactor/attention-reference/`，包含源码/库hash、CANN完整
选项、输入hash、设备及计时口径。`regressions.log`为本次回归；f16-g1、bf16-g20
分别存小例，bf16-large存大例。新AST FA没有借用这些设备支持标记。

## 首批大例性能与硬件分析

BF16 8K GQA在20 AIC/40 AIV的910B3上，warmup5/repeats30，**一批profile**设备
时长p50=18.641612 ms，范围18.592592…18.684216 ms。是热重复分配的固定参考
基线；没有三批独立比较、buffer轮换或与ACLNN/新生成代码对照，不作为性能验收。

[分析脚本](../examples/analyze_attention_policy.py)绑定实际CSV、source/library
hash与运行合同后计算：262144次128³ MAD，共1,099,511,627,776 FLOPs；GM请求
36,641,439,744字节，其中Score往返16 GiB、P往返8 GiB、PV往返2 GiB。请求速率
约1965.6 GB/s，**不是HBM实测带宽**。IO+workspace为192.5 MiB，仅略大于实机
192 MiB L2，不能因此视为冷HBM测试。

按20核×1.8 GHz×8192 ops/core/cycle的假设，Cube峰值294.912 TFLOP/s、仅Cube
理想下界3.72827 ms；当前有效吞吐约58.98 TFLOP/s（条件峰值的20%）。还需处理
2,147,483,648个probability exp元素及归约、广播和输出递推，不能只用Cube
下界预测完整FA延迟。profile中AIC MAC约0.226、Scalar0.324、AIV Vector0.672，
这些活动率可重叠；不能单独据此认定瓶颈。Vector、数据路径和CV依赖的差距
尚未归因，持续HBM/L2/Vector吞吐仍未知，没有恢复算子调参。

## 复现

在工作区根目录先做纯CPU检查，再离线构建：

```bash
.venv/bin/python -m pytest -q compiler/tests/test_attention_policy.py compiler/tests/test_attention_target.py
.venv/bin/python compiler/examples/run_attention_policy.py --output runs/refactor/attention-reference/f16-g1 --heads 1 --kv-heads 1 --groups 1 --build
```

选择健康空闲物理卡后（已有记录为1），执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python compiler/examples/run_attention_policy.py --output runs/refactor/attention-reference/f16-g1 --heads 1 --kv-heads 1 --groups 1 --physical 1 --cases 128:1024 256:1024 384:3072
```

大例需按B1/H32/Hkv8/G20/bf16单独`--build`，Sq/Sk仍是同binary的runtime参数。
使用`profile_attention_policy.sh`给msprof提供venv入口，避免profiler解析Python
符号链接后丢失venv。例如对已经构建的大例运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 XDSL_NPU_PYTHON=/workspace/xdsl-910b/.venv/bin/python /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --output=/workspace/xdsl-910b/runs/refactor/attention-reference/bf16-large/profile --application='/workspace/xdsl-910b/compiler/examples/profile_attention_policy.sh --output /workspace/xdsl-910b/runs/refactor/attention-reference/bf16-large --heads 32 --kv-heads 8 --groups 20 --dtype bf16 --physical 1 --cases 8192:8192 --scope sampled --warmup 5 --repeats 30 --record /workspace/xdsl-910b/runs/refactor/attention-reference/bf16-large/device-large.json' --task-time=on --ai-core=on --aic-metrics=PipeUtilization
```

分析已完成profile：

```bash
.venv/bin/python compiler/examples/profile_summary.py runs/refactor/attention-reference/bf16-large/profile
.venv/bin/python compiler/examples/analyze_attention_policy.py runs/refactor/attention-reference/bf16-large/device-large.json runs/refactor/attention-reference/bf16-large/profile --output runs/refactor/attention-reference/bf16-large/analysis.json
```

P0的参考及政策差异已落地；下一步按P1a收敛actor绑定/region入口，接通父体
init→child→use。同步规划、生成代码和大例对照仍是后续重构要求，不能以完成
手排参考宣布新编译器已支持FA。
