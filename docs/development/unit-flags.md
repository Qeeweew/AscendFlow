# L0C unitflag 合同

910B / dav-2201 的 MAD 与 FIX 可以在 L0C 的计算、拷出之间使用硬件细粒度握手。
DSL 仍显式描述全部内存访问：

```python
for k in T.Pipelined(tiles * panels):
    with T.Stage(1):
        T.matmul(a, b, accum=c, initialize=k % panels == 0,
                 unit_flag=3 if k % panels == panels - 1 else 2)
        if k % panels == panels - 1:
            T.copy(c, out, unit_flag=3)
```

`matmul` 的缺省值为常量 `0`；启用时实际 index SSA 只能取 `2` 或 `3`。
`2` 检查可写许可、保持本轮累加，`3` 还发布本轮结果；FIX 的常量 `3` 检查
结果可读并在读取后归还许可。K 清零与 flag 发布是两个独立条件。
后端直接使用 CANN 的 `asc_mmad` 与 `asc_copy_l0c2gm` 参数。

`analysis/unit_flags.py` 在 access order 构造前消费已建立的 memory/dispatch，
一次核验以下前提：

- AIC 私有、固定地址、完整连续的同一 L0C 平面；当前不支持旋转 L0C 槽。
- 一组中间检查、一个可能更新的 MAD、一个 FIX，位于同一实际 scope/stage，
  按真实派发次序构成发布和消费的交替 word。不能交错其他平面的累加。
- 更新 MAD 与 FIX 的实际激活完全相同；flag 范围和激活等价使用已有整数事实。
- FIX 消费完整 M/N fractal 和全部已发布行；部分行、部分平面、同 scope 内
  共同执行的未标记访问，以及缺少发布或消费均拒绝。

原始 RAW/WAR/WAW、native effects、值版本和初始化义务全部保留。已核验的条件
FIX reader 保留真实 native 接口，避免为它建立提前读取 metadata 的分支端口。
跨调用、混合其他访问的边界继续使用原来的队列接口与事件。

notification binding 将对应 RAW/WAR 记录为独立 `UnitFlagCoverage`，引用原始
binding 和本次拥有的 word。它不产生 `OrderingWitness`、整个 MAD 的完成事实、
queue prefix、事件消费或额外 event ID。L0A/L0B 最后 reader、M↔MTE1 和最终
输出退休仍遵守原合同。资源台账不会因设置 flag 而扩大。

独立有限模型直接解释原生 flag SSA 和实际 L0C 字节，检查发布/消费 word。
它使用独立 L0C access 节点表达握手，不建立 `MAD.end → FIX.begin`；相应负例
与空、多轮、重入测试见 `tests/sync/test_unit_flags.py`。

依据为安装的 CANN 9.1.0 `npu_arch_2201` 的公开 C API 及其直接 intrinsic 映射，
以及 CATLASS 上游提交 `698379e0d4b1b94d12728e7181df2cf83a958e8f` 的
`include/catlass/gemm/block/block_mmad_pingpong.hpp`。参考源码仅用于核对协议，
不成为产品依赖；设备行为和收益以实际产物验收。
