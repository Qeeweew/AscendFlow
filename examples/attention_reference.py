"""Installed CANN Flash Attention baseline for this example."""

import ctypes as ct

from xdsl_npu.cann_reference import OperatorReference


class FlashAttentionReference(OperatorReference):
    """Installed CANN dense BNSD kernel, with its own tiling/rounding choices."""

    interface = "aclnnFlashAttentionScore BNSD, keepProb1, innerPrecise0, sparseMode0"

    def __init__(self, runtime, q, k, v, output, shape, dtype, scale):
        super().__init__(runtime)
        batch, heads, kv_heads, sq, sk, dim = shape
        code = {"f16": 1, "bf16": 27}[dtype]
        stat_shape = (batch, heads, sq, 8)
        maximum = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        denominator = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        self.auxiliary_bytes = 2 * batch * heads * sq * 8 * 4
        self.prepare(
            "aclnnFlashAttentionScore",
            [
                self.tensor(q, (batch, heads, sq, dim), dtype=code),
                self.tensor(k, (batch, kv_heads, sk, dim), dtype=code),
                self.tensor(v, (batch, kv_heads, sk, dim), dtype=code),
                None,
                None,
                None,
                None,
                None,
                scale,
                1.0,
                2**31 - 1,
                2**31 - 1,
                heads,
                b"BNSD",
                0,
                0,
                self.tensor(maximum, stat_shape),
                self.tensor(denominator, stat_shape),
                None,
                self.tensor(output, (batch, heads, sq, dim), dtype=code),
            ],
            [ct.c_void_p] * 8
            + [ct.c_double] * 2
            + [ct.c_int64] * 3
            + [ct.c_char_p]
            + [ct.c_int64] * 2
            + [ct.c_void_p] * 4,
        )
