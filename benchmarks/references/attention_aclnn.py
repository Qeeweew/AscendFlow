"""Installed CANN Flash Attention baseline for this example."""

import ctypes as ct

from ascendflow.diagnostics import require
from benchmarks.references.aclnn import OperatorReference


class FlashAttentionReference(OperatorReference):
    """Installed CANN dense BNSD kernel, with its own tiling/rounding choices."""

    interface = "aclnnFlashAttentionScore BNSD, keepProb1, innerPrecise0, sparseMode0"

    def __init__(
        self, runtime, q, k, v, output, shape, dtype, scale, *, causal=False, query_start=0
    ):
        super().__init__(runtime)
        batch, heads, kv_heads, sq, sk, dim = shape
        require(
            not causal or query_start in {0, sk - sq},
            "ACLNN causal comparison supports left-up or right-down alignment only",
        )
        code = {"f16": 1, "bf16": 27}[dtype]
        mask, sparse_mode = None, 0
        if causal:
            import numpy as np

            # True means masked; mode 2/3 consumes the compressed 2048² mask.
            mask = self.tensor(
                runtime.allocate(np.triu(np.ones((2048, 2048), dtype=np.bool_), k=1).tobytes()),
                (2048, 2048),
                dtype=12,
            )
            sparse_mode = 2 if query_start == 0 else 3
        self.interface = (
            f"aclnnFlashAttentionScore BNSD, keepProb1, innerPrecise0, sparseMode{sparse_mode}"
        )
        stat_shape = (batch, heads, sq, 8)
        maximum = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        denominator = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        self.auxiliary_bytes = 2 * batch * heads * sq * 8 * 4 + (2048**2 if causal else 0)
        self.prepare(
            "aclnnFlashAttentionScore",
            [
                self.tensor(q, (batch, heads, sq, dim), dtype=code),
                self.tensor(k, (batch, kv_heads, sk, dim), dtype=code),
                self.tensor(v, (batch, kv_heads, sk, dim), dtype=code),
                None,
                None,
                None,
                mask,
                None,
                scale,
                1.0,
                2**31 - 1,
                2**31 - 1,
                heads,
                b"BNSD",
                0,
                sparse_mode,
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
