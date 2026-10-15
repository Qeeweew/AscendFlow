"""Deterministic inputs and an independent numerical reference."""

from xdsl_npu import language as N
from xdsl_npu.model import require


def inputs_and_oracle(policy, panels):
    import numpy as np

    m, n, bk, groups = (policy[key] for key in ("M", "N", "BK", "GROUPS"))
    require(panels * bk * 4 <= 1 << 24, "exact-oracle input bound exceeded")
    # Each product is an integer/64. The conservative bound above makes every
    # possible intermediate sum exactly representable in f32, regardless of
    # MAD's internal grouping. Non-square, nonsymmetric inputs test the layout.
    a = ((np.arange(5 * m * bk, dtype=np.int64) * 3 + 1) % 5 - 2).reshape(5, m, bk)
    br, bc = (n, bk) if policy["TRANSPOSE_B"] else (bk, n)
    b = ((np.arange(5 * br * bc, dtype=np.int64) * 2 + 3) % 5 - 2).reshape(5, br, bc)
    products = a @ (b.transpose(0, 2, 1) if policy["TRANSPOSE_B"] else b)
    expected = np.zeros((groups, m, n), dtype=np.int64)
    for group in range(groups):
        start = group * panels
        for residue in range(5):
            occurrences = (start + panels - 1 - residue) // 5 - (start - 1 - residue) // 5
            expected[group] += occurrences * products[residue]
    selection = np.arange(groups * panels) % 5

    def pack(values):
        values = values.astype(np.float32) / 8
        encoded = (
            values.astype(np.float16)
            if policy["DTYPE"] == N.f16
            else (values.view(np.uint32) >> 16).astype(np.uint16)
        )
        return encoded[selection].tobytes()

    return pack(a), pack(b), expected.astype(np.float32) / 64
