"""Deterministic inputs and an independent numerical reference."""

from xdsl_npu.model import require


def inputs_and_oracle(groups, count, tile, *, rounding=False):
    import numpy as np

    if rounding:
        require(count == 2, "rounding counterexample uses two ordered additions")
        initial = np.full((groups, tile), 2**24, dtype=np.float32)
        x = np.empty((groups, 2, tile), dtype=np.float32)
        x[:, 0, :], x[:, 1, :] = 1.0, -(2**24)
        expected = np.zeros_like(initial)  # (2**24 + 1) rounds first, then subtract 2**24.
    else:
        require(0 <= count <= 65536, "exact dyadic oracle length exceeds its bound")
        rng = np.random.default_rng(9102201 + count)
        initial = rng.integers(-31, 32, size=(groups, tile), dtype=np.int16).astype(np.float32) / 32
        x = (
            rng.integers(-31, 32, size=(groups, count, tile), dtype=np.int16).astype(np.float32)
            / 32
        )
        expected = (initial.astype(np.float64) + x.sum(axis=1, dtype=np.float64)).astype(np.float32)
    return initial.tobytes(), x.tobytes(), expected
