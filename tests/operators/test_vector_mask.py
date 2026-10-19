"""vector mask contracts."""

from __future__ import annotations

from itertools import product

import pytest

from ascendflow import compile
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.mask_predicates import _evaluate


def test_generic_mask_example_oracle_matches_source_predicates():
    import numpy as np

    from benchmarks.vector import run_masks
    from examples.vector.masks import masks

    result = compile(masks.specialize(ROWS=8, COLUMNS=128), timeout_s=180)
    rows, columns = 8, 128
    x, z = run_masks.field_tensors(rows, columns)
    for q, lo, hi in ((0, 0, 0), (7, 32, 96), ((1 << 63) - 1, 0, 0)):
        expected0, expected1, expected2 = run_masks.oracle(x, z, q, lo, hi)
        # Pure per-lane Python/NumPy logical reference, not the compiler.
        for r in range(rows):
            for c in range(columns):
                interval = lo <= c < hi
                periodic = (c // 64) % 2 == 0
                origin = (c + q) > r
                fill = (interval != periodic) and origin
                update = (x[r, c] * np.float32(2) if interval else x[r, c]) + np.float32(1)
                assert expected0[r, c] == (np.float32(-3) if fill else update)
                p = x[r, c] > np.float32(0.25)
                data = x[r, c] * np.float32(5)
                q_pred = data > z[r, c]
                both = bool(p) and bool(q_pred)
                either = both or not p
                selected = data if either else z[r, c]
                assert expected1[r, c] == selected
                assert expected2[r, c] == (np.float32(-9) if both else selected)
        module = parse(result.source_ir)
        fill_op = next(
            op for op in module.walk() if op.name == "npu_source.element" and op.kind.data == "fill"
        )
        program = fill_op.mask.owner.program()
        assert program.shape == (rows, columns)
        fill_mask = {
            index: _evaluate(program, fill_op.mask.owner, index, {"q": q, "lo": lo, "hi": hi}, {})
            for index in product(range(rows), range(columns))
        }
        for (r, c), value in fill_mask.items():
            interval = lo <= c < hi
            periodic = (c // 64) % 2 == 0
            origin = (c + q) > r
            assert value == ((interval != periodic) and origin)
    assert run_masks._parse_cases(["0:0:0", f"{(1 << 63) - 1}:3:9"])
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_masks._parse_cases(["-1:0:0"])
    with pytest.raises(CompileError, match="q:lo:hi"):
        run_masks._parse_cases(["1:2"])
