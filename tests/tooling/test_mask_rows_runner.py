"""Independent row-mask oracle, case parsing and large-width lane coverage."""

import numpy as np
import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.vector import run_mask_rows


def test_oracle_applies_prefix_complement_and_row_origin():
    x = run_mask_rows.field_tensor(2, 8)
    y0, y1, y2 = run_mask_rows.oracle(x, 1, 3, 5)
    cols = np.arange(8)
    assert np.array_equal(y0, np.where(cols < 5, np.float32(-7), x))
    assert np.array_equal(y1, np.where(cols != 3, np.float32(-9), x))
    expected_y2 = np.array(
        [[-11.0 if c <= r + 1 else x[r, c] for c in range(8)] for r in range(2)],
        dtype=np.float32,
    )
    assert np.array_equal(y2, expected_y2)
    # Untouched lanes retain the exact input bytes, and inputs are never fills.
    assert np.array_equal(y0[0, 5:], x[0, 5:]) and np.array_equal(y1[:, 3], x[:, 3])
    assert not np.isin(x, np.float32([-7.0, -9.0, -11.0])).any()


def test_oracle_is_exact_at_int64_max():
    x = run_mask_rows.field_tensor(2, 8)
    y0, y1, y2 = run_mask_rows.oracle(
        x, run_mask_rows.INT64_MAX, run_mask_rows.INT64_MAX, run_mask_rows.INT64_MAX
    )
    assert np.all(y0 == np.float32(-7)) and np.all(y1 == np.float32(-9))
    assert np.all(y2 == np.float32(-11))
    y0, y1, y2 = run_mask_rows.oracle(x, run_mask_rows.INT64_MAX, 0, 0)
    assert np.array_equal(y0, x) and np.array_equal(y1[:, 0], x[:, 0])
    assert np.all(y1[:, 1:] == np.float32(-9)) and np.all(y2 == np.float32(-11))


def test_large_width_case_covers_whole_blocks_and_a_63_lane_tail():
    x = run_mask_rows.field_tensor(1, 16448)
    y0, _, _ = run_mask_rows.oracle(x, 3, 100, 16447)
    assert np.count_nonzero(y0 != x) == 16447 == 256 * 64 + 63


def test_parse_cases_requires_nonnegative_int64_values():
    assert list(run_mask_rows.parse_cases(["3:100:127", "0:0:0"])) == [(3, 100, 127), (0, 0, 0)]
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_mask_rows.parse_cases(["-1:0:0"])
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_mask_rows.parse_cases([f"0:0:{run_mask_rows.INT64_MAX + 1}"])
