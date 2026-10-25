"""Small row-mask example: one f32 UB tile, three coordinate-prefix outputs.

Each result starts from an independent copy of ``X`` into the single work tile,
then one masked fill. ``Y0`` is a column prefix ``cols < hi``, ``Y1`` is the
complement of a runtime point ``cols != lo``, and ``Y2`` is the negative-unit
row predicate ``rows >= cols - q`` (a per-row origin). Runtime ``q``, ``lo``
and ``hi`` are nonnegative int64 sizes; the compiler evaluates the lane
comparison exactly, including values near ``INT64_MAX``. Untouched lanes keep
the input value, so a wrong mask is bit-visible.

The default shape is 4x256; one tile also covers 1x16448 (64 KiB UB, below
192 KiB) for width-heavy evidence. Device validation lives in benchmarks.
"""

from __future__ import annotations

# ruff: noqa: F821 - DSL signature annotations resolve at capture time.
from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def mask_rows(
    X: N.Tensor((ROWS, COLUMNS), N.f32),
    Y0: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y1: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y2: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    q: N.i64,
    lo: N.i64,
    hi: N.i64,
    ROWS: N.constexpr = 4,
    COLUMNS: N.constexpr = 256,
):
    with N.Kernel(1, kind="vector"):
        assert ROWS > 0 and COLUMNS > 0, "positive row and column extents are required"
        work = N.alloc_ub((ROWS, COLUMNS), N.f32, name="work")
        rows = N.arange(ROWS)[:, None]
        cols = N.arange(COLUMNS)[None, :]
        # Column prefix: cols < hi.
        N.copy(X, work)
        prefix = N.fill(work, -7.0, mask=cols < hi)
        N.copy(prefix, Y0)
        # Complement of one runtime point: cols != lo.
        N.copy(X, work)
        complement = N.fill(work, -9.0, mask=cols != lo)
        N.copy(complement, Y1)
        # Negative unit comparison with a per-row origin: rows >= cols - q.
        # A one-row tile has no non-singleton row lane axis, so the same
        # predicate is written with its degenerate row origin (row 0).
        N.copy(X, work)
        if ROWS == 1:
            diagonal = N.fill(work, -11.0, mask=cols <= q)
        else:
            diagonal = N.fill(work, -11.0, mask=rows >= cols - q)
        N.copy(diagonal, Y2)
