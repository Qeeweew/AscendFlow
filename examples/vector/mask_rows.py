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

import argparse
from pathlib import Path

# ruff: noqa: F821 - DSL signature annotations resolve at capture time.
from ascendflow import language as N
from ascendflow.build import build_file


@N.prim_func(disjoint_external=True)
def mask_rows(
    X: N.Tensor((ROWS, COLUMNS), N.f32),
    Y0: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y1: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y2: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    q: N.int64,
    lo: N.int64,
    hi: N.int64,
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
        N.fill(work, -7.0, mask=cols < hi)
        N.copy(work, Y0)
        # Complement of one runtime point: cols != lo.
        N.copy(X, work)
        N.fill(work, -9.0, mask=cols != lo)
        N.copy(work, Y1)
        # Negative unit comparison with a per-row origin: rows >= cols - q.
        # A one-row tile has no non-singleton row lane axis, so the same
        # predicate is written with its degenerate row origin (row 0).
        N.copy(X, work)
        if ROWS == 1:
            N.fill(work, -11.0, mask=cols <= q)
        else:
            N.fill(work, -11.0, mask=rows >= cols - q)
        N.copy(work, Y2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report = build_file(
        Path(__file__),
        args.output,
        entry="mask_rows",
        parameters={"ROWS": args.rows, "COLUMNS": args.columns},
        compile_asc=args.compile,
        shared=args.shared,
        timeout_s=args.timeout,
    )
    print(report["compilation"])


if __name__ == "__main__":
    main()
