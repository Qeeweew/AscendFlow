"""Generic vector predicates, preserve-update masks and packed compare/select.

The same DSL entry covers non-triangular coordinate masks: an interval, a
genuinely adjacent-chunk-different 64-lane periodic predicate, their XOR, and a
nontrivial runtime-origin condition ``(cols + q) > rows``. The nonnegative
runtime origin ``q`` is tested at ordinary values and at ``INT64_MAX``; the
compiler evaluates the sum exactly in signed 128-bit arithmetic, so the oracle
derives the same condition from Python integers. Empty/all/partial intervals and
an interleaved full-count op are covered. Data comparisons, materialized
NOT/OR/AND and a ``where`` select are arranged so that NOT, OR and AND each
change an observable output lane. An earlier data comparison is consumed after
its source tile is overwritten, which checks that the compare result is a real
earlier computation rather than a deferred expression.

``Y0`` is the coordinate-mask result, ``Y1`` the compare/where/boolean result
and ``Y2`` the data-predicate preserve-update result.
Scratch and packed predicate storage are explicit allocations. The device
oracle is independent NumPy/Python logic, never the compiler's predicate
interpreter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ruff: noqa: F821 - DSL signature annotations resolve at capture time.
from ascendflow import language as N
from ascendflow.build import build_file


@N.prim_func(disjoint_external=True)
def masks(
    X: N.Tensor((ROWS, COLUMNS), N.f32),
    Z: N.Tensor((ROWS, COLUMNS), N.f32),
    Y0: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y1: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    Y2: N.Tensor((ROWS, COLUMNS), N.f32, access="write", write_disjoint=True),
    q: N.int64,
    lo: N.int64,
    hi: N.int64,
    ROWS: N.constexpr = 8,
    COLUMNS: N.constexpr = 128,
):
    with N.Kernel(1, kind="vector"):
        assert COLUMNS % 64 == 0, "packed predicates need whole 64-lane blocks"
        assert ROWS > 0, "at least one row is required"
        coord_work = N.alloc_ub((ROWS, COLUMNS), N.f32, name="coord_work")
        data_work = N.alloc_ub((ROWS, COLUMNS), N.f32, name="data_work")
        z_work = N.alloc_ub((ROWS, COLUMNS), N.f32, name="z_work")
        scratch = N.alloc_ub((ROWS, COLUMNS), N.f32, name="scratch")
        results = N.alloc_ub((ROWS, COLUMNS), N.f32, name="results")
        fill_work = N.alloc_ub((ROWS, COLUMNS), N.f32, name="fill_work")
        bits_p = N.alloc_mask((ROWS, COLUMNS), name="bits_p")
        bits_q = N.alloc_mask((ROWS, COLUMNS), name="bits_q")
        bits_not = N.alloc_mask((ROWS, COLUMNS), name="bits_not")
        bits_or = N.alloc_mask((ROWS, COLUMNS), name="bits_or")
        bits_and = N.alloc_mask((ROWS, COLUMNS), name="bits_and")
        N.copy(X, coord_work)
        N.copy(X, data_work)
        N.copy(Z, z_work)
        rows = N.arange(ROWS)[:, None]
        cols = N.arange(COLUMNS)[None, :]
        interval = (cols >= lo) & (cols < hi)
        # Adjacent 64-lane chunks differ, unlike a predicate whose period
        # repeats inside one packed lane block.
        periodic = ((cols // 64) % 2) == 0
        origin = (cols + q) > rows
        # XOR keeps both interval and non-interval lanes outside the fill, so
        # preserved updated (interval) and untouched lanes are both observable.
        different = (interval | periodic) & ~(interval & periodic)
        boolean = different & origin
        # Preserve-update masked arithmetic, then an ordinary count op, then a
        # masked fill whose false lanes must keep the count result.
        N.mul(coord_work, 2.0, out=coord_work, mask=interval)
        N.add(coord_work, 1.0, out=coord_work)
        N.fill(coord_work, -3.0, mask=boolean)
        N.copy(coord_work, Y0)
        # Data predicates: p is bound before data_work is overwritten.
        p = N.compare(data_work, 0.25, kind="gt", out=bits_p)
        N.mul(data_work, 5.0, out=data_work)
        q_pred = N.compare(data_work, z_work, kind="gt", out=bits_q)
        n = N.mask_not(p, out=bits_not)
        both = N.mask_and(p, q_pred, out=bits_and)
        # either is neither q_pred&p alone nor ~p alone: true-p lanes need the
        # AND, false-p lanes need the NOT+OR, so all three are observable.
        either = N.mask_or(both, n, out=bits_or)
        N.where(either, data_work, z_work, out=results)
        # A data-predicate preserve update reuses explicit scratch storage.
        N.copy(results, fill_work)
        N.fill(fill_work, -9.0, mask=both, scratch=scratch)
        N.copy(results, Y1)
        N.copy(fill_work, Y2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=128)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report = build_file(
        Path(__file__),
        args.output,
        entry="masks",
        parameters={"ROWS": args.rows, "COLUMNS": args.columns},
        compile_asc=args.compile,
        shared=args.shared,
        timeout_s=args.timeout,
    )
    print(report["compilation"])


if __name__ == "__main__":
    main()
