"""mask predicates contracts."""

from __future__ import annotations

from itertools import product

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from tests.fixtures.masks import (
    _masked_fill,
)
from tests.models.mask_predicates import mask_values


@pytest.mark.parametrize(
    "expression,environment,expected",
    [
        (
            "(cols + q) > rows",
            {"q": 5, "lo": 0, "hi": 0},
            lambda r, c, e: (c + 5) > r,
        ),
        (
            "((cols >= lo) & (cols < hi)) & ((rows % 3) == 1)",
            {"q": 0, "lo": 70, "hi": 190},
            lambda r, c, e: 70 <= c < 190 and r % 3 == 1,
        ),
        (
            "((cols + 17) % 5) == 2",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c, e: (c + 17) % 5 == 2,
        ),
        (
            "((cols % 2) == 0) | ((rows + q) < cols)",
            {"q": -1000, "lo": 0, "hi": 0},
            lambda r, c, e: c % 2 == 0 or (r - 1000) < c,
        ),
        (
            "~((cols + q) > rows)",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c, e: not (c > r),
        ),
    ],
)
def test_coordinate_predicate_values_broadcast_and_boolean(expression, environment, expected):
    result = compile(_masked_fill(expression), timeout_s=120)
    truth = mask_values(result, environment)
    assert truth
    for shape, lanes in truth[-1:]:
        assert shape == (16, 256)
        for (r, c), value in lanes.items():
            assert value == expected(r, c, environment), (r, c)


@pytest.mark.parametrize(
    "expression,environment,expected",
    [
        ("(cols + q) > rows", {"q": 0, "lo": 0, "hi": 0}, lambda r, c: c > r),
        (
            "((cols >= lo) & (cols < hi))",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c: False,
        ),
        (
            "((cols >= lo) & (cols < hi))",
            {"q": 0, "lo": 0, "hi": 256},
            lambda r, c: True,
        ),
        (
            "(cols + 9223372036854775807) > (rows + (-9223372036854775807 - 1))",
            {"q": 0, "lo": 0, "hi": 0},
            lambda r, c: (c + (1 << 63) - 1) > (r - (1 << 63)),
        ),
    ],
)
def test_predicate_empty_all_and_extreme_signed_origins(expression, environment, expected):
    result = compile(_masked_fill(expression), timeout_s=120)
    _shape, lanes = mask_values(result, environment)[-1]
    assert tuple(lanes) == tuple(product(range(16), range(256)))
    assert all(value == expected(r, c) for (r, c), value in lanes.items())


@pytest.mark.parametrize(
    "shape,message",
    [((8, 100), "whole 64-lane"), ((3, 40), "whole 64-lane")],
)
def test_non_whole_lane_domains_are_rejected(shape, message):
    rows, cols = shape
    program = N.from_source(
        f"""
        def kernel(X:N.Tensor(({rows},{cols}),N.f32), Y:N.Tensor(({rows},{cols}),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub(({rows},{cols}),N.f32)
                bits=N.alloc_mask(({rows},{cols}))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises((CompileError, Exception), match=message):
        compile(program, timeout_s=60)


def test_masked_conversion_and_row_broadcast_are_rejected():
    program = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f16)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.copy(X,work)
                N.cast(work,out=other,mask=(cols + 1) > rows)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="masked row broadcast and conversion"):
        compile(program, timeout_s=60)


def test_where_requires_materialized_predicate_and_defined_branches():
    symbolic = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                rows=N.arange(8)[:, None]
                cols=N.arange(128)[None, :]
                N.copy(X,work)
                N.copy(X,other)
                N.where((cols > rows),work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="materialized data predicate"):
        compile(symbolic, timeout_s=60)
    undefined = N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.0,kind="gt",out=bits)
                N.where(p,work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(undefined, timeout_s=60)


def test_exact_integer_limits_reject_unsafe_forms_and_emit_floor_division():
    def fill_expression(expression):
        return N.from_source(
            f"""
            def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write"),
                       q:N.int64):
                with N.Kernel(1, kind="vector"):
                    work=N.alloc_ub((8,128),N.f32)
                    N.copy(X,work)
                    rows=N.arange(8)[:, None]
                    cols=N.arange(128)[None, :]
                    N.fill(work,-1.0,mask={expression})
                    N.copy(work,Y)
            """,
            disjoint_external=True,
        )

    extreme = compile(
        fill_expression("(cols + 9223372036854775807) > (rows + (-9223372036854775807 - 1))"),
        timeout_s=120,
    )
    # These exact extremes prove every lane true, so the admitted whole-tile
    # coverage replays one repeated full-mask run and evaluates no word.
    assert "npu_mask_monotone((__int128)(" not in extreme.asc
    assert all(mask_values(extreme, {})[-1][1].values())
    # A runtime atom keeps the tile mixed, so the extreme int64 literal is
    # still emitted inside the exact 128-bit word arithmetic.
    mixed_extreme = compile(
        fill_expression(
            "((cols + q) + (-9223372036854775807 - 1)) > (rows + (-9223372036854775807 - 1))"
        ),
        timeout_s=120,
    )
    assert "(-9223372036854775807LL - 1)" in mixed_extreme.asc
    assert "npu_mask_monotone(" in mixed_extreme.asc
    with pytest.raises(CompileError, match="int64"):
        compile(fill_expression("(cols * 9223372036854775808) > rows"), timeout_s=60)
    with pytest.raises(CompileError, match="lane coefficient"):
        compile(
            fill_expression("(cols * 4611686018427387904) > rows"),
            timeout_s=60,
        )
    with pytest.raises(CompileError, match="positive int64 divisor"):
        compile(fill_expression("(cols // 9223372036854775808) == 0"), timeout_s=60)
    negative = compile(
        fill_expression("((rows - 5) % 2) == 1"),
        timeout_s=120,
    )
    assert "npu_mask_floormod(" in negative.asc
    truth = mask_values(negative, {})[-1][1]
    assert all(value == ((r - 5) % 2 == 1) for (r, _c), value in truth.items())
    floor = compile(
        fill_expression("((cols - 7) // 5) == -1"),
        timeout_s=120,
    )
    assert "npu_mask_floordiv(" in floor.asc
    truth = mask_values(floor, {})[-1][1]
    assert all(value == ((c - 7) // 5 == -1) for (_r, c), value in truth.items())


def test_scalar_and_periodic_combinations_accept_both_operand_orders(tmp_path):
    for expression in (
        "((q + rows) > 0) & (((cols // 8) % 2) == 0)",
        "(((cols // 8) % 2) == 0) & ((q + rows) > 0)",
        "(((-cols) % 5) == 0) & (rows >= 0)",
    ):
        result = compile(_masked_fill(expression), timeout_s=120)
        truth = mask_values(result, {"q": 3, "lo": 0, "hi": 0})[-1][1]
        for (r, c), value in truth.items():
            if "// 8" in expression:
                scalar_first = "(q + rows) > 0" in expression.split("&")[0]
                expected = (
                    (3 + r) > 0 and (c // 8) % 2 == 0
                    if scalar_first
                    else (c // 8) % 2 == 0 and (3 + r) > 0
                )
            elif "% 5" in expression:
                expected = (-c) % 5 == 0 and r >= 0
            else:
                expected = False
            assert value == expected, (expression, r, c)
