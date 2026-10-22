"""Arithmetic admission and control share the actual serial iteration domain."""

import pytest

from ascendflow import language as N
from ascendflow.dialects.source import ProgramOp
from ascendflow.indexing import validate_static_arithmetic
from ascendflow.passes.lower_pipeline import normalize_loop_controls


def program(lower, upper, expression, *, step=1):
    module = N.from_source(f"""
        def domain(n: N.i64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial({lower},{upper},{step}):
                    unused={expression}
                    N.fill(x,0.0)
    """).to_ir()
    return module, next(op for op in module.walk() if isinstance(op, ProgramOp))


def test_nonzero_lower_bound_is_used_before_unsigned_division_simplification():
    module, source = program(3, 10, "(i-3)//2", step=2)
    controls = normalize_loop_controls(module)
    (control,) = controls.loops
    assert control.domain.concrete({}) == (3, 10, 2)
    validate_static_arithmetic(source)


def test_dynamic_serial_dividend_definedness_is_an_input_contract():
    _, source = program(-1, 3, "i//2")
    validate_static_arithmetic(source)


@pytest.mark.parametrize("lower,upper", [(3, 3), (3, 0), (-2, -3), ("n", "n")])
def test_empty_range_has_no_executed_division_even_when_upper_is_nonzero(lower, upper):
    _, source = program(lower, upper, "(i-8)//2")
    validate_static_arithmetic(source)


def test_uniform_runtime_lower_is_preserved_in_the_range_proof():
    _, source = program("n", "n+8", "(i-n)//2", step=3)
    validate_static_arithmetic(source)


@pytest.mark.parametrize("origin", ["i-3", "i-8"])
def test_unrelated_scalar_carry_does_not_hide_or_require_division_premises(origin):
    module = N.from_source(f"""
        def carried(n: N.i64):
            with N.Kernel():
                total=7
                for k in N.serial(n):
                    total=total+1
                for t in N.Pipelined(total):
                    pass
                for i in N.serial(3,7):
                    unused=({origin})//2
""").to_ir()
    source = module.body.block.first_op
    validate_static_arithmetic(source)
