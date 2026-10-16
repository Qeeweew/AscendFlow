"""Arithmetic admission and control share the actual serial iteration domain."""

import pytest

from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.source import ProgramOp
from xdsl_npu.indexing import validate_source_arithmetic
from xdsl_npu.passes.lower_pipeline import normalize_loop_controls


def program(lower, upper, expression, *, step=1):
    module = N.from_source(f"""
        def domain(n: N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                for i in N.serial({lower},{upper},{step}):
                    unused={expression}
                    N.fill(x,0.0)
    """).to_ir()
    return module, next(op for op in module.walk() if isinstance(op, ProgramOp))


def test_nonzero_lower_bound_is_used_before_unsigned_division_simplification():
    module, source = program(3, 10, "(i-3)//2", step=2)
    (control,) = normalize_loop_controls(module).loops
    assert control.domain.concrete({}) == (3, 10, 2)
    validate_source_arithmetic(source)


def test_negative_serial_coordinate_cannot_be_proved_nonnegative_from_zero():
    module, source = program(-1, 3, "i//2")
    normalize_loop_controls(module)
    with pytest.raises(CompileError, match="nonnegative dividend"):
        validate_source_arithmetic(source)


@pytest.mark.parametrize("lower,upper", [(3, 3), (3, 0), (-2, -3), ("n", "n")])
def test_empty_range_has_no_executed_division_even_when_upper_is_nonzero(lower, upper):
    module, source = program(lower, upper, "(i-8)//2")
    normalize_loop_controls(module)
    validate_source_arithmetic(source)


def test_uniform_runtime_lower_is_preserved_in_the_range_proof():
    module, source = program("n", "n+8", "(i-n)//2", step=3)
    normalize_loop_controls(module)
    validate_source_arithmetic(source)
