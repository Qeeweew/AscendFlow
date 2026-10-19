"""Independent execution of frontend scalar SSA, including zero-trip loops."""

import pytest
from xdsl.dialects import arith, scf

from ascendflow import compile
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.source import PipelineOp, SizeOp, YieldOp
from ascendflow.frontend import FrontendError, from_source


def interpret(block, values, arguments):
    """Only scalar test arithmetic; no production expression evaluator is used."""
    for op in block.ops:
        if isinstance(op, SizeOp):
            result = (arguments[op.sym_name.data],)
        elif isinstance(op, arith.ConstantOp):
            result = (op.value.value.data,)
        elif isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp)):
            a, b = (values[operand] for operand in op.operands)
            result = (
                a + b
                if isinstance(op, arith.AddiOp)
                else a - b
                if isinstance(op, arith.SubiOp)
                else a * b,
            )
        elif isinstance(op, arith.CmpiOp):
            a, b = (values[operand] for operand in op.operands)
            result = (
                {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[
                    op.predicate.value.data
                ],
            )
        elif isinstance(op, scf.IfOp):
            region = op.true_region if values[op.cond] else op.false_region
            result = interpret(region.block, dict(values), arguments)
        elif isinstance(op, scf.ForOp):
            state = tuple(values[value] for value in op.iter_args)
            for iv in range(values[op.lb], values[op.ub], values[op.step]):
                local = dict(values)
                local.update(zip(op.body.block.args, (iv, *state)))
                state = interpret(op.body.block, local, arguments)
            result = state
        elif isinstance(op, scf.YieldOp):
            return tuple(values[value] for value in op.arguments)
        elif isinstance(op, PipelineOp):
            return values[op.limit] if op.limit is not None else op.count.data
        elif isinstance(op, YieldOp):
            return None
        else:
            raise TypeError(f"unsupported oracle instruction {op.name}")
        values.update(zip(op.results, result))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 9, 32])
def test_serial_loop_carries_ssa_through_if_and_preserves_zero_iteration_value(n):
    definition = from_source("""
        def accumulate(n: N.int64):
            with N.Kernel():
                total = 7
                for i in N.serial(n):
                    if i < 3:
                        total = total + i
                    else:
                        total = total + 1
                for k in N.Pipelined(total, stages=1):
                    pass
    """)
    source = parse(dump(definition.to_ir()))
    loop = next(op for op in source.walk() if isinstance(op, scf.ForOp))
    conditional = next(op for op in source.walk() if isinstance(op, scf.IfOp))
    assert len(loop.iter_args) == len(loop.results) == len(conditional.results) == 1
    assert interpret(source.body.block.first_op.body.block, {}, {"n": n}) == 7 + sum(
        i if i < 3 else 1 for i in range(n)
    )
    with pytest.raises(CompileError, match="host integer safety lacks a uniform scalar range"):
        compile(definition)


def test_serial_bound_is_captured_at_entry_even_when_carried_value_changes():
    source = from_source("""
        def countdown(n: N.int64):
            with N.Kernel():
                remaining = n
                for i in N.serial(remaining):
                    remaining = remaining - 1
                for k in N.Pipelined(remaining, stages=1):
                    pass
    """).to_ir()
    for n in (0, 1, 17):
        assert interpret(source.body.block.first_op.body.block, {}, {"n": n}) == 0


def test_if_can_define_a_new_scalar_on_both_paths_and_merge_it():
    source = from_source("""
        def select(n: N.int64):
            with N.Kernel():
                if n < 4:
                    count = 2
                else:
                    count = n
                for k in N.Pipelined(count, stages=1):
                    pass
    """).to_ir()
    for n in (0, 3, 4, 17):
        assert interpret(source.body.block.first_op.body.block, {}, {"n": n}) == (2 if n < 4 else n)


def test_one_sided_definition_is_not_exported_as_an_initialized_scalar():
    definition = from_source("""
        def invalid(n: N.int64):
            with N.Kernel():
                if n < 4:
                    count = 2
                for k in N.Pipelined(count, stages=1):
                    pass
    """)
    with pytest.raises(FrontendError, match="undefined DSL name count"):
        definition.to_ir()


@pytest.mark.parametrize("count", [0, 3])
def test_explicit_unroll_preserves_constant_updates(count):
    source = from_source(
        """
        def unrolled():
            with N.Kernel():
                total = 2
                for i in N.unroll(COUNT):
                    total = total + 1
                for k in N.Pipelined(total, stages=1):
                    pass
    """,
        constants={"COUNT": count},
    ).to_ir()
    assert interpret(source.body.block.first_op.body.block, {}, {}) == count + 2


def test_boolean_merge_is_not_a_host_boolean():
    source = from_source("""
        def choose(n: N.int64):
            with N.Kernel():
                chosen = False
                if n < 4:
                    chosen = True
                if chosen:
                    count = 2
                else:
                    count = 5
                for k in N.Pipelined(count, stages=1):
                    pass
    """).to_ir()
    for n in (1, 5):
        assert interpret(source.body.block.first_op.body.block, {}, {"n": n}) == (2 if n < 4 else 5)
