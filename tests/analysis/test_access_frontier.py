"""Actual native identities and finite FIRST/LAST ordinal contracts."""

from dataclasses import fields

import pytest
from xdsl.dialects import scf

from ascendflow import language as N
from ascendflow.analysis.access_frontier import (
    AccessEndpoint,
    LoopResidueSelection,
    ResiduePosition,
)
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.structured import PipelineLoopOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target.ascend910b import Actor


@pytest.mark.parametrize("serial", [False, True])
def test_actual_loop_selection_guards_empty_and_short_before_last_ordinal(serial):
    loop_source = "N.serial(3,10,2)" if serial else "N.Pipelined(7)"
    plan = lower_native_structure(
        N.from_source(f"""
        def select():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in {loop_source}:
                    N.fill(x,0.0)
    """)
    )
    loop = next(op for op in plan.module.walk() if isinstance(op, (PipelineLoopOp, scf.ForOp)))
    for residue in range(3):
        first = LoopResidueSelection(loop, 3, residue, ResiduePosition.FIRST)
        last = LoopResidueSelection(loop, 3, residue, ResiduePosition.LAST)
        for count in range(10):
            actual = list(range(residue, count, 3))
            assert first.evaluate(count) == (actual[0] if actual else None)
            assert last.evaluate(count) == (actual[-1] if actual else None)
        assert first.loop is last.loop is loop
        assert first.evaluate(0) is last.evaluate(0) is None
        for count in (10**6, 10**12):
            selected = last.evaluate(count)
            assert 0 <= selected < count and selected % 3 == residue
            assert selected + 3 >= count
        assert len(fields(last)) == 4


def test_residue_selection_preserves_loop_identity_and_rejects_bad_domains():
    plan = lower_native_structure(
        N.from_source("""
        def loops():
            with N.Kernel():
                for a in N.Pipelined(2):
                    pass
                for b in N.Pipelined(2):
                    pass
""")
    )
    first, second = tuple(plan.loops.values())
    a = LoopResidueSelection(first, 2, 0, ResiduePosition.LAST)
    b = LoopResidueSelection(second, 2, 0, ResiduePosition.LAST)
    assert a != b and a.evaluate(10**12) == b.evaluate(10**12)
    for period, residue in ((0, 0), (2, 2), (2, -1)):
        with pytest.raises(CompileError, match="actual loop, period/residue"):
            LoopResidueSelection(first, period, residue, ResiduePosition.LAST)
    with pytest.raises(CompileError, match="nonnegative"):
        a.evaluate(-1)


def test_endpoint_identities_do_not_depend_on_diagnostic_names():
    structure = lower_native_structure(
        N.from_source("""
        def identities():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,0.0)
                N.mul(x,2.0,out=y)
    """)
    )
    facts = collect_native_memory(structure.module)
    writer, reader = (AccessEndpoint.from_template(t, Actor.AIV0) for t in facts.templates)
    writer.operation.attributes["uid"] = reader.operation.uid
    assert writer.operation.uid == reader.operation.uid
    assert writer != reader
