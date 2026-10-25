"""L0C unitflag protocol admission and independent partial-completion checks."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import compile
from ascendflow import language as T
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)

SOURCE = """
def flags(X:T.Tensor((128,128),T.bf16),
          Y:T.Tensor((n,128,128),T.bf16,access="write",write_disjoint=True),n:T.i64):
    with T.Kernel(1,kind="cube"):
        a1=T.alloc_l1((128,128),T.bf16)
        b1=T.alloc_l1((128,128),T.bf16)
        a0=T.alloc_l0a((128,128),T.bf16)
        b0=T.alloc_l0b((128,128),T.bf16)
        c0=T.alloc_l0c((128,128),T.f32)
        a=T.copy(X[:,:],a1)
        b=T.copy(X[:,:],b1)
        aa=T.copy(a,a0)
        bb=T.copy(b,b0)
        for k in T.Pipelined(n*4):
            with T.Stage(0):
                T.matmul(aa,bb,accum=c0,initialize=k%4==0,
                         unit_flag=3 if k%4==3 else 2)
                if k%4==3:
                    T.copy(c0,Y[k//4,:,:],unit_flag=3)
"""


def source(text=SOURCE):
    return T.from_source(text, disjoint_external=True)


def test_unitflag_keeps_access_requirements_without_whole_pipe_handshakes():
    plan = synchronize_structure(source())
    assert len(plan.selection.unitflags.pairs) == 1
    covered = [c for c in plan.selection.coverage if c.unitflag is not None]
    assert {c.requirement.kind.value for c in covered} == {"RAW", "WAR"}
    assert all(c.unitflag.binding is c.binding for c in covered)
    assert not any(
        {a.source.value, a.destination.value} == {"M", "FIX"} for a in plan.resources.actions
    )
    result = compile(source())
    assert ", 128, 128, 128, 0, false, false," not in result.asc
    for n in (0, 1, 3):
        execution = check_compiled_events(plan, {"n": n}, invocations=2)
        if n:
            publish = next(
                c for c in execution.commands if c.source.value == "M" and c.unit_flag == 3
            )
            consume = next(
                c for c in execution.commands if c.source.value == "FIX" and c.unit_flag == 3
            )
            assert execution.happens_before(publish.cc_end, consume.cc_begin)
            assert not execution.happens_before(publish.end, consume.begin)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("3 if k%4==3 else 2", "1", "must be 2 or 3"),
        ("3 if k%4==3 else 2", "3 if k%4==2 else 2", "identical activation"),
        ("T.copy(c0,Y[k//4,:,:],unit_flag=3)", "T.copy(c0,Y[k//4,:,:])", "no final FIX"),
        (
            "T.copy(c0,Y[k//4,:,:],unit_flag=3)",
            "T.copy(c0,Y[k//4,:,:64],unit_flag=3)",
            "exact published",
        ),
        (
            "T.copy(c0,Y[k//4,:,:],unit_flag=3)",
            "T.copy(c0,Y[k//4,:,:],unit_flag=3,valid_rows=64)",
            "unconsumed runtime rows",
        ),
        (
            "T.copy(c0,Y[k//4,:,:],unit_flag=3)",
            "T.copy(c0,Y[k//4,:,:],unit_flag=3)\n                    T.copy(c0,Y[k//4,:,:])",
            "unflagged access interferes",
        ),
        (
            "                if k%4==3:\n                    T.copy",
            "            with T.Stage(1):\n                if k%4==3:\n                    T.copy",
            "crosses dispatch stages",
        ),
    ],
)
def test_invalid_flag_words_are_rejected(old, new, message):
    with pytest.raises(CompileError, match=message):
        synchronize_structure(source(SOURCE.replace(old, new)))


def test_rotating_l0c_slots_require_a_separate_unitflag_protocol():
    text = (
        SOURCE.replace("alloc_l0c((128,128),T.f32)", "alloc_l0c((128,128),T.f32,slots=2)")
        .replace("accum=c0,", "accum=c0.at(k//4),")
        .replace("T.copy(c0,Y", "T.copy(c0.at(k//4),Y")
    )
    with pytest.raises(CompileError, match="fixed L0C plane"):
        synchronize_structure(source(text))


def test_finite_model_detects_a_missing_actual_mad_publication():
    plan = synchronize_structure(source())
    before = collect_native_memory(parse(plan.before_ir))
    publisher = plan.selection.unitflags.pairs[0].publisher.operation.uid
    actual = next(op for op in plan.module.walk() if isinstance(op, TaskOp) and op.uid == publisher)
    flag = arith.ConstantOp.from_int_and_width(2, IndexType())
    actual.parent.insert_op_before(flag, actual)
    refs = tuple(actual.refs)
    actual.operands = (*refs[:3], flag.result, *refs[4:])
    with pytest.raises(EventModelError, match="unitflag FIX has no publication"):
        check_structured_events(
            collect_native_memory(plan.module), {"n": 2}, logical_reference=before, invocations=2
        )
