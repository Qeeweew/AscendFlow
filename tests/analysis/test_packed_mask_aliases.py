"""Packed boolean storage uses actual ranges within one allocation root."""

from dataclasses import replace

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.source import CompareOp, MaskCombineOp
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def kernel(destination, *, guarded=False):
    body = f"""
        p=N.compare(loaded,0.0,kind="gt",out=bits.slot(0))
        inverted=N.mask_not(p,out=bits.slot({destination}))
        selected=N.where(inverted,loaded,loaded,out=work)
        N.copy(selected,Out)
"""
    if guarded:
        body = "        if which==2:\n" + "\n".join(
            "    " + line for line in body.splitlines() if line
        )
    return N.from_source(
        """
def packed(X:N.Tensor((64,),N.f32),Out:N.Tensor((64,),N.f32,access="write"),which:N.i64):
    with N.Kernel(kind="vector"):
        work=N.alloc_ub((64,),N.f32)
        bits=N.alloc_mask((64,),slots=3)
        loaded=N.copy(X,work)
"""
        + body,
        disjoint_external=True,
    )


def test_separate_packed_slots_are_admitted_by_their_real_native_ranges():
    plan = synchronize_structure(kernel("2"))
    native = next(t for t in plan.admission.memory.templates if t.spec.native == "mask_not")
    left, right = native.accesses
    assert left.address.root is right.address.root
    assert left.address.offset != right.address.offset
    check_compiled_events(plan, {"which": 2}, invocations=2)


def test_guarded_runtime_slot_uses_the_actual_activation_domain():
    plan = synchronize_structure(kernel("which", guarded=True))
    for which in (0, 1, 2):
        check_compiled_events(plan, {"which": which}, invocations=2)


@pytest.mark.parametrize("destination", ["0", "which"])
def test_identical_and_unknown_aliases_do_not_gain_a_disjointness_permission(destination):
    with pytest.raises(CompileError, match="distinct predicate storage"):
        synchronize_structure(kernel(destination))


def test_initialized_sibling_slot_cannot_replace_a_predicates_actual_writer():
    definition = kernel("2")
    original = 'p=N.compare(loaded,0.0,kind="gt",out=bits.slot(0))'
    changed = definition.source.replace(
        original, original + '\n        sibling=N.compare(loaded,1.0,kind="gt",out=bits.slot(1))'
    )
    assert changed != definition.source
    source = replace(definition, source=changed).to_ir()
    comparisons = [op for op in source.walk() if isinstance(op, CompareOp)]
    combination = next(op for op in source.walk() if isinstance(op, MaskCombineOp))
    assert len(comparisons) == 2
    combination.lhs_storage.replace_all_uses_with(comparisons[1].out)
    with pytest.raises(CompileError):
        compile(source)
