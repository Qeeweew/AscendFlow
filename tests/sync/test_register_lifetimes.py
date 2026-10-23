"""SPR definitions, configuration readiness and reader retirement compose."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from ascendflow.analysis.registers import admit_register_definitions
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine
from tests.fixtures.register_matrices import register_matrices
from tests.models.structured_events import EventModelError, check_compiled_events


@pytest.mark.parametrize("branch", [False, True])
def test_different_matrix_configurations_compose_through_branches_and_loop_reentry(branch):
    plan = synchronize_structure(register_matrices(branch=branch))
    assert any(t.spec.native == "load3d_a" for t in plan.admission.memory.templates)
    assert any(t.spec.native == "load3d_b_nn" for t in plan.admission.memory.templates)
    for count in (0, 1, 3):
        for choice in (0, 1):
            execution = check_compiled_events(plan, {"n": count, "choice": choice})
            assert not count or execution.register_hazards


def test_deleting_the_real_retirement_handshake_exposes_a_register_war():
    plan = synchronize_structure(register_matrices())
    check_compiled_events(plan, {"n": 1, "choice": 0})
    retirements = [
        op
        for op in plan.module.walk()
        if isinstance(op, NotificationOp)
        and op.source.data == Engine.MTE1.value
        and op.destination.data == Engine.S.value
    ]
    assert retirements
    for op in retirements:
        op.parent.erase_op(op)
    with pytest.raises(EventModelError, match="unordered register WAR"):
        check_compiled_events(plan, {"n": 1, "choice": 0})


@pytest.mark.parametrize("change", ["remove", "wrong_value"])
def test_register_reader_requires_the_actual_setter_and_its_required_value(change):
    structure = lower_native_structure(register_matrices())
    setter = next(
        op
        for op in structure.module.walk()
        if getattr(op, "native", None) is not None and op.native.data == "set_fmatrix_a"
    )
    if change == "remove":
        setter.parent.erase_op(setter)
    else:
        setter.attributes["params"] = ArrayAttr((IntAttr(32),))
    with pytest.raises(
        CompileError, match="FMATRIX_A reader lacks its required dominating configuration"
    ):
        admit_register_definitions(collect_native_memory(structure.module))
