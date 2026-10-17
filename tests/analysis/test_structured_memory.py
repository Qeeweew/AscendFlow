"""One target rule/address decode per actual native, not full synchronization."""

import pytest

from ascendflow import language as N
from ascendflow.analysis import structured_memory
from ascendflow.analysis.structured_control import build_control_facts
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.structured import ActorDomainOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target import native
from ascendflow.target.ascend910b import Actor
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.control.test_shared_aiv import SHARED


@pytest.mark.parametrize(
    "source,count",
    [(scale, 3), (batched_matmul, 6), (N.from_source(SHARED, disjoint_external=True), 9)],
)
def test_native_memory_facts_preserve_owned_operations_and_call_target_rule_once(
    source, count, monkeypatch
):
    plan = lower_native_structure(source)
    control = build_control_facts(plan.module)
    rule = native.native_contract
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return rule(*args, **kwargs)

    def no_rebuild(*args):
        pytest.fail("unchanged control facts were rebuilt")

    monkeypatch.setattr(native, "native_contract", counted)
    monkeypatch.setattr(structured_memory, "build_control_facts", no_rebuild)
    memory = collect_native_memory(plan.module, control=control)
    assert memory.control is control
    assert len(calls) == memory.native_contract_calls == count
    assert memory.module is plan.module
    assert tuple(item.operation for item in memory.templates) == tuple(
        op for op in plan.module.walk() if isinstance(op, TaskOp)
    )
    refs = {access.operand for item in memory.templates for access in item.accesses}
    assert set(memory.addresses) == refs
    for template in memory.templates:
        for access in template.accesses:
            assert access.operand is template.operation.refs[access.operand_index]
            assert access.address is memory.addresses[access.operand]
            assert access.address.root is access.operand.owner.source
    assert not hasattr(memory, "asc")


def test_one_aiv_template_keeps_two_participants_and_symbolic_sid_address():
    plan = lower_native_structure(N.from_source(SHARED, disjoint_external=True))
    memory = collect_native_memory(plan.module)
    vector = [item for item in memory.templates if item.participants == (Actor.AIV0, Actor.AIV1)]
    assert len(vector) == 3  # no clone per participant
    assert all(isinstance(item.frame.operation, ActorDomainOp) for item in vector)
    load = vector[0]
    gm, ub = (access.address for access in load.accesses)
    loop = next(iter(memory.control.loops.values()))
    values = {loop.ordinal.args[0]: 0, memory.control.group.args[0]: 0, "n": 3}
    assert (
        gm.offset.evaluate(values | {memory.control.sid.args[0]: 1})
        - gm.offset.evaluate(values | {memory.control.sid.args[0]: 0})
        == 16 * 16 * 4
    )
    assert ub.offset.constant == 0


@pytest.mark.parametrize("origin,accepted", [("64:128", True), ("32:96", False), ("0:64", False)])
def test_ub_copy_alias_premise_uses_actual_root_and_full_offsets(origin, accepted):
    source = N.from_source(f"""
        def aliases():
            with N.Kernel():
                x=N.alloc_ub((128,))
                N.copy(x[0:64],x[{origin}])
    """)
    plan = lower_native_structure(source)
    if accepted:
        memory = collect_native_memory(plan.module)
        assert len(memory.templates) == 1
    else:
        with pytest.raises(CompileError, match="provably disjoint"):
            collect_native_memory(plan.module)


def test_another_modules_control_cannot_authorize_memory_collection():
    first, second = lower_native_structure(scale), lower_native_structure(scale)
    control = build_control_facts(first.module)
    with pytest.raises(CompileError, match="another physical module"):
        collect_native_memory(second.module, control=control)
