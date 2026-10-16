"""Real serial-query/KV/child clocks reach physical lowering without frames."""

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType

from xdsl_npu import language as N
from xdsl_npu.indexing import arguments
from xdsl_npu.ir import (
    CyclicCountOp,
    FrameOp,
    LoopSyncOp,
    ScopeSyncOp,
    StreamOp,
    TaskOp,
    dump,
)
from xdsl_npu.lowering import lower_source
from xdsl_npu.lowering_context import ActorBindings
from xdsl_npu.model import CompileError
from xdsl_npu.scalar import Expr
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.source_ir import AllocationOp, PipelineOp

SOURCE = """
def scoped_serial(Q:N.Tensor((total_queries,2,16,16),N.f16),
                  O:N.Tensor((total_queries,4,64),N.f32,access="write"),
                  total_queries:N.int64,kv_tiles:N.int64(minimum=1),
                  GROUPS:N.constexpr=4):
    with N.Kernel(GROUPS,kind="mixed") as group:
        q=N.alloc_l1((16,16),N.f16)
        x=N.alloc_ub((64,),N.f32)
        local_queries=N.ceildiv(N.max(0,total_queries-group),GROUPS)
        for query in N.serial(local_queries):
            for kv in N.Pipelined(kv_tiles,stages=2):
                flat=group+query*GROUPS
                with N.Stage(0,name="load"),N.Cube():
                    for part in N.Pipelined(2,stages=1):
                        N.copy(Q[flat,part,:,:],q)
                with N.Stage(1,name="export"),N.Vector() as lane:
                    for part in N.Pipelined(2,stages=1):
                        N.fill(x,2.0)
                        if kv == kv_tiles - 1:
                            N.copy(x,O[flat,lane*2+part,:])
"""


def source(text=SOURCE):
    return N.from_source(text, disjoint_external=True).to_ir()


def bindings(text=SOURCE, family="vector"):
    program = source(text).body.block.first_op
    serial = next(op for op in program.body.block.ops if isinstance(op, scf.ForOp))
    allocations = tuple(op for op in program.walk() if isinstance(op, AllocationOp))
    return program, serial, ActorBindings(program, family, allocations)


@pytest.mark.parametrize("family", ("cube", "vector"))
def test_cyclic_query_bound_keeps_exact_source_owner_and_never_becomes_uniform(family):
    program, serial, environment = bindings(family=family)
    physical = environment.scoped_serial_bound(serial)
    assert isinstance(physical.owner, CyclicCountOp)
    assert physical.owner.parent_op() is environment.kernel.op
    assert serial.ub not in environment.uniform
    assert not hasattr(environment, "cyclic_total")
    assert environment.scoped_serial_bound(serial) is physical
    assert sum(isinstance(op, CyclicCountOp) for op in environment.kernel.block.ops) == 1
    assert set(environment.owner_domains) == {serial.body.block.args[0]}
    domain = environment.owner_domains[serial.body.block.args[0]]
    assert domain.total == Expr("argument", ("total_queries",))
    assert domain.groups == program.groups.data
    for total in (0, 1, 3, 4, 5, 51):
        for core in range(domain.groups):
            expected = len(range(core, total, domain.groups))
            assert domain.local_bound.evaluate({"total_queries": total, "$core": core}) == expected


def test_uniform_serial_bound_uses_actual_existing_uniform_ssa_without_owner_metadata():
    _, serial, environment = bindings(
        SOURCE.replace("N.ceildiv(N.max(0,total_queries-group),GROUPS)", "total_queries")
    )
    physical = environment.scoped_serial_bound(serial)
    assert physical is environment.uniform[serial.ub]
    assert not environment.owner_domains
    assert not any(isinstance(op, CyclicCountOp) for op in environment.kernel.block.ops)


def test_scoped_serial_does_not_relax_the_old_mixed_runtime_bound_contract():
    _, serial, environment = bindings()
    pipeline = next(op for op in serial.walk() if isinstance(op, PipelineOp))
    pipeline.operands = [serial.ub]
    with pytest.raises(CompileError, match="uniform count or a proved cyclic owner domain"):
        environment.runtime_bound(pipeline)


def test_wrong_owner_divisor_cannot_create_a_cyclic_query_certificate():
    _, serial, environment = bindings(
        SOURCE.replace("total_queries-group),GROUPS)", "total_queries-group),GROUPS+1)")
    )
    with pytest.raises(CompileError, match="scoped serial bound requires"):
        environment.scoped_serial_bound(serial)


def test_real_source_keeps_serial_query_kv_and_fixed_child_ssa_clocks():
    original = source()
    before = dump(original)
    physical = lower_source(original)
    assert dump(original) == before
    physical.verify()
    scoped = extract_scopes(physical)
    assert set(scoped.invocations) == {"aic", "aiv0", "aiv1"}
    assert len(scoped.leaves) == 3
    counts = tuple(op for op in physical.walk() if isinstance(op, CyclicCountOp))
    assert len(counts) == 3 and len({op.total for op in counts}) == 1
    for actor, invocation in scoped.invocations.items():
        parent = scoped.parents[actor]
        assert tuple(type(op) for op in invocation.op.body.block.ops) == (StreamOp, scf.YieldOp)
        assert parent.parent_op() is invocation.op
        assert invocation.op.ub.owner in counts
        assert invocation.cyclic.groups == 4
        assert parent.limit is not invocation.op.ub
        leaves = [leaf for leaf in scoped.leaves.values() if leaf.actor == actor]
        assert len(leaves) == 1 and leaves[0].stream.count.data == 2
        assert leaves[0].call.generation is parent.body.block.args[0]
        assert leaves[0].call.generation is not invocation.op.body.block.args[0]
        assert any("$parent1" in arguments(captured.offset) for captured in leaves[0].captures)
    assert sum(isinstance(op, TaskOp) for op in physical.walk()) == 5
    assert not any(isinstance(op, (FrameOp, LoopSyncOp, ScopeSyncOp)) for op in physical.walk())


def test_uniform_query_keeps_real_axes_without_legacy_frame_mapping():
    text = (
        SOURCE.replace("N.ceildiv(N.max(0,total_queries-group),GROUPS)", "total_queries")
        .replace("((total_queries,2,16,16)", "((GROUPS,total_queries,2,16,16)")
        .replace("((total_queries,4,64)", "((GROUPS,total_queries,4,64)")
        .replace("Q[flat,part,:,:]", "Q[group,query,part,:,:]")
        .replace("O[flat,lane*2+part,:]", "O[group,query,lane*2+part,:]")
    )
    physical = lower_source(source(text))
    scoped = extract_scopes(physical)
    assert len(scoped.invocations) == 3
    assert all(invocation.cyclic is None for invocation in scoped.invocations.values())
    assert not any(isinstance(op, (CyclicCountOp, FrameOp)) for op in physical.walk())


@pytest.mark.parametrize("offset", (1, -1))
def test_flat_query_axis_proof_uses_query_owner_domain_and_rejects_out_of_bounds(offset):
    text = SOURCE.replace("flat=group+query*GROUPS", f"flat=group+query*GROUPS+({offset})")
    with pytest.raises(CompileError, match="outside|exceed|bound|negative"):
        lower_source(source(text))


def test_kv_coordinate_cannot_borrow_the_query_cyclic_ownership_domain():
    text = SOURCE.replace("flat=group+query*GROUPS", "flat=group+kv*GROUPS")
    with pytest.raises(CompileError, match="bounded by its tensor extent"):
        lower_source(source(text))


@pytest.mark.parametrize("field,value", (("lb", 1), ("step", 2)))
def test_serial_query_requires_actual_zero_start_and_unit_step(field, value):
    module = source()
    program = module.body.block.first_op
    serial = next(op for op in program.body.block.ops if isinstance(op, scf.ForOp))
    constant = arith.ConstantOp.from_int_and_width(value, IndexType())
    program.body.block.insert_op_before(constant, serial)
    operands = list(serial.operands)
    operands[0 if field == "lb" else 2] = constant.result
    serial.operands = operands
    with pytest.raises(CompileError, match="zero-based unit-step"):
        lower_source(module)


def test_ordinary_work_beside_kv_pipeline_is_not_silently_dropped():
    text = SOURCE.replace(
        "for kv in N.Pipelined(kv_tiles,stages=2):",
        "with N.Vector():\n                N.fill(x,0.0)\n            for kv in N.Pipelined(kv_tiles,stages=2):",
    )
    with pytest.raises(CompileError, match="one actual KV pipeline and no other work"):
        lower_source(source(text))


def test_unsynchronized_source_fails_at_its_actual_child_memory_hazard():
    from xdsl_npu.sync import verify_plan

    module = lower_source(source())
    before = dump(module)
    with pytest.raises(CompileError, match="missing loop (RAW|WAR|WAW)"):
        verify_plan(module)
    assert dump(module) == before
