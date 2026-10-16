"""Actual single-AIC event DAGs preserve source values across L0C occupants."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr

from tests.fixtures.nested import shared_l0c_qk_pv
from tests.models.scoped_local import _scope_epoch, finite_check
from tests.test_definition_lifetimes import _pv_recurrence, _read_qk_operand_zero
from tests.test_scope_completion import Parent
from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.dialects.physical import (
    LoopChannelOp,
    LoopSyncOp,
    ScopeChannelOp,
    ScopeSyncOp,
    TaskOp,
)

COUNTS = (0, 1, 2, 3, 7)


def activation(*, minimum=0, exact=None, first=(False, 0), last=(True, 0)):
    fields = {
        "count_period": 1,
        "count_residue": 0,
        "count_minimum": minimum,
        "first_end": int(first[0]),
        "first_offset": first[1],
        "last_end": int(last[0]),
        "last_offset": last[1],
    }
    if exact is not None:
        fields["count_exact"] = exact
    return DictionaryAttr({name: IntAttr(value) for name, value in fields.items()})


def local(kernel, name, source, destination, event_id, *, distance=0):
    declaration = LoopChannelOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "kind": StringAttr("local"),
            "sources": ArrayAttr([StringAttr("aic/" + source.uid.data)]),
            "destinations": ArrayAttr([StringAttr("aic/" + destination.uid.data)]),
            "distance": IntAttr(distance),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(event_id)]),
        }
    )
    kernel.block.insert_op_before(declaration, kernel.stream)
    for task, action in ((source, "set"), (destination, "wait")):
        sync = LoopSyncOp.create(
            operands=[kernel.task_block.args[0]],
            attributes={
                "action": StringAttr(action),
                "channel": StringAttr(name),
                "lower": IntAttr(distance if action == "wait" else 0),
                "upper_gap": IntAttr(distance if action == "set" else 0),
                "period": IntAttr(1),
                "residues": ArrayAttr([IntAttr(0)]),
            },
        )
        if action == "set":
            kernel.task_block.insert_op_after(sync, task)
        else:
            kernel.task_block.insert_op_before(sync, task)


def child(name, *, export, count=1):
    kernel = Kernel(name, core="aic", disjoint_external=True)
    x = kernel.buffer("X", "GM", 512, dtype="f16", external=True, initialized=True)
    l1 = kernel.buffer("l1", "L1", 512, dtype="f16", layout="NZ")
    a = kernel.buffer("a", "L0A", 512, dtype="f16", layout="ZZ")
    b = kernel.buffer("b", "L0B", 512, dtype="f16", layout="ZN")
    c = kernel.buffer("c", "L0C", 1024, layout="NZ")
    if export:
        out = kernel.buffer("Score", "GM", 1024, external=True)
    kernel.loop(count, symbolic=True)
    load = kernel.task(
        name + "_load",
        "nd2nz",
        kernel.view(x, 32, rows=16, row_stride=32),
        kernel.view(l1, 512),
        params=(16, 16),
    )
    aa = kernel.task(
        name + "_a", "load2d_a", kernel.view(l1, 512), kernel.view(a, 512), params=(16, 16)
    )
    bb = kernel.task(
        name + "_b", "load2d_b_nt", kernel.view(l1, 512), kernel.view(b, 512), params=(16, 16)
    )
    mad = kernel.task(
        name + "_mad",
        "mad" if export else "mad_first",
        kernel.view(a, 512),
        kernel.view(b, 512),
        kernel.view(c, 1024),
        params=(16, 16, 16, 1) if export else (16, 16, 16),
    )
    local(kernel, name + "_load_ready", load, aa, 0 if export else 1)
    local(kernel, name + "_operands_ready", bb, mad, 0 if export else 1)
    if export:
        fix = kernel.task(
            name + "_fix",
            "fix",
            kernel.view(c, 1024),
            kernel.view(out, 64, rows=16, row_stride=64),
            params=(16, 16),
        )
        local(kernel, name + "_result_ready", mad, fix, 0)
    else:
        local(kernel, name + "_slot_return", mad, load, 0, distance=1)
    return kernel.finish()


def scope(parent, name, source, destination, event_id, *, distance=0, active=None):
    src_call, src_uid, src_g = source
    dst_call, dst_uid, dst_g = destination
    declaration = ScopeChannelOp.create(
        attributes={
            "sym_name": StringAttr(name),
            "kind": StringAttr("local"),
            "sources": ArrayAttr([StringAttr(f"aic/{src_call}/{src_uid}/{src_g}")]),
            "destinations": ArrayAttr([StringAttr(f"aic/{dst_call}/{dst_uid}/{dst_g}")]),
            "distance": IntAttr(distance),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(event_id)]),
            **({} if active is None else {"activation": active}),
        }
    )
    parent.k.block.insert_op_before(declaration, parent.k.stream)
    for uid, task_uid, g, action in ((*source, "set"), (*destination, "wait")):
        call = parent.calls[uid][0]
        child = next(op for op in call.body.block.ops if hasattr(op, "body") and op is not call)
        task = next(op for op in child.walk() if isinstance(op, TaskOp) and op.uid.data == task_uid)
        sync = ScopeSyncOp.create(
            operands=[call.generation, child.body.block.args[0]],
            attributes={
                "action": StringAttr(action),
                "channel": StringAttr(name),
                "position": IntAttr(g),
            },
        )
        if action == "set":
            task.parent.insert_op_after(sync, task)
        else:
            task.parent.insert_op_before(sync, task)
    return declaration


@pytest.fixture(scope="module")
def manual_shared():
    parent = Parent(core="aic")
    parent.add("QK", child("qk", export=True), None, shared=("c",))
    parent.add("PV", child("pv", export=False, count=2), None, stage=1, order=1, shared=("c",))
    output = Kernel("output", core="aic", disjoint_external=True)
    c = output.buffer("c", "L0C", 1024, layout="NZ")
    y = output.buffer("Y", "GM", 1024, external=True)
    output.once("output")
    output.task(
        "output_fix",
        "fix",
        output.view(c, 1024),
        output.view(y, 64, rows=16, row_stride=64),
        params=(16, 16),
    )
    parent.add("FIX", output.finish(), None, stage=1, order=2, shared=("c",))
    qk_mad, qk_fix, pv_mad, output_fix = (
        ("QK", "qk_mad", 0),
        ("QK", "qk_fix", 0),
        ("PV", "pv_mad", 0),
        ("FIX", "output_fix", 0),
    )
    # Private operand storage is returned at its last MAD, before result FIX.
    # These edges cannot stand in for the separate shared-C retirement edges.
    scope(parent, "qk_operands_return", qk_mad, ("QK", "qk_load", 0), 2, distance=1)
    scope(parent, "pv_operands_return", ("PV", "pv_mad", 1), ("PV", "pv_load", 0), 1, distance=1)
    scope(parent, "pv_result_ready", ("PV", "pv_mad", 1), output_fix, 1)
    scope(parent, "startup_one", qk_fix, pv_mad, 0, active=activation(exact=1, last=(False, 0)))
    scope(
        parent,
        "startup_many",
        qk_fix,
        qk_mad,
        1,
        distance=1,
        active=activation(minimum=2, last=(False, 0)),
    )
    scope(
        parent,
        "steady_to_pv",
        qk_fix,
        pv_mad,
        2,
        distance=-1,
        active=activation(minimum=2, first=(False, 1)),
    )
    scope(
        parent,
        "steady_to_qk",
        output_fix,
        qk_mad,
        3,
        distance=2,
        active=activation(minimum=3, last=(True, 2)),
    )
    scope(
        parent,
        "drain",
        output_fix,
        pv_mad,
        4,
        distance=1,
        active=activation(minimum=2, first=(True, 1), last=(True, 1)),
    )
    parent.k.module.verify()
    return parent.k.module


def check(plan, count):
    return finite_check(plan, count, parameters={"tasks": count})


@pytest.mark.parametrize("count", COUNTS)
def test_manual_full_cube_pipeline_keeps_raw_versions_and_physical_c_occupants(
    manual_shared, count
):
    result = check(manual_shared, count)
    assert len(result.issued_tasks) == 14 * count
    assert result.bytes_read == 10752 * count
    assert result.bytes_written == 9728 * count


def erase_channel(plan, name):
    for op in list(plan.walk()):
        if (
            isinstance(op, ScopeSyncOp)
            and op.channel.data == name
            or isinstance(op, ScopeChannelOp)
            and op.sym_name.data == name
        ):
            op.parent.erase_op(op)


@pytest.mark.parametrize(
    "name,count",
    (
        ("startup_one", 1),
        ("startup_many", 2),
        ("drain", 2),
        ("steady_to_pv", 3),
        ("steady_to_qk", 3),
    ),
)
def test_missing_boundary_family_cannot_be_replaced_by_a_steady_path(manual_shared, name, count):
    plan = manual_shared.clone()
    erase_channel(plan, name)
    with pytest.raises(AssertionError, match="unordered physical|binary event overwrite"):
        check(plan, count)


def test_missing_actual_cube_wait_is_an_unmatched_physical_event(manual_shared):
    plan = manual_shared.clone()
    wait = next(
        op
        for op in plan.walk()
        if isinstance(op, ScopeSyncOp) and op.action.data == "wait" and op.channel.data == "drain"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(AssertionError, match="unmatched physical Set/Wait"):
        check(plan, 3)


def test_wrong_cube_id_cannot_overwrite_another_childs_unconsumed_event(manual_shared):
    plan = manual_shared.clone()
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, LoopChannelOp) and op.sym_name.data == "pv_load_ready"
    )
    channel.attributes["event_ids"] = ArrayAttr([IntAttr(0)])
    with pytest.raises(
        AssertionError, match="different logical notification|binary event overwrite"
    ):
        check(plan, 7)


@pytest.fixture(scope="module")
def compiled_shared():
    # One public compilation serves all counts and all cloned-IR mutations.
    return compile(shared_l0c_qk_pv, timeout_s=60).plan


@pytest.mark.parametrize("count", COUNTS)
def test_compiled_original_shared_l0c_has_fixed_raws_and_safe_physical_occupants(
    compiled_shared, count
):
    result = finite_check(compiled_shared, count)
    assert len(result.issued_tasks) == 73 * count
    assert result.bytes_read == 57344 * count
    assert result.bytes_written == 50176 * count


@pytest.mark.parametrize(
    "name,count",
    (("scope_c1", 1), ("scope_c7", 2), ("scope_c12", 2), ("scope_c0", 3), ("scope_c6", 3)),
)
def test_compiled_startup_steady_and_drain_families_are_independently_needed(
    compiled_shared, name, count
):
    plan = compiled_shared.clone()
    erase_channel(plan, name)
    with pytest.raises(AssertionError, match="unordered physical|binary event overwrite"):
        finite_check(plan, count)


def test_compiled_cube_removed_wait_cannot_be_hidden_by_activation_summary(compiled_shared):
    plan = compiled_shared.clone()
    wait = next(
        op
        for op in plan.walk()
        if isinstance(op, ScopeSyncOp)
        and op.action.data == "wait"
        and op.channel.data == "scope_c12"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(AssertionError, match="unmatched physical Set/Wait"):
        finite_check(plan, 3)


def test_compiled_cube_wrong_id_overwrites_a_live_c1_retirement_notification(compiled_shared):
    plan = compiled_shared.clone()
    protocol = next(
        op
        for op in plan.walk()
        if isinstance(op, ScopeChannelOp) and op.sym_name.data == "scope_c0"
    )
    protocol.attributes["event_ids"] = ArrayAttr([IntAttr(6)])
    with pytest.raises(
        AssertionError, match="binary event overwrite|different logical notification"
    ):
        finite_check(plan, 7)


@pytest.mark.parametrize("mutation", (_read_qk_operand_zero, _pv_recurrence))
def test_actual_dag_cannot_change_shared_operand_or_first_rw_producer(compiled_shared, mutation):
    plan = compiled_shared.clone()
    mutation(plan)
    with pytest.raises(AssertionError, match="physical WAW|physical WAR|logical RAW"):
        finite_check(plan, 2)


@pytest.mark.parametrize("action", ("set", "wait", "barrier"))
def test_actual_ir_activation_decodes_count_residue_maximum_and_signed_source_epoch(action):
    protocol = ScopeChannelOp.create(
        attributes={
            "sym_name": StringAttr("predicate"),
            "kind": StringAttr("local"),
            "sources": ArrayAttr([]),
            "destinations": ArrayAttr([]),
            "distance": IntAttr(-1),
            "period": IntAttr(3),
            "residues": ArrayAttr([IntAttr(0), IntAttr(2)]),
            "event_ids": ArrayAttr([IntAttr(0)]),
            "activation": DictionaryAttr(
                {
                    "count_period": IntAttr(3),
                    "count_residue": IntAttr(2),
                    "count_minimum": IntAttr(2),
                    "count_maximum": IntAttr(5),
                    "first_end": IntAttr(0),
                    "first_offset": IntAttr(1),
                    "last_end": IntAttr(1),
                    "last_offset": IntAttr(0),
                }
            ),
        }
    )
    for count in range(9):
        for parent in range(count):
            epoch = parent if action == "set" else parent + 1
            active = count in (2, 5) and 1 <= epoch < count and epoch % 3 in (0, 2)
            assert _scope_epoch(protocol, action, parent, count) == (epoch if active else None)
