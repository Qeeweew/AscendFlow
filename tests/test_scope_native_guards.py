"""Actual parent first/last guards retain complete native recipes and two clocks."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from tests.fixtures.attention_target import attention_target
from xdsl_npu import language as N
from xdsl_npu.ir import (
    CallOp,
    EndOp,
    LoopChannelOp,
    LoopSyncOp,
    OnceOp,
    ScopeGuardOp,
    StreamOp,
    TaskOp,
    ViewOp,
    dump,
    parse,
)
from xdsl_npu.lowering import _Actor, _work
from xdsl_npu.model import CompileError
from xdsl_npu.scope_lowering import _segments, lower_scopes
from xdsl_npu.scope_predicates import PARENT_CASES, case_activation, classify_guard, guard_active
from xdsl_npu.source_ir import AllocationOp, PipelineOp
from xdsl_npu.source_scopes import ClockFrame, collect_source_scopes

SOURCE = """
def guarded(O:N.Tensor((2,8,64),N.f16,access="write"),n:N.int64):
    with N.Kernel():
        a=N.alloc_ub((2,8,64),N.f32)
        b=N.alloc_ub((2,8,64),N.f16)
        tail=N.alloc_ub((2,8,64),N.f32)
        statistics=N.alloc_ub((8,),N.f32)
        broadcast=N.alloc_ub((8,8),N.f32)
        for t in N.Pipelined(n,stages=3):
            with N.Stage(0,name="initialize"):
                if t == 0:
                    N.fill(statistics,1.0)
            with N.Stage(2,name="update"):
                for row in N.Pipelined(2,stages=3):
                    with N.Stage(0,name="prepare"):
                        N.fill(a[row%2,:,:],2.0)
                    with N.Stage(1,name="normalize"):
                        N.add(a[row%2,:,:],a[row%2,:,:],out=a[row%2,:,:])
                        if t == n - 1:
                            N.mul(a[row%2,:,:],statistics[:,None],out=a[row%2,:,:],scratch=broadcast)
                            N.cast(a[row%2,:,:],out=b[row%2,:,:])
                    with N.Stage(2,name="export"):
                        if t == n - 1:
                            N.copy(b[row%2,:,:],O[row,:,:])
                        N.fill(tail[row%2,:,:],3.0)
"""


def source_parts(text=SOURCE):
    program = N.from_source(text, disjoint_external=True).to_ir().body.block.first_op
    (parent,) = collect_source_scopes(program).items
    guards = {
        op: kind
        for op in program.walk()
        if isinstance(op, scf.IfOp)
        if (kind := classify_guard(op, parent.op)) is not None
    }
    return program, parent, guards


def native_leaf(*, once=False, guarded=True, parent_generation=True):
    """Inspect complete recipes directly at the native lowering boundary.

    Views in this fixture are parent-independent. Ordinary Actor allocation
    binding suffices; production scope_lowering separately builds captures and
    still runs its complete final extract_scopes validation.
    """
    program, parent, guards = source_parts()
    scope, items, is_once = next(
        segment for segment in _segments(parent, "vector") if segment[2] == once
    )
    assert is_once == once
    pipeline = parent.op if once else scope.op
    records = _work(
        program,
        pipeline,
        "vector",
        0,
        operations=tuple(item.op for item in items) if once else None,
        scoped=items,
        scope=scope,
        nested=True,
        parent_guards=guards,
    )
    if not guarded:
        records = [replace(record, parent_guard=None) for record in records]
    physical_parent = StreamOp.create(
        attributes={"count": IntAttr(3)}, regions=[Region(Block(arg_types=[IndexType()]))]
    )
    call = CallOp.create(
        operands=[physical_parent.body.block.args[0]],
        attributes={
            "uid": StringAttr("native_guard"),
            "stage": IntAttr(0 if once else 2),
            "logical_phase": IntAttr(0),
            "issue_order": IntAttr(0),
        },
        regions=[Region(Block())],
    )
    actor = _Actor(
        program,
        pipeline,
        "aiv0",
        "vector",
        0,
        tuple(op for op in program.walk() if isinstance(op, AllocationOp)),
        symbolic=True,
        records=records,
        once=once,
        region_name="native_guard",
        ambient_frames=(ClockFrame(parent.op),),
        parent_generation=call.generation if parent_generation else None,
    )
    actor.emit()
    actor.kernel.block.detach_op(actor.region)
    call.body.block.add_ops([actor.region, EndOp()])
    physical_parent.body.block.add_ops([call, EndOp()])
    actor.kernel.block.add_ops([physical_parent, EndOp()])
    actor.kernel.module.verify()
    return actor, call


def test_normal_scope_lowering_retains_one_guarded_child_template():
    program = N.from_source(SOURCE, disjoint_external=True).to_ir().body.block.first_op
    physical = lower_scopes(program)
    physical.verify()
    calls = [op for op in physical.walk() if isinstance(op, CallOp)]
    assert len(calls) == 2
    children = [
        next(op for op in call.body.block.ops if isinstance(op, (OnceOp, StreamOp)))
        for call in calls
    ]
    assert isinstance(children[0], OnceOp)
    assert isinstance(children[1], StreamOp) and children[1].count.data == 2
    for call, child in zip(calls, children):
        guards = [op for op in child.walk() if isinstance(op, ScopeGuardOp)]
        assert [guard.kind.data for guard in guards] == (
            ["first"] if isinstance(child, OnceOp) else ["last"] * 3
        )
        assert all(guard.parent_generation is call.generation for guard in guards)
        assert all(guard.child_generation is child.body.block.args[0] for guard in guards)


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7))
def test_four_cases_partition_actual_parent_generations_without_runtime_expansion(count):
    assert PARENT_CASES == ("single", "first", "middle", "last")
    covered = []
    for label in PARENT_CASES:
        activation = case_activation(label)
        if not activation.count.contains(count):
            continue
        for generation in range(
            activation.first.evaluate(count), activation.last.evaluate(count) + 1
        ):
            covered.append(generation)
            assert guard_active("first", label) == (generation == 0)
            assert guard_active("last", label) == (generation == count - 1)
    assert sorted(covered) == list(range(count))


@pytest.mark.parametrize("kind,label", (("middle", "first"), ("first", "unknown")))
def test_case_helpers_reject_unknown_kind_or_partition(kind, label):
    with pytest.raises(CompileError):
        guard_active(kind, label)
    if label == "unknown":
        with pytest.raises(CompileError):
            case_activation(label)


def test_once_initialization_uses_parent_first_and_its_own_zero_child_clock():
    actor, call = native_leaf(once=True)
    assert isinstance(actor.region, OnceOp)
    guards = [op for op in actor.region.walk() if isinstance(op, ScopeGuardOp)]
    assert len(guards) == 1 and guards[0].kind.data == "first"
    assert guards[0].parent_generation is call.generation
    assert guards[0].child_generation is actor.region.body.block.args[0]
    assert guards[0].parent_generation is not guards[0].child_generation
    assert all(task.stage.data == 0 for task in guards[0].walk() if isinstance(task, TaskOp))


def test_child_last_parent_guard_keeps_brcb_cast_store_in_their_native_stages():
    actor, call = native_leaf()
    guards = [op for op in actor.region.walk() if isinstance(op, ScopeGuardOp)]
    assert len(guards) == 3 and all(op.kind.data == "last" for op in guards)
    for guard in guards:
        assert guard.parent_generation is call.generation
        assert guard.child_generation is actor.region.body.block.args[0]
        assert any(isinstance(op, ViewOp) for op in guard.body.block.ops)
    tasks = [[op for op in guard.body.block.ops if isinstance(op, TaskOp)] for guard in guards]
    assert [task.native.data for task in tasks[0]] == ["brcb8", "mul_row_broadcast"]
    assert [task.native.data for task in tasks[1]] == ["convert_rn"]
    assert [task.native.data for task in tasks[2]] == ["copy"]
    assert [[task.stage.data for task in group] for group in tasks] == [[1, 1], [1], [2]]
    # An ordinary effect after the last-parent store remains ordinary: this
    # condition is neither a child terminal suffix nor a whole-call guard.
    trailing = [op for op in actor.region.body.block.ops if isinstance(op, TaskOp)]
    assert trailing[-1].native.data == "fill" and trailing[-1].stage.data == 2


@pytest.mark.parametrize("once", (False, True))
def test_native_recipe_uids_stages_and_issue_order_survive_guard_wrapping(once):
    guarded, _ = native_leaf(once=once)
    ordinary, _ = native_leaf(once=once, guarded=False)

    def schedule(actor):
        return [
            (task.uid.data, task.native.data, task.stage.data, task.issue_order.data, task.params)
            for task in actor.region.walk()
            if isinstance(task, TaskOp)
        ]

    assert schedule(guarded) == schedule(ordinary)
    assert guarded.region.count == ordinary.region.count


def test_parent_guard_cannot_emit_without_a_bound_parent_generation():
    with pytest.raises(CompileError, match="actual parent generation"):
        native_leaf(parent_generation=False)


@pytest.mark.parametrize(
    "condition",
    ("t % 2 == 0", "t == n - 2", "row == 0", "t < n", "t == 1"),
)
def test_source_predicates_cannot_guess_a_parent_clock_or_boundary(condition):
    text = SOURCE.replace("if t == n - 1:", "if " + condition + ":")
    with pytest.raises(CompileError, match="scope guard|scope guards|predicate"):
        source_parts(text)


def test_effect_free_runtime_cycle_is_rejected_before_native_construction(monkeypatch):
    text = SOURCE.replace(
        "if t == 0:\n                    N.fill(statistics,1.0)",
        "if t % 2 == 0:\n                    pass",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported runtime-cycle guard reached native construction")

    monkeypatch.setattr(_Actor, "emit", forbidden)
    with pytest.raises(CompileError, match="runtime-cycle"):
        lower_scopes(N.from_source(text, disjoint_external=True).to_ir().body.block.first_op)


def test_source_else_and_results_are_not_silently_treated_as_parent_boundaries():
    text = SOURCE.replace(
        "N.fill(statistics,1.0)\n            with",
        "N.fill(statistics,1.0)\n                else:\n                    N.fill(statistics,2.0)\n            with",
    )
    with pytest.raises(CompileError, match="no else or results"):
        source_parts(text)
    program, parent, _ = source_parts()
    old = next(op for op in program.walk() if isinstance(op, scf.IfOp))
    regions = []
    for _ in range(2):
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        regions.append(Region(Block([zero, scf.YieldOp(zero.result)])))
    returning = scf.IfOp.create(operands=[old.cond], result_types=[IndexType()], regions=regions)
    old.parent.insert_op_before(returning, old)
    with pytest.raises(CompileError, match="no else or results"):
        classify_guard(returning, parent.op)


def test_child_local_last_keeps_its_existing_terminal_suffix_rule():
    program, parent, guards = source_parts(SOURCE.replace("t == n - 1", "row == 1"))
    assert set(guards.values()) == {"first"}
    scope, items, _ = next(segment for segment in _segments(parent, "vector") if not segment[2])
    with pytest.raises(CompileError, match="terminal logical effect suffix"):
        _work(
            program,
            scope.op,
            "vector",
            0,
            scoped=items,
            scope=scope,
            nested=True,
            parent_guards=guards,
        )


def test_original_attention_guards_are_parent_boundaries_only_in_nested_organization():
    for continuous in (False, True):
        program = attention_target(continuous=continuous).to_ir().body.block.first_op
        parent = next(op for op in program.walk() if isinstance(op, PipelineOp))
        predicates = [op for op in program.walk() if isinstance(op, scf.IfOp)]
        assert len(predicates) == 5
        if continuous:
            with pytest.raises(CompileError, match="runtime-cycle"):
                classify_guard(predicates[0], parent)
        else:
            assert [classify_guard(op, parent) for op in predicates] == [
                "first",
                "first",
                "first",
                "last",
                "last",
            ]


def loop_protocol(parent_generation, child_generation):
    sync = LoopSyncOp.create(
        operands=[child_generation, parent_generation],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr(""),
            "lower": IntAttr(0),
            "upper_gap": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "scope_cases": ArrayAttr([StringAttr("single"), StringAttr("last")]),
        },
    )
    channel = LoopChannelOp.create(
        attributes={
            "sym_name": StringAttr("conditional"),
            "kind": StringAttr("local"),
            "sources": ArrayAttr([]),
            "destinations": ArrayAttr([]),
            "distance": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
            "event_ids": ArrayAttr([IntAttr(0)]),
            "scope_cases": ArrayAttr([StringAttr("single"), StringAttr("last")]),
        }
    )
    return sync, channel


def test_guard_ir_and_conditional_loop_protocol_operands_round_trip():
    parent = StreamOp.create(
        attributes={"count": IntAttr(3)}, regions=[Region(Block(arg_types=[IndexType()]))]
    )
    call = CallOp.create(
        operands=[parent.body.block.args[0]],
        attributes={
            "uid": StringAttr("conditional"),
            "stage": IntAttr(0),
            "logical_phase": IntAttr(0),
            "issue_order": IntAttr(0),
        },
        regions=[Region(Block())],
    )
    child = StreamOp.create(
        attributes={"count": IntAttr(2)}, regions=[Region(Block(arg_types=[IndexType()]))]
    )
    generation = child.body.block.args[0]
    guard = ScopeGuardOp.create(
        operands=[call.generation, generation],
        attributes={"kind": StringAttr("last")},
        regions=[Region(Block([EndOp()]))],
    )
    sync, channel = loop_protocol(call.generation, generation)
    child.body.block.add_ops([guard, sync, channel, EndOp()])
    call.body.block.add_ops([child, EndOp()])
    parent.body.block.add_ops([call, EndOp()])
    module = ModuleOp([parent])
    module.verify()
    assert dump(parse(dump(module))) == dump(module)
    assert sync.parent_generation is call.generation
    unconditional = LoopSyncOp.create(
        operands=[generation],
        attributes={
            name: value for name, value in sync.attributes.items() if name != "scope_cases"
        },
    )
    assert unconditional.parent_generation is None and unconditional.scope_cases is None
    unconditional.verify()


@pytest.mark.parametrize("annotation", ("channel_cases", "sync_cases", "sync_parent", "sync_both"))
def test_conditional_loop_annotations_require_actual_call_ancestry(annotation):
    parent = arith.ConstantOp.from_int_and_width(3, IndexType())
    child = arith.ConstantOp.from_int_and_width(1, IndexType())
    sync, channel = loop_protocol(parent.result, child.result)
    if annotation == "channel_cases":
        protocol = channel
    else:
        if annotation == "sync_cases":
            sync.operands = [child.result]
        elif annotation == "sync_parent":
            del sync.attributes["scope_cases"]
        protocol = sync
    module = ModuleOp([parent, child, protocol])
    with pytest.raises(VerifyException, match="actual enclosing CallOp"):
        module.verify()


def test_case_filter_can_normalize_conditional_protocol_to_ordinary_loop():
    parent = arith.ConstantOp.from_int_and_width(3, IndexType())
    child = arith.ConstantOp.from_int_and_width(1, IndexType())
    sync, channel = loop_protocol(parent.result, child.result)
    sync.operands = [child.result]
    del sync.attributes["scope_cases"]
    del channel.attributes["scope_cases"]
    ModuleOp([parent, child, sync, channel]).verify()


@pytest.mark.parametrize("kind,terminated", (("middle", True), ("first", False)))
def test_guard_ir_rejects_unknown_kind_and_unterminated_body(kind, terminated):
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    guard = ScopeGuardOp.create(
        operands=[zero.result, zero.result],
        attributes={"kind": StringAttr(kind)},
        regions=[Region(Block([EndOp()] if terminated else []))],
    )
    with pytest.raises(VerifyException, match="scope[ _]guard"):
        guard.verify()
