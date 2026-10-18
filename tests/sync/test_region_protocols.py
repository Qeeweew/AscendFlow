"""Actual queue-prefix protocols across an irregular, possibly empty child.

The first fixture spells a rule's instructions on lowered DSL IR. It checks
the rule before the dependency builder is migrated; public compile coverage
belongs to the integration cases below, not to this explicit instruction test.
"""

import pytest

from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.dialects.physical import TaskOp
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import EventModelError, check_structured_events

PREFIX_STATE = """
def prefix_state(X:N.Tensor((2,64),N.f32),
                 Before:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),
                 After:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel():
        state=N.alloc_ub((64,))
        N.copy(X[0,:],state)
        for parent in N.serial(n):
            N.copy(state,Before[parent,:])
            for child in N.Pipelined(parent%2,stages=1):
                N.copy(X[1,:],state)
            N.copy(state,After[parent,:])
"""


def prefix_protocol(*, remove=None):
    module = lower_native_structure(N.from_source(PREFIX_STATE, disjoint_external=True)).module
    memory = collect_native_memory(module)
    original = collect_native_memory(module.clone())
    root = module.body.block.first_op
    initial, before, _write, after = tuple(op for op in module.walk() if isinstance(op, TaskOp))
    parent, child = memory.control.loops.values()
    actions = []

    def action(anchor, position, kind, src, dst, *, guards=(), name):
        if name != remove:
            actions.append(
                SyncAction(
                    anchor,
                    position,
                    (Actor.AIV0,),
                    kind,
                    src,
                    dst,
                    literal(0),
                    ActionGuard(tuple(guards)),
                    (name,),
                )
            )

    live = Comparison(parent.count, ">", literal(0))
    first = Comparison(parent.ordinal, "==", literal(0))
    action(
        initial,
        Placement.AFTER,
        SyncKind.LOCAL_SET,
        Engine.MTE2,
        Engine.MTE3,
        guards=(live,),
        name="initial-ready",
    )
    action(
        before,
        Placement.BEFORE,
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.MTE3,
        guards=(first,),
        name="initial-acquire",
    )
    action(
        before,
        Placement.AFTER,
        SyncKind.LOCAL_SET,
        Engine.MTE3,
        Engine.MTE2,
        name="readers-release",
    )
    action(
        child.operation,
        Placement.ENTRY,
        SyncKind.LOCAL_WAIT,
        Engine.MTE3,
        Engine.MTE2,
        name="writer-acquire",
    )
    action(
        child.operation,
        Placement.EXIT,
        SyncKind.LOCAL_SET,
        Engine.MTE2,
        Engine.MTE3,
        name="prefix-ready",
    )
    action(
        after,
        Placement.BEFORE,
        SyncKind.LOCAL_WAIT,
        Engine.MTE2,
        Engine.MTE3,
        name="reader-acquire",
    )
    for engine in (Engine.MTE2, Engine.MTE3):
        for kind in (SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT):
            actions.append(
                SyncAction(
                    root,
                    Placement.EXIT,
                    (Actor.AIV0,),
                    kind,
                    engine,
                    Engine.S,
                    literal(7),
                    protocols=("final-drain",),
                )
            )
    materialize_notifications(memory, actions)
    return collect_native_memory(module), original


@pytest.mark.parametrize("n", (0, 1, 2, 3, 6, 9))
def test_prefix_capture_keeps_empty_child_state_and_consumes_every_publication(n):
    final, original = prefix_protocol()
    check_structured_events(final, {"n": n}, logical_reference=original)


@pytest.mark.parametrize(
    "removed",
    ("initial-acquire", "readers-release", "writer-acquire", "prefix-ready", "reader-acquire"),
)
def test_prefix_protocol_missing_real_instruction_has_a_counterexample(removed):
    final, original = prefix_protocol(remove=removed)
    with pytest.raises(EventModelError):
        check_structured_events(final, {"n": 6}, logical_reference=original)


@pytest.fixture(scope="module")
def compiled_prefix():
    from ascendflow.passes.structured_sync import synchronize_structure

    return synchronize_structure(N.from_source(PREFIX_STATE, disjoint_external=True))


@pytest.mark.parametrize("n", (0, 1, 2, 3, 6, 9))
def test_public_compile_generates_balanced_empty_child_prefixes(compiled_prefix, n):
    from ascendflow.analysis.access_frontier import QueueEndpoint
    from ascendflow.target.ascend910b import Milestone
    from tests.models.structured_events import check_compiled_events

    ports = [
        need
        for need in compiled_prefix.dependencies.needs
        if isinstance(need.source.endpoint, QueueEndpoint)
    ]
    assert ports and all(need.source_cut.milestone is Milestone.ISSUE for need in ports)
    check_compiled_events(compiled_prefix, {"n": n}, invocations=2)


def test_conditional_write_with_multiple_holes_keeps_all_old_state_readers():
    from ascendflow.passes.structured_sync import synchronize_structure
    from tests.models.structured_events import check_compiled_events

    source = PREFIX_STATE.replace("for child in N.Pipelined(parent%2,stages=1):", "if parent%3==1:")
    result = synchronize_structure(N.from_source(source, disjoint_external=True))
    for n in (0, 1, 2, 3, 6, 10):
        check_compiled_events(result, {"n": n}, invocations=2)


def test_nested_reentry_reinitializes_state_without_reusing_an_unconsumed_event():
    from ascendflow.passes.structured_sync import synchronize_structure
    from tests.models.structured_events import check_compiled_events

    source = PREFIX_STATE.replace("(n,64)", "(3,n,64)")
    source = source.replace("Before[parent,:]", "Before[round,parent,:]")
    source = source.replace("After[parent,:]", "After[round,parent,:]")
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "N.copy(X[0" in line)
    source = "\n".join(
        lines[:start]
        + ["        for round in N.serial(3):"]
        + ["    " + line for line in lines[start:]]
    )
    result = synchronize_structure(N.from_source(source, disjoint_external=True))
    for n in (0, 1, 2, 5):
        check_compiled_events(result, {"n": n}, invocations=2)


def test_compiled_prefix_loses_safety_when_actual_wait_is_deleted():
    from ascendflow.dialects.structured import NotificationOp
    from ascendflow.passes.structured_sync import synchronize_structure
    from tests.models.structured_events import check_compiled_events

    result = synchronize_structure(N.from_source(PREFIX_STATE, disjoint_external=True))
    wait = next(
        op
        for op in result.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.LOCAL_WAIT.value
        and op.source.data == Engine.MTE3.value
        and op.destination.data == Engine.MTE2.value
    )
    wait.parent.erase_op(wait)
    with pytest.raises(EventModelError):
        check_compiled_events(result, {"n": 6})


def test_two_aiv_use_one_body_with_different_child_activity_and_full_quorum():
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from ascendflow import compile
    from ascendflow.dialects.io import parse
    from ascendflow.dialects.structured import NotificationOp

    source = """
    def lanes(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
              Work:N.Tensor((16,16),N.f32,access="write"),
              Out:N.Tensor((3,n,16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
        with N.Kernel(1,kind="mixed"):
            a1=N.alloc_l1((16,16),N.f16)
            b1=N.alloc_l1((16,16),N.f16)
            a0=N.alloc_l0a((16,16),N.f16)
            b0=N.alloc_l0b((16,16),N.f16)
            c=N.alloc_l0c((16,16),N.f32)
            x=N.alloc_ub((8,16),N.f32)
            for i in N.Pipelined(n,stages=1):
                with N.Cube():
                    N.copy(A,a1)
                    N.copy(B,b1)
                    N.copy(a1,a0)
                    N.copy(b1,b0)
                    N.matmul(a0,b0,accum=c,initialize=True)
                    N.copy(c,Work)
                with N.Vector() as sid:
                    N.copy(Work[sid*8:sid*8+8,:],x)
                    for child in N.Pipelined((i+sid)%2,stages=1):
                        N.mul(x,2.0,out=x)
                    N.copy(x,Out[round,i,sid*8:sid*8+8,:])
    """
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "for i in N.Pipelined" in line)
    source = "\n".join(
        lines[:start]
        + ["            for round in N.serial(3):"]
        + ["    " + line for line in lines[start:]]
    )
    artifact = compile(N.from_source(source, disjoint_external=True))
    assert artifact.asc.count("if ASC_IS_AIV") == 1
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for n in (0, 1, 2, 5):
        check_structured_events(final, {"n": n}, logical_reference=original)
    release = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.CROSS_SET.value
        and op.source.data == Engine.MTE2.value
    )
    release.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
    with pytest.raises(EventModelError, match="quorum|contribution"):
        check_structured_events(final, {"n": 2}, logical_reference=original)
