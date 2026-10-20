"""Real per-actor capture/acquire prefixes retain two AIV contributions."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.event_contracts import SUPPORTED_CROSS_IDS
from ascendflow.analysis.event_sharing import CounterSharing
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)


@pytest.mark.parametrize("messages,reverse", [(2, False), (2, True), (16, False)])
def test_counter_packing_requires_fifo_receivers_and_bounds_the_whole_backlog(messages, reverse):
    # The sixteen-message case is a finite broadcast with no feedback; the
    # two-message cases additionally exercise the reusable physical ring.
    slots, index, count = ("1", "0", "N.min(n,1)") if messages > 15 else ("2", "i%2", "n")
    source = N.from_source(
        f"""
def messages(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
             Work:N.Tensor(({slots},{messages},16,16),N.f32,access="write"),
             Out:N.Tensor((n,{messages},16,16),N.f32,access="write",write_disjoint=True),
             n:N.int64):
    with N.Kernel(1,kind="mixed"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        x=N.alloc_ub((8,16),N.f32)
        with N.Cube():
            N.copy(A,a1)
            N.copy(B,b1)
            N.copy(a1,a0)
            N.copy(b1,b0)
            N.matmul(a0,b0,accum=c0,initialize=True)
        for i in N.Dataflow({count}):
            with N.Cube(), N.Stage(0):
                for panel in N.unroll({messages}):
                    N.copy(c0,Work[{index},panel,:,:])
            with N.Vector() as lane, N.Stage(1):
                for panel in N.unroll({messages}):
                    column={messages}-1-panel if {reverse} else panel
                    N.copy(Work[{index},column,lane*8:lane*8+8,:],x)
                    N.copy(x,Out[i,column,lane*8:lane*8+8,:])
""",
        disjoint_external=True,
    )
    from ascendflow.passes.structured_sync import synchronize_structure

    plan = synchronize_structure(source)
    packed = [witness for witness in plan.resources.sharing if isinstance(witness, CounterSharing)]
    assert bool(packed) != reverse
    assert all(
        len(witness.earlier.sources) == len(witness.later.sources) == 1 for witness in packed
    )
    assert 0 < plan.resources.work.maximum_counter_backlog <= 15
    if messages > 15:
        # An ordered stream still needs another ID once its complete credits
        # could collectively leave sixteen unconsumed contributions.
        assert plan.resources.work.maximum_counter_backlog == 15
    for count in (0, 1, 2, 3):
        # One kernel invocation; count=3 also exercises physical-ring reuse.
        # Repeating complete mixed bodies without a launch completion edge
        # would let another AIC overwrite GM before the old AIV has returned.
        execution = check_compiled_events(plan, {"n": count}, max_commands=30000)
        if count == 1 and not reverse:
            publications = [
                c
                for c in execution.natives
                if c.source is Engine.FIX and c.occurrence.actor is Actor.AIC
            ]
            first_import = next(
                c
                for c in execution.natives
                if c.source is Engine.MTE2 and c.occurrence.actor is Actor.AIV0
            )
            assert not execution.happens_before(publications[1].end, first_import.begin)


def test_two_publishers_retain_separate_counters_without_a_quorum_round_proof():
    source = N.from_source(
        """
def messages(A:N.Tensor((n,2,32,16),N.f16),B:N.Tensor((16,16),N.f16),
             Work:N.Tensor((2,2,32,16),N.f16,access="write"),
             Out:N.Tensor((n,2,32,16),N.f32,access="write",write_disjoint=True),
             n:N.int64):
    with N.Kernel(1,kind="mixed"):
        x=N.alloc_ub((16,16),N.f16)
        a1=N.alloc_l1((2,32,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((32,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((32,16),N.f32)
        with N.Cube():
            N.copy(B,b1)
            N.copy(b1,b0)
        for i in N.Dataflow(n):
            with N.Vector() as lane, N.Stage(0):
                for panel in N.unroll(2):
                    N.copy(A[i,panel,lane*16:lane*16+16,:],x)
                    N.copy(x,Work[i%2,panel,lane*16:lane*16+16,:])
            with N.Cube(), N.Stage(1):
                for panel in N.unroll(2):
                    N.copy(Work[i%2,panel,:,:],a1[panel,:,:])
                    N.copy(a1[panel,:,:],a0)
                    N.matmul(a0,b0,accum=c0,initialize=True)
                    N.copy(c0,Out[i,panel,:,:])
""",
        disjoint_external=True,
    )
    from ascendflow.passes.structured_sync import synchronize_structure

    plan = synchronize_structure(source)
    publications = [
        assignment
        for assignment in plan.resources.assignments
        if len(assignment.notification.sources) == 2
    ]
    assert len(publications) >= 2
    assert len({assignment.ids for assignment in publications}) >= 2
    assert all(
        len(witness.earlier.sources) == len(witness.later.sources) == 1
        for witness in plan.resources.sharing
        if isinstance(witness, CounterSharing)
    )
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count})


@pytest.mark.parametrize("cube", [False, True])
def test_independent_outputs_retain_separate_completion_cuts(cube):
    source = (
        """
def independent(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
                Out:N.Tensor((2,16,16),N.f32,access="write",write_disjoint=True)):
    with N.Kernel(1,kind="cube"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        c1=N.alloc_l0c((16,16),N.f32)
        N.copy(A,a1)
        N.copy(B,b1)
        N.copy(a1,a0)
        N.copy(b1,b0)
        N.matmul(a0,b0,accum=c0,initialize=True)
        N.matmul(a0,b0,accum=c1,initialize=True)
        N.copy(c0,Out[0,:,:])
        N.copy(c1,Out[1,:,:])
"""
        if cube
        else """
def independent(X:N.Tensor((2,64),N.f32),
                Out:N.Tensor((2,64),N.f32,access="write",write_disjoint=True)):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((64,),N.f32)
        y=N.alloc_ub((64,),N.f32)
        u=N.alloc_ub((64,),N.f32)
        v=N.alloc_ub((64,),N.f32)
        N.copy(X[0,:],x)
        N.copy(X[1,:],y)
        N.mul(x,2.0,out=u)
        N.mul(y,2.0,out=v)
        N.copy(u,Out[0,:])
        N.copy(v,Out[1,:])
"""
    )
    artifact = compile(N.from_source(source, disjoint_external=True))
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    checked = check_structured_events(final, logical_reference=original, invocations=2)
    producer, consumer = (Engine.M, Engine.FIX) if cube else (Engine.MTE2, Engine.V)
    for invocation in (0, 1):
        producers = [
            command
            for command in checked.natives
            if command.occurrence.invocation == invocation and command.source is producer
        ]
        consumers = [
            command
            for command in checked.natives
            if command.occurrence.invocation == invocation and command.source is consumer
        ]
        assert len(producers) == len(consumers) == 2
        for writer, reader in zip(producers, consumers, strict=True):
            assert checked.happens_before(writer.end, reader.begin)
        assert not checked.happens_before(producers[1].end, consumers[0].begin), (
            "an independent first reader must not wait for the second producer's completion"
        )


def test_bulk_load_waits_for_every_old_reader_before_reusing_its_ring():
    program = N.from_source(
        """
def local_prefix(X:N.Tensor((n,8,64),N.f32),
                 Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        inputs=N.alloc_ub((2,8,64),N.f32)
        total=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for i in N.Pipelined(n):
                with N.Stage(0,name="load"):
                    N.copy(X[i,:,:],inputs[i%2,:,:])
                with N.Stage(1,name="sum"):
                    N.copy(inputs[i%2,0,:],total[i%2,:])
                    for panel in N.unroll(1,8):
                        N.add(total[i%2,:],inputs[i%2,panel,:],out=total[i%2,:])
                with N.Stage(2,name="store"):
                    N.copy(total[i%2,:],Out[i,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for count in (0, 1, 2, 5, 7):
        check_structured_events(final, {"n": count}, logical_reference=original, invocations=2)
    release = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.LOCAL_WAIT.value
        and op.source.data == Engine.V.value
        and op.destination.data == Engine.MTE2.value
    )
    release.parent.erase_op(release)
    with pytest.raises(EventModelError):
        check_structured_events(final, {"n": 7}, logical_reference=original)


def test_independent_gm_handoffs_keep_complete_quorums_and_fit_real_capacity():
    program = N.from_source(
        """
def paired(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
           Work:N.Tensor((2,2,16,16),N.f32,access="write"),
           Out:N.Tensor((n,16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="mixed"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        x=N.alloc_ub((8,16),N.f32)
        y=N.alloc_ub((8,16),N.f32)
        for i in N.Dataflow(n):
            with N.Cube(), N.Stage(0,name="cube"):
                N.copy(A,a1)
                N.copy(B,b1)
                N.copy(a1,a0)
                N.copy(b1,b0)
                N.matmul(a0,b0,accum=c0,initialize=True)
                N.copy(c0,Work[i%2,0,:,:])
                N.copy(c0,Work[i%2,1,:,:])
            with N.Vector() as lane, N.Stage(1,name="vector"):
                N.copy(Work[i%2,0,lane*8:lane*8+8,:],x)
                N.copy(Work[i%2,1,lane*8:lane*8+8,:],y)
                N.add(x,y,out=x)
                N.copy(x,Out[i,lane*8:lane*8+8,:])
""",
        disjoint_external=True,
    )
    artifact = compile(program)
    assert artifact.report["analysis_work"]["resources"]["allocated_cross_ids"] <= len(
        SUPPORTED_CROSS_IDS
    )
    final = collect_native_memory(artifact.plan)
    original = collect_native_memory(parse(artifact.before_ir))
    for count in (0, 1, 2, 5):
        check_structured_events(final, {"n": count}, logical_reference=original)
    release = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.CROSS_SET.value
        and op.source.data == Engine.MTE2.value
    )
    release.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
    with pytest.raises(EventModelError, match="quorum|contribution"):
        check_structured_events(final, {"n": 5}, logical_reference=original)
