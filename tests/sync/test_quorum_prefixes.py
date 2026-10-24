"""Real per-actor capture/acquire prefixes retain two AIV contributions."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.event_contracts import SUPPORTED_CROSS_IDS, CounterFIFOGroup
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)


@pytest.mark.parametrize(
    "messages,reverse", [(2, False), (2, True), (16, False), (17, False), (17, True)]
)
def test_counter_packing_requires_fifo_receivers_and_bounds_the_whole_backlog(messages, reverse):
    # The sixteen-message case is a finite broadcast with no feedback; the
    # two-message cases additionally exercise the reusable physical ring.
    slots, index, count = ("1", "0", "N.min(n,1)") if messages > 15 else ("2", "i%2", "n")
    source = N.from_source(
        f"""
def messages(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
             Work:N.Tensor(({slots},{messages},16,16),N.f32,access="write"),
             Out:N.Tensor((n,{messages},16,16),N.f32,access="write",write_disjoint=True),
             n:N.i64):
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
    packed = [
        witness
        for witness in plan.resources.sharing
        if isinstance(witness, CounterFIFOGroup)
        and witness.certificate.messages[0][2].sources[0].actor is Actor.AIC
    ]
    # Reversing the imports needs the last publication first. Its existing
    # prefix acquire covers the remaining imports before resource allocation.
    assert bool(packed) == (messages > len(SUPPORTED_CROSS_IDS) and not reverse)
    if reverse:
        # Reversed messages may still pack mutually exclusive ring residues
        # at identical sites; they cannot share distinct reversed messages.
        assert all(
            len({n.sources[0].operation for r, _, n in w.certificate.messages if r in w.resources})
            == 1
            for w in packed
        )
    assert all(len(n.sources) == 1 for w in packed for _, _, n in w.certificate.messages)
    assert 0 < plan.resources.work.maximum_counter_backlog <= 15
    if messages == 16:
        # Keep the capacity counterexample independently of the allocator's
        # preference for exclusive IDs: forcing all sixteen messages onto
        # one counter permits more than fifteen outstanding contributions.
        from xdsl.dialects import arith
        from xdsl.dialects.builtin import IndexType

        mutant = plan.module.clone()
        root = mutant.body.block.first_op.body.block
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        root.insert_op_before(zero, root.first_op)
        for op in mutant.walk():
            if isinstance(op, NotificationOp) and op.kind.data in {
                SyncKind.CROSS_SET.value,
                SyncKind.CROSS_WAIT.value,
            }:
                op.operands = (zero.result,)
        with pytest.raises(EventModelError, match="counter backlog"):
            check_structured_events(collect_native_memory(mutant), {"n": 1}, max_commands=30000)
    for count in (0, 1, 2, 3):
        # One kernel invocation; count=3 also exercises physical-ring reuse.
        # Repeating complete mixed bodies without a launch completion edge
        # would let another AIC overwrite GM before the old AIV has returned.
        execution = check_compiled_events(plan, {"n": count}, max_commands=30000)
        if count == 1:
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
            if reverse:
                assert execution.happens_before(publications[-1].end, first_import.begin)
            else:
                assert not execution.happens_before(publications[1].end, first_import.begin)


@pytest.mark.parametrize(
    "messages,reverse", [(2, False), (2, True), (4, False), (5, False), (5, True), (17, False)]
)
def test_two_publishers_share_only_with_matching_fifo_and_complete_quorums(messages, reverse):
    count = "N.min(n,1)" if messages > 15 else "n"
    source = N.from_source(
        f"""
def messages(A:N.Tensor((n,{messages},32,16),N.f16),B:N.Tensor((16,16),N.f16),
             Work:N.Tensor((2,{messages},32,16),N.f16,access="write"),
             Out:N.Tensor((n,{messages},32,16),N.f32,access="write",write_disjoint=True),
             n:N.i64):
    with N.Kernel(1,kind="mixed"):
        x=N.alloc_ub((16,16),N.f16)
        a1=N.alloc_l1(({messages},32,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((32,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((32,16),N.f32)
        with N.Cube():
            N.copy(B,b1)
            N.copy(b1,b0)
        for i in N.Dataflow({count}):
            with N.Vector() as lane, N.Stage(0):
                for panel in N.unroll({messages}):
                    N.copy(A[i,panel,lane*16:lane*16+16,:],x)
                    N.copy(x,Work[i%2,panel,lane*16:lane*16+16,:])
            with N.Cube(), N.Stage(1):
                for panel in N.unroll({messages}):
                    column={messages}-1-panel if {reverse} else panel
                    destination=0 if {messages}>15 else column
                    N.copy(Work[i%2,column,:,:],a1[destination,:,:])
                    if {messages}<=15:
                        N.copy(a1[column,:,:],a0)
                        N.matmul(a0,b0,accum=c0,initialize=True)
                        N.copy(c0,Out[i,column,:,:])
                if {messages}>15:
                    N.copy(a1[0,:,:],a0)
                    N.matmul(a0,b0,accum=c0,initialize=True)
                    N.copy(c0,Out[i,0,:,:])
""",
        disjoint_external=True,
    )
    from ascendflow.passes.structured_sync import synchronize_structure

    plan = synchronize_structure(source)
    packed = [
        witness
        for witness in plan.resources.sharing
        if isinstance(witness, CounterFIFOGroup)
        and len(witness.certificate.messages[0][2].sources) == 2
    ]
    # Five ring-buffered panels pressure the combined namespace, but either
    # direction's complete words may supply the necessary packing. Seventeen
    # finite forward messages force the actual two-publisher path to pack.
    if messages > 15:
        assert packed
    if messages <= 4:
        assert not packed
    for group in plan.resources.counter_groups:
        certificate = group.certificate
        assert set(group.resources) <= set(certificate.resources)
        assert all(word == certificate.words[0][2] for _, _, word in certificate.words)
        assert all(bound <= 15 for _, bound in certificate.contributions)
    if reverse:
        assert all(
            len({n.sources[0].operation for r, _, n in w.certificate.messages if r in w.resources})
            == 1
            for w in packed
        )
    assignments = {
        assignment.notification: assignment.ids for assignment in plan.resources.assignments
    }
    for witness in packed:
        for resource, _, notification in witness.certificate.messages:
            if resource in witness.resources:
                assert assignments[notification] == (witness.event,)
                assert {point.actor for point in notification.sources} == {Actor.AIV0, Actor.AIV1}
    assert 0 < plan.resources.work.maximum_counter_backlog <= 15
    for count in (0, 1, 3):
        checked = check_compiled_events(plan, {"n": count})
        if count != 1:
            continue
        readers = [
            command
            for command in checked.natives
            if command.occurrence.actor is Actor.AIC
            and any(
                span.root.sym_name.data == "Work" and "R" in span.mode for span in command.spans
            )
        ]
        writers = {
            actor: [
                command
                for command in checked.natives
                if command.occurrence.actor is actor
                and command.source is Engine.MTE3
                and any(
                    span.root.sym_name.data == "Work" and "W" in span.mode for span in command.spans
                )
            ]
            for actor in (Actor.AIV0, Actor.AIV1)
        }
        assert len(readers) == messages
        for actor, publications in writers.items():
            assert len(publications) == messages
            for writer in publications:
                written = next(span for span in writer.spans if span.root.sym_name.data == "Work")
                reader = next(
                    command
                    for command in readers
                    if any(
                        span.root is written.root and span.start <= written.start < span.end
                        for span in command.spans
                    )
                )
                assert checked.happens_before(writer.end, reader.begin)
            if not reverse:
                assert not checked.happens_before(publications[1].end, readers[0].begin), (
                    "the first acquire must not wait for a later message's data"
                )
                other = Actor.AIV1 if actor is Actor.AIV0 else Actor.AIV0
                assert not checked.happens_before(writers[other][0].end, publications[1].end), (
                    "one AIV may publish its second message before its peer publishes the first"
                )

    if messages == 2 and not reverse:
        damaged = plan.module.clone()
        publication = next(
            operation
            for operation in damaged.walk()
            if isinstance(operation, NotificationOp)
            and operation.kind.data == SyncKind.CROSS_SET.value
            and operation.source.data == Engine.MTE3.value
        )
        publication.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
        with pytest.raises(EventModelError, match="quorum|contribution"):
            check_structured_events(
                collect_native_memory(damaged),
                {"n": 3},
                logical_reference=collect_native_memory(parse(plan.before_ir)),
            )


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
                 Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.i64):
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
           Out:N.Tensor((n,16,16),N.f32,access="write",write_disjoint=True),n:N.i64):
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
