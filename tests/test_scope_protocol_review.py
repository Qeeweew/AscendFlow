"""Independent checks of actual parent synchronization order and lifetimes."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import LoopSyncOp, ScopeSyncOp, TaskOp
from xdsl_npu.loop_boundary import stream_operations
from xdsl_npu.loop_graph import Command
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_sync import ParentCompletion, ScopeProtocol, _analyses, _install, verify_scopes

ORDER = """
def order(X:N.Tensor((n,16,16),N.f16),Y:N.Tensor((n,8),N.f32),n:N.int64):
    with N.Kernel(kind="mixed"):
        q=N.alloc_l1((16,16))
        a=N.alloc_ub((8,))
        b=N.alloc_ub((8,))
        for t in N.Pipelined(n,stages=1):
            with N.Stage(0,name="cube"),N.Cube():
                for j in N.Pipelined(1,stages=1):
                    N.copy(X[t,:,:],q)
            with N.Stage(0,name="load"),N.Vector():
                for j in N.Pipelined(1,stages=1):
                    N.copy(Y[t,:],a)
            with N.Stage(0,name="fill"),N.Vector():
                for j in N.Pipelined(1,stages=1):
                    N.fill(b,1.0)
"""

FIFO = """
def fifo(X:N.Tensor((4,8),N.f32)):
    with N.Kernel():
        a=N.alloc_ub((8,))
        b=N.alloc_ub((8,))
        c=N.alloc_ub((8,))
        for t in N.Pipelined(PARENT_COUNT,stages=1):
            with N.Stage(0,name="load"):
                for j in N.Pipelined(1,stages=1):
                    N.copy(X[t,:],a)
            with N.Stage(0,name="fill"):
                for j in N.Pipelined(1,stages=1):
                    N.fill(b,1.0)
            with N.Stage(0,name="later_load"):
                for j in N.Pipelined(1,stages=1):
                    N.copy(X[t,:],c)
"""


def fifo_proof(count, period):
    source = N.from_source(FIFO.replace("PARENT_COUNT", str(count)), disjoint_external=True).to_ir()
    program = extract_scopes(lower_scopes(source.body.block.first_op))
    access = analyze_scope_access(program)
    _, analyses = _analyses(program)
    return ParentCompletion(program, analyses, access, period)


def test_parent_set_begin_reaches_next_native_queue_without_forwarding_its_effect():
    proof = fifo_proof(1, 1)
    source, destination, later = proof.access.endpoints
    proof.insert(ScopeProtocol("p", "local", (source,), (destination,), 0, 1, (0,)))
    send = proof.events[source[0], "p", "set"]
    uid, site = proof.access.endpoints[later]
    command = proof.completion.commands[uid, site]
    assert proof.graph.path(send, 0, command, 0, source_point="begin") is not None
    assert proof.graph.path(send, 0, command, 0) is None


@pytest.mark.parametrize(
    "period,count,distance,residues", [(1, 1, 1, (0,)), (3, 1, 1, (0,)), (3, 4, 3, (0, 2))]
)
def test_skipped_and_sparse_protocols_preserve_prefixes_for_later_wider_protocols(
    period, count, distance, residues
):
    proof = fifo_proof(count, period)
    source, destination, _ = proof.access.endpoints
    proof.insert(
        ScopeProtocol("narrow", "local", (source,), (destination,), distance, period, residues)
    )
    proof.insert(
        ScopeProtocol("wide", "local", (source,), (destination,), 0, period, tuple(range(period)))
    )
    uid, site = proof.access.endpoints[source]
    source_begin = proof.completion.commands[uid, site]
    uid, site = proof.access.endpoints[destination]
    queue = next(
        point
        for point in proof.completion.projections[uid].endpoints
        if point.task == site.task and point.point == "queue"
    )
    destination_queue = proof.completion.commands[uid, queue]
    sent = proof.events[source[0], "wide", "set"]
    wait = proof.events[destination[0], "wide", "wait"]
    for residue in range(min(period, count)):
        assert (
            proof.graph.path(source_begin, residue, sent, residue, destination_point="begin")
            is not None
        )
        assert proof.graph.path(destination_queue, residue, wait, residue) is not None


def ordered_protocols():
    module = lower_scopes(N.from_source(ORDER, disjoint_external=True).to_ir().body.block.first_op)
    program = extract_scopes(module)
    access = analyze_scope_access(program)
    tasks = {(a, t.uid): t for a, d in access.loop.actors.items() for t in d.program.tasks}
    cube = next(key for key in tasks if key[0] == "aic")
    loads = {
        actor: next(key for key, task in tasks.items() if key[0] == actor and task.pipe == "MTE2")
        for actor in ("aiv0", "aiv1")
    }
    fills = {
        actor: next(key for key, task in tasks.items() if key[0] == actor and task.pipe == "V")
        for actor in loads
    }
    protocols = [
        ScopeProtocol("local_" + actor, "local", (loads[actor],), (fills[actor],), 0, 1, (0,), (1,))
        for actor in loads
    ]
    protocols.extend(
        [
            ScopeProtocol("ready", "cross", (cube,), tuple(fills.values()), 0, 1, (0,), (0,)),
            ScopeProtocol("release", "cross", tuple(fills.values()), (cube,), 1, 1, (0,), (1,)),
        ]
    )
    protocols.extend(
        ScopeProtocol("fence_" + actor, "barrier", (), (loads[actor],), 0, 1, (0,))
        for actor in loads
    )
    _install(program, access, protocols)
    return module


def finite_binary_reuse(checked):
    """Read the actual one-position children, without ParentCompletion.insert."""
    sequences = {actor: [] for actor in checked.program.parents}
    tasks = {
        (actor, task.uid): task
        for actor, data in checked.access.loop.actors.items()
        for task in data.program.tasks
    }
    endpoints, number = {}, 0
    for uid, leaf in checked.program.leaves.items():
        owners = [key for key, (call, _) in checked.access.endpoints.items() if call == uid]
        assert len(owners) == 1 and leaf.stream.count.data == 1
        owner = owners[0]
        for op, _ in stream_operations(leaf.stream):
            if isinstance(op, TaskOp):
                command = Command(number, leaf.actor, tasks[owner], "task")
            elif isinstance(op, ScopeSyncOp):
                protocol = checked.channels[op.channel.data]
                action = op.action.data
                kind = "cross_" + action if protocol.kind == "cross" else action
                command = Command(
                    number,
                    leaf.actor,
                    tasks[owner],
                    kind,
                    protocol.name,
                    protocol.distance if action == "wait" else 0,
                    protocol.distance if action == "set" else 0,
                )
                endpoints[leaf.actor, protocol.name, action] = command
            else:
                continue
            number += 1
            sequences[leaf.actor].append(command)
    ids, reach = finite_hardware_dag(checked.access.loop, sequences, checked.channels, 2)
    return {
        actor: bool(
            reach[ids[endpoints[actor, "local_" + actor, "wait"].index, 0, "begin"]]
            & (1 << ids[endpoints[actor, "local_" + actor, "set"].index, 1, "end"])
        )
        for actor in ("aiv0", "aiv1")
    }


def test_actual_wait_order_cannot_be_replaced_by_channel_declaration_order():
    module = ordered_protocols()
    _, checked = verify_scopes(module, _details=True)
    assert finite_binary_reuse(checked) == {"aiv0": True, "aiv1": True}
    for leaf in checked.program.leaves.values():
        waits = [
            op
            for op in leaf.stream.walk()
            if isinstance(op, ScopeSyncOp) and op.action.data == "wait"
        ]
        if leaf.actor == "aic" or len(waits) != 2:
            continue
        local = next(op for op in waits if op.channel.data.startswith("local_"))
        ready = next(op for op in waits if op.channel.data == "ready")
        local.parent.detach_op(ready)
        local.parent.insert_op_before(ready, local)
    # The finite actual-hardware model loses BOTH old-Wait -> next-Set paths.
    assert finite_binary_reuse(checked) == {"aiv0": False, "aiv1": False}
    with pytest.raises(CompileError, match="synchronization order differs"):
        verify_scopes(module)


def test_missing_real_aiv_quorum_contribution_is_rejected():
    module = ordered_protocols()
    program = extract_scopes(module)
    contribution = next(
        op
        for leaf in program.leaves.values()
        if leaf.actor == "aiv1"
        for op in leaf.stream.walk()
        if isinstance(op, ScopeSyncOp) and op.action.data == "set" and op.channel.data == "release"
    )
    contribution.parent.erase_op(contribution)
    with pytest.raises(
        CompileError, match="missing or changes an actual native Set/Wait placement"
    ):
        verify_scopes(module)


def test_parent_wait_cannot_cross_a_real_child_barrier():
    module = ordered_protocols()
    program = extract_scopes(module)
    leaf, task = next(
        (leaf, task)
        for leaf in program.leaves.values()
        if leaf.actor == "aiv0"
        for task in leaf.stream.walk()
        if isinstance(task, TaskOp) and task.native.data == "fill"
    )
    barrier = LoopSyncOp.create(
        operands=[leaf.stream.body.block.args[0]],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr(""),
            "lower": IntAttr(0),
            "upper_gap": IntAttr(0),
            "period": IntAttr(1),
            "residues": ArrayAttr([IntAttr(0)]),
        },
    )
    task.parent.insert_op_before(barrier, task)
    with pytest.raises(CompileError, match="acquire must be adjacent"):
        verify_scopes(module)
