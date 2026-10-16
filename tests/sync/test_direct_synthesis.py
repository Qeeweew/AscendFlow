"""Direct protocols preserve native milestones and the earliest read retirement."""

import random

import pytest

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp
from xdsl_npu.loop_graph import channels, completion_graph
from xdsl_npu.loop_planning import DirectCompletion


def _identity(command):
    return command.actor, command.task.uid, command.kind, command.channel


@pytest.mark.parametrize("case", ["vector", "cube", "cv"])
def test_incremental_native_facts_match_materialized_instruction_stream(monkeypatch, case):
    import xdsl_npu.loop_sync as sync
    from examples.gemm.reduction import reduction_gemm
    from tests.fixtures.cv import cv_source, vector_source

    graphs, checked = [], []
    original_init, original_insert = DirectCompletion.__init__, DirectCompletion.insert
    original_materialize = sync._materialize

    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        graphs.append(self)

    def insert(self, protocol, native):
        if not self.selected:
            # There are fixed native insertion cuts, but no prospective sync
            # instructions, events, or alternative protocol effects.
            assert {node.command.kind for node in self.nodes} == {"task"}
        original_insert(self, protocol, native)

    def materialize(loop, protocols):
        graph = graphs[-1]
        assert not any(
            isinstance(op, (LoopSyncOp, LoopChannelOp)) for op in loop.protocol_scope.walk()
        )
        original_materialize(loop, protocols)
        actual, streams, _, _ = completion_graph(loop, channels(loop, virtual=True))
        by_identity = {_identity(c): c for stream in streams.values() for c in stream}
        commands = list({n.command.index: n.command for n in graph.nodes}.values())
        assert {_identity(c) for c in commands} == set(by_identity)

        # Include negative queries, individual notification effects, native
        # prefixes, and carries. Testing only the required dependencies would
        # compare two sets of all-True answers and miss invented ordering.
        rng = random.Random(20261015)
        queries, mapped = [], []
        if case != "cv":
            for source in commands:
                for destination in commands:
                    for distance in range(3):
                        queries.append((source, 0, destination, distance, 0, False))
                        mapped.append(
                            (
                                by_identity[_identity(source)],
                                0,
                                by_identity[_identity(destination)],
                                distance,
                                0,
                                False,
                            )
                        )
        for _ in range(160):
            source, destination = rng.choices(commands, k=2)
            sg, distance = rng.randrange(2), rng.randrange(3)
            prefix = source.kind == "task" and bool(rng.randrange(2))
            queries.append((source, sg, destination, sg + distance, 0, prefix))
            mapped.append(
                (
                    by_identity[_identity(source)],
                    sg,
                    by_identity[_identity(destination)],
                    sg + distance,
                    0,
                    prefix,
                )
            )
        graph.register(queries)
        wanted = actual.covered_many(mapped)
        observed = [graph.covered(q) for q in queries]
        assert not all(wanted) and any(wanted)
        assert observed == wanted, [
            (_identity(q[0]), _identity(q[2]), q[1], q[3], a, b)
            for q, a, b in zip(queries, observed, wanted)
            if a != b
        ]
        checked.append(len(queries))

    monkeypatch.setattr(DirectCompletion, "__init__", capture)
    monkeypatch.setattr(DirectCompletion, "insert", insert)
    monkeypatch.setattr(sync, "_materialize", materialize)
    definition = {
        "vector": lambda: vector_source(N.dynamic("tasks")),
        "cube": lambda: reduction_gemm,
        "cv": lambda: cv_source(N.dynamic("tasks")),
    }[case]()
    artifact = compile(definition)
    assert checked
    operations = set(artifact.plan.walk())
    for op in operations:
        for region in op.regions:
            for block in region.blocks:
                for argument in block.args:
                    assert all(use.operation in operations for use in argument.uses)


def test_release_captures_actual_last_reader_before_unrelated_vector_work():
    artifact = compile(
        N.from_source(
            """
        def early(X:N.Tensor((n,16),N.f32), Z:N.Tensor((n,16),N.f32),
                  Y:N.Tensor((n,16),N.f32,access="write"),
                  U:N.Tensor((n,16),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((2,16),name="x")
                u=N.alloc_ub((2,16),name="u")
                y=N.alloc_ub((2,16),name="y")
                z=N.alloc_ub((2,16),name="z")
                for t in N.Pipelined(n,stages=3):
                    with N.Stage(0,name="load"):
                        N.copy(X[t,:],x[t%2,:])
                        N.copy(Z[t,:],u[t%2,:])
                    with N.Stage(1,name="compute"):
                        N.mul(x[t%2,:],2.0,out=y[t%2,:])
                        N.mul(u[t%2,:],3.0,out=z[t%2,:])
                    with N.Stage(2,name="store"):
                        N.copy(y[t%2,:],Y[t,:])
                        N.copy(z[t%2,:],U[t,:])
    """,
            disjoint_external=True,
        )
    )
    from xdsl_npu.loop_graph import commands
    from xdsl_npu.loop_model import extract_loop

    loop = extract_loop(artifact.plan)
    declared = channels(loop)
    (release,) = [
        p
        for p in declared.values()
        if p.destinations == (("aiv0", "load_0_copy"),) and p.distance == 2
    ]
    assert release.sources == (("aiv0", "compute_2_mul_scalar"),)
    streams, _, native = commands(loop, declared)
    stream = streams["aiv0"]
    (send,) = [c for c in stream if c.channel == release.name and c.kind == "set"]
    assert native["aiv0", "compute_2_mul_scalar"].index < send.index
    assert send.index < native["aiv0", "compute_3_mul_scalar"].index


def test_direct_domains_propagate_new_edges_and_nodes_without_rebuilding():
    from types import SimpleNamespace

    from xdsl_npu.loop_graph import Command

    task = SimpleNamespace(stage=0)
    graph = DirectCompletion(1, None)
    source, target = (Command(i, "aiv0", task, "task") for i in range(2))
    a = graph.node(source, 0, "end")
    b = graph.node(target, 0, "begin")
    query = source, 0, target, 0, 0, False
    graph.register((query,))
    proof = graph.completion()
    guard, flow = next(iter(proof.flows.items()))
    assert not graph.covered(query)
    capture = graph.node(Command(2, "aiv0", task, "set"), 0, "end")
    graph.edge((a, 0), (capture, 0))
    assert not graph.covered(query)
    graph.edge((capture, 0), (b, 0))
    assert graph.covered(query)
    assert proof is graph.completion() and flow is proof.flows[guard]


def test_skipped_earlier_wait_preserves_scalar_dispatch_facts(monkeypatch):
    """g=0 skips a carried credit but still executes the later ready wait."""
    from types import SimpleNamespace

    import xdsl_npu.loop_graph as hardware
    from xdsl_npu.loop_graph import Command
    from xdsl_npu.loop_planning import Protocol

    actor = "aiv0"
    work = [
        Command(
            i,
            actor,
            SimpleNamespace(
                uid=uid, pipe=pipe, stage=0, implicit_completion=(pipe == "S"), last=False
            ),
            "task",
        )
        for i, (uid, pipe) in enumerate(
            (("load", "MTE2"), ("scalar", "S"), ("reader", "V"), ("consumer", "S"))
        )
    ]
    native = {(actor, c.task.uid): c for c in work}
    stream, endpoints = list(work), {}
    monkeypatch.setattr(
        hardware, "commands", lambda *args, **kwargs: ({actor: stream}, endpoints, native)
    )
    loop = SimpleNamespace(protocol_period=1, constant_count=None)
    graph = DirectCompletion(1, None)
    completion_graph(loop, {}, _planning=graph)
    credit = Protocol("credit", "local", ((actor, "reader"),), ((actor, "consumer"),), 2, 1, (0,))
    ready = Protocol("ready", "local", ((actor, "load"),), ((actor, "consumer"),), 0, 1, (0,))
    graph.insert(credit, native)
    graph.insert(ready, native)
    endpoints.update(graph.endpoints)
    stream[:] = [
        work[0],
        endpoints[actor, "ready", "set"],
        work[1],
        work[2],
        endpoints[actor, "credit", "set"],
        endpoints[actor, "credit", "wait"],
        endpoints[actor, "ready", "wait"],
        work[3],
    ]
    actual, *_ = completion_graph(loop, {p.name: p for p in (credit, ready)})
    query = work[1], 0, endpoints[actor, "ready", "wait"], 0
    assert actual.covered_many([query]) == [True]
    graph.register([query])
    assert graph.covered(query)
