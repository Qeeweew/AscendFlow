"""Real fixed-child proofs composed on the retained parent pipeline clock."""

from collections import deque
from dataclasses import replace
from random import Random

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.ir import Block, Region

from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.child_completion import ChildPort, project_fixed_child
from xdsl_npu.completion import Position, Site
from xdsl_npu.ir import (
    BufferOp,
    CallOp,
    CaptureOp,
    EndOp,
    FixConfigOp,
    LoopChannelOp,
    OnceOp,
    StreamOp,
    TaskOp,
    dump,
)
from xdsl_npu.loop_graph import Graph
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.loop_sync import _plan_leaf, _verify_leaf, verify_loop
from xdsl_npu.model import CompileError
from xdsl_npu.scope_completion import _transitive_reduction, compose_calls
from xdsl_npu.symbolic_completion import native_family


def checked(module, *, live_in=()):
    if live_in:
        loop = extract_loop(module)
        _plan_leaf(loop, live_in=live_in)
        return module, _verify_leaf(loop, live_in=live_in, _details=True)[1]
    plan = compile(module).plan
    return plan, verify_loop(plan, _details=True)[1]


def site(analysis, uid, point, position=None, **kwargs):
    actor = next(actor for actor, data in analysis[0].actors.items() if uid in data.operations)
    return Site(
        (actor, uid),
        Position(False, 0) if position is None else position,
        point,
        region="fixed/child",
        **kwargs,
    )


def bfs(component_count, arcs, source):
    """Independent, unweighted reachability oracle for small test DAGs."""
    following = [[] for _ in range(component_count)]
    for start, target in arcs:
        following[start].append(target)
    reached, pending = {source}, deque([source])
    while pending:
        for target in following[pending.popleft()]:
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def test_dag_reduction_matches_independent_bfs_and_keeps_only_necessary_arcs():
    cases = [
        (0, ()),
        (1, ()),
        # A diamond, a redundant shortcut, and a disconnected component.
        (7, ((5, 2), (5, 4), (5, 0), (2, 0), (4, 0), (3, 1))),
        # Descendant bitsets cross machine-word boundaries.
        (80, tuple((source, target) for source in range(80) for target in range(source + 1, 80))),
    ]
    random = Random(9173)
    for size in range(2, 11):
        for _ in range(8):
            order = list(range(size))
            random.shuffle(order)
            arcs = [
                (source, target)
                for rank, source in enumerate(order)
                for target in order[rank + 1 :]
                if random.random() < 0.4
            ]
            random.shuffle(arcs)
            cases.append((size, tuple(arcs)))
    for size, arcs in cases:
        reduced = _transitive_reduction(size, arcs)
        assert set(reduced) <= set(arcs)
        assert len(reduced) == len(set(reduced))
        for source in range(size):
            assert bfs(size, reduced, source) == bfs(size, arcs, source)
        for source, target in reduced:
            without_edge = tuple(edge for edge in reduced if edge != (source, target))
            assert target not in bfs(size, without_edge, source)


class Parent:
    """Attach actual CallOps and captured SSA allocations to one parent stream."""

    def __init__(self, count=None, *, core="aiv"):
        self.k = Kernel("parent", core=core, disjoint_external=True)
        bound = self.k.runtime_size("tasks") if count is None else count
        self.k.loop(bound, symbolic=True)
        self.k.task_block.add_op(EndOp())
        self.k.block.add_op(EndOp())
        self.roots, self.calls = {}, {}

    def add(self, uid, plan, projection, *, stage=0, order=0, shared=()):
        kernel = plan.body.block.first_op
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        captures, mapper = [], {}
        for root in kernel.body.block.ops:
            if not isinstance(root, BufferOp):
                continue
            name = root.sym_name.data
            key = name if name in shared else uid + "_" + name
            if key not in self.roots:
                parent = root.clone()
                parent.attributes["sym_name"] = StringAttr(key)
                self.k.block.insert_op_before(parent, self.k.stream)
                self.roots[key] = parent
            parent = self.roots[key]
            assert parent.ref.type == root.ref.type and parent.nbytes == root.nbytes
            capture = CaptureOp.create(
                operands=[parent.ref, zero.result],
                result_types=[root.ref.type],
                attributes={"sym_name": StringAttr(name), "nbytes": root.nbytes},
            )
            captures.append(capture)
            mapper[root.ref] = capture.ref
        child = next(op for op in kernel.body.block.ops if isinstance(op, (StreamOp, OnceOp)))
        child = child.clone(value_mapper=mapper)
        declarations = [op.clone() for op in kernel.body.block.ops if isinstance(op, LoopChannelOp)]
        child.body.block.insert_ops_before(declarations, child.body.block.first_op)
        if any(isinstance(op, FixConfigOp) for op in kernel.body.block.ops) and not any(
            isinstance(op, FixConfigOp) for op in self.k.block.ops
        ):
            self.k.block.insert_op_before(FixConfigOp(), self.k.stream)
        call = CallOp.create(
            operands=[self.k.task_block.args[0]],
            attributes={
                "uid": StringAttr(uid),
                "stage": IntAttr(stage),
                "logical_phase": IntAttr(0),
                "issue_order": IntAttr(order),
            },
            regions=[Region(Block([zero, *captures, child, EndOp()]))],
        )
        self.k.task_block.insert_op_before(call, self.k.task_block.last_op)
        self.calls[uid] = call, projection
        return call


@pytest.fixture(scope="module")
def children():
    load = Kernel("load_child", disjoint_external=True)
    x = load.buffer("X", "GM", 32, external=True, initialized=True)
    a = load.buffer("shared", "UB", 32)
    load.loop(1, symbolic=True)
    load.copy("load", load.view(x, 32), load.view(a, 32))
    load_plan, load_analysis = checked(load.finish())
    load_end = site(load_analysis, "load", "end")
    load_prefix = replace(load_end, point="prefix")
    load_projection = project_fixed_child(load_analysis, (load_end, load_prefix))

    capture = Kernel("capture_child", disjoint_external=True)
    x = capture.buffer("X", "GM", 32, external=True, initialized=True)
    a = capture.buffer("a", "UB", 32)
    capture.loop(1, symbolic=True)
    capture.copy("load", capture.view(x, 32), capture.view(a, 32))
    capture.scalar_load("capture", capture.view(a, 4), stage=1)
    capture_plan, capture_analysis = checked(capture.finish())
    capture_end = site(capture_analysis, "capture", "end")
    capture_projection = project_fixed_child(capture_analysis, (capture_end,))

    consumer = Kernel("consumer_child")
    a = consumer.buffer("shared", "UB", 32)
    b = consumer.buffer("result", "UB", 32)
    consumer.loop(1, symbolic=True)
    consumer.scale("consume", consumer.view(a, 32), consumer.view(b, 32), 2.0)
    consumer_plan, consumer_analysis = checked(consumer.finish(), live_in=("aiv0/shared",))
    consume_begin = site(consumer_analysis, "consume", "begin")
    consume_projection = project_fixed_child(consumer_analysis, (consume_begin,))
    return {
        "load": (load_plan, load_projection, load_end),
        "capture": (capture_plan, capture_projection, capture_end),
        "consume": (consumer_plan, consume_projection, consume_begin),
    }


@pytest.mark.parametrize("stages", [(0, 2, 0), (2, 0, 2)])
def test_skipped_scalar_child_cannot_invent_raw_completion_in_fill_or_drain(children, stages):
    parent = Parent(1)
    for order, (uid, stage) in enumerate(zip(("load", "capture", "consume"), stages)):
        plan, projection, _ = children[uid]
        parent.add(uid, plan, projection, stage=stage, order=order, shared=("shared",))
    scope = compose_calls(parent.calls, period=1, count=1)
    scope.graph.potential()
    projection = children["capture"][1]
    actor = next(iter(scope.call_order))
    assert projection.reaches(
        ChildPort(actor, "MTE2", "retire", "input"), ChildPort(actor, None, "issue", "output")
    )
    query = scope.query("load", children["load"][2], 0, "consume", children["consume"][2], 0)
    assert not scope.graph.covered_many([query])[0]
    assert set(scope.call_order) == {"aiv0"}
    # The skipped call's matching-kind physical state still crosses its slot.
    for key, node in scope.prefixes.items():
        uid, _, _, kind, _ = key
        meta = scope.graph.nodes[node]
        assert native_family(meta, scope.graph.period).monotone
        assert meta.lower == -parent.calls[uid][0].stage.data
        assert kind in {"issue", "queue", "retire", "implicit"}


def test_actual_scalar_child_can_supply_its_proved_controller_gate(children):
    parent = Parent(1)
    for order, uid in enumerate(("load", "capture", "consume")):
        plan, projection, _ = children[uid]
        parent.add(uid, plan, projection, order=order, shared=("shared",))
    scope = compose_calls(parent.calls, period=1, count=1)
    query = scope.query("load", children["load"][2], 0, "consume", children["consume"][2], 0)
    assert scope.graph.covered_many([query]) == [True]
    assert all(
        not native_family(scope.graph.nodes[node], scope.graph.period).monotone
        for node in {*scope.sites.values(), *scope.ports.values()}
    )


@pytest.mark.parametrize("period", [1, 3])
@pytest.mark.parametrize(
    "stages,count", [((0, 0, 0), 1), ((0, 2, 0), 1), ((2, 0, 2), 1), ((0, 1, 0), None)]
)
def test_reduced_real_projection_matches_all_unreduced_endpoint_queries(
    children, monkeypatch, period, stages, count
):
    parent = Parent(count)
    for order, (uid, stage) in enumerate(zip(("load", "capture", "consume"), stages)):
        plan, projection, _ = children[uid]
        parent.add(uid, plan, projection, stage=stage, order=order, shared=("shared",))
    reduced = compose_calls(parent.calls, period=period, count=count)
    with monkeypatch.context() as original:
        original.setattr("xdsl_npu.scope_completion._transitive_reduction", lambda _, arcs: arcs)
        unreduced = compose_calls(parent.calls, period=period, count=count)
    assert reduced.aliases == unreduced.aliases
    assert reduced.sites == unreduced.sites
    assert reduced.prefixes == unreduced.prefixes
    assert len(reduced.graph.nodes) == len(unreduced.graph.nodes)
    assert len(reduced.graph.edges) < len(unreduced.graph.edges)
    locations = [
        (uid, endpoint, generation)
        for uid, (_, projection) in parent.calls.items()
        for endpoint in projection.endpoints
        for generation in range(3 if count is None else count)
    ]
    answers = []
    for scope in (reduced, unreduced):
        scope.graph.potential()
        queries = [scope.query(*source, *target) for source in locations for target in locations]
        answers.append(scope.graph.covered_many(queries))
    assert answers[0] == answers[1]
    assert any(answers[0]) and not all(answers[0])


@pytest.fixture(scope="module")
def cube_child():
    from xdsl_npu import language as N

    plan, analysis = checked(
        N.from_source(
            """
            def cube(X:N.Tensor((8,16,16),N.f16),Y:N.Tensor((16,16),N.f32,access="write")):
                with N.Kernel(kind="cube"):
                    x=N.alloc_l1((16,16))
                    a=N.alloc_l0a((16,16))
                    b=N.alloc_l0b((16,16))
                    c=N.alloc_l0c((16,16))
                    for i in N.Pipelined(8,stages=1):
                        v=N.copy(X[i,:,:],x)
                        aa=N.copy(v,a)
                        bb=N.copy(v,b,transpose=True)
                        z=N.matmul(aa,bb,accum=c,initialize=(i==0))
                        if i==7:
                            N.copy(z,Y)
            """,
            disjoint_external=True,
        )
    )
    actor, data = next(iter(analysis[0].actors.items()))
    fix = next(task.uid for task in data.program.tasks if task.pipe == "FIX")
    begin = Site((actor, fix), Position(False, 7), "begin", region="cube/inner")
    projection = project_fixed_child(analysis, (begin,))
    return plan, projection, begin


@pytest.mark.parametrize("period", [1, 3])
def test_qk_then_pv_uses_stage_difference_and_exact_wrap_distance(cube_child, period):
    plan, projection, begin = cube_child
    parent = Parent(4, core="aic")
    parent.add("qk", plan, projection, stage=0, order=0)
    parent.add("pv", plan, projection, stage=1, order=1)
    scope = compose_calls(dict(reversed(tuple(parent.calls.items()))), period=period, count=4)
    assert scope.call_order == {"aic": ("qk", "pv")}
    for residue in range(period):
        target = residue - 1
        assert (
            scope.prefixes["qk", "aic", None, "issue", residue],
            scope.prefixes["pv", "aic", None, "issue", target % period],
            target // period,
        ) in scope.graph.edges
        target = residue + 2
        assert (
            scope.prefixes["pv", "aic", None, "issue", residue],
            scope.prefixes["qk", "aic", None, "issue", target % period],
            target // period,
        ) in scope.graph.edges
    scope.graph.potential()
    queries = [
        scope.query("qk", begin, 1, "pv", begin, 0),
        scope.query("pv", begin, 0, "qk", begin, 1),
        scope.query("pv", begin, 0, "qk", begin, 2),
    ]
    assert scope.graph.covered_many(queries) == [True, False, True]


def fill_child(count):
    k = Kernel("fill_child")
    a = k.buffer("a", "UB", 64)
    k.loop(count, symbolic=True)
    k.task("fill", "fill", k.view(a, 64), scalar=1.0, stage=2)
    plan, analysis = checked(k.finish())
    head = site(analysis, "fill", "end")
    tail = site(analysis, "fill", "prefix", Position(True, 0))
    return plan, project_fixed_child(analysis, (head, tail)), head, tail


def test_graph_size_does_not_depend_on_parent_or_child_trip_count(monkeypatch):
    children = [fill_child(count) for count in (8, (1 << 40) + 3)]

    def no_expansion(*args, **kwargs):
        pytest.fail("parent completion must use symbolic families")

    monkeypatch.setattr(Graph, "domain", no_expansion)
    sizes = []
    for plan, projection, head, tail in children:
        before = dump(plan)
        for count in (1, (1 << 50) + 7, None):
            parent = Parent(count)
            parent.add("fill", plan, projection, stage=2)
            scope = compose_calls(parent.calls, period=3, count=count)
            scope.graph.potential()
            query = scope.query("fill", head, 0, "fill", tail, 0)
            assert scope.graph.covered_many([query]) == [True]
            stats = scope.statistics()
            assert stats["native_templates"] == 1
            assert scope.graph.domain_states == 0
            sizes.append((stats["node_families"], stats["arc_families"]))
            assert sum(isinstance(op, TaskOp) for op in parent.calls["fill"][0].walk()) == 1
        assert dump(plan) == before
    assert len(set(sizes)) == 1


def test_only_certified_child_alias_sccs_are_collapsed_and_new_parent_cycle_is_rejected():
    plan, projection, head, _ = fill_child(1)
    _, analysis = verify_loop(plan, _details=True)
    begin = site(analysis, "fill", "begin")
    alias = replace(head, position=Position(True, 0))
    projection = project_fixed_child(analysis, (begin, head, alias))
    parent = Parent()
    parent.add("fill", plan, projection)
    scope = compose_calls(parent.calls, period=1, count=None)
    assert scope.at("fill", head, 0) == scope.at("fill", alias, 0)
    assert scope.aliases["fill", head] == scope.aliases["fill", alias]
    scope.graph.potential()
    assert scope.graph.path(*scope.query("fill", begin, 0, "fill", head, 0)) is not None
    assert scope.graph.domains
    # A real new Wait before fill waiting for that fill's own publication.
    scope.edge(scope.at("fill", head, 0), scope.at("fill", begin, 0))
    assert scope.graph.potentials is None and not scope.graph.domains
    with pytest.raises(CompileError, match="completion cycle|non-well-founded"):
        scope.graph.potential()


def test_same_call_local_flag_is_not_assumed_consumed_before_next_outer_generation():
    k = Kernel("ready_child", disjoint_external=True)
    x = k.buffer("X", "GM", 32, external=True, initialized=True)
    a, b = k.buffer("a", "UB", 32), k.buffer("b", "UB", 32)
    k.loop(1, symbolic=True)
    k.copy("load", k.view(x, 32), k.view(a, 32))
    k.scale("scale", k.view(a, 32), k.view(b, 32), 2.0, stage=1)
    plan, analysis = checked(k.finish())
    actor = next(iter(analysis[2]))
    sent = next(command for command in analysis[2][actor] if command.kind == "set")
    wait = next(command for command in analysis[2][actor] if command.kind == "wait")
    published = site(analysis, sent.task.uid, "end", channel=sent.channel, action="set")
    consumed = site(analysis, wait.task.uid, "begin", channel=wait.channel, action="wait")
    native_prefix = site(analysis, "load", "prefix")
    projection = project_fixed_child(analysis, (published, consumed, native_prefix))
    parent = Parent()
    parent.add("ready", plan, projection)
    scope = compose_calls(parent.calls, period=1, count=None)
    assert scope.graph.covered_many(
        [
            scope.query("ready", published, 0, "ready", consumed, 0),
            scope.query("ready", consumed, 0, "ready", published, 1),
            scope.query("ready", published, 0, "ready", native_prefix, 0),
        ]
    ) == [True, False, False]


def test_actual_issue_order_and_lexical_ties_override_dictionary_order(children):
    plan, projection, _ = children["capture"]
    parent = Parent()
    parent.add("first", plan, projection, order=7)
    parent.add("second", plan, projection, order=7)
    parent.add("urgent", plan, projection, order=0)
    scope = compose_calls(dict(reversed(tuple(parent.calls.items()))), period=1, count=None)
    assert scope.call_order == {"aiv0": ("urgent", "first", "second")}


def test_real_insertion_cut_sites_keep_their_direction_and_opaque_identity(children):
    plan = children["load"][0]
    _, analysis = verify_loop(plan, _details=True)
    kinds = ("issue", "queue", "retire", "implicit", "admit_issue", "admit_queue", "begin")
    endpoints = {kind: site(analysis, "load", kind) for kind in kinds}
    projection = project_fixed_child(analysis, tuple(endpoints.values()), insertion_cuts=True)
    parent = Parent()
    parent.add("load", plan, projection)
    scope = compose_calls(parent.calls, period=1, count=None)
    scope.graph.potential()
    assert scope.at("load", endpoints["queue"], 0) != scope.at("load", endpoints["admit_queue"], 0)
    queries = [
        scope.query("load", endpoints[a], 0, "load", endpoints[b], 0)
        for a, b in (("queue", "admit_queue"), ("admit_queue", "queue"), ("admit_queue", "begin"))
    ]
    assert scope.graph.covered_many(queries) == [True, False, True]
    assert all(("load", endpoint, 0) in scope.sites for endpoint in endpoints.values())


def test_zero_trip_child_preserves_state_without_adding_its_scalar_gate(children):
    empty = children["capture"][0].clone()
    child = next(op for op in empty.walk() if isinstance(op, StreamOp))
    child.attributes["count"] = IntAttr(0)
    _, analysis = verify_loop(empty, _details=True)
    projection = project_fixed_child(analysis, ())
    parent = Parent(1)
    for order, uid in enumerate(("load", "empty", "consume")):
        if uid == "empty":
            parent.add(uid, empty, projection, order=order)
        else:
            plan, projected, _ = children[uid]
            parent.add(uid, plan, projected, order=order, shared=("shared",))
    scope = compose_calls(parent.calls, period=1, count=1)
    scope.graph.potential()
    assert not any(uid == "empty" for uid, _, _ in scope.sites)
    query = scope.query("load", children["load"][2], 0, "consume", children["consume"][2], 0)
    assert scope.graph.covered_many([query]) == [False]


def test_different_actors_have_no_implicit_prefix_inheritance(children, cube_child):
    cube_plan, cube_projection, cube_begin = cube_child
    vector_plan, vector_projection, vector_end = children["capture"]
    cube, vector = Parent(core="aic"), Parent()
    cube.add("cube", cube_plan, cube_projection)
    vector.add("vector", vector_plan, vector_projection)
    scope = compose_calls({**cube.calls, **vector.calls}, period=1, count=None)
    assert set(scope.call_order) == {"aic", "aiv0"}
    assert scope.graph.covered_many(
        [scope.query("vector", vector_end, 0, "cube", cube_begin, 0)]
    ) == [False]


@pytest.mark.parametrize("period,count", [(0, None), (True, None), (1, -1), (1, True)])
def test_invalid_parent_clock_contract_is_rejected(children, period, count):
    plan, projection, _ = children["capture"]
    parent = Parent()
    parent.add("capture", plan, projection)
    with pytest.raises(CompileError):
        compose_calls(parent.calls, period=period, count=count)


def test_call_count_and_endpoint_identity_are_read_from_actual_ir(children):
    plan, projection, endpoint = children["capture"]
    parent = Parent()
    op = parent.add("capture", plan, projection)
    child = next(op for op in op.body.block.ops if isinstance(op, StreamOp))
    child.attributes["count"] = IntAttr(2)
    with pytest.raises(CompileError, match="actual fixed-count child"):
        compose_calls(parent.calls, period=1, count=None)
    child.attributes["count"] = IntAttr(1)
    scope = compose_calls(parent.calls, period=1, count=None)
    with pytest.raises(CompileError, match="unknown scoped"):
        scope.query("capture", endpoint, 0, "capture", replace(endpoint, region="elsewhere"), 0)
