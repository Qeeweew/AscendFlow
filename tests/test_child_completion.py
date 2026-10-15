"""Fixed child projection keeps native completion and actor identity intact."""

from collections import defaultdict
from dataclasses import replace

import pytest

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.child_completion import ChildPort, project_fixed_child
from xdsl_npu.completion import Position, Site
from xdsl_npu.ir import TaskOp, dump
from xdsl_npu.loop_graph import Graph
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.mixed import compose
from xdsl_npu.model import CompileError
from xdsl_npu.sequence_sync import verify_sequence


def checked(module):
    artifact = compile(module)
    return artifact.plan, verify_loop(artifact.plan, _details=True)[1]


@pytest.fixture(scope="module")
def cube():
    return checked(
        N.from_source(
            """
            def child(X:N.Tensor((8,16,16),N.f16),
                      Y:N.Tensor((16,16),N.f32,access="write")):
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


def task_site(analysis, native, generation, point):
    tasks = [
        (actor, task.uid)
        for actor, data in analysis[0].actors.items()
        for task in data.program.tasks
        if task.native == native or task.pipe == native
    ]
    assert len(tasks) == 1
    return Site(tasks[0], Position(False, generation), point, region="outer/qk/inner")


def parent_path(projection, source, target, bridge=None):
    """Compose two invocations by matching prefix state, using a small BFS."""
    edges = defaultdict(set)
    for call in (0, 1):
        for left, right in projection.relations:
            edges[call, left].add((call, right))
    for output in projection.outputs:
        following = replace(output, side="input")
        assert following in projection.inputs
        edges[0, output].add((1, following))
    if bridge is not None:
        edges[0, bridge].add((1, target))
    seen, pending = {(0, source)}, [(0, source)]
    while pending:
        for point in edges[pending.pop()]:
            if point not in seen:
                seen.add(point)
                pending.append(point)
    return (1, target) in seen


def test_terminal_fix_read_needs_its_actual_release_not_mad_or_whole_child_retirement(cube):
    plan, analysis = cube
    before = dump(plan)
    fix = task_site(analysis, "FIX", 7, "end")
    release = replace(fix, point="prefix")
    clear = task_site(analysis, "mad_first", 0, "begin")
    wrong = task_site(analysis, "mad_first", 7, "prefix")
    projection = project_fixed_child(analysis, (fix, release, clear, wrong))
    assert projection.count == 8 and projection.native_templates == 5
    assert projection.endpoints == (fix, release, clear, wrong)
    assert projection.reaches(fix, release)
    assert not parent_path(projection, fix, clear)
    assert parent_path(projection, fix, clear, bridge=release)
    assert not parent_path(projection, fix, clear, bridge=wrong)
    assert dump(plan) == before
    assert sum(isinstance(op, TaskOp) for op in plan.walk()) == 5


@pytest.fixture(scope="module")
def one_copy():
    k = Kernel("copy_then_vector", disjoint_external=True)
    x = k.buffer("X", "GM", 64, external=True, initialized=True)
    a = k.buffer("a", "UB", 64)
    b = k.buffer("b", "UB", 64)
    k.loop(1, symbolic=True)
    ar, br = k.view(a, 64), k.view(b, 64)
    k.copy("load", k.view(x, 64), ar)
    k.scale("scale", ar, br, 2.0, stage=1)
    return checked(k.finish())


def test_ports_distinguish_queue_retirement_implicit_and_notification_effect(one_copy):
    _, analysis = one_copy
    actor = next(iter(analysis[0].actors))
    begin = Site((actor, "load"), Position(False, 0), "begin", region="child")
    end, prefix = replace(begin, point="end"), replace(begin, point="prefix")
    notification = next(command for command in analysis[2][actor] if command.kind == "set")
    notified = Site(
        (actor, notification.task.uid),
        Position(False, 0),
        "end",
        notification.channel,
        "set",
        region="child",
    )
    scale = Site((actor, "scale"), Position(False, 0), "begin", region="child")
    projection = project_fixed_child(analysis, (begin, end, prefix, notified, scale))

    def port(kind, side, pipe="MTE2"):
        return ChildPort(actor, pipe, kind, side)

    assert projection.reaches(port("queue", "input"), begin)
    assert projection.reaches(port("implicit", "input"), begin)
    assert projection.reaches(ChildPort(actor, None, "issue", "input"), begin)
    assert not projection.reaches(port("retire", "input"), begin)
    assert projection.reaches(port("retire", "input"), prefix)
    assert projection.reaches(begin, port("queue", "output"))
    assert not projection.reaches(end, port("queue", "output"))
    assert projection.reaches(end, port("retire", "output"))
    assert not projection.reaches(end, port("implicit", "output"))
    assert projection.reaches(notified, scale)
    assert not projection.reaches(notified, prefix)
    assert not projection.reaches(notified, port("retire", "output"))


def fill_child(count):
    k = Kernel("fill_child")
    a = k.buffer("a", "UB", 64)
    bound = k.runtime_size("count") if count is None else count
    k.loop(bound, symbolic=True)
    k.task("same", "fill", k.view(a, 64), scalar=1.0, stage=2)
    return k.finish()


def test_projection_does_not_expand_count_or_mutate_the_native_graph(monkeypatch):
    children = [checked(fill_child(count)) for count in (8, (1 << 40) + 3)]

    def no_instance_graph(*args, **kwargs):
        pytest.fail("fixed projection must not materialize native instances")

    monkeypatch.setattr(Graph, "domain", no_instance_graph)
    sizes = []
    for plan, analysis in children:
        before = dump(plan)
        nodes, edges = len(analysis[3].nodes), tuple(analysis[3].edges)
        actor = next(iter(analysis[0].actors))
        head = Site((actor, "same"), Position(False, 0), "end", region="child")
        tail = replace(head, position=Position(True, 0), point="prefix")
        projection = project_fixed_child(analysis, (head, tail))
        stats = projection.statistics()
        assert stats["native_templates"] == 1
        assert stats["native_nodes"] == nodes
        assert stats["native_edges"] == len(edges)
        assert stats["projection_nodes"] == nodes + len(projection.inputs) * 2
        assert projection.reaches(head, tail)
        assert len(analysis[3].nodes) == nodes and tuple(analysis[3].edges) == edges
        assert dump(plan) == before
        sizes.append((projection.node_count, projection.edge_count, stats["relations"]))
    assert sizes[0] == sizes[1]


def test_zero_trip_child_has_only_matching_prefix_identity_transfers():
    _, analysis = checked(fill_child(0))
    projection = project_fixed_child(analysis, ())
    assert projection.count == 0
    assert projection.endpoints == ()
    assert projection.relations == frozenset(
        (port, replace(port, side="output")) for port in projection.inputs
    )
    actor = next(iter(analysis[0].actors))
    with pytest.raises(CompileError, match="inactive"):
        project_fixed_child(analysis, (Site((actor, "same"), Position(False, 0), "end"),))


def test_runtime_child_count_is_rejected():
    _, analysis = checked(fill_child(None))
    with pytest.raises(CompileError, match="actual constant child count"):
        project_fixed_child(analysis, ())


def test_relation_budget_precedes_proof_graph_materialization(one_copy, monkeypatch):
    import xdsl_npu.child_completion as child

    monkeypatch.setattr(child, "MAX_PROJECTION_RELATIONS", 1)
    monkeypatch.setattr(
        child, "SymbolicCompletion", lambda: pytest.fail("materialized before relation budget")
    )
    with pytest.raises(CompileError, match="endpoint relation budget"):
        child.project_fixed_child(one_copy[1], ())


def test_sequence_leaf_keeps_its_parent_invocation_metadata():
    k = Kernel("two_children")
    outputs = k.runtime_size("outputs")
    a, b = k.buffer("a", "UB", 64), k.buffer("b", "UB", 64)
    k.invocations(outputs)
    for name, root in (("first", a), ("second", b)):
        k.region(name, 8)
        k.task("fill_" + name, "fill", k.view(root, 64), scalar=1.0)
    plan = compile(k.finish()).plan
    _, sequence = verify_sequence(plan, _details=True)
    for name, analysis in sequence.analyses.items():
        loop = analysis[0]
        assert loop.invocation is not None and loop.region is not None
        actor = next(iter(loop.actors))
        site = Site((actor, "fill_" + name), Position(False, 0), "end", region=name)
        projection = project_fixed_child(analysis, (site,))
        assert projection.count == 8 and projection.endpoints == (site,)


def test_complete_invocation_analysis_cannot_be_mistaken_for_one_child():
    k = Kernel("single_child")
    outputs = k.runtime_size("outputs")
    a = k.buffer("a", "UB", 64)
    k.invocations(outputs)
    k.loop(8, symbolic=True)
    k.task("fill", "fill", k.view(a, 64), scalar=1.0)
    k.scalar_load("capture", k.view(a, 4), stage=1)
    _, analysis = checked(k.finish())
    assert analysis[0].invocation is not None and analysis[0].region is None
    with pytest.raises(CompileError, match="one verified leaf invocation"):
        project_fixed_child(analysis, ())


def test_repeated_bare_fill_still_requires_actual_completion_across_calls():
    # The pre-projection planner also rejects this: its inner V barrier starts
    # at generation 1, and no reader supplies retirement before the next fill(0).
    k = Kernel("unreleased_child")
    outputs = k.runtime_size("outputs")
    a = k.buffer("a", "UB", 64)
    k.invocations(outputs)
    k.loop(8, symbolic=True)
    k.task("fill", "fill", k.view(a, 64), scalar=1.0)
    with pytest.raises(CompileError, match="missing invocation WAW"):
        checked(k.finish())


@pytest.mark.parametrize(
    "change",
    [
        {"task": ("missing_actor", "missing")},
        {"task": ("aic", "missing_task")},
        {"position": Position(False, 8)},
        {"position": Position(True, 8)},
        {"position": Position(False, -1)},
        {"position": Position(False, 0)},  # FIX only executes at inner 7.
        {"point": "child_done"},
        {"channel": "made_up"},
        {"action": "set"},
    ],
)
def test_inactive_or_invalid_native_sites_are_not_projected(cube, change):
    _, analysis = cube
    fix = task_site(analysis, "FIX", 7, "end")
    with pytest.raises(CompileError, match="endpoint"):
        project_fixed_child(analysis, (replace(fix, **change),))


@pytest.mark.parametrize("change", [{"region": "other/child"}, {"call": 1}])
def test_different_scope_or_call_cannot_alias_one_child(cube, change):
    _, analysis = cube
    fix = task_site(analysis, "FIX", 7, "end")
    with pytest.raises(CompileError, match="scopes or calls"):
        project_fixed_child(analysis, (fix, replace(fix, **change)))


def test_different_inner_position_names_of_one_milestone_keep_their_relations(cube):
    _, analysis = cube
    head = task_site(analysis, "FIX", 7, "end")
    tail = replace(head, position=Position(True, 0))
    projection = project_fixed_child(analysis, (head, tail))
    assert projection.endpoints == (head, tail)
    assert projection.reaches(head, tail) and projection.reaches(tail, head)
    assert all(source != target for source, target in projection.relations)


def test_mixed_actors_with_identical_uids_have_separate_endpoints_and_ports():
    k = Kernel("cube", core="aic")
    x = k.buffer("X", "GM", 512, "f16", layout="NZ", external=True, initialized=True)
    a = k.buffer("a", "L1", 512, "f16", layout="NZ")
    k.loop(8, symbolic=True)
    k.copy("same", k.view(x, 512), k.view(a, 512))
    vectors = []
    for _ in range(2):
        v = Kernel("vector")
        a = v.buffer("a", "UB", 64)
        v.loop(8, symbolic=True)
        v.task("same", "fill", v.view(a, 64), scalar=1.0)
        vectors.append(v.finish())
    _, analysis = checked(compose("independent", k.finish(), *vectors, disjoint_external=True))
    actors = ("aic", "aiv0", "aiv1")
    endpoints = tuple(
        Site((actor, "same"), Position(False, 0), point, region="outer/shared_child")
        for actor in actors
        for point in ("begin", "end", "prefix")
    )
    projection = project_fixed_child(analysis, endpoints)
    assert {port.actor for port in projection.inputs} == set(actors)
    for source_actor in actors:
        begin = next(
            site for site in endpoints if site.task[0] == source_actor and site.point == "begin"
        )
        end = replace(begin, point="end")
        assert projection.reaches(begin, end)
        for destination_actor in actors:
            destination = replace(begin, task=(destination_actor, "same"))
            if source_actor != destination_actor:
                assert not projection.reaches(end, destination)
                issue = ChildPort(source_actor, None, "issue", "input")
                assert not projection.reaches(issue, destination)
    assert all(
        source.task[0] == target.actor
        for source, target in projection.relations
        if isinstance(source, Site) and isinstance(target, ChildPort)
    )
