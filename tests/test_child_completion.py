"""Fixed child projection keeps native completion and actor identity intact."""

from collections import defaultdict
from dataclasses import replace

import pytest

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.child_completion import ChildPort, project_fixed_child
from xdsl_npu.completion import CountDomain, Position, Site, covers
from xdsl_npu.completion_flow import ALL
from xdsl_npu.ir import LoopSyncOp, TaskOp, dump
from xdsl_npu.loop_graph import Command, Graph
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.mixed import compose
from xdsl_npu.model import CompileError
from xdsl_npu.native_order import instruction_points
from xdsl_npu.sequence_sync import verify_sequence
from xdsl_npu.symbolic_completion import Family, SymbolicCompletion


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


def _relation_closure(relations, sources, targets):
    following = defaultdict(set)
    for source, target in relations:
        following[source].add(target)
    result = set()
    for source in sources:
        seen, pending = {source}, [source]
        while pending:
            for target in following[pending.pop()]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        result.update((source, target) for target in targets if source != target and target in seen)
    return frozenset(result)


def _projection_relations(projection):
    return frozenset(
        (source, target)
        for source in (*projection.inputs, *projection.endpoints)
        for target in (*projection.endpoints, *projection.outputs)
        if projection.reaches(source, target)
    )


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


@pytest.mark.parametrize("insertion_cuts", [False, True])
def test_projection_does_not_expand_count_or_mutate_the_native_graph(monkeypatch, insertion_cuts):
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
        projection = project_fixed_child(analysis, (head, tail), insertion_cuts=insertion_cuts)
        stats = projection.statistics()
        assert stats["native_templates"] == 1
        if insertion_cuts:
            commands = sum(len(sequence) for sequence in analysis[2].values())
            assert stats["native_nodes"] == nodes + analysis[3].period * (
                2 * commands + stats["native_templates"]
            )
            assert stats["native_edges"] > len(edges)
        else:
            assert stats["native_nodes"] == nodes
            assert stats["native_edges"] == len(edges)
        assert stats["projection_nodes"] == stats["native_nodes"] + len(projection.inputs) * 2
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
    assert _projection_relations(projection) == frozenset(
        (port, replace(port, side="output")) for port in projection.inputs
    )
    actor = next(iter(analysis[0].actors))
    with pytest.raises(CompileError, match="inactive"):
        project_fixed_child(analysis, (Site((actor, "same"), Position(False, 0), "end"),))


def test_runtime_child_count_is_rejected():
    _, analysis = checked(fill_child(None))
    with pytest.raises(CompileError, match="actual constant child count"):
        project_fixed_child(analysis, ())


@pytest.mark.parametrize("insertion_cuts", [False, True])
def test_relation_budget_precedes_proof_graph_materialization(
    one_copy, monkeypatch, insertion_cuts
):
    import xdsl_npu.child_completion as child
    import xdsl_npu.loop_planning as planning

    monkeypatch.setattr(child, "MAX_PROJECTION_RELATIONS", 1)
    monkeypatch.setattr(
        child, "SymbolicCompletion", lambda: pytest.fail("materialized before relation budget")
    )
    monkeypatch.setattr(
        planning,
        "DirectCompletion",
        lambda *args: pytest.fail("rebuilt cuts before relation budget"),
    )
    with pytest.raises(CompileError, match="endpoint relation budget"):
        child.project_fixed_child(one_copy[1], (), insertion_cuts=insertion_cuts)


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


def command_site(command, generation, point):
    return Site(
        (command.actor, command.task.uid),
        Position(False, generation),
        point,
        command.channel,
        "" if command.kind == "task" else command.kind,
        region="outer/child",
    )


def test_admission_cuts_are_forward_only_and_retain_existing_wait_semantics(one_copy):
    plan, analysis = one_copy
    before = dump(plan)
    actor = next(iter(analysis[0].actors))
    wait = next(command for command in analysis[2][actor] if command.kind == "wait")
    task = next(
        command
        for command in analysis[2][actor]
        if command.task.uid == "scale" and command.kind == "task"
    )
    wait_before = command_site(wait, 0, "queue")
    wait_admit = replace(wait_before, point="admit_queue")
    wait_issue = replace(wait_before, point="issue")
    wait_admit_issue = replace(wait_before, point="admit_issue")
    wait_begin = replace(wait_before, point="begin")
    task_queue = command_site(task, 0, "queue")
    task_issue = replace(task_queue, point="issue")
    task_begin = replace(task_queue, point="begin")
    endpoints = (
        wait_before,
        wait_admit,
        wait_issue,
        wait_admit_issue,
        wait_begin,
        task_queue,
        task_issue,
        task_begin,
    )
    projection = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    assert projection.reaches(wait_before, wait_admit)
    assert not projection.reaches(wait_admit, wait_before)
    assert projection.reaches(wait_issue, wait_admit_issue)
    assert not projection.reaches(wait_admit_issue, wait_issue)
    assert projection.reaches(wait_admit, wait_begin)
    assert projection.reaches(wait_admit, task_queue)
    assert projection.reaches(wait_admit, task_begin)
    assert projection.reaches(wait_admit, ChildPort(actor, "V", "queue", "output"))
    assert projection.reaches(wait_admit_issue, task_issue)
    assert projection.reaches(wait_admit_issue, ChildPort(actor, None, "issue", "output"))
    # A real V Wait gates its consumer queue, not Scalar dispatch. An outer
    # Scalar acquisition belongs to the separate admission-issue cut.
    assert projection.reaches(wait_begin, task_begin)
    assert not projection.reaches(wait_begin, task_issue)
    assert not projection.reaches(wait_admit, task_issue)
    assert dump(plan) == before


def test_final_command_admission_is_preserved_at_the_exit(one_copy):
    _, analysis = one_copy
    actor = next(iter(analysis[0].actors))
    last = analysis[2][actor][-1]
    before = command_site(last, 0, "issue")
    admit = replace(before, point="admit_issue")
    projection = project_fixed_child(analysis, (before, admit), insertion_cuts=True)
    assert projection.reaches(admit, ChildPort(actor, None, "issue", "output"))
    assert not projection.reaches(admit, before)


def test_native_fifo_exit_forwards_inserted_set_begin_to_the_actual_next_command():
    k = Kernel("fifo_exit", disjoint_external=True)
    x = k.buffer("X", "GM", 64, external=True, initialized=True)
    a, b = (k.buffer(name, "UB", 64) for name in ("a", "b"))
    k.loop(1, symbolic=True)
    k.copy("first", k.view(x, 64), k.view(a, 64))
    k.copy("later", k.view(x, 64), k.view(b, 64))
    _, analysis = checked(k.finish())
    actor = next(iter(analysis[0].actors))
    first, later = [command for command in analysis[2][actor] if command.kind == "task"]
    before = {
        point: command_site(first, 0, point)
        for point in ("admit_issue", "begin", "end", "prefix", "after_queue")
    }
    target = command_site(later, 0, "begin")
    projection = project_fixed_child(analysis, (*before.values(), target), insertion_cuts=True)
    after = before["after_queue"]
    assert projection.reaches(before["begin"], after)
    assert projection.reaches(after, target)
    assert not projection.reaches(after, before["begin"])
    assert not projection.reaches(before["end"], after)
    edges = defaultdict(set)
    for source, destination in projection.relations:
        edges[source].add(destination)
    actual = instruction_points(
        Command(999, actor, first.task, "set"),
        {"issue": before["admit_issue"], "queue": before["begin"], "retire": before["prefix"]},
        lambda point: ("inserted_set", point),
        lambda source, destination: edges[source].add(destination),
    )
    edges[actual["begin"]].add(after)

    def reaches(source, destination):
        seen, pending = {source}, [source]
        while pending:
            for point in edges[pending.pop()]:
                if point not in seen:
                    seen.add(point)
                    pending.append(point)
        return destination in seen

    assert reaches(actual["begin"], target)
    assert not reaches(actual["begin"], before["begin"])
    assert not reaches(actual["end"], target)
    assert not reaches(actual["end"], ChildPort(actor, "MTE2", "queue", "output"))


def test_fifo_exit_reaches_real_child_sync_and_crosses_skipped_tail_to_next_call(one_copy):
    _, analysis = one_copy
    actor = next(iter(analysis[0].actors))
    source = next(
        command
        for command in analysis[2][actor]
        if command.kind == "task" and command.task.uid == "load"
    )
    following = next(command for command in analysis[2][actor] if command.kind == "set")
    after = command_site(source, 0, "after_queue")
    begin = command_site(source, 0, "begin")
    end = command_site(source, 0, "end")
    sent = command_site(following, 0, "begin")
    projection = project_fixed_child(analysis, (after, begin, end, sent), insertion_cuts=True)
    output = ChildPort(actor, "MTE2", "queue", "output")
    assert projection.reaches(after, sent)
    assert projection.reaches(after, output)
    assert not projection.reaches(end, output)
    assert parent_path(projection, after, begin)
    assert not parent_path(projection, end, begin)


@pytest.mark.parametrize("count", [1, 8, (1 << 40) + 3])
def test_final_active_fifo_exit_crosses_to_the_next_call_without_clock_expansion(count):
    _, analysis = checked(fill_child(count))
    actor = next(iter(analysis[0].actors))
    source = next(command for command in analysis[2][actor] if command.kind == "task")
    tail = command_site(source, count - 1, "after_queue")
    head = command_site(source, 0, "begin")
    end = command_site(source, count - 1, "end")
    projection = project_fixed_child(analysis, (tail, head, end), insertion_cuts=True)
    assert projection.native_templates == 1
    assert projection.reaches(tail, ChildPort(actor, "V", "queue", "output"))
    assert not projection.reaches(end, ChildPort(actor, "V", "queue", "output"))
    assert parent_path(projection, tail, head)
    assert not parent_path(projection, end, head)


def test_terminal_native_fifo_exit_reaches_the_next_call(cube):
    _, analysis = cube
    after = task_site(analysis, "FIX", 7, "after_queue")
    begin = replace(after, point="begin")
    end = replace(after, point="end")
    projection = project_fixed_child(analysis, (after, begin, end), insertion_cuts=True)
    assert projection.reaches(after, ChildPort(after.task[0], "FIX", "queue", "output"))
    assert parent_path(projection, after, begin)
    assert not parent_path(projection, end, begin)


def test_fifo_exit_requires_opt_in_and_an_active_native_task(one_copy):
    _, analysis = one_copy
    actor = next(iter(analysis[0].actors))
    source = next(command for command in analysis[2][actor] if command.kind == "task")
    wait = next(command for command in analysis[2][actor] if command.kind == "wait")
    with pytest.raises(CompileError, match="milestone"):
        project_fixed_child(analysis, (command_site(source, 0, "after_queue"),))
    with pytest.raises(CompileError, match="native task"):
        project_fixed_child(analysis, (command_site(wait, 0, "after_queue"),), insertion_cuts=True)
    with pytest.raises(CompileError, match="inactive"):
        project_fixed_child(
            analysis, (command_site(source, 1, "after_queue"),), insertion_cuts=True
        )


def test_multiple_same_pipe_commands_keep_their_admission_order():
    k = Kernel("multiple_commands")
    buffers = [k.buffer(name, "UB", 64) for name in ("first", "second", "third")]
    k.loop(3, symbolic=True)
    for rank, root in enumerate(buffers):
        k.task("fill_" + str(rank), "fill", k.view(root, 64), scalar=float(rank))
    _, analysis = checked(k.finish())
    actor = next(iter(analysis[0].actors))
    commands = [c for c in analysis[2][actor] if c.kind == "task"]
    endpoints = tuple(
        command_site(command, 1, point)
        for command in commands
        for point in ("queue", "admit_queue", "issue", "admit_issue", "begin", "end", "prefix")
    )
    projection = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    for rank, command in enumerate(commands):
        for point in ("queue", "issue"):
            before = command_site(command, 1, point)
            admit = replace(before, point="admit_" + point)
            assert projection.reaches(before, admit)
            assert not projection.reaches(admit, before)
            for later in commands[rank + 1 :]:
                assert projection.reaches(admit, command_site(later, 1, point))
            for earlier in commands[:rank]:
                assert not projection.reaches(admit, command_site(earlier, 1, point))


@pytest.fixture(scope="module")
def mixed_child():
    from tests.fixtures.cv import cv_source

    return checked(cv_source(2))


@pytest.fixture(scope="module")
def barrier_child():
    return checked(fill_child(3))


@pytest.mark.parametrize("fixture", ["one_copy", "cube", "barrier_child", "mixed_child"])
def test_cut_projection_preserves_all_real_native_and_sync_relations(request, fixture):
    _, analysis = request.getfixturevalue(fixture)
    count = analysis[0].constant_count
    endpoints = []
    kinds = set()
    for sequence in analysis[2].values():
        for command in sequence:
            g = count - 1 - command.upper
            if g < command.lower or g % command.period not in command.residues:
                continue
            if command.last_gap is not None and g != count - 1 - command.last_gap:
                continue
            kinds.add(command.kind)
            for point in ("begin", "end", "prefix") if command.kind == "task" else ("begin", "end"):
                endpoints.append(command_site(command, g, point))
    ordinary = project_fixed_child(analysis, endpoints)
    cuts = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    assert _projection_relations(cuts) == _projection_relations(ordinary)
    assert cuts.native_templates == ordinary.native_templates
    assert cuts.native_node_count > ordinary.native_node_count
    if fixture == "one_copy":
        assert {"task", "set", "wait"} <= kinds
    elif fixture == "barrier_child":
        assert "barrier" in kinds
    elif fixture == "mixed_child":
        assert {"cross_set", "cross_wait"} <= kinds


def test_zero_trip_cut_projection_is_only_prefix_identity():
    _, analysis = checked(fill_child(0))
    projection = project_fixed_child(analysis, (), insertion_cuts=True)
    assert _projection_relations(projection) == frozenset(
        (port, replace(port, side="output")) for port in projection.inputs
    )
    actor = next(iter(analysis[0].actors))
    for point in ("begin", "end", "prefix", "issue", "queue", "admit_issue", "admit_queue"):
        with pytest.raises(CompileError, match="inactive"):
            project_fixed_child(
                analysis, (Site((actor, "same"), Position(False, 0), point),), insertion_cuts=True
            )


def test_cut_points_are_opt_in_and_syncs_do_not_gain_native_retirement_prefixes(one_copy):
    _, analysis = one_copy
    actor = next(iter(analysis[0].actors))
    wait = next(command for command in analysis[2][actor] if command.kind == "wait")
    for point in ("issue", "queue", "retire", "implicit", "admit_issue", "admit_queue"):
        site = command_site(wait, 0, point)
        with pytest.raises(CompileError, match="milestone"):
            project_fixed_child(analysis, (site,))
        assert project_fixed_child(analysis, (site,), insertion_cuts=True).endpoints == (site,)
    for point in ("prefix", "admit_retire", "admit_implicit"):
        with pytest.raises(CompileError, match="milestone"):
            project_fixed_child(analysis, (command_site(wait, 0, point),), insertion_cuts=True)


def test_ambiguous_barrier_command_cuts_are_rejected(barrier_child):
    plan = barrier_child[0].clone()
    barrier = next(
        op for op in plan.walk() if isinstance(op, LoopSyncOp) and op.action.data == "barrier"
    )
    barrier.parent.insert_op_before(barrier.clone(), barrier)
    _, analysis = verify_loop(plan, _details=True)
    actor = next(iter(analysis[0].actors))
    command = next(c for c in analysis[2][actor] if c.kind == "barrier")
    with pytest.raises(CompileError, match="ambiguous native commands"):
        project_fixed_child(
            analysis, (command_site(command, 1, "admit_queue"),), insertion_cuts=True
        )
    native = next(c for c in analysis[2][actor] if c.kind == "task")
    endpoint = command_site(native, 1, "prefix")
    assert project_fixed_child(analysis, (endpoint,), insertion_cuts=True).endpoints == (endpoint,)


def _literal_projection_relations(flow, sources, targets, bindings):
    """Finite native-instance BFS, independent of stored completion facts."""
    instances = {
        (node, generation)
        for node, family in enumerate(flow.families)
        for generation in range(family.lower, family.horizon[1] + 1)
        if generation % family.period == family.residue
        and (family.last is None or generation == family.last[1])
    }
    following = defaultdict(set)
    for node, generation in instances:
        family = flow.families[node]
        if family.monotone and (node, generation + family.period) in instances:
            following[node, generation].add((node, generation + family.period))
        for target, shift, source_point, target_point, guard in flow.edges[node]:
            assert guard == ALL
            if source_point is None:
                destination = target, generation + shift
            elif generation == source_point[1]:
                assert source_point[0] == target_point[0] == 0
                destination = target, target_point[1]
            else:
                continue
            if destination in instances:
                following[node, generation].add(destination)
    result = set()
    for source in sources:
        node, point = bindings[source]
        assert point[0] == 0
        seen, pending = {(node, point[1])}, [(node, point[1])]
        while pending:
            for destination in following[pending.pop()]:
                if destination not in seen:
                    seen.add(destination)
                    pending.append(destination)
        for target in targets:
            node, point = bindings[target]
            if source != target and (node, point[1]) in seen:
                result.add((source, target))
    return frozenset(result)


@pytest.mark.parametrize("fixture", ("one_copy", "cube", "barrier_child", "mixed_child"))
@pytest.mark.parametrize("insertion_cuts", (False, True))
def test_complete_relation_set_matches_literal_paths_and_scans_each_target_binding_once(
    request, fixture, insertion_cuts, monkeypatch
):
    import xdsl_npu.child_completion as child

    _, analysis = request.getfixturevalue(fixture)
    count = analysis[0].constant_count
    endpoints = []
    for sequence in analysis[2].values():
        for command in sequence:
            generation = count - 1 - command.upper
            if generation < command.lower or generation % command.period not in command.residues:
                continue
            if command.last_gap is not None and generation != count - 1 - command.last_gap:
                continue
            points = ("begin", "end", "prefix") if command.kind == "task" else ("begin", "end")
            if insertion_cuts:
                points += ("issue", "queue", "admit_issue", "admit_queue")
                if command.kind == "task":
                    points += ("after_queue",)
            for point in points:
                site = command_site(command, generation, point)
                endpoints.append(site)
                if point == "begin":
                    endpoints.append(replace(site, position=Position(True, count - 1 - generation)))

    class TargetReads(dict):
        reads = 0
        records = 0

        def get(self, key, default=None):
            facts = super().get(key, default)
            self.reads += 1
            self.records += len(facts)
            return facts

    original = child._fixed_relations
    measured = []

    def checked_relations(flow, sources, targets, bindings):
        facts = flow.facts
        flow.facts = counted = TargetReads(facts)
        try:
            result = original(flow, sources, targets, bindings)
        finally:
            flow.facts = facts
        unique = {bindings[target] for target in targets}
        assert counted.reads == len(unique)
        assert counted.records == sum(len(facts.get(node, {})) for node, _ in unique)
        # Origins are shared only for identical actual bindings. Retain every
        # external Site alias in both independent complete-reach oracles.
        all_sources = tuple(
            dict.fromkeys((*sources, *(target for target in targets if isinstance(target, Site))))
        )
        origin = {bindings[source]: 1 << bit for bit, source in enumerate(sources)}
        expected = frozenset(
            (source, target)
            for source in all_sources
            for target in targets
            if source != target
            and covers(ALL, flow.coverage(*bindings[target], origin[bindings[source]], ALL))
        )
        assert result <= expected
        assert _relation_closure(result, all_sources, targets) == expected
        assert expected == _literal_projection_relations(flow, all_sources, targets, bindings)
        old_reads = sum(source != target for source in all_sources for target in targets)
        assert counted.reads < old_reads
        assert len(unique) < len(targets)  # Constant and end-relative Site aliases.
        measured.append((len(result), counted.reads, old_reads))
        return result

    monkeypatch.setattr(child, "_fixed_relations", checked_relations)
    projection = project_fixed_child(analysis, endpoints, insertion_cuts=insertion_cuts)
    assert measured and measured[0][0] == len(projection.relations)
    assert projection.endpoints == tuple(endpoints)
    assert all(source != target for source, target in projection.relations)


def test_fixed_projection_does_not_query_each_source_bit_again(one_copy, monkeypatch):
    _, analysis = one_copy
    actor = next(iter(analysis[2]))
    command = next(command for command in analysis[2][actor] if command.kind == "task")
    endpoints = tuple(command_site(command, 0, point) for point in ("begin", "end", "prefix"))

    def no_pairwise_coverage(*args, **kwargs):
        pytest.fail("fixed projection rescanned target facts for one source bit")

    monkeypatch.setattr(SymbolicCompletion, "coverage", no_pairwise_coverage)
    projection = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    assert projection.reaches(endpoints[0], endpoints[1])


@pytest.mark.parametrize("guard", (CountDomain(1, 0, exact=1), CountDomain(2, 0, minimum=0)))
def test_fixed_origin_masks_reject_parent_count_conditioned_facts(guard):
    from xdsl_npu.child_completion import _fixed_origins

    flow = SymbolicCompletion()
    node = flow.node(Family(0, (0, 0)))
    flow.seed(node, (0, 0), guard, 1)
    flow.propagate()
    with pytest.raises(CompileError, match="unconditional facts"):
        _fixed_origins(flow, node, (0, 0))


@pytest.mark.parametrize("point,horizon", (((1, -1), (0, 3)), ((0, 0), (1, -1))))
def test_fixed_origin_masks_reject_unresolved_parent_points_and_horizons(point, horizon):
    from xdsl_npu.child_completion import _fixed_origins

    flow = SymbolicCompletion()
    node = flow.node(Family(0, horizon))
    with pytest.raises(CompileError, match="constant points and family bounds"):
        _fixed_origins(flow, node, point)
