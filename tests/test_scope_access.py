"""Real nested captures and parent hazards checked by an independent byte oracle."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr

from xdsl_npu import budget
from xdsl_npu import language as N
from xdsl_npu.ir import StreamOp, TaskOp, dump
from xdsl_npu.loop_sync import _requirements
from xdsl_npu.model import CompileError, Ref
from xdsl_npu.scalar import Expr, expression
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes

PITCHED = """
def nested(Score:N.Tensor((2,2,3,32),N.f32,access="write"),
           Y:N.Tensor((2,n,3,32),N.f32,access="write"), n:N.int64):
    with N.Kernel(2) as group:
        tmp=N.alloc_ub((3,8),name="tmp")
        full=N.alloc_ub((3,32),name="full")
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="write"):
                for j in N.Pipelined(4,stages=1):
                    N.fill(tmp,1.0)
                    N.copy(tmp,Score[group,t%2,:,j*8:(j+1)*8])
            with N.Stage(1,name="consume"):
                N.copy(Score[group,t%2,:,:],full)
                N.copy(full,Y[group,t,:,:])
"""


def program(source=PITCHED):
    source = N.from_source(source, disjoint_external=True).to_ir()
    return extract_scopes(lower_scopes(source.body.block.first_op))


@pytest.fixture(scope="module")
def pitched():
    scope = program()
    return scope, analyze_scope_access(scope)


def _oracle(scope, count):
    # Interpret capture expressions and native affine/modulo terms directly.
    # No boundary summary, access_cells, Ref.segments or production frontier.
    ordered = []
    for uid, leaf in scope.leaves.items():
        normalized, data = next(iter(leaf.loop.actors.items()))
        captures = {
            (
                capture.op.sym_name.data
                if capture.root.external.data
                else normalized + "/" + capture.op.sym_name.data
            ): capture
            for capture in leaf.captures
        }
        for inner in range(leaf.loop.constant_count):
            for index, task in enumerate(data.program.tasks):
                ordered.append(
                    (
                        leaf.call.logical_phase.data,
                        inner,
                        task.logical_phase,
                        index,
                        leaf.actor,
                        uid,
                        task,
                        leaf,
                        captures,
                    )
                )
    ordered.sort(key=lambda item: item[:6])
    state, edges = {}, set()
    for outer in range(count):
        for _, inner, _, _, actor, uid, task, leaf, captures in ordered:
            touched = {}
            for number, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                normalized = next(iter(leaf.loop.actors))
                capture = captures[leaf.loop.root(normalized, ref)]
                name = capture.root.sym_name.data
                if not capture.root.external.data:
                    name = actor + "/" + name
                bindings = {"$parent0": outer, "$core": 0, "n": count}
                start = capture.offset.evaluate(bindings) + sum(
                    offset + stride * (inner // divisor % modulus if modulus else inner // divisor)
                    for offset, stride, modulus, divisor in ref.terms
                )
                if task.native == "mad_first" and number == 2 and inner == 0:
                    mode = "W"
                for row in range(ref.rows):
                    for byte in range(
                        start + row * ref.row_stride, start + row * ref.row_stride + ref.nbytes
                    ):
                        touched.setdefault((name, byte), set()).update(mode)
            event = outer, actor, uid, inner, task.uid

            def add(hazard, source, destination, root, byte):
                if source is not None and not (
                    source[0] == destination[0] and source[2] == destination[2]
                ):
                    edges.add((root, byte, hazard, source, destination))

            for (root, byte), modes in touched.items():
                writer, readers = state.setdefault((root, byte), [None, []])
                if "R" in modes:
                    add("RAW", writer, event, root, byte)
                    readers.append(event)
                if "W" in modes:
                    add("WAW", writer, event, root, byte)
                    for reader in readers:
                        add("WAR", reader, event, root, byte)
                    writer, readers = event, []
                state[root, byte] = writer, readers
    return edges


def _expanded(access, scope, count):
    edges = set()
    for dependency in access.obligations:
        source_call, source = access.endpoints[dependency.source]
        target_call, target = access.endpoints[dependency.destination]
        for generation in range(dependency.distance, count):
            if generation % dependency.period not in dependency.residues:
                continue
            first = (
                generation - dependency.distance,
                source.task[0],
                source_call,
                source.position.evaluate(scope.leaves[source_call].loop.constant_count),
                source.task[1],
            )
            last = (
                generation,
                target.task[0],
                target_call,
                target.position.evaluate(scope.leaves[target_call].loop.constant_count),
                target.task[1],
            )
            for byte in range(dependency.start, dependency.stop):
                edges.add((dependency.storage, byte, dependency.hazard, first, last))
    return edges


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
def test_parent_hazards_match_actual_native_byte_events(pitched, count):
    scope, access = pitched
    assert _expanded(access, scope, count) == _oracle(scope, count)


def test_parent_slot_period_and_real_inner_writers_are_separate(pitched):
    scope, access = pitched
    assert access.loop.storage_periods["Score"] == 2
    assert "Score" not in access.loop.frames
    assert access.loop.frames["Y"] == 384
    assert len(access.endpoints) == 8
    assert all(
        task.native == "scope_access"
        for actor in access.loop.actors.values()
        for task in actor.program.tasks
    )
    raw = [d for d in access.obligations if d.storage == "Score" and d.hazard == "RAW"]
    assert {access.endpoints[d.source][1].position.offset for d in raw} == {0, 1, 2, 3}
    assert all(d.distance == 0 for d in raw)
    assert {
        d.distance for d in access.obligations if d.storage == "Score" and d.hazard == "WAR"
    } == {2}
    assert any(
        d.storage == "aiv0/tmp" and d.distance == 1 and d.hazard == "WAR"
        for d in access.obligations
    )
    original = {
        capture.root.sym_name.data: capture.root.ref
        for leaf in scope.leaves.values()
        for capture in leaf.captures
    }
    assert access.origins["Score"] is original["Score"]
    assert access.origins["aiv0/tmp"] is original["tmp"]
    assert access.visits < 1000


def test_memory_view_does_not_clone_or_mutate_native_ir(pitched, monkeypatch):
    scope, _ = pitched
    before = dump(scope.module)
    monkeypatch.setattr(
        TaskOp, "clone", lambda *args, **kwargs: pytest.fail("cloned a native task")
    )
    monkeypatch.setattr(StreamOp, "clone", lambda *args, **kwargs: pytest.fail("cloned a loop"))
    analyze_scope_access(scope)
    assert dump(scope.module) == before
    assert sum(isinstance(op, TaskOp) for op in scope.module.walk()) == 4


def test_missing_child_column_is_not_parent_initialized_storage():
    with pytest.raises(CompileError, match="undefined.*read on Score"):
        analyze_scope_access(program(PITCHED.replace("N.Pipelined(4,", "N.Pipelined(3,")))


def test_empty_child_does_not_initialize_parent_output():
    with pytest.raises(CompileError, match="undefined.*read on Score"):
        analyze_scope_access(program(PITCHED.replace("N.Pipelined(4,", "N.Pipelined(0,")))


def test_equal_call_issue_ranks_keep_whole_child_lexical_dispatch(pitched):
    module = pitched[0].module.clone()
    scope = extract_scopes(module)
    for leaf in scope.leaves.values():
        leaf.call.attributes["issue_order"] = IntAttr(0)
    scope = extract_scopes(module)
    access = analyze_scope_access(scope)
    first, second = scope.leaves
    ranks = {uid: [] for uid in scope.leaves}
    for actor, data in access.loop.actors.items():
        for task in data.program.tasks:
            ranks[access.endpoints[actor, task.uid][0]].append(task.issue_order)
    assert max(ranks[first]) < min(ranks[second])
    assert _expanded(access, scope, 3) == _oracle(scope, 3)


def test_multiple_calls_in_one_parent_stage_keep_global_logical_order():
    source = PITCHED.replace('N.Stage(1,name="consume")', 'N.Stage(0,name="consume")')
    scope = program(source)
    assert {leaf.call.stage.data for leaf in scope.leaves.values()} == {0}
    assert len({leaf.call.logical_phase.data for leaf in scope.leaves.values()}) == 2
    access = analyze_scope_access(scope)
    assert _expanded(access, scope, 3) == _oracle(scope, 3)


def test_shared_l0c_qk_fix6_releases_pv_first_mad_and_keeps_slot1_fix7():
    source = """
def cube(X:N.Tensor((1,n,8,16,16),N.f16),
         Score:N.Tensor((1,2,16,128),N.f32,access="write"),
         Y:N.Tensor((1,n,16,16),N.f32,access="write"), n:N.int64):
    with N.Kernel(1,kind="cube") as group:
        l1=N.alloc_l1((2,16,16),name="l1")
        a=N.alloc_l0a((2,16,16),name="a")
        b=N.alloc_l0b((2,16,16),name="b")
        c=N.alloc_l0c((2,16,16),name="c")
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="qk"):
                for j in N.Pipelined(8,stages=4,issue_order={"cube":("load","mad","operands","export")}):
                    with N.Stage(0,name="load"):
                        x=N.copy(X[group,t,j,:,:],l1[j%2,:,:])
                    with N.Stage(1,name="operands"):
                        aa=N.copy(x,a[j%2,:,:])
                        bb=N.copy(x,b[j%2,:,:],transpose=True)
                    with N.Stage(2,name="mad"):
                        z=N.matmul(aa,bb,accum=c[j%2,:,:],initialize=True)
                    with N.Stage(3,name="export"):
                        N.copy(z,Score[group,t%2,:,j*16:(j+1)*16])
            with N.Stage(1,name="pv"):
                for j in N.Pipelined(8,stages=1):
                    N.matmul(a[0,:,:],b[0,:,:],accum=c[0,:,:],initialize=(j==0))
                N.copy(c[0,:,:],Y[group,t,:,:])
"""
    scope = program(source)
    access = analyze_scope_access(scope)
    qk, pv, export = scope.leaves
    actual = {
        name: {task.uid: task for data in leaf.loop.actors.values() for task in data.program.tasks}
        for name, leaf in scope.leaves.items()
    }
    matches = []
    for dependency in access.obligations:
        if dependency.storage != "aic/c" or dependency.hazard != "WAR":
            continue
        first_call, first = access.endpoints[dependency.source]
        last_call, last = access.endpoints[dependency.destination]
        matches.append(
            (
                first_call,
                actual[first_call][first.task[1]].pipe,
                first.position.offset,
                last_call,
                actual[last_call][last.task[1]].native,
                last.position.offset,
                dependency.distance,
                dependency.start,
            )
        )
    assert (qk, "FIX", 6, pv, "mad_first", 0, 0, 0) in matches
    assert (qk, "FIX", 7, qk, "mad", 1, 1, 1024) in matches
    assert any(
        d.storage == "aic/c"
        and d.hazard == "RAW"
        and access.endpoints[d.source][0] == pv
        and access.endpoints[d.source][1].position.offset == 7
        and access.endpoints[d.destination][0] == export
        for d in access.obligations
    )
    roots = {
        capture.root.ref
        for leaf in scope.leaves.values()
        for capture in leaf.captures
        if capture.root.sym_name.data == "c"
    }
    assert roots == {access.origins["aic/c"]}


def test_cross_group_capture_alias_is_rejected():
    with pytest.raises(CompileError, match="cross-group.*ownership overlap"):
        analyze_scope_access(program(PITCHED.replace("Score[group,", "Score[0,")))


def _child_fragments(scope):
    return sum(
        leaf.loop.constant_count * ref.rows * (scope.groups if ref.storage.external else 1)
        for leaf in scope.leaves.values()
        for actor in leaf.loop.actors.values()
        for task in actor.program.tasks
        for ref in task.refs
    )


def test_child_fragment_group_budget_precedes_segments(pitched, monkeypatch):
    import xdsl_npu.scope_access as analysis

    monkeypatch.setattr(analysis, "MAX_SCOPE_ACCESS_SEGMENTS", 1)
    monkeypatch.setattr(
        Ref, "segments", lambda *args: pytest.fail("materialized before child budget")
    )
    with pytest.raises(CompileError, match="fixed-child fragment/group budget"):
        analysis.analyze_scope_access(pitched[0])


def test_parent_period_group_budget_precedes_owner_expansion(pitched, monkeypatch):
    import xdsl_npu.scope_access as analysis

    scope, _ = pitched
    monkeypatch.setattr(analysis, "MAX_SCOPE_ACCESS_SEGMENTS", _child_fragments(scope))
    monkeypatch.setattr(
        analysis,
        "verify_access_ownership",
        lambda *args: pytest.fail("owner expansion before budget"),
    )
    with pytest.raises(CompileError, match="parent fragment/period/group budget"):
        analysis.analyze_scope_access(scope)


def test_cut_budget_precedes_parent_frontier_materialization(pitched, monkeypatch):
    import xdsl_npu.scope_access as analysis

    scope, access = pitched
    cost = _child_fragments(scope) + sum(summary.visits for summary in access.summaries.values())
    cost += sum(
        access.loop.storage_periods[access.loop.root(actor, ref)]
        * (scope.groups if ref.storage.external else 1)
        for actor, data in access.loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
    )
    monkeypatch.setattr(analysis, "MAX_SCOPE_ACCESS_VISITS", cost)
    monkeypatch.setattr(
        analysis, "loop_dependencies", lambda *args: pytest.fail("frontiers before cut budget")
    )
    with pytest.raises(CompileError, match="visit budget|cut/frontier budget"):
        analysis.analyze_scope_access(scope)


def test_parent_analysis_polls_cooperative_cancellation(pitched, monkeypatch):
    import xdsl_npu.scope_access as analysis

    phases, original = [], analysis.poll

    def poll(phase, **counts):
        phases.append(phase)
        original(phase, **counts)

    monkeypatch.setattr(analysis, "poll", poll)
    with (
        pytest.raises(CompileError, match="cancelled at scope parent ownership"),
        budget.limit(cancel=lambda: "scope parent ownership" in phases),
    ):
        analysis.analyze_scope_access(pitched[0])


@pytest.mark.parametrize(
    "offset",
    [
        expression("floordiv", Expr("argument", ("$parent0",)), Expr("argument", ("n",))),
        expression("mul", Expr("argument", ("$parent0",)), Expr("argument", ("$core",))),
    ],
)
def test_unsupported_runtime_quotient_and_mixed_capture_atoms_are_rejected(pitched, offset):
    scope, _ = pitched
    uid, leaf = next(iter(scope.leaves.items()))
    captures = (replace(leaf.captures[0], offset=offset), *leaf.captures[1:])
    leaves = {**scope.leaves, uid: replace(leaf, captures=captures)}
    with pytest.raises(
        CompileError, match="quotient|divisor|unsupported.*clocks|unsupported parent/core"
    ):
        analyze_scope_access(replace(scope, leaves=leaves))


def test_fixed_parent_zero_and_one_do_not_create_inactive_backedges():
    for count in (0, 1):
        source = PITCHED.replace("N.Pipelined(n,", f"N.Pipelined({count},")
        # A fixed parent also has a fixed per-group output pitch. Keeping an
        # unrelated runtime n in that pitch is outside this initial contract.
        source = source.replace("Y:N.Tensor((2,n,3,32)", "Y:N.Tensor((2,1,3,32)")
        scope = program(source)
        access = analyze_scope_access(scope)
        assert all(d.distance < count for d in access.obligations)
        if count == 0:
            assert access.obligations == ()


def test_child_first_last_branch_is_explicitly_unsupported(pitched):
    scope, _ = pitched
    uid, leaf = next(iter(scope.leaves.items()))
    actor, data = next(iter(leaf.loop.actors.items()))
    first, *rest = data.program.tasks
    data = replace(data, program=replace(data.program, tasks=(replace(first, last=True), *rest)))
    leaf = replace(leaf, loop=replace(leaf.loop, actors={actor: data}))
    with pytest.raises(CompileError, match="first/last child branches"):
        analyze_scope_access(replace(scope, leaves={**scope.leaves, uid: leaf}))


def test_twin_aiv_local_uids_keep_actual_endpoints_and_collective_quorum():
    source = """
def cv(Q:N.Tensor((1,n,16,16),N.f16), Score:N.Tensor((1,2,16,16),N.f32,access="write"),
       Y:N.Tensor((1,n,16,16),N.f32,access="write"), n:N.int64):
    with N.Kernel(1,kind="mixed") as group:
        q=N.alloc_l1((16,16))
        a=N.alloc_l0a((16,16))
        b=N.alloc_l0b((16,16))
        c=N.alloc_l0c((16,16))
        u=N.alloc_ub((8,16))
        for t in N.Pipelined(n,stages=2):
            with N.Stage(0,name="produce"),N.Cube():
                for j in N.Pipelined(1,stages=1):
                    v=N.copy(Q[group,t,:,:],q)
                    aa=N.copy(v,a)
                    bb=N.copy(v,b,transpose=True)
                    z=N.matmul(aa,bb,accum=c,initialize=True)
                    N.copy(z,Score[group,t%2,:,:])
            with N.Stage(1,name="consume"),N.Vector() as lane:
                for j in N.Pipelined(1,stages=1):
                    v=N.copy(Score[group,t%2,lane*8:lane*8+8,:],u)
                    N.copy(v,Y[group,t,lane*8:lane*8+8,:])
"""
    scope = program(source)
    access = analyze_scope_access(scope)
    twins = {
        actor: {uid for owner, uid in access.endpoints if owner == actor}
        for actor in ("aiv0", "aiv1")
    }
    assert twins["aiv0"] == twins["aiv1"]
    for uid in twins["aiv0"]:
        first, second = access.endpoints["aiv0", uid], access.endpoints["aiv1", uid]
        assert first[0] != second[0]
        assert first[1].task[0] == "aiv0" and second[1].task[0] == "aiv1"
        assert first[1].task[1] != second[1].task[1]
    protocols = _requirements(access.loop, access.obligations)
    assert any(
        p.kind == "cross" and {actor for actor, _ in p.destinations} == {"aiv0", "aiv1"}
        for p, _ in protocols
    )
    assert _expanded(access, scope, 3) == _oracle(scope, 3)
