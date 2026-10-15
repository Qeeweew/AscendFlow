"""Actual parent/child finite byte histories and event lifetimes, not planner witnesses."""

import pytest
from ast_grouped_state_sum import grouped_state_sum
from test_sequence_sync import finite_check
from xdsl.dialects import scf
from xdsl.dialects.builtin import IntAttr

from xdsl_npu import compile, verify_plan
from xdsl_npu.ir import FrameOp, RegionChannelOp, RegionSyncOp, parse
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.fixture(scope="module")
def artifact():
    return compile(grouped_state_sum.specialize(B=16, GROUPS=1))


@pytest.mark.parametrize(
    "calls,count", [(0, 0), (0, 7), (1, 0), (3, 0), (1, 1), (3, 1), (2, 2), (3, 3), (2, 7), (3, 17)]
)
def test_actual_parent_sequence_bytes_and_binary_events(artifact, calls, count):
    finite_check(artifact.plan, count, invocations=calls)


def test_parent_and_child_are_retained_and_zero_child_executes_init_use(artifact):
    assert artifact.report["native_templates"] == 4
    assert artifact.asc.count("for (int64_t") == 2
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert "npu_invocation = 0; npu_invocation <" in artifact.asc
    assert verify_plan(parse(artifact.plan_ir)) == artifact.report
    carried = [o for o in artifact.plan.walk() if isinstance(o, RegionChannelOp) and o.carried]
    assert len(carried) == 1
    assert "PIPE_MTE3, PIPE_MTE2" in artifact.asc
    for calls, count in [(0, 7), (3, 0), (2, 7), (1048576, 1048576)]:
        work = runtime_work(artifact.plan, {"outputs": calls, "tiles": count})
        assert work["native_invocations"]["copy"] == calls * (count + 2)
        assert work["native_invocations"]["add"] == calls * count
        assert "tasks_per_invocation" not in work
        assert work["invocations_per_group"] == calls
        assert work["region_tasks_per_invocation"] == {"r0": 1, "r1": count, "r2": 1}
        assert work["region_tasks_per_group"] == {"r0": calls, "r1": calls * count, "r2": calls}
        assert work["terminal_drains"] == 1
        assert work["sync_invocations"].get("wait:MTE2", 0) == calls * max(0, count - 2) + max(
            0, calls - 1
        )


def test_missing_call_return_is_not_hidden_by_leaf_verification(artifact):
    plan = artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    for op in list(plan.walk()):
        if isinstance(op, RegionSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing sequence invocation"):
        verify_plan(plan)


def test_carried_guard_cannot_be_reinterpreted_as_forward_same_call(artifact):
    plan = artifact.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    channel.attributes["carried"] = IntAttr(0)
    with pytest.raises(CompileError, match="lexical dispatch order"):
        verify_plan(plan)


def test_real_parent_frame_coordinate_is_required(artifact):
    plan = artifact.plan
    parent = next(o for o in plan.walk() if isinstance(o, scf.ForOp))
    frame = next(o for o in plan.walk() if isinstance(o, FrameOp))
    frame.operands = (frame.source, parent.lb, frame.extent)
    with pytest.raises(CompileError, match="actual root, invocation coordinate"):
        verify_plan(plan)


def test_notify_effect_does_not_fence_later_same_pipe_work(artifact):
    from xdsl_npu.invocation_sync import Site
    from xdsl_npu.region_summary import CountDomain, Position
    from xdsl_npu.sequence_calls import CallProof, Mark, mark
    from xdsl_npu.sequence_sync import Point, verify_sequence

    _, checked = verify_sequence(artifact.plan, _details=True)
    seq = checked.sequence
    channel = next(c for c in checked.analyses["r1"][1].values() if c.distance == 0)
    source = Mark("r1", Site(channel.sources[0], Position(False, 0), "end", channel.name, "set"))
    target = mark(seq, Point("r1", channel.sources[0][1], Position(False, 1)), "prefix")
    proof = CallProof(seq, checked.analyses, ())
    assert not proof.direct(source, target, CountDomain(1, 0, exact=3))


def test_shared_stationary_frame_writes_compose_inside_the_actual_call():
    from dataclasses import replace

    source = grouped_state_sum.source.replace(
        "T.copy(Initial[group, output, :], state)",
        "T.copy(Initial[group, output, :], state)\n            T.copy(state, Output[group, output, :])",
    )
    result = compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
    for calls, count in [(0, 7), (1, 0), (3, 0), (2, 7)]:
        finite_check(result.plan, count, invocations=calls)


def test_two_children_in_one_parent_share_real_storage_and_event_lifetimes():
    from dataclasses import replace

    source = grouped_state_sum.source.replace(
        "T.copy(state, Output[group, output, :])",
        """
            for u in T.Pipelined(tiles, stages=2):
                with T.Stage(0, name="reload"):
                    T.copy(X[group, output, u, :], x[u % 2, :])
                with T.Stage(1, name="again"), T.Vector():
                    T.add(state, x[u % 2, :], out=state)
            T.copy(state, Output[group, output, :])""",
    )
    result = compile(
        replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1), timeout_s=10
    )
    assert result.report["native_templates"] == 6
    for calls, count in [(2, 0), (2, 1), (3, 3), (2, 7)]:
        finite_check(result.plan, count, invocations=calls)


def test_whole_last_reader_not_compute_completion_protects_state_reuse():
    from dataclasses import replace

    from xdsl_npu.ir import TaskOp

    source = grouped_state_sum.source.replace(
        "(GROUPS, outputs, tiles, B)", "(GROUPS, outputs, 1, B)"
    )
    source = source.replace("T.Pipelined(tiles,", "T.Pipelined(1,")
    result = compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
    plan = result.plan
    channel = next(o for o in plan.walk() if isinstance(o, RegionChannelOp) and o.carried)
    add = next(o for o in plan.walk() if isinstance(o, TaskOp) and o.native.data == "add")
    region = add.parent_op()
    send = next(
        o
        for o in plan.walk()
        if isinstance(o, RegionSyncOp) and o.channel == channel.sym_name and o.action.data == "set"
    )
    send.parent.detach_op(send)
    add.parent.insert_op_after(send, add)
    send.operands = (region.body.block.args[0],)
    channel.attributes.update(
        source_region=region.uid, source_task=add.uid, source_end=IntAttr(1), event_id=IntAttr(2)
    )
    with pytest.raises(CompileError, match="missing sequence invocation WAR"):
        verify_plan(plan)
    with pytest.raises(AssertionError):
        finite_check(plan, 0, invocations=2)


def test_parent_scalar_reader_is_retired_without_promoting_wait_to_completion():
    from dataclasses import replace

    source = grouped_state_sum.source.replace(
        "T.copy(state, Output[group, output, :])",
        "T.copy(state, Output[group, output, :])\n            sample = T.load_scalar(state[0:1])",
    )
    artifact = compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
    for calls, count in [(2, 0), (2, 1), (3, 3)]:
        finite_check(artifact.plan, count, invocations=calls)
    # An implicit Scalar completion must never reverse its own begin/end.
    from xdsl_npu.region_summary import CountDomain, Position
    from xdsl_npu.sequence_calls import CallProof, mark
    from xdsl_npu.sequence_sync import Point, verify_sequence

    _, checked = verify_sequence(artifact.plan, _details=True)
    seq = checked.sequence
    task = next(
        t for d in seq.regions["r2"].actors.values() for t in d.program.tasks if t.pipe == "S"
    )
    p = Point("r2", task.uid, Position(False, 0))
    proof = CallProof(seq, checked.analyses, ())
    assert not proof.direct(mark(seq, p, "end"), mark(seq, p, "begin"), CountDomain(1, 0, exact=1))


def test_canonical_predicate_intersection_matches_independent_finite_membership():
    from itertools import product

    from xdsl_npu.region_summary import CountDomain
    from xdsl_npu.sequence_calls import canonical, intersection

    domains = [
        CountDomain(p, r, minimum=m) for p in (1, 2, 3, 5) for r in range(p) for m in (0, 1, 4)
    ]
    domains += [CountDomain(1, 0, exact=n) for n in (0, 1, 7)]
    for a, b in product(domains, repeat=2):
        result = intersection(canonical(a), canonical(b))
        for n in range(32):
            assert canonical(a).contains(n) == a.contains(n)
            assert bool(result is not None and result.contains(n)) == (
                a.contains(n) and b.contains(n)
            )


def test_large_fixed_parent_and_child_do_not_expand_proof_windows():
    from xdsl_npu import language as N

    source = """
        def fixed(I:N.Tensor((1,CALLS,16),N.f32), X:N.Tensor((1,CALLS,TILES,16),N.f32),
                  O:N.Tensor((1,CALLS,16),N.f32,access="write")):
            with N.Kernel() as g:
                state=N.alloc_ub((16,))
                x=N.alloc_ub((2,16))
                for p in N.serial(CALLS):
                    N.copy(I[g,p,:],state)
                    for t in N.Pipelined(TILES,stages=2):
                        with N.Stage(0,name="load"):
                            N.copy(X[g,p,t,:],x[t%2,:])
                        with N.Stage(1,name="update"):
                            N.add(state,x[t%2,:],out=state)
                    N.copy(state,O[g,p,:])
    """
    reports = []
    for size in (1024, 1048576):
        reports.append(
            compile(
                N.from_source(
                    source, constants={"CALLS": size, "TILES": size}, disjoint_external=True
                ),
                timeout_s=10,
            ).report
        )
    for key in ("native_templates", "proof_domain_states", "invocation_path_visits"):
        assert reports[0][key] == reports[1][key]
    assert reports[0]["native_templates"] == 4
