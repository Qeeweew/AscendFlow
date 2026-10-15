"""Actual parent/child finite byte histories and event lifetimes, not planner witnesses."""

import pytest

from examples.state.grouped import grouped_state_sum
from tests.models.local_sequence import finite_check
from xdsl_npu import compile, verify_plan
from xdsl_npu.ir import RegionChannelOp, parse
from xdsl_npu.performance import runtime_work


@pytest.mark.parametrize(
    "calls,count", [(0, 0), (0, 7), (1, 0), (3, 0), (1, 1), (3, 1), (2, 2), (3, 3), (2, 7), (3, 17)]
)
def test_actual_parent_sequence_bytes_and_binary_events(grouped_artifact, calls, count):
    finite_check(grouped_artifact.plan, count, invocations=calls)


def test_parent_and_child_are_retained_and_zero_child_executes_init_use(grouped_artifact):
    assert grouped_artifact.report["native_templates"] == 4
    assert grouped_artifact.asc.count("for (int64_t") == 2
    assert grouped_artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert "npu_invocation = 0; npu_invocation <" in grouped_artifact.asc
    assert verify_plan(parse(grouped_artifact.plan_ir)) == grouped_artifact.report
    carried = [
        o for o in grouped_artifact.plan.walk() if isinstance(o, RegionChannelOp) and o.carried
    ]
    assert len(carried) == 1
    assert "PIPE_MTE3, PIPE_MTE2" in grouped_artifact.asc
    for calls, count in [(0, 7), (3, 0), (2, 7), (1048576, 1048576)]:
        work = runtime_work(grouped_artifact.plan, {"outputs": calls, "tiles": count})
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


def test_notify_effect_does_not_fence_later_same_pipe_work(grouped_artifact):
    from xdsl_npu.completion import Requirement, Site
    from xdsl_npu.region_completion import RegionCompletion, mark
    from xdsl_npu.region_summary import CountDomain, Position
    from xdsl_npu.sequence_sync import Point, verify_sequence

    _, checked = verify_sequence(grouped_artifact.plan, _details=True)
    seq = checked.sequence
    channel = next(c for c in checked.analyses["r1"][1].values() if c.distance == 0)
    source = Site(channel.sources[0], Position(False, 0), "end", channel.name, "set", region="r1")
    target = mark(seq, Point("r1", channel.sources[0][1], Position(False, 1)), "prefix")
    required = Requirement(source, target, CountDomain(1, 0, exact=3), "Set is not a fence")
    proof = RegionCompletion(seq, checked.analyses, (required,))
    assert not proof.covered(required)


def test_fifo_begin_composition_matches_the_actual_native_graph(grouped_artifact):
    from xdsl_npu.completion import CountDomain, LeafCompletion, Position, Requirement, Site
    from xdsl_npu.region_completion import RegionCompletion
    from xdsl_npu.sequence_sync import verify_sequence

    _, checked = verify_sequence(grouped_artifact.plan, _details=True)
    requirements, oracles = [], {}
    for region, (loop, _, streams, graph, _) in checked.analyses.items():
        native = {
            (actor, c.task.uid): c
            for actor, commands in streams.items()
            for c in commands
            if c.kind == "task"
        }
        endpoints = {
            (actor, c.channel, c.kind): c
            for actor, commands in streams.items()
            for c in commands
            if c.kind in {"set", "wait"}
        }
        oracles[region] = LeafCompletion(loop, graph, native, endpoints)
        for a in native:
            for b in native:
                for point in ("begin", "end", "prefix"):
                    requirements.append(
                        Requirement(
                            Site(a, Position(False, 0), "begin", region=region),
                            Site(b, Position(False, 0), point, region=region),
                            CountDomain(1, 0, exact=3),
                            "native relation",
                        )
                    )
    proof = RegionCompletion(checked.sequence, checked.analyses, requirements, calls=(0,))
    for r in requirements:
        # Old per-pair proof is an oracle in this test, never called by planning.
        if proof.covered(r):
            assert oracles[r.source.region].prove(r.source, r.destination, r.domain)


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


def test_parent_scalar_reader_is_retired_without_promoting_wait_to_completion():
    from dataclasses import replace

    source = grouped_state_sum.source.replace(
        "T.copy(state, Output[group, output, :])",
        "T.copy(state, Output[group, output, :])\n            sample = T.load_scalar(state[0:1])",
    )
    grouped_artifact = compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
    for calls, count in [(2, 0), (2, 1), (3, 3)]:
        finite_check(grouped_artifact.plan, count, invocations=calls)
    # An implicit Scalar completion must never reverse its own begin/end.
    from xdsl_npu.region_completion import RegionCompletion, mark
    from xdsl_npu.region_summary import CountDomain, Position
    from xdsl_npu.sequence_sync import Point, verify_sequence

    _, checked = verify_sequence(grouped_artifact.plan, _details=True)
    seq = checked.sequence
    task = next(
        t for d in seq.regions["r2"].actors.values() for t in d.program.tasks if t.pipe == "S"
    )
    p = Point("r2", task.uid, Position(False, 0))
    from xdsl_npu.completion import Requirement

    required = Requirement(
        mark(seq, p, "end"),
        mark(seq, p, "begin"),
        CountDomain(1, 0, exact=1),
        "Scalar end cannot precede its start",
    )
    proof = RegionCompletion(seq, checked.analyses, (required,))
    assert not proof.covered(required)


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
