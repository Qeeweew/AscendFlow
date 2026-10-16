"""Independent finite actual-command DAG and byte oracle for local sequences."""

import pytest

from examples.state.recurrence import state_sum
from tests.models.local_sequence import finite_check, physical_regions
from tests.report_contract import semantic_report
from xdsl_npu import autosync, compile, verify_plan
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.ir import (
    parse,
)
from xdsl_npu.model import CompileError
from xdsl_npu.performance import runtime_work


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 17])
def test_actual_instructions_cover_bytes_events_and_progress(state_artifact, count):
    finite_check(state_artifact.plan, count)


def test_entry_capture_zero_exit_and_single_drain_are_explicit(state_artifact):
    plan = state_artifact.plan
    assert len(physical_regions(plan)) == 3
    assert state_artifact.report["native_templates"] == 4
    assert state_artifact.report["sequence_channels"] == 2
    assert state_artifact.report["sequence_inherited_paths"] > 0
    assert state_artifact.asc.count("for (int64_t step") == 1
    assert state_artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert semantic_report(verify_plan(parse(state_artifact.plan_ir))) == semantic_report(state_artifact.report)
    for count in (0, 1, 7, 1048576):
        w = runtime_work(plan, {"tiles": count})
        assert w["native_invocations"]["copy"] == count + 2
        assert w["native_invocations"]["add"] == count
        assert w["terminal_drains"] == 1
        assert w["region_tasks_per_group"] == {"r0": 1, "r1": count, "r2": 1}


def readers():
    k = Kernel("readers", disjoint_external=True)
    initial = k.buffer("initial", "GM", 64, external=True, initialized=True)
    output = k.buffer("output", "GM", 64, external=True)
    state = k.buffer("state", "UB", 64)
    scratch = k.buffer("scratch", "UB", 64)
    count = k.runtime_size("n")
    k.region("entry")
    k.copy("init", k.view(initial, 64), k.view(state, 64))
    k.region("readers", count)
    k.task("scalar", "scalar_load", k.view(state, 4))
    k.scale("vector", k.view(state, 64), k.view(scratch, 64), 2.0)
    k.copy("dma", k.view(state, 64), k.view(output, 64))
    k.region("exit")
    k.task("overwrite", "fill", k.view(state, 64))
    return k.finish()


@pytest.fixture(scope="module")
def readers_plan():
    return autosync(readers())


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 17])
def test_all_reader_families_retire_before_overwrite(readers_plan, count):
    finite_check(readers_plan, count)


def test_two_retained_children_use_same_allocation_and_checked_resources():
    definition = N.from_source(
        """
        def twice(A:N.Tensor((64,),N.f32), O:N.Tensor((64,),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.copy(A,x)
                for t in N.Pipelined(n,stages=1):
                    N.mul(x,2.0,out=x)
                for t in N.Pipelined(n,stages=1):
                    N.add(x,x,out=x)
                N.copy(x,O)
    """,
        disjoint_external=True,
    )
    result = compile(definition)
    for n in (0, 1, 2, 7):
        finite_check(result.plan, n)


def test_borrowed_cross_region_value_is_diagnosed_until_version_binding_is_supported():
    source = state_sum.source.replace(
        "T.copy(Initial[group, :], state)", "saved = T.copy(Initial[group, :], state)"
    ).replace("T.add(state,", "T.add(saved,")
    # Capture's actual module alias is preserved by the definition environment.
    from dataclasses import replace

    with pytest.raises(CompileError, match="ambiguous ownership|producer"):
        compile(replace(state_sum, source=source))


def test_large_constant_bound_uses_boundary_families_instead_of_finite_history():
    source = """
        def large(A:N.Tensor((64,),N.f32), X:N.Tensor((COUNT,64),N.f32),
                  O:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((2,64))
                s=N.alloc_ub((64,))
                N.copy(A,s)
                for t in N.Pipelined(COUNT,stages=2,issue_order={"vector":("compute","load")}):
                    with N.Stage(0,name="load"):
                        N.copy(X[t,:],x[t%2,:])
                    with N.Stage(1,name="compute"):
                        N.add(s,x[t%2,:],out=s)
                N.copy(s,O)
    """
    reports = [
        compile(
            N.from_source(source, constants={"COUNT": count}, disjoint_external=True), timeout_s=3
        ).report
        for count in (1024, 1048576)
    ]
    assert all(r["native_templates"] == 4 for r in reports)
    assert reports[0]["completion_statistics"] == reports[1]["completion_statistics"]
    assert 0 < reports[1]["completion_statistics"]["condition_facts"] < 10000


def test_two_dma_pipelines_share_readonly_gm_and_release_only_actual_readers():
    source = state_sum.source.replace(
        "T.copy(state, Output[group, :])",
        """
        for u in T.Pipelined(tiles, stages=2):
            with T.Stage(0, name="reload"):
                T.copy(X[group, u, :], x[u % 2, :])
            with T.Stage(1, name="again"), T.Vector():
                T.add(state, x[u % 2, :], out=state)
        T.copy(state, Output[group, :])""",
    )
    from dataclasses import replace

    result = compile(replace(state_sum, source=source).specialize(B=16, GROUPS=1))
    for n in (0, 1, 2, 3, 7, 17):
        finite_check(result.plan, n)
    assert "asc_sync_pipe(PIPE_MTE2)" not in result.asc
    assert len(physical_regions(result.plan)) == 4


def test_cross_region_gm_ownership_is_checked_as_a_union():
    k = Kernel("bad_owners", disjoint_external=True, blocks=2)
    gm = k.buffer("gm", "GM", 256, external=True)
    ub = k.buffer("ub", "UB", 64)
    k.region("init")
    k.task("fill", "fill", k.view(ub, 64))
    k.region("left")
    k.copy("left_write", k.view(ub, 64), k.view(gm, 64, core_stride=64))
    k.region("right")
    k.copy("right_write", k.view(ub, 64), k.view(gm, 64, offset=64, core_stride=128))
    with pytest.raises(CompileError, match="sequential regions disagree on GM ownership"):
        autosync(k.finish())
