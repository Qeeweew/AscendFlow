"""Small physical CV counterexamples, not an executable attention operator."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.transforms.dead_code_elimination import dce

from tests.fixtures.cv import cv_program
from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import (
    BufferOp,
    CrossChannelOp,
    CrossDrainOp,
    CrossSyncOp,
    TaskOp,
    ViewOp,
)
from xdsl_npu.dispatch import extract_dispatch, static_dispatch
from xdsl_npu.mixed import ACTORS, global_dependencies, mixed_execution, mixed_graph
from xdsl_npu.sync import autosync, verify_plan


@pytest.fixture(scope="module")
def segmented_plan():
    return compile(cv_program(3, segments=2, groups=2)).plan


@pytest.fixture(scope="module")
def fragment_plan():
    return compile(cv_program(1, segments=2)).plan


def test_three_rings_keep_complete_contributions_and_every_physical_reader(segmented_plan):
    plan = parse(dump(segmented_plan))
    mixed, channels, commands = mixed_execution(plan)
    graph, begin, end, sends, receives, *_ = mixed_graph(mixed, channels, commands)
    assert set(channels) == {"scores_ready", "probabilities_ready", "pv_ready"}
    assert channels["probabilities_ready"].sources == ACTORS[1:]
    assert channels["probabilities_ready"].destinations == ("aic",)
    for actor in ACTORS[1:]:
        assert graph.precedes(
            sends[actor, "probabilities_ready", 0], receives["aic", "probabilities_ready", 0]
        )
    cross_reads = set()
    for source, destination, obligation in global_dependencies(mixed):
        assert graph.precedes(end[source], begin[destination])
        if obligation.hazard == "WAR" and source[0] in ACTORS[1:]:
            cross_reads.add(source[0])
    assert cross_reads == set(ACTORS[1:])
    assert all(c.pipe == "S" for seq in commands.values() for c in seq if c.kind == "cross_wait")
    # Generic passes cannot discard unused task IDs or collective effects.
    original = dump(plan)
    dce(plan)
    assert dump(plan) == original
    assert verify_plan(plan)["groups"] == 2


@pytest.mark.parametrize("channel", ["scores_ready", "probabilities_ready", "pv_ready"])
def test_missing_three_ring_visibility_is_rejected(segmented_plan, channel):
    plan = segmented_plan.clone()
    for op in list(plan.walk()):
        if (
            isinstance(op, CrossChannelOp)
            and op.sym_name.data == channel
            or isinstance(op, (CrossSyncOp, CrossDrainOp))
            and op.channel.data == channel
        ):
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match="missing mixed (RAW|WAR)"):
        verify_plan(plan)


@pytest.mark.parametrize(
    "channel,first", [("scores_ready", "qk_export0"), ("probabilities_ready", "vector_export0")]
)
def test_first_native_fragment_cannot_publish_the_complete_ring(fragment_plan, channel, first):
    # One epoch isolates incomplete publication from later slot reuse.
    plan = fragment_plan.clone()
    for send in list(plan.walk()):
        if (
            isinstance(send, CrossSyncOp)
            and send.channel.data == channel
            and send.action.data == "set"
        ):
            block = send.parent
            initial = next(
                op for op in block.ops if isinstance(op, TaskOp) and op.uid.data == first
            )
            block.detach_op(send)
            block.insert_op_after(send, initial)
    with pytest.raises(CompileError, match="missing mixed RAW"):
        verify_plan(plan)


def test_probabilities_need_both_aiv_contributions(segmented_plan):
    plan = segmented_plan.clone()
    channel = next(
        op
        for op in plan.walk()
        if isinstance(op, CrossChannelOp) and op.sym_name.data == "probabilities_ready"
    )
    channel.attributes["sources"] = ArrayAttr([StringAttr("aiv0")])
    with pytest.raises(CompileError, match="both AIV participants"):
        verify_plan(plan)


def test_logical_phase_is_independent_of_physical_issue_stage():
    source = cv_program(1, segments=2)
    load = next(
        op for op in source.walk() if isinstance(op, TaskOp) and op.uid.data == "pv_load_p0"
    )
    assert load.logical_phase.data == 2 and load.stage.data == 1
    load.attributes["logical_phase"] = IntAttr(0)
    with pytest.raises(CompileError, match="undefined probabilities"):
        autosync(source)


def test_delaying_cube_pv_past_a_blocking_consumer_creates_a_real_cycle():
    source = cv_program(5)
    for task in source.walk():
        if (
            isinstance(task, TaskOp)
            and task.logical_phase is not None
            and task.logical_phase.data == 2
        ):
            task.attributes["stage"] = IntAttr(4)
    with pytest.raises(CompileError, match="cycle|fixed schedule"):
        autosync(source)


def _delayed_state(slots):
    k = Kernel("delayed_state")
    state = k.buffer("state", "UB", slots * 32)
    result = k.buffer("result", "UB", 32)
    k.loop(4)
    owner_state = k.view(state, 32, stride=32, slots=slots, divisor=2)
    k.task("initialize", "fill", owner_state, scalar=1, period=2)
    k.task(
        "normalize", "muls", owner_state, k.view(result, 32), scalar=2, period=2, phase=1, stage=3
    )
    return k.finish()


def test_next_owner_state_cannot_overwrite_the_previous_normalization_reader():
    # Each owner spans two logical generations. Its last state read is issued
    # late, after the next owner's initialization. Distinct state slots make
    # that overlap legal; collapsing them exposes a physical WAR cycle.
    plan = autosync(_delayed_state(2))
    verify_plan(plan)
    state = next(
        op for op in plan.walk() if isinstance(op, BufferOp) and op.sym_name.data == "state"
    )
    for view in plan.walk():
        if isinstance(view, ViewOp) and view.source is state.ref:
            view.attributes["modulus"] = IntAttr(1)
    with pytest.raises(CompileError, match="missing WAR read_retired.*state"):
        verify_plan(plan)


def _two_case_source():
    cases = [cv_program(1), cv_program(1)]
    for index, module in enumerate(cases):
        roots = {
            op.sym_name.data: op
            for op in module.body.block.first_op.body.block.ops
            if isinstance(op, BufferOp)
        }
        sizes = {name: op.nbytes.data for name, op in roots.items()}
        for root in roots.values():
            root.attributes["nbytes"] = IntAttr(root.nbytes.data * 2)
        if index:
            for view in module.walk():
                if (
                    isinstance(view, ViewOp)
                    and isinstance(view.source.owner, BufferOp)
                    and view.source.owner.external.data
                ):
                    view.attributes["offset"] = IntAttr(
                        view.offset.data + sizes[view.source.owner.sym_name.data]
                    )
    return static_dispatch("cv_cases", cases)


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [("hole", "partition"), ("binding", "identical ordered"), ("workspace", "overlap GM scores")],
)
def test_two_small_cases_recheck_selectors_arguments_and_shared_ownership(mutation, diagnostic):
    plan = compile(_two_case_source()).plan
    case, mixed = extract_dispatch(plan).cases[1]
    if mutation == "hole":
        case.attributes["lower"] = IntAttr(case.lower.data + 1)
    elif mutation == "binding":
        root = next(
            op for op in case.walk() if isinstance(op, BufferOp) and op.sym_name.data == "Q"
        )
        root.attributes["nbytes"] = IntAttr(root.nbytes.data + 32)
    else:
        for view in case.walk():
            if (
                isinstance(view, ViewOp)
                and isinstance(view.source.owner, BufferOp)
                and view.source.owner.sym_name.data == "scores"
            ):
                view.attributes["offset"] = IntAttr(
                    view.offset.data - mixed.roots["scores"].nbytes.data // 2
                )
    with pytest.raises(CompileError, match=diagnostic):
        verify_plan(plan)
