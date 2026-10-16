"""Finite execution traces of the fixed PLE phases in original dispatch order."""

from collections import defaultdict
from dataclasses import dataclass

import pytest
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException

from tests.fixtures.pipelines import vector_pipeline
from xdsl_npu.backend.loops import CountBound, emit_pipeline_body
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.loops import LoopScheduleAttr, PhaseKind
from xdsl_npu.dialects.physical import StreamOp
from xdsl_npu.loop_model import extract_loop


@dataclass(frozen=True)
class Task:
    uid: str
    stage: int


@dataclass(frozen=True)
class Command:
    task: Task
    kind: str = "task"
    lower: int = 0
    upper: int = 0


def make_commands(stages):
    return tuple(Command(Task(f"task{i}", stage)) for i, stage in enumerate(stages))


def render(commands, *, schedule=None):
    calls = []

    def render_group(group, domain):
        calls.append((group, domain))
        lines = []
        for command in group:
            if not domain.permits(lower=command.lower, upper_gap=command.upper):
                continue
            guards = []
            if not domain.implies_lower(command.lower):
                guards.append(f"g >= {command.lower}")
            if not domain.implies_upper(command.upper):
                guards.append(f"g < {domain.count} - {command.upper}")
            line = f"execute_{command.task.uid}_{command.kind}(g);"
            if guards:
                line = f"if ({' && '.join(guards)}) {{ {line} }}"
            lines.append(line)
        return lines

    return emit_pipeline_body(commands, render_group, schedule=schedule), calls


def reference_trace(commands, count):
    """Literal physical steps, with the unsplit native and event predicates."""
    maximum = max((command.task.stage for command in commands), default=0)
    result = []
    for step in range(count + maximum):
        for command in commands:
            generation = step - command.task.stage
            if (
                0 <= generation < count
                and generation >= command.lower
                and generation < count - command.upper
            ):
                result.append((command.task.uid, command.kind, step, generation))
    return result


def phase_trace(calls, count):
    """Execute the structured bounds/guards that the emitter supplies."""
    phases = defaultdict(list)
    for group, domain in calls:
        phases[domain.phase].append((group, domain))
    result = []
    for groups in phases.values():
        first = groups[0][1]
        if not first.active(count):
            continue
        for step in range(first.step_lower.evaluate(count), first.step_upper.evaluate(count)):
            for group, domain in groups:
                if not domain.contains(step, count):
                    continue
                generation = step - domain.stage
                for command in group:
                    if not domain.permits(lower=command.lower, upper_gap=command.upper):
                        continue
                    if (domain.implies_lower(command.lower) or generation >= command.lower) and (
                        domain.implies_upper(command.upper) or generation < count - command.upper
                    ):
                        result.append((command.task.uid, command.kind, step, generation))
    return result


@pytest.mark.parametrize(
    "stages",
    [(0,), (0, 0), (0, 1), (0, 2), (0, 2, 1), (5, 0, 2, 0, 5), (2, 2)],
)
def test_phase_trace_matches_literal_dispatch_for_empty_short_equal_and_large_counts(stages):
    commands = make_commands(stages)
    _, calls = render(commands)
    maximum = max(stages)
    for count in range(maximum + 5):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_interleaved_lexical_groups_keep_exact_task_sync_order_and_generation():
    load, store, scale = Task("load", 0), Task("store", 2), Task("scale", 1)
    commands = (
        Command(load),
        Command(store, "wait"),
        Command(scale),
        Command(load, "set"),
        Command(store),
    )
    _, calls = render(commands)
    assert phase_trace(calls, 1) == [
        ("load", "task", 0, 0),
        ("load", "set", 0, 0),
        ("scale", "task", 1, 0),
        ("store", "wait", 2, 0),
        ("store", "task", 2, 0),
    ]
    assert phase_trace(calls, 2) == [
        ("load", "task", 0, 0),
        ("load", "set", 0, 0),
        ("load", "task", 1, 1),
        ("scale", "task", 1, 0),
        ("load", "set", 1, 1),
        ("store", "wait", 2, 0),
        ("scale", "task", 2, 1),
        ("store", "task", 2, 0),
        ("store", "wait", 3, 1),
        ("store", "task", 3, 1),
    ]


def test_only_original_consecutive_equal_stages_share_one_group():
    commands = make_commands((0, 0, 2, 0, 1, 2, 2))
    _, calls = render(commands)
    groups = defaultdict(list)
    for group, domain in calls:
        groups[domain.phase].append(group)
    expected = (commands[:2], commands[2:3], commands[3:4], commands[4:5], commands[5:])
    assert groups["steady"] == list(expected)
    assert groups["short"] == list(expected)
    assert groups["prologue"] == [expected[0], expected[2], expected[3]]
    assert groups["epilogue"] == [expected[1], expected[3], expected[4]]
    for group, _ in calls:
        for command in group:
            assert any(command is original for original in commands)


def test_stage_zero_has_one_ordinary_loop_without_stage_activation_if():
    commands = make_commands((0, 0))
    lines, calls = render(commands)
    assert lines[0] == "for (int64_t step = 0; step < n; ++step) { // ordinary"
    assert "    const int64_t g = step;" in lines
    assert not any("if (" in line for line in lines)
    assert [domain.phase for _, domain in calls] == ["ordinary"]
    assert phase_trace(calls, 0) == []


@pytest.mark.parametrize("maximum", [1, 2, 5])
def test_phase_bounds_and_count_branches_cover_only_real_iterations(maximum):
    commands = make_commands((0, maximum))
    lines, calls = render(commands)
    by_phase = {domain.phase: domain for _, domain in calls}
    assert lines[0] == f"if (n >= {maximum}) {{"
    assert (by_phase["prologue"].step_lower, by_phase["prologue"].step_upper) == (
        CountBound(),
        CountBound(0, maximum),
    )
    assert (by_phase["steady"].step_lower, by_phase["steady"].step_upper) == (
        CountBound(0, maximum),
        CountBound(1),
    )
    assert (by_phase["epilogue"].step_lower, by_phase["epilogue"].step_upper) == (
        CountBound(1),
        CountBound(1, maximum),
    )
    assert not any(domain.active(0) for domain in by_phase.values())
    if maximum == 1:
        assert "short" not in by_phase
        assert not any("else if" in line for line in lines)
    else:
        short = by_phase["short"]
        assert short.count_minimum == 1 and short.count_limit == maximum
        assert (short.step_lower, short.step_upper) == (CountBound(), CountBound(1, maximum))
        assert "} else if (n > 0) {" in lines
    for count in range(maximum + 4):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_phase_stage_guards_follow_their_actual_step_domains():
    _, calls = render(make_commands((0, 1, 2)))
    domains = {(domain.phase, domain.stage): domain for _, domain in calls}
    assert ("prologue", 2) not in domains
    assert ("epilogue", 0) not in domains
    for stage in range(3):
        steady = domains["steady", stage]
        assert not steady.stage_lower_guard and not steady.stage_upper_guard
    assert not domains["prologue", 0].stage_lower_guard
    assert domains["prologue", 1].stage_lower_guard
    assert not domains["prologue", 1].stage_upper_guard
    for stage in (1, 2):
        assert not domains["epilogue", stage].stage_lower_guard
    assert domains["epilogue", 1].stage_upper_guard
    assert not domains["epilogue", 2].stage_upper_guard
    assert not domains["short", 0].stage_lower_guard
    assert domains["short", 0].stage_upper_guard
    assert domains["short", 1].stage_lower_guard
    assert domains["short", 1].stage_upper_guard
    assert domains["short", 2].stage_lower_guard
    assert not domains["short", 2].stage_upper_guard


def test_highest_stage_epilogue_uses_loop_bound_without_an_extra_if():
    lines, calls = render(make_commands((0, 2)))
    text = "\n".join(lines)
    epilogue = text.split("// epilogue\n", 1)[1].split("} else if", 1)[0]
    assert "const int64_t g = step - 2;" in epilogue
    assert "execute_task1_task(g);" in epilogue
    assert "if (" not in epilogue
    for count in (0, 1, 2, 3, 7):
        assert phase_trace(calls, count) == reference_trace(make_commands((0, 2)), count)


def test_domain_generation_facts_hold_for_every_executed_small_trace():
    _, calls = render(make_commands((0, 3, 1, 0, 2)))
    for _, domain in calls:
        for count in range(9):
            for step in range(count + 3):
                if not domain.contains(step, count):
                    continue
                generation = step - domain.stage
                assert generation >= domain.generation_minimum
                assert generation < count - domain.generation_upper_gap
                for distance in range(6):
                    if domain.implies_lower(distance):
                        assert generation >= distance
                    if domain.implies_upper(distance):
                        assert generation < count - distance


@pytest.mark.parametrize("stages,selected", [((0,), 0), ((0, 1, 2), 1)])
def test_pipeline_fill_does_not_remove_distance_two_startup_or_release(stages, selected):
    tasks = tuple(command.task for command in make_commands(stages))
    commands = tuple(
        command
        for task in tasks
        for command in (
            Command(task, "wait", lower=2),
            Command(task),
            Command(task, "set", upper=2),
        )
    )
    lines, calls = render(commands)
    phase = "ordinary" if max(stages) == 0 else "steady"
    domain = next(
        domain for _, domain in calls if domain.phase == phase and domain.stage == selected
    )
    assert not domain.implies_lower(2)
    assert not domain.implies_upper(2)
    assert domain.permits(lower=2)
    assert domain.permits(upper_gap=2)
    text = "\n".join(lines)
    assert f"if (g >= 2) {{ execute_{tasks[selected].uid}_wait(g); }}" in text
    assert f"if (g < n - 2) {{ execute_{tasks[selected].uid}_set(g); }}" in text
    for count in range(9):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_phase_domains_remove_only_impossible_startup_and_release_commands():
    commands = tuple(
        command
        for task in (Task("load", 0), Task("scale", 1), Task("store", 2))
        for command in (
            Command(task, "wait", lower=2),
            Command(task),
            Command(task, "set", upper=2),
        )
    )
    lines, calls = render(commands)
    domains = {(domain.phase, domain.stage): domain for _, domain in calls}
    assert not domains["prologue", 0].permits(lower=2)
    assert not domains["epilogue", 1].permits(upper_gap=2)
    assert not domains["epilogue", 2].permits(upper_gap=2)
    for stage in range(3):
        assert not domains["short", stage].permits(lower=2)
        assert not domains["short", stage].permits(upper_gap=2)
    text = "\n".join(lines)
    prologue = text.split("// prologue\n", 1)[1].split("// steady\n", 1)[0]
    epilogue = text.split("// epilogue\n", 1)[1].split("} else if", 1)[0]
    short = text.split("// short\n", 1)[1]
    assert "execute_load_wait" not in prologue
    assert "execute_scale_set" not in epilogue
    assert "execute_store_set" not in epilogue
    assert "_wait(g)" not in short and "_set(g)" not in short
    for count in (0, 1, 2, 3, 4, 7):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_permits_keeps_short_count_limit_exclusive_and_long_minimum_inclusive():
    _, calls = render(make_commands((0, 1, 2)))
    domains = {(domain.phase, domain.stage): domain for _, domain in calls}
    short, prologue = domains["short", 0], domains["prologue", 0]
    assert not short.permits(lower=1)
    assert not short.permits(upper_gap=1)
    assert not short.active(2)
    assert prologue.active(2) and prologue.contains(1, 2)
    assert prologue.permits(lower=1)
    assert prologue.permits(upper_gap=1)
    assert [(step, step) for step in range(3) if short.contains(step, 1)] == [(0, 0)]
    assert [(step, step) for step in range(4) if prologue.contains(step, 2)] == [(0, 0), (1, 1)]


@pytest.mark.parametrize("maximum", [0, 1, 2, 3, 5])
def test_permits_matches_finite_literal_witnesses_for_all_small_bounds(maximum):
    _, calls = render(make_commands(tuple(range(maximum + 1))))
    for _, domain in calls:
        for lower in range(6):
            for upper_gap in range(6):
                # All offsets here are bounded by maximum/lower/upper_gap.
                # This finite window includes every possible minimum witness;
                # the implementation itself does not enumerate N or steps.
                limit = 2 * maximum + lower + upper_gap + 3
                witness = any(
                    domain.contains(step, count)
                    and lower <= step - domain.stage < count - upper_gap
                    for count in range(limit)
                    for step in range(count + maximum)
                )
                assert domain.permits(lower=lower, upper_gap=upper_gap) == witness


def test_permits_handles_large_constants_without_finite_domain_evaluation(monkeypatch):
    huge = 10**80
    _, calls = render(make_commands((0, 1, 32)))
    domains = {(domain.phase, domain.stage): domain for _, domain in calls}

    def fail(*args):
        raise AssertionError("permits must compare bounds without enumerating a domain")

    monkeypatch.setattr(CountBound, "evaluate", fail)
    assert not domains["prologue", 0].permits(lower=huge)
    assert not domains["epilogue", 1].permits(upper_gap=1)
    assert not domains["short", 0].permits(lower=huge - 1)
    assert domains["steady", 1].permits(lower=huge**2, upper_gap=huge**2)


@pytest.mark.parametrize("maximum,phase_count", [(0, 1), (1, 3), (2, 4), (32, 4)])
def test_rendering_has_at_most_four_fixed_phases_and_never_expands_runtime_count(
    maximum, phase_count
):
    commands = make_commands((0, maximum))
    lines, calls = render(commands)
    assert len({domain.phase for _, domain in calls}) == phase_count
    assert sum("for (int64_t step" in line for line in lines) == phase_count
    assert len(calls) <= 4 * len(commands)
    assert sum(len(group) for group, _ in calls) <= 4 * len(commands)
    assert len(lines) < 75


@pytest.mark.parametrize("maximum", [0, 1, 2, 5, 32])
def test_explicit_schedule_bounds_match_the_canonical_phase_descriptors(maximum):
    commands = make_commands((maximum, 0))
    schedule = LoopScheduleAttr(IntAttr(maximum))
    lines, calls = render(commands, schedule=schedule)
    assert lines == render(commands)[0]
    names = {
        PhaseKind.SHORT: "short",
        PhaseKind.PROLOGUE: "prologue",
        PhaseKind.LOOP: "ordinary" if maximum == 0 else "steady",
        PhaseKind.EPILOGUE: "epilogue",
    }
    canonical = {names[phase.kind]: phase for phase in schedule.phases}
    for _, domain in calls:
        phase = canonical[domain.phase]
        assert domain.step_lower == CountBound(phase.lower.coefficient, phase.lower.offset)
        assert domain.step_upper == CountBound(phase.upper.coefficient, phase.upper.offset)
        assert domain.count_minimum == phase.minimum_count
        assert domain.count_limit == (
            None if phase.maximum_count is None else phase.maximum_count + 1
        )
    for count in sorted({0, 1, maximum, maximum + 1, 2 * maximum + 3}):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_real_stream_schedule_is_consumed_by_identity_and_round_trips(monkeypatch):
    module = vector_pipeline()
    stream = next(operation for operation in module.walk() if isinstance(operation, StreamOp))
    schedule = LoopScheduleAttr(IntAttr(2))
    stream.attributes["npu.loop_schedule"] = schedule
    original_body = tuple(module.walk())
    original_phases = LoopScheduleAttr.phases.fget
    seen = []

    def phases(actual):
        seen.append(actual)
        return original_phases(actual)

    monkeypatch.setattr(LoopScheduleAttr, "phases", property(phases))
    loop = extract_loop(module)
    commands = tuple(Command(task) for task in next(iter(loop.actors.values())).program.tasks)
    lines, calls = render(commands, schedule=stream.attributes["npu.loop_schedule"])
    assert len(seen) == 1 and seen[0] is schedule
    assert stream.attributes["npu.loop_schedule"] is schedule
    assert tuple(module.walk()) == original_body
    text = dump(module)
    restored = parse(text)
    assert dump(restored) == text
    restored_stream = next(
        operation for operation in restored.walk() if isinstance(operation, StreamOp)
    )
    restored_schedule = restored_stream.attributes["npu.loop_schedule"]
    assert isinstance(restored_schedule, LoopScheduleAttr)
    assert render(commands, schedule=restored_schedule)[0] == lines
    assert seen[-1] is restored_schedule
    for count in (0, 1, 2, 7):
        assert phase_trace(calls, count) == reference_trace(commands, count)


def test_schedule_cannot_override_actual_stages_or_the_canonical_horizon_limit():
    commands = make_commands((0, 2))
    for maximum in (0, 1, 3):
        with pytest.raises(CompileError, match="horizon.*actual maximum stage"):
            render(commands, schedule=LoopScheduleAttr(IntAttr(maximum)))
    with pytest.raises(CompileError, match="typed loop schedule"):
        render(commands, schedule=IntAttr(2))
    with pytest.raises(VerifyException, match=r"\[0,32\]"):
        render(make_commands((0, 33)))
    with pytest.raises(CompileError, match="horizon.*actual maximum stage"):
        render((), schedule=LoopScheduleAttr(IntAttr(1)))
    assert render((), schedule=LoopScheduleAttr(IntAttr(0)))[0] == []


def test_count_alias_and_multiline_renderer_are_retained():
    def render_group(group, domain):
        assert domain.count == "tiles"
        return ["first(g);\nsecond(g);"]

    lines = emit_pipeline_body(make_commands((0, 2)), render_group, count="tiles")
    text = "\n".join(lines)
    assert "if (tiles >= 2)" in text
    assert "step < tiles + 2" in text
    assert "if (g < tiles)" in text
    assert "first(g);\n      second(g);" in text


@pytest.mark.parametrize(
    "bound,expected",
    [
        (CountBound(0, 3), "3"),
        (CountBound(1), "n"),
        (CountBound(1, 2), "n + 2"),
        (CountBound(1, -2), "n - 2"),
    ],
)
def test_count_bounds_have_only_one_existing_count_and_constant_offset(bound, expected):
    assert bound.c() == expected
    assert bound.evaluate(7) == 7 * bound.coefficient + bound.offset


def test_empty_body_and_invalid_stage_or_bound_have_no_fallback():
    def fail(*args):
        raise AssertionError("empty body must not invoke renderer")

    assert emit_pipeline_body((), fail) == []
    for stage in (-1, 0.5, True):
        with pytest.raises(CompileError, match="nonnegative stage"):
            emit_pipeline_body((Command(Task("invalid", stage)),), fail)
    with pytest.raises(CompileError, match="coefficient"):
        CountBound(2)
    with pytest.raises(CompileError, match="integer offset"):
        CountBound(1, 0.5)
