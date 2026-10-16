"""Source-epoch guards in actual parent synchronization ASC instructions."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import StringAttr

from tests.test_scope_codegen import VECTOR, checked_source, task_for_sync
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import ScopeSyncOp
from xdsl_npu.scope_codegen import _activation_guard, scope_boundary
from xdsl_npu.scope_domains import Activation


def channel(activation, distance=0):
    return SimpleNamespace(
        activation=activation,
        distance=distance,
        period=activation.period,
        residues=activation.residues,
    )


def evaluate(conditions, count, generation):
    expression = " && ".join(conditions) or "true"
    # Emitted guards contain integer affine arithmetic, comparisons, and
    # short-circuit conjunction/disjunction; no C parser or compiler is needed.
    return eval(
        expression.replace("&&", " and ").replace("||", " or "),
        {"__builtins__": {}},
        {"npu_parent_n": count, "npu_parent_g": generation, "true": True},
    )


CASES = (
    (Activation(), 0),
    (Activation(last=Position(True, 2)), 2),
    (Activation(first=Position(False, 2)), -2),
    (Activation(CountDomain(1, 0, exact=1), Position(False, 0), Position(False, 0)), 0),
    (Activation(CountDomain(1, 0, exact=3), Position(True, 0), Position(True, 0)), -2),
    (Activation(CountDomain(1, 0, minimum=2), Position(False, 1), Position(True, 0)), -1),
    (Activation(CountDomain(2, 1, minimum=1, maximum=7), period=3, residues=(0, 2)), 0),
    (
        Activation(
            CountDomain(3, 2, minimum=2, maximum=11),
            Position(False, 1),
            Position(True, 0),
            4,
            (1, 3),
        ),
        -1,
    ),
    (
        Activation(
            CountDomain(1, 0, minimum=4),
            Position(True, 2),
            Position(True, 1),
            3,
            (0, 1),
        ),
        1,
    ),
)


@pytest.mark.parametrize("activation,distance", CASES)
@pytest.mark.parametrize("action", ("set", "wait", "barrier"))
def test_guards_equal_source_epoch_predicates_for_small_counts(activation, distance, action):
    conditions = _activation_guard(channel(activation, distance), action)
    for count in range(12):
        for generation in range(count):
            epoch = generation if action == "set" else generation - distance
            expected = (
                activation.count.contains(count)
                and activation.first.evaluate(count) <= epoch <= activation.last.evaluate(count)
                and epoch % activation.period in activation.residues
            )
            assert evaluate(conditions, count, generation) == expected, (
                action,
                count,
                generation,
                conditions,
            )


def test_negative_distance_checks_destination_tail_before_forming_source_remainder():
    activation = Activation(first=Position(False, 2), period=3, residues=(0, 2))
    conditions = _activation_guard(channel(activation, -2), "wait")
    text = " && ".join(conditions)
    assert "npu_parent_g <= (npu_parent_n - 3)" in text
    assert text.index("npu_parent_n - 3") < text.index("npu_parent_g + 2")
    assert "npu_parent_g >= 0" not in text
    assert "+ 0" not in text and "- 0" not in text and "- -" not in text
    # Short-circuit bounds also protect the largest host-representable N.
    for generation in ((1 << 63) - 4, (1 << 63) - 3, (1 << 63) - 2):
        assert evaluate(conditions, (1 << 63) - 1, generation) == (
            generation <= (1 << 63) - 4 and (generation + 2) % 3 in (0, 2)
        )


def test_point_and_count_guards_simplify_without_redundant_boundary_arithmetic():
    activation = Activation(CountDomain(1, 0, exact=3), Position(True, 0), Position(True, 0))
    assert _activation_guard(channel(activation, -2), "set") == [
        "npu_parent_n == 3",
        "npu_parent_g == 2",
    ]
    assert _activation_guard(channel(activation, -2), "wait") == [
        "npu_parent_n == 3",
        "npu_parent_g == 0",
    ]


@pytest.mark.parametrize(
    "activation,distance",
    (
        (Activation(count=CountDomain(0, 0)), 0),
        (Activation(count=CountDomain(2, 2)), 0),
        (Activation(count=CountDomain(1, 0, minimum=-1)), 0),
        (Activation(count=CountDomain(1, 0, minimum=4, maximum=2)), 0),
        (Activation(first=Position(False, -1)), 0),
        (Activation(last=Position(True, -1)), 0),
        (Activation(period=0), 0),
        (Activation(period=True), 0),
        (Activation(period=3, residues=()), 0),
        (Activation(period=3, residues=(2, 0)), 0),
        (Activation(period=3, residues=(0, 0)), 0),
        (Activation(period=3, residues=(3,)), 0),
        (Activation(), -1),
        (Activation(), 1),
        (Activation(), True),
        (Activation(count=CountDomain(1, 0, exact=1 << 64)), 0),
        (
            Activation(first=Position(False, (1 << 63) - 1), last=Position(True, (1 << 63) - 1)),
            -((1 << 63) - 1),
        ),
    ),
)
def test_invalid_explicit_activation_cannot_become_an_unguarded_protocol(activation, distance):
    with pytest.raises(CompileError):
        _activation_guard(channel(activation, distance), "wait")


@pytest.mark.parametrize("field,value", (("period", 2), ("residues", (1,))))
def test_activation_and_event_ordinal_require_the_same_residue_stream(field, value):
    protocol = channel(Activation())
    setattr(protocol, field, value)
    with pytest.raises(CompileError, match="period/residues disagree"):
        _activation_guard(protocol, "set")


@pytest.fixture(scope="module")
def vector():
    return checked_source(VECTOR, tie=True)


def changed(vector, activation, *, distance=-1, ids=(0, 1, 2), kind="local"):
    original = next(protocol for protocol in vector.channels.values() if protocol.kind == "local")
    protocol = replace(
        original,
        kind=kind,
        distance=distance,
        period=activation.period if activation is not None else 1,
        residues=activation.residues if activation is not None else (0,),
        ids=ids,
        activation=activation,
    )
    return replace(vector, channels=vector.channels | {protocol.name: protocol}), protocol


def rendered(checked, protocol, action):
    leaf, op = next(
        (leaf, op)
        for leaf in checked.program.leaves.values()
        for op in leaf.stream.walk()
        if isinstance(op, ScopeSyncOp)
        and op.channel.data == protocol.name
        and op.action.data == ("wait" if action == "barrier" else action)
    )
    original = op.action
    if action == "barrier":
        op.attributes["action"] = StringAttr("barrier")
    try:
        task = task_for_sync(checked, leaf, op)
        text = scope_boundary(checked, leaf)(task, "NATIVE();")
    finally:
        op.attributes["action"] = original
    return (
        next(line for line in text.splitlines() if line.endswith(f"/* {protocol.name} */")),
        task,
        op,
    )


@pytest.mark.parametrize("action", ("set", "wait"))
def test_actual_local_sync_guards_and_rotating_ids_share_the_source_epoch(vector, action):
    activation = Activation(CountDomain(1, 0, minimum=2), Position(False, 1), Position(True, 0))
    checked, protocol = changed(vector, activation)
    text, _, op = rendered(checked, protocol, action)
    assert f"if (g == {op.position.data} && npu_parent_n >= 2" in text
    assert (
        "npu_parent_g >= 1" if action == "set" else "npu_parent_g <= (npu_parent_n - 2)"
    ) in text
    event = re.search(r"event_t\((.*?)\)\);", text).group(1)
    for generation in range(7):
        actual = eval(event, {"__builtins__": {}}, {"npu_parent_g": generation})
        expected = (generation if action == "set" else generation + 1) % 3
        assert actual == expected
    assert ("npu_parent_g + 1" in event) == (action == "wait")
    assert not re.search(r"\bg\b", event)
    assert "@npu_scalar_" not in text and "- -" not in text


def test_actual_scope_barrier_uses_shifted_source_epoch_residues(vector):
    activation = Activation(first=Position(False, 1), period=3, residues=(0, 2))
    checked, protocol = changed(vector, activation, kind="barrier", ids=())
    text, task, _ = rendered(checked, protocol, "barrier")
    assert "npu_parent_g <= (npu_parent_n - 2)" in text
    assert "((npu_parent_g + 1) % 3) == 0" in text
    assert "((npu_parent_g + 1) % 3) == 2" in text
    assert f"asc_sync_pipe(PIPE_{task.pipe});" in text
    assert "event_t" not in text


@pytest.mark.parametrize("action", ("set", "wait"))
def test_actual_cross_notification_preserves_count_and_boundary_guards(vector, action):
    activation = Activation(CountDomain(1, 0, exact=3), Position(True, 0), Position(True, 0))
    checked, protocol = changed(vector, activation, distance=-2, ids=(3,), kind="cross")
    text, task, _ = rendered(checked, protocol, action)
    assert "npu_parent_n == 3" in text
    assert f"npu_parent_g == {2 if action == 'set' else 0}" in text
    function = "asc_sync_block_arrive" if action == "set" else "asc_sync_block_wait"
    assert f"{function}(PIPE_{task.pipe}, 3);" in text


@pytest.mark.parametrize("action", ("set", "wait"))
def test_legacy_no_activation_keeps_the_original_guard_and_instruction_text(vector, action):
    checked, protocol = changed(vector, None, distance=2, ids=(2,))
    text, _, op = rendered(checked, protocol, action)
    tasks = {
        (actor, task.uid): task
        for actor, data in checked.access.loop.actors.items()
        for task in data.program.tasks
    }
    source = tasks[protocol.sources[0]].pipe
    destination = tasks[protocol.destinations[0]].pipe
    boundary = "npu_parent_g < npu_parent_n - 2" if action == "set" else "npu_parent_g >= 2"
    function = "asc_sync_notify" if action == "set" else "asc_sync_wait"
    assert text == (
        f"if (g == {op.position.data} && {boundary}) {{ "
        f"{function}(PIPE_{source}, PIPE_{destination}, event_t(2)); }} /* {protocol.name} */"
    )


def test_legacy_signed_distance_does_not_silently_emit_an_unbounded_range(vector):
    checked, protocol = changed(vector, None, distance=-1)
    with pytest.raises(CompileError, match="requires an explicit activation"):
        rendered(checked, protocol, "wait")
