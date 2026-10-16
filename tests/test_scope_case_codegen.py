"""One actual child template emits its verified parent-case native and flags."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr

from tests.test_scope_native_guards import SOURCE
from xdsl_npu import language as N
from xdsl_npu.completion import Position
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import LoopSyncOp, ScopeGuardOp, ScopeSyncOp, TaskOp
from xdsl_npu.loop_codegen import _guard, _region_lines
from xdsl_npu.loop_graph import channels, commands
from xdsl_npu.loop_sync import _verify_leaf
from xdsl_npu.scope_codegen import (
    _parent_case_guard,
    emit_scope_actor,
    scope_boundary,
    scope_command_guard,
)
from xdsl_npu.scope_lowering import lower_scopes
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_predicates import PARENT_CASES, case_activation
from xdsl_npu.scope_sync import ScopeProtocol
from xdsl_npu.scope_variants import plan_child_variants


def evaluate(guard, count, generation):
    return eval(
        guard.replace("&&", " and ").replace("||", " or ") or "True",
        {"__builtins__": {}},
        {"npu_parent_n": count, "npu_parent_g": generation, "false": False},
    )


ALL_SUBSETS = tuple(
    subset for size in range(len(PARENT_CASES) + 1) for subset in combinations(PARENT_CASES, size)
)


@pytest.mark.parametrize("selected", ALL_SUBSETS)
def test_case_boolean_guards_equal_activation_unions_at_parent_boundaries(selected):
    guard = _parent_case_guard(selected)
    for count in (1, 2, 3, 4, 7, (1 << 63) - 1):
        positions = {0, count - 1, *range(1, min(4, count)), max(0, count - 2)}
        for generation in sorted(positions):
            expected = any(
                activation.count.contains(count)
                and activation.first.evaluate(count)
                <= generation
                <= activation.last.evaluate(count)
                for label in selected
                for activation in (case_activation(label),)
            )
            assert evaluate(guard, count, generation) == expected


def test_case_unions_simplify_shared_first_last_and_interior_boundaries():
    assert _parent_case_guard(PARENT_CASES) == ""
    assert _parent_case_guard(("single", "first")) == "npu_parent_g == 0"
    assert _parent_case_guard(("single", "last")) == "npu_parent_g == (npu_parent_n - 1)"
    assert _parent_case_guard(("first", "middle")) == "npu_parent_g < (npu_parent_n - 1)"
    assert _parent_case_guard(("middle", "last")) == "npu_parent_g > 0"
    assert _parent_case_guard(("single",)) == "npu_parent_n == 1"
    with pytest.raises(CompileError, match="unknown ASC parent case"):
        _parent_case_guard(("unknown",))


def roots(loop):
    return {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
    }


@pytest.fixture
def checked():
    # Use the normal source and physical extraction path. Only the parent
    # proof object is a small stand-in: these tests exercise native emission,
    # while each surviving child case has its real local completion proof.
    program = N.from_source(SOURCE, disjoint_external=True).to_ir().body.block.first_op
    physical = lower_scopes(program)
    extracted = extract_scopes(physical)
    for leaf in extracted.leaves.values():
        plan_child_variants(
            leaf.stream,
            leaf.call,
            {label: case.loop for label, case in leaf.cases.items()},
            live_in={
                label: roots(case.loop) if case.loop else () for label, case in leaf.cases.items()
            },
        )
    extracted = extract_scopes(physical)
    analyses, verified_cases = {}, {}
    for uid, leaf in extracted.leaves.items():
        channel_map = channels(leaf.loop)
        sequences, _, _ = commands(leaf.loop, channel_map)
        analyses[uid] = leaf.loop, channel_map, sequences, None, None
        for label, case in leaf.cases.items():
            if case.loop is not None:
                _, verified_cases[uid, label] = _verify_leaf(
                    case.loop, live_in=roots(case.loop), _details=True
                )
    parent_data = SimpleNamespace(program=SimpleNamespace(storages=(), tasks=()))
    return SimpleNamespace(
        program=extracted,
        analyses=analyses,
        cases=verified_cases,
        channels={},
        access=SimpleNamespace(
            loop=SimpleNamespace(actors={actor: parent_data for actor in extracted.parents}),
            endpoints={},
        ),
    )


def signature(command):
    return (
        command.task.uid,
        command.kind,
        command.channel,
        command.lower,
        command.upper,
        command.period,
        command.residues,
        command.last_gap,
    )


def parent_guard(code):
    if code == "EFFECT();":
        return ""
    assert code.startswith("if (") and code.endswith(") { EFFECT(); }")
    return code[len("if (") : -len(") { EFFECT(); }")]


def test_actual_commands_emit_exactly_the_native_and_syncs_in_each_verified_case(checked):
    native_count = sync_count = 0
    for uid, leaf in checked.program.leaves.items():
        analysis = checked.analyses[uid]
        sequence = next(iter(analysis[2].values()))
        render = scope_command_guard(leaf, sequence)
        assert render is not None
        for command in sequence:
            native_count += command.kind == "task"
            sync_count += command.kind != "task"
            guard = parent_guard(render(command, "EFFECT();"))
            for count in (1, 2, 3, 7):
                for generation in range(count):
                    label = (
                        "single"
                        if count == 1
                        else "first"
                        if generation == 0
                        else "last"
                        if generation == count - 1
                        else "middle"
                    )
                    case = checked.cases.get((uid, label))
                    surviving = (
                        {signature(op) for op in next(iter(case[2].values()))} if case else set()
                    )
                    assert evaluate(guard, count, generation) == (signature(command) in surviving)
    assert native_count == 8 and sync_count > 0


def test_scope_actor_keeps_one_child_and_never_guards_the_ordinary_tail(checked):
    actor = next(iter(checked.program.parents))
    text = "\n".join(emit_scope_actor(checked, actor, {}))
    assert text.count("npu_parent_step < npu_parent_n + 2") == 1
    assert text.count("for (int64_t step = 0; step < n + 2; ++step)") == 1
    assert text.count("ordinary region: execute once") == 1
    assert "npu_invocation" not in text and "@npu_scalar_" not in text
    for uid, leaf in checked.program.leaves.items():
        data = next(iter(checked.analyses[uid][0].actors.values()))
        for task in data.program.tasks:
            marker = f"// {actor}/{task.uid}: task, stage {task.stage}\n"
            assert text.count(marker) == 1
            line = text.split(marker, 1)[1].splitlines()[0]
            native = next(
                op
                for op in leaf.stream.walk()
                if isinstance(op, TaskOp) and op.uid.data == task.uid
            )
            if isinstance(native.parent_op(), ScopeGuardOp):
                assert "if (npu_parent_g == " in line
            else:
                assert "npu_parent_g" not in line
    assert "asc_sync_notify" in text and "asc_sync_wait" in text


def region_args(checked, uid):
    loop, channel_map, sequences, _, _ = checked.analyses[uid]
    actor, data = next(iter(loop.actors.items()))
    native = {(actor, task.uid): task for task in data.program.tasks}
    return actor, data, sequences, channel_map, native, None, (), {}


def test_command_callback_runs_after_inner_guards_before_scalar_materialization(checked):
    uid, leaf = next(
        (uid, leaf) for uid, leaf in checked.program.leaves.items() if leaf.stream.count.data == 2
    )
    args = region_args(checked, uid)
    normal = _region_lines(*args)
    assert normal == _region_lines(*args, command_guard=None)
    assert normal == _region_lines(*args, command_guard=lambda command, code: code)
    # This fixture's real selected events all have zero child distance. Give
    # one descriptor an inner lower bound only to probe callback ordering.
    sequence = [replace(args[2][args[0]][0], lower=1), *args[2][args[0]][1:]]
    args = (*args[:2], args[2] | {args[0]: sequence}, *args[3:])
    render, seen = scope_command_guard(leaf, sequence), []

    def observe(command, code):
        seen.append((command, code))
        return render(command, code)

    emitted = "\n".join(_region_lines(*args, command_guard=observe))
    assert len(seen) == len(sequence)
    assert any("@npu_scalar_" in code for _, code in seen)
    assert "@npu_scalar_" not in emitted
    guarded = [
        (command, code) for command, code in seen if _guard(command, args[3].get(command.channel))
    ]
    assert guarded and all(code.startswith("if (") for _, code in guarded)
    # The final enclosing brace precedes a local event's line comment.
    for line in emitted.splitlines():
        if "// case_c" in line:
            assert line.split("//", 1)[0].rstrip().endswith("}")


def test_a_single_case_descriptor_cannot_omit_actual_native_or_sync_templates(checked):
    uid, leaf = next(
        (uid, leaf) for uid, leaf in checked.program.leaves.items() if leaf.stream.count.data == 2
    )
    first_case = checked.cases[uid, "first"]
    with pytest.raises(CompileError, match="complete actual child descriptor"):
        scope_command_guard(leaf, next(iter(first_case[2].values())))
    sequence = next(iter(checked.analyses[uid][2].values()))
    duplicate = next(command for command in sequence if command.kind != "task")
    with pytest.raises(CompileError, match="duplicated"):
        scope_command_guard(leaf, [*sequence, replace(duplicate, index=10000)])


def test_parent_sync_inside_actual_native_guard_keeps_its_source_epoch_condition(checked):
    uid, leaf = next(
        (uid, leaf) for uid, leaf in checked.program.leaves.items() if leaf.stream.count.data == 2
    )
    native = next(
        op
        for op in leaf.stream.walk()
        if isinstance(op, TaskOp) and isinstance(op.parent_op(), ScopeGuardOp)
    )
    data = next(iter(checked.analyses[uid][0].actors.values()))
    task = next(task for task in data.program.tasks if task.uid == native.uid.data)
    sync = ScopeSyncOp.create(
        operands=[leaf.call.generation, leaf.stream.body.block.args[0]],
        attributes={
            "action": StringAttr("barrier"),
            "channel": StringAttr("parent_tail"),
            "position": IntAttr(0),
        },
    )
    native.parent.insert_op_before(sync, native)
    endpoint = leaf.actor, "native_tail_endpoint"
    checked.access.endpoints[endpoint] = (
        uid,
        SimpleNamespace(task=(leaf.actor, task.uid), position=Position(False, 0)),
    )
    checked.channels["parent_tail"] = ScopeProtocol(
        "parent_tail",
        "barrier",
        (endpoint,),
        (endpoint,),
        0,
        1,
        (0,),
        activation=case_activation("last"),
    )
    code = scope_boundary(checked, leaf)(task, "NATIVE();")
    assert "if (g == 0 && npu_parent_n >= 2 && npu_parent_g == (npu_parent_n - 1))" in code
    assert f"asc_sync_pipe(PIPE_{task.pipe});" in code
    assert code.index("parent_tail") < code.index("NATIVE();")


def test_normalized_sync_clones_do_not_supply_the_actual_case_metadata(checked):
    uid, leaf = next(
        (uid, leaf) for uid, leaf in checked.program.leaves.items() if leaf.stream.count.data == 2
    )
    sequence = next(iter(checked.analyses[uid][2].values()))
    syncs = [command for command in sequence if command.kind != "task"]
    assert syncs and all(isinstance(command.operation, LoopSyncOp) for command in syncs)
    assert all(command.operation.scope_cases is None for command in syncs)
    assert any(
        op.scope_cases is not None for op in leaf.stream.walk() if isinstance(op, LoopSyncOp)
    )
    render = scope_command_guard(leaf, sequence)
    assert any(parent_guard(render(command, "EFFECT();")) for command in syncs)
