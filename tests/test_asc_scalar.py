"""Standard-pass integration and independently executed emitted scalar code."""

import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tests.fixtures.pipelines import vector_pipeline
from xdsl_npu import autosync, emit_asc
from xdsl_npu.asc_scalar import ScalarScope
from xdsl_npu.events import selection_expression
from xdsl_npu.ir import dump
from xdsl_npu.loop_codegen import _event, _guard


def test_shared_stage_scope_executes_the_same_guarded_command_trace(tmp_path, monkeypatch):
    from examples.normalization.softmax import softmax
    from xdsl_npu import compile, loop_codegen
    from xdsl_npu.loop_sync import verify_loop

    compiler = shutil.which("c++")
    assert compiler is not None
    artifact = compile(softmax)
    _, (loop, channels, sequences, _, _) = verify_loop(artifact.plan, _details=True)
    actor, data = next(iter(loop.actors.items()))
    tasks = {(actor, task.uid): task for task in data.program.tasks}
    monkeypatch.setattr(
        loop_codegen,
        "native_line",
        lambda task, *args, **kwargs: f'printf("T {task.uid} %lld\\n", (long long)g);',
    )
    body = "\n".join(
        loop_codegen._region_lines(actor, data, sequences, channels, tasks, None, (), {})
    )
    assert body.count("const int64_t g =") == 3
    source = tmp_path / "commands.cpp"
    source.write_text(
        "#include <cstdio>\n#include <cstdint>\n"
        "enum {PIPE_V=1, PIPE_MTE2=2, PIPE_MTE3=3};\nusing event_t=int;\n"
        'void asc_sync_notify(int a,int b,int id) { printf("S %d %d %d\\n",a,b,id); }\n'
        'void asc_sync_wait(int a,int b,int id) { printf("W %d %d %d\\n",a,b,id); }\n'
        'void asc_sync_pipe(int p) { printf("B %d\\n",p); }\n'
        + 'int main() { for (int64_t n=0; n<=7; ++n) { printf("N %lld\\n",(long long)n);\n'
        + body
        + "\n} }\n"
    )
    executable = tmp_path / "commands"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=True,
        timeout=20,
    )
    actual = subprocess.check_output([str(executable)], text=True, timeout=5).splitlines()
    expected, pipes = [], {"V": 1, "MTE2": 2, "MTE3": 3}
    for count in range(8):
        expected.append(f"N {count}")
        for step in range(count + 2):
            for command in sequences[actor]:
                g = step - command.task.stage
                if not 0 <= g < count:
                    continue
                if not command.lower <= g < count - command.upper:
                    continue
                if g % command.period not in command.residues:
                    continue
                if command.last_gap is not None and g != count - 1 - command.last_gap:
                    continue
                if command.kind == "task":
                    expected.append(f"T {command.task.uid} {g}")
                elif command.kind == "barrier":
                    expected.append(f"B {pipes[command.task.pipe]}")
                else:
                    channel = channels[command.channel]
                    epoch = g if command.kind == "set" else g - channel.distance
                    # Independent small notification enumeration, not the emitter.
                    events = [i for i in range(epoch + 1) if i % channel.period in channel.residues]
                    event = channel.ids[(len(events) - 1) % len(channel.ids)]
                    a, b = (
                        pipes[tasks[end].pipe]
                        for end in (channel.sources[0], channel.destinations[0])
                    )
                    expected.append(f"{'S' if command.kind == 'set' else 'W'} {a} {b} {event}")
    assert actual == expected


def test_standard_passes_fold_identities_and_share_real_ssa_nodes():
    scope = ScalarScope()
    a = scope.div(scope.sub(scope.add("g", 0), 0), 1)
    b = scope.mul("g", 1)
    left, right = scope.rem(a, 2), scope.rem(b, 2)
    code = "\n".join(scope.finish([f"consume({scope.use(left)}, {scope.use(right)});"]))
    assert " - 0" not in code and " / 1" not in code and " * 1" not in code
    assert code.count("g % 2") == 1
    assert "consume(npu_scalar_0, npu_scalar_0)" in code


def test_scalar_optimization_keeps_async_commands_and_the_verified_ir(monkeypatch):
    from xdsl_npu.asc_scalar import CanonicalizePass, CommonSubexpressionElimination

    invocations = []
    for cls in (CanonicalizePass, CommonSubexpressionElimination):
        original = cls.apply

        def inspect(self, ctx, module, original=original):
            module.verify()
            assert all(op.name.startswith(("builtin.", "func.", "arith.")) for op in module.walk())
            invocations.append(self.name)
            original(self, ctx, module)
            module.verify()

        monkeypatch.setattr(cls, "apply", inspect)
    plan = autosync(vector_pipeline(4))
    before = dump(plan)
    optimized = emit_asc(plan)
    assert dump(plan) == before
    assert set(invocations) == {"canonicalize", "cse"}
    for cls in (CanonicalizePass, CommonSubexpressionElimination):
        monkeypatch.setattr(cls, "apply", lambda *args: None)
    raw = emit_asc(plan)
    # Every native and sync call remains in the same textual dispatch order.
    calls = r"\basc_\w+\("
    assert re.findall(calls, optimized) == re.findall(calls, raw)
    assert optimized != raw


def test_emitted_event_and_address_expressions_execute_exactly(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required to execute generated expressions")
    ids = (0, 1) * 20 + (0, 2, 1, 3) * 6 + (0, 1) * 4 + (1, 3, 0, 2) * 14
    scope = ScalarScope()
    event = scope.use(selection_expression(ids, scope.sub("g", 0), scalar=scope))
    addr = scope.use(
        scope.add(
            scope.mul(scope.rem(scope.div("g", 8), 2), 262144), scope.mul(scope.rem("g", 8), 16384)
        )
    )
    finite = "\n".join(
        scope.finish(
            [
                f"assert({event} == expected[g]);",
                f"assert({addr} == (g / 8 % 2) * 262144 + (g % 8) * 16384);",
            ]
        )
    )
    scope = ScalarScope()
    channel = SimpleNamespace(period=3, residues=(0, 2), ids=(2, 5))
    event = scope.use(_event(channel, scope.value("g"), scope))
    periodic = "\n".join(scope.finish([f"assert({event} == (g % 3 == 0 ? 2 : 5));"]))
    scope = ScalarScope()
    remainder = scope.rem(scope.sub("g", 2), 3)
    # Exercise signed division/remainder and a collision with an external name.
    a, b = scope.use(remainder), scope.use(remainder)
    signed = "\n".join(
        scope.finish(
            [f"assert({a} == (g - 2) % 3);", f"assert({b} + npu_scalar_0 == (g - 2) % 3 + 7);"]
        )
    )
    source = tmp_path / "scalars.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\nint main() {\n"
        f"const int expected[] = {{{', '.join(map(str, ids))}}};\n"
        f"for (int64_t g = 0; g < {len(ids)}; ++g) {{ {finite} }}\n"
        f"for (int64_t g = 0; g < 300; ++g) if (g % 3 != 1) {{ {periodic} }}\n"
        f"for (int64_t g = -64; g < 64; ++g) {{ const int64_t npu_scalar_0 = 7; {signed} }}\n"
        "}\n"
    )
    executable = tmp_path / "scalars"
    subprocess.run(
        [compiler, "-std=c++17", "-O0", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)


def test_runtime_event_id_progressions_execute_under_source_and_wait_guards(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required to execute generated expressions")
    # Tuple fields are assigned IDs, protocol period/residues, and wait distance.
    # The ID assignments are inputs to _event, never reallocated by emission.
    cases = [
        ((7,), 1, (0,), 0, True),
        ((0, 1), 1, (0,), 0, True),
        ((1, 3, 5, 7), 1, (0,), 3, True),
        ((7, 5, 3, 1), 1, (0,), 4, True),
        ((-5, -2, 1, 4), 1, (0,), 2, True),
        ((4, 4, 4), 1, (0,), 1, True),
        ((0, 4, 1, 7), 1, (0,), 5, False),
        ((2, 4, 6), 5, (0, 2, 4), 0, True),
        ((7, 4, 1), 3, (0, 2), 5, True),
        ((7, 0, 5, 1), 5, (1, 4), 3, False),
        ((3,), 7, (2, 6), 9, True),
    ]
    functions, checks = [], []
    limit, inactive = 97, -1000000
    for number, (ids, period, residues, distance, affine) in enumerate(cases):
        channel = SimpleNamespace(period=period, residues=residues, ids=ids, carry=False)
        scope = ScalarScope()
        epoch = scope.sub("g", distance) if distance else scope.value("g")
        event = scope.use(_event(channel, epoch, scope))
        body = "\n".join(scope.finish([f"return {event};"]))
        if affine:
            # A periodic ordinal may still contain its original rank selects.
            # The arithmetic ID mapping contributes no equality/ternary chain.
            assert " == " not in body
            if period == 1 or len(ids) == 1:
                assert "?" not in body
            else:
                assert body.count("?") <= len(residues)
        elif period == 1:
            assert "?" in body
        functions.append(f"int64_t event_{number}(int64_t g) {{\n{body}\n}}\n")
        command = SimpleNamespace(
            last_gap=None,
            lower=distance,
            upper=0,
            period=period,
            residues=tuple(sorted((residue + distance) % period for residue in residues)),
        )
        guard = _guard(command, channel) or "true"
        expected = [inactive] * limit
        # Independent occurrence oracle: list only legal source epochs, then
        # cycle the already assigned IDs. No quotient/rank formula is reused.
        active_epochs = (epoch for epoch in range(limit) if epoch % period in residues)
        for occurrence, epoch in enumerate(active_epochs):
            wait_generation = epoch + distance
            if wait_generation < limit:
                expected[wait_generation] = ids[occurrence % len(ids)]
        checks.append(
            "{\n"
            f"  const int64_t expected[] = {{{', '.join(map(str, expected))}}};\n"
            f"  for (int64_t g = 0; g < {limit}; ++g) {{\n"
            f"    const bool active = {guard};\n"
            f"    assert(active == (expected[g] != {inactive}));\n"
            f"    if (active) assert(event_{number}(g) == expected[g]);\n"
            "  }\n}\n"
        )
    source, executable = tmp_path / "runtime_events.cpp", tmp_path / "runtime_events"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n"
        + "".join(functions)
        + "int main() {\n"
        + "".join(checks)
        + "}\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)
