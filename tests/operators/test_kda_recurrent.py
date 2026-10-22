"""Token prefetch must preserve the carry read by every snapshot export."""

import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from examples.kda.recurrent import recurrent_kda
from tests.models.structured_events import _Interpret, check_structured_events


@pytest.mark.parametrize("inplace", (True, False))
@pytest.mark.parametrize("heads", (1, 16))
def test_token_pipeline_keeps_each_carry_until_its_snapshot_retires(monkeypatch, inplace, heads):
    # Empty, single-token, two-token and steady-state loops share one compiled
    # program. Distinct snapshot destinations expose a one-token shift even
    # when a caller would normally overwrite the same state slot each step.
    boundaries = (0, 0, 1, 3, 7)
    indices = ((0, 0, 0, 0), (1, 0, 0, 0), (2, 3, 0, 0), (4, 5, 6, 7))
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            if root.sym_name.data == "Cu":
                values[operation.value] = boundaries[offset // 8]
            elif root.sym_name.data == "Indices":
                values[operation.value] = indices[offset // 32][offset % 32 // 8]
            else:
                raise AssertionError(root.sym_name.data)

    monkeypatch.setattr(_Interpret, "native", metadata)
    result = compile(
        recurrent_kda.specialize(
            TOKENS=7,
            SEQUENCES=4,
            SLOTS=8,
            INDEX_WIDTH=4,
            H=heads,
            HV=heads,
            K=64,
            V=32,
            BV=16,
            INPLACE=inplace,
        ),
        debug_verify=True,
    )
    trace = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        groups=(0, heads, 2 * heads, 3 * heads),
        invocations=2,
        max_commands=20000,
    )
    destination = "State" if inplace else "Snapshots"
    exports = [
        command
        for command in trace.natives
        if any(
            span.root.sym_name.data == destination and span.mode == "W" for span in command.spans
        )
    ]
    assert len(exports) == 14
