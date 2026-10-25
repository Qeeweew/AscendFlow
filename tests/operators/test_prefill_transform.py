"""Whole-chunk transform prefetches retire raw readers before reuse."""

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill_transform import prefill_transform
from tests.models.structured_events import _Interpret, check_compiled_events


def test_transform_preserves_raw_readers_and_repeated_output_publication(monkeypatch):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Lengths" and offset in (0, 8)
            values[operation.value] = (1, 13, 63, 64)[invocation]

    monkeypatch.setattr(_Interpret, "native", metadata)
    plan = synchronize_structure(prefill_transform.specialize(CHUNKS=2, H=1, GROUPS=1))
    checked = check_compiled_events(plan, invocations=4, max_commands=10000)
    assert checked.bytes_read and checked.bytes_written
