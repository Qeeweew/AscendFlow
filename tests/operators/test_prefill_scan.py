"""Grouped scan prefetch retains FP32 carry and every slot's last reader."""

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill_scan import prefill_scan
from tests.models.structured_events import _Interpret, check_compiled_events


def test_scan_grouped_prefetch_reuses_both_slots_across_sequences(monkeypatch):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Lengths" and offset % 8 == 0
            chunk = offset // 8
            assert 0 <= chunk < 6
            values[operation.value] = 13 if chunk % 3 == 2 and invocation == 0 else 64

    monkeypatch.setattr(_Interpret, "native", metadata)
    plan = synchronize_structure(
        prefill_scan.specialize(CHUNKS=6, SEQUENCES=2, NC=3, H=1, GROUPS=1)
    )
    checked = check_compiled_events(plan, invocations=2, max_commands=30000)
    assert checked.bytes_read and checked.bytes_written and checked.consumptions
