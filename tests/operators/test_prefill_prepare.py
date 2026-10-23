"""Fused Prepare retains byte visibility across AIV peers and scratch reuse."""

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill_prepare import prefill_prepare
from tests.models.structured_events import _Interpret, check_compiled_events


def test_prepare_orders_peer_rows_full_chunk_reuse_tails_and_empty_owners(monkeypatch):
    native = _Interpret.native
    chunks = 4

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert offset % 8 == 0 and 0 <= offset < chunks * 8
            name = root.sym_name.data
            assert name in {"Starts", "Lengths"}
            chunk = offset // 8
            if name == "Starts":
                values[operation.value] = chunk * 64
            else:
                values[operation.value] = 13 if chunk == chunks - 1 and invocation == 0 else 64

    monkeypatch.setattr(_Interpret, "native", metadata)
    plan = synchronize_structure(prefill_prepare.specialize(CHUNKS=chunks, NC=chunks, H=8))
    # Group 0 reuses full-chunk scratch, group 19 has no last-chunk work.
    # Repeat the actual instruction streams without resetting their events.
    execution = check_compiled_events(
        plan, {"TOKENS": chunks * 64}, groups=(0, 19), invocations=2, max_commands=80000
    )
    assert execution.bytes_read and execution.bytes_written
