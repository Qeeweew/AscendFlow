"""Fused Prepare preserves peer visibility and overlapped workspace retirement."""

import pytest

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill_prepare import prefill_prepare
from tests.models.structured_events import _Interpret, check_compiled_events


@pytest.mark.parametrize("sequences,nc", [(2, 3), (1, 1)])
def test_prepare_orders_peer_rows_reuse_tails_and_empty_owners(monkeypatch, sequences, nc):
    native = _Interpret.native
    chunks = sequences * nc

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
                values[operation.value] = (
                    (13, 16, 63, 64)[invocation] if chunk % nc == nc - 1 else 64
                )

    monkeypatch.setattr(_Interpret, "native", metadata)
    plan = synchronize_structure(
        prefill_prepare.specialize(CHUNKS=chunks, SEQUENCES=sequences, NC=nc, H=8)
    )
    # Three tasks exercise both W planes and their first reuse across sequence
    # boundaries. NC=1 covers a tail-only invocation and a completely idle owner.
    # Repeat the actual instruction streams without resetting their events.
    execution = check_compiled_events(
        plan, {"TOKENS": chunks * 64}, groups=(0, 19), invocations=4, max_commands=160000
    )
    assert execution.bytes_read and execution.bytes_written


def test_prepare_production_domain_has_a_finite_event_assignment():
    from ascendflow.analysis.event_contracts import SUPPORTED_CROSS_IDS

    # Long task streams exercise reentry/resource facts absent from NC=1/3.
    # Keep this as an IR check; the finite byte model above covers actual events.
    plan = synchronize_structure(prefill_prepare.specialize(CHUNKS=64, SEQUENCES=1, NC=64, H=32))
    assert 0 < plan.resources.work.allocated_cross_ids <= len(SUPPORTED_CROSS_IDS)
