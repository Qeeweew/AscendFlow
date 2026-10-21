"""Check buffered Gram lifetimes against the independent pre-sync native trace."""

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill import prefill_gram
from examples.kda.prefill_gram import prefill_gram_cross
from tests.models.structured_events import _Interpret, check_compiled_events


def varying_lengths(monkeypatch, chunks):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Lengths" and offset % 8 == 0 and 0 <= offset < chunks * 8
            values[operation.value] = 64 if (offset // 8) % 2 == 0 else 13

    monkeypatch.setattr(_Interpret, "native", metadata)


def test_buffered_gram_preserves_reads_through_full_chunks_tails_and_reentry(monkeypatch):
    varying_lengths(monkeypatch, 2)
    compiled = synchronize_structure(prefill_gram.specialize(CHUNKS=2, H=1, GROUPS=1))
    check_compiled_events(compiled, invocations=2, max_commands=10000)


def test_cross_gram_retains_quorum_counters_through_cohort_ring_reuse(monkeypatch):
    from ascendflow.analysis.event_contracts import SUPPORTED_CROSS_IDS

    varying_lengths(monkeypatch, 8)
    compiled = synchronize_structure(prefill_gram_cross.specialize(CHUNKS=8, H=1, PACK=2, GROUPS=1))
    assert 0 < compiled.resources.work.allocated_cross_ids <= len(SUPPORTED_CROSS_IDS)
    check_compiled_events(compiled, max_commands=40000)
