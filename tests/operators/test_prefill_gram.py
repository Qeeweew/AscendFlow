"""Check buffered Gram lifetimes against the independent pre-sync native trace."""

from ascendflow.passes.structured_sync import synchronize_structure
from examples.kda.prefill import prefill_gram
from tests.models.structured_events import _Interpret, check_compiled_events


def test_buffered_gram_preserves_reads_through_full_chunks_tails_and_reentry(monkeypatch):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Lengths" and offset in (0, 8)
            values[operation.value] = 64 if offset == 0 else 13

    monkeypatch.setattr(_Interpret, "native", metadata)
    compiled = synchronize_structure(prefill_gram.specialize(CHUNKS=2, H=1, GROUPS=1))
    check_compiled_events(compiled, invocations=2, max_commands=10000)
