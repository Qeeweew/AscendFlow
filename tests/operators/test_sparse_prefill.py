"""Sparse prefill row selection and circular-window lifetime contracts."""

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from examples.sparse_attention.attention import sparse_mla
from tests.models.structured_events import check_structured_events


def test_prefill_window_ring_retires_across_wraps_and_invocations(monkeypatch):
    from tests.operators.test_sparse_persistent import bind_metadata, metadata_inputs

    data = metadata_inputs(1, 160)
    data["seqused_ori_kv"][:] = 4095
    data["seqused_cmp_kv"][:] = 2047
    data["cmp_residual_kv"][:] = 1
    data["ori_topk_length"][:] = 0
    # A narrowed query forces a full visible reload on its next worker query.
    data["ori_topk_length"][40] = 7
    bind_metadata(monkeypatch, data)
    result = compile(sparse_mla.specialize(B=1, T=160), timeout_s=300)
    trace = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
        max_commands=200000,
    )
    assert trace.bytes_read and trace.bytes_written and trace.consumptions
