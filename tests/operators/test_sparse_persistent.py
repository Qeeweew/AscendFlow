"""Cross-token/request retirement and actual paged addresses in resident CSA."""

from dataclasses import replace

import numpy as np
import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from benchmarks.attention.sparse_reference import query_geometry
from examples.sparse_attention.decode import sparse_mla_decode
from tests.models.structured_events import EventModelError, check_structured_events


def specialize(definition):
    return definition.specialize(B=5, T=7, H=32, BM=32, D=128, GROUPS=1, SCALE=128**-0.5)


@pytest.fixture(scope="module")
def resident_artifact():
    return compile(specialize(sparse_mla_decode), timeout_s=300)


def metadata_inputs(batch=5, tokens=7, topk=512, valid=512):
    from benchmarks.attention.sparse_reference import query_counts

    counts = query_counts({"B": batch, "T": tokens})
    if (batch, tokens) == (5, 7):
        counts = np.array([0, 2, 0, 1, 4], dtype=np.int32)
    lengths = np.resize(np.array([255, 383, 511, 4095], dtype=np.int32), batch)
    indices = np.full((tokens, 1, topk), -1, dtype=np.int32)
    indices[:, :, :valid] = np.arange(min(valid, topk)) % 64
    return {
        "cu_seqlens_q": np.r_[0, counts.cumsum()].astype(np.int32),
        "seqused_ori_kv": lengths,
        "seqused_cmp_kv": lengths // 2,
        "cmp_residual_kv": lengths % 2,
        "ori_topk_length": np.full((tokens, 1), 64, dtype=np.int32),
        "cmp_sparse_indices": indices,
        "ori_block_table": np.arange(batch * 32, dtype=np.int32).reshape(batch, 32) % 128,
        "cmp_block_table": np.arange(batch * 32, dtype=np.int32).reshape(batch, 32) % 128,
    }


def bind_metadata(monkeypatch, data):
    from tests.models.structured_events import _Interpret

    original = _Interpret.native
    contents = {}

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data in {"copy", "copy_bytes"}:
            source, offset, _ = values[operation.refs[0]]
            target, destination, size = values[operation.refs[1]]
            if source.sym_name.data in data and target.ref.type.bank.data == "UB":
                key = (self, invocation, group, actor, target)
                width = self.templates[operation].accesses[0].address.geometry.total_nbytes
                raw = data[source.sym_name.data].tobytes()
                contents.setdefault(key, bytearray(size))[destination : destination + width] = raw[
                    offset : offset + width
                ]
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            if root.ref.type.bank.data == "UB":
                key = (self, invocation, group, actor, root)
                width = self.templates[operation].accesses[0].address.geometry.nbytes
                values[operation.value] = int.from_bytes(
                    contents[key][offset : offset + width], "little", signed=True
                )
            else:
                array = data[root.sym_name.data]
                values[operation.value] = int(array.reshape(-1)[offset // array.itemsize])

    monkeypatch.setattr(_Interpret, "native", metadata)


def check_trace(result):
    return check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        max_commands=100000,
    )


@pytest.mark.parametrize("valid", (512, 319, 0))
def test_resident_worker_retires_old_token_across_ragged_and_empty_requests(
    monkeypatch, resident_artifact, valid
):
    data = metadata_inputs(valid=valid)
    bind_metadata(monkeypatch, data)
    trace = check_trace(resident_artifact)
    output = [
        span
        for command in trace.natives
        for span in command.spans
        if span.root.sym_name.data == "attention_out" and "W" in span.mode
    ]
    assert sum(span.end - span.start for span in output) == 7 * 32 * 128 * 2
    assert sorted((s.start, s.end) for s in output) == [
        (token * 8192 + row * 256, token * 8192 + (row + 8) * 256)
        for token in range(7)
        for row in (0, 8, 16, 24)
    ]
    # Check real GM accesses independently of the DSL's binary request search.
    for command in trace.natives:
        for span in command.spans:
            if span.root.sym_name.data not in {"ori_kv", "cmp_kv"}:
                continue
            token = command.occurrence.coordinates[0][1]
            batch, right, visible, available, cutoff = query_geometry(
                data, {"WINDOW": 128, "RATIO": 2}, token
            )
            if span.root.sym_name.data == "ori_kv":
                allowed = {
                    int(data["ori_block_table"][batch, p // 128]) * 128 + p % 128
                    for p in range(right - visible, right)
                }
            else:
                allowed = {
                    int(data["cmp_block_table"][batch, i // 64]) * 64 + i % 64
                    for i in data["cmp_sparse_indices"][token, 0, : min(512, available, cutoff)]
                    if 0 <= i < cutoff
                }
            assert set(range(span.start // 256, (span.end - 1) // 256 + 1)) <= allowed


def test_new_softmax_cannot_overwrite_previous_token_denominator():
    source = sparse_mla_decode.source.replace("final_l[slot, local : local + UR]", "row_sum[0:UR]")
    assert source != sparse_mla_decode.source
    with pytest.raises(CompileError, match="issued|ownership|same-engine|queue"):
        compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)


@pytest.mark.parametrize("gather_first", (False, True))
def test_two_cache_slots_require_prior_score_acquire_before_gather(monkeypatch, gather_first):
    source = sparse_mla_decode.source.replace("3, S - WINDOW", "2, S - WINDOW").replace(
        "cache_slot = query % 3", "cache_slot = query % 2"
    )
    if not gather_first:
        source = source.replace(
            '"vector": ("gather", "softmax", "finish")',
            '"vector": ("softmax", "gather", "finish")',
        )
    result = compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)
    bind_metadata(monkeypatch, metadata_inputs())
    if gather_first:
        with pytest.raises(EventModelError, match="unordered physical WAR: Bias"):
            check_trace(result)
    else:
        check_trace(result)


def test_mask_read_before_cube_score_acquire_has_no_peer_publication(monkeypatch):
    source = sparse_mla_decode.source
    start = source.index("                # Acquire the complete compressed score")
    middle = source.index("                N.fill(window_bias, 0.0)", start)
    end = source.index("                for block in N.serial", middle)
    source = source[:start] + source[middle:end] + source[start:middle] + source[end:]
    result = compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)
    bind_metadata(monkeypatch, metadata_inputs())
    with pytest.raises(EventModelError, match="unordered physical RAW: Bias"):
        check_trace(result)
