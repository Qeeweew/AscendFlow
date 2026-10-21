"""Cross-request state and scratch retirement in a resident CSA worker."""

from dataclasses import replace

import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from examples.sparse_attention.decode import sparse_mla_decode
from tests.models.structured_events import EventModelError, check_structured_events


def specialize(definition):
    return definition.specialize(B=4, H=32, BM=32, D=128, GROUPS=1, SCALE=128**-0.5)


@pytest.fixture(scope="module")
def resident_artifact():
    return compile(specialize(sparse_mla_decode), timeout_s=300)


def bind_metadata(monkeypatch, counts=512):
    from tests.models.structured_events import _Interpret

    original = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            name = root.sym_name.data
            values[operation.value] = {
                "Lengths": (255, 383, 511, 4095)[offset // 4] if name == "Lengths" else 0,
                "WindowLengths": 128,
                "Counts": counts,
                "Indices": (offset // 4) % (256 if counts == 319 else 64),
                "OriTable": (offset // 4) % 128,
                "CmpTable": (offset // 4) % 128,
            }[name]

    monkeypatch.setattr(_Interpret, "native", metadata)


def check_trace(result):
    return check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        max_commands=40000,
    )


@pytest.mark.parametrize("counts", (512, 319, 0))
def test_resident_worker_retires_old_request_before_reset_and_reuses_gm_slots(
    monkeypatch, resident_artifact, counts
):
    bind_metadata(monkeypatch, counts)
    trace = check_trace(resident_artifact)
    output = [
        s
        for command in trace.natives
        for s in command.spans
        if s.root.sym_name.data == "O" and "W" in s.mode
    ]
    assert sum(span.end - span.start for span in output) == 4 * 32 * 128 * 2


def test_new_softmax_cannot_overwrite_previous_request_denominator():
    source = sparse_mla_decode.source.replace("final_l[slot, local : local + UR]", "row_sum[0:UR]")
    assert source != sparse_mla_decode.source
    with pytest.raises(CompileError, match="issued|ownership|same-engine|queue"):
        compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)


def test_two_cache_slots_do_not_retire_the_other_aiv_mask_reader(monkeypatch):
    source = sparse_mla_decode.source.replace("3, S - WINDOW", "2, S - WINDOW").replace(
        "cache_slot = query % 3", "cache_slot = query % 2"
    )
    result = compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)
    bind_metadata(monkeypatch)
    with pytest.raises(EventModelError, match="unordered physical WAR: Bias"):
        check_trace(result)


def test_mask_read_before_cube_score_acquire_has_no_peer_publication(monkeypatch):
    source = sparse_mla_decode.source
    start = source.index("                # Acquire the complete Cube score")
    middle = source.index("                N.fill(bias, NEG_INF)", start)
    end = source.index("                for block in N.serial", middle)
    source = source[:start] + source[middle:end] + source[start:middle] + source[end:]
    result = compile(specialize(replace(sparse_mla_decode, source=source)), timeout_s=300)
    bind_metadata(monkeypatch)
    with pytest.raises(EventModelError, match="unordered physical RAW: Bias"):
        check_trace(result)
