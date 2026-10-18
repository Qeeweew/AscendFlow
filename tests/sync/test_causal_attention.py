"""Actual causal DSL bytes and finite event consumption across varying Q lengths."""

from pathlib import Path

import pytest

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.dialects.io import parse
from benchmarks.attention.contract import AttentionPolicy
from tests.models.structured_events import check_structured_events


@pytest.fixture(scope="module")
def causal_attention(tmp_path_factory):
    directory = tmp_path_factory.mktemp("causal-fa")
    source = Path(__file__).parents[2] / "examples/flash_attention/attention.py"
    policy = AttentionPolicy(causal=True, kv_tile=512, vector_rows=16)
    build_file(
        source, directory, entry="attention_target", parameters=policy.parameters(), timeout_s=300
    )
    return (
        collect_native_memory(parse((directory / "after.xdsl").read_text())),
        collect_native_memory(parse((directory / "before.xdsl").read_text())),
        (directory / "attention_target.asc").read_text(),
    )


@pytest.mark.parametrize(
    "sq,sk,groups", [(256, 1024, (0, 19)), (768, 3072, (0, 7, 19)), (2048, 1024, (0,))]
)
def test_causal_parent_lengths_and_slot_releases_match_actual_source(
    causal_attention, sq, sk, groups
):
    final, original, asc = causal_attention
    checked = check_structured_events(
        final, {"Sq": sq, "Sk": sk}, groups=groups, logical_reference=original, max_commands=100000
    )
    assert checked.bytes_read and checked.bytes_written and checked.consumptions
    assert asc.count("if ASC_IS_AIV") == 1
    assert "npu_sid ==" not in asc
    assert sum(t.contract.native == "mask_fill" for t in final.templates) == 1
    assert "PIPE_ALL" not in asc
