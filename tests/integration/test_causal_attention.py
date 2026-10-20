"""Actual causal DSL bytes and finite event consumption across varying Q lengths."""

from collections import Counter
from pathlib import Path

import pytest

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.dialects.io import parse
from benchmarks.attention.contract import AttentionPolicy
from tests.models.structured_events import check_structured_events


@pytest.fixture(
    scope="module", params=[(128, 256, 16), (256, 256, 16), (256, 512, 16), (256, 1024, 8)]
)
def causal_attention(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("causal-fa")
    source = Path(__file__).parents[2] / "examples/flash_attention/attention.py"
    query_tile, kv_tile, vector_rows = request.param
    policy = AttentionPolicy(
        causal=True, query_tile=query_tile, kv_tile=kv_tile, vector_rows=vector_rows
    )
    build_file(
        source, directory, entry="attention_target", parameters=policy.parameters(), timeout_s=300
    )
    return (
        collect_native_memory(parse((directory / "after.xdsl").read_text())),
        collect_native_memory(parse((directory / "before.xdsl").read_text())),
        (directory / "attention_target.asc").read_text(),
        policy,
    )


@pytest.mark.parametrize(
    "sq,sk,groups", [(256, 1024, (0, 19)), (768, 3072, (0, 7, 19)), (2048, 1024, (0,))]
)
def test_causal_parent_lengths_and_slot_releases_match_actual_source(
    causal_attention, sq, sk, groups
):
    final, original, asc, policy = causal_attention
    checked = check_structured_events(
        final, {"Sq": sq, "Sk": sk}, groups=groups, logical_reference=original, max_commands=100000
    )
    assert checked.bytes_read and checked.bytes_written and checked.consumptions
    if (sq, sk) == (256, 1024):
        # Group zero owns the first query block; group 19 is empty here.
        # QK evaluates one BK tile and PV reads only the BQ visible columns.
        loads = Counter()
        for command in checked.natives:
            for span in command.spans:
                if span.root.sym_name.data in {"K", "V"}:
                    loads[span.root.sym_name.data] += span.end - span.start
        assert loads == {"K": policy.kv_tile * 128 * 2, "V": policy.query_tile * 128 * 2}
    assert asc.count("if ASC_IS_AIV") == 1
    # Boundary notifications may have participant guards; the native body
    # stays shared and the finite execution checks both AIV contributions.
    assert sum(t.spec.native == "mask_fill" for t in final.templates) == 1
    assert "PIPE_ALL" not in asc
