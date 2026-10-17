"""Actual default FA structure, one symbolic public source build.

These finite byte/event checks neither execute floating point arithmetic nor
claim CANN/device qualification. No constexpr policy or runtime length is
specialized for this regression.
"""

from pathlib import Path

import pytest

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


@pytest.fixture(scope="module")
def default_attention(tmp_path_factory):
    directory = tmp_path_factory.mktemp("default-fa")
    source = Path(__file__).parents[2] / "examples/flash_attention/attention.py"
    report = build_file(source, directory, entry="attention_target", timeout_s=300)
    assert report["native_templates"] == 44
    final = collect_native_memory(parse((directory / "after.xdsl").read_text()))
    original = collect_native_memory(parse((directory / "before.xdsl").read_text()))
    return final, original


@pytest.mark.parametrize("sq,sk", [(128, 1024), (128, 2048), (128, 3072), (512, 2048)])
def test_default_attention_native_versions_and_real_event_consumption(default_attention, sq, sk):
    final, original = default_attention
    execution = check_structured_events(
        final,
        {"Sq": sq, "Sk": sk},
        logical_reference=original,
        groups=(0, 19),
        max_commands=20000,
    )
    assert execution.natives and execution.consumptions
    assert execution.bytes_read and execution.bytes_written
