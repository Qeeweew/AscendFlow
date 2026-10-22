"""Eviction keeps the checksum live while its DMA input slots wrap."""

import pytest

from ascendflow.passes.structured_sync import synchronize_structure
from benchmarks.cache_sweep import cache_sweep
from tests.models.structured_events import check_compiled_events


@pytest.fixture(scope="module")
def sweep():
    return synchronize_structure(cache_sweep.specialize(B=16, GROUPS=2))


@pytest.mark.parametrize("tiles", [0, 1, 7])
def test_checksum_export_waits_for_the_complete_read_stream(sweep, tiles):
    check_compiled_events(sweep, {"tiles": tiles}, groups=(0, 1), invocations=2)
