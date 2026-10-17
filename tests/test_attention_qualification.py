import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.attention.compare import qualification


def test_clear_slowdown_fails_and_overlap_is_not_called_success():
    assert qualification([200, 201, 202], [100, 101, 102])["status"] == "failed"
    assert qualification([100, 110, 120], [101, 102, 103])["status"] == "unresolved"
    assert qualification([99, 100, 101], [100, 101, 102])["status"] == "passed"


def test_single_batch_cannot_qualify_an_operator():
    with pytest.raises(CompileError, match="three paired"):
        qualification([90], [100])
