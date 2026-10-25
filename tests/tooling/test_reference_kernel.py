"""Handwritten references keep their explicit ABI and propagate launch failures."""

from types import SimpleNamespace

import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.references import reference_kernel


@pytest.mark.parametrize("status", [0, 7])
def test_reference_binding_forwards_stream_tensors_sizes_and_checks_status(monkeypatch, status):
    calls = []

    def function(*arguments):
        calls.append(arguments)
        return status

    monkeypatch.setattr(
        "benchmarks.references.ct.CDLL", lambda _: SimpleNamespace(launch_reference=function)
    )
    launch = reference_kernel(
        SimpleNamespace(stream=99), "reference.so", "reference", 2, parameters=(3, 5)
    )
    if status:
        with pytest.raises(CompileError, match="reference launch failed"):
            launch(128, 256)
    else:
        launch(128, 256)
    assert calls == [(99, 128, 256, 3, 5)]
    with pytest.raises(CompileError, match="tensor count"):
        launch(128)
    assert len(calls) == 1
