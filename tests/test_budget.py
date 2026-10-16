"""Cancellation at real proof work, bounded diagnostics and context cleanup."""

import pytest

from examples.vector.scale import scale
from xdsl_npu import budget, compile
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.loop_planning import DirectCompletion


def test_cancellation_during_direct_synthesis_leaves_input_usable(monkeypatch):
    stopped = False
    original = DirectCompletion.insert

    def stop(self, *args, **kwargs):
        nonlocal stopped
        stopped = bool(self.selected)
        return original(self, *args, **kwargs)

    source = scale.to_ir()
    monkeypatch.setattr(DirectCompletion, "insert", stop)
    with pytest.raises(
        CompileError, match=r"cancelled at direct protocol insertion.*selected=1"
    ) as caught:
        compile(source, cancel=lambda: stopped)
    assert caught.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert budget._active.get() is None
    # The cancelled context and its partly mutated owned clone do not leak.
    assert compile(source).report["native_templates"] == 3


def test_deadline_includes_phase_and_structure(monkeypatch):
    times = iter([0.0, 0.0, 0.25])
    monkeypatch.setattr(budget, "perf_counter", lambda: next(times))
    with (
        pytest.raises(
            CompileError, match=r"deadline exceeded at physical accesses.*pieces=19"
        ) as caught,
        budget.limit(0.1),
    ):
        budget.poll("physical accesses", pieces=19)
    assert caught.value.kind is ErrorKind.ANALYSIS_LIMIT
    assert str(caught.value) == "compilation deadline exceeded at physical accesses (pieces=19)"
    assert budget._active.get() is None


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_policy_is_rejected(seconds):
    with pytest.raises(CompileError, match="timeout_s") as caught:
        compile(scale, timeout_s=seconds)
    assert caught.value.kind is ErrorKind.INVALID


def test_deadline_does_not_change_successful_generated_code():
    assert compile(scale, timeout_s=3).asc == compile(scale, timeout_s=60).asc
