"""Cancellation at real proof work, bounded diagnostics and context cleanup."""

import pytest

from examples.vector.scale import scale
from xdsl_npu import budget, compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError, ErrorKind


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
    source = N.from_source(
        """
        def finite(X: N.Tensor((64,), N.f32), Y: N.Tensor((64,), N.f32, access="write")):
            with N.Kernel():
                x=N.alloc_ub((64,))
                y=N.alloc_ub((64,))
                N.copy(X, x)
                N.mul(x, 2.0, out=y)
                N.copy(y, Y)
        """,
        disjoint_external=True,
    )
    assert compile(source, timeout_s=3).asc == compile(source, timeout_s=60).asc
