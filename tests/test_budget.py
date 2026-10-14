"""Cancellation at real proof work, bounded diagnostics and context cleanup."""

import pytest
from ast_vector import scale

from xdsl_npu import budget, compile
from xdsl_npu.loop_graph import Graph
from xdsl_npu.model import CompileError


def test_cancellation_during_completion_query_leaves_input_usable(monkeypatch):
    stopped = False
    original = Graph.path

    def stop(self, *args, **kwargs):
        nonlocal stopped
        stopped = True
        return original(self, *args, **kwargs)

    source = scale.to_ir()
    monkeypatch.setattr(Graph, "path", stop)
    with pytest.raises(CompileError, match=r"cancelled at completion path.*queries="):
        compile(source, cancel=lambda: stopped)
    # The cancelled context and its partly mutated owned clone do not leak.
    assert compile(source).report["native_templates"] == 3


def test_deadline_includes_phase_and_structure(monkeypatch):
    times = iter([0.0, 0.0, 0.25])
    monkeypatch.setattr(budget, "perf_counter", lambda: next(times))
    with (
        pytest.raises(CompileError, match=r"deadline exceeded at physical accesses.*pieces=19"),
        budget.limit(0.1),
    ):
        budget.poll("physical accesses", pieces=19)
    assert budget._active.get() is None


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_policy_is_rejected(seconds):
    with pytest.raises(CompileError, match="timeout_s"):
        compile(scale, timeout_s=seconds)


def test_deadline_does_not_change_successful_generated_code():
    assert compile(scale, timeout_s=3).asc == compile(scale, timeout_s=60).asc
