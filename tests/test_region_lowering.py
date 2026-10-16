"""Control admission must be independent of surrounding ordinary work."""

from dataclasses import replace

import pytest

from examples.state.grouped import grouped_state_sum
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump


def test_work_outside_nested_parent_is_refused_before_native_construction(monkeypatch):
    from xdsl_npu.lowering import _Actor

    def no_partial_lowering(*args, **kwargs):
        raise AssertionError("unsupported control reached native construction")

    monkeypatch.setattr(_Actor, "emit", no_partial_lowering)
    source = N.from_source("""
        def nested(n: N.int64):
            with N.Kernel(kind="vector"):
                x = N.alloc_ub((2, 64), N.f32)
                N.fill(x[0, :], 1.0)
                for i in N.Pipelined(n, stages=1):
                    for j in N.Pipelined(3, stages=1):
                        N.fill(x[j % 2, :], 0.0)
    """).to_ir()
    before = dump(source)
    with pytest.raises(CompileError, match="one parent Pipelined is required"):
        compile(source)
    assert dump(source) == before


def test_parent_admission_cannot_drop_work_outside_its_control():
    source = grouped_state_sum.source.replace(
        "for output in T.serial(outputs):",
        "T.fill(state, 0.0)\n        for output in T.serial(outputs):",
    )
    with pytest.raises(CompileError, match="work outside a serial sequence"):
        compile(replace(grouped_state_sum, source=source).specialize(B=16, GROUPS=1))
