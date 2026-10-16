"""An actual native end is an anchor, without an invented retirement cut."""

import pytest

from examples.vector.scale import scale
from xdsl_npu.analysis.frontier import Clock, PrefixRank
from xdsl_npu.analysis.native import NativeCut
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def test_native_end_cannot_manufacture_a_retirement_capture():
    module = lower_native_structure(scale).module
    operation = next(op for op in module.walk() if isinstance(op, TaskOp))
    rank = PrefixRank(Clock("actual native"), 0, 0)
    for milestone in (Milestone.RETIREMENT_CAPTURE, Milestone.IMPLICIT_RETIREMENT):
        with pytest.raises(CompileError, match="cannot establish a retirement capture") as error:
            NativeCut(operation, Actor.AIV0, Engine.MTE2, rank, milestone)
        assert error.value.kind is ErrorKind.UNSUPPORTED
