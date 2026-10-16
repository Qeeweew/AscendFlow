"""Typed reduction scratch, scalar source effects and expert issue policies."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

from examples.normalization.rms_norm import rms_normalize
from xdsl_npu import compile
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.source import PipelineOp, ScalarReadOp
from xdsl_npu.frontend import from_source


@pytest.mark.parametrize(
    "labels", [(), ("load", "compute"), ("load", "compute", "store", "unknown")]
)
def test_expert_policy_cannot_silently_drop_stage_constraints(labels):
    source = rms_normalize.to_ir()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    pipeline.attributes["issue_order"] = DictionaryAttr(
        {"vector": ArrayAttr(StringAttr(label) for label in labels)}
    )
    with pytest.raises(CompileError, match="every active Stage"):
        compile(source)


def test_reduction_scratch_alias_cannot_be_hidden_by_sync():
    source = rms_normalize.source.replace("scratch=partial", "scratch=y[slot, :64]")
    with pytest.raises(CompileError, match="shifted in-place Vector alias"):
        compile(from_source(source, disjoint_external=True))


def test_scalar_read_is_an_effectful_source_op():
    source = rms_normalize.to_ir()
    assert len([op for op in source.walk() if isinstance(op, ScalarReadOp)]) == 1
