"""Typed reduction scratch, scalar source effects and expert issue policies."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

from ascendflow import compile
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.source import IterationOp, ScalarReadOp
from ascendflow.frontend import from_source
from examples.normalization.rms_norm import rms_compute, rms_normalize


@pytest.mark.parametrize(
    "labels", [(), ("load", "compute"), ("load", "compute", "store", "unknown")]
)
def test_expert_policy_cannot_silently_drop_stage_constraints(labels):
    source = rms_normalize.to_ir()
    pipeline = next(op for op in source.walk() if isinstance(op, IterationOp))
    pipeline.attributes["issue_order"] = DictionaryAttr(
        {"vector": ArrayAttr(StringAttr(label) for label in labels)}
    )
    with pytest.raises(CompileError, match="every participating Stage exactly once"):
        compile(source)


def test_reduction_scratch_alias_cannot_be_hidden_by_sync():
    helper = replace(
        rms_compute, source=rms_compute.source.replace("scratch=partial", "scratch=y[:64]")
    )
    assert helper.source != rms_compute.source
    source = from_source(
        rms_normalize.source, functions={"rms_compute": helper}, disjoint_external=True
    )
    with pytest.raises(CompileError, match="native read/write aliases"):
        compile(source)


def test_scalar_read_is_an_effectful_source_op():
    source = rms_normalize.to_ir()
    assert len([op for op in source.walk() if isinstance(op, ScalarReadOp)]) == 1
