"""Explicit Cube operand syntax and typed storage contracts."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException

from examples.gemm.batched import batched_matmul
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.source import CopyOp, MatmulOp, PipelineOp
from xdsl_npu.dialects.source import ViewOp as SourceViewOp
from xdsl_npu.frontend import FrontendError


def test_cube_bf16_uses_the_same_explicit_storage_semantics():
    result = compile(batched_matmul.specialize(DTYPE=N.bf16, TRANSPOSE_B=False))
    assert "__cbuf__ bfloat16_t" in result.asc
    assert result.report["native_templates"] == 6


def test_ast_rejects_hidden_l0_recipe_and_runtime_conditional_expressions():
    source = batched_matmul.source.replace("T.matmul(aa, bb, accum=", "T.gemm(aa, bb, ")
    with pytest.raises(FrontendError, match="unknown DSL operation gemm"):
        replace(batched_matmul, source=source).to_ir()
    source = batched_matmul.source.replace("N if TRANSPOSE_B else K", "N if tiles > 0 else K")
    with pytest.raises(FrontendError, match="constexpr boolean"):
        replace(batched_matmul, source=source).to_ir()


def test_transpose_flag_does_not_license_other_dma_routes():
    source = batched_matmul.to_ir()
    copy = next(op for op in source.walk() if isinstance(op, CopyOp))
    copy.attributes["transpose"] = IntAttr(1)
    with pytest.raises(VerifyException, match="L1→L0B"):
        source.verify()


def test_first_mad_cannot_accumulate_uninitialized_local_storage():
    source = batched_matmul.source.replace("initialize=True", "initialize=False")
    with pytest.raises(CompileError, match="undefined first read"):
        compile(replace(batched_matmul, source=source))


@pytest.mark.parametrize("sequence", [False, True])
def test_serialized_ast_cannot_bypass_explicit_operand_contract(sequence):
    source = batched_matmul.to_ir()
    if sequence:
        pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
        pipeline.parent.insert_op_after(pipeline.clone(), pipeline)
    matmul = next(op for op in source.walk() if isinstance(op, MatmulOp))
    a1 = next(
        op.ref
        for op in source.walk()
        if isinstance(op, SourceViewOp) and op.source.owner.sym_name.data == "a1"
    )
    b1 = next(
        op.ref
        for op in source.walk()
        if isinstance(op, SourceViewOp) and op.source.owner.sym_name.data == "b1"
    )
    matmul.operands = (a1, b1, *matmul.operands[2:])
    with pytest.raises(VerifyException, match="explicit L0A/L0B/L0C"):
        source.verify()
