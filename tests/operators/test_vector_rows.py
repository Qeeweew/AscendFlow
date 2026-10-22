"""Typed FA Vector building blocks through source, physical effects and final ASC."""

from dataclasses import replace

import pytest
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as T
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.physical import RuntimeSizeOp
from ascendflow.dialects.source import ElementOp
from examples.normalization.softmax import softmax, softmax_compute


def mutated(change):
    """Mutate an owned main/helper pair, retaining the actual inline binding."""
    source = change(softmax.source)
    helper = replace(softmax_compute, source=change(softmax_compute.source))
    assert source != softmax.source or helper.source != softmax_compute.source
    return T.from_source(source, functions={"softmax_compute": helper}, disjoint_external=True)


@pytest.fixture(scope="module")
def artifact():
    return compile(softmax)


def test_packed_softmax_keeps_vector_statistics_and_runtime_pipeline(artifact):
    assert {op.sym_name.data for op in artifact.plan.walk() if isinstance(op, RuntimeSizeOp)} == {
        "tiles"
    }
    assert "GetValue" not in artifact.asc and "npu_value_" not in artifact.asc
    assert "asc_brcb" in artifact.asc
    assert "asc_sub(" in artifact.asc and "asc_div(" in artifact.asc
    assert "asc_repeat_reduce_max_only_value" in artifact.asc
    assert artifact.asc.count("asc_brcb") == 2


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        (lambda s: s.replace("scratch=broadcast", ""), "explicit scratch"),
        (lambda s: s.replace("(R, 8)", "(R, 16)"), "row broadcast"),
        (lambda s: s.replace("r64_then_packed_merge", "automatic"), "reduction"),
        (lambda s: s.replace("D // 64", "D // 32"), "partial/statistic"),
    ],
)
def test_invalid_reduction_and_broadcast_contracts_are_rejected(change, diagnostic):
    with pytest.raises((CompileError, VerifyException), match=diagnostic):
        compile(mutated(change))


def test_row_new_axis_does_not_create_a_dense_broadcast_buffer():
    source = softmax.to_ir()
    broadcasts = [
        op for op in source.walk() if isinstance(op, ElementOp) and op.kind.data.endswith("_rows")
    ]
    assert len(broadcasts) == 2
    for op in broadcasts:
        assert tuple(n.data for n in op.inputs[1].type.shape) == (8,)
        assert tuple(n.data for n in op.scratch.type.shape) == (8, 8)


def test_maximum_is_a_vector_primitive_with_ordinary_physical_effects():
    definition = T.from_source(
        """
    def maximum(X:T.Tensor((n,64),T.f32), Y:T.Tensor((n,64),T.f32,access="write"), n:T.i64):
        with T.Kernel():
            x=T.alloc_ub((64,),T.f32)
            y=T.alloc_ub((64,),T.f32)
            for i in T.Pipelined(n):
                T.copy(X[i,:],x)
                T.fill(y,0.0)
                T.maximum(x,y,out=y)
                T.copy(y,Y[i,:])
""",
        disjoint_external=True,
    )
    assert "asc_max(" in compile(definition).asc


def test_broadcast_requires_complete_supported_physical_rows():
    with pytest.raises((CompileError, VerifyException), match="row broadcast"):
        compile(softmax.specialize(R=4))


def test_broadcast_reads_the_borrowed_statistic_before_it_is_overwritten():
    source = mutated(
        lambda s: s.replace(
            "T.sub(x[slot, :, :], maximum[:, None]",
            "T.fill(statistic, 0.0)\n    T.sub(x[slot, :, :], maximum[:, None]",
        )
    )
    with pytest.raises(CompileError, match="overwrite before its actual native reader"):
        compile(source)


def test_broadcast_result_keeps_the_destination_as_its_logical_producer():
    source = mutated(
        lambda s: s.replace("T.sub(x[", "shifted = T.sub(x[").replace(
            "T.exp(y[slot, :, :]", "T.exp(shifted"
        )
    )
    assert "asc_exp" in compile(source).asc


def test_small_columns_preserve_partial_row_alignment():
    artifact = compile(softmax.specialize(D=64))
    assert {op.sym_name.data for op in artifact.plan.walk() if isinstance(op, RuntimeSizeOp)} == {
        "tiles"
    }
    # Without physical row padding the packed native merge must refuse it.
    source = mutated(lambda s: s.replace("((D // 64 + 7) // 8) * 8", "D // 64"))
    with pytest.raises(CompileError, match="aligned rows"):
        compile(source.specialize(D=64))
