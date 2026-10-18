"""Triangular updates preserve old lanes and use the ordinary memory contract."""

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


def source(*, initialized=True, rows=8, cols=128, dtype="f32"):
    load = "N.copy(X, work)" if initialized else ""
    return N.from_source(
        f"""
        def upper(X:N.Tensor(({rows},{cols}),N.{dtype}),
                  Y:N.Tensor(({rows},{cols}),N.{dtype},access="write"),
                  q:N.int64, k:N.int64):
            with N.Kernel():
                work=N.alloc_ub(({rows},{cols}),N.{dtype})
                {load}
                N.fill_upper(work, NEG_INF, row_start=q, col_start=k)
                N.copy(work,Y)
    """,
        constants={"NEG_INF": float("-inf")},
        disjoint_external=True,
    )


def test_preserved_lanes_require_initialization_and_async_load_store_order():
    result = compile(source())
    memory = collect_native_memory(result.plan)
    (mask,) = [t for t in memory.templates if t.contract.native == "fill_upper"]
    assert mask.contract.modes == ("RW",)
    assert mask.contract.packet.closed_mask
    assert mask.contract.packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}
    original = collect_native_memory(parse(result.before_ir))
    for q, k in ((0, 0), (63, 0), (64, 0), (127, 0), (0, 128), (2**63 - 1, 0)):
        execution = check_structured_events(memory, {"q": q, "k": k}, logical_reference=original)
        assert execution.bytes_read and execution.bytes_written
    with pytest.raises(CompileError, match="initialized writer"):
        compile(source(initialized=False))


@pytest.mark.parametrize("options", [{"dtype": "f16"}, {"cols": 32}, {"rows": 256}])
def test_unsupported_mask_geometry_is_rejected(options):
    with pytest.raises(Exception, match="upper fill requires"):
        compile(source(**options))


def test_clipped_runtime_inner_loop_preserves_both_operand_bounds():
    definition = N.from_source(
        """
        def clipped(X:N.Tensor((n,64),N.f32), Y:N.Tensor((n,64),N.f32,access="write"),
                    n:N.int64(minimum=1), bound:N.int64):
            with N.Kernel():
                work=N.alloc_ub((64,))
                for j in N.Pipelined(N.min(n,bound),stages=1):
                    N.copy(X[j,:],work)
                    N.copy(work,Y[j,:])
    """,
        disjoint_external=True,
    )
    result = compile(definition)
    memory = collect_native_memory(result.plan)
    for n, bound in ((1, 0), (1, 5), (7, 3), (3, 7)):
        check_structured_events(memory, {"n": n, "bound": bound})
