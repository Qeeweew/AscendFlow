"""Terminal-region and first-clear proofs checked against finite actual accesses."""

from dataclasses import replace

import pytest

from examples.gemm.reduction import reduction_gemm
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError


def test_first_clear_does_not_initialize_other_ring_slots():
    source = reduction_gemm.source.replace(
        "c0 = T.alloc_l0c((M, N)", "c0 = T.alloc_l0c((2, M, N)"
    ).replace("c0[:, :]", "c0[k % 2, :, :]")
    with pytest.raises(CompileError, match="undefined first read"):
        compile(replace(reduction_gemm, source=source))


def test_conditional_clear_keeps_the_accumulator_numerical_reader():
    source = reduction_gemm.source.replace(
        "value = T.matmul(aa, bb, accum=c0[:, :], initialize=(k == 0))",
        "prior = T.matmul(aa, bb, accum=c0[:, :], initialize=True)\n"
        "                value = T.matmul(aa, bb, accum=prior, initialize=(k == 0))",
    )
    result = compile(replace(reduction_gemm, source=source))
    assert result.report["native_templates"] == 7
    assert result.asc.count("asc_mmad(") == 2


def test_unsupported_boundary_and_late_first_clear_are_not_assumed_safe():
    for original, changed, message in [
        ("k == panels - 1", "k == panels - 2", "pipeline bound - 1"),
        ("k == 0", "k == 1", "own generation == 0"),
    ]:
        with pytest.raises(CompileError, match=message):
            compile(
                replace(reduction_gemm, source=reduction_gemm.source.replace(original, changed))
            )


def test_terminal_writer_cannot_initialize_a_steady_first_read():
    definition = N.from_source("""
        def uninitialized(n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                y = N.alloc_ub((64,))
                for t in N.Pipelined(n, stages=2):
                    with N.Stage(0):
                        N.mul(x, 2.0, out=y)
                    with N.Stage(1):
                        if t == n - 1:
                            N.fill(x, 0.0)
    """)
    with pytest.raises(CompileError, match="undefined first read"):
        compile(definition)
