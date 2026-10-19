"""Terminal-region and first-clear proofs checked against finite actual accesses."""

from dataclasses import replace

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.target.ascend910b import Engine
from examples.gemm.reduction import reduction_gemm


def test_first_clear_does_not_initialize_other_ring_slots():
    source = reduction_gemm.source.replace(
        "c0 = T.alloc_l0c((M, N)", "c0 = T.alloc_l0c((2, M, N)"
    ).replace("c0[:, :]", "c0[k % 2, :, :]")
    with pytest.raises(CompileError, match="without a proved initialized writer coverage"):
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


def test_nonterminal_export_is_supported_while_late_first_clear_remains_invalid():
    source = reduction_gemm.source.replace("k == panels - 1", "k == panels - 2")
    result = compile(replace(reduction_gemm, source=source))
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    final = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    for panels in (0, 1, 2, 3, 5):
        check_structured_events(final, {"panels": panels}, logical_reference=original)
    with pytest.raises(CompileError, match="own pipeline's first logical iteration"):
        compile(replace(reduction_gemm, source=reduction_gemm.source.replace("k == 0", "k == 1")))


@pytest.mark.parametrize("reader_engine", [Engine.FIX, Engine.MTE3])
@pytest.mark.parametrize("predicate", ["i == panels - 1", "i + 1 == panels"])
def test_singleton_reader_does_not_synchronize_its_idle_engine_each_iteration(
    reader_engine, predicate
):
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    definition = (
        replace(
            reduction_gemm,
            source=reduction_gemm.source.replace("k == panels - 1", predicate.replace("i", "k")),
        )
        if reader_engine is Engine.FIX
        else N.from_source(f"""
        def last_store(Y: N.Tensor((64,), N.f32, access="write"), panels: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                for i in N.Pipelined(panels, stages=1):
                    N.fill(x, 1.0)
                    if {predicate}:
                        N.copy(x, Y)
        """)
    )
    result = compile(definition)
    final = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    counts = []
    for panels in (0, 1, 2, 7):
        checked = check_structured_events(
            final, {"panels": panels}, logical_reference=original, invocations=2
        )
        if panels:
            counts.append(
                sum(
                    command.kind != "native"
                    and reader_engine in (command.source, command.destination)
                    for command in checked.commands
                )
            )
    assert len(set(counts)) == 1, "inactive readers must not add per-iteration queue handoffs"


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
    with pytest.raises(CompileError, match="without a proved initialized writer coverage"):
        compile(definition)
