"""Explicit block8 trees preserve exact native effects and synchronization."""

import pytest
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import check_structured_events


def definition(kind="sum", rows=8, columns=1024, workspace=1152):
    return N.from_source(
        f"""
        def block8(n: N.int64):
            with N.Kernel(kind="vector"):
                x = N.alloc_ub(({rows}, {columns}), N.f32)
                workspace = N.alloc_ub((2, {workspace}), N.f32)
                statistic = N.alloc_ub(({rows},), N.f32)
                for i in N.Pipelined(n):
                    values = N.fill(x, 1.0)
                    reduced = N.reduce_{kind}(values, axis=1, out=statistic,
                        scratch=workspace[i % 2, :], order="block8_then_packed_merge")
                    N.mul(reduced, 2.0, out=statistic)
        """,
        disjoint_external=True,
    )


@pytest.mark.parametrize("columns", [576, 704, 960, 1088, 1984, 4032])
def test_block8_order_rejects_unaligned_derived_stages_without_fallback(columns):
    with pytest.raises(VerifyException, match="block8.*whole 64-element repeats"):
        definition(columns=columns).to_ir()


@pytest.mark.parametrize("words", [1144, 1160])
def test_block8_workspace_extent_must_be_exact(words):
    with pytest.raises(VerifyException, match="exact partial/statistic shapes"):
        definition(workspace=words).to_ir()


def test_block8_workspace_must_be_one_dimensional():
    source = definition().source.replace("scratch=workspace[i % 2, :]", "scratch=x[:, :144]")
    with pytest.raises(VerifyException, match="exact partial/statistic shapes"):
        N.from_source(source, disjoint_external=True).to_ir()


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_eight_column_reduction_packs_whole_row_repeats_without_a_merge(kind):
    result = compile(definition(kind=kind, rows=32, columns=8, workspace=32))
    memory = collect_native_memory(result.plan)
    natives = [t.spec.native for t in memory.templates]
    assert natives.count("reduce_" + kind + "8") == 1
    assert not any(native.startswith("merge_") for native in natives)
    original = collect_native_memory(parse(result.before_ir))
    for count in (0, 1, 3):
        check_structured_events(memory, {"n": count}, logical_reference=original, invocations=2)


@pytest.mark.parametrize("rows", [1, 7, 9, 256])
def test_eight_column_reduction_rejects_partial_or_excessive_row_repeats(rows):
    with pytest.raises(VerifyException, match="8-aligned rows"):
        definition(rows=rows, columns=8, workspace=rows).to_ir()


def test_eight_column_reduction_keeps_the_native_nonoverlap_contract():
    source = definition(rows=8, columns=8, workspace=8).source.replace(
        "out=statistic,", "out=x[0,:],"
    )
    with pytest.raises(CompileError, match="native read/write aliases"):
        compile(N.from_source(source, disjoint_external=True), timeout_s=10)


@pytest.mark.parametrize("destination", [False, True])
def test_block8_does_not_relax_native_in_place_overlap_rules(destination):
    source = definition(rows=1, workspace=144).source
    # Keep the physical allocation aligned so the fixture reaches alias
    # admission rather than failing the unrelated local capacity contract.
    source = source.replace("statistic = N.alloc_ub((1,),", "statistic = N.alloc_ub((8,),")
    source = source.replace("out=statistic", "out=statistic[:1]")
    if destination:
        source = source.replace("out=statistic[:1],", "out=workspace[i % 2, 128:129],")
    else:
        source = source.replace("scratch=workspace[i % 2, :]", "scratch=x[0, :144]")
    with pytest.raises(CompileError, match="native read/write aliases"):
        compile(N.from_source(source, disjoint_external=True), timeout_s=10)


def test_block8_tracks_the_input_reader_and_final_value_producer():
    source = definition().source.replace(
        "reduced = N.reduce_sum", "N.fill(x, 0.0)\n            reduced = N.reduce_sum"
    )
    with pytest.raises(CompileError, match="overwrite before its actual native reader"):
        compile(N.from_source(source, disjoint_external=True), timeout_s=10)
