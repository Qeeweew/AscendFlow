"""Real contiguous matrices, cyclic rectangle owners and finite event traces."""

from dataclasses import replace
from pathlib import Path

import pytest

from ascendflow.analysis.structured_addresses import nd_rectangle
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import expression, literal
from examples.gemm.dense import accumulate_panel, dense_gemm, import_panel
from tests.models.structured_events import EventModelError, check_structured_events

EXAMPLE = Path(__file__).parents[2] / "examples/gemm/dense.py"


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("row_begin", (0, 32))
def test_b16_strided_a_panel_uses_one_load3d_with_its_real_resident_pitch(dtype, row_begin):
    from ascendflow import compile
    from ascendflow import language as T

    result = compile(
        T.from_source(f"""
def panel(X:T.Tensor((128,128),T.{dtype})):
    with T.Kernel(kind="cube"):
        a1=T.alloc_l1((128,128),T.{dtype})
        a0=T.alloc_l0a((64,64),T.{dtype})
        v=T.copy(X,a1)
        T.copy(v[{row_begin}:{row_begin + 64},32:96],a0)
""")
    )
    memory = collect_native_memory(result.plan)
    load = next(t for t in memory.templates if t.spec.native == "load3d_a")
    source, destination = load.spec.operands
    assert (source.nbytes, source.rows, source.row_stride) == (2048, 4, 4096)
    assert source.leading_rows == 128
    assert destination.total_nbytes == 64 * 64 * 2
    check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    assert "npu_chunk" not in result.asc
    assert result.asc.count("asc_copy_l12l0a(") == 1


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_b16_complete_b_panel_transposes_block_axes_with_one_load3d(dtype):
    from ascendflow import compile
    from ascendflow import language as T

    result = compile(
        T.from_source(f"""
def panel(X:T.Tensor((128,128),T.{dtype})):
    with T.Kernel(kind="cube"):
        b1=T.alloc_l1((128,128),T.{dtype})
        b0=T.alloc_l0b((128,128),T.{dtype})
        v=T.copy(X,b1)
        T.copy(v,b0)
""")
    )
    memory = collect_native_memory(result.plan)
    load = next(t for t in memory.templates if t.spec.native == "load3d_b_nn")
    assert load.spec.operands[0].leading_rows == 128
    check_structured_events(
        memory, logical_reference=collect_native_memory(parse(result.before_ir))
    )
    assert "npu_chunk" not in result.asc
    assert result.asc.count("asc_copy_l12l0b(") == 1


@pytest.mark.parametrize("bk,k", [(16, 16), (16, 32), (16, 48), (32, 32), (64, 128)])
def test_contiguous_nt_gemm_keeps_byte_versions_and_actual_event_consumption(tmp_path, bk, k):
    report = build_file(
        EXAMPLE,
        tmp_path,
        entry="dense_gemm",
        parameters={
            "M": 64,
            "N": 80,
            "K": k,
            "BM": 16,
            "BN": 16,
            "BK": bk,
            "CK": 16,
            "COL_PARTS": 1,
            "GROUPS": 3,
        },
    )
    assert report["native_templates"] == 3 + 3 * (bk // 16)
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1, 2))
    assert trace.bytes_read and trace.bytes_written and trace.consumptions


def test_tiled_geometry_retains_every_actual_access_without_owner_admission():
    source = dense_gemm.specialize(
        M=64, N=80, K=16, BM=16, BN=16, BK=16, CK=16, COL_PARTS=1, GROUPS=3
    )
    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    admitted = admit_structured_memory(structure, memory, bounds)
    assert admitted.memory is memory
    access = next(
        a.address
        for t in memory.templates
        for a in t.accesses
        if a.address.geometry.bank == "GM" and a.mode.writes
    )
    assert access.rectangle.extents == (64, 80)
    # Geometry identity must not trust axes independently of real bytes.
    facts = memory.control.environment.facts
    assert (
        nd_rectangle(
            expression("add", access.offset, literal(4)), access.geometry, access.axes, facts
        )
        is None
    )
    assert (
        nd_rectangle(access.offset, replace(access.geometry, row_stride=16 * 4), access.axes, facts)
        is None
    )


@pytest.mark.parametrize("prefetch", [1, 2])
def test_rectangular_output_and_single_accumulator_retirement(tmp_path, prefetch):
    build_file(
        EXAMPLE,
        tmp_path,
        entry="dense_gemm",
        parameters={
            "M": 64,
            "N": 160,
            "K": 128,
            "BM": 16,
            "BN": 32,
            "BK": 64,
            "CK": 16,
            "C_SLOTS": 1,
            "COL_PARTS": 1,
            "PREFETCH": prefetch,
            "GROUPS": 3,
        },
    )
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1, 2))
    assert trace.bytes_read and trace.bytes_written and trace.consumptions


@pytest.mark.parametrize("mapping", ["group % 2 + tile * 3", "group + tile * 2"])
def test_aliasing_owner_maps_violate_the_source_parallel_contract(mapping):
    from ascendflow import compile
    from ascendflow import language as T

    text = EXAMPLE.read_text().replace(
        "index = group + tile * GROUPS", "index = (" + mapping + ") % tiles"
    )
    text = text[text.index("def dense_gemm(") :]
    source = T.from_source(
        text,
        functions={"import_panel": import_panel, "accumulate_panel": accumulate_panel},
        disjoint_external=True,
    )
    source = source.specialize(M=64, N=80, K=16, BM=16, BN=16, BK=16, CK=16, COL_PARTS=1, GROUPS=3)
    memory = collect_native_memory(compile(source).plan)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(memory, groups=(0, 1, 2))


def test_nz_column_slice_needs_fractal_alignment_even_when_bytes_are_aligned():
    from ascendflow import language as T

    source = T.from_source("""
        def slice():
            with T.Kernel(1,kind="cube"):
                x=T.alloc_l1((32,64),T.f16)
                a=T.alloc_l0a((32,16),T.f16)
                T.copy(x[:,8:24],a)
    """)
    memory = collect_native_memory(lower_native_structure(source).module)
    with pytest.raises(CompileError, match="aligned NZ rectangle"):
        admit_memory_bounds(memory)


@pytest.mark.parametrize("k", [32, 96])
def test_paired_columns_share_a_operands_and_retire_each_accumulator(tmp_path, k):
    report = build_file(
        EXAMPLE,
        tmp_path,
        entry="dense_gemm",
        parameters={
            "M": 64,
            "N": 192,
            "K": k,
            "BM": 16,
            "BN": 16,
            "BK": 32,
            "CK": 16,
            "COL_PARTS": 2,
            "C_SLOTS": 1,
            "GROUPS": 3,
        },
    )
    assert report["native_templates"] == 15
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1, 2))
    assert trace.bytes_read and trace.bytes_written and trace.consumptions
