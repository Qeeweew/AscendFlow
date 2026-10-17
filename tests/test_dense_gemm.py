"""Real contiguous matrices, cyclic rectangle owners and finite event traces."""

from dataclasses import replace
from pathlib import Path

import pytest

from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_ownership import admit_owners
from ascendflow.analysis.structured_tiles import nd_grid_cell
from ascendflow.build import build_file
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.scalar import expression, literal
from examples.gemm.dense import dense_gemm
from tests.models.structured_events import check_structured_events

EXAMPLE = Path(__file__).parents[1] / "examples/gemm/dense.py"


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


def test_tiled_owner_proof_uses_grid_cells_and_retains_every_actual_access():
    source = dense_gemm.specialize(
        M=64, N=80, K=16, BM=16, BN=16, BK=16, CK=16, COL_PARTS=1, GROUPS=3
    )
    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    admitted = admit_structured_memory(structure, memory, bounds)
    (proof,) = admitted.owner_proofs
    assert proof.tile_shape == (16, 16) and proof.stride.constant == 1
    assert all(access.address.root is proof.root for access in proof.accesses)
    access = proof.accesses[0].address
    cell = access.tile
    assert cell.extents == (64, 80)
    # Geometry identity must not trust axes independently of real bytes.
    facts = memory.control.environment.facts
    assert (
        nd_grid_cell(
            expression("add", access.offset, literal(4)), access.geometry, access.axes, facts
        )
        is None
    )
    assert (
        nd_grid_cell(access.offset, replace(access.geometry, row_stride=16 * 4), access.axes, facts)
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
def test_aliasing_owner_maps_remain_rejected(mapping):
    from ascendflow import language as T

    text = EXAMPLE.read_text().replace("index = group + tile * GROUPS", "index = " + mapping)
    text = text[text.index("def dense_gemm(") :]
    source = T.from_source(text, disjoint_external=True)
    source = source.specialize(M=64, N=80, K=16, BM=16, BN=16, BK=16, CK=16, COL_PARTS=1, GROUPS=3)
    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    with pytest.raises(CompileError, match="owner"):
        admit_owners(memory)


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
    with pytest.raises(CompileError, match="aligned full-row L1 NZ column slice"):
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
