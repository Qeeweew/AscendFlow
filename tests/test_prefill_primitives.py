"""Exact initializer rectangles and narrow row broadcasts used by chunk kernels."""

from collections import Counter
from types import SimpleNamespace

import pytest

from ascendflow.analysis.structured_admission import _covers, _Span
from ascendflow.diagnostics import CompileError
from ascendflow.scalar import literal
from ascendflow.target.native import NativeGeometry, native_contract
from ascendflow.target.packets import MaskStep, audited_packet


def test_strided_initializer_coverage_matches_actual_bytes():
    writer = _Span(literal(256), literal(16), 3, 32)
    initialized = {256 + r * 32 + col for r in range(3) for col in range(16)}
    frame = SimpleNamespace(domains=lambda: ())
    for origin in range(248, 364, 4):
        for rows in (1, 2, 3):
            for width in (4, 8, 16, 20):
                reader = _Span(literal(origin), literal(width), rows, 32)
                accessed = {origin + r * 32 + col for r in range(rows) for col in range(width)}
                assert _covers([writer], reader, frame, {}, Counter()) == (accessed <= initialized)


@pytest.mark.parametrize("columns", [8, 16, 24, 32, 48, 64, 128])
def test_row_broadcast_has_exact_matrix_and_scalar_footprints(columns):
    rows = 16
    matrix = NativeGeometry("UB", "f32", "linear", rows * columns * 4)
    scalars = NativeGeometry("UB", "f32", "linear", rows * 32)
    assert native_contract("mul_row_broadcast", (matrix, scalars, matrix), (rows, columns)) == (
        "V",
        ("R", "R", "W"),
    )
    packet = audited_packet("mul_row_broadcast")
    assert packet.closed_mask
    assert MaskStep.SET_COLUMNS in packet.mask_steps
    assert packet.clobbers == {"VECTOR_MASK", "VECTOR_MASK_MODE"}


@pytest.mark.parametrize("columns", [4, 63, 65])
def test_row_broadcast_rejects_unaligned_or_unmodeled_repeats(columns):
    matrix = NativeGeometry("UB", "f32", "linear", 16 * columns * 4)
    scalars = NativeGeometry("UB", "f32", "linear", 16 * 32)
    with pytest.raises(CompileError, match="row broadcast arithmetic"):
        native_contract("mul_row_broadcast", (matrix, scalars, matrix), (16, columns))


@pytest.mark.parametrize("width", [32, 64, 128, 256])
def test_vector_row_copy_uses_exact_pitched_footprints(width):
    source = NativeGeometry("UB", "f32", "linear", width, rows=16, row_stride=width + 32)
    destination = NativeGeometry("UB", "f32", "linear", width, rows=16, row_stride=width + 64)
    assert native_contract(
        "vector_copy_rows", (source, destination), (), disjoint=lambda: True
    ) == ("V", ("R", "W"))
    with pytest.raises(CompileError, match="provably disjoint"):
        native_contract("vector_copy_rows", (source, destination), (), disjoint=lambda: False)


@pytest.mark.parametrize(
    "width,rows,pitch", [(16, 16, 64), (288, 16, 320), (64, 256, 64), (64, 16, 8192)]
)
def test_vector_row_copy_rejects_unrepresentable_native_geometry(width, rows, pitch):
    ref = NativeGeometry("UB", "f32", "linear", width, rows=rows, row_stride=pitch)
    with pytest.raises(CompileError, match="Vector row copy"):
        native_contract("vector_copy_rows", (ref, ref), (), disjoint=lambda: True)
