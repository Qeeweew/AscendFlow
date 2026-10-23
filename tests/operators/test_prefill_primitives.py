"""Exact initializer rectangles and narrow row broadcasts used by chunk kernels."""

from collections import Counter
from types import SimpleNamespace

import pytest

from ascendflow.analysis.structured_admission import _covers, _Span, _summarize
from ascendflow.diagnostics import CompileError
from ascendflow.scalar import Expr, expression, literal
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


@pytest.mark.parametrize("count", [1, 2, 16])
def test_repeated_rectangles_preserve_row_holes_and_exact_byte_coverage(count):
    ordinal = Expr("argument", ("tile",))
    loop = SimpleNamespace(ordinal=ordinal, count=literal(count), entry_domains=())
    source = _Span(
        expression("add", literal(256), expression("mul", ordinal, literal(96))), literal(16), 3, 32
    )
    summary = _summarize(source, loop, {}, {})
    written = {
        256 + tile * 96 + row * 32 + col
        for tile in range(count)
        for row in range(3)
        for col in range(16)
    }
    represented = {
        summary.base.constant + row * summary.pitch + col
        for row in range(summary.rows)
        for col in range(summary.width.constant)
    }
    assert represented == written
    frame = SimpleNamespace(domains=lambda: ())
    for row in range(count * 3):
        for column, width in ((0, 16), (8, 8), (8, 16), (16, 8)):
            reader = _Span(literal(256 + row * 32 + column), literal(width))
            accessed = set(range(256 + row * 32 + column, 256 + row * 32 + column + width))
            assert _covers([summary], reader, frame, {}, Counter()) == (accessed <= written)


@pytest.mark.parametrize("stride", [64, 128])
def test_repeated_rectangles_with_different_group_pitch_still_reject(stride):
    ordinal = Expr("argument", ("tile",))
    loop = SimpleNamespace(ordinal=ordinal, count=literal(4), entry_domains=())
    source = _Span(expression("mul", ordinal, literal(stride)), literal(16), 3, 32)
    assert _summarize(source, loop, {}, {}) is None


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


@pytest.mark.parametrize("rows,columns", [(16, 16), (32, 32)])
def test_fp32_load3d_b_keeps_parent_pitch_and_reads_only_its_matrix(rows, columns):
    source = NativeGeometry(
        "L1",
        "f32",
        "NZ",
        rows * 32,
        rows=columns // 8,
        row_stride=64 * 32,
        matrix_shape=(64, columns),
    )
    destination = NativeGeometry("L0B", "f32", "ZN", rows * columns * 4)
    assert native_contract("load3d_b_nn", (source, destination), (rows, columns)) == (
        "MTE1",
        ("R", "W"),
    )
    touched = {plane * 64 * 32 + byte for plane in range(columns // 8) for byte in range(rows * 32)}
    assert len(touched) == rows * columns * 4
    assert rows * 32 not in touched
    invalid = NativeGeometry(
        "L1", "f32", "NZ", rows * 32, rows=columns // 8, row_stride=64 * 32 + 32
    )
    with pytest.raises(CompileError, match="FP32 Load3D"):
        native_contract("load3d_b_nn", (invalid, destination), (rows, columns))
