"""One static native contract; UB alias proofs belong to its address consumer."""

from dataclasses import replace

import pytest

from ascendflow.diagnostics import CompileError
from ascendflow.target import native as target_native
from ascendflow.target.native import NativeGeometry, native_contract


def test_geometry_counts_accessed_bytes_without_including_pitched_holes():
    partials = NativeGeometry("UB", "f32", "linear", 8, rows=8, row_stride=32)
    assert partials.total_nbytes == 64
    assert not partials.contiguous
    assert replace(partials, row_stride=8).contiguous
    assert replace(partials, rows=1).contiguous


@pytest.mark.parametrize(
    "changes",
    [
        {"nbytes": 0},
        {"nbytes": -32},
        {"rows": 0},
        {"rows": -1},
        {"nbytes": -32, "rows": -1, "row_stride": -32},
        {"row_stride": -1},
        {"rows": 2, "row_stride": 0},
        {"rows": 2, "row_stride": 16},
    ],
)
def test_geometry_rejects_invalid_rows_before_they_can_mimic_valid_native_lengths(changes):
    with pytest.raises(CompileError, match="native geometry requires"):
        replace(NativeGeometry("UB", "f32", "linear", 32), **changes)


@pytest.mark.parametrize(
    "native,refs,params,expected",
    [
        (
            "nd2nz",
            (
                NativeGeometry("GM", "f16", "linear", 32, rows=16, row_stride=64),
                NativeGeometry("L1", "f16", "NZ", 512),
            ),
            (16, 16),
            ("MTE2", ("R", "W")),
        ),
        (
            "mad",
            (
                NativeGeometry("L0A", "f16", "ZZ", 512),
                NativeGeometry("L0B", "f16", "ZN", 512),
                NativeGeometry("L0C", "f32", "NZ", 1024),
            ),
            (16, 16, 16, 1),
            ("M", ("R", "R", "W")),
        ),
        (
            "mad_first",
            (
                NativeGeometry("L0A", "f16", "ZZ", 512),
                NativeGeometry("L0B", "f16", "ZN", 512),
                NativeGeometry("L0C", "f32", "NZ", 1024),
            ),
            (16, 16, 16),
            ("M", ("R", "R", "RW")),
        ),
        (
            "fix",
            (
                NativeGeometry("L0C", "f32", "NZ", 1024),
                NativeGeometry("GM", "f32", "linear", 64, rows=16, row_stride=128),
            ),
            (16, 16),
            ("FIX", ("R", "W")),
        ),
    ],
)
def test_matrix_contract_uses_static_tiles_and_preserves_accumulator_modes(
    native, refs, params, expected
):
    assert native_contract(native, refs, params) == expected


@pytest.mark.parametrize("operation", ["sum", "max"])
def test_reduction_and_packed_merge_keep_exact_rows_and_accessed_bytes(operation):
    matrix = NativeGeometry("UB", "f32", "linear", 4096)
    partials = NativeGeometry("UB", "f32", "linear", 8, rows=8, row_stride=32)
    compact = NativeGeometry("UB", "f32", "linear", 32)
    expected = "V", ("R", "W")
    assert native_contract(f"reduce_{operation}64", (matrix, partials), (8, 128)) == expected
    assert native_contract(f"merge_{operation}_packed", (partials, compact), (8, 2)) == expected
    with pytest.raises(CompileError, match="exact row footprint"):
        native_contract(
            f"reduce_{operation}64", (matrix, replace(partials, nbytes=64, rows=1)), (8, 128)
        )
    for source, destination in (
        (replace(partials, nbytes=64, rows=1), compact),
        (replace(partials, row_stride=8), compact),
        (partials, replace(compact, nbytes=4, rows=8, row_stride=32)),
    ):
        with pytest.raises(CompileError, match="packed partial reduction"):
            native_contract(f"merge_{operation}_packed", (source, destination), (8, 2))


@pytest.mark.parametrize("answer", [False, 1])
def test_explicit_alias_proof_must_establish_true(answer):
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        native_contract("copy", ub_geometries(), (), disjoint=lambda: answer)


def test_alias_proof_exception_is_propagated():
    def unavailable():
        raise RuntimeError("address proof is unavailable")

    with pytest.raises(RuntimeError, match="address proof is unavailable"):
        native_contract("copy", ub_geometries(), (), disjoint=unavailable)


def test_noncallable_alias_premise_is_rejected():
    with pytest.raises(CompileError, match="premise must be callable"):
        native_contract("copy", ub_geometries(), (), disjoint=True)


@pytest.mark.parametrize(
    "source_changes,destination_changes,params,diagnostic",
    [
        ({"nbytes": 16}, {"nbytes": 16}, (), "32-byte aligned"),
        ({}, {"nbytes": 64}, (), "equal native footprints"),
        ({}, {"dtype": "f16"}, (), "cannot silently convert dtype"),
        ({}, {"layout": "NZ"}, (), "preserve physical layout"),
        ({"nbytes": 16, "rows": 2, "row_stride": 32}, {}, (), "contiguous physical views"),
        ({}, {}, (1,), "no unchecked native parameters"),
    ],
)
def test_invalid_static_ub_copy_is_rejected_before_requesting_alias_proof(
    source_changes, destination_changes, params, diagnostic
):
    source, destination = ub_geometries()
    calls = []

    def prove():
        calls.append("proof")
        return True

    with pytest.raises(CompileError, match=diagnostic):
        native_contract(
            "copy",
            (replace(source, **source_changes), replace(destination, **destination_changes)),
            params,
            disjoint=prove,
        )
    assert not calls


@pytest.mark.parametrize(
    "operation,destination_bank,destination_layout,delegate",
    [
        ("load3d_a", "L0A", "ZZ", "load2d_a"),
        ("load3d_b_nn", "L0B", "ZN", "load2d_b_nn"),
    ],
)
def test_load3d_forwards_proof_identity_without_consuming_it(
    monkeypatch, operation, destination_bank, destination_layout, delegate
):
    refs = (
        NativeGeometry("L1", "f16", "NZ", 512),
        NativeGeometry(destination_bank, "f16", destination_layout, 512),
    )
    calls = []

    def proof():
        raise AssertionError("Load2D has no UB alias premise to consume")

    def trace(native, operands, params, *, disjoint=None):
        calls.append((native, operands, params, disjoint))
        return native_contract(native, operands, params, disjoint=disjoint)

    monkeypatch.setattr(target_native, "native_contract", trace)
    assert native_contract(operation, refs, (16, 16), disjoint=proof) == ("MTE1", ("R", "W"))
    assert calls == [(delegate, refs, (16, 16), proof)]


def ub_geometries():
    source = NativeGeometry("UB", "f32", "linear", 32)
    return source, replace(source)


def test_ub_copy_consumes_the_explicit_address_proof_once():
    calls = []

    def prove():
        calls.append("proof")
        return True

    assert native_contract("copy", ub_geometries(), (), disjoint=prove) == ("V", ("R", "W"))
    assert calls == ["proof"]


@pytest.mark.parametrize("same_object", [False, True])
def test_equal_geometry_cannot_establish_address_disjointness(same_object):
    source, destination = ub_geometries()
    with pytest.raises(CompileError, match="geometry inputs need an explicit alias proof"):
        native_contract("copy", (source, source if same_object else destination), ())
