"""One static native contract; UB alias proofs belong to its address consumer."""

import sys
from dataclasses import replace

import pytest

from tests.fixtures.pipelines import cube_pipeline, vector_pipeline
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import Ref, Storage
from xdsl_npu.target import native as target_native
from xdsl_npu.target.native import NativeGeometry, native_contract


def geometry(ref):
    return NativeGeometry(
        ref.storage.bank,
        ref.dtype,
        ref.layout,
        ref.nbytes,
        ref.rows,
        ref.row_stride,
        ref.tail_nbytes,
        ref.prefix_extent,
    )


def ub_refs():
    storage = Storage("source", "UB", 4096, "f32", "linear", False, True)
    source = Ref(storage, ((0, 64, 2, 1),), 32, "f32", "linear")
    return source, replace(source, storage=replace(storage, name="destination"))


def test_legacy_static_adapter_does_not_import_program_model(monkeypatch):
    refs = ub_refs()
    monkeypatch.setitem(sys.modules, "xdsl_npu.model", None)
    assert native_contract("copy", refs, ()) == ("V", ("R", "W"))


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


@pytest.mark.parametrize("pipeline", [vector_pipeline, cube_pipeline], ids=["vector", "cube"])
def test_real_pipeline_native_results_are_identical_for_refs_and_static_geometry(pipeline):
    loop = extract_loop(pipeline())
    for actor in loop.actors.values():
        for task in actor.program.tasks:
            expected = task.pipe, task.modes
            assert native_contract(task.native, task.refs, task.params) == expected
            assert (
                native_contract(task.native, tuple(map(geometry, task.refs)), task.params)
                == expected
            )


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


@pytest.mark.parametrize("kind", ["geometry", "mixed", "legacy"])
def test_ub_copy_uses_explicit_proof_once_without_legacy_alias_fallback(monkeypatch, kind):
    source, destination = ub_refs()
    refs = (
        (geometry(source), geometry(destination))
        if kind == "geometry"
        else (source, geometry(destination))
        if kind == "mixed"
        else (source, destination)
    )
    calls = []

    def prove():
        calls.append("proof")
        return True

    def forbidden(*args):
        raise AssertionError("an explicit address proof cannot use legacy Ref aliases")

    monkeypatch.setattr(target_native, "_disjoint_ub_copy", forbidden)
    assert native_contract("copy", refs, (), disjoint=prove) == ("V", ("R", "W"))
    assert calls == ["proof"]


@pytest.mark.parametrize("kind", ["same_geometry", "equal_geometry", "mixed"])
def test_geometry_ub_copy_requires_an_address_proof_even_when_footprints_are_equal(kind):
    source, _ = ub_refs()
    static = geometry(source)
    refs = (
        (static, static)
        if kind == "same_geometry"
        else (static, replace(static))
        if kind == "equal_geometry"
        else (source, static)
    )
    with pytest.raises(CompileError, match="geometry inputs need an explicit alias proof"):
        native_contract("copy", refs, ())


@pytest.mark.parametrize("answer", [False, 1])
def test_explicit_alias_proof_must_establish_true_even_for_disjoint_legacy_roots(answer):
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        native_contract("copy", ub_refs(), (), disjoint=lambda: answer)


def test_alias_proof_exception_is_not_replaced_with_legacy_success():
    def unavailable():
        raise RuntimeError("address proof is unavailable")

    with pytest.raises(RuntimeError, match="address proof is unavailable"):
        native_contract("copy", ub_refs(), (), disjoint=unavailable)


def test_noncallable_alias_premise_is_rejected():
    with pytest.raises(CompileError, match="premise must be callable"):
        native_contract("copy", ub_refs(), (), disjoint=True)


@pytest.mark.parametrize(
    "source_changes,destination_changes,params,diagnostic",
    [
        ({"nbytes": 16}, {"nbytes": 16}, (), "32-byte aligned"),
        ({}, {"nbytes": 64}, (), "equal native footprints"),
        ({}, {"dtype": "f16"}, (), "cannot silently convert dtype"),
        ({}, {"layout": "NZ"}, (), "preserve physical layout"),
        ({"nbytes": 16, "rows": 2, "row_stride": 32}, {}, (), "contiguous physical views"),
        ({"prefix_extent": object()}, {}, (), "fixed complete 32-byte blocks"),
        ({}, {"tail_nbytes": 16}, (), "fixed complete 32-byte blocks"),
        ({}, {}, (1,), "no unchecked native parameters"),
    ],
)
def test_invalid_static_ub_copy_is_rejected_before_requesting_alias_proof(
    source_changes, destination_changes, params, diagnostic
):
    source, destination = map(geometry, ub_refs())
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


def test_legacy_default_keeps_root_separation_and_equal_translation_alias_rule():
    source, destination = ub_refs()
    assert native_contract("copy", (source, destination), ()) == ("V", ("R", "W"))
    source = replace(
        source,
        terms=((0, 64, 2, 1), (0, 32, 2, 2)),
        core_terms=((128, 1, 2),),
        count_offset=32,
        count_core_stride=64,
    )
    destination = replace(
        source,
        terms=((32, 16, 2, 2), (0, 64, 2, 1), (0, 16, 2, 2)),
        core_terms=((64, 1, 2), (64, 1, 2)),
    )
    assert native_contract("copy", (source, destination), ()) == ("V", ("R", "W"))


@pytest.mark.parametrize(
    "changes",
    [
        {"terms": ((16, 64, 2, 1),)},
        {"terms": ((64, 32, 2, 1),)},
        {"core_terms": ((32, 1, 2),)},
        {"count_offset": 32},
        {"count_core_stride": 32},
    ],
)
def test_legacy_default_rejects_overlapping_or_changing_alias_translations(changes):
    source, _ = ub_refs()
    destination = replace(source, terms=((64, 64, 2, 1),))
    destination = replace(destination, **changes)
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        native_contract("copy", (source, destination), ())


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
