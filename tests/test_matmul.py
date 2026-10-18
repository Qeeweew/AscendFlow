"""NN b16 GEMM, exact edge accesses, zero padding and FIXPIPE conversion."""

from pathlib import Path

import pytest

from ascendflow import language as T
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.target.native import NativeGeometry, native_contract
from examples.gemm.schedule import l2_tile
from tests.models.structured_events import check_structured_events

EXAMPLE = Path(__file__).parents[1] / "examples/gemm/matmul.py"


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 19, 21), (48, 64, 48)])
@pytest.mark.parametrize("bk", [16, 32])
def test_nn_gemm_edge_bytes_and_event_retirement(tmp_path, dtype, shape, bk):
    m, n, k = shape
    build_file(
        EXAMPLE,
        tmp_path,
        entry="gemm",
        parameters={
            "M": m,
            "N": n,
            "K": k,
            "BM": 16,
            "BN": 16,
            "BK": bk,
            "CK": 16,
            "COL_PARTS": 2,
            "GROUPS": 2,
            "DTYPE": dtype,
        },
        timeout_s=300,
    )
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1))
    assert trace.bytes_read and trace.bytes_written and trace.consumptions
    asc = (tmp_path / "gemm.asc").read_text()
    mode = "F322F16" if dtype == "f16" else "F322BF16"
    assert f"QuantMode_t::{mode}" in asc
    assert "load_cbuf_to_cb" in asc
    if any(size % 16 for size in shape):
        assert "asc_fill_l1" in asc


@pytest.mark.parametrize("dtype,itemsize", [("f32", 4), ("f16", 2), ("bf16", 2)])
def test_fixpipe_cropped_fractal_reads_and_element_pitched_output(dtype, itemsize):
    source = NativeGeometry("L0C", "f32", "NZ", 3 * 16 * 4, rows=2, row_stride=32 * 16 * 4)
    output = NativeGeometry("GM", dtype, "linear", 17 * itemsize, rows=3, row_stride=19 * itemsize)
    assert native_contract("fix", (source, output), (3, 17, 32)) == ("FIX", ("R", "W"))
    bad = NativeGeometry("L0C", "f32", "NZ", 3 * 16 * 4, rows=1)
    with pytest.raises(CompileError, match="exact f32 NZ reads"):
        native_contract("fix", (bad, output), (3, 17, 32))


@pytest.mark.parametrize("bank", ["UB", "GM"])
def test_ordinary_copy_still_rejects_implicit_conversion(bank):
    program = T.from_source(f"""
        def invalid(A:T.Tensor((16,16),T.f32), C:T.Tensor((16,16),T.f16,access="write")):
            with T.Kernel(1,kind="vector"):
                x=T.alloc_ub((16,16),T.f32)
                T.copy({"x" if bank == "UB" else "A"},C)
    """)
    with pytest.raises(CompileError, match="explicit cast"):
        program.to_ir()


def test_zero_padding_cannot_hide_an_undersized_nz_destination():
    program = T.from_source("""
        def invalid(A:T.Tensor((17,19),T.f16)):
            with T.Kernel(1,kind="cube"):
                x=T.alloc_l1((16,16),T.f16)
                T.copy(A,x,pad_value=0)
    """)
    with pytest.raises(CompileError, match="pad_value=0"):
        program.to_ir()


def test_constexpr_boolean_policy_short_circuits_undefined_branch():
    program = T.from_source("""
        def policy(A:T.Tensor((16,16),T.f16)):
            with T.Kernel(1,kind="cube"):
                enabled = True or undefined
                assert enabled, "short circuit must remain constexpr"
                x=T.alloc_l1((16,16),T.f16)
                T.copy(A,x)
    """)
    program.to_ir().verify()


@pytest.mark.parametrize("mapping", ["group % 2 + tile * GROUPS", "group + tile * (GROUPS - 1)"])
def test_group_dependent_first_iteration_cannot_hide_owner_collisions(mapping):
    from ascendflow import compile
    from tests.models.structured_events import EventModelError

    text = EXAMPLE.read_text()
    text = text[text.index("def gemm(") :].replace(
        "group + tile * GROUPS", "(" + mapping + ") % count"
    )
    source = T.from_source(text, functions={"l2_tile": l2_tile}, disjoint_external=True).specialize(
        M=64, N=80, K=16, BM=16, BN=16, BK=16, CK=16, GROUPS=3
    )
    memory = collect_native_memory(compile(source).plan)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(memory, groups=(0, 1, 2))


def test_distinct_loop_names_do_not_hide_overlapping_output_rectangles(tmp_path):
    path = tmp_path / "overlap.py"
    path.write_text("""
from __future__ import annotations
from ascendflow import language as T
@T.prim_func(disjoint_external=True)
def overlap(A:T.Tensor((16,16),T.f16),B:T.Tensor((16,16),T.f16),C:T.Tensor((R,32),T.f16,access="write"),R:T.int64(minimum=32,multiple_of=16)):
    with T.Kernel(1,kind="cube"):
        a1=T.alloc_l1((16,16),T.f16)
        b1=T.alloc_l1((16,16),T.f16)
        a0=T.alloc_l0a((16,16),T.f16)
        b0=T.alloc_l0b((16,16),T.f16)
        c0=T.alloc_l0c((16,16),T.f32)
        for phase in T.unroll(2):
            for row in T.serial(R//16-1):
                T.copy(A,a1)
                T.copy(B,b1)
                aa=T.copy(a1,a0)
                bb=T.copy(b1,b0)
                T.matmul(aa,bb,accum=c0,initialize=True)
                T.copy(c0,C[(row+phase)*16:(row+phase+1)*16,0:16])
""")
    output = tmp_path / "build"
    build_file(path, output, entry="overlap", timeout_s=300)
    original = collect_native_memory(parse((output / "before.xdsl").read_text()))
    final = collect_native_memory(parse((output / "after.xdsl").read_text()))
    for rows in (32, 48, 80):
        check_structured_events(final, {"R": rows}, logical_reference=original)


def test_long_gm_leading_dimension_uses_exact_row_descriptors(tmp_path):
    build_file(
        EXAMPLE,
        tmp_path,
        entry="gemm",
        parameters={
            "M": 3,
            "N": 5,
            "K": 65537,
            "BM": 16,
            "BN": 16,
            "BK": 16,
            "CK": 16,
            "GROUPS": 1,
        },
        timeout_s=300,
    )
    asc = (tmp_path / "gemm.asc").read_text()
    assert "npu_row * 131074" in asc


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_strided_load3d_uses_parent_height_and_shared_operand_retirement(tmp_path, dtype):
    build_file(
        EXAMPLE,
        tmp_path,
        entry="gemm",
        parameters={
            "M": 32,
            "N": 64,
            "K": 64,
            "BM": 16,
            "BN": 32,
            "BK": 64,
            "CK": 32,
            "COL_PARTS": 2,
            "GROUPS": 2,
            "DTYPE": dtype,
        },
        timeout_s=300,
    )
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1))
    assert trace.bytes_read and trace.consumptions
    assert any(t.contract.native == "load3d_b_nn" for t in final.templates)
    from ascendflow.dialects.physical import Load3DConfigOp

    (config,) = [op for op in final.module.walk() if isinstance(op, Load3DConfigOp)]
    assert config.b_width.data == 64


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("l2_tile", [1, 2, 4])
def test_l2_tiles_preserve_all_output_bytes_and_owner_retirement(tmp_path, dtype, l2_tile):
    build_file(
        EXAMPLE,
        tmp_path,
        entry="gemm",
        parameters={
            "M": 64,
            "N": 128,
            "K": 32,
            "BM": 16,
            "BN": 16,
            "BK": 32,
            "CK": 16,
            "COL_PARTS": 2,
            "L2_TILE": l2_tile,
            "GROUPS": 3,
            "DTYPE": dtype,
        },
        timeout_s=300,
    )
    original = collect_native_memory(parse((tmp_path / "before.xdsl").read_text()))
    final = collect_native_memory(parse((tmp_path / "after.xdsl").read_text()))
    trace = check_structured_events(final, logical_reference=original, groups=(0, 1, 2))
    assert trace.bytes_written and trace.consumptions


@pytest.mark.parametrize(
    "before,after",
    [
        ("(index // tile_n) % tile_m", "(index // tile_n) % 1"),
        ("index % tile_n", "index % 1"),
    ],
)
def test_swizzle_collisions_are_not_an_ownership_contract(before, after):
    from dataclasses import replace

    from ascendflow import compile
    from tests.models.structured_events import EventModelError

    text = EXAMPLE.read_text()
    text = text[text.index("def gemm(") :]
    assert before in l2_tile.source
    broken = replace(l2_tile, source=l2_tile.source.replace(before, after))
    source = T.from_source(text, functions={"l2_tile": broken}, disjoint_external=True).specialize(
        M=64, N=128, K=16, BM=16, BN=16, BK=16, CK=16, GROUPS=3, L2_TILE=2
    )
    memory = collect_native_memory(compile(source).plan)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(memory, groups=(0, 1, 2))
