"""FP32 operand footprints, explicit IEEE mode and transposed b16 A."""

from pathlib import Path

import pytest

from ascendflow.build import build_file
from ascendflow.diagnostics import CompileError
from ascendflow.target.native import NativeGeometry, native_contract


def test_fp32_mad_cannot_use_b16_byte_footprints():
    a = NativeGeometry("L0A", "f32", "ZZ", 64 * 64 * 4)
    b = NativeGeometry("L0B", "f32", "ZN", 64 * 128 * 4)
    c = NativeGeometry("L0C", "f32", "NZ", 64 * 128 * 4)
    assert native_contract("mad", (a, b, c), (64, 128, 64, 1)) == ("M", ("R", "R", "W"))
    bad = NativeGeometry("L0A", "f32", "ZZ", 64 * 64 * 2)
    with pytest.raises(CompileError, match="MAD requires"):
        native_contract("mad", (bad, b, c), (64, 128, 64, 1))


@pytest.mark.parametrize("transpose", [False, True])
def test_fp32_operands_keep_32byte_nz_columns_and_ieee_mode(tmp_path, transpose):
    build_file(
        Path(__file__).parents[1] / "examples/gemm/batched.py",
        tmp_path,
        entry="batched_matmul",
        parameters={
            "M": 32,
            "N": 64,
            "K": 64,
            "GROUPS": 1,
            "DTYPE": "f32",
            "TRANSPOSE_B": transpose,
        },
        timeout_s=300,
    )
    source = (tmp_path / "batched_matmul.asc").read_text()
    assert "asc_set_fp32_mode()" in source
    assert "asc_sync_notify(PIPE_S, PIPE_M, event_t(0))" in source
    assert "__ca__ float*" in source and "__cb__ float*" in source
    assert "asc_enable_hf32" not in source


def test_nested_scalar_reader_is_covered_by_full_row_writer(tmp_path):
    path = tmp_path / "prefix.py"
    path.write_text("""
from ascendflow import language as N
@N.prim_func(disjoint_external=True)
def prefix(A:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        a=N.alloc_ub((64,),N.f32)
        b=N.alloc_ub((64,),N.f32)
        N.copy(A,a)
        for row in N.serial(64):
            for previous in N.serial(row):
                factor=N.load_scalar(a[previous:previous+1])
                N.mul(a,factor,out=b)
                N.copy(b,O)
""")
    build_file(path, tmp_path / "result", entry="prefix", timeout_s=300)


def test_scalar_scatter_to_ub_keeps_element_width_and_order(tmp_path):
    path = tmp_path / "scatter.py"
    path.write_text("""
from ascendflow import language as N
@N.prim_func(disjoint_external=True)
def scatter(A:N.Tensor((64,),N.f32),O:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        b=N.alloc_ub((64,),N.f32)
        for i in N.serial(64):
            value=N.load_scalar(A[i:i+1])
            N.store_scalar(value,out=b[i:i+1])
        N.mul(b,2.0,out=b)
        N.copy(b,O)
""")
    build_file(path, tmp_path / "result", entry="scatter", timeout_s=300)
    asc = (tmp_path / "result/scatter.asc").read_text()
    assert "volatile float*" in asc and "= npu_value_" in asc
    assert "asc_sync_notify(PIPE_S, PIPE_V" in asc
    with pytest.raises(CompileError, match="one f32 UB"):
        native_contract("scalar_store", (NativeGeometry("GM", "f32", "linear", 4),), ())


def test_scalar_store_loop_then_scalar_load_needs_no_scalar_pipe_fence(tmp_path):
    path = tmp_path / "roundtrip.py"
    path.write_text("""
from ascendflow import language as N
@N.prim_func(disjoint_external=True)
def roundtrip(A:N.Tensor((64,),N.f32),O:N.Tensor((64,64),N.f32,access="write")):
    with N.Kernel(1,kind="vector"):
        b=N.alloc_ub((64,),N.f32)
        ones=N.alloc_ub((64,),N.f32)
        out=N.alloc_ub((64,),N.f32)
        N.fill(ones,1.0)
        for i in N.serial(64):
            value=N.load_scalar(A[i:i+1])
            N.store_scalar(value,out=b[i:i+1])
        for j in N.serial(64):
            value=N.load_scalar(b[j:j+1])
            N.mul(ones,value,out=out)
            N.copy(out,O[j,:])
""")
    build_file(path, tmp_path / "result", entry="roundtrip", timeout_s=300)
    asc = (tmp_path / "result/roundtrip.asc").read_text()
    assert "asc_sync_pipe(PIPE_S)" not in asc
