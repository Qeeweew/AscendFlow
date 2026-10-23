"""b16 Vector Dup: IEEE bits, exact count, complete writes and closed masks."""

import re

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.frontend import FrontendError
from ascendflow.target.ascend910b import Engine
from ascendflow.target.native import NativeGeometry, native_contract
from ascendflow.target.packets import MaskStep
from tests.fixtures.b16_fill import b16_fill
from tests.models.structured_events import check_structured_events


@pytest.mark.parametrize(
    "dtype,value,expected",
    (
        ("f16", 0.0, 0x0000),
        ("bf16", -0.0, 0x8000),
        ("f16", 1.0 + 2.0**-11, 0x3C00),
        ("f16", 1.0 + 3.0 * 2.0**-11, 0x3C02),
        ("bf16", 1.0 + 2.0**-8, 0x3F80),
        ("bf16", 1.0 + 3.0 * 2.0**-8, 0x3F82),
        ("f16", 2.0**-24, 0x0001),
        ("f16", 2.0**-25, 0x0000),
        ("bf16", 2.0**-133, 0x0001),
        ("bf16", 2.0**-134, 0x0000),
        ("f16", 65520.0, 0x7C00),
        ("f16", -65520.0, 0xFC00),
        ("bf16", float.fromhex("0x1.fffffep+127"), 0x7F80),
        ("bf16", float("inf"), 0x7F80),
        ("f16", float("-inf"), 0xFC00),
    ),
)
def test_generated_fill_bits_and_element_count_match_ieee_reference(dtype, value, expected):
    function = N.from_source(
        f"""
def fill_word(Y:N.Tensor((96,),N.{dtype},access="write")):
    with N.Kernel(1,kind="vector"):
        work=N.alloc_ub((96,),N.{dtype})
        N.fill(work[16:64],VALUE)
        N.copy(work[16:64],Y[16:64])
""",
        constants={"VALUE": value},
    )
    result = compile(function)
    calls = re.findall(r"asc_duplicate\([^;]*, uint16_t\((\d+)\), uint32_t\((\d+)\)\);", result.asc)
    assert calls == [(str(expected), "48")]
    assert "__ubuf__ uint16_t*" in result.asc
    memory = collect_native_memory(parse(result.before_ir))
    fill = next(t for t in memory.templates if t.operation.native.data == "fill")
    assert fill.spec.engine is Engine.V and fill.spec.modes == ("W",)
    assert fill.spec.operands[0].dtype == dtype
    assert fill.spec.operands[0].total_nbytes == 96
    assert fill.spec.packet.closed_mask
    assert MaskStep.SET_COUNT in fill.spec.packet.mask_steps
    assert fill.spec.packet.restores_normal_mode


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_b16_fill_complete_write_and_reuse_use_the_existing_synchronization_contract(dtype):
    result = compile(b16_fill.specialize(DTYPE=dtype))
    memory = collect_native_memory(result.plan)
    original = collect_native_memory(parse(result.before_ir))
    execution = check_structured_events(memory, logical_reference=original)
    assert execution.bytes_read and execution.bytes_written


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_b16_masked_fill_is_not_silently_enabled(dtype):
    source = f"""
def masked_fill():
    with N.Kernel():
        x=N.alloc_ub((64,),N.{dtype})
        N.fill(x,0.0,mask=N.arange(64)<32)
"""
    with pytest.raises(FrontendError, match="requires f32"):
        N.from_source(source).to_ir()


@pytest.mark.parametrize(
    "geometry",
    (
        NativeGeometry("UB", "bf16", "linear", 48, 2, 64),
        NativeGeometry("GM", "f16", "linear", 96),
        NativeGeometry("UB", "i32", "linear", 96),
        NativeGeometry("UB", "bf16", "linear", 95),
    ),
)
def test_fill_does_not_enlarge_unsupported_footprints(geometry):
    with pytest.raises(CompileError, match="Vector fill needs"):
        native_contract("fill", (geometry,), ())
