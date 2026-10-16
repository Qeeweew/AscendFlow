"""Generic f32 fill literals through source and both physical checkers."""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest
from xdsl.dialects.builtin import FloatAttr, FloatData, f16, f32, f64
from xdsl.utils.exceptions import VerifyException

from tests.report_contract import semantic_report
from xdsl_npu import compile, verify_plan
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import parse
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.dialects.source import ElementOp
from xdsl_npu.frontend import FrontendError
from xdsl_npu.target.scalars import round_f32

POS_INF, NEG_INF = float("inf"), float("-inf")


@N.prim_func
def infinite_fill(Y: N.Tensor((2, 64), N.f32, access="write")):
    """Primitive device fixture: Y[0] is +Inf and Y[1] is -Inf."""
    with N.Kernel(1, kind="vector"):
        values = N.alloc_ub((2, 64), N.f32)
        for task in N.Pipelined(1, stages=2):
            with N.Stage(0):
                N.fill(values[0, :], POS_INF)
                N.fill(values[1, :], NEG_INF)
            with N.Stage(1):
                N.copy(values[0, :], Y[0, :])
                N.copy(values[1, :], Y[1, :])


def scalar_source(value, operation="fill", dtype="f32"):
    statement = "N.fill(x, VALUE)" if operation == "fill" else f"N.{operation}(x, VALUE, out=x)"
    return N.from_source(
        f"""
        def scalar_constant():
            with N.Kernel():
                x = N.alloc_ub((64,), N.{dtype})
                for task in N.Pipelined(1, stages=1):
                    {statement}
        """,
        constants={"VALUE": value},
    )


def physical_scalar(value, *, symbolic, native="fill"):
    kernel = Kernel("physical_scalar")
    storage = kernel.buffer("values", "UB", 256)
    kernel.loop(1, symbolic=symbolic)
    tile = kernel.view(storage, 256)
    if native != "fill":
        kernel.task("initialize", "fill", tile)
    if native == "stat_fill":
        refs, params = (kernel.view(storage, 4),), (1,)
    else:
        refs, params = ((tile,) if native == "fill" else (tile, tile)), ()
    kernel.task("constant", native, *refs, params=params, scalar=value)
    return kernel.finish()


def scalar_task(module):
    return next(op for op in module.walk() if isinstance(op, TaskOp) and op.uid.data == "constant")


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.fixture(scope="module")
def infinite_artifact():
    return compile(infinite_fill)


def test_real_dsl_infinities_survive_ir_and_emit_valid_float_builtins(infinite_artifact):
    artifact = infinite_artifact
    source = parse(artifact.source_ir)
    literals = [op.scalar for op in source.walk() if isinstance(op, ElementOp)]
    assert [bits(value.value.data) for value in literals] == [0x7F800000, 0xFF800000]
    for snapshot in (artifact.before_ir, artifact.plan_ir):
        module = parse(snapshot)
        tasks = [op for op in module.walk() if isinstance(op, TaskOp)]
        fills = [op for op in tasks if op.native.data == "fill"]
        assert len(fills) == 2
        assert [bits(op.scalar.value.data) for op in fills] == [0x7F800000, 0xFF800000]
        assert all(op.scalar.type == f32 for op in fills)
        assert not any(op.native.data == "fill_neg_inf" for op in tasks)
    assert semantic_report(verify_plan(parse(artifact.plan_ir))) == semantic_report(artifact.report)
    assert artifact.asc.count("__builtin_inff()") == 2
    assert "-__builtin_inff()" in artifact.asc
    assert "float(inf)" not in artifact.asc and "float(-inf)" not in artifact.asc


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("value,expected", [(POS_INF, 0x7F800000), (NEG_INF, 0xFF800000)])
def test_explicit_physical_fill_has_the_same_contract(symbolic, value, expected):
    artifact = compile(physical_scalar(value, symbolic=symbolic))
    task = scalar_task(parse(artifact.plan_ir))
    assert task.native.data == "fill"
    assert task.scalar.type == f32 and bits(task.scalar.value.data) == expected
    literal = "-__builtin_inff()" if expected == 0xFF800000 else " __builtin_inff()"
    assert literal in artifact.asc
    verify_plan(parse(artifact.plan_ir))


@pytest.mark.parametrize("dtype", [f16, f64])
def test_source_ir_cannot_forge_another_float_type(dtype):
    module = infinite_fill.to_ir()
    operation = next(op for op in module.walk() if isinstance(op, ElementOp))
    operation.attributes["scalar"] = FloatAttr(POS_INF, dtype)
    with pytest.raises(VerifyException, match="f32"):
        compile(module)


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("dtype", [f16, f64])
def test_physical_ir_requires_f32_in_both_checkers(symbolic, dtype):
    module = physical_scalar(0.0, symbolic=symbolic)
    scalar_task(module).attributes["scalar"] = FloatAttr(POS_INF, dtype)
    with pytest.raises(CompileError, match="native Vector scalar literals must be f32"):
        compile(module)


def test_source_nan_is_rejected_even_if_a_forged_ir_bypasses_language_construction():
    with pytest.raises(FrontendError, match="Vector scalar must be finite"):
        scalar_source(float("nan")).to_ir()
    module = infinite_fill.to_ir()
    operation = next(op for op in module.walk() if isinstance(op, ElementOp))
    operation.attributes["scalar"] = FloatAttr(float("nan"), f32)
    with pytest.raises(CompileError, match="Vector scalar must be finite"):
        compile(module)


@pytest.mark.parametrize("symbolic", [False, True])
def test_physical_fill_nan_is_rejected(symbolic):
    with pytest.raises(CompileError, match="Vector scalar must be finite"):
        compile(physical_scalar(float("nan"), symbolic=symbolic))


@pytest.mark.parametrize("value", [1e39, -1e39])
def test_finite_overflow_is_not_promoted_to_an_allowed_infinity(value):
    with pytest.raises(CompileError, match="outside finite f32 range"):
        round_f32(value, allow_infinity=True)
    with pytest.raises(FrontendError, match="outside finite f32 range"):
        scalar_source(value).to_ir()
    for symbolic in (False, True):
        module = physical_scalar(0.0, symbolic=symbolic)
        # Bypass FloatAttr's construction-time f32 rounding to exercise the
        # final checker on an explicitly forged, out-of-range finite literal.
        scalar_task(module).attributes["scalar"] = FloatAttr.new([FloatData(value), f32])
        with pytest.raises(CompileError, match="outside finite f32 range"):
            compile(module)


@pytest.mark.parametrize("value", [POS_INF, NEG_INF])
def test_nonfill_language_scalars_and_default_rounding_remain_finite(value):
    with pytest.raises(CompileError, match="Vector scalar must be finite"):
        round_f32(value)
    for operation in ("mul", "add"):
        with pytest.raises(FrontendError, match="Vector scalar must be finite"):
            scalar_source(value, operation).to_ir()
    with pytest.raises(FrontendError, match="requires f32"):
        scalar_source(value, dtype="f16").to_ir()


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("native", ["muls", "adds", "stat_fill"])
def test_other_physical_native_scalars_keep_their_finite_contract(symbolic, native):
    for value in (POS_INF, NEG_INF):
        with pytest.raises(CompileError, match="Vector scalar must be finite"):
            compile(physical_scalar(value, symbolic=symbolic, native=native))


def fill_literals(asc):
    values = re.findall(r"asc_duplicate\([^;\n]*, (.*?), uint32_t\(64\)\);", asc)
    assert values, "the fixture must contain actual generated f32 fill calls"
    return values


def test_generated_literals_preserve_ieee_bits_and_existing_finite_rounding(
    tmp_path, infinite_artifact
):
    # IEEE binary32 constants are an independent oracle for sign and the
    # unchanged round-to-nearest-even behavior at two adjacent midpoint ties.
    finite = [
        (-0.0, 0x80000000),
        (1.0 + 2.0**-24, 0x3F800000),
        (1.0 + 3.0 * 2.0**-24, 0x3F800002),
        (float.fromhex("0x1.fffffep+127"), 0x7F7FFFFF),
    ]
    literals = fill_literals(infinite_artifact.asc)
    expected = [0x7F800000, 0xFF800000]
    assert len(literals) == len(expected)
    for value, wanted in finite:
        artifact = compile(scalar_source(value))
        generated = fill_literals(artifact.asc)
        assert len(generated) == 1
        literals.extend(generated)
        expected.append(wanted)
        task = next(op for op in parse(artifact.plan_ir).walk() if isinstance(op, TaskOp))
        assert bits(task.scalar.value.data) == wanted
    compiler = shutil.which("c++")
    assert compiler is not None, "CPU fill-literal tests require a C++ compiler"
    source, binary = tmp_path / "fill_literals.cpp", tmp_path / "fill_literals"
    source.write_text(
        "#include <cstdint>\n#include <cstring>\n#include <cassert>\n"
        "int main() {\n"
        f"  const float values[] = {{{', '.join(literals)}}};\n"
        f"  const uint32_t expected[] = {{{', '.join(hex(value) for value in expected)}}};\n"
        f"  for (unsigned i = 0; i < {len(expected)}; ++i) {{\n"
        "    uint32_t actual;\n"
        "    std::memcpy(&actual, &values[i], sizeof(actual));\n"
        "    assert(actual == expected[i]);\n"
        "  }\n}\n"
    )
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=5)
