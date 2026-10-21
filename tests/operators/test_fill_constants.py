"""Generic f32 fill literals through source, final SSA and emitted C++."""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest
from xdsl.dialects.builtin import FloatAttr, f16, f32, f64
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import TaskOp
from ascendflow.dialects.source import ElementOp
from ascendflow.frontend import FrontendError
from ascendflow.target.scalars import round_f32

POS_INF, NEG_INF = float("inf"), float("-inf")


@N.prim_func
def infinite_fill(Y: N.Tensor((2, 64), N.f32, access="write")):
    """Primitive device fixture: Y[0] is +Inf and Y[1] is -Inf."""
    with N.Kernel(1, kind="vector"):
        values = N.alloc_ub((2, 64), N.f32)
        for task in N.Pipelined(1):
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
                for task in N.Pipelined(1):
                    {statement}
        """,
        constants={"VALUE": value},
    )


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.fixture(scope="module")
def infinite_artifact():
    return compile(infinite_fill)


@pytest.mark.parametrize("dtype", [f16, f64])
def test_source_ir_cannot_forge_another_float_type(dtype):
    module = infinite_fill.to_ir()
    operation = next(op for op in module.walk() if isinstance(op, ElementOp))
    operation.attributes["scalar"] = FloatAttr(POS_INF, dtype)
    with pytest.raises(VerifyException, match="f32"):
        compile(module)


def test_source_nan_is_rejected_even_if_a_forged_ir_bypasses_language_construction():
    with pytest.raises(FrontendError, match="Vector scalar must be finite"):
        scalar_source(float("nan")).to_ir()
    module = infinite_fill.to_ir()
    operation = next(op for op in module.walk() if isinstance(op, ElementOp))
    operation.attributes["scalar"] = FloatAttr(float("nan"), f32)
    with pytest.raises(CompileError, match="Vector scalar must be finite"):
        compile(module)


@pytest.mark.parametrize("value", [POS_INF, NEG_INF])
def test_nonfill_language_scalars_and_default_rounding_remain_finite(value):
    with pytest.raises(CompileError, match="Vector scalar must be finite"):
        round_f32(value)
    for operation in ("mul", "add"):
        with pytest.raises(FrontendError, match="Vector scalar must be finite"):
            scalar_source(value, operation).to_ir()
    with pytest.raises(FrontendError, match="requires f32"):
        scalar_source(value, operation="mul", dtype="f16").to_ir()


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
    assert artifact.asc.count("__builtin_inff()") == 2
    assert "-__builtin_inff()" in artifact.asc
    assert "float(inf)" not in artifact.asc and "float(-inf)" not in artifact.asc
