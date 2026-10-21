"""Capture identities retain the proof DAG and checked integer semantics."""

import shutil
import subprocess

import pytest

from ascendflow.analysis.integer_facts import IntegerFacts
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.scalar import INT64_MAX, Expr, bindings, expression, literal


@pytest.mark.parametrize("kind", ("floordiv", "ceildiv"))
def test_unit_division_renders_its_identity_without_erasing_unsigned_proof(kind):
    coordinate = Expr("argument", ("$g",))
    numerator = expression("sub", coordinate, literal(1))
    quotient = expression(kind, numerator, literal(1))
    assert quotient.c() == numerator.c()
    assert quotient.kind == kind and quotient.args[0] is numerator
    for g in (1, 2, INT64_MAX):
        assert quotient.evaluate({"$g": g}) == g - 1
    with pytest.raises(CompileError, match="nonnegative numerator"):
        quotient.evaluate({"$g": 0})
    # The final expression cancels algebraically. Its original unsigned
    # division must still reject a negative dividend at generation zero.
    canceled = expression("sub", quotient, numerator)
    assert (
        IntegerFacts({}).ordered_definedness(
            (
                Comparison(coordinate, ">=", literal(0)),
                Comparison(coordinate, "<=", literal(3)),
                Comparison(canceled, "==", literal(0)),
            )
        )
        is not None
    )
    assert quotient.c({quotient: "already_checked"}) == "already_checked"


def test_emitted_unit_identities_preserve_overflow_checks_and_short_circuit(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None, "scalar emission tests require a host C++ compiler"
    argument = Expr("argument", ("a",))
    numerator = expression("add", argument, literal(1))
    functions = []
    for kind in ("floordiv", "ceildiv", "rem"):
        value = expression(kind, numerator, literal(1))
        names, lines = bindings([value], checked=True)
        assert any("uniform sum overflow" in line for line in lines)
        functions.append(
            f"int {kind}(int64_t npu_size_a, int64_t& out) {{\n"
            + "\n".join(lines)
            + f"\nout = {names[value]};\nreturn 0;\n}}\n"
        )
    # Render directly inside a C short-circuit operand. The overflow-prone
    # numerator must stay unevaluated when the left operand is false.
    inline = expression("ceildiv", numerator, literal(1)).c({argument: "a"})
    source = tmp_path / "identities.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n"
        + "".join(functions)
        + "int main() {\n"
        + "for (int64_t a : {int64_t(0), int64_t(1), INT64_MAX-1}) {\n"
        + "  int64_t out = -1;\n"
        + "  assert(floordiv(a, out) == 0 && out == a+1);\n"
        + "  assert(ceildiv(a, out) == 0 && out == a+1);\n"
        + "  assert(rem(a, out) == 0 && out == 0);\n}\n"
        + "int64_t out = -1;\n"
        + "assert(floordiv(INT64_MAX, out) == -1 && out == -1);\n"
        + "assert(ceildiv(INT64_MAX, out) == -1 && out == -1);\n"
        + "assert(rem(INT64_MAX, out) == -1 && out == -1);\n"
        + "for (int64_t a : {int64_t(0), INT64_MAX}) {\n"
        + f"  assert((a < INT64_MAX && {inline} > a) == (a < INT64_MAX));\n"
        + "}\n}\n"
    )
    executable = tmp_path / "identities"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-include",
            "initializer_list",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)
