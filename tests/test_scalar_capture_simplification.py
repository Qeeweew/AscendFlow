"""Capture identities retain the proof DAG and checked integer semantics."""

import re
import shutil
import subprocess

import pytest
from xdsl.dialects import arith

from xdsl_npu import language as N
from xdsl_npu.indexing import check_axis
from xdsl_npu.ir import TaskOp, dump
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scalar import INT64_MAX, Expr, bindings, expression, literal
from xdsl_npu.scope_model import extract_scopes


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
    with pytest.raises(CompileError, match="unproven nonnegative dividend"):
        check_axis(canceled, 1, literal(1), [(coordinate, literal(0), literal(3))], {})
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


@pytest.fixture(params=("floor", "ceil"))
def captured_source(request):
    head = "(outer//2)%8//1" if request.param == "floor" else "N.ceildiv((outer//2)%8,1)"
    source = N.from_source(f"""
        def nested(X:N.Tensor((n,8,2,64),N.f32), n:N.int64):
            with N.Kernel():
                a=N.alloc_ub((2,64))
                for outer in N.Pipelined(n*16,stages=1):
                    for part in N.Pipelined(2,stages=1):
                        N.copy(X[outer//16,{head},part,:],a[part,:])
    """).to_ir()
    before = dump(source)
    physical = lower_source(source)
    assert dump(source) == before
    return physical


def test_real_source_capture_removes_only_the_unit_quotient(captured_source, monkeypatch):
    from xdsl_npu import scope_model

    def fixed_child_range(limit):
        assert limit <= 2, "runtime parent length must stay symbolic"
        return range(limit)

    monkeypatch.setattr(scope_model, "range", fixed_child_range, raising=False)
    physical = captured_source
    physical.verify()
    before = dump(physical)
    program = extract_scopes(physical)
    (parent,) = program.parents.values()
    (leaf,) = program.leaves.values()
    assert parent.limit is not None and leaf.stream.count.data == 2
    assert sum(isinstance(op, TaskOp) for op in physical.walk()) == 1
    captured = next(capture for capture in leaf.captures if capture.root.sym_name.data == "X")
    coordinate = Expr("argument", ("$parent0",))
    code = captured.offset.c({coordinate: "outer"})
    assert not re.search(r"/ 1\b|\* 1\b|\+ 0\b", code)
    assert "/ 16" in code and "/ 2" in code and "% 8" in code
    # Independent row-major source indexing, including a large runtime epoch.
    for outer in (0, 1, 15, 16, 33, (1 << 44) - 1):
        expected = ((outer // 16) * 8 + (outer // 2) % 8) * 2 * 64 * 4
        assert captured.offset.evaluate({"$parent0": outer}) == expected
    assert dump(physical) == before


def test_actual_unit_division_still_rejects_a_negative_unsigned_dividend(captured_source):
    program = extract_scopes(captured_source)
    (leaf,) = program.leaves.values()
    unit = next(
        op
        for op in leaf.call.body.block.ops
        if isinstance(op, (arith.DivUIOp, arith.CeilDivUIOp))
        and isinstance(op.rhs.owner, arith.ConstantOp)
        and op.rhs.owner.value.value.data == 1
    )
    negative = arith.SubiOp(leaf.call.generation, unit.rhs)
    unit.parent.insert_op_before(negative, unit)
    unit.operands = (negative.result, unit.rhs)
    with pytest.raises(CompileError, match="unproven nonnegative dividend"):
        extract_scopes(captured_source)
