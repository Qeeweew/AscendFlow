"""Nonnegative index proofs and CPU execution of checked emitted expressions."""

import itertools
import operator
import shutil
import subprocess

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.ir import Block

from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import RuntimeSizeOp
from xdsl_npu.dialects.source import SizeOp
from xdsl_npu.scalar import (
    INT64_MAX,
    Expr,
    Fact,
    bindings,
    expression,
    facts_from_sizes,
    from_ssa,
    infer,
    literal,
    positive,
    uniform,
)


def argument(name):
    return Expr("argument", (name,))


def size(kind, name, *, minimum=0, multiple=1):
    return kind.create(
        attributes={
            "sym_name": StringAttr(name),
            "minimum": IntAttr(minimum),
            "multiple": IntAttr(multiple),
        },
        result_types=[IndexType()],
    )


def ceil_quotient(a, b):
    quotient, remainder = divmod(a, b)
    return quotient + bool(remainder)


@pytest.mark.parametrize("kind", [SizeOp, RuntimeSizeOp])
def test_size_contract_uses_its_first_feasible_value(kind):
    sizes = [size(kind, "n", minimum=1, multiple=64), size(kind, "empty", multiple=64)]
    facts = facts_from_sizes([arith.ConstantOp.from_int_and_width(0, IndexType()), *sizes])
    assert facts["n"].minimum == 64
    assert facts["empty"].minimum == 0
    assert facts["n"].maximum <= INT64_MAX
    assert facts["n"].maximum % 64 == 0
    assert positive(expression("floordiv", argument("n"), literal(64)), facts)
    assert not positive(argument("empty"), facts)


def test_fact_bounds_and_duplicate_contracts_must_have_feasible_values():
    assert Fact(3, 4, 31) == Fact(4, 4, 28)
    with pytest.raises(CompileError, match="no feasible value"):
        Fact(5, 4, 7)
    with pytest.raises(CompileError, match="no feasible value"):
        facts_from_sizes([size(SizeOp, "n", minimum=INT64_MAX, multiple=2)])
    with pytest.raises(CompileError, match="contract disagrees"):
        facts_from_sizes([size(SizeOp, "n"), size(RuntimeSizeOp, "n", minimum=1)])


@pytest.mark.parametrize(
    "a,b",
    [
        (0, 1),
        (0, INT64_MAX),
        (1, 1),
        (6, 7),
        (7, 7),
        (8, 7),
        (INT64_MAX, 1),
        (INT64_MAX, 2),
        (INT64_MAX, INT64_MAX),
    ],
)
def test_division_semantics_match_independent_divmod_at_boundaries(a, b):
    quotient, remainder = divmod(a, b)
    expected = {"floordiv": quotient, "rem": remainder, "ceildiv": quotient + bool(remainder)}
    for kind, value in expected.items():
        expr = expression(kind, argument("a"), argument("b"))
        assert expr.evaluate({"a": a, "b": b}) == value
        assert expression(kind, literal(a), literal(b)) == literal(value)


@pytest.mark.parametrize("kind", ["floordiv", "rem", "ceildiv"])
def test_division_requires_positive_uniform_divisor(kind):
    a, b = argument("a"), argument("b")
    value = expression(kind, a, b)
    assert not uniform(value, {"a", "b"})
    assert not uniform(value, {"a": Fact(), "b": Fact(multiple=64)})
    assert uniform(value, {"a": Fact(), "b": Fact(minimum=1)})
    assert uniform(expression(kind, a, literal(3)), {"a"})
    assert not uniform(value, {"a": Fact(), "other": Fact(minimum=1)})
    for divisor in (0, -1):
        with pytest.raises(CompileError, match="divisor must be positive"):
            expression(kind, a, literal(divisor))
        with pytest.raises(CompileError, match="positive divisor"):
            value.evaluate({"a": 3, "b": divisor})
    with pytest.raises(CompileError, match="nonnegative numerator"):
        value.evaluate({"a": -1, "b": 2})


def test_fact_context_does_not_change_expression_identity():
    value = expression("floordiv", argument("a"), argument("b"))
    table = {value: "one expression"}
    assert not uniform(value, {"a", "b"})
    assert uniform(value, {"a": Fact(), "b": Fact(minimum=1)})
    assert table[expression("floordiv", argument("a"), argument("b"))] == "one expression"
    assert uniform(expression("add", argument("a"), literal(1)), {"a"})
    assert not uniform(expression("sub", argument("a"), literal(1)), {"a"})
    assert not uniform(literal(INT64_MAX + 1), {})


def test_inferred_bounds_and_multiples_cover_independent_integer_oracles():
    a, b = argument("a"), argument("b")
    facts = {"a": Fact(3, 4, 100), "b": Fact(1, 3, 30)}
    operations = {
        "add": operator.add,
        "mul": operator.mul,
        "min": min,
        "max": max,
        "floordiv": operator.floordiv,
        "rem": operator.mod,
        "ceildiv": ceil_quotient,
    }
    for kind, operation in operations.items():
        result = infer(expression(kind, a, b), facts)
        assert result is not None
        for left, right in itertools.product(range(4, 101, 4), range(3, 31, 3)):
            value = operation(left, right)
            assert result.minimum <= value
            assert result.maximum is None or value <= result.maximum
            assert value % result.multiple == 0
    # a >= 4 and b >= 3 do not imply a // b >= 1: b may be 30.
    assert infer(expression("floordiv", a, b), facts).minimum == 0
    unbounded = {"a": Fact(10), "b": Fact(1)}
    assert infer(expression("floordiv", a, b), unbounded).minimum == 0
    assert infer(expression("ceildiv", a, b), unbounded).minimum == 1
    assert infer(expression("rem", a, b), unbounded).maximum is None


def test_positive_derived_divisors_are_conservative():
    a, b = argument("a"), argument("b")
    facts = {"a": Fact(1, 64), "b": Fact(1)}
    assert positive(expression("min", a, b), facts)
    assert positive(expression("mul", a, b), facts)
    assert positive(expression("max", literal(1), argument("zero")), {"zero": Fact()})
    assert not positive(expression("floordiv", a, b), facts)
    assert not positive(expression("rem", a, b), facts)
    assert infer(argument("missing"), facts) is None
    assert infer(expression("floordiv", a, b), {"a": Fact(), "b": Fact()}) is None


@pytest.mark.parametrize(
    "operation,kind",
    [(arith.DivUIOp, "floordiv"), (arith.RemUIOp, "rem"), (arith.CeilDivUIOp, "ceildiv")],
)
def test_unsigned_ssa_extraction_checks_dynamic_divisors_even_with_cached_ir(operation, kind):
    a, b = size(SizeOp, "a"), size(RuntimeSizeOp, "b", minimum=1, multiple=8)
    value = operation(a.value, b.value).result
    facts, memo = facts_from_sizes([a, b]), {}
    result = from_ssa(value, facts=facts, memo=memo)
    assert result == expression(kind, argument("a"), argument("b"))
    for environment in (None, {"a": Fact(), "b": Fact()}):
        with pytest.raises(CompileError, match="uniform and provably positive"):
            from_ssa(value, facts=environment, memo=memo)
    constant = arith.ConstantOp.from_int_and_width(8, IndexType())
    assert from_ssa(operation(a.value, constant.result).result).kind == kind


def test_coordinate_numerator_uses_a_separate_range_environment():
    block = Block(arg_types=[IndexType()])
    divisor = size(SizeOp, "d", minimum=1)
    quotient = arith.DivUIOp(block.args[0], divisor.value).result
    facts = facts_from_sizes([divisor])
    value = from_ssa(quotient, coordinates={block.args[0]: "$g"}, facts=facts)
    assert value.c() == "(g / npu_size_d)"
    assert not uniform(value, facts)
    assert infer(value, {**facts, "$g": Fact(0, maximum=63)}).maximum == 63
    bad = arith.DivUIOp(divisor.value, block.args[0]).result
    with pytest.raises(CompileError, match="uniform and provably positive"):
        from_ssa(bad, coordinates={block.args[0]: "$g"}, facts=facts)


def test_unknown_kinds_fail_before_any_ceildiv_emission():
    with pytest.raises(CompileError, match="unsupported scalar expression kind: mystery"):
        Expr("mystery", (argument("a"), literal(2)))
    with pytest.raises(CompileError, match="unsupported scalar expression kind: mystery"):
        expression("mystery", literal(6), literal(2))


@pytest.fixture(scope="module")
def checked_expression_binary(tmp_path_factory):
    """Run actual generated C, including guards, with undefined-behavior checks."""
    a, b = argument("a"), argument("b")
    expressions = [
        [expression(kind, a, b) for kind in ("floordiv", "rem", "ceildiv")],
        [expression("add", a, b)],
        [expression("mul", a, b)],
        [expression("floordiv", a, expression("add", b, literal(1)))],
    ]
    branches = []
    for operation, values in enumerate(expressions):
        names, lines = bindings(values, checked=True)
        branches += [
            f"  if (operation == {operation}) {{",
            *lines,
            *(f"    out[{i}] = {names[value]};" for i, value in enumerate(values)),
            "    return 0;",
            "  }",
        ]
    directory = tmp_path_factory.mktemp("runtime_index_scalar")
    source, binary = directory / "checked.cpp", directory / "checked"
    source.write_text(
        "#include <cstdint>\n#include <cstdlib>\n#include <iostream>\n"
        "int evaluate(int operation, int64_t npu_size_a, int64_t npu_size_b, int64_t* out) {\n"
        "  if (npu_size_a < 0 || npu_size_b < 0) return -1;\n"
        + "\n".join(branches)
        + "\n  return -2;\n}\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc != 4) return 2;\n"
        "  int operation = std::atoi(argv[1]);\n"
        "  int64_t out[3] = {0, 0, 0};\n"
        "  int status = evaluate(operation, std::strtoll(argv[2], nullptr, 10),\n"
        "                        std::strtoll(argv[3], nullptr, 10), out);\n"
        "  std::cout << status;\n"
        "  if (!status) for (int i = 0; i < (operation == 0 ? 3 : 1); ++i)\n"
        "    std::cout << ' ' << out[i];\n"
        "  std::cout << '\\n';\n"
        "}\n"
    )
    compiler = shutil.which("c++")
    assert compiler is not None, "CPU scalar emission tests require a C++ compiler"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    def run(operation, a, b):
        result = subprocess.run(
            [str(binary), str(operation), str(a), str(b)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.stderr == ""
        return tuple(map(int, result.stdout.split()))

    return run


def test_checked_c_divmod_matches_independent_oracle_and_rejects_zero(checked_expression_binary):
    run = checked_expression_binary
    for a, b in [(0, 1), (6, 7), (7, 7), (8, 7), (INT64_MAX, 1), (INT64_MAX, 2), (1, INT64_MAX)]:
        quotient, remainder = divmod(a, b)
        assert run(0, a, b) == (0, quotient, remainder, quotient + bool(remainder))
    assert run(0, 0, 0) == (-1,)
    assert run(0, INT64_MAX, 0) == (-1,)
    assert run(0, -1, 1) == (-1,)
    assert run(0, 1, -1) == (-1,)


def test_checked_c_rejects_overflow_before_evaluating_arithmetic(checked_expression_binary):
    run = checked_expression_binary
    assert run(1, INT64_MAX - 1, 1) == (0, INT64_MAX)
    assert run(1, INT64_MAX, 1) == (-1,)
    assert run(2, INT64_MAX, 1) == (0, INT64_MAX)
    assert run(2, INT64_MAX, 2) == (-1,)
    assert run(2, 0, INT64_MAX) == (0, 0)
    assert run(3, INT64_MAX, INT64_MAX) == (-1,)
    assert run(3, INT64_MAX, INT64_MAX - 1) == (0, 1)


def test_shared_uniform_dag_is_emitted_once():
    denominator = expression("max", literal(1), argument("d"))
    quotient = expression("floordiv", argument("a"), denominator)
    names, lines = bindings([quotient, quotient, expression("rem", argument("a"), denominator)])
    assert len(lines) == 3
    assert len(names[quotient]) < 30
