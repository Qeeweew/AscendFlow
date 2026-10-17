"""Runtime prefix source contracts and scalar expression identities."""

import pytest

from ascendflow import compile
from ascendflow.diagnostics import CompileError
from ascendflow.frontend import from_source
from examples.vector.prefix import scale_prefix


def test_incorrect_valid_expression_is_not_treated_as_padding():
    definition = from_source("""
        def invalid(X: N.Tensor((n,), N.f32), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for t in N.Pipelined(N.ceildiv(n, 64), stages=1):
                    valid = N.min(64, n - t * 64 - 1)
                    N.copy(X[t * 64:(t + 1) * 64], x[t % 2, :], valid=valid)
    """)
    with pytest.raises(CompileError, match="valid prefix native addresses are not supported"):
        compile(definition)


def test_uniform_scalar_codegen_has_no_exponential_ternary_expansion():
    from ascendflow.scalar import Expr, bindings, expression

    value = Expr("argument", ("a0",))
    for i in range(1, 32):
        value = expression("min", value, Expr("argument", (f"a{i}",)))
    names, lines = bindings([value])
    assert len(lines) == 31
    assert len("\n".join(lines)) < 6000
    assert len(names[value]) < 30


def test_equivalent_scalar_bound_does_not_bypass_unsupported_prefix_contract():
    source = scale_prefix.source.replace(
        "N.ceildiv(length, B)", "N.min(N.ceildiv(length, B), N.ceildiv(length, B))"
    )
    equivalent = from_source(source, disjoint_external=True)
    for definition in (scale_prefix, equivalent):
        definition.to_ir().verify()
        with pytest.raises(CompileError, match="valid prefix native addresses are not supported"):
            compile(definition)
