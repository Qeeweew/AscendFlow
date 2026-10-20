"""Real AST runtime descriptors through native addresses and thin launch wrappers."""

from dataclasses import replace

import pytest

from ascendflow import compile
from ascendflow import language as T
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.physical import RuntimeSizeOp
from examples.vector.grouped import grouped_scale
from tests.support.host_launcher import host_launcher


@pytest.fixture(scope="module")
def grouped():
    return compile(grouped_scale)


@pytest.mark.parametrize("groups", [1, 3, 20])
def test_owner_count_is_policy_not_an_input_length_specialization(groups):
    artifact = compile(grouped_scale.specialize(GROUPS=groups))
    assert artifact.report["native_templates"] == 3
    assert {op.sym_name.data for op in artifact.plan.walk() if isinstance(op, RuntimeSizeOp)} == {
        "S"
    }


def test_actual_launcher_accepts_one_binary_for_different_lengths(grouped, tmp_path):
    library, launch = host_launcher(grouped, tmp_path)
    for index, length in enumerate((16, 48, 64, 4096, 131072), 1):
        assert launch(length) == 0
        assert library.dispatches() == index


def test_cyclic_count_cannot_cancel_an_invalid_unsigned_dividend():
    source = grouped_scale.source.replace("total - group", "((total - group) * 3) // 3")
    with pytest.raises(CompileError, match="source unsigned division"):
        compile(replace(grouped_scale, source=source))


@pytest.mark.parametrize("cancel", ["bad - bad", "bad * 0"])
def test_source_division_contract_survives_frontend_constant_cancellation(cancel):
    definition = T.from_source(f"""
        def bad_index():
            with T.Kernel():
                x = T.alloc_ub((64,))
                for t in T.Pipelined(2):
                    bad = (t - 1) // 2
                    offset = {cancel}
                    T.fill(x[offset:offset+64], 0.0)
""")
    with pytest.raises(CompileError, match="source unsigned division"):
        compile(definition)
