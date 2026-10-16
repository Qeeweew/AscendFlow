"""DSL compile-time assertions are strict and survive host Python optimization."""

import subprocess
import sys
from dataclasses import replace

import pytest
from xdsl.dialects import arith, scf

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.source import PredicateOp
from xdsl_npu.frontend import FrontendError

SOURCE = """\
def asserted(n:N.int64, B:N.constexpr=64):
    ASSERTION
    with N.Kernel():
        x=N.alloc_ub((B,),N.f32)
        for i in N.Pipelined(n,stages=1):
            N.fill(x,1.0)
"""


def definition(assertion, **kwargs):
    return N.from_source(
        SOURCE.replace("ASSERTION", assertion), filename="assert_contract.py", **kwargs
    )


def test_successful_assertion_does_not_emit_a_guard_assumption_or_change_native_work():
    checked = definition('assert B % 64 == 0, "tile alignment"')
    unchecked = definition("pass")
    source = checked.to_ir()
    assert not any(isinstance(op, (arith.CmpiOp, scf.IfOp, PredicateOp)) for op in source.walk())
    baseline, result = compile(unchecked, timeout_s=5), compile(checked, timeout_s=5)
    assert result.asc == baseline.asc
    assert result.report["native_templates"] == baseline.report["native_templates"] == 1


@pytest.mark.parametrize("message", [None, '"positive tile required"', '""'])
def test_false_assertion_aborts_with_the_original_source_location(message):
    assertion = "assert B > 0" + ("" if message is None else ", " + message)
    contract = definition(assertion).specialize(B=0)
    contract = replace(contract, first_line=27)
    with pytest.raises(FrontendError) as caught:
        compile(contract, timeout_s=5)
    diagnostic = str(caught.value)
    assert diagnostic.startswith("assert_contract.py:28:5: compile-time assertion failed")
    if message is not None:
        assert diagnostic.endswith(": " + message[1:-1])


@pytest.mark.parametrize("condition", ["0", "1", "1.0", "None", '"true"', "(True,)", "N.f32"])
def test_assertion_condition_is_strict_bool_and_never_implicit_truthiness(condition):
    with pytest.raises(FrontendError, match="assert requires a constexpr bool"):
        definition("assert " + condition).to_ir()


@pytest.mark.parametrize("condition", ["n", "n > 0", "n % 64 == 0", "i == 0"])
def test_runtime_values_cannot_become_assertions_assumptions_or_loop_guards(condition):
    if condition == "i == 0":
        contract = definition("pass")
        contract = replace(
            contract,
            source=contract.source.replace(
                "N.fill(x,1.0)", "assert i == 0\n            N.fill(x,1.0)"
            ),
        )
    else:
        contract = definition("assert " + condition)
    with pytest.raises(FrontendError, match="runtime assertions are unsupported"):
        compile(contract, timeout_s=5)


@pytest.mark.parametrize("condition", ["True", "False"])
@pytest.mark.parametrize("message", ["None", "False", "1", "1.0", "('static',)", "n", "n == 0"])
def test_assertion_message_is_static_str_even_when_the_condition_succeeds(condition, message):
    with pytest.raises(FrontendError, match="assert message must be a constexpr str"):
        definition(f"assert {condition}, {message}").to_ir()


def test_captured_static_messages_and_boolean_specializations_are_frozen():
    contract = definition("assert ENABLED, MESSAGE", constants={"ENABLED": True, "MESSAGE": "ok"})
    contract.to_ir()
    contract = definition("assert B, MESSAGE", constants={"MESSAGE": "enabled is required"})
    contract = replace(contract, source=contract.source.replace("((B,),N.f32)", "((64,),N.f32)"))
    contract.specialize(B=True).to_ir()
    with pytest.raises(FrontendError, match="compile-time assertion failed: enabled is required"):
        contract.specialize(B=False).to_ir()


def test_constexpr_control_selects_assertions_without_changing_runtime_branch_semantics():
    contract = definition(
        "if B == 64:\n        assert True\n    else:\n        assert False, 'unsupported tile'"
    )
    contract.to_ir()
    with pytest.raises(FrontendError, match="compile-time assertion failed: unsupported tile"):
        contract.specialize(B=128).to_ir()


def test_decorated_assertion_diagnostic_refers_to_captured_file_and_line():
    @N.prim_func
    def invalid(B: N.constexpr = 64):
        assert B == 128, "requires 128"
        with N.Kernel():
            pass

    expected_line = invalid.first_line + 2
    with pytest.raises(
        FrontendError,
        match=rf"test_frontend_assert.py:{expected_line}:5: compile-time assertion failed: requires 128",
    ):
        invalid.to_ir()


def test_host_optimization_cannot_remove_a_captured_dsl_assertion(tmp_path):
    script = tmp_path / "optimized_dsl_assert.py"
    script.write_text("""from xdsl_npu import language as N
from xdsl_npu.frontend import FrontendError

@N.prim_func
def checked(B:N.constexpr=64):
    assert B == 128, "retained under optimization"
    with N.Kernel():
        pass

try:
    checked.to_ir()
except FrontendError as error:
    print(error)
else:
    raise RuntimeError("DSL assertion was removed")
""")
    result = subprocess.run(
        [sys.executable, "-O", str(script)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert (
        f"{script}:6:5: compile-time assertion failed: retained under optimization" in result.stdout
    )


def test_bad_assertion_does_not_mutate_a_previously_returned_source_module():
    contract = definition("assert B == 64")
    source = contract.to_ir()
    before = dump(source)
    with pytest.raises(FrontendError, match="compile-time assertion failed"):
        contract.specialize(B=128).to_ir()
    assert dump(source) == before
