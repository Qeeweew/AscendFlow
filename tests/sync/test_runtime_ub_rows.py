"""GM→UB row prefixes keep exact access counts and ordinary UB lifetimes."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import compile, language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from tests.models.structured_events import EventModelError, check_structured_events


def definition(dtype="bf16", *, strided=False, repeated=False):
    pitch = 48 if strided else 16
    column = 16 if strided else 0
    body = f"""
def rows(A:N.Tensor((n,{pitch}),N.{dtype}), Initial:N.Tensor((16,{pitch}),N.{dtype}),
         O:N.Tensor((3,16,16),N.{dtype},access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(1,kind="vector"):
        raw=N.alloc_ub((16,{pitch}),N.{dtype})
        N.copy(Initial,raw)
        loaded=N.copy(A[:16,{column}:{column+16}],raw[:,{column}:{column+16}],valid_rows=N.min(n,16))
        N.copy(loaded,O[0,:,:])
"""
    if repeated:
        body = body.replace(
            "        N.copy(Initial,raw)\n        loaded=",
            "        for task in N.Pipelined(3):\n            with N.Stage(0):\n                N.copy(Initial,raw)\n                loaded=",
        ).replace(
            "        N.copy(loaded,O[0,:,:])",
            "            with N.Stage(1):\n                N.copy(loaded,O[task,:,:])",
        )
    return N.from_source(body, disjoint_external=True)


@pytest.mark.parametrize("dtype", ("f16", "bf16", "f32", "i32", "i64"))
@pytest.mark.parametrize("strided", (False, True))
def test_import_preserves_runtime_count_and_pitched_holes(dtype, strided):
    result = compile(definition(dtype, strided=strided))
    before = collect_native_memory(parse(result.before_ir))
    after = collect_native_memory(result.plan)
    prefix = next(t for t in after.templates if t.spec.native == "copy_rows_prefix")
    assert prefix.spec.engine.value == "MTE2"
    itemsize = 8 if dtype == "i64" else 4 if dtype in {"f32", "i32"} else 2
    pitch, column = (48, 16) if strided else (16, 0)
    for count in (0, 1, 7, 15, 16, 17):
        checked = check_structured_events(after, {"n": count}, logical_reference=before, invocations=2)
        for command in checked.natives:
            if command.occurrence.operation.native.data != "copy_rows_prefix":
                continue
            source = [s for s in command.spans if s.root.sym_name.data == "A"]
            assert [(s.start, s.end) for s in source] == [
                ((row*pitch+column)*itemsize, (row*pitch+column+16)*itemsize)
                for row in range(min(count, 16))
            ]
            local = [s for s in command.spans if s.root.ref.type.bank.data == "UB"]
            assert sum(s.end-s.start for s in local) == min(count, 16)*16*itemsize
    assert "if ((" in result.asc and "asc_copy_gm2ub" in result.asc
    assert "uint16_t(" in result.asc if strided else "uint32_t((" in result.asc


@pytest.mark.parametrize("strided", (False, True))
def test_prefix_buffer_reuse_retires_previous_output(strided):
    result = compile(definition(strided=strided, repeated=True))
    for count in (0, 1, 15, 16, 17):
        check_structured_events(collect_native_memory(result.plan), {"n": count},
                                logical_reference=collect_native_memory(parse(result.before_ir)),
                                invocations=2)
    assert "PIPE_MTE3, PIPE_MTE2" in result.asc


@pytest.mark.parametrize("operand", (0, 1))
def test_both_prefix_addresses_must_match_the_instruction_count(operand):
    result = compile(definition())
    memory = collect_native_memory(parse(result.before_ir))
    prefix = next(t for t in memory.templates if t.spec.native == "copy_rows_prefix")
    address = prefix.accesses[operand].address.operation
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    address.parent.insert_op_before(zero, address)
    address.operands = (*address.operands[:-1], zero.result)
    with pytest.raises(CompileError, match="address and instruction counts disagree"):
        collect_native_memory(memory.module)


@pytest.mark.parametrize("count", (-1, 17))
def test_invalid_static_import_prefix_is_rejected(count):
    source = definition().source.replace("N.min(n,16)", str(count))
    with pytest.raises(CompileError, match="static row count.*declared capacity"):
        compile(N.from_source(source, disjoint_external=True))


def test_actual_prefix_capacity_remains_the_callers_contract():
    source = definition().source.replace("N.min(n,16)", "n")
    result = compile(N.from_source(source, disjoint_external=True))
    with pytest.raises(EventModelError, match="row count.*capacity"):
        check_structured_events(collect_native_memory(result.plan), {"n": 17})


def test_partial_import_does_not_define_unwritten_rows():
    source = definition().source.replace("        N.copy(Initial,raw)\n", "")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(N.from_source(source, disjoint_external=True))


def test_gm_ub_prefix_does_not_claim_unsupported_padding():
    source = definition().source.replace("valid_rows=N.min(n,16)", "valid_rows=N.min(n,16),pad_value=0")
    with pytest.raises(CompileError, match="valid_rows requires"):
        compile(N.from_source(source, disjoint_external=True))
