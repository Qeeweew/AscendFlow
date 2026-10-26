"""Typed integer Scalar stores retain exact widths and ordinary UB lifetimes."""
import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.target.native import NativeGeometry, native_contract
from tests.models.structured_events import _Interpret, check_structured_events


@pytest.mark.parametrize("dtype,width", [("i32", 4), ("i64", 8)])
def test_integer_store_has_typed_scalar_effects_and_mte3_retirement(monkeypatch, dtype, width):
    result = compile(N.from_source(f"""
def write(Meta:N.Tensor((8,),N.{dtype}),Out:N.Tensor((3,8),N.{dtype},access="write",write_disjoint=True)):
    with N.Kernel(1,kind="vector"):
        local=N.alloc_ub((8,),N.{dtype})
        for generation in N.Pipelined(3):
            with N.Stage(0):
                for lane in N.serial(8):
                    value=N.load_index(Meta[lane:lane+1])
                    N.store_index(value-generation,out=local[lane:lane+1])
                N.copy(local,Out[generation,:])
""", disjoint_external=True))
    facts = collect_native_memory(result.plan)
    stores = [t for t in facts.templates if t.spec.native == "index_store"]
    assert len(stores) == 1
    assert stores[0].spec.modes == ("W",) and not stores[0].spec.implicit_completion
    assert stores[0].accesses[0].address.geometry.nbytes == width
    assert f"volatile int{width * 8}_t*" in result.asc
    assert "asc_sync_notify(PIPE_S, PIPE_MTE3" in result.asc
    original = _Interpret.native

    def metadata(self, operation, values, *args):
        original(self, operation, values, *args)
        if operation.native.data == "index_load":
            values[operation.value] = -17

    monkeypatch.setattr(_Interpret, "native", metadata)
    check_structured_events(facts, logical_reference=collect_native_memory(parse(result.before_ir)),
                            invocations=2)


@pytest.mark.parametrize("dtype,value", [("i32", -(1 << 31)), ("i32", (1 << 31) - 1),
                                         ("i64", -(1 << 63)), ("i64", (1 << 63) - 1)])
def test_integer_store_preserves_signed_endpoints(dtype, value):
    result = compile(N.from_source(f"""
def endpoint(Out:N.Tensor((8,),N.{dtype},access="write")):
    with N.Kernel(1,kind="vector"):
        local=N.alloc_ub((8,),N.{dtype})
        for lane in N.serial(8):
            N.store_index({value},out=local[lane:lane+1])
        N.copy(local,Out)
"""))
    assert ("INT64_MIN" if value == -(1 << 63) else str(abs(value))) in result.asc


@pytest.mark.parametrize("dtype,value", [("i32", -(1 << 31) - 1), ("i32", 1 << 31),
                                         ("i64", -(1 << 63) - 1), ("i64", 1 << 63)])
def test_integer_store_rejects_static_signed_overflow(dtype, value):
    with pytest.raises(CompileError, match="fit the destination signed dtype"):
        compile(N.from_source(f"""
def invalid(Out:N.Tensor((8,),N.{dtype},access="write")):
    with N.Kernel(1,kind="vector"):
        local=N.alloc_ub((8,),N.{dtype})
        N.store_index({value},out=local[0:1])
"""))


@pytest.mark.parametrize("bank,dtype,nbytes", [("GM", "i32", 4), ("UB", "f32", 4),
                                              ("UB", "i32", 8), ("UB", "i64", 4)])
def test_integer_store_rejects_incorrect_native_footprints(bank, dtype, nbytes):
    with pytest.raises(CompileError, match="exactly one int32/int64 UB"):
        native_contract("index_store", (NativeGeometry(bank, dtype, "linear", nbytes),), ())


def test_integer_fill_is_not_a_runtime_integer_store():
    with pytest.raises(CompileError, match="fill requires"):
        compile(N.from_source("""
def invalid(Meta:N.Tensor((8,),N.i32)):
    with N.Kernel(1,kind="vector"):
        local=N.alloc_ub((8,),N.i32)
        value=N.load_index(Meta[:1])
        N.fill(local,value)
"""))
