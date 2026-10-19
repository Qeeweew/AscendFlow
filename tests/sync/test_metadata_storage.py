"""Metadata's captured generation is distinct from its loaded index value."""

import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import _Interpret, check_compiled_events


@pytest.fixture
def metadata_values(monkeypatch):
    native = _Interpret.native

    def interpret(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Meta" and offset % 8 == 0
            values[operation.value] = (offset // 16) % 2

    monkeypatch.setattr(_Interpret, "native", interpret)


def test_unrelated_metadata_read_preserves_an_ordinary_two_buffer_pipeline(metadata_values):
    source = N.from_source(
        """
    def buffered(X:N.Tensor((n,64),N.f32),Y:N.Tensor((n,64),N.f32),
                 Meta:N.Tensor((n,),N.i64),
                 Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
        with N.Kernel():
            a=N.alloc_ub((2,64))
            b=N.alloc_ub((64,))
            result=N.alloc_ub((64,))
            for i in N.Pipelined(n,stages=2):
                with N.Stage(0):
                    unused=N.load_index(Meta[i:i+1],minimum=0,maximum=1)
                    N.copy(X[i,:],a[i%2,:])
                with N.Stage(1):
                    N.copy(Y[i,:],b)
                    N.add(a[i%2,:],b,out=result)
                    N.copy(result,Out[i,:])
    """,
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for count in (0, 1, 2, 3, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_a_loaded_index_still_retains_real_aliases_and_repeated_slot_values(metadata_values):
    source = N.from_source(
        """
    def indexed(X:N.Tensor((n,64),N.f32),Meta:N.Tensor((n,),N.i64),
                Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.int64):
        with N.Kernel():
            a=N.alloc_ub((2,64))
            for i in N.Pipelined(n,stages=1):
                slot=N.load_index(Meta[i:i+1],minimum=0,maximum=1)
                N.copy(X[i,:],a[slot,:])
                N.copy(a[slot,:],Out[i,:])
    """,
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for count in (0, 1, 2, 3, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)
