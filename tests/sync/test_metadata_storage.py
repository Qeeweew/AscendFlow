"""Metadata's captured generation is distinct from its loaded index value."""

import pytest

from ascendflow import language as N
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import _Interpret, check_compiled_events


@pytest.mark.parametrize("dtype,width", [("i32", 4), ("i64", 8)])
def test_local_integer_capture_keeps_snapshot_and_retires_each_generation(
    monkeypatch, dtype, width
):
    from ascendflow import compile
    from ascendflow.analysis.structured_memory import collect_native_memory
    from ascendflow.dialects.io import parse
    from tests.models.structured_events import check_structured_events

    native = _Interpret.native
    contents = {}
    inputs = [[[3, -1, 7][i]] * (32 // width) for i in range(3)]
    raw = b"".join(v.to_bytes(width, "little", signed=True) for row in inputs for v in row)

    def capture(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "copy":
            source, offset, _ = values[operation.refs[0]]
            destination, target, size = values[operation.refs[1]]
            if source.sym_name.data == "Meta":
                key = (self, invocation, group, actor, destination)
                contents.setdefault(key, bytearray(size))[target : target + 32] = raw[
                    offset : offset + 32
                ]
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            key = (self, invocation, group, actor, root)
            values[operation.value] = int.from_bytes(
                contents[key][offset : offset + width], "little", signed=True
            )

    monkeypatch.setattr(_Interpret, "native", capture)
    result = compile(
        N.from_source(
            f"""
def cached(Meta:N.Tensor((3,{32 // width}),N.{dtype}),X:N.Tensor((8,64),N.f32),
           Out:N.Tensor((3,64),N.f32,access="write",write_disjoint=True)):
    with N.Kernel():
        cache=N.alloc_ub(({32 // width},),N.{dtype})
        row=N.alloc_ub((64,),N.f32)
        for i in N.Pipelined(3):
            with N.Stage(0):
                N.copy(Meta[i,:],cache)
                index=N.load_index(cache[0:1])
                # Reusing the storage does not change the captured SSA value.
                N.copy(Meta[(i+1)%3,:],cache)
                N.fill(row,0.0)
                if index>=0:
                    N.copy(X[index,:],row)
                N.copy(row,Out[i,:])
""",
            disjoint_external=True,
        )
    )
    trace = check_structured_events(
        collect_native_memory(result.plan),
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
    )
    reads = [
        span.start // 256
        for command in trace.natives
        for span in command.spans
        if span.root.sym_name.data == "X"
    ]
    assert reads == [3, 7, 3, 7]
    assert f"__ubuf__ volatile int{width * 8}_t" in result.asc


def test_local_index_capture_still_requires_initialized_storage():
    from ascendflow import compile
    from ascendflow.diagnostics import CompileError

    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(
            N.from_source(
                """
def invalid(X:N.Tensor((8,64),N.f32),Out:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        cache=N.alloc_ub((8,),N.i32)
        row=N.alloc_ub((64,),N.f32)
        index=N.load_index(cache[0:1])
        N.copy(X[index,:],row)
        N.copy(row,Out)
""",
                disjoint_external=True,
            )
        )


def test_local_index_capture_does_not_read_an_overwritten_tile_version():
    from ascendflow import compile
    from ascendflow.diagnostics import CompileError

    with pytest.raises(CompileError, match="overwrit|value|version|writer"):
        compile(
            N.from_source(
                """
def stale(Meta:N.Tensor((2,8),N.i32),X:N.Tensor((8,64),N.f32),
          Out:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        cache=N.alloc_ub((8,),N.i32)
        row=N.alloc_ub((64,),N.f32)
        first=N.copy(Meta[0,:],cache)
        N.copy(Meta[1,:],cache)
        index=N.load_index(first[0:1])
        N.copy(X[index,:],row)
        N.copy(row,Out)
""",
                disjoint_external=True,
            )
        )


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
            for i in N.Pipelined(n):
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
            for i in N.Pipelined(n):
                slot=N.load_index(Meta[i:i+1],minimum=0,maximum=1)
                N.copy(X[i,:],a[slot,:])
                N.copy(a[slot,:],Out[i,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    for count in (0, 1, 2, 3, 7):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_scalar_store_exit_still_publishes_before_vector_reads(metadata_values):
    from ascendflow.analysis.access_frontier import QueueEndpoint
    from ascendflow.target.ascend910b import Engine

    source = N.from_source(
        """
def store_exit(Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),
               Meta:N.Tensor((n,),N.i64),n:N.int64):
    with N.Kernel(1,kind="vector"):
        scalar=N.alloc_ub((8,),N.f32)
        row=N.alloc_ub((64,),N.f32)
        N.fill(scalar,1.0)
        value=N.load_scalar(scalar[0:1])
        for i in N.Pipelined(n):
            N.fill(row,0.0)
            for child in N.serial(i%2+1):
                condition=N.load_index(Meta[i:i+1],minimum=0,maximum=1)
                if condition==0:
                    N.store_scalar(value,out=row[0:1])
            N.mul(row,2.0,out=row)
            N.copy(row,Out[i,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    publications = [
        p
        for n in plan.selection.notifications
        if n.source_engine is Engine.S and n.destination_engine is Engine.V
        for p in n.sources
    ]
    assert publications
    exits = [
        p.occurrence.endpoint
        for p in publications
        if isinstance(p.occurrence.endpoint, QueueEndpoint)
    ]
    assert exits and all(not point.retirements for point in exits)
    for count in (0, 1, 2, 5):
        check_compiled_events(plan, {"n": count}, invocations=2)
