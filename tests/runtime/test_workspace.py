"""Kernel-owned GM: real layout/ABI execution and ordinary memory contracts."""

import ctypes as ct
import re
import subprocess

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.device import ACL
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import BufferOp
from ascendflow.dialects.source import AllocationOp
from examples.flash_attention.simple import simple_attention
from tests.models.structured_events import check_structured_events
from tests.support.asc_trace import TRACE_SHIM, compile_cpp

PROGRAM = """
def owned(Out:N.Tensor((n,64),N.f32,access="write"),n:N.i64):
    with N.Kernel():
        head=N.alloc_workspace((65,),name="head")
        middle=N.alloc_workspace((n,64),name="middle")
        ring=N.alloc_workspace((64,),slots=2,name="ring")
        x=N.alloc_ub((64,))
        for t in N.serial(n):
            N.fill(x,2.0)
            N.copy(x,head[:64])
            N.copy(head[:64],x)
            N.copy(x,middle[t,:])
            N.copy(middle[t,:],x)
            N.copy(x,ring.at(t))
            N.copy(ring.at(t),x)
            N.copy(x,Out[t,:])
"""


@pytest.fixture(scope="module")
def artifact():
    return compile(N.from_source(PROGRAM))


def test_workspace_retains_uninitialized_roots_slots_and_repeated_launch_hazards(artifact):
    source = [op for op in parse(artifact.source_ir).walk() if isinstance(op, AllocationOp)]
    workspaces = [
        op for op in source if op.ref.type.memory.data == "GM" and op.access.data == "local"
    ]
    assert [op.sym_name.data for op in workspaces] == ["head", "middle", "ring"]
    buffers = [op for op in artifact.plan.walk() if isinstance(op, BufferOp)]
    assert [op.sym_name.data for op in buffers if op.external.data] == ["Out"]
    gm = [op for op in buffers if not op.external.data and op.ref.type.bank.data == "GM"]
    assert len(gm) == 3 and all(op.initialized.data == 0 for op in gm)
    before = collect_native_memory(parse(artifact.before_ir))
    after = collect_native_memory(artifact.plan)
    for count in (0, 1, 2, 5):
        check_structured_events(after, {"n": count}, logical_reference=before, invocations=2)
    assert artifact.report["workspace"]["static_bytes"] is None
    assert "arg_head" not in artifact.asc and "arg_middle" not in artifact.asc


def test_workspace_read_before_write_remains_an_initialization_error():
    definition = N.from_source("""
def invalid():
    with N.Kernel():
        w=N.alloc_workspace((64,))
        x=N.alloc_ub((64,))
        N.copy(w,x)
""")
    with pytest.raises(CompileError, match="without a proved initialized writer"):
        compile(definition)


@pytest.mark.parametrize("shape", ["(0,)", "(-1,)", "(True,)", "(1.5,)", "()"])
def test_invalid_workspace_shapes_are_rejected(shape):
    with pytest.raises(CompileError, match="positive static"):
        N.from_source(
            f"def invalid():\n    with N.Kernel():\n        w=N.alloc_workspace({shape})"
        ).to_ir()


def test_runtime_workspace_allocation_cannot_be_declared_in_a_loop():
    definition = N.from_source("""
def invalid():
    with N.Kernel():
        for i in N.serial(2):
            w=N.alloc_workspace((64,))
""")
    with pytest.raises(CompileError, match="kernel entry"):
        compile(definition)


def test_external_tensor_cannot_request_internal_ownership():
    definition = N.from_source("""
def invalid(X:N.Tensor((64,),N.f32,access="local")):
    with N.Kernel():
        pass
""")
    with pytest.raises(CompileError, match="external Tensor access"):
        definition.to_ir()


def host_library(artifact, tmp_path):
    """Execute the actual sizing and launcher code; only dispatch is a stub."""
    host = artifact.asc[artifact.asc.index('extern "C" int workspace_size_') :]
    host = re.sub(
        r"(?m)^  \w+<<<[^\n]+;",
        "  seen_workspace = npu_workspace; seen_size = npu_size_n; ++dispatches;",
        host,
    )
    source = tmp_path / "host.cpp"
    source.write_text(
        "#include <cstdint>\n#include <climits>\n"
        "void* seen_workspace; int64_t seen_size; int dispatches=0;\n"
        + host
        + '\nextern "C" void* workspace_seen() { return seen_workspace; }\n'
    )
    library = tmp_path / "host.so"
    subprocess.run(
        ["c++", "-shared", "-fPIC", "-std=c++17", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return ct.CDLL(str(library))


def test_actual_query_packs_odd_sizes_and_device_uses_the_same_aligned_offsets(artifact, tmp_path):
    host = host_library(artifact, tmp_path)
    query = host.workspace_size_owned
    query.argtypes, query.restype = [ct.POINTER(ct.c_uint64), ct.c_int64], ct.c_int
    for n, total in ((0, 1024), (1, 1536), (2, 1536), (3, 2048), (5, 2560)):
        size = ct.c_uint64()
        assert query(ct.byref(size), n) == 0 and size.value == total
    for n in (-1, (1 << 63) - 1):
        size = ct.c_uint64(123)
        assert query(ct.byref(size), n) != 0 and size.value == 0
    # Run the emitted device body through the existing CPU address trace. This
    # checks the real pointer bindings and native addresses, without NPU work.
    asc = compile(N.from_source(PROGRAM), host_launcher=False).asc
    asc = "\n".join(line for line in asc.splitlines() if not line.startswith("#include"))
    binary = compile_cpp(
        tmp_path,
        TRACE_SHIM
        + "#include <climits>\n"
        + asc
        + "\nint main(){owned((uint8_t*)1000000,(uint8_t*)2000000,3);}\n",
    )
    lines = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=5
    ).stdout
    addresses = {int(word) for line in lines.splitlines() for word in line.split()[2:]}
    assert {2_000_000, 2_000_512, 2_001_536, 2_001_792} <= addresses
    assert "PIPE_ALL" not in artifact.asc


def test_multi_dimension_query_checks_overflow_and_accepts_zero_extents(tmp_path):
    artifact = compile(
        N.from_source("""
def sizing(n:N.i64,m:N.i64):
    with N.Kernel():
        w=N.alloc_workspace((n,m,64))
        tail=N.alloc_workspace((17,),N.bf16)
""")
    )
    host = host_library(artifact, tmp_path)
    query = host.workspace_size_sizing
    query.argtypes = [ct.POINTER(ct.c_uint64), ct.c_int64, ct.c_int64]
    query.restype = ct.c_int
    size = ct.c_uint64()
    assert query(ct.byref(size), (1 << 63) - 1, 0) == 0 and size.value == 512
    assert query(ct.byref(size), 0, (1 << 63) - 1) == 0 and size.value == 512
    assert query(ct.byref(size), 2, 3) == 0 and size.value == 2048
    assert query(ct.byref(size), 1 << 32, 1 << 32) != 0 and size.value == 0
    assert query(ct.byref(size), ((1 << 63) - 1) // 256, 1) != 0 and size.value == 0


def test_runtime_allocates_once_reuses_on_one_stream_and_isolates_bindings(artifact, tmp_path):
    library = host_library(artifact, tmp_path)
    runtime = object.__new__(ACL)
    runtime._closed, runtime._bound_kernels = False, False
    runtime.stream = ct.c_void_p(99)
    allocations = []

    def allocate(size):
        allocations.append(size)
        return ct.c_void_p(4096 * len(allocations))

    runtime.allocate_uninitialized = allocate
    kernel = runtime.kernel(library, "owned", parameters=(3,))
    other = runtime.kernel(library, "owned", parameters=(5,))
    assert allocations == [2048, 2560]
    assert kernel.workspace.value != other.workspace.value
    library.workspace_seen.restype = ct.c_void_p
    for _ in range(3):
        kernel(ct.c_void_p(123))
        assert library.workspace_seen() == kernel.workspace.value
    assert allocations == [2048, 2560]
    with pytest.raises(CompileError, match="expects 1"):
        kernel()
    runtime._closed = True
    with pytest.raises(CompileError, match="closed"):
        kernel(ct.c_void_p(123))


def test_teaching_attention_exposes_only_io_and_shared_internal_gm():
    artifact = compile(simple_attention)
    buffers = [op for op in artifact.plan.walk() if isinstance(op, BufferOp)]
    assert [op.sym_name.data for op in buffers if op.external.data] == ["Q", "K", "V", "O"]
    assert artifact.report["workspace"]["static_bytes"] == 655360
    assert artifact.asc.count("if ASC_IS_AIV") == 1
    assert artifact.asc.count("if ASC_IS_AIC") == 1
    assert "arg_Score" not in artifact.asc


def test_bound_kernel_retires_stream_before_releasing_workspace():
    runtime = object.__new__(ACL)
    runtime._closed, runtime._bound_kernels = False, True
    runtime.events, runtime.allocations, runtime.stream = [], [ct.c_void_p(512)], ct.c_void_p(99)
    calls = []
    runtime.call = lambda name, *args: calls.append(name)
    runtime.close()
    assert calls.index("aclrtSynchronizeStream") < calls.index("aclrtFree")
    runtime.close()
    assert calls.count("aclrtFree") == 1
