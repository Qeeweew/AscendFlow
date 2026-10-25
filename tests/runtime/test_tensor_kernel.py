"""Execute generated host metadata and argument binding without an NPU."""

import ctypes as ct
import json
import re
import subprocess
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace

import pytest

from ascendflow import DeviceResources, TorchRuntime, compile
from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.runtime.torch import TensorKernel

SOURCE = """
def transfer(X:N.Tensor((n,64),N.f32),Out:N.Tensor((n,64),N.f32,access="write"),n:N.i64(minimum=1)):
    with N.Kernel():
        work=N.alloc_workspace((n,64))
        row=N.alloc_ub((64,))
        for i in N.serial(n):
            value=N.copy(X[i,:],row)
            N.copy(value,work[i,:])
            value=N.copy(work[i,:],row)
            N.copy(value,Out[i,:])
"""


@pytest.fixture(scope="module")
def artifact():
    return compile(N.from_source(SOURCE, disjoint_external=True))


@pytest.fixture
def library(artifact, tmp_path):
    host = artifact.asc[artifact.asc.index('extern "C" int workspace_size_') :]
    host = re.sub(
        r"(?m)^  \w+<<<[^\n]+;",
        "  last_stream=uintptr_t(stream); last_x=uintptr_t(arg_X); last_out=uintptr_t(arg_Out); last_size=npu_size_n; ++calls;",
        host,
    )
    source = (
        "#include <cstdint>\n#include <climits>\nuintptr_t last_stream,last_x,last_out; int64_t last_size,calls;\n"
        + host
    )
    source += '\nextern "C" int64_t seen(int which) { return which==0 ? last_stream : which==1 ? last_x : which==2 ? last_out : which==3 ? last_size : calls; }\n'
    path, output = tmp_path / "host.cpp", tmp_path / "host.so"
    path.write_text(source)
    subprocess.run(
        ["c++", "-shared", "-fPIC", "-std=c++17", str(path), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    result = ct.CDLL(str(output))
    result.seen.argtypes, result.seen.restype = [ct.c_int], ct.c_int64
    return result


class Tensor:
    def __init__(self, shape=(3, 64), pointer=8192, dtype="f32", device="npu", contiguous=True):
        self.shape, self.pointer, self.dtype = shape, pointer, dtype
        self.device = SimpleNamespace(type=device, index=0)
        self.contiguous = contiguous
        self.streams = []

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return 1 if self.dtype == "u8" else 4

    def data_ptr(self):
        return self.pointer

    def record_stream(self, stream):
        self.streams.append(stream.npu_stream)


def runtime():
    result = object.__new__(TorchRuntime)
    result._closed, result._streams = False, {}
    result.resources = DeviceResources(20, 40)
    allocations, calls = [], []
    npu = SimpleNamespace(selected=SimpleNamespace(npu_stream=99))
    npu.current_stream = lambda device: npu.selected
    result._stream_handle = lambda device: npu.selected.npu_stream

    @contextmanager
    def stream(selected):
        previous, npu.selected = npu.selected, selected
        try:
            yield
        finally:
            npu.selected = previous

    npu.stream = stream

    def empty(size, *, dtype, device):
        storage = Tensor((size,), pointer=65543 + len(allocations) * 8192, dtype=dtype)
        allocations.append((storage, npu.selected.npu_stream))
        return storage

    result.torch = SimpleNamespace(Tensor=Tensor, float32="f32", uint8="u8", npu=npu, empty=empty)
    result.tensor_format = lambda tensor: getattr(tensor, "storage_format", 2)
    result.call = lambda name, types, *args: calls.append((name, args))
    return result, allocations, calls


def test_actual_library_metadata_matches_report_and_real_argument_order(artifact, library):
    owner, allocations, _ = runtime()
    kernel = owner.kernel(library, "transfer")
    assert kernel.metadata == artifact.report["abi"]
    assert kernel.tensors == ("X", "Out") and kernel.parameters == ("n",)
    x, out = Tensor(), Tensor(pointer=16384)
    kernel(Out=out, X=x)
    assert [library.seen(i) for i in range(5)] == [99, 8192, 16384, 3, 1]
    assert len(allocations) == 1 and kernel.workspace_nbytes == 1024
    assert allocations[0][0].streams == [99]
    assert next(iter(kernel._bindings.values())).workspace.value % 512 == 0


def test_packed_storage_cannot_masquerade_as_linear(library):
    owner, _, _ = runtime()
    x = Tensor()
    x.storage_format = 29
    with pytest.raises(CompileError, match="expected ND storage"):
        owner.kernel(library, "transfer").bind(x, Tensor(pointer=16384))


def test_nz_tensor_binding_requires_actual_packed_storage():
    definition = N.from_source("""
def packed(W:N.Tensor((16,16),N.bf16,layout="NZ")):
    with N.Kernel(kind="cube"):
        pass
""")
    metadata = compile(definition).report["abi"]

    def query():
        return json.dumps(metadata).encode()

    owner, _, _ = runtime()
    owner.torch.bfloat16 = "bf16"
    kernel = owner.kernel(SimpleNamespace(kernel_metadata_packed=query), "packed")
    w = Tensor(shape=(16, 16), dtype="bf16")
    with pytest.raises(CompileError, match="expected FRACTAL_NZ"):
        kernel.bind(w)
    w.storage_format = 29
    assert callable(kernel.bind(w))


def test_bound_call_reuses_arena_and_follows_current_stream(library):
    owner, allocations, calls = runtime()
    kernel = owner.kernel(library, "transfer")
    x, out = Tensor(), Tensor(pointer=16384)
    invoke = kernel.bind(x, out)
    invoke()
    invoke()
    owner.torch.npu.selected = SimpleNamespace(npu_stream=123)
    invoke()
    assert library.seen(0) == 123
    assert [stream for _, stream in allocations] == [99, 123]
    assert x.streams == out.streams == [99, 123]
    assert kernel.workspace_nbytes == 2048
    assert not calls
    owner.close()
    assert [name for name, _ in calls] == ["aclrtSynchronizeStream"] * 2
    assert all(not context.allocations for context in owner._streams.values())
    with pytest.raises(CompileError, match="closed"):
        invoke()


def test_hot_calls_only_resolve_stream_objects_for_new_streams(library):
    owner, _, _ = runtime()
    resolved = []

    def current_stream(device):
        resolved.append(owner.torch.npu.selected.npu_stream)
        return owner.torch.npu.selected

    owner.torch.npu.current_stream = current_stream
    invoke = owner.kernel(library, "transfer").bind(Tensor(), Tensor(pointer=16384))
    invoke()
    invoke()
    owner.torch.npu.selected = SimpleNamespace(npu_stream=123)
    invoke()
    invoke()
    owner.torch.npu.selected = SimpleNamespace(npu_stream=99)
    invoke()
    assert resolved == [99, 123]


@pytest.mark.parametrize(
    "x,out,parameters,message",
    [
        (Tensor(), Tensor(pointer=16384), {"n": 4}, "conflicting runtime dimension"),
        (Tensor(), Tensor(shape=(4, 64), pointer=16384), {}, "conflicting runtime dimension"),
        (Tensor(shape=(3, 32)), Tensor(pointer=16384), {}, "shape differs"),
        (Tensor(dtype="bf16"), Tensor(pointer=16384), {}, "dtype differs"),
        (Tensor(device="cpu"), Tensor(pointer=16384), {}, "logical NPU"),
        (Tensor(contiguous=False), Tensor(pointer=16384), {}, "contiguous"),
        (Tensor(), Tensor(pointer=8256), {}, "disjoint"),
        (Tensor(shape=(0, 64)), Tensor(shape=(0, 64), pointer=16384), {}, "input contract"),
        (Tensor(), Tensor(pointer=16384), {"n": True}, "int64"),
    ],
)
def test_invalid_tensor_contract_is_rejected_before_allocation(
    library, x, out, parameters, message
):
    owner, allocations, _ = runtime()
    with pytest.raises(CompileError, match=message):
        owner.kernel(library, "transfer")(x, out, **parameters)
    assert not allocations and library.seen(4) == 0


def test_different_dynamic_sizes_use_independent_bindings(library):
    owner, allocations, _ = runtime()
    kernel = owner.kernel(library, "transfer")
    kernel(Tensor(), Tensor(pointer=16384))
    kernel(Tensor(shape=(1, 64)), Tensor(shape=(1, 64), pointer=16384))
    assert len(allocations) == 2 and kernel.workspace_nbytes == 1536


def test_loaded_library_identity_cannot_silently_change(library):
    from pathlib import Path

    owner, _, _ = runtime()
    path = Path(library._name)
    owner.kernel(path, "transfer")
    with path.open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(CompileError, match="library changed"):
        owner.kernel(path, "transfer")


def test_missing_scalar_cannot_be_guessed_from_a_product_axis(library):
    query = library.kernel_metadata_transfer
    query.restype = ct.c_char_p
    data = json.loads(query())
    for tensor in data["tensors"]:
        tensor["shape"][0] = {"mul": ["n", 1]}

    def metadata():
        return json.dumps(data).encode()

    owner, allocations, _ = runtime()
    kernel = TensorKernel(owner, SimpleNamespace(kernel_metadata_transfer=metadata), "transfer")
    with pytest.raises(CompileError, match="missing runtime parameter n"):
        kernel(Tensor(), Tensor(pointer=16384))
    assert not allocations


def test_missing_compiled_metadata_requires_an_explicit_rebuild():
    owner, _, _ = runtime()
    with pytest.raises(CompileError, match="metadata is missing"):
        owner.kernel(SimpleNamespace(), "transfer")


def test_tensor_callable_rejects_unknown_abi_version(library):
    metadata = library.kernel_metadata_transfer
    metadata.restype = ct.c_char_p
    data = json.loads(metadata())
    data["version"] = 99

    def query():
        return json.dumps(data).encode()

    owner, _, _ = runtime()
    with pytest.raises(CompileError, match="ABI version"):
        TensorKernel(owner, SimpleNamespace(kernel_metadata_transfer=query), "transfer")


def test_compile_binds_resources_before_source_and_build(monkeypatch, tmp_path):
    from ascendflow import build

    definition = N.from_source("""
def resources(Out:N.Tensor((128,),N.f32,access="write")):
    groups=N.get_core_num("mixed")
    with N.Kernel(groups,kind="mixed"):
        x=N.alloc_ub((64,))
        with N.Vector() as sid:
            N.fill(x,1.0)
            N.copy(x,Out[sid*64:sid*64+64])
""")
    owner, _, _ = runtime()
    received = []

    def build_module(construct, output, **options):
        bound = construct()
        assert bound.device_resources == owner.resources
        assert bound.to_ir().body.block.first_op.groups.data == 20
        assert options == {"shared": True, "timeout_s": 300}
        return {"kernel": "resources"}

    monkeypatch.setattr(build, "build_module", build_module)
    owner.kernel = lambda library, name: received.append((library, name))
    owner.compile(definition, tmp_path)
    assert received == [(tmp_path / "resources.so", "resources")]
