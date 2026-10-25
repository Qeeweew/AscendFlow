"""Tensor calls on Torch's current NPU stream, using the compiled launch ABI.

Torch remains optional: it is imported only when constructing this runtime.
Outputs stay explicit because write access does not imply a complete write.
"""

import ctypes as ct
import hashlib
import inspect
import json
from functools import cached_property
from itertools import pairwise
from os import PathLike, fsdecode
from pathlib import Path

from ..backend.cann import CANN
from ..device import query_device_resources
from ..diagnostics import require
from .kernel import KernelLaunch

_DTYPES = {"f16": "float16", "bf16": "bfloat16", "f32": "float32", "i32": "int32", "i64": "int64"}
_LIBRARIES = {}


def _library(value):
    if not isinstance(value, (str, bytes, PathLike)):
        return value
    path = Path(fsdecode(value)).resolve()
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    if path in _LIBRARIES:
        previous, library = _LIBRARIES[path]
        require(
            previous == identity, "loaded kernel library changed; build into a new output directory"
        )
        return library
    library = ct.CDLL(str(path))
    _LIBRARIES[path] = (identity, library)
    return library


def _extent(expression, parameters):
    if type(expression) is int:
        return expression
    if isinstance(expression, str):
        require(
            expression in parameters, f"missing runtime parameter {expression}; pass it by name"
        )
        return parameters[expression]
    require(
        isinstance(expression, dict) and set(expression) == {"mul"},
        "invalid compiled tensor extent",
    )
    factors = expression["mul"]
    require(
        isinstance(factors, list) and len(factors) == 2, "invalid compiled tensor extent product"
    )
    return _extent(factors[0], parameters) * _extent(factors[1], parameters)


class TensorKernel:
    """One compiled signature; arenas are isolated by stream and scalar binding."""

    def __init__(self, runtime, library, name):
        self.runtime = runtime
        self.library = _library(library)
        require(
            hasattr(self.library, "kernel_metadata_" + name),
            "compiled tensor metadata is missing; rebuild the kernel",
        )
        query = getattr(self.library, "kernel_metadata_" + name)
        query.argtypes, query.restype = [], ct.c_char_p
        self.metadata = json.loads(query())
        require(
            self.metadata["version"] == 1 and self.metadata["name"] == name,
            "compiled kernel ABI version/name differs",
        )
        require(self.metadata["target"] == "dav-2201", "tensor runtime requires dav-2201")
        self.name = name
        self.tensors = tuple(item["name"] for item in self.metadata["tensors"])
        self.parameters = tuple(item["name"] for item in self.metadata["parameters"])
        self.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in self.tensors]
            + [
                inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None)
                for n in self.parameters
            ]
        )
        self._bindings = {}

    @property
    def workspace_nbytes(self):
        return sum(binding.workspace_nbytes for binding in self._bindings.values())

    def bind(self, *args, **kwargs):
        """Validate once; bound tensors must keep their shape, dtype and storage."""
        require(not self.runtime._closed, "kernel runtime is closed")
        values = self.__signature__.bind(*args, **kwargs).arguments
        parameters = {n: values[n] for n in self.parameters if values.get(n) is not None}
        require(
            all(type(n) is int and 0 <= n < 1 << 63 for n in parameters.values()),
            "runtime parameters require nonnegative int64 integers",
        )
        tensors = tuple(values[n] for n in self.tensors)
        device = None
        intervals = []
        pointers = []
        for tensor, spec in zip(tensors, self.metadata["tensors"]):
            name = spec["name"]
            require(
                isinstance(tensor, self.runtime.torch.Tensor), f"{name}: expected a Torch tensor"
            )
            require(
                tensor.device.type == "npu" and tensor.device.index == 0,
                f"{name}: expected the explicitly mapped logical NPU 0",
            )
            device = tensor.device if device is None else device
            require(tensor.device == device, "kernel tensors must share one device")
            require(tensor.is_contiguous(), f"{name}: expected contiguous tensor storage")
            storage_format = self.runtime.tensor_format(tensor)
            if spec["layout"] == "NZ":
                require(storage_format == 29, f"{name}: expected FRACTAL_NZ storage")
            else:
                require(spec["layout"] == "linear", f"{name}: unsupported compiled tensor layout")
                require(storage_format in {0, 2}, f"{name}: expected ND storage")
            require(
                tensor.dtype == getattr(self.runtime.torch, _DTYPES[spec["dtype"]]),
                f"{name}: tensor dtype differs from compiled ABI",
            )
            require(
                len(tensor.shape) == len(spec["shape"]),
                f"{name}: tensor rank differs from compiled ABI",
            )
            for actual, dimension in zip(tensor.shape, spec["shape"]):
                if isinstance(dimension, str):
                    if dimension in parameters:
                        require(
                            parameters[dimension] == actual,
                            f"{name}: conflicting runtime dimension {dimension}",
                        )
                    else:
                        parameters[dimension] = int(actual)
            pointer = tensor.data_ptr()
            pointers.append(pointer)
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes:
                intervals.append((pointer, pointer + nbytes, name))
        for spec, tensor in zip(self.metadata["tensors"], tensors):
            expected = tuple(_extent(n, parameters) for n in spec["shape"])
            require(
                tuple(tensor.shape) == expected,
                f"{spec['name']}: tensor shape differs from compiled ABI",
            )
        for spec in self.metadata["parameters"]:
            name = spec["name"]
            require(name in parameters, f"missing runtime parameter {name}; pass it by name")
            value = parameters[name]
            require(
                spec["minimum"] <= value < 1 << 63 and value % spec["multiple"] == 0,
                f"{name}: runtime input contract differs",
            )
        if self.metadata["disjoint_external"]:
            ordered = sorted(intervals)
            require(
                all(a[1] <= b[0] for a, b in pairwise(ordered)),
                "external tensors violate the compiled disjoint contract",
            )
        sizes = tuple(parameters[n] for n in self.parameters)
        recorded_streams = set()

        def invoke():
            require(not self.runtime._closed, "kernel runtime is closed")
            context = self.runtime._context(device)
            key = (context.stream.value, sizes)
            if key not in self._bindings:
                self._bindings[key] = KernelLaunch(
                    context, self.library, self.name, parameters=sizes
                )
            # Includes inputs created on another stream: allocation lifetime
            # follows the actual consumer stream. Readiness is caller-owned.
            if context.stream.value not in recorded_streams:
                for tensor in tensors:
                    tensor.record_stream(context.torch_stream)
                recorded_streams.add(context.stream.value)
            self._bindings[key](*pointers)

        return invoke

    def __call__(self, *args, **kwargs):
        return self.bind(*args, **kwargs)()


class _StreamContext:
    def __init__(self, owner, stream):
        self.owner, self.torch_stream = owner, stream
        self.stream = ct.c_void_p(stream.npu_stream)
        self.allocations = []

    @property
    def _closed(self):
        return self.owner._closed

    @property
    def resources(self):
        return self.owner.resources

    def allocate_uninitialized(self, nbytes):
        torch = self.owner.torch
        with torch.npu.stream(self.torch_stream):
            storage = torch.empty(max(nbytes, 1) + 511, dtype=torch.uint8, device="npu:0")
        storage.record_stream(self.torch_stream)
        self.allocations.append(storage)
        return ct.c_void_p((storage.data_ptr() + 511) // 512 * 512)


class TorchRuntime:
    """Own compiled tensor calls and their per-stream arenas until close()."""

    def __init__(self, torch=None):
        if torch is None:
            import torch
        from torch_npu import get_npu_format
        from torch_npu._C import _npu_getCurrentRawStreamNoWait

        self.torch = torch
        self.tensor_format = get_npu_format
        self._stream_handle = _npu_getCurrentRawStreamNoWait
        self._closed = False
        self._streams = {}
        self.lib = ct.CDLL(str(CANN / "lib64/libascendcl.so"), mode=ct.RTLD_GLOBAL)

    def call(self, name, types, *arguments):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, ct.c_int
        status = fn(*arguments)
        if status:
            raise RuntimeError(
                f"{name} failed with ACL error {status}; device retries are disabled"
            )

    @cached_property
    def resources(self):
        require(not self._closed, "kernel runtime is closed")
        require(
            self.torch.npu.current_device() == 0,
            "runtime requires the explicitly mapped logical NPU 0",
        )
        return query_device_resources(self.call)

    def _context(self, device):
        key = self._stream_handle(0)
        if key not in self._streams:
            stream = self.torch.npu.current_stream(device)
            self._streams[key] = _StreamContext(self, stream)
        return self._streams[key]

    def kernel(self, library, name):
        require(not self._closed, "kernel runtime is closed")
        return TensorKernel(self, library, name)

    def compile(self, definition, output, **constexpr):
        """Bind actual device resources before the ordinary bounded build."""
        from ..build import build_module

        require(not self._closed, "kernel runtime is closed")
        resources = self.resources
        result = build_module(
            lambda: definition.specialize(**constexpr).bind_device(resources),
            output,
            shared=True,
            timeout_s=300,
        )
        name = result["kernel"]
        return self.kernel(Path(output).resolve() / (name + ".so"), name)

    def close(self):
        if not self._closed:
            for context in self._streams.values():
                self.call("aclrtSynchronizeStream", [ct.c_void_p], context.stream)
            for context in self._streams.values():
                context.allocations.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
