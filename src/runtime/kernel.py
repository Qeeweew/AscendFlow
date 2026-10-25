"""A stream-owned callable with one reusable, automatically sized GM arena."""

import ctypes as ct
from os import PathLike, fspath

from ..diagnostics import require


class KernelLaunch:
    """Runtime sizes are bound once; concurrent streams need separate bindings."""

    def __init__(self, runtime, library, name, *, parameters=()):
        self.parameters = tuple(parameters)
        require(
            all(type(n) is int and 0 <= n < 1 << 63 for n in self.parameters),
            "kernel size parameters must be nonnegative int64 integers",
        )
        self.runtime = runtime
        self.stream = ct.c_void_p(runtime.stream.value)
        self.library = (
            ct.CDLL(fspath(library)) if isinstance(library, (str, bytes, PathLike)) else library
        )
        config = getattr(self.library, "kernel_config_" + name)
        config.argtypes, config.restype = [ct.POINTER(ct.c_int64)], ct.c_int
        values = (ct.c_int64 * 5)()
        require(config(values) == 0, "failed to read compiled kernel configuration")
        tensors, parameters_count, has_workspace, cube, vector = values
        require(
            len(self.parameters) == parameters_count,
            f"kernel expects {parameters_count} size parameters",
        )
        if cube or vector:
            resources = runtime.resources
            require(
                (cube, vector) == (resources.cube_cores, resources.vector_cores),
                "compiled device resources differ from the runtime device; bind and rebuild for this device",
            )
        self.tensors = tensors
        self.launch = getattr(self.library, "launch_" + name)
        self.query = getattr(self.library, "workspace_size_" + name) if has_workspace else None
        self.workspace, self.workspace_nbytes = None, 0
        if self.query is not None:
            self.query.argtypes = [ct.POINTER(ct.c_uint64)] + [ct.c_int64] * len(self.parameters)
            self.query.restype = ct.c_int
            size = ct.c_uint64()
            status = self.query(ct.byref(size), *self.parameters)
            require(status == 0, f"{name}: workspace byte extent exceeds int64 or is invalid")
            self.workspace_nbytes = size.value
            self.workspace = runtime.allocate_uninitialized(size.value)
        self.launch.argtypes = [ct.c_void_p] * (1 + tensors + int(self.query is not None)) + [
            ct.c_int64
        ] * len(self.parameters)
        self.launch.restype = ct.c_int

    def __call__(self, *tensors):
        require(not self.runtime._closed, "kernel runtime is closed")
        require(len(tensors) == self.tensors, f"kernel expects {self.tensors} tensor pointers")
        workspace = () if self.query is None else (self.workspace,)
        status = self.launch(self.stream, *tensors, *workspace, *self.parameters)
        if status:
            raise RuntimeError(
                f"kernel launch failed with status {status}; device retries are disabled"
            )
