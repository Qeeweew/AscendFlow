"""Small direct ACLNN baseline adapter; no Torch or foreign workspace needed."""

import ctypes as ct
import hashlib
import json
import struct
from pathlib import Path

from .device import ACL, health_snapshot
from .model import extract, require
from .oracles import operator_inputs
from .toolchain import CANN


class OperatorReference:
    def __init__(self, runtime):
        self.runtime = runtime
        self.meta = ct.CDLL(str(CANN / "lib64/libnnopbase.so"), mode=ct.RTLD_GLOBAL)
        self.ops = ct.CDLL(str(CANN / "lib64/libopapi.so"), mode=ct.RTLD_GLOBAL)
        self.tensors = []

    def prepare(self, operator, arguments, types):
        self.operator = operator
        self.workspace_size = ct.c_uint64()
        self.executor = ct.c_void_p()
        get = getattr(self.ops, operator + "GetWorkspaceSize")
        get.argtypes = types + [
            ct.POINTER(ct.c_uint64),
            ct.POINTER(ct.c_void_p),
        ]
        get.restype = ct.c_int32
        status = get(*arguments, ct.byref(self.workspace_size), ct.byref(self.executor))
        require(status == 0, f"{operator}GetWorkspaceSize failed: {status}")
        repeat = self.meta.aclSetAclOpExecutorRepeatable
        repeat.argtypes, repeat.restype = [ct.c_void_p], ct.c_int32
        require(repeat(self.executor) == 0, "ACLNN executor is not repeatable")
        self.workspace = self.runtime.allocate_uninitialized(self.workspace_size.value)
        self.launch = getattr(self.ops, operator)
        self.launch.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
        self.launch.restype = ct.c_int32

    def tensor(self, ptr, shape):
        dims = (ct.c_int64 * len(shape))(*shape)
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        layout = (ct.c_int64 * len(shape))(*strides)
        create = self.meta.aclCreateTensor
        create.argtypes = [
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_int64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_void_p,
        ]
        create.restype = ct.c_void_p
        tensor = create(dims, len(shape), 0, layout, 0, 2, dims, len(shape), ptr)
        require(tensor is not None, "aclCreateTensor failed")
        self.tensors.append(tensor)
        return tensor

    def __call__(self):
        status = self.launch(
            self.workspace, self.workspace_size, self.executor, self.runtime.stream
        )
        require(status == 0, f"{self.operator} failed: {status}; no retry")

    def close(self):
        destroy = self.meta.aclDestroyAclOpExecutor
        destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int32
        require(destroy(self.executor) == 0, "ACLNN executor destroy failed")
        destroy_tensor = self.meta.aclDestroyTensor
        destroy_tensor.argtypes, destroy_tensor.restype = [ct.c_void_p], ct.c_int32
        for tensor in self.tensors:
            require(destroy_tensor(tensor) == 0, "ACL tensor destroy failed")


class RmsNormReference(OperatorReference):
    interface = "aclnnRmsNorm; computes Y and auxiliary Rstd"

    def __init__(self, runtime, x, gamma, y, rstd, rows, columns, epsilon):
        super().__init__(runtime)
        self.prepare(
            "aclnnRmsNorm",
            [
                self.tensor(x, (rows, columns)),
                self.tensor(gamma, (columns,)),
                epsilon,
                self.tensor(y, (rows, columns)),
                self.tensor(rstd, (rows, 1)),
            ],
            [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_void_p, ct.c_void_p],
        )


class SoftmaxReference(OperatorReference):
    interface = "aclnnSoftmax; last dimension, f32 Y"

    def __init__(self, runtime, x, y, rows, columns):
        super().__init__(runtime)
        self.prepare(
            "aclnnSoftmax",
            [self.tensor(x, (rows, columns)), -1, self.tensor(y, (rows, columns))],
            [ct.c_void_p, ct.c_int64, ct.c_void_p],
        )


def run_reference(plan, output: Path, *, physical=1, repeats=30, warmup=5, data_case="standard"):
    p = extract(plan)
    require(p.name in {"rms_norm", "softmax"}, "reference adapter implements RMSNorm and Softmax")
    require(repeats > 0 and warmup > 0, "correctness and timing require positive launch counts")
    health = health_snapshot(physical, output)
    contents, expected, tolerance, shape = operator_inputs(p, data_case=data_case)
    runtime = ACL()
    baseline = None
    try:
        pointers = [runtime.allocate(data) for data in contents]
        y = pointers[-1]
        if p.name == "rms_norm":
            rstd = runtime.allocate_uninitialized(shape[0] * 4)
            epsilon = next(t.scalar for t in p.tasks if t.uid == "epsilon")
            baseline = RmsNormReference(runtime, *pointers, rstd, *shape, epsilon)
        else:
            epsilon = None
            baseline = SoftmaxReference(runtime, *pointers, *shape)
        for _ in range(warmup):
            baseline()
        runtime.synchronize()
        raw = runtime.read(y, len(expected) * 4)
        actual = struct.unpack(f"<{len(expected)}f", raw)
        error = max((abs(a - b) for a, b in zip(actual, expected)), default=0)
        require(
            all(
                abs(a - b) <= tolerance[1] + tolerance[0] * abs(b) for a, b in zip(actual, expected)
            ),
            f"ACLNN reference mismatch: max_abs_error={error}",
        )
        (output / "reference-output.bin").write_bytes(raw)
        first, last = runtime.event(), runtime.event()
        runtime.record(first)
        for _ in range(repeats):
            baseline()
        runtime.record(last)
        runtime.synchronize()
        ms = ct.c_float()
        runtime.call(
            "aclrtEventElapsedTime",
            [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
            ct.byref(ms),
            first,
            last,
        )
        record = {
            "health": health,
            "shape": shape,
            "data_case": data_case,
            "argument_sha256": [hashlib.sha256(data).hexdigest() for data in contents],
            "dtype": "f32",
            "epsilon": epsilon,
            "warmup": warmup,
            "repeats": repeats,
            "max_abs_error": error,
            "stream_interval_us_per_launch": ms.value * 1000 / repeats,
            "timing_scope": "ACL stream interval including submission gaps",
            "interface": baseline.interface,
            "library": str(CANN / "lib64/libopapi.so"),
            "library_sha256": hashlib.sha256((CANN / "lib64/libopapi.so").read_bytes()).hexdigest(),
            "workspace_bytes": baseline.workspace_size.value,
        }
        (output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        if baseline is not None:
            baseline.close()
        runtime.close()
