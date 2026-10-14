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
        self.scalars = []

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

    def tensor(self, ptr, shape, *, dtype=0, strides=None, storage_shape=None):
        dims = (ct.c_int64 * len(shape))(*shape)
        if strides is None:
            strides = []
            stride = 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= size
        layout = (ct.c_int64 * len(shape))(*strides)
        storage_shape = shape if storage_shape is None else storage_shape
        storage_dims = (ct.c_int64 * len(storage_shape))(*storage_shape)
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
        tensor = create(
            dims, len(shape), dtype, layout, 0, 2, storage_dims, len(storage_shape), ptr
        )
        require(tensor is not None, "aclCreateTensor failed")
        self.tensors.append(tensor)
        return tensor

    def __call__(self):
        status = self.launch(
            self.workspace, self.workspace_size, self.executor, self.runtime.stream
        )
        require(status == 0, f"{self.operator} failed: {status}; no retry")

    def scalar(self, value):
        create = self.meta.aclCreateScalar
        create.argtypes, create.restype = [ct.c_void_p, ct.c_int], ct.c_void_p
        data = ct.c_float(value)
        scalar = create(ct.byref(data), 0)
        require(scalar is not None, "aclCreateScalar failed")
        self.scalars.append(scalar)
        return scalar

    def close(self):
        destroy = self.meta.aclDestroyAclOpExecutor
        destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int32
        require(destroy(self.executor) == 0, "ACLNN executor destroy failed")
        destroy_tensor = self.meta.aclDestroyTensor
        destroy_tensor.argtypes, destroy_tensor.restype = [ct.c_void_p], ct.c_int32
        for tensor in self.tensors:
            require(destroy_tensor(tensor) == 0, "ACL tensor destroy failed")
        destroy_scalar = self.meta.aclDestroyScalar
        destroy_scalar.argtypes, destroy_scalar.restype = [ct.c_void_p], ct.c_int32
        for scalar in self.scalars:
            require(destroy_scalar(scalar) == 0, "ACL scalar destroy failed")


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


class GemmReference(OperatorReference):
    interface = "aclnnMatmul; f16 A and Bt.T view, f32 C; cubeMathType=0"

    def __init__(self, runtime, a, bt, c, m, n, reduction):
        super().__init__(runtime)
        self.prepare(
            "aclnnMatmul",
            [
                self.tensor(a, (m, reduction), dtype=1),
                self.tensor(
                    bt,
                    (reduction, n),
                    dtype=1,
                    strides=(1, reduction),
                    storage_shape=(n, reduction),
                ),
                self.tensor(c, (m, n)),
                0,
            ],
            [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int8],
        )


class GemmBiasSiluReference:
    interface = "aclnnMatmul f16->f32; aclnnAdd broadcast f32 bias; aclnnSilu f32, three calls"

    def __init__(self, runtime, a, bt, bias, y, m, n, reduction):
        c = runtime.allocate_uninitialized(m * n * 4)
        biased = runtime.allocate_uninitialized(m * n * 4)
        self.calls = [GemmReference(runtime, a, bt, c, m, n, reduction)]
        add = OperatorReference(runtime)
        add.prepare(
            "aclnnAdd",
            [
                add.tensor(c, (m, n)),
                add.tensor(bias, (n,)),
                add.scalar(1),
                add.tensor(biased, (m, n)),
            ],
            [ct.c_void_p] * 4,
        )
        self.calls.append(add)
        silu = OperatorReference(runtime)
        silu.prepare(
            "aclnnSilu", [silu.tensor(biased, (m, n)), silu.tensor(y, (m, n))], [ct.c_void_p] * 2
        )
        self.calls.append(silu)
        self.workspace_size = ct.c_uint64(sum(op.workspace_size.value for op in self.calls))

    def __call__(self):
        for op in self.calls:
            op()

    def close(self):
        for op in reversed(self.calls):
            op.close()


class FlashAttentionReference(OperatorReference):
    """Installed CANN dense BNSD kernel, with its own tiling/rounding choices."""

    interface = "aclnnFlashAttentionScore BNSD, keepProb1, innerPrecise0, sparseMode0"

    def __init__(self, runtime, q, k, v, output, shape, dtype, scale):
        super().__init__(runtime)
        batch, heads, kv_heads, sq, sk, dim = shape
        code = {"f16": 1, "bf16": 27}[dtype]
        stat_shape = (batch, heads, sq, 8)
        maximum = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        denominator = runtime.allocate_uninitialized(batch * heads * sq * 8 * 4)
        self.auxiliary_bytes = 2 * batch * heads * sq * 8 * 4
        self.prepare(
            "aclnnFlashAttentionScore",
            [
                self.tensor(q, (batch, heads, sq, dim), dtype=code),
                self.tensor(k, (batch, kv_heads, sk, dim), dtype=code),
                self.tensor(v, (batch, kv_heads, sk, dim), dtype=code),
                None,
                None,
                None,
                None,
                None,
                scale,
                1.0,
                2**31 - 1,
                2**31 - 1,
                heads,
                b"BNSD",
                0,
                0,
                self.tensor(maximum, stat_shape),
                self.tensor(denominator, stat_shape),
                None,
                self.tensor(output, (batch, heads, sq, dim), dtype=code),
            ],
            [ct.c_void_p] * 8
            + [ct.c_double] * 2
            + [ct.c_int64] * 3
            + [ct.c_char_p]
            + [ct.c_int64] * 2
            + [ct.c_void_p] * 4,
        )


def run_reference(plan, output: Path, *, physical=1, repeats=30, warmup=5, data_case="standard"):
    from .ir import MixedOp

    mixed = isinstance(plan.body.block.first_op, MixedOp)
    if mixed:
        from types import SimpleNamespace

        p = SimpleNamespace(name=plan.body.block.first_op.sym_name.data)
    else:
        p = extract(plan)
    require(
        p.name in {"rms_norm", "softmax", "gemm_nt", "gemm_bias_silu"},
        "reference adapter implements RMSNorm, Softmax, GEMM and GEMM+bias+SiLU",
    )
    require(repeats > 0 and warmup > 0, "correctness and timing require positive launch counts")
    health = health_snapshot(physical, output)
    if mixed:
        from .oracles import mixed_inputs

        require(data_case == "standard", "mixed baseline requires standard inputs")
        contents, expected, tolerance, shape = mixed_inputs(plan)
    else:
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
        elif p.name == "softmax":
            epsilon = None
            baseline = SoftmaxReference(runtime, *pointers, *shape)
        elif p.name == "gemm_bias_silu":
            epsilon = None
            # The generated workspace is not part of the ACLNN algorithm.
            # Keep its bytes/hash in the common argument manifest for comparison.
            baseline = GemmBiasSiluReference(
                runtime, pointers[0], pointers[1], pointers[3], pointers[4], *shape
            )
        else:
            epsilon = None
            baseline = GemmReference(runtime, *pointers, *shape)
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
            "dtype": "f16 operands/f32 output"
            if p.name in {"gemm_nt", "gemm_bias_silu"}
            else "f32",
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
