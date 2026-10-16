"""Direct ACLNN references called by runners with explicit tensors and shapes."""

import ctypes as ct

from xdsl_npu.backend.cann import CANN
from xdsl_npu.diagnostics import require


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
