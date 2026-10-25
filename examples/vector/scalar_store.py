"""Ordered scalar capture into UB, followed by whole-vector consumption."""

from ascendflow import language as N


@N.prim_func(disjoint_external=True)
def scalar_store(A: N.Tensor((64,), N.f32), O: N.Tensor((64,), N.f32, access="write")):
    with N.Kernel(1, kind="vector"):
        values = N.alloc_ub((64,), N.f32)
        for i in N.serial(64):
            value = N.load_scalar(A[i : i + 1])
            N.store_scalar(value, out=values[i : i + 1])
        scaled = N.mul(values, 2.0, out=values)
        N.copy(scaled, O)
