"""Real source fixtures for fixed-child row geometry and boundary composition."""

from xdsl_npu import language as N


@N.prim_func(disjoint_external=True)
def pitched_children(
    X: N.Tensor((2, 4, 32), N.f32),
    Y: N.Tensor((2, 4, 32), N.f32, access="write"),
    O: N.Tensor((2, 4, 32), N.f32, access="write"),
):
    with N.Kernel(2, kind="vector") as group:
        tiles = N.alloc_ub((2, 4, 8), N.f32)
        full = N.alloc_ub((4, 32), N.f32)
        for j in N.Pipelined(4, stages=3):
            with N.Stage(0, name="load"):
                value = N.copy(X[group, :, j * 8 : (j + 1) * 8], tiles[j % 2, :, :])
            with N.Stage(1, name="scale"):
                scaled = N.mul(value, 2.0, out=tiles[j % 2, :, :])
            with N.Stage(2, name="store"):
                N.copy(scaled, Y[group, :, j * 8 : (j + 1) * 8])
        for k in N.Pipelined(1, stages=1):
            imported = N.copy(Y[group, :, :], full)
            N.copy(imported, O[group, :, :])


@N.prim_func(disjoint_external=True)
def cube_columns(
    X: N.Tensor((2, 8, 16, 16), N.f16),
    Y: N.Tensor((2, 16, 128), N.f32, access="write"),
):
    with N.Kernel(2, kind="cube") as group:
        l1 = N.alloc_l1((2, 16, 16))
        a = N.alloc_l0a((2, 16, 16))
        b = N.alloc_l0b((2, 16, 16))
        c = N.alloc_l0c((2, 16, 16))
        for i in N.Pipelined(8, stages=4):
            with N.Stage(0, name="load"):
                x = N.copy(X[group, i, :, :], l1[i % 2, :, :])
            with N.Stage(1, name="operands"):
                aa = N.copy(x, a[i % 2, :, :])
                bb = N.copy(x, b[i % 2, :, :], transpose=True)
            with N.Stage(2, name="mad"):
                z = N.matmul(aa, bb, accum=c[i % 2, :, :], initialize=True)
            with N.Stage(3, name="export"):
                N.copy(z, Y[group, :, i * 16 : (i + 1) * 16])
