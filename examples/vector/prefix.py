"""Runtime element lengths, exact aligned tails, and explicit double buffering.

Process min(n, m) elements. The rest of Y is untouched; neither allocation is
padded to the pipeline tile. The launch contract requires min(n, m) % 8 == 0.
"""

from __future__ import annotations

from xdsl_npu import language as N


@N.prim_func(disjoint_external=True)
def scale_prefix(
    X: N.Tensor((n,), N.f32),  # noqa: F821 - DSL signature scope
    Y: N.Tensor((m,), N.f32, access="write"),  # noqa: F821
    n: N.int64,
    m: N.int64,
    B: N.constexpr = 64,
):
    with N.Kernel(1, kind="vector"):
        x = N.alloc_ub((2, B), N.f32, name="inputs")
        y = N.alloc_ub((2, B), N.f32, name="outputs")
        length = N.min(n, m)
        for t in N.Pipelined(N.ceildiv(length, B), stages=3):
            valid = N.min(B, length - t * B)
            slot = t % 2
            with N.Stage(0, name="load"):
                N.copy(X[t * B : (t + 1) * B], x[slot, :], valid=valid)
            with N.Stage(1, name="compute"), N.Vector():
                N.mul(x[slot, :], 2.0, out=y[slot, :], valid=valid)
            with N.Stage(2, name="store"):
                N.copy(y[slot, :], Y[t * B : (t + 1) * B], valid=valid)
