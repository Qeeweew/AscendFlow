"""Actual DSL inputs for bounded row transfers; no alternate implementation."""

from ascendflow import language as N


def matrix_import():
    return N.from_source(
        """
def rows_import(A:N.Tensor((n,16),N.f16), B:N.Tensor((16,16),N.f16),
                O:N.Tensor((16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((16,16),N.f32)
        N.copy(A[0:16,:],a1,valid_rows=N.min(16,n),pad_value=0)
        N.copy(B,b1)
        aa=N.copy(a1,a0)
        bb=N.copy(b1,b0)
        N.matmul(aa,bb,accum=c0,initialize=True)
        N.copy(c0,O)
""",
        disjoint_external=True,
    )


def row_output():
    return N.from_source(
        """
def rows_output(O:N.Tensor((n,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((4,16),N.f32)
        N.fill(x,3.0)
        N.copy(x,O[0:4,:],valid_rows=N.min(4,n))
""",
        disjoint_external=True,
    )
