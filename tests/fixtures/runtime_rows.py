"""Actual DSL inputs for bounded row transfers; no alternate implementation."""

from ascendflow import language as N


def matrix_import(dtype="f16"):
    return N.from_source(
        """
def rows_import(A:N.Tensor((n,16),N.f16), B:N.Tensor((16,16),N.f16),
                O:N.Tensor((16,16),N.f32,access="write",write_disjoint=True),n:N.i64):
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
""".replace("N.f16", f"N.{dtype}"),
        disjoint_external=True,
    )


def row_output():
    return N.from_source(
        """
def rows_output(O:N.Tensor((n,16),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel():
        x=N.alloc_ub((4,16),N.f32)
        N.fill(x,3.0)
        N.copy(x,O[0:4,:],valid_rows=N.min(4,n))
""",
        disjoint_external=True,
    )


def matrix_output(dtype="bf16", columns=48):
    """A full NZ producer with a runtime row prefix read by FIXPIPE."""
    return N.from_source(
        f"""
def rows_fix(A:N.Tensor((16,16),N.{dtype}), B:N.Tensor((16,{columns}),N.{dtype}),
             O:N.Tensor((n,{columns}),N.{dtype},access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.{dtype})
        b1=N.alloc_l1((16,{columns}),N.{dtype})
        a0=N.alloc_l0a((16,16),N.{dtype})
        b0=N.alloc_l0b((16,{columns}),N.{dtype})
        c0=N.alloc_l0c((16,{columns}),N.f32)
        av=N.copy(A,a1)
        bv=N.copy(B,b1)
        aa=N.copy(av,a0)
        bb=N.copy(bv,b0)
        value=N.matmul(aa,bb,accum=c0,initialize=True)
        N.copy(value,O[0:16,:],valid_rows=N.min(n,16))
""",
        disjoint_external=True,
    )


def fp32_padding(*, runtime=False, pitch=19):
    prefix = ",valid_rows=N.min(n,3)" if runtime else ""
    return N.from_source(f"""
def fp32_padding(A:N.Tensor((3,{pitch}),N.f32),n:N.i64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1((16,16),N.f32)
        y=N.alloc_l0a((16,16),N.f32)
        N.copy(A[:,:5],x,pad_value=0{prefix})
        N.copy(x,y)
""")


def matrix_overlay():
    """Identity GEMM exposes every preserved/overlaid NZ row and column."""
    return N.from_source(
        """
def overlay(A:N.Tensor((32,64),N.f16), B:N.Tensor((32,64),N.f16),
            Identity:N.Tensor((64,64),N.f16),
            O:N.Tensor((32,64),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((32,64),N.f16)
        b1=N.alloc_l1((64,64),N.f16)
        a0=N.alloc_l0a((32,64),N.f16)
        b0=N.alloc_l0b((64,64),N.f16)
        c0=N.alloc_l0c((32,64),N.f32)
        N.copy(A,a1)
        N.copy(B[:27,:],a1[5:32,:],valid_rows=N.min(n,27))
        N.copy(Identity,b1)
        aa=N.copy(a1,a0)
        bb=N.copy(b1,b0)
        N.matmul(aa,bb,accum=c0,initialize=True)
        N.copy(c0,O)
""",
        disjoint_external=True,
    )
