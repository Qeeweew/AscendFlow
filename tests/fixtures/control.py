"""Source inputs for recursive control and shared AIV participant tests."""

NESTED = """
def nested(X: N.Tensor((n,k,PARTS,64),N.f32), n: N.int64, k: N.int64,
           PARTS: N.constexpr=3):
    with N.Kernel():
        x=N.alloc_ub((2,64),name="x")
        N.fill(x[0,:],0.0)
        for i in N.serial(1,n,2):
            N.fill(x[0,:],1.0)
            for j in N.Pipelined(k):
                with N.Stage(2,name="children"):
                    N.fill(x[0,:],2.0)
                    for t in N.Pipelined(PARTS):
                        with N.Stage(1,name="load"):
                            N.copy(X[i,j,t,:],x[t%2,:])
                    N.fill(x[0,:],3.0)
            N.fill(x[0,:],4.0)
        N.fill(x[0,:],5.0)
"""


SHARED = """
def shared(A:N.Tensor((GROUPS,n,32,16),N.f16),
           B:N.Tensor((GROUPS,n,16,16),N.f16),
           Scratch:N.Tensor((GROUPS,2,32,16),N.f32,access="write"),
           Y:N.Tensor((GROUPS,n,32,16),N.f32,access="write"),
           n:N.int64,GROUPS:N.constexpr=3):
    with N.Kernel(GROUPS,kind="mixed") as group:
        a1=N.alloc_l1((32,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((32,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((32,16),N.f32)
        x=N.alloc_ub((16,16),N.f32,name="x")
        y=N.alloc_ub((16,16),N.f32,name="y")
        for t in N.Dataflow(n,
                 issue_order={"cube":("produce",),"vector":("consume",)}):
            with N.Cube(), N.Stage(0,name="produce"):
                av=N.copy(A[group,t,:,:],a1)
                bv=N.copy(B[group,t,:,:],b1)
                aa=N.copy(av,a0)
                bb=N.copy(bv,b0)
                acc=N.matmul(aa,bb,accum=c0,initialize=True)
                N.copy(acc,Scratch[group,t%2,:,:])
            with N.Vector() as sid, N.Stage(1,name="consume"):
                tile=N.copy(Scratch[group,t%2,sid*16:(sid+1)*16,:],x)
                value=N.mul(tile,2.0,out=y)
                N.copy(value,Y[group,t,sid*16:(sid+1)*16,:])
"""


CARRY = """
def carry(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((64,),name="x")
        total=7
        for i in N.serial(1,n,2):
            if i<3:
                total=total+i
            else:
                total=total+2
        for j in N.Pipelined(total):
            N.fill(x,1.0)
"""
