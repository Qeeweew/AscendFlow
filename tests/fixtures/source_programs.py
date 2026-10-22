"""Small retained source programs; no physical builder or planner."""

from ascendflow import language as T


def vector_source(count=4):
    runtime = type(count) is not int
    name = count.name if runtime else "tasks"
    return T.from_source(
        f"""
        def scale(X:T.Tensor((2,EXTENT,64),T.f32),
                  Y:T.Tensor((2,EXTENT,64),T.f32,access="write"){", " + name + ":T.i64" if runtime else ""}):
            with T.Kernel(2) as group:
                inputs=T.alloc_ub((2,64),name="inputs")
                outputs=T.alloc_ub((2,64),name="outputs")
                for task in T.Pipelined({name if runtime else count}):
                    with T.Stage(0,name="load"):
                        value=T.copy(X[group,task,:],inputs[task%2,:])
                    with T.Vector(),T.Stage(1,name="compute"):
                        value=T.mul(value,2.0,out=outputs[task%2,:])
                    with T.Stage(2,name="store"):
                        T.copy(value,Y[group,task,:])
        """.replace("EXTENT", name if runtime else str(max(1, count))),
        disjoint_external=True,
    ).to_ir()


def cv_source(count=4):
    runtime = type(count) is not int
    name = count.name if runtime else "tasks"
    return T.from_source(
        f"""
        def cv(Q:T.Tensor((1,EXTENT,16,16),T.f16),
               K:T.Tensor((1,EXTENT,16,16),T.f16),
               V:T.Tensor((1,EXTENT,16,16),T.f16),
               O:T.Tensor((1,EXTENT,16,16),T.f32,access="write"),
               scores:T.Tensor((1,2,16,16),T.f32,access="write"),
               probabilities:T.Tensor((1,2,16,16),T.f16,access="write"),
               pv:T.Tensor((1,2,16,16),T.f32,access="write"){", " + name + ":T.i64" if runtime else ""}):
            with T.Kernel(1,kind="mixed") as group:
                q=T.alloc_l1((16,16))
                k=T.alloc_l1((2,16,16))
                p=T.alloc_l1((2,16,16))
                v=T.alloc_l1((2,16,16))
                qa=T.alloc_l0a((16,16))
                kb=T.alloc_l0b((16,16))
                pa=T.alloc_l0a((16,16))
                vb=T.alloc_l0b((16,16))
                sc=T.alloc_l0c((16,16))
                pc=T.alloc_l0c((16,16))
                su=T.alloc_ub((2,8,16))
                pu=T.alloc_ub((2,8,16),T.f16)
                ou=T.alloc_ub((2,8,16))
                for task in T.Dataflow({name if runtime else count}):
                    slot=task%2
                    with T.Cube(),T.Stage(0,name="qk"):
                        T.copy(Q[group,task,:,:],q)
                        T.copy(K[group,task,:,:],k[slot,:,:])
                        T.copy(q,qa)
                        T.copy(k[slot,:,:],kb,transpose=True)
                        score=T.matmul(qa,kb,accum=sc,initialize=True)
                        T.copy(score,scores[group,slot,:,:])
                    with T.Vector() as lane,T.Stage(1,name="vector"):
                        value=T.copy(scores[group,slot,lane*8:lane*8+8,:],su[slot,:,:])
                        value=T.mul(value,2.0,out=su[slot,:,:])
                        value=T.cast(value,out=pu[slot,:,:])
                        T.copy(value,probabilities[group,slot,lane*8:lane*8+8,:])
                    with T.Cube(),T.Stage(1,name="pv"):
                        T.copy(probabilities[group,slot,:,:],p[slot,:,:])
                        T.copy(V[group,task,:,:],v[slot,:,:])
                        T.copy(p[slot,:,:],pa)
                        T.copy(v[slot,:,:],vb)
                        result=T.matmul(pa,vb,accum=pc,initialize=True)
                        T.copy(result,pv[group,slot,:,:])
                    with T.Vector() as lane,T.Stage(2,name="output"):
                        result=T.copy(pv[group,slot,lane*8:lane*8+8,:],ou[slot,:,:])
                        T.copy(result,O[group,task,lane*8:lane*8+8,:])
        """.replace("EXTENT", name if runtime else str(max(1, count))),
        disjoint_external=True,
    ).to_ir()
