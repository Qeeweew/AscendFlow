"""Nested slot source input shared by event-domain and protocol tests."""


def nested_slot_source(slots):
    return f"""
    def slots(X:N.Tensor((n,m,64),N.f32),n:N.i64,m:N.i64):
        with N.Kernel():
            x,y=N.alloc_ub(({slots},64)),N.alloc_ub(({slots},64))
            for p in N.serial(n):
                for g in N.Pipelined(m):
                    slot=(p*m+g)%{slots}
                    with N.Stage(0):
                        N.copy(X[p,g,:],x[slot,:])
                    with N.Stage(1):
                        N.mul(x[slot,:],2.0,out=y[slot,:])
"""
