"""CV compiler counterexamples, not an operator API or nested-AST support claim.

The AST case uses one retained outer pipeline. The segmented physical case
expresses a small, fixed set of native contributions directly, so removing the
Python tracer does not remove the complete-contribution/early-release tests.
"""

from xdsl.dialects.builtin import StringAttr

from xdsl_npu import language as T
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.mixed import compose
from xdsl_npu.model import require


def vector_source(count=4):
    runtime = type(count) is not int
    name = count.name if runtime else "tasks"
    return T.from_source(
        f"""
        def scale(X:T.Tensor((2,EXTENT,64),T.f32),
                  Y:T.Tensor((2,EXTENT,64),T.f32,access="write"){", " + name + ":T.int64" if runtime else ""}):
            with T.Kernel(2) as group:
                inputs=T.alloc_ub((2,64),name="inputs")
                outputs=T.alloc_ub((2,64),name="outputs")
                for task in T.Pipelined({name if runtime else count},stages=3):
                    with T.Stage(0,name="load"):
                        value=T.copy(X[group,task,:],inputs[task%2,:])
                    with T.Stage(1,name="compute"),T.Vector():
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
               pv:T.Tensor((1,2,16,16),T.f32,access="write"){", " + name + ":T.int64" if runtime else ""}):
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
                for task in T.Pipelined({name if runtime else count},stages=3):
                    slot=task%2
                    with T.Stage(0,name="qk"),T.Cube():
                        T.copy(Q[group,task,:,:],q)
                        T.copy(K[group,task,:,:],k[slot,:,:])
                        T.copy(q,qa)
                        T.copy(k[slot,:,:],kb,transpose=True)
                        score=T.matmul(qa,kb,accum=sc,initialize=True)
                        T.copy(score,scores[group,slot,:,:])
                    with T.Stage(1,name="vector"),T.Vector() as lane:
                        value=T.copy(scores[group,slot,lane*8:lane*8+8,:],su[slot,:,:])
                        value=T.mul(value,2.0,out=su[slot,:,:])
                        value=T.cast(value,out=pu[slot,:,:])
                        T.copy(value,probabilities[group,slot,lane*8:lane*8+8,:])
                    with T.Stage(1,name="pv"),T.Cube():
                        T.copy(probabilities[group,slot,:,:],p[slot,:,:])
                        T.copy(V[group,task,:,:],v[slot,:,:])
                        T.copy(p[slot,:,:],pa)
                        T.copy(v[slot,:,:],vb)
                        result=T.matmul(pa,vb,accum=pc,initialize=True)
                        T.copy(result,pv[group,slot,:,:])
                    with T.Stage(2,name="output"),T.Vector() as lane:
                        result=T.copy(pv[group,slot,lane*8:lane*8+8,:],ou[slot,:,:])
                        T.copy(result,O[group,task,lane*8:lane*8+8,:])
        """.replace("EXTENT", name if runtime else str(max(1, count))),
        disjoint_external=True,
    ).to_ir()


def cv_program(count=4, segments=1, groups=1):
    """Small native three-ring producer/consumer test; no user DSL execution."""
    require(type(segments) is int and 1 <= segments <= 4, "test segments must be 1..4")
    require(type(groups) is int and 1 <= groups <= 2, "test groups must be 1..2")
    runtime = type(count) is not int

    def actor(role):
        k = Kernel(
            "aic" if role == 0 else f"aiv{role - 1}",
            core="aic" if role == 0 else "aiv",
            blocks=groups,
            disjoint_external=True,
        )
        bound = k.runtime_size(count.name) if runtime else count

        def external(name, size, dtype="f16", initialized=True):
            return k.buffer(
                name,
                "GM",
                0 if runtime else max(1, count) * groups * size,
                dtype,
                external=True,
                initialized=initialized,
                extent_count=bound if runtime else None,
                count_scale=groups * size if runtime else 0,
            )

        q = external("Q", 512)
        key = external("K", segments * 512)
        value = external("V", segments * 512)
        output = external("O", 1024, "f32", False)
        score = k.buffer("scores", "GM", groups * 2 * segments * 1024, external=True)
        probability = k.buffer(
            "probabilities", "GM", groups * 2 * segments * 512, "f16", external=True
        )
        pv = k.buffer("pv", "GM", groups * 2 * 1024, external=True)
        if role == 0:
            q1 = k.buffer("q1", "L1", 512, "f16", layout="NZ")
            k1 = k.buffer("k1", "L1", 1024, "f16", layout="NZ")
            p1 = k.buffer("p1", "L1", 1024, "f16", layout="NZ")
            v1 = k.buffer("v1", "L1", 1024, "f16", layout="NZ")
            qa = k.buffer("qa", "L0A", 512, "f16", layout="ZZ")
            kb = k.buffer("kb", "L0B", 512, "f16", layout="ZN")
            pa = k.buffer("pa", "L0A", 512, "f16", layout="ZZ")
            vb = k.buffer("vb", "L0B", 512, "f16", layout="ZN")
            sc = k.buffer("sc", "L0C", 1024, layout="NZ")
            pc = k.buffer("pc", "L0C", 1024, layout="NZ")
            k.init_fixpipe()
        else:
            su = k.buffer("su", "UB", 1024)
            pu = k.buffer("pu", "UB", 512, "f16")
            ou = k.buffer("ou", "UB", 1024)
        k.loop(bound)

        def stream(root, total, nbytes, **kwargs):
            return k.view(
                root,
                nbytes,
                stride=total,
                extent_count=bound if runtime else None,
                count_core_stride=total if runtime else 0,
                core_stride=0 if runtime else count * total,
                **kwargs,
            )

        def ring(root, total, nbytes, **kwargs):
            return k.view(root, nbytes, stride=total, slots=2, core_stride=2 * total, **kwargs)

        def local(root, size, part=0):
            # The two inner-prefetch slots alternate by the static chunk;
            # cross-task reuse must be proved, never declared independent.
            return k.view(root, size, offset=(part % 2) * size)

        def task(uid, native, *refs, stage=0, phase=0, **kwargs):
            op = k.task(uid, native, *refs, stage=stage, logical_phase=phase, **kwargs)
            op.attributes["source_region"] = StringAttr(
                "qk" if phase == 0 else "vector" if phase == 1 else "pv" if phase == 2 else "output"
            )
            return op

        if role == 0:
            qr = k.view(q1, 512)
            task(
                "qk_load_q",
                "nd2nz",
                stream(q, 512, 32, rows=16, row_stride=32),
                qr,
                params=(16, 16),
            )

            def load_key(part):
                task(
                    f"qk_load_k{part}",
                    "nd2nz",
                    stream(key, segments * 512, 32, offset=part * 512, rows=16, row_stride=32),
                    local(k1, 512, part),
                    params=(16, 16),
                )

            load_key(0)
            for part in range(segments):
                if part + 1 < segments:
                    load_key(part + 1)
                ar, br, cr = k.view(qa, 512), k.view(kb, 512), k.view(sc, 1024)
                task(f"qk_operand_a{part}", "load2d_a", qr, ar, params=(16, 16))
                task(
                    f"qk_operand_b{part}", "load2d_b_nt", local(k1, 512, part), br, params=(16, 16)
                )
                task(f"qk_mad{part}", "mad", ar, br, cr, params=(16, 16, 16, 1))
                task(
                    f"qk_export{part}",
                    "fix",
                    cr,
                    ring(
                        score,
                        segments * 1024,
                        64,
                        offset=part * 64,
                        rows=16,
                        row_stride=segments * 64,
                    ),
                    params=(16, 16),
                )

            def load_pv(part):
                task(
                    f"pv_load_p{part}",
                    "nd2nz",
                    ring(
                        probability,
                        segments * 512,
                        32,
                        offset=part * 32,
                        rows=16,
                        row_stride=segments * 32,
                    ),
                    local(p1, 512, part),
                    stage=1,
                    phase=2,
                    params=(16, 16),
                )
                task(
                    f"pv_load_v{part}",
                    "nd2nz",
                    stream(value, segments * 512, 32, offset=part * 512, rows=16, row_stride=32),
                    local(v1, 512, part),
                    stage=1,
                    phase=2,
                    params=(16, 16),
                )

            load_pv(0)
            for part in range(segments):
                if part + 1 < segments:
                    load_pv(part + 1)
                ar, br, cr = k.view(pa, 512), k.view(vb, 512), k.view(pc, 1024)
                task(
                    f"pv_operand_a{part}",
                    "load2d_a",
                    local(p1, 512, part),
                    ar,
                    stage=1,
                    phase=2,
                    params=(16, 16),
                )
                task(
                    f"pv_operand_b{part}",
                    "load2d_b_nn",
                    local(v1, 512, part),
                    br,
                    stage=1,
                    phase=2,
                    params=(16, 16),
                )
                task(
                    f"pv_mad{part}",
                    "mad",
                    ar,
                    br,
                    cr,
                    stage=1,
                    phase=2,
                    params=(16, 16, 16, int(part == 0)),
                )
            task(
                "pv_export",
                "fix",
                k.view(pc, 1024),
                ring(pv, 1024, 64, rows=16, row_stride=64),
                stage=1,
                phase=2,
                params=(16, 16),
            )
        else:
            lane = role - 1
            for part in range(segments):
                a, b = local(su, 512, part), local(pu, 256, part)
                task(
                    f"vector_import{part}",
                    "copy_rows",
                    ring(
                        score,
                        segments * 1024,
                        64,
                        offset=lane * 8 * segments * 64 + part * 64,
                        rows=8,
                        row_stride=segments * 64,
                    ),
                    k.view(su, 64, offset=(part % 2) * 512, rows=8, row_stride=64),
                    stage=1,
                    phase=1,
                )
                task(f"vector_scale{part}", "muls", a, a, scalar=2, stage=1, phase=1)
                task(f"vector_round{part}", "convert_rn", a, b, stage=1, phase=1)
                task(
                    f"vector_export{part}",
                    "copy_rows",
                    k.view(pu, 32, offset=(part % 2) * 256, rows=8, row_stride=32),
                    ring(
                        probability,
                        segments * 512,
                        32,
                        offset=lane * 8 * segments * 32 + part * 32,
                        rows=8,
                        row_stride=segments * 32,
                    ),
                    stage=1,
                    phase=1,
                )
            local_output = k.view(ou, 512, stride=512, slots=2)
            task(
                "output_import",
                "copy",
                ring(pv, 1024, 512, offset=lane * 512),
                local_output,
                stage=2,
                phase=3,
            )
            task(
                "output_store",
                "copy",
                local_output,
                stream(output, 1024, 512, offset=lane * 512),
                stage=2,
                phase=3,
            )
        return k.finish()

    return compose("segmented_cv", *(actor(role) for role in range(3)), disjoint_external=True)


def segmented_cv_program(count=4, segments=3, groups=2):
    return cv_program(count, segments, groups)
