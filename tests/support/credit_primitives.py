"""Explicit primitive probes for a proposed persistent ownership rule.

These fixtures replace selected instructions on a real, compiled DSL program.
They qualify the rule and target primitives, not compiler inference of that
rule. Neither protocol names nor the old resource plan qualify the mutation.
"""

from dataclasses import replace

from ascendflow import language as N
from ascendflow.analysis.smt import query_scope
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_dispatch import bind_structured_dispatch
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement, SyncAction
from ascendflow.passes.generate_notifications import generate_notifications
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.structured_sync import _materialize_owned
from ascendflow.scalar import expression, literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.support.structured import allocated

LOCAL_SOURCE = """
def local_free(X:N.Tensor((n,4,64),N.f32),
               Out:N.Tensor((n,4,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        scratch=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for parent in N.serial(n):
                for child in N.Pipelined((parent//2)%5):
                    with N.Stage(0,name="load"):
                        N.copy(X[parent,child,:],scratch[child%2,:])
                    with N.Stage(1,name="store"):
                        N.copy(scratch[child%2,:],Out[parent,child,:])
"""


CROSS_SOURCE = """
def cross_free(X:N.Tensor((16,16),N.f16),
               Work:N.Tensor((2,16,16),N.f32,access="write"),
               Out:N.Tensor((n,4,16,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="mixed"):
        a1=N.alloc_l1((16,16),N.f16)
        b1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        b0=N.alloc_l0b((16,16),N.f16)
        c0=N.alloc_l0c((2,16,16),N.f32)
        y=N.alloc_ub((8,16),N.f32)
        with N.Cube():
            N.copy(X[:,:],a1)
            N.copy(X[:,:],b1)
            N.copy(a1,a0)
            N.copy(b1,b0,transpose=True)
        for parent in N.serial(n):
            for child in N.Dataflow((parent//2)%5):
                with N.Cube(), N.Stage(0,name="produce"):
                    N.matmul(a0,b0,accum=c0[child%2,:,:],initialize=True)
                    N.copy(c0[child%2,:,:],Work[child%2,:,:])
                with N.Vector() as lane, N.Stage(0,name="consume"):
                    N.copy(Work[child%2,lane*8:(lane+1)*8,:],y)
                    N.copy(y,Out[parent,child,lane*8:(lane+1)*8,:])
"""


@query_scope()
def seeded_credit_probe(*, cross=False, remove=None, missing_lane=False, late_reclaim=False):
    """Return actual seeded instructions and an independent native snapshot."""
    structure = lower_native_structure(
        N.from_source(CROSS_SOURCE if cross else LOCAL_SOURCE, disjoint_external=True)
    )
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    admission = admit_structured_memory(structure, memory, bounds)
    dispatch = bind_structured_dispatch(memory)
    dependencies = build_structured_dependencies(memory, admission=admission, dispatch=dispatch)
    generation = generate_notifications(dependencies, dispatch)
    resources = allocated(generation)
    original = collect_native_memory(structure.module.clone())
    root = structure.module.body.block.first_op
    if cross:
        writer = next(t for t in memory.templates if t.spec.engine is Engine.FIX)
        reader = next(
            t
            for t in memory.templates
            if any(a.address.root.owner.sym_name.data == "Work" for a in t.accesses)
            and t.spec.engine is Engine.MTE2
        )
        source, destination = Engine.MTE2, Engine.FIX
        producers, consumers = (Actor.AIV0, Actor.AIV1), (Actor.AIC,)
        ready_source, ready_destination = Engine.FIX, Engine.MTE2
        set_kind, wait_kind, stride = SyncKind.CROSS_SET, SyncKind.CROSS_WAIT, 1024
        write_offset = writer.accesses[-1].address.offset
        read_offset = next(
            a.address.offset
            for a in reader.accesses
            if a.address.root.owner.sym_name.data == "Work"
        )
    else:
        writer, reader = memory.templates
        source, destination = Engine.MTE3, Engine.MTE2
        producers = consumers = (Actor.AIV0,)
        ready_source, ready_destination = Engine.MTE2, Engine.MTE3
        set_kind, wait_kind, stride = SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT, 256
        write_offset, read_offset = (
            writer.accesses[-1].address.offset,
            reader.accesses[0].address.offset,
        )
    retained = [
        action
        for action in resources.actions
        if not (
            action.kind in {SyncKind.CROSS_SET, SyncKind.CROSS_WAIT}
            if cross
            else action.kind in {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT}
            and action.destination is not Engine.S
        )
    ]
    actions = []

    def add(anchor, placement, participants, kind, src, dst, event, name, guard=None):
        if name != remove:
            actions.append(
                SyncAction(
                    anchor,
                    placement,
                    participants,
                    kind,
                    src,
                    dst,
                    literal(event),
                    ActionGuard() if guard is None else guard,
                    (name,),
                )
            )

    for slot in range(2):
        free, ready = slot, slot + 2 if cross else slot
        write_guard, read_guard = (
            ActionGuard(
                (Comparison(expression("floordiv", offset, literal(stride)), "==", literal(slot)),)
            )
            for offset in (write_offset, read_offset)
        )
        add(
            root,
            Placement.ENTRY,
            (Actor.AIV0,) if cross and missing_lane else producers,
            set_kind,
            source,
            destination,
            free,
            "seed",
        )
        add(
            writer.operation,
            Placement.BEFORE,
            consumers,
            wait_kind,
            source,
            destination,
            free,
            "acquire",
            write_guard,
        )
        add(
            writer.operation,
            Placement.AFTER,
            consumers,
            set_kind,
            ready_source,
            ready_destination,
            ready,
            "ready",
            write_guard,
        )
        add(
            reader.operation,
            Placement.BEFORE,
            producers,
            wait_kind,
            ready_source,
            ready_destination,
            ready,
            "read",
            read_guard,
        )
        add(
            reader.operation,
            Placement.AFTER,
            producers,
            set_kind,
            source,
            destination,
            free,
            "return",
            read_guard,
        )
        add(root, Placement.EXIT, consumers, wait_kind, source, destination, free, "reclaim")
    if late_reclaim:
        # A local Wait does not stop the controller. Reclaim after its final
        # queue->Scalar drain can leak the credit into the next invocation.
        final = [action for action in actions if action.protocols == ("reclaim",)]
        actions = [action for action in actions if action.protocols != ("reclaim",)]
        actions = actions + retained + final
    else:
        actions = actions + retained
    plan = _materialize_owned(
        structure,
        admission,
        dependencies,
        generation,
        replace(resources, actions=tuple(actions)),
        (),
    )
    return plan, original
