"""Uncertain aliases preserve source byte versions through the normal pipeline."""

from itertools import product

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_regions import AliasRegion
from ascendflow.dialects.io import parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.target.ascend910b import Engine, SyncKind
from tests.models.structured_events import EventModelError, check_structured_events

ALIASES = """
def aliases(X:N.Tensor((2,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),
            n:N.int64,a:N.int64,b:N.int64,c:N.int64):
    with N.Kernel():
        x=N.alloc_ub((4,64))
        y=N.alloc_ub((64,))
        N.fill(x,1.0)
        for i in N.Pipelined(n):
            N.copy(X[i%2,:],x[(a+i)%4,:])
            N.mul(x[(b+i)%4,:],2.0,out=y)
            N.copy(x[(b+i)%4,:],Y[i,:])
            N.fill(x[(c+i)%4,:],3.0)
"""


def _memory(source):
    result = compile(N.from_source(source, disjoint_external=True))
    return collect_native_memory(result.plan), collect_native_memory(parse(result.before_ir))


def test_uncertain_aliases_keep_both_reader_engines_and_empty_short_loop_versions():
    final, original = _memory(ALIASES)
    hazards = set()
    for n, a, b, c in product((0, 1, 2, 5), (0, 1), (0, 1), (0, 3)):
        checked = check_structured_events(
            final, {"n": n, "a": a, "b": b, "c": c}, logical_reference=original
        )
        hazards.update(hazard.kind for hazard in checked.hazards)
    assert hazards >= {"RAW", "WAR", "WAW"}


def test_uncertain_waw_without_a_reader_still_needs_completion():
    final, original = _memory("""
    def writes(X:N.Tensor((64,),N.f32),a:N.int64,b:N.int64):
        with N.Kernel():
            x=N.alloc_ub((4,64))
            N.fill(x[a%4,:],1.0)
            N.copy(X,x[b%4,:])
    """)
    check_structured_events(final, {"a": 0, "b": 0}, logical_reference=original)
    for op in tuple(final.module.walk()):
        if (
            isinstance(op, NotificationOp)
            and (op.source.data, op.destination.data) == (Engine.V.value, Engine.MTE2.value)
            and op.kind.data in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}
        ):
            op.parent.erase_op(op)
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(final, {"a": 0, "b": 0}, logical_reference=original)


def test_alias_regions_are_ordering_facts_not_exact_kills_or_private_packet_credits():
    from ascendflow.analysis.structured_dependencies import build_structured_dependencies
    from ascendflow.passes.lower_native import lower_native_structure

    memory = collect_native_memory(lower_native_structure(N.from_source(ALIASES)).module)
    dependencies = build_structured_dependencies(memory)
    assert any(
        isinstance(n.cell, AliasRegion) and n.kind is HazardKind.RAW for n in dependencies.required
    )
    assert not hasattr(dependencies, "values")
    assert any(
        isinstance(n.cell, AliasRegion) and n.kind is HazardKind.WAR for n in dependencies.required
    )
    assert all(not isinstance(p.cell, AliasRegion) for p in dependencies.order.packet_partitions)


def test_shifted_ring_reads_preserve_the_previous_iteration_value():
    final, original = _memory("""
    def shifted(X:N.Tensor((n,64),N.f32),Y:N.Tensor((n,64),N.f32,access="write"),n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((2,64))
            N.fill(x,1.0)
            for i in N.Pipelined(n):
                N.copy(X[i,:],x[i%2,:])
                N.copy(x[(i+1)%2,:],Y[i,:])
""")
    for n in (0, 1, 2, 3, 7):
        check_structured_events(final, {"n": n}, logical_reference=original)


def test_uncertain_shared_gm_retains_both_aiv_readers_and_actual_quorum():
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from ascendflow.target.ascend910b import Actor

    final, original = _memory("""
    def mixed(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
              Work:N.Tensor((2,16,16),N.f32,access="write"),
              Y:N.Tensor((n,16,16),N.f32,access="write",write_disjoint=True),
              n:N.int64,a:N.int64):
        with N.Kernel(1,kind="mixed"):
            a1=N.alloc_l1((16,16),N.f16)
            b1=N.alloc_l1((16,16),N.f16)
            a0=N.alloc_l0a((16,16),N.f16)
            b0=N.alloc_l0b((16,16),N.f16)
            c0=N.alloc_l0c((16,16),N.f32)
            x=N.alloc_ub((8,16),N.f32)
            for i in N.Dataflow(n):
                slot=(a+i)%2
                with N.Cube():
                    N.copy(A,a1)
                    N.copy(B,b1)
                    N.copy(a1,a0)
                    N.copy(b1,b0)
                    N.matmul(a0,b0,accum=c0,initialize=True)
                    N.copy(c0,Work[slot,:,:])
                with N.Vector() as sid:
                    N.copy(Work[slot,sid*8:sid*8+8,:],x)
                    N.copy(x,Y[i,sid*8:sid*8+8,:])
""")
    for n, a in product((0, 1, 2, 5), (0, 1, 7)):
        check_structured_events(final, {"n": n, "a": a}, logical_reference=original)
    release = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.CROSS_SET.value
        and op.source.data == Engine.MTE2.value
    )
    release.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
    with pytest.raises(EventModelError, match="quorum|contribution"):
        check_structured_events(final, {"n": 2, "a": 0}, logical_reference=original)


def _metadata_rows(monkeypatch, choices):
    from tests.models.structured_events import _Interpret

    original = _Interpret.native

    def native(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            _, offset, _ = values[operation.refs[0]]
            values[operation.value] = choices[(offset // 4) % len(choices)]

    monkeypatch.setattr(_Interpret, "native", native)


def test_shared_enclosures_keep_group_slots_and_both_aiv_publications():
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from ascendflow.target.ascend910b import Actor

    final, original = _memory("""
def shared(Work:N.Tensor((2,3,16,16),N.f16,access="write"),n:N.int64):
    with N.Kernel(2,kind="mixed") as group:
        x=N.alloc_ub((8,16),N.f16)
        values=N.alloc_ub((8,16),N.f32)
        a1=N.alloc_l1((16,16),N.f16)
        a0=N.alloc_l0a((16,16),N.f16)
        for i in N.Dataflow(n):
            with N.Vector() as sid,N.Stage(0,name="produce"):
                N.fill(values,1.0)
                N.cast(values,out=x)
                N.copy(x,Work[group,i%3,sid*8:sid*8+8,:])
            with N.Cube(),N.Stage(1,name="consume"):
                N.copy(Work[group,i%3,:,:],a1)
                N.copy(a1,a0)
""")
    for n in (0, 1, 2, 3, 7):
        # Raw mixed controller bodies have no cross-controller reentry
        # contract. Sequential kernel launches use runtime completion.
        check_structured_events(final, {"n": n}, logical_reference=original, groups=(0, 1))
    ready = next(
        op
        for op in final.module.walk()
        if isinstance(op, NotificationOp)
        and op.kind.data == SyncKind.CROSS_SET.value
        and op.source.data == Engine.MTE3.value
    )
    ready.attributes["participants"] = ArrayAttr([StringAttr(Actor.AIV0.value)])
    with pytest.raises(EventModelError, match="quorum|contribution|deadlock"):
        check_structured_events(final, {"n": 7}, logical_reference=original, groups=(0, 1))


def _fragment_ring(slots, *, complete=True, surrounding=False):
    initial = (
        """
        N.copy(A,x[0,:,:])
        N.copy(x[0,:,:],a0[0,:,:])
"""
        if surrounding
        else ""
    )
    final = "        N.copy(A,x[0,:,:])" if surrounding else ""
    fill = "N.copy(A,x[slot,:,:])" if complete else "pass"
    return f"""
def fragments(A:N.Tensor((16,32),N.f16),Rows:N.Tensor((4,),N.i32),n:N.int64):
    with N.Kernel(kind="cube"):
        x=N.alloc_l1(({slots},16,32),N.f16)
        a0=N.alloc_l0a((2,16,32),N.f16)
{initial}
        for i in N.Pipelined(n):
            slot=i%{slots}
            with N.Stage(0,name="load"):
                row=N.load_index(Rows[i%4:i%4+1],minimum=0,maximum=15)
                {fill}
                N.copy(A[2:3,:],x[slot,row:row+1,:])
            with N.Stage(1,name="read"):
                N.copy(x[slot,:,:],a0[i%2,:,:])
{final}
"""


@pytest.mark.parametrize("slots", (2, 3))
def test_dynamic_nz_fragments_keep_disjoint_slots_and_same_slot_retirement(slots, monkeypatch):
    _metadata_rows(monkeypatch, (0, 15, 3, 8))
    final, original = _memory(_fragment_ring(slots))
    for n in (0, 1, 2, 3, 5, 8):
        execution = check_structured_events(
            final, {"n": n}, logical_reference=original, invocations=2
        )
        assert execution.bytes_read or n == 0
    removed = 0
    for op in tuple(final.module.walk()):
        if (
            isinstance(op, NotificationOp)
            and (op.source.data, op.destination.data) == (Engine.MTE1.value, Engine.MTE2.value)
            and op.kind.data in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}
        ):
            op.parent.erase_op(op)
            removed += 1
    assert removed
    with pytest.raises(EventModelError, match="unordered physical WAR|binary event overwritten"):
        check_structured_events(final, {"n": 8}, logical_reference=original, invocations=2)


def test_empty_fragment_child_preserves_the_incoming_last_reader(monkeypatch):
    _metadata_rows(monkeypatch, (0, 15, 3, 8))
    final, original = _memory(_fragment_ring(3, surrounding=True))
    for n in (0, 1, 2, 5):
        check_structured_events(final, {"n": n}, logical_reference=original, invocations=2)


def test_enclosing_region_cannot_initialize_untouched_nz_rows():
    from ascendflow.diagnostics import CompileError

    with pytest.raises(CompileError, match="initialized writer coverage"):
        _memory(_fragment_ring(3, complete=False))


def test_enclosing_region_does_not_authorize_an_out_of_bounds_fragment():
    from ascendflow.diagnostics import CompileError

    source = _fragment_ring(3).replace("maximum=15", "maximum=16")
    with pytest.raises(CompileError, match="axis|bound|allocation"):
        _memory(source)


def test_single_slot_keeps_the_old_reader_retirement(monkeypatch):
    _metadata_rows(monkeypatch, (0, 15, 3, 8))
    final, original = _memory(_fragment_ring(1))
    execution = check_structured_events(final, {"n": 5}, logical_reference=original, invocations=2)
    assert any(h.kind == "WAR" and h.root.ref.type.bank.data == "L1" for h in execution.hazards)


@pytest.mark.parametrize("slots", (3, 4))
def test_linear_enclosures_preserve_both_vector_and_dma_last_readers(slots, monkeypatch):
    _metadata_rows(monkeypatch, (0, 7, 3, 6))
    final, original = _memory(f"""
def fragments(A:N.Tensor((4,64),N.f32),Rows:N.Tensor((4,),N.i32),
              O:N.Tensor((n,8,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub(({slots},8,64),N.f32)
        y=N.alloc_ub((8,64),N.f32)
        for i in N.Pipelined(n):
            slot=i%{slots}
            with N.Stage(0,name="load"):
                row=N.load_index(Rows[i%4:i%4+1],minimum=0,maximum=7)
                N.fill(x[slot,:,:],0.0)
                N.copy(A[i%4,:],x[slot,row,:])
            with N.Stage(1,name="vector_read"):
                N.mul(x[slot,:,:],2.0,out=y)
            with N.Stage(2,name="dma_read"):
                N.copy(x[slot,:,:],O[i,:,:])
""")
    for n in (0, 1, 2, 5, 9):
        check_structured_events(final, {"n": n}, logical_reference=original, invocations=2)
    removed = 0
    for op in tuple(final.module.walk()):
        if (
            isinstance(op, NotificationOp)
            and (op.source.data, op.destination.data) == (Engine.MTE3.value, Engine.V.value)
            and op.kind.data in {SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value}
        ):
            op.parent.erase_op(op)
            removed += 1
    assert removed
    with pytest.raises(EventModelError, match="unordered physical WAR|binary event overwritten"):
        check_structured_events(final, {"n": 9}, logical_reference=original, invocations=2)
