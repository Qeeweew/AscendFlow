"""Repeated queue-boundary credits need both intra-iteration and reentry handbacks."""

import pytest

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from tests.models.structured_events import check_compiled_events


def conditional_loads(*, reuse):
    buffers, index = (1, "0") if reuse else (9, "panel")
    return N.from_source(
        f"""
def repeated(X:N.Tensor((n,m,9,64),N.f32),
             Out:N.Tensor((n,m,9,64),N.f32,access="write",write_disjoint=True),
             Meta:N.Tensor((n,m,9),N.i64),
             n:N.int64,m:N.int64):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub(({buffers},64),N.f32)
        with N.Vector():
            for outer in N.Pipelined(n):
                for inner in N.serial(m):
                    for panel in N.unroll(9):
                        N.copy(X[outer,inner,panel,:],x[{index},:])
                        condition=N.load_index(Meta[outer,inner,panel:panel+1],minimum=0,maximum=1)
                        if condition==0:
                            N.mul(x[{index},:],2.0,out=x[{index},:])
                        N.copy(x[{index},:],Out[outer,inner,panel,:])
""",
        disjoint_external=True,
    )


def test_repeated_queue_ports_share_only_after_consumption_and_nested_reentry(monkeypatch):
    from ascendflow.analysis import event_planning
    from tests.models.structured_events import _Interpret

    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Meta"
            values[operation.value] = (offset // 8) % 2

    monkeypatch.setattr(_Interpret, "native", metadata)

    establish = event_planning.share_events
    queried = set()

    def once(domains, rule, left, right):
        pair = frozenset((left, right))
        assert pair not in queried, "candidate IDs must reuse unchanged lifetime relations"
        queried.add(pair)
        return establish(domains, rule, left, right)

    monkeypatch.setattr(event_planning, "share_events", once)
    plan = synchronize_structure(conditional_loads(reuse=True))
    assert plan.resources.sharing
    assert queried
    for outer, inner in ((0, 0), (0, 3), (1, 0), (1, 1), (2, 2), (3, 4)):
        check_compiled_events(plan, {"n": outer, "m": inner}, invocations=2)


def test_source_order_alone_does_not_license_sharing_repeated_local_events():
    with pytest.raises(CompileError, match="capacity 8"):
        synchronize_structure(conditional_loads(reuse=False))


def disjoint_loop_episodes(*, scalar):
    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,),N.f32)" for i in range(9))
    episodes = []
    for panel in range(9):
        reader = (
            f"N.load_scalar(x{panel}[:1])"
            if scalar
            else f"N.copy(x{panel},Out[{panel},i{panel},:])"
        )
        prefix = "" if scalar else f"        N.fill(x{panel},1.0)\n"
        writer = f"                N.fill(x{panel},1.0)\n" if scalar else ""
        episodes.append(
            prefix + f"        for i{panel} in N.Pipelined(N.min(n,m+{panel % 2})):\n"
            f"            if i{panel}%3=={panel % 3}:\n" + writer + f"                {reader}"
        )
    return N.from_source(
        'def episodes(Out:N.Tensor((9,n,64),N.f32,access="write",write_disjoint=True),'
        "n:N.int64,m:N.int64):\n"
        '    with N.Kernel(1,kind="vector"):\n'
        + declarations
        + "\n        with N.Vector():\n"
        + "\n".join("    " + line for line in "\n".join(episodes).splitlines())
        + "\n",
        disjoint_external=True,
    )


def test_whole_loop_episodes_share_after_last_consumption_with_empty_and_sparse_uses():
    plan = synchronize_structure(disjoint_loop_episodes(scalar=True))
    assert plan.resources.sharing
    for n, m in ((0, 0), (0, 4), (1, 0), (2, 3), (7, 4)):
        check_compiled_events(plan, {"n": n, "m": m}, invocations=2)


def test_disjoint_loop_episodes_still_require_a_consumption_handback():
    with pytest.raises(CompileError, match="capacity 8"):
        synchronize_structure(disjoint_loop_episodes(scalar=False))


def test_two_required_handoffs_may_join_at_later_work_in_the_same_queue():
    definition = N.from_source(
        """
def patches(X:N.Tensor((n,16,64),N.f32),
            O:N.Tensor((n,16,64),N.f32,access="write",write_disjoint=True),
            Masks:N.Tensor((n,16),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((2,16,64),N.f32)
        masks=N.alloc_ub((2,16),N.f32)
        zero=N.alloc_ub((8,),N.f32)
        N.fill(zero,0.0)
        value=N.load_scalar(zero[:1])
        for i in N.Pipelined(n):
            with N.Stage(0,name="patch"):
                N.fill(x[i%2,:,:],0.0)
                N.fill(masks[i%2,:],-1.0)
                for row in N.serial(16):
                    if row<i%17:
                        N.copy(X[i,row,:],x[i%2,row,:])
                        N.store_scalar(value,out=masks[i%2,row:row+1])
            with N.Stage(1,name="publish"):
                N.copy(masks[i%2,:],Masks[i,:])
                N.copy(x[i%2,:,:],O[i,:,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(definition)
    for n in (0, 1, 2, 5, 19):
        check_compiled_events(plan, {"n": n}, invocations=2)


def test_nested_bursts_check_last_consumption_and_next_outer_invocation():
    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,),N.f32)" for i in range(9))
    bursts = "\n".join(
        f"            for child{i} in N.serial({i % 3 + 1}):\n"
        f"                N.fill(x{i},1.0)\n"
        f"                N.load_scalar(x{i}[:1])"
        for i in range(9)
    )
    source = (
        'def bursts(n:N.int64):\n    with N.Kernel(1,kind="vector"):\n'
        + declarations
        + "\n        for outer in N.Pipelined(n):\n"
        + bursts
        + "\n"
    )
    plan = synchronize_structure(N.from_source(source))
    for n in (0, 1, 2, 5):
        check_compiled_events(plan, {"n": n}, invocations=2)


@pytest.mark.parametrize("panels", (3, 9))
def test_contained_windows_reuse_only_under_pressure_and_keep_real_consumption(panels):
    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,),N.f32)" for i in range(panels))
    bursts = "\n".join(
        f"            if outer>={i}:\n"
        f"                for child{i} in N.serial({i % 3 + 1}):\n"
        f"                    N.fill(x{i},1.0)\n"
        f"                    N.load_scalar(x{i}[:1])"
        for i in range(panels)
    )
    source = (
        'def prefixes(n:N.int64):\n    with N.Kernel(1,kind="vector"):\n'
        + declarations
        + "\n        for outer in N.Pipelined(n):\n"
        + bursts
        + "\n"
    )
    plan = synchronize_structure(N.from_source(source))
    assert bool(plan.resources.sharing) == (panels == 9)
    for n in (0, 1, 2, 9, 12):
        check_compiled_events(plan, {"n": n}, invocations=2)


@pytest.mark.parametrize("scalar", (True, False))
def test_alternating_child_bursts_preserve_consumptions_across_empty_and_short_invocations(scalar):
    read_even = "N.load_scalar(a[:1])" if scalar else "N.copy(a,Out[i,j,:])"
    read_odd = "N.load_scalar(b[:1])" if scalar else "N.copy(b,Out[i,k,:])"
    definition = N.from_source(
        f"""
def phases(Out:N.Tensor((n,3,64),N.f32,access="write",write_disjoint=True),n:N.int64):
    with N.Kernel(1,kind="vector"):
        a=N.alloc_ub((64,),N.f32)
        b=N.alloc_ub((64,),N.f32)
        for i in N.Pipelined(n):
            if i%2==0:
                for j in N.serial(2):
                    N.fill(a,1.0)
                    {read_even}
            else:
                for k in N.serial(3):
                    N.fill(b,2.0)
                    {read_odd}
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(definition)
    for n in (0, 1, 2, 3, 6, 9):
        check_compiled_events(plan, {"n": n}, invocations=2)
