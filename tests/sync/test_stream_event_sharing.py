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
    from ascendflow.analysis import structured_resources
    from tests.models.structured_events import _Interpret

    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Meta"
            values[operation.value] = (offset // 8) % 2

    monkeypatch.setattr(_Interpret, "native", metadata)

    establish = structured_resources.share_events
    queried = set()

    def once(domains, rule, left, right):
        pair = frozenset((left, right))
        assert pair not in queried, "candidate IDs must reuse unchanged lifetime relations"
        queried.add(pair)
        return establish(domains, rule, left, right)

    monkeypatch.setattr(structured_resources, "share_events", once)
    plan = synchronize_structure(conditional_loads(reuse=True))
    assert plan.resources.sharing
    assert queried
    for outer, inner in ((0, 0), (0, 3), (1, 0), (1, 1), (2, 2), (3, 4)):
        check_compiled_events(plan, {"n": outer, "m": inner}, invocations=2)


def test_source_order_alone_does_not_license_sharing_repeated_local_events():
    with pytest.raises(CompileError, match="capacity 8"):
        synchronize_structure(conditional_loads(reuse=False))
