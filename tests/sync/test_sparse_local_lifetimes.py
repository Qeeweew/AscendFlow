"""Real metadata may skip any number of local binary-flag invocations."""

from itertools import product
from types import SimpleNamespace

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine, SyncKind
from tests.fixtures.nested import metadata_mixed_panels, metadata_panels
from tests.models.structured_events import EventModelError, _Interpret, check_compiled_events


@pytest.fixture(scope="module")
def plan():
    return synchronize_structure(metadata_panels)


def set_metadata(monkeypatch, patterns, *, lanes=1):
    native = _Interpret.native

    def metadata(self, operation, values, invocation, group, actor, coordinates):
        native(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "Meta" and offset % 8 == 0
            if lanes == 1:
                values[operation.value] = patterns[invocation][offset // 8]
            else:
                t, lane = divmod(offset // 8, lanes)
                values[operation.value] = patterns[invocation][lane][t]

    monkeypatch.setattr(_Interpret, "native", metadata)


def test_full_and_tail_paths_consume_flags_for_all_lengths_and_sparse_words(plan, monkeypatch):
    patterns = [()] + [(n,) for n in range(33)]
    patterns += list(product((0, 16, 32), repeat=3))
    patterns += [(1, 32, 32, 32, 17), (32, 0, 32, 15, 32), (15, 16, 17, 31, 32)]
    for pattern in patterns:
        with monkeypatch.context() as patch:
            set_metadata(patch, (pattern, tuple(reversed(pattern))))
            checked = check_compiled_events(plan, {"n": len(pattern)}, invocations=2)
            assert checked is not None


@pytest.mark.parametrize("route", ((Engine.MTE3, Engine.V), (Engine.V, Engine.MTE2)))
def test_missing_actual_return_or_acquire_breaks_sparse_reuse(plan, monkeypatch, route):
    module = plan.module.clone()
    removed = []
    for operation in list(module.walk()):
        if (
            isinstance(operation, NotificationOp)
            and operation.kind.data == SyncKind.LOCAL_WAIT.value
            and (operation.source.data, operation.destination.data) == tuple(e.value for e in route)
        ):
            removed.append(operation)
            operation.parent.erase_op(operation)
    assert removed
    set_metadata(monkeypatch, ((1, 32, 32, 17), (32, 1, 32, 15)))
    with pytest.raises(EventModelError):
        check_compiled_events(
            SimpleNamespace(module=module, before_ir=plan.before_ir), {"n": 4}, invocations=2
        )


def test_public_source_to_asc_keeps_a_constant_id_and_real_tail_control():
    result = compile(metadata_panels)
    streams = result.report["event_resources"]["streams"]
    sparse = [
        s for s in streams if any(c["policy"] == "ordered-local-handback" for c in s["cases"])
    ]
    assert sparse and all(s["width"] == 1 for s in sparse)
    assert all(c["phase"] == "0" for s in sparse for c in s["cases"])


def test_two_aiv_keep_private_consumptions_in_one_body_with_different_metadata(monkeypatch):
    result = compile(metadata_mixed_panels)
    assert result.asc.count("if ASC_IS_AIV") == 1
    assert "npu_sid ==" not in result.asc
    set_metadata(
        monkeypatch,
        (((0, 32, 32, 1), (16, 17, 0, 32)), ((32, 1, 32, 0), (0, 32, 15, 32))),
        lanes=2,
    )
    check_compiled_events(
        SimpleNamespace(module=result.plan, before_ir=result.before_ir), {"n": 4}, invocations=2
    )


@pytest.fixture(scope="module")
def cube_parent_plan():
    definition = N.from_source(
        """
def changing_children(X:N.Tensor((n,8,2,16,16),N.f16),
                      Meta:N.Tensor((n,),N.i64),
                      Y:N.Tensor((n,8,16,16),N.f32,access="write"), n:N.i64):
    with N.Kernel(1,kind="cube"):
        l1=N.alloc_l1((16,16),N.f16,slots=2)
        a=N.alloc_l0a((16,16),N.f16,slots=2)
        b=N.alloc_l0b((16,16),N.f16,slots=2)
        c=N.alloc_l0c((16,16),N.f32)
        for parent in N.serial(n):
            count=N.load_index(Meta[parent:parent+1],minimum=0,maximum=8)
            for tile in N.serial(count):
                for k in N.Pipelined(2,issue_order=("load","compute")):
                    with N.Stage(0,name="load"):
                        v=N.copy(X[parent,tile,k,:,:],l1.at(k))
                    with N.Stage(1,name="compute"):
                        aa=N.copy(v,a.at(k))
                        bb=N.copy(v,b.at(k),transpose=True)
                        N.matmul(aa,bb,accum=c,initialize=(k==0))
                        if k==1:
                            N.copy(c,Y[parent,tile,:,:])
""",
        disjoint_external=True,
    )
    return synchronize_structure(definition)


def test_parent_varying_cube_children_get_finite_ring_and_retire_on_reentry(
    cube_parent_plan, monkeypatch
):
    assert any(
        lease.width > 1 and any(c.policy == "ordered-local-handback" for c in lease.cases)
        for lease in cube_parent_plan.generation.leases
    )
    for pattern in ((0,), (1,), (8,), (0, 8, 0, 1), (1, 8, 2, 0, 5), (8, 1, 8, 0, 8)):
        with monkeypatch.context() as patch:
            set_metadata(patch, (pattern, tuple(reversed(pattern))))
            checked = check_compiled_events(cube_parent_plan, {"n": len(pattern)}, invocations=2)
            for invocation in (0, 1):
                writes = [
                    span
                    for command in checked.natives
                    if command.occurrence.invocation == invocation
                    for span in command.spans
                    if span.root.sym_name.data == "Y"
                ]
                assert sum(span.end - span.start for span in writes) == sum(pattern) * 16 * 16 * 4


def test_parent_varying_cube_does_not_replace_actual_handback_with_queue_order(
    cube_parent_plan, monkeypatch
):
    module = cube_parent_plan.module.clone()
    removed = []
    for operation in list(module.walk()):
        if (
            isinstance(operation, NotificationOp)
            and operation.kind.data == SyncKind.LOCAL_WAIT.value
            and (operation.source.data, operation.destination.data)
            == (Engine.FIX.value, Engine.M.value)
        ):
            removed.append(operation)
            operation.parent.erase_op(operation)
    assert removed
    set_metadata(monkeypatch, ((8, 0, 1, 8), (1, 8, 0, 8)))
    with pytest.raises(EventModelError):
        check_compiled_events(
            SimpleNamespace(module=module, before_ir=cube_parent_plan.before_ir),
            {"n": 4},
            invocations=2,
        )
