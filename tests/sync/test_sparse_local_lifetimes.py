"""Real metadata may skip any number of local binary-flag invocations."""

from itertools import product
from types import SimpleNamespace

import pytest

from ascendflow import compile
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
