"""Actual parent reentry and two-participant CV boundary contracts."""

from dataclasses import replace

import pytest

from tests.fixtures.nested import nested_cv
from tests.support.structured import prepared, selected
from xdsl_npu.analysis.event_progress import prove_progress
from xdsl_npu.analysis.structured_protocols import NotificationKind
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.target.ascend910b import Actor

PARENT_REUSE = """
def parent_reuse(X:N.Tensor((n,2,64),N.f32),n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((2,64))
        for p in N.Pipelined(n,stages=1):
            for j in N.Pipelined(2,stages=2):
                with N.Stage(0):
                    v=N.copy(X[p,j,:],x[j,:])
                with N.Stage(1):
                    N.mul(v,2.0,out=x[j,:])
"""


def test_reused_child_storage_returns_to_the_next_parent_invocation():
    domains = prepared(selected(PARENT_REUSE))
    proof = prove_progress(domains)
    parent, child = proof.region.children[0], proof.region.children[0].children[0]
    assert parent.stream is not None and child.stream is not None
    assert len(domains.memory.templates) == 2
    assert len(parent.stream.ports) == len(child.stream.ports) == 2
    assert len(child.ports) == 2
    release = parent.stream.ports[0].notification
    assert all(port.notification is release for port in child.ports)
    assert all(port.notification is not release for port in child.stream.ports)
    # The queue reentry is a positive parent-clock edge. Folding the boundary
    # into child-local zero-step progress would create a false cycle.
    assert any(edge.steps == 1 for edge in parent.stream.relations)
    for region in (parent, child):
        stream = region.stream
        for edge in stream.relations:
            assert (
                stream.rank[edge.source] + 1
                <= stream.stride * edge.steps + stream.rank[edge.destination]
            )
    # The original endpoints still name the previous and current invocation;
    # a proof binding must not rewrite the materialization input.
    ordinal = domains.control.loops[parent.operation].ordinal
    source = dict(release.sources[0].coordinates)[parent.operation]
    consumer = dict(release.consumers[0].coordinates)[parent.operation]
    for invocation in (1, 2, 5):
        context = {ordinal.args[0]: invocation}
        assert consumer.evaluate(context) - source.evaluate(context) == 1


@pytest.fixture(scope="module")
def cv_selection():
    return selected(nested_cv)


def test_cv_domains_keep_both_private_vector_participants(cv_selection):
    domains = prepared(cv_selection)
    cross = [n for n in domains.notifications if n.mechanism is NotificationKind.CROSS_EVENT]
    assert cross
    assert any({p.actor for p in n.sources} == {Actor.AIV0, Actor.AIV1} for n in cross)
    assert any({p.actor for p in n.consumers} == {Actor.AIV0, Actor.AIV1} for n in cross)
    for notification in cross:
        vector = notification.sources if len(notification.sources) == 2 else notification.consumers
        # One native operation in the shared body, two physical participants.
        assert vector[0].operation is vector[1].operation


@pytest.mark.parametrize("missing", (Actor.AIV0, Actor.AIV1))
def test_cv_release_rejects_a_missing_actual_contribution(cv_selection, missing):
    original = next(
        n
        for n in cv_selection.notifications
        if n.mechanism is NotificationKind.CROSS_EVENT and len(n.sources) == 2
    )
    changed = replace(
        original, sources=tuple(p for p in original.sources if p.actor is not missing)
    )
    broken = replace(
        cv_selection,
        notifications=tuple(changed if n is original else n for n in cv_selection.notifications),
        coverage=tuple(
            replace(item, notification=changed) if item.notification is original else item
            for item in cv_selection.coverage
        ),
    )
    with pytest.raises(CompileError, match="complete 1 AIC : 2 AIV quorum"):
        prepared(broken)
