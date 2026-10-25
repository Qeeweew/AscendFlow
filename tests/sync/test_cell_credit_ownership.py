"""A release credit keeps its physical owners at nested-loop boundaries."""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine
from tests.models.structured_events import check_compiled_events
from tests.support.credit_primitives import LOCAL_SOURCE
from tests.support.structured import selected


def test_selected_cell_anchors_are_original_bindings_and_reject_detached_copies():
    selection = selected(N.from_source(LOCAL_SOURCE, disjoint_external=True))
    originals = {id(b): b for b in selection.bindings}
    assert selection.notifications
    assert all(
        originals.get(id(anchor)) is anchor
        for notification in selection.notifications
        for anchor in notification.anchors
    )
    original = next(n for n in selection.notifications if n.anchors)
    changed = replace(original, anchors=tuple(replace(b) for b in original.anchors))
    detached = replace(
        selection,
        notifications=tuple(changed if n is original else n for n in selection.notifications),
        coverage=tuple(
            replace(c, notification=changed) if c.notification is original else c
            for c in selection.coverage
        ),
    )
    with pytest.raises(CompileError, match="anchor lost its original owned requirement binding"):
        detached.require_closed()


def test_eight_release_cells_reuse_their_inner_credit_at_outer_reentry():
    source = N.from_source(
        """
def parts(X:N.Tensor((n,8,4,64),N.f32),
          Out:N.Tensor((n,8,4,64),N.f32,access="write",write_disjoint=True),
          n:N.i64):
    with N.Kernel(1,kind="vector"):
        scratch=N.alloc_ub((2,4,64),N.f32)
        with N.Vector():
            for outer in N.serial(n):
                for inner in N.Pipelined(8):
                    with N.Stage(0,name="load"):
                        for part in N.unroll(4):
                            N.copy(X[outer,inner,part,:],scratch[inner%2,part,:])
                    with N.Stage(1,name="compute"):
                        for part in N.unroll(4):
                            N.mul(scratch[inner%2,part,:],2.0,out=scratch[inner%2,part,:])
                    with N.Stage(2,name="store"):
                        for part in N.unroll(4):
                            N.copy(scratch[inner%2,part,:],Out[outer,inner,part,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    releases = [
        channel
        for channel in plan.generation.channels
        if channel.members[0].mechanism is NotificationKind.LOCAL_EVENT
        and channel.members[0].source_engine is Engine.MTE3
        and channel.members[0].destination_engine is Engine.MTE2
    ]
    assert len(releases) == 8
    assert any(len(channel.members) > 1 for channel in releases)
    for channel in releases:
        owner_sets = [
            {b.requirement.cell.physical_key for b in member.anchors} for member in channel.members
        ]
        assert all(owners == owner_sets[0] for owners in owner_sets)
    for outer in (0, 1, 2, 3):
        check_compiled_events(plan, {"n": outer}, invocations=2)
