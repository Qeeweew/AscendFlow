"""Actual endpoint cycles that must survive nested progress projection."""

from dataclasses import replace

import pytest

from tests.models.structured_domains import actual_dispatch
from tests.support.structured import selected
from xdsl_npu.analysis.access_frontier import AccessEndpoint
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.event_progress import prove_progress
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_regions import AccessOccurrence, LoopBinding
from xdsl_npu.analysis.structured_schedule import ActionGuard, Placement
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.scalar import literal
from xdsl_npu.target.ascend910b import Actor


def nested_channel_cycle():
    selection = selected("""
    def anchors(X:N.Tensor((16,16),N.f16),O:N.Tensor((16,16),N.f32,access="write"),n:N.int64):
        with N.Kernel(kind="cube"):
            x=N.alloc_l1((16,16))
            a=N.alloc_l0a((16,16))
            b=N.alloc_l0b((16,16))
            c=N.alloc_l0c((16,16))
            N.copy(X[:,:],x)
            for i in N.Pipelined(n,stages=1):
                N.copy(X[:,:],x)
                N.copy(x,a)
                N.copy(x,b,transpose=True)
                for j in N.Pipelined(n,stages=1):
                    N.copy(x,a)
                    N.matmul(a,b,accum=c,initialize=True)
                N.matmul(a,b,accum=c,initialize=True)
                N.copy(c,O[:,:])
            N.copy(c,O[:,:])
    """)
    memory = selection.dependencies.memory

    def port(index, source):
        template = memory.templates[index]
        coordinates = tuple(
            (loop.operation, loop.ordinal)
            for loop in memory.control.frames[template.operation].loops()
        )
        return BoundPlacement(
            AccessOccurrence(
                AccessEndpoint.from_template(template, Actor.AIC),
                tuple(LoopBinding(*coordinate) for coordinate in coordinates),
            ),
            Placement.AFTER if source else Placement.BEFORE,
            ActionGuard(()),
            (),
            (),
            (),
            coordinates,
        )

    def notification(name, source, destination):
        return LogicalNotification(
            name,
            (port(source, True),),
            (port(destination, False),),
            memory.templates[source].contract.engine,
            memory.templates[destination].contract.engine,
            NotificationKind.LOCAL_EVENT,
            (),
            None,
            literal(0),
            1,
            None,
        )

    # The outer stream changes MTE2 to MTE1 and M to FIX. Only its real
    # child provides MTE1 to M, completing the MTE2 to FIX dependency.
    notifications = (
        notification("root", 8, 0),
        notification("outer_left", 1, 2),
        notification("inner", 4, 5),
        notification("outer_right", 6, 7),
    )
    return replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=notifications,
        coverage=(),
    )


def test_exported_channel_composes_paths_through_a_real_grandchild():
    selection = nested_channel_cycle()
    trace = tuple(actual_dispatch(selection, {"n": 1, "$group": 0, "$sid": 0}, Actor.AIC))
    positions = {key[0]: index for index, (key, _) in enumerate(trace)}
    assert len(positions) == len(trace) == 9
    notifications = selection.notifications
    for left, right in zip(notifications, notifications[1:] + notifications[:1]):
        wait, source = left.consumers[0], right.sources[0]
        # Matching Set -> Wait arcs plus these actual FIFO Wait -> Set arcs
        # form one cycle. This check uses no production progress relation.
        assert wait.actor is source.actor is Actor.AIC
        assert left.destination_engine is right.source_engine
        assert positions[wait.operation] < positions[source.operation]
    with pytest.raises(CompileError):
        prove_progress(EventDomains(selection, ResourceWork()))
