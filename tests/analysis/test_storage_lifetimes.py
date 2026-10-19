"""Dispatch lifetime order preserves source values and actual reader retirement."""

import pytest

from ascendflow import language as N
from ascendflow.analysis.hazards import HazardKind
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Engine, SyncKind
from tests.fixtures.nested import shared_l0c_qk_pv
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
)
from tests.support.structured import selected


def scratch(order=("b_fill", "b_read", "a_fill", "a_read")):
    return N.from_source(
        f"""
        def scratch(O:N.Tensor((n,2,64),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                s = N.alloc_ub((64,))
                for g in N.Pipelined(n, stages=1, issue_order={{"vector":{order!r}}}):
                    with N.Stage(0,name="a_fill"):
                        N.fill(s,1.0)
                    with N.Stage(0,name="a_read"):
                        N.copy(s,O[g,0,:])
                    with N.Stage(0,name="b_fill"):
                        N.fill(s,3.0)
                    with N.Stage(0,name="b_read"):
                        N.copy(s,O[g,1,:])
    """,
        disjoint_external=True,
    )


def test_complete_private_lifetimes_follow_dispatch_without_rebinding_reads():
    result = synchronize_structure(scratch())
    assert len(result.dependencies.order.packet_partitions) == 1
    a, ar, b, br = (t.operation for t in result.admission.memory.templates)
    raw = [n for n in result.dependencies.needs if n.kind is HazardKind.RAW]
    assert {(n.source.operation, n.consumer.operation) for n in raw} == {(a, ar), (b, br)}
    for n in range(5):
        check_compiled_events(result, {"n": n})


def test_interleaved_copy_reads_keep_their_values_through_actual_engine_events():
    result = synchronize_structure(scratch(("a_fill", "b_fill", "a_read", "b_read")))
    for n in range(5):
        check_compiled_events(result, {"n": n})


def test_shifted_private_packets_keep_values_through_fill_and_drain():
    result = synchronize_structure(
        N.from_source(
            """
        def shifted(O:N.Tensor((n,2,64),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                s=N.alloc_ub((64,))
                for g in N.Pipelined(n,stages=2,issue_order={"vector":("a","b")}):
                    with N.Stage(0,name="a"):
                        N.fill(s,1.0)
                        N.copy(s,O[g,0,:])
                    with N.Stage(1,name="b"):
                        N.fill(s,3.0)
                        N.copy(s,O[g,1,:])
    """,
            disjoint_external=True,
        )
    )
    assert len(result.dependencies.order.packet_partitions) == 1
    a, ar, b, br = (t.operation for t in result.executable.natives)
    for n in range(5):
        execution = check_compiled_events(result, {"n": n})
        expected = []
        for clock in range(n + 1):
            if clock < n:
                expected.extend(((a, clock), (ar, clock)))
            if 0 <= clock - 1 < n:
                expected.extend(((b, clock - 1), (br, clock - 1)))
        assert [
            (command.occurrence.operation, command.occurrence.coordinates[-1][1])
            for command in execution.natives
        ] == expected


def test_same_vector_overwrite_before_the_old_value_read_is_rejected():
    source = N.from_source(
        """
        def clobber(O:N.Tensor((n,2,64),N.f32,access="write"), n:N.int64):
            with N.Kernel():
                s,x,y=N.alloc_ub((64,)),N.alloc_ub((64,)),N.alloc_ub((64,))
                for g in N.Pipelined(n,stages=1,
                    issue_order={"vector":("a_fill","b_fill","a_read","b_read")}):
                    with N.Stage(0,name="a_fill"):
                        N.fill(s,1.0)
                    with N.Stage(0,name="a_read"):
                        N.mul(s,2.0,out=x)
                        N.copy(x,O[g,0,:])
                    with N.Stage(0,name="b_fill"):
                        N.fill(s,3.0)
                    with N.Stage(0,name="b_read"):
                        N.mul(s,4.0,out=y)
                        N.copy(y,O[g,1,:])
    """,
        disjoint_external=True,
    )
    with pytest.raises(CompileError, match="same-engine source"):
        synchronize_structure(source)


def test_deleting_actual_store_release_leaves_an_unordered_war():
    result = synchronize_structure(
        N.from_source(
            """
        def release(O:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                s=N.alloc_ub((64,))
                N.fill(s,1.0)
                N.copy(s,O)
                N.mul(s,2.0,out=s)
    """,
            disjoint_external=True,
        )
    )
    baseline = check_compiled_events(result)
    _, store, overwrite = (t.operation for t in result.executable.natives)
    assert any(
        hazard.kind == "WAR"
        and hazard.source.operation is store
        and hazard.consumer.operation is overwrite
        for hazard in baseline.hazards
    )
    release = [
        operation
        for operation in result.module.walk()
        if isinstance(operation, NotificationOp)
        and operation.source.data == Engine.MTE3.value
        and operation.destination.data == Engine.V.value
    ]
    assert sorted(operation.kind.data for operation in release) == sorted(
        (SyncKind.LOCAL_SET.value, SyncKind.LOCAL_WAIT.value)
    )
    # Remove the real matched pair so event imbalance cannot mask the missing
    # reader retirement. The existing Vector RAW fence still orders both writes.
    for operation in release:
        operation.parent.erase_op(operation)
    result.module.verify()
    with pytest.raises(EventModelError, match="unordered physical WAR"):
        check_compiled_events(result)


def test_shared_l0c_uses_dispatch_neighbors_and_keeps_accumulation_raw():
    result = selected(shared_l0c_qk_pv)
    dependencies = result.dependencies
    assert len(dependencies.order.packet_partitions) == 1
    partition = dependencies.order.packet_partitions[0]
    qk, pv = partition.packets
    qk_write, pv_read = (
        qk.summary.first_writes[0].operation,
        pv.summary.tail_reads[0].operation,
    )
    shared = [n for n in dependencies.needs if n.cell == partition.cell]
    assert any(
        n.kind is HazardKind.WAR
        and n.source.operation is pv_read
        and n.consumer.operation is qk_write
        and n.distance == 2
        for n in shared
    )
    assert not any(
        n.kind is HazardKind.WAR
        and n.source.operation is pv_read
        and n.consumer.operation is qk_write
        and n.distance == 1
        for n in shared
    )
    assert any(
        n.kind is HazardKind.RAW and n.source.operation is n.consumer.operation and n.distance == 1
        for n in shared
    )
