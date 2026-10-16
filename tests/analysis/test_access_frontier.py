"""Real native anchors for finite access contracts, without completion claims."""

from dataclasses import fields

import pytest
from xdsl.dialects import scf

from xdsl_npu import language as N
from xdsl_npu.analysis.access_frontier import (
    AccessCollector,
    AccessEndpoint,
    Cell,
    CellPort,
    CellTransfer,
    LoopResidueSelection,
    PortKind,
    ResiduePosition,
    compose_cell_sequence,
    compose_cells,
    native_cell,
)
from xdsl_npu.analysis.hazards import HazardKind
from xdsl_npu.analysis.native import AccessMode
from xdsl_npu.analysis.structured_memory import collect_native_memory
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.structured import PipelineLoopOp
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def memory(source):
    return collect_native_memory(lower_native_structure(N.from_source(source)).module)


def vector(*, rmw=False):
    return memory(f"""
        def accesses():
            with N.Kernel():
                x,y,z=N.alloc_ub((64,)),N.alloc_ub((64,)),N.alloc_ub((64,))
                N.fill(x,0.0)
                N.mul(x,2.0,out={"x" if rmw else "y"})
                N.mul(x,3.0,out=z)
                N.fill(x,1.0)
    """)


def cell_and_endpoints(facts):
    root = facts.templates[0].accesses[0].address.root
    cell = Cell(root, 0, 256, Actor.AIV0)
    return cell, tuple(AccessEndpoint.from_template(t, Actor.AIV0) for t in facts.templates)


def on_cell(cell, template, participant):
    modes = [a.mode for a in template.accesses if a.address.root is cell.root]
    if not modes:
        return CellTransfer.identity(cell)
    reads, writes = any(mode.reads for mode in modes), any(mode.writes for mode in modes)
    mode = (
        AccessMode.READ_WRITE
        if reads and writes
        else AccessMode.READ
        if reads
        else AccessMode.WRITE
    )
    return native_cell(cell, AccessEndpoint.from_template(template, participant), mode)


def edges(dependencies):
    return {(n.kind, n.source, n.consumer) for n in dependencies.needs}


def test_write_read_write_pairs_a_direct_carry_without_dropping_waw():
    cell, (writer, reader, _, overwrite) = cell_and_endpoints(vector())
    collector = AccessCollector()
    summary = compose_cell_sequence(
        cell,
        (
            native_cell(cell, writer, AccessMode.WRITE),
            native_cell(cell, reader, AccessMode.READ),
            native_cell(cell, overwrite, AccessMode.WRITE),
        ),
        collector,
    )
    result = collector.freeze()
    assert edges(result) == {
        (HazardKind.RAW, writer, reader),
        (HazardKind.WAR, reader, overwrite),
        (HazardKind.WAW, writer, overwrite),
    }
    (carry,) = result.read_carries
    assert carry.raw.source is carry.waw.source is writer
    assert carry.raw.consumer is carry.war.source is reader
    assert carry.war.consumer is carry.waw.consumer is overwrite
    assert all(
        n is next(x for x in result.needs if x is n) for n in (carry.raw, carry.war, carry.waw)
    )
    assert summary.first_write is writer and summary.last_write is overwrite
    assert not summary.head_reads and not summary.tail_reads
    assert summary.entry_needs[0].source == cell.writer_port
    assert summary.entry_needs[1].source == cell.readers_port


def test_scan_uses_logical_access_order_with_reverse_stage_issue_order():
    facts = memory("""
        def logical():
            with N.Kernel():
                x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
                for g in N.Pipelined(4,stages=3,
                    issue_order={"vector":("overwrite","read","initialize")}):
                    with N.Stage(0,name="initialize"):
                        N.fill(x,0.0)
                    with N.Stage(1,name="read"):
                        N.mul(x,2.0,out=y)
                    with N.Stage(2,name="overwrite"):
                        N.fill(x,1.0)
    """)
    cell, (writer, reader, overwrite) = cell_and_endpoints(facts)
    loop = writer.frame.operation.parent_op()
    assert isinstance(loop, PipelineLoopOp)
    assert [name.data for name in loop.issue_order.data["vector"]] == [
        "overwrite",
        "read",
        "initialize",
    ]
    assert [endpoint.frame.operation.label.data for endpoint in (writer, reader, overwrite)] == [
        "initialize",
        "read",
        "overwrite",
    ]
    collector = AccessCollector()
    compose_cell_sequence(cell, (on_cell(cell, t, Actor.AIV0) for t in facts.templates), collector)
    assert edges(collector.freeze()) == {
        (HazardKind.RAW, writer, reader),
        (HazardKind.WAR, reader, overwrite),
        (HazardKind.WAW, writer, overwrite),
    }


def test_no_write_forwards_both_ports_and_keeps_every_reader():
    cell, (_, first, second, _) = cell_and_endpoints(vector())
    collector = AccessCollector()
    summary = compose_cell_sequence(
        cell,
        (
            native_cell(cell, first, AccessMode.READ),
            native_cell(cell, second, AccessMode.READ),
        ),
        collector,
    )
    assert summary.forwards_writer and summary.forwards_readers
    assert summary.writer == cell.writer_port
    assert [r.reader for r in summary.head_reads] == [first, second]
    assert [r.reader for r in summary.tail_reads] == [first, second]
    assert len(summary.entry_needs) == 2
    assert all(
        n.kind is HazardKind.RAW and n.source == cell.writer_port for n in summary.entry_needs
    )
    assert collector.freeze().needs == ()


def test_prepend_writer_binds_only_head_reads_and_preserves_reader_passthrough():
    cell, (writer, first, second, _) = cell_and_endpoints(vector())
    collector = AccessCollector()
    child = compose_cell_sequence(
        cell,
        (
            native_cell(cell, first, AccessMode.READ),
            native_cell(cell, second, AccessMode.READ),
        ),
        collector,
    )
    combined = compose_cells(native_cell(cell, writer, AccessMode.WRITE), child, collector)
    assert not combined.head_reads and not combined.forwards_readers
    assert combined.last_write is writer
    assert [r.reader for r in combined.tail_reads] == [first, second]
    assert all(r.writer is writer for r in combined.tail_reads)
    assert edges(collector.freeze()) == {
        (HazardKind.RAW, writer, first),
        (HazardKind.RAW, writer, second),
    }


def test_child_head_read_carry_binds_to_incoming_writer_without_copying_internal_needs():
    cell, (writer, first, second, overwrite) = cell_and_endpoints(vector())
    collector = AccessCollector()
    left = compose_cells(
        native_cell(cell, writer, AccessMode.WRITE),
        native_cell(cell, first, AccessMode.READ),
        collector,
    )
    right = compose_cells(
        native_cell(cell, second, AccessMode.READ),
        native_cell(cell, overwrite, AccessMode.WRITE),
        collector,
    )
    (deferred,) = right.entry_carries
    assert isinstance(deferred.waw.source, CellPort)
    assert deferred.waw.source.kind is PortKind.WRITER
    war = deferred.war
    summary = compose_cells(left, right, collector)
    result = collector.freeze()
    assert collector.work.needs_appended == len(result.needs) == 5
    assert len(result.read_carries) == collector.work.carries_appended == 2
    assert sum(n is war for n in result.needs) == 1
    assert next(c for c in result.read_carries if c.raw.consumer is second).war is war
    assert not summary.entry_carries
    assert collector.freeze() is result
    with pytest.raises(CompileError, match="already been frozen"):
        compose_cells(summary, CellTransfer.identity(cell), collector)


def test_rmw_reads_old_writer_before_write_and_never_emits_same_instruction_war():
    facts = vector(rmw=True)
    cell, (writer, rmw, later_read, overwrite) = cell_and_endpoints(facts)
    collector = AccessCollector()
    summary = compose_cell_sequence(
        cell, (on_cell(cell, template, Actor.AIV0) for template in facts.templates), collector
    )
    result = collector.freeze()
    assert (HazardKind.RAW, writer, rmw) in edges(result)
    assert (HazardKind.WAW, writer, rmw) in edges(result)
    assert (HazardKind.RAW, rmw, later_read) in edges(result)
    assert (HazardKind.WAR, later_read, overwrite) in edges(result)
    assert not any(n.source == n.consumer for n in result.needs)
    rmw_carry = next(c for c in result.read_carries if c.waw.consumer == rmw)
    assert rmw_carry.war is None and rmw_carry.raw.consumer == rmw
    assert summary.last_write == overwrite


def test_rmw_at_entry_keeps_its_incoming_raw_and_direct_boundary_carry():
    cell, (_, rmw, _, _) = cell_and_endpoints(vector(rmw=True))
    transfer = native_cell(cell, rmw, AccessMode.READ_WRITE)
    assert len(transfer.head_reads) == 1 and not transfer.tail_reads
    assert [n.kind for n in transfer.entry_needs] == [
        HazardKind.RAW,
        HazardKind.WAW,
        HazardKind.WAR,
    ]
    (carry,) = transfer.entry_carries
    assert carry.war is None and carry.raw.source == carry.waw.source == cell.writer_port
    assert carry.raw.consumer is carry.waw.consumer is rmw


def test_same_vector_task_has_two_distinct_gm_reader_participants():
    facts = memory("""
        def shared(Scratch:N.Tensor((16,16),N.f32,access="read_write")):
            with N.Kernel(kind="mixed"):
                c=N.alloc_l0c((16,16),N.f32)
                x=N.alloc_ub((16,16),N.f32)
                with N.Cube():
                    N.copy(c,Scratch)
                with N.Vector():
                    N.copy(Scratch,x)
                with N.Cube():
                    N.copy(c,Scratch)
    """)
    initial, shared, final = facts.templates
    root = initial.accesses[1].address.root
    cell = Cell(root, 0, 1024)
    writer, overwrite = (AccessEndpoint.from_template(t, Actor.AIC) for t in (initial, final))
    first, second = (AccessEndpoint.from_template(shared, p) for p in (Actor.AIV0, Actor.AIV1))
    assert first.operation is second.operation and first != second
    collector = AccessCollector()
    compose_cell_sequence(
        cell,
        (
            native_cell(cell, writer, AccessMode.WRITE),
            native_cell(cell, first, AccessMode.READ),
            native_cell(cell, second, AccessMode.READ),
            native_cell(cell, overwrite, AccessMode.WRITE),
        ),
        collector,
    )
    result = collector.freeze()
    releases = [n for n in result.needs if n.kind is HazardKind.WAR]
    assert {n.source.participant for n in releases} == {Actor.AIV0, Actor.AIV1}
    assert all(n.source.operation is shared.operation for n in releases)
    assert len(result.read_carries) == 2
    assert shared.participants == (Actor.AIV0, Actor.AIV1)


def test_empty_transfer_is_transparent_and_emits_no_need():
    cell, (_, reader, _, _) = cell_and_endpoints(vector())
    transfer = native_cell(cell, reader, AccessMode.READ)
    collector = AccessCollector()
    left = compose_cells(CellTransfer.identity(cell), transfer, collector)
    right = compose_cells(left, CellTransfer.identity(cell), collector)
    assert left == right == transfer
    assert not collector.freeze().needs
    assert CellTransfer.identity(cell).entry_needs == ()


@pytest.mark.parametrize("mode", [AccessMode.READ, AccessMode.WRITE, AccessMode.READ_WRITE])
def test_repeated_unbound_endpoint_cannot_silently_become_same_instruction_rmw(mode):
    cell, (endpoint, _, _, _) = cell_and_endpoints(vector())
    collector = AccessCollector()
    with pytest.raises(CompileError, match="explicit occurrence binding"):
        compose_cells(
            native_cell(cell, endpoint, mode),
            native_cell(cell, endpoint, mode),
            collector,
        )
    assert not collector.freeze().needs


def test_repeated_endpoint_inside_child_is_rejected_and_child_cannot_be_reused():
    cell, (writer, first, second, overwrite) = cell_and_endpoints(vector())
    collector = AccessCollector()
    child = compose_cell_sequence(
        cell,
        (
            native_cell(cell, writer, AccessMode.WRITE),
            native_cell(cell, first, AccessMode.READ),
            native_cell(cell, overwrite, AccessMode.WRITE),
        ),
        collector,
    )
    assert not child.head_reads and not child.tail_reads
    with pytest.raises(CompileError, match="explicit occurrence binding"):
        compose_cells(child, native_cell(cell, first, AccessMode.READ), collector)
    parent = compose_cells(child, native_cell(cell, second, AccessMode.READ), collector)
    with pytest.raises(CompileError, match="already been spliced"):
        compose_cells(parent, child, collector)


def test_local_cell_cannot_be_shared_by_other_participant_or_different_cell_owner():
    cell, (writer, reader, _, _) = cell_and_endpoints(vector())
    other = Cell(cell.root, 0, 256, Actor.AIV1)
    with pytest.raises(CompileError, match="another participant's private cell"):
        native_cell(other, reader, AccessMode.READ)
    with pytest.raises(CompileError, match="distinct physical cells"):
        compose_cells(
            native_cell(cell, writer, AccessMode.WRITE),
            CellTransfer.identity(other),
            AccessCollector(),
        )
    collector = AccessCollector()
    owned = compose_cells(
        native_cell(cell, writer, AccessMode.WRITE),
        native_cell(cell, reader, AccessMode.READ),
        collector,
    )
    with pytest.raises(CompileError, match="different internal-obligation owners"):
        compose_cells(owned, CellTransfer.identity(cell), AccessCollector())


def test_l1_release_uses_actual_operand_reader_before_unrelated_mad_and_fix():
    facts = memory("""
        def early(A:N.Tensor((16,16),N.f16),B:N.Tensor((16,16),N.f16),
                  Y:N.Tensor((16,16),N.f32,access="write")):
            with N.Kernel(kind="cube"):
                a,b=N.alloc_l1((16,16),N.f16),N.alloc_l1((16,16),N.f16)
                aa,bb=N.alloc_l0a((16,16),N.f16),N.alloc_l0b((16,16),N.f16)
                c=N.alloc_l0c((16,16),N.f32)
                N.copy(A,a)
                N.copy(B,b)
                N.copy(a,aa)
                N.copy(b,bb)
                N.matmul(aa,bb,accum=c,initialize=True)
                N.copy(c,Y)
                N.copy(A,a)
    """)
    root = facts.templates[0].accesses[1].address.root
    cell = Cell(root, 0, 512, Actor.AIC)
    reader = next(t for t in facts.templates if t.contract.native == "load2d_a")
    assert any(t.contract.engine is Engine.M for t in facts.templates)
    assert any(t.contract.engine is Engine.FIX for t in facts.templates)
    collector = AccessCollector()
    compose_cell_sequence(cell, (on_cell(cell, t, Actor.AIC) for t in facts.templates), collector)
    (release,) = [n for n in collector.freeze().needs if n.kind is HazardKind.WAR]
    assert release.source.operation is reader.operation
    assert release.source_cut.milestone is Milestone.NATIVE_END
    assert release.consumer_cut.milestone is Milestone.ADMISSION
    assert release.consumer.operation is facts.templates[-1].operation


@pytest.mark.parametrize("serial", [False, True])
def test_actual_loop_selection_guards_empty_and_short_before_last_ordinal(serial):
    loop_source = "N.serial(3,10,2)" if serial else "N.Pipelined(7,stages=1)"
    plan = lower_native_structure(
        N.from_source(f"""
        def select():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for g in {loop_source}:
                    N.fill(x,0.0)
    """)
    )
    loop = next(op for op in plan.module.walk() if isinstance(op, (PipelineLoopOp, scf.ForOp)))
    for residue in range(3):
        first = LoopResidueSelection(loop, 3, residue, ResiduePosition.FIRST)
        last = LoopResidueSelection(loop, 3, residue, ResiduePosition.LAST)
        for count in range(10):
            actual = list(range(residue, count, 3))
            assert first.evaluate(count) == (actual[0] if actual else None)
            assert last.evaluate(count) == (actual[-1] if actual else None)
        assert first.loop is last.loop is loop
        assert first.evaluate(0) is last.evaluate(0) is None
        for count in (10**6, 10**12):
            selected = last.evaluate(count)
            assert 0 <= selected < count and selected % 3 == residue
            assert selected + 3 >= count
        assert len(fields(last)) == 4


def test_residue_selection_preserves_loop_identity_and_rejects_bad_domains():
    plan = lower_native_structure(
        N.from_source("""
        def loops():
            with N.Kernel():
                for a in N.Pipelined(2,stages=1):
                    pass
                for b in N.Pipelined(2,stages=1):
                    pass
    """)
    )
    first, second = tuple(plan.loops.values())
    a = LoopResidueSelection(first, 2, 0, ResiduePosition.LAST)
    b = LoopResidueSelection(second, 2, 0, ResiduePosition.LAST)
    assert a != b and a.evaluate(10**12) == b.evaluate(10**12)
    for period, residue in ((0, 0), (2, 2), (2, -1)):
        with pytest.raises(CompileError, match="actual loop, period/residue"):
            LoopResidueSelection(first, period, residue, ResiduePosition.LAST)
    with pytest.raises(CompileError, match="nonnegative"):
        a.evaluate(-1)


def test_cells_and_endpoint_identities_do_not_depend_on_diagnostic_names():
    cell, (writer, reader, _, _) = cell_and_endpoints(vector())
    writer.operation.attributes["uid"] = reader.operation.uid
    assert writer != reader
    assert cell.root.owner.sym_name.data
    with pytest.raises(CompileError, match="actual allocation root SSA"):
        Cell(cell.root.owner.sym_name.data, 0, 256, Actor.AIV0)
