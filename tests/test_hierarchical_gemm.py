"""Finite native L1/L0/Cube/FIX lifetime oracles, independent of public DSL syntax."""

import pytest
from xdsl.dialects.builtin import IntAttr

from tests.fixtures.physical_operators import gemm_nt
from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.check import bounded_check
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import ChannelOp, SyncOp, TaskOp, ViewOp
from xdsl_npu.model import extract


@pytest.mark.parametrize("ratio,slots", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)])
def test_l1_has_its_own_clock_and_all_l0_readers(ratio, slots):
    source = gemm_nt(
        32,
        64,
        ratio * 16 * 4,
        tile_m=16,
        tile_n=32,
        l1_k=ratio * 16,
        l0_k=16,
        l1_slots=slots,
    )
    p = extract(source)
    assert len([i for i in p.instances() if p.tasks[i.task].uid == "load_a"]) == 4
    assert len([i for i in p.instances() if p.tasks[i.task].uid == "operand_a"]) == 4 * ratio
    operand = next(t for t in p.tasks if t.uid == "operand_a").refs[0]
    width = 16 * 16 * 2
    for g in range(p.count):
        assert operand.address(g) == ((g // ratio) % slots) * width * ratio + (g % ratio) * width
    plan = autosync(source)
    verify_plan(parse(dump(plan)))
    assert extract(plan) == p
    assert emit_asc(plan).count("asc_sync_pipe(PIPE_ALL)") == 1


def test_gemm_two_dimensional_grid_writes_disjoint_column_strips():
    p = extract(gemm_nt(64, 128, 192, tile_m=32, tile_n=64, l1_k=64, l0_k=32))
    output = next(t for t in p.tasks if t.uid == "export").refs[-1]
    assert [output.address(5, core) for core in range(4)] == [0, 256, 16384, 16640]
    assert output.segments(5, 1)[0] == (256, 512)
    assert output.segments(5, 1)[1] == (768, 1024)


def test_gemm_releases_l1_after_the_last_slice_before_whole_mad():
    plan = autosync(gemm_nt(16, 16, 96, tile_m=16, tile_n=16, l1_k=32, l0_k=16))
    releases = [
        c
        for c in plan.walk()
        if isinstance(c, ChannelOp) and c.destination.data in {"load_a", "load_bt"}
    ]
    assert {c.source.data for c in releases} == {"operand_a", "operand_b"}
    assert all(c.lower.data == 1 and c.distance.data == 3 for c in releases)
    assert bounded_check(plan, max_states=500_000).exhaustive


def test_sparse_event_bank_rotates_notification_ordinals():
    plan = autosync(gemm_nt(16, 16, 160, tile_m=16, tile_n=16, l1_k=32, l0_k=16))
    ready = next(c for c in plan.walk() if isinstance(c, ChannelOp) and c.source.data == "load_a")
    assert ready.stride.data == 2 and ready.width.data == 2
    assert "/ 2" in emit_asc(plan)
    ready.attributes["width"] = IntAttr(1)
    with pytest.raises(CompileError, match="re-set before old Wait"):
        verify_plan(plan)


def test_deleting_last_l1_reader_release_is_not_repaired_by_exit_drain():
    plan = autosync(gemm_nt(16, 16, 96, tile_m=16, tile_n=16, l1_k=32, l0_k=16))
    channel = next(
        c for c in plan.walk() if isinstance(c, ChannelOp) and c.destination.data == "load_a"
    )
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.channel.data == channel.sym_name.data:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


@pytest.mark.parametrize("field,value", [("divisor", 0), ("core_divisor", 0), ("core_modulus", -1)])
def test_coordinate_contract_rejects_invalid_index_arithmetic(field, value):
    source = gemm_nt(16, 16, 32, tile_m=16, tile_n=16, l1_k=32, l0_k=16)
    view = next(v for v in source.walk() if isinstance(v, ViewOp))
    view.attributes[field] = IntAttr(value)
    with pytest.raises(CompileError):
        extract(source)


def test_periodic_task_cannot_read_a_cache_before_its_first_writer():
    source = gemm_nt(16, 16, 64, tile_m=16, tile_n=16, l1_k=32, l0_k=16)
    load = next(t for t in source.walk() if isinstance(t, TaskOp) and t.uid.data == "load_a")
    load.attributes["phase"] = IntAttr(1)
    with pytest.raises(CompileError, match="undefined l1a"):
        autosync(source)


def test_core_dependent_internal_alias_does_not_inherit_core_zero_proof():
    k = Kernel("core_dependent_alias", blocks=2)
    gm = k.buffer("gm", "GM", 256, external=True, initialized=True)
    ub = k.buffer("ub", "UB", 64)
    k.loop(3)
    a = k.view(ub, 32, stride=32, slots=2)
    # Core0 R[0:32], W[64:96]; Core1 both R/W[96:128]. Cross-core writes
    # are disjoint, yet only core1 needs Store(g)->Load(g+1) synchronization.
    k.copy("load", k.view(gm, 32, core_stride=96), a)
    k.scale("compute", a, a, 2, stage=1)
    k.copy("store", a, k.view(gm, 32, offset=64, core_stride=32), stage=2)
    with pytest.raises(CompileError, match="core-dependent internal aliases"):
        autosync(k.finish())


@pytest.mark.parametrize("cache_b", [False, True])
@pytest.mark.parametrize("c_slots", [1, 2])
def test_persistent_rows_initialize_every_output_and_export_exactly_once(cache_b, c_slots):
    source = gemm_nt(
        48,
        32,
        32,
        tile_m=16,
        tile_n=16,
        l1_k=32,
        l0_k=16,
        grid_m=1,
        c_slots=c_slots,
        cache_b=cache_b,
    )
    p = extract(source)
    mad = next(t for t in p.tasks if t.uid == "mad_step")
    assert [mad.modes_for(g)[-1] for g in range(p.count)] == ["W", "RW"] * 3
    exports = [i.generation for i in p.instances() if p.tasks[i.task].uid == "export"]
    assert exports == [1, 3, 5]
    loads = [i for i in p.instances() if p.tasks[i.task].uid == "load_bt"]
    assert len(loads) == (1 if cache_b else 3)
    plan = autosync(source)
    verify_plan(parse(dump(plan)))
    assert bounded_check(plan, max_states=500_000).exhaustive


def test_l0c_read_retirement_from_fixpipe_protects_next_output_tile():
    plan = autosync(
        gemm_nt(
            48, 16, 32, tile_m=16, tile_n=16, l1_k=32, l0_k=16, grid_m=1, c_slots=2, cache_b=True
        )
    )
    release = next(
        c
        for c in plan.walk()
        if isinstance(c, ChannelOp)
        and c.source.data == "export"
        and c.destination.data == "mad_step"
    )
    assert release.distance.data == 3  # Export g1 -> first MAD of third output, g4.
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.channel.data == release.sym_name.data:
            op.parent.erase_op(op)
    release.parent.erase_op(release)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_missing_periodic_clear_cannot_read_the_next_uninitialized_l0c_slot():
    source = gemm_nt(32, 16, 32, tile_m=16, tile_n=16, l1_k=32, l0_k=16, grid_m=1, c_slots=2)
    from xdsl.dialects.builtin import ArrayAttr

    mad = next(t for t in source.walk() if isinstance(t, TaskOp) and t.uid.data == "mad_step")
    mad.attributes["params"] = ArrayAttr(IntAttr(i) for i in (16, 16, 16, 1000))
    with pytest.raises(CompileError, match="undefined accumulator"):
        autosync(source)


def test_requested_b_cache_must_fit_alongside_a_ring():
    with pytest.raises(CompileError, match="L1 capacity exceeded"):
        autosync(gemm_nt(128, 256, 1024, tile_m=128, tile_n=256, l1_k=256, l0_k=64, cache_b=True))


def test_short_cube_accumulation_needs_a_real_m_completion_fence():
    plan = autosync(gemm_nt(16, 16, 64, tile_m=16, tile_n=16, l1_k=32, l0_k=16))
    barriers = [
        s
        for s in plan.walk()
        if isinstance(s, SyncOp) and s.action.data == "barrier" and s.pipe.data == "M"
    ]
    assert len(barriers) == 1 and (barriers[0].lower.data, barriers[0].upper.data) == (1, 4)
    for s in barriers:
        s.parent.erase_op(s)
    with pytest.raises(CompileError, match="missing RAW"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_short_m_fences_do_not_serialize_independent_output_accumulators():
    plan = autosync(
        gemm_nt(
            48, 16, 32, tile_m=16, tile_n=16, l1_k=32, l0_k=16, grid_m=1, c_slots=2, cache_b=True
        )
    )
    barriers = [
        s
        for s in plan.walk()
        if isinstance(s, SyncOp) and s.action.data == "barrier" and s.pipe.data == "M"
    ]
    covered = {g for s in barriers for g in range(s.lower.data, s.upper.data, s.stride.data)}
    assert covered == {1, 3, 5}


def test_large_cube_keeps_the_documented_implicit_order_without_m_fences():
    plan = autosync(gemm_nt(128, 256, 128, tile_m=128, tile_n=256, l1_k=128, l0_k=64))
    assert not any(
        isinstance(s, SyncOp) and s.action.data == "barrier" and s.pipe.data == "M"
        for s in plan.walk()
    )
