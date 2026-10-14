import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.transforms.dead_code_elimination import dce

from xdsl_npu import autosync, emit_asc, gemm_bias_silu, verify_plan
from xdsl_npu.check import bounded_check
from xdsl_npu.examples import mixed_pipeline
from xdsl_npu.ir import (
    BufferOp,
    CrossChannelOp,
    CrossDrainOp,
    CrossSyncOp,
    EndOp,
    KernelOp,
    StreamOp,
    TaskOp,
    TaskType,
    ViewOp,
    dump,
    parse,
    ref_type,
    task_attrs,
)
from xdsl_npu.mixed import (
    cross_op,
    extract_mixed,
    global_dependencies,
    mixed_execution,
    mixed_graph,
)
from xdsl_npu.mixed_check import bounded_mixed_check
from xdsl_npu.model import CompileError, Instance
from xdsl_npu.oracles import mixed_inputs
from xdsl_npu.sync import explain_plan
from xdsl_npu.sync import verify_plan as local_verify


@pytest.mark.parametrize("count,slots", [(0, 2), (1, 1), (2, 1), (3, 2), (5, 3), (17, 2)])
def test_collective_boundaries_reuse_and_counter_capacity(count, slots):
    source = mixed_pipeline(count, slots)
    before = dump(source)
    plan = autosync(parse(before))
    report = verify_plan(parse(dump(plan)))
    assert dump(source) == before
    assert report["epochs"] == count
    assert report["tasks_per_group"] == count * 12
    assert all(peak <= slots for peak in report["cross_backlog_upper_bounds"].values())
    bounded = bounded_mixed_check(plan, scope="shared")
    assert bounded.exhaustive and bounded.terminal_states == 1 and bounded.scope == "shared"
    asc = emit_asc(plan)
    assert "__mix__(1, 2)" in asc and asc.count("asc_sync_pipe(PIPE_ALL)") == 3
    assert "asc_get_sub_block_id() == 0" in asc and "asc_get_sub_block_id() == 1" in asc
    if count:
        expected = {"gm_ready": 0}
        if count > slots:
            expected["gm_credit"] = 1
        assert report["cross_counter_ids"] == expected
    else:
        assert not any(
            isinstance(o, (CrossChannelOp, CrossSyncOp, CrossDrainOp)) for o in plan.walk()
        )


def remove_cross_channel(plan, name):
    for op in list(plan.walk()):
        if (
            isinstance(op, CrossChannelOp)
            and op.sym_name.data == name
            or isinstance(op, (CrossSyncOp, CrossDrainOp))
            and op.channel.data == name
        ):
            op.parent.erase_op(op)


@pytest.mark.parametrize("name,hazard", [("gm_ready", "RAW"), ("gm_credit", "WAR")])
def test_counterexamples_are_rejected_independently(name, hazard):
    plan = autosync(mixed_pipeline(3, 2))
    remove_cross_channel(plan, name)
    with pytest.raises(CompileError, match=f"missing mixed {hazard}"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_mixed_check(plan, scope="shared")
    with pytest.raises(CompileError):
        emit_asc(plan)


def test_actor_import_is_a_condition_not_an_initialization_or_standalone_proof():
    plan = autosync(mixed_pipeline(2, 1))
    m = extract_mixed(plan)
    assert not m.roots["scratch"].initialized.data
    with pytest.raises(CompileError, match="undefined scratch"):
        local_verify(m.projections["aiv0"])
    conditional = local_verify(m.projections["aiv0"], _imports=m.imports)
    assert "conditional" in conditional["verification"]
    with pytest.raises(CompileError, match="undefined scratch"):
        emit_asc(m.projections["aiv0"])
    assert all(
        not o.initialized.data
        for o in plan.walk()
        if isinstance(o, BufferOp) and o.sym_name.data == "scratch"
    )


def test_external_pointer_names_do_not_grant_a_noalias_contract():
    source = mixed_pipeline(1, 1)
    source.body.block.first_op.attributes["disjoint_external"] = IntAttr(0)
    with pytest.raises(CompileError, match="disjoint contract"):
        autosync(source)


def test_projection_cannot_hide_misplaced_cross_effects():
    plan = autosync(mixed_pipeline(3, 2))
    actor = extract_mixed(plan).actors["aic"]
    stream = next(o for o in actor.body.block.ops if isinstance(o, StreamOp))
    drain = next(o for o in actor.body.block.ops if isinstance(o, CrossDrainOp))
    stream.body.block.insert_op_before(drain.clone(), stream.body.block.last_op)
    with pytest.raises(CompileError, match="actor epilogue"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="actor epilogue"):
        emit_asc(plan)


def test_zero_epoch_cross_effects_still_require_a_valid_activation():
    plan = autosync(mixed_pipeline(0, 2))
    actor = extract_mixed(plan).actors["aiv0"]
    stream = next(o for o in actor.body.block.ops if isinstance(o, StreamOp))
    task = next(o for o in stream.body.block.ops if isinstance(o, TaskOp))
    stream.body.block.insert_op_before(
        cross_op(stream.body.block.args[0], "wait", "gm_ready", 0, 1), task
    )
    with pytest.raises(CompileError, match="activation/epoch clock"):
        verify_plan(plan)


def test_release_is_read_retirement_and_does_not_wait_for_vector_or_store():
    plan = autosync(mixed_pipeline(3, 2))
    m, channels, commands = mixed_execution(plan)
    graph, begin, end, sends, _, _, _ = mixed_graph(m, channels, commands)
    for actor in ("aiv0", "aiv1"):
        tasks = {t.uid: i for i, t in enumerate(m.programs[actor].tasks)}
        read = (actor, Instance(0, tasks["import_tile"]))
        scale = (actor, Instance(0, tasks["scale"]))
        store = (actor, Instance(0, tasks["store"]))
        credit = sends[(actor, "gm_credit", 0)]
        assert graph.precedes(end[read], credit)
        assert not graph.precedes(end[scale], credit)
        assert not graph.precedes(end[store], credit)
        assert not graph.precedes(credit, begin[scale])
    report = explain_plan(plan)
    assert any(o["milestone"] == "read_retired" for o in report["cross_obligations"])


def test_2201_cross_wait_is_controller_gate_not_an_mte2_queue_entry():
    # UB slot 1 needs store[0], dispatched at step 2, before import[1].
    # Its MTE2 local Wait must NOT prevent the controller's cross Wait[1]
    # from consuming ready[1] and then dispatching that earlier store.
    plan = autosync(mixed_pipeline(2, 1))
    _, _, commands = mixed_execution(plan)
    assert all(c.pipe == "S" for seq in commands.values() for c in seq if c.kind == "cross_wait")
    assert verify_plan(plan)["epochs"] == 2


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("participant", "collective"),
        ("tail", "epoch"),
        ("id", "namespace"),
        ("reorder", "epoch"),
        ("inactive", "common epoch"),
    ],
)
def test_invalid_collective_plans_are_rejected(mutation, diagnostic):
    plan = autosync(mixed_pipeline(3, 2))
    m = extract_mixed(plan)
    channel = next(
        o for o in plan.walk() if isinstance(o, CrossChannelOp) and o.sym_name.data == "gm_credit"
    )
    if mutation == "participant":
        channel.attributes["sources"] = ArrayAttr([StringAttr("aiv0")])
    elif mutation == "tail":
        drain = next(o for o in plan.walk() if isinstance(o, CrossDrainOp))
        drain.parent.erase_op(drain)
    elif mutation == "id":
        channel.attributes["flag_id"] = IntAttr(16)
    elif mutation == "reorder":
        sync = next(
            o
            for o in plan.walk()
            if isinstance(o, CrossSyncOp)
            and o.channel.data == "gm_ready"
            and o.action.data == "wait"
        )
        sync.attributes["epoch_offset"] = IntAttr(1)
    elif mutation == "inactive":
        stream = next(o for o in m.actors["aiv1"].walk() if isinstance(o, StreamOp))
        stream.attributes["count"] = IntAttr(2)
    with pytest.raises(CompileError, match=diagnostic):
        verify_plan(plan)


def test_second_gm_reader_cannot_be_released_by_the_first_import():
    plan = autosync(mixed_pipeline(3, 2))
    m = extract_mixed(plan)
    actor = m.actors["aiv0"]
    stream = next(o for o in actor.walk() if isinstance(o, StreamOp))
    reader = next(
        o for o in stream.body.block.ops if isinstance(o, TaskOp) and o.uid.data == "import_tile"
    )
    buffer = BufferOp.create(
        result_types=[ref_type("UB", "f32")],
        attributes={
            "sym_name": StringAttr("late_ub"),
            "nbytes": IntAttr(512),
            "external": IntAttr(0),
            "initialized": IntAttr(0),
        },
    )
    actor.body.block.insert_op_before(buffer, stream)
    ref = ViewOp.create(
        operands=[buffer.ref, stream.body.block.args[0]],
        result_types=[ref_type("UB", "f32")],
        attributes={
            "offset": IntAttr(0),
            "stride": IntAttr(0),
            "modulus": IntAttr(0),
            "nbytes": IntAttr(512),
            "tail_nbytes": IntAttr(0),
        },
    )
    stream.body.block.insert_op_before(ref, reader)
    attrs = task_attrs("late_reader", "copy", 3)
    attrs.update(lower=IntAttr(0), upper=IntAttr(1), issue_order=IntAttr(30))
    late = TaskOp.create(
        operands=[reader.refs[0], ref.ref], result_types=[TaskType()], attributes=attrs
    )
    stream.body.block.insert_op_before(late, stream.body.block.last_op)
    assert isinstance(stream.body.block.last_op, EndOp)
    with pytest.raises(CompileError, match="missing mixed WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_mixed_check(plan, scope="shared")


def test_missing_physical_writer_cannot_be_assumed_initialized():
    source = mixed_pipeline(1, 1)
    actor = next(o for o in source.walk() if isinstance(o, KernelOp) and o.sym_name.data == "aic")
    producer = next(o for o in actor.walk() if isinstance(o, TaskOp) and o.uid.data == "export")
    producer.parent.erase_op(producer)
    with pytest.raises(CompileError, match="undefined scratch"):
        autosync(source)


def test_sixteen_unacknowledged_notifications_cannot_compile():
    source = mixed_pipeline(16, 1)
    m = extract_mixed(source)
    # The input/operand/C rings remain bounded by their local read retirements.
    # Make only shared GM large enough to permit sixteen independent exports.
    m.roots["scratch"].attributes["nbytes"] = IntAttr(16 * 1024)
    for actor in m.actors.values():
        for op in actor.walk():
            if isinstance(op, ViewOp) and op.source is m.roots["scratch"].ref:
                op.attributes["modulus"] = IntAttr(16)
    with pytest.raises(CompileError, match="15-notification"):
        autosync(source)


def test_small_full_interleavings_and_explicit_budget_boundary():
    plan = autosync(mixed_pipeline(1, 1))
    reduced = bounded_mixed_check(plan)
    unreduced = bounded_mixed_check(plan, reduce=False, max_states=200_000)
    assert reduced.exhaustive and unreduced.exhaustive
    assert reduced.terminal_states == unreduced.terminal_states == 1
    assert reduced.states <= unreduced.states
    incomplete = bounded_check(plan, max_states=1)
    assert not incomplete.exhaustive


def test_fused_epilogue_has_explicit_rounding_and_native_row_dma():
    source = gemm_bias_silu(
        96, 64, 64, tile_m=32, tile_n=32, l1_k=64, l0_k=32, grid_m=1, cache_b=True
    )
    plan = autosync(source)
    m = extract_mixed(plan)
    assert verify_plan(plan)["groups"] == 2
    source_tasks = {(a, t.uid): t for a, p in m.programs.items() for t in p.tasks}
    assert source_tasks[("aic", "export")].refs[1].row_stride == 128
    assert source_tasks[("aiv0", "store")].native == "copy_rows"
    code = emit_asc(plan)
    assert "uint16_t(16), uint16_t(4), uint16_t(0), uint16_t(4)" in code
    data, expected, tolerance, shape = mixed_inputs(plan)
    assert shape == [96, 64, 64] and len(expected) == 96 * 64 and tolerance == (5e-5, 5e-6)
    assert len(data[2]) == m.roots["scratch"].nbytes.data
    assert data[2] == b"\xcd" * len(data[2])
    dce(plan)
    assert verify_plan(plan)["epochs"] == 3


@pytest.mark.parametrize("l1_slots,l0_slots", [(1, 1), (2, 3), (3, 2)])
def test_fused_expert_buffer_counts_change_the_actual_retirement_distance(l1_slots, l0_slots):
    source = gemm_bias_silu(
        48,
        32,
        64,
        tile_m=16,
        tile_n=16,
        l1_k=32,
        l0_k=16,
        grid_m=1,
        l1_slots=l1_slots,
        l0_slots=l0_slots,
    )
    dependencies = global_dependencies(extract_mixed(source))
    assert any(
        o.hazard == "WAR"
        and o.storage == "aic/l1a"
        and src[1].generation == 1
        and dst[1].generation == 2 * l1_slots
        for src, dst, o in dependencies
    )
    assert any(
        o.hazard == "WAR"
        and o.storage == "aic/l0a"
        and src[1].generation == 0
        and dst[1].generation == l0_slots
        for src, dst, o in dependencies
    )
    assert verify_plan(autosync(source))["epochs"] == 3


def test_shared_written_views_cannot_change_alias_relationship_in_another_group():
    source = gemm_bias_silu(32, 32, 16, tile_m=16, tile_n=16, l1_k=16, l0_k=16)
    m = extract_mixed(source)
    view = next(
        o
        for o in m.actors["aiv1"].walk()
        if isinstance(o, ViewOp) and o.source is m.roots["scratch"].ref
    )
    view.attributes["core_stride"] = IntAttr(0)
    with pytest.raises(CompileError, match="translations"):
        autosync(source)


def test_actual_physical_effects_derive_both_half_reader_releases():
    source = mixed_pipeline(3, 2)
    dependencies = global_dependencies(extract_mixed(source))
    releases = [
        (src, dst, o)
        for src, dst, o in dependencies
        if o.storage == "scratch" and o.hazard == "WAR"
    ]
    assert {src[0] for src, _, _ in releases} == {"aiv0", "aiv1"}
    assert {o.stop - o.start for _, _, o in releases} == {512}


def test_uneven_aiv_arrival_preserves_the_collective_and_scalar_dispatch_gate():
    plan = autosync(mixed_pipeline(3, 2, aiv1_delay=4))
    m, channels, commands = mixed_execution(plan)
    graph, begin, end, _, _, _, _ = mixed_graph(m, channels, commands)
    tasks = {t.uid: i for i, t in enumerate(m.programs["aiv1"].tasks)}
    assert graph.precedes(
        end[("aiv1", Instance(0, tasks["capture_delay"]))],
        begin[("aiv1", Instance(0, tasks["import_tile"]))],
    )
    assert verify_plan(plan)["cross_backlog_upper_bounds"] == {"gm_ready": 2, "gm_credit": 2}
    assert bounded_mixed_check(plan, scope="shared").exhaustive
    code = emit_asc(plan)
    assert "volatile float*" in code and "npu_value_capture_delay" in code
