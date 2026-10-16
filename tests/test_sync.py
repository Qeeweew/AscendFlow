import pytest
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.transforms.dead_code_elimination import dce

from tests.fixtures.pipelines import cube_pipeline, vector_pipeline
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.check import bounded_check
from xdsl_npu.codegen import emit_asc
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import ChannelOp, SyncOp, ViewOp
from xdsl_npu.model import dependencies, extract
from xdsl_npu.sync import autosync, explain_plan, verify_plan


@pytest.mark.parametrize("build", [vector_pipeline, cube_pipeline])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_finite_boundaries_and_parse_roundtrip(build, count, slots):
    source = build(count, slots=slots)
    text = dump(source)
    plan = autosync(parse(text))
    report = verify_plan(parse(dump(plan)))
    assert report["tasks"] == count * len(extract(source).tasks)
    assert dump(source) == text
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert not any("_sync(" in line for line in code.splitlines())


@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("build", [vector_pipeline, cube_pipeline])
def test_independent_interleaving_model(build, count):
    result = bounded_check(autosync(build(count)), max_states=250_000)
    assert result.exhaustive and result.terminal_states == 1


def channels(plan):
    return [op for op in plan.walk() if isinstance(op, ChannelOp)]


def remove_channel(plan, ch):
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.channel.data == ch.sym_name.data:
            op.parent.erase_op(op)
    ch.parent.erase_op(ch)


def test_last_reader_releases_input_before_unrelated_postprocessing():
    plan = autosync(vector_pipeline())
    release = next(c for c in channels(plan) if c.destination.data == "load")
    assert release.source.data == "scale"
    code = emit_asc(plan)
    assert code.index("// " + release.sym_name.data) < code.index("asc_add_scalar(")
    late = autosync(vector_pipeline(late_reader=True))
    release = next(c for c in channels(late) if c.destination.data == "load")
    assert release.source.data == "last_read"


def test_delete_war_release_rejected_by_two_independent_checkers():
    plan = autosync(vector_pipeline(4))
    release = next(c for c in channels(plan) if c.destination.data == "load")
    remove_channel(plan, release)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_ssa_last_use_does_not_release_async_store_input():
    plan = autosync(vector_pipeline(5))
    release = next(c for c in channels(plan) if c.source.data == "store")
    assert release.destination.data == "scale" and release.distance.data == 2
    remove_channel(plan, release)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)


def test_last_native_piece_can_release_a_lower_generation_in_a_later_phase():
    k = Kernel("signed_generation", disjoint_external=True)
    x = k.buffer("x", "GM", 32, external=True, initialized=True)
    y = k.buffer("y", "GM", 64, external=True)
    ub = k.buffer("tile", "UB", 32)
    k.loop(2)
    src, tile, dst = k.view(x, 32), k.view(ub, 32), k.view(y, 32, stride=32)
    k.task("load", "copy", src, tile, active=(0, 1))
    k.task("store", "copy", tile, dst, stage=1)
    k.task("reuse", "adds", tile, tile, scalar=1, stage=3, active=(0, 1), logical_phase=1)
    source = k.finish()
    source.body.block.first_op.attributes["epoch_divisor"] = IntAttr(2)
    plan = autosync(parse(dump(source)))
    release = next(
        c for c in channels(plan) if c.source.data == "store" and c.destination.data == "reuse"
    )
    assert release.distance.data == -1 and release.lower.data == 1
    assert verify_plan(plan)["tasks"] == 4
    assert bounded_check(plan).exhaustive
    remove_channel(plan, release)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_binary_event_ring_reduction_needs_reverse_completion():
    plan = autosync(vector_pipeline(6))
    ch = next(c for c in channels(plan) if c.source.data == "load")
    assert ch.width.data == 2
    ch.attributes["width"] = IntAttr(1)
    with pytest.raises(CompileError, match="re-set"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="re-Set|wrong epoch"):
        bounded_check(plan)


@pytest.mark.parametrize("mutation", ["wait", "clock", "pair", "bank", "barrier"])
def test_negative_plan_mutations(mutation):
    plan = autosync(vector_pipeline(4))
    syncs = [o for o in plan.walk() if isinstance(o, SyncOp)]
    wait = next(o for o in syncs if o.action.data == "wait")
    if mutation == "wait":
        wait.parent.erase_op(wait)
    elif mutation == "clock":
        wait.attributes["distance"] = IntAttr(7)
    elif mutation == "pair":
        wait.attributes["pipe"] = StringAttr("MTE3")
    elif mutation == "bank":
        channels(plan)[0].attributes["base"] = IntAttr(8)
    else:
        barrier = next(o for o in syncs if o.action.data == "barrier")
        barrier.parent.erase_op(barrier)
    with pytest.raises(CompileError):
        verify_plan(plan)


def test_generic_dce_cannot_delete_unused_async_task_ids():
    plan = autosync(vector_pipeline(3))
    text = dump(plan)
    dce(plan)
    assert dump(plan) == text
    verify_plan(plan)


def minimal(*, alias=False, size=64, unknown=False):
    k = Kernel("minimal", disjoint_external=not alias)
    x = k.buffer("x", "GM", size, external=True, initialized=True)
    y = k.buffer("y", "GM", size, external=True)
    u = k.buffer("u", "UB", size)
    k.loop(1)
    a, b, c = k.view(x, 32), k.view(y, 32), k.view(u, 32)
    k.copy("load", a, c)
    if unknown:
        k.task("opaque", "unknown_call", c)
    k.copy("store", c, b)
    return k.finish()


@pytest.mark.parametrize(
    "kwargs, diagnostic",
    [
        ({"alias": True}, "alias is unknown"),
        ({"size": 192 * 1024 + 32}, "UB capacity exceeded"),
        ({"unknown": True}, "effects/completion contract"),
    ],
)
def test_input_contract_diagnostics(kwargs, diagnostic):
    with pytest.raises(CompileError, match=diagnostic):
        autosync(minimal(**kwargs))


def test_padding_native_footprint_is_not_logical_extent():
    module = cube_pipeline(1)
    op = next(o for o in module.walk() if isinstance(o, ViewOp) and o.nbytes.data == 512)
    op.attributes["nbytes"] = IntAttr(256)
    with pytest.raises(CompileError, match="equal native footprints"):
        autosync(module)


def test_layout_route_cannot_silently_add_staging():
    k = Kernel("bad_route", core="aic")
    c = k.buffer("c", "L0C", 1024, layout="NZ")
    u = k.buffer("u", "L1", 1024, layout="NZ")
    k.loop(1)
    k.copy("direct", k.view(c, 1024), k.view(u, 1024))
    with pytest.raises(CompileError, match="unsupported dav-2201 DMA route"):
        autosync(k.finish())


def test_partial_view_and_recast_preserve_storage_provenance():
    k = Kernel("aliases", disjoint_external=True)
    x = k.buffer("x", "GM", 128, external=True, initialized=True)
    y = k.buffer("y", "GM", 32, external=True)
    u = k.buffer("u", "UB", 128)
    k.loop(1)
    source = k.view(x, 64)
    parent = k.view(u, 128)
    half = k.view(parent, 64, dtype="f16")
    whole = k.view(parent, 64)
    k.copy("load", source, whole)
    sub = k.view(half, 32, offset=32, dtype="f32")
    k.copy("store", sub, k.view(y, 32))
    module = k.finish()
    obs = dependencies(extract(module))
    assert any(
        o.storage == "u" and o.hazard == "RAW" and (o.start, o.stop) == (32, 64) for o in obs
    )
    verify_plan(autosync(module))


def test_uninitialized_partial_write_is_rejected():
    k = Kernel("undefined", disjoint_external=True)
    x = k.buffer("x", "GM", 64, external=True, initialized=True)
    y = k.buffer("y", "GM", 64, external=True)
    u = k.buffer("u", "UB", 64)
    k.loop(1)
    k.copy("load", k.view(x, 32), k.view(u, 32))
    k.copy("store", k.view(u, 64), k.view(y, 64))
    with pytest.raises(CompileError, match="reads undefined u\\[32:64\\]"):
        autosync(k.finish())


def test_budget_exhaustion_is_explicitly_incomplete():
    result = bounded_check(autosync(vector_pipeline(3)), max_states=10)
    assert not result.exhaustive


def test_expert_stage_and_buffers_are_preserved():
    source = cube_pipeline(6, slots=3)
    before = extract(source)
    after = extract(autosync(source))
    assert before == after


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_static_tail_preserves_native_footprints(count):
    source = vector_pipeline(count, tail=24)
    plan = autosync(source)
    verify_plan(parse(dump(plan)))
    if count:
        refs = extract(plan).tasks[0].refs
        assert refs[0].interval(count - 1)[1] == (count - 1) * 512 + 96
        assert "? 96 : 512" in emit_asc(plan)
    if count < 4:
        assert bounded_check(plan).exhaustive


def test_unaligned_tail_requires_a_byte_dma_recipe():
    with pytest.raises(CompileError, match="32-byte aligned"):
        autosync(vector_pipeline(3, tail=7))


def test_source_notifications_do_not_fence_later_vector_accesses():
    plan = autosync(vector_pipeline(6))
    barriers = [o for o in plan.walk() if isinstance(o, SyncOp) and o.action.data == "barrier"]
    assert len(barriers) == 1
    assert (barriers[0].lower.data, barriers[0].upper.data) == (0, 6)
    barriers[0].attributes["lower"] = IntAttr(4)
    with pytest.raises(CompileError, match="missing RAW"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_cube_uses_source_verified_no_transpose_load():
    source = emit_asc(autosync(cube_pipeline(1)))
    assert "asc_copy_l12l0a(" in source
    assert "load_cbuf_to_ca(" not in source


def test_independent_components_cannot_copy_event_numbers():
    plan = autosync(cube_pipeline(3))
    ready = [c for c in channels(plan) if c.source.data in {"load_a", "load_b"}]
    assert len(ready) == 2
    ready[1].attributes["base"] = ready[0].base
    with pytest.raises(CompileError, match="re-set"):
        verify_plan(plan)


def test_event_pressure_cannot_trigger_silent_serialization():
    source = vector_pipeline(20, slots=9)
    before = dump(source)
    with pytest.raises(CompileError, match="event bank has 8 IDs"):
        autosync(source)
    assert dump(source) == before


def test_witnesses_distinguish_write_done_read_retired_and_event_consumption():
    proof = explain_plan(autosync(vector_pipeline(4)))
    obligations = proof["physical_obligations"]
    assert any(
        o["required_milestone"] == "read_retired" and o["hazard"] == "WAR" for o in obligations
    )
    assert any(
        o["required_milestone"] == "write_done" and o["hazard"] == "RAW" for o in obligations
    )
    assert proof["binary_event_reuse"]
    assert all("wait" in o["consumption_path"][0] for o in proof["binary_event_reuse"])
