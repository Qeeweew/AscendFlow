import pytest

from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.check import bounded_check
from xdsl_npu.ir import ChannelOp, SyncOp, dump, parse
from xdsl_npu.kernels import batched_matmul_nt, rms_norm, rope, softmax, swiglu
from xdsl_npu.model import CompileError, dependencies, extract


@pytest.mark.parametrize(
    "factory,columns",
    [
        (rms_norm, 64),
        (rms_norm, 128),
        (rms_norm, 1024),
        (rms_norm, 4096),
        (rope, 16),
        (rope, 128),
        (rope, 4096),
        (softmax, 64),
        (softmax, 4096),
        (swiglu, 8),
        (swiglu, 4096),
    ],
)
@pytest.mark.parametrize("rows,blocks", [(0, 1), (8, 1), (32, 4)])
def test_operators_have_verified_native_plans(factory, columns, rows, blocks):
    source = factory(rows, columns, blocks=blocks)
    p = extract(source)
    plan = autosync(source)
    assert verify_plan(parse(dump(plan)))["independent_cores"] == blocks
    assert extract(plan) == p
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert not any("_sync(" in line for line in code.splitlines())


@pytest.mark.parametrize("factory", [rms_norm, rope, softmax, swiglu])
def test_operator_interleavings_include_first_slot_reuse(factory):
    p = autosync(factory(3, 64, row_block=1))
    result = bounded_check(p, max_states=300_000)
    assert result.exhaustive and result.terminal_states == 1


def test_rmsnorm_caches_gamma_once_and_retires_input_before_gamma():
    plan = autosync(rms_norm(8, 128, row_block=2))
    p = extract(plan)
    assert len([i for i in p.instances() if p.tasks[i.task].uid == "load_gamma"]) == 1
    assert next(s.nbytes for s in p.storages if s.name == "stats") == 64
    release = next(
        c for c in plan.walk() if isinstance(c, ChannelOp) and c.destination.data == "load"
    )
    assert release.source.data.startswith("normalize_r")
    code = emit_asc(plan)
    assert code.index("// " + release.sym_name.data) < code.index("// gamma:")
    assert len([t for t in p.tasks if t.native == "scalar_load"]) == 2


def test_rmsnorm_fixed_grain_cannot_be_silently_reduced():
    with pytest.raises(CompileError, match="row_block exceeds"):
        rms_norm(16, 4096, row_block=8)


@pytest.mark.parametrize("factory", [rms_norm, rope, softmax, swiglu])
def test_default_grain_does_not_reduce_the_requested_slot_count(factory):
    with pytest.raises(CompileError, match="requested slot count"):
        factory(8, 4096, slots=8)


def test_rope_trig_alias_requires_all_cos_readers_before_sin_dma():
    plan = autosync(rope(4, 128, row_block=1))
    ch = next(
        c for c in plan.walk() if isinstance(c, ChannelOp) and c.destination.data == "load_sin"
    )
    assert ch.source.data == "cos_high" and ch.distance.data == 0
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.channel.data == ch.sym_name.data:
            op.parent.erase_op(op)
    ch.parent.erase_op(ch)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_rope_two_halves_have_precise_byte_holes_and_two_rounding_nodes():
    source = rope(8, 128, row_block=2)
    p = extract(source)
    low = next(t for t in p.tasks if t.uid == "cos_low").refs[-1]
    high = next(t for t in p.tasks if t.uid == "cos_high").refs[-1]
    assert low.segments(0) == ((0, 256), (512, 768))
    assert high.segments(0) == ((256, 512), (768, 1024))
    assert not any(
        p.tasks[o.source.task].uid == "cos_low" and p.tasks[o.destination.task].uid == "cos_high"
        for o in dependencies(p)
    )
    assert [t.native for t in p.tasks if t.uid in {"sin_high", "rotate_low"}] == [
        "mul_rows",
        "sub_rows",
    ]


@pytest.mark.parametrize("epsilon", [1e-46, 1e39])
def test_rmsnorm_rejects_nonpositive_or_infinite_f32_epsilon(epsilon):
    with pytest.raises(CompileError, match="f32"):
        rms_norm(1, 64, epsilon=epsilon)


def test_swiglu_has_distinct_gate_and_up_release_milestones():
    plan = autosync(swiglu(4, 128, row_block=1))
    channels = [c for c in plan.walk() if isinstance(c, ChannelOp)]
    gate = next(c for c in channels if c.destination.data == "load_gate")
    up = next(c for c in channels if c.destination.data == "load_up")
    assert gate.source.data == "silu"
    assert up.source.data == "multiply_up"


def test_softmax_input_retires_before_exponential_and_sum():
    plan = autosync(softmax(4, 128, row_block=1))
    p = extract(plan)
    readers = {
        p.tasks[o.source.task].uid
        for o in dependencies(p)
        if o.hazard == "WAR" and o.storage == "input_slots"
    }
    assert readers == {"partial_max", "shift_r0"}
    # The required later V->S wait already gates next-step DMA dispatch.
    # A second, redundant V->MTE2 release channel would not improve this plan.
    assert not any(isinstance(c, ChannelOp) and c.destination.data == "load" for c in plan.walk())
    ready = next(
        c
        for c in plan.walk()
        if isinstance(c, ChannelOp) and c.destination.data == "capture_inv_r0"
    )
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.channel.data == ready.sym_name.data:
            op.parent.erase_op(op)
    ready.parent.erase_op(ready)
    with pytest.raises(CompileError, match="missing (RAW|WAR)"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


@pytest.mark.parametrize(
    "factory,args", [(rms_norm, (7, 65)), (rms_norm, (7, 64)), (rope, (8, 17))]
)
def test_concrete_shape_diagnostics(factory, args):
    with pytest.raises(CompileError):
        factory(*args, blocks=2)


@pytest.mark.parametrize("m,n", [(16, 32), (32, 64), (64, 32), (128, 128)])
@pytest.mark.parametrize("chunks", [1, 3, 5])
def test_batched_matmul_has_one_clear_and_one_final_export(m, n, chunks):
    source = batched_matmul_nt(4, m, n, chunks * 32, k_block=32)
    p = extract(source)
    instances = p.instances()
    assert len([i for i in instances if p.tasks[i.task].uid == "mad_init"]) == 1
    assert len([i for i in instances if p.tasks[i.task].uid == "mad_acc"]) == chunks - 1
    assert len([i for i in instances if p.tasks[i.task].uid == "export"]) == 1
    plan = autosync(source)
    assert verify_plan(plan)["independent_cores"] == 4
    code = emit_asc(plan)
    assert "asc_copy_gm2l1_nd2nz" in code
    if chunks > 1 and (m // 16) * (n // 16) < 10:
        assert "asc_sync_pipe(PIPE_M)" in code
    else:
        assert "asc_sync_pipe(PIPE_M)" not in code
    assert "asc_copy_l0c2gm" in code and code.count("asc_sync_pipe(PIPE_ALL)") == 1


def test_matmul_releases_l1_at_load2d_and_l0_at_mad():
    plan = autosync(batched_matmul_nt(2, 32, 64, 96, k_block=32))
    channels = [c for c in plan.walk() if isinstance(c, ChannelOp)]
    assert {c.source.data for c in channels if c.destination.data in {"load_a", "load_bt"}} <= {
        "operand_a",
        "operand_b",
    }
    assert {
        c.source.data
        for c in channels
        if c.destination.data in {"operand_a", "operand_b"} and c.distance.data > 0
    } <= {"mad_init", "mad_acc"}
    assert bounded_check(plan, max_states=300_000).exhaustive


def test_matmul_cannot_accumulate_an_uninitialized_l0c_carry():
    from xdsl.dialects.builtin import ArrayAttr, IntAttr

    from xdsl_npu.ir import TaskOp

    source = batched_matmul_nt(1, 16, 16, 32, k_block=16)
    task = next(t for t in source.walk() if isinstance(t, TaskOp) and t.uid.data == "mad_init")
    task.attributes["params"] = ArrayAttr(IntAttr(p) for p in (16, 16, 16, 0))
    with pytest.raises(CompileError, match="undefined accumulator"):
        autosync(source)


def test_fixed_matmul_tile_cannot_be_silently_shrunk_to_fit():
    with pytest.raises(CompileError, match="capacity exceeded"):
        autosync(batched_matmul_nt(1, 256, 256, 256, k_block=256))
