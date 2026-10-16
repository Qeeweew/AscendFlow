import pytest
from xdsl.dialects.builtin import IntAttr

from xdsl_npu._physical_builder import Kernel
from xdsl_npu.check import bounded_check
from xdsl_npu.codegen import emit_asc
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import ChannelOp
from xdsl_npu.model import dependencies, extract
from xdsl_npu.sync import autosync, verify_plan


@pytest.mark.parametrize("native", ["reduce_max8", "reduce_sum8"])
def test_datablock_reduction_requires_the_complete_packed_output_footprint(native):
    k = Kernel("block_max")
    source = k.buffer("source", "UB", 512)
    destination = k.buffer("destination", "UB", 32)
    k.loop(1)
    a, b = k.view(source, 512), k.view(destination, 32)
    k.task("initialize", "fill", a)
    k.task("reduce", native, a, b)
    with pytest.raises(CompileError, match="datablock"):
        extract(k.finish())


def scalar_pipeline(count=3, blocks=1):
    k = Kernel("scalar_pipeline", blocks=blocks, disjoint_external=True)
    x = k.buffer("x", "GM", count * blocks * 32, external=True, initialized=True)
    y = k.buffer("y", "GM", count * blocks * 32, external=True)
    a = k.buffer("a", "UB", 64)
    b = k.buffer("b", "UB", 64)
    k.loop(count)
    xr = k.view(x, 32, stride=32, core_stride=count * 32)
    yr = k.view(y, 32, stride=32, core_stride=count * 32)
    ar, br = k.view(a, 32, stride=32, slots=2), k.view(b, 32, stride=32, slots=2)
    k.copy("load", xr, ar)
    word = k.view(ar, 4)
    value = k.scalar_load("factor", word, stage=1)
    k.scale_value("multiply", ar, br, value, stage=1)
    k.copy("store", br, yr, stage=2)
    return k.finish()


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_scalar_capture_and_blocking_dispatch(count):
    plan = autosync(scalar_pipeline(count))
    verify_plan(parse(dump(plan)))
    assert bounded_check(plan).exhaustive
    if count:
        code = emit_asc(plan)
        assert "volatile float*" in code and "npu_value_factor" in code


def test_scalar_capture_cannot_be_shifted_without_descriptor_storage():
    module = scalar_pipeline(2)
    op = next(o for o in module.walk() if o.name == "npu.task" and o.uid.data == "multiply")
    op.attributes["stage"] = IntAttr(2)
    with pytest.raises(CompileError, match="delayed descriptors"):
        autosync(module)


def test_scalar_ready_event_cannot_be_deleted():
    plan = autosync(scalar_pipeline(3))
    channel = next(
        c for c in plan.walk() if isinstance(c, ChannelOp) and c.destination.data == "factor"
    )
    name = channel.sym_name.data
    for op in list(plan.walk()):
        if op.name == "npu.sync" and op.channel.data == name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing RAW"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


def test_independent_owner_grid_accepts_disjoint_gm_slices():
    plan = autosync(scalar_pipeline(2, blocks=4))
    assert verify_plan(plan)["independent_cores"] == 4
    code = emit_asc(plan)
    assert "asc_get_block_idx()" in code and "<<<4," in code


def test_independent_owner_grid_rejects_shared_writes():
    module = scalar_pipeline(2, blocks=2)
    view = next(
        o
        for o in module.walk()
        if o.name == "npu.view"
        and o.core_stride is not None
        and o.source.owner.name == "npu.buffer"
        and o.source.owner.sym_name.data == "y"
    )
    view.attributes["core_stride"] = IntAttr(0)
    with pytest.raises(CompileError, match="ownership overlap"):
        autosync(module)


def test_independent_core_stride_preserves_dma_alignment():
    k = Kernel("misaligned_core", blocks=2)
    x = k.buffer("x", "GM", 64, external=True, initialized=True)
    a = k.buffer("a", "UB", 32)
    k.loop(1)
    k.copy("load", k.view(x, 32, core_stride=4), k.view(a, 32))
    with pytest.raises(CompileError, match="32-byte alignment"):
        autosync(k.finish())


def test_strided_accesses_preserve_the_holes():
    k = Kernel("strided", disjoint_external=True)
    x = k.buffer("x", "GM", 128, external=True, initialized=True)
    y = k.buffer("y", "GM", 128, external=True)
    a = k.buffer("a", "UB", 128)
    b = k.buffer("b", "UB", 128)
    k.loop(1)
    af, bf = k.view(a, 128), k.view(b, 128)
    k.copy("load", k.view(x, 128), af)
    left = k.view(af, 32, rows=2, row_stride=64)
    right = k.view(af, 32, offset=32, rows=2, row_stride=64)
    bl = k.view(bf, 32, rows=2, row_stride=64)
    br = k.view(bf, 32, offset=32, rows=2, row_stride=64)
    k.task("left", "mul_rows", left, left, bl, params=(2, 8))
    k.task("right", "mul_rows", right, right, br, params=(2, 8))
    k.copy("store", bf, k.view(y, 128))
    module = k.finish()
    deps = dependencies(extract(module))
    assert not any({o.source.task, o.destination.task} == {1, 2} for o in deps)
    plan = autosync(module)
    assert bounded_check(plan).exhaustive
    assert "npu_row * 64" in emit_asc(plan)


def test_subview_cannot_read_a_strided_parent_hole():
    k = Kernel("hole")
    a = k.buffer("a", "UB", 128)
    k.loop(1)
    parent = k.view(a, 32, rows=2, row_stride=64)
    bad = k.view(parent, 32, offset=32)
    k.scale("bad", bad, bad, 2.0)
    with pytest.raises(CompileError, match="exceeds its parent"):
        autosync(k.finish())


def test_once_cache_has_one_writer_and_no_repeated_notification():
    k = Kernel("cache", disjoint_external=True)
    x = k.buffer("x", "GM", 32, external=True, initialized=True)
    y = k.buffer("y", "GM", 96, external=True)
    a = k.buffer("a", "UB", 32)
    b = k.buffer("b", "UB", 64)
    k.loop(3)
    ar = k.view(a, 32)
    br = k.view(b, 32, stride=32, slots=2)
    k.task("cache", "copy", k.view(x, 32), ar, active=(0, 1))
    k.scale("compute", ar, br, 2.0, stage=1)
    k.copy("store", br, k.view(y, 32, stride=32), stage=2)
    plan = autosync(k.finish())
    assert len([i for i in extract(plan).instances() if i.task == 0]) == 1
    verify_plan(plan)
    assert bounded_check(plan).exhaustive


def test_fixed_issue_rank_is_distinct_from_semantic_source_order():
    module = scalar_pipeline(3)
    op = next(o for o in module.walk() if o.name == "npu.task" and o.uid.data == "store")
    op.attributes["issue_order"] = IntAttr(0)
    plan = autosync(module)
    assert bounded_check(plan).exhaustive


def test_periodic_phase_interval_keeps_async_last_read_before_reinitialization():
    k = Kernel("periodic_rows", disjoint_external=True)
    ub = k.buffer("state", "UB", 64)
    gm = k.buffer("output", "GM", 256, external=True)
    k.loop(8)
    state = k.view(ub, 32, stride=32, slots=2)
    k.task("init", "fill", state, scalar=1, period=4, phase_width=2, issue_order=1)
    k.task(
        "store",
        "copy",
        state,
        k.view(gm, 32, stride=32),
        period=4,
        phase=2,
        phase_width=2,
        stage=2,
        issue_order=0,
    )
    plan = autosync(parse(dump(k.finish())))
    p = extract(plan)
    assert [i.generation for i in p.instances() if i.task == 0] == [0, 1, 4, 5]
    assert verify_plan(plan)["tasks"] == 8
    assert bounded_check(plan).exhaustive
    channel = next(c for c in plan.walk() if isinstance(c, ChannelOp) and c.source.data == "store")
    for op in list(plan.walk()):
        if op.name == "npu.sync" and op.channel.data == channel.sym_name.data:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing WAR"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


@pytest.mark.parametrize("phase,width", [(0, 0), (3, 2), (0, 5)])
def test_periodic_phase_interval_must_be_nonempty_and_fit_its_period(phase, width):
    k = Kernel("invalid_phase")
    ub = k.buffer("state", "UB", 32)
    k.loop(8)
    k.task("init", "fill", k.view(ub, 32), period=4, phase=phase, phase_width=width)
    with pytest.raises(CompileError, match="nonempty phase interval"):
        autosync(k.finish())


@pytest.mark.parametrize("rows,cols", [(1, 1024), (8, 1024), (8, 512), (64, 128)])
def test_emitted_row_broadcast_repeat_addresses_match_every_logical_lane(rows, cols):
    import re

    from xdsl_npu.codegen import native_line

    k = Kernel("broadcast")
    matrix = k.buffer("matrix", "UB", rows * cols * 4)
    expanded = k.buffer("expanded", "UB", rows * 32)
    k.loop(1)
    mr, er = k.view(matrix, rows * cols * 4), k.view(expanded, rows * 32)
    k.task("data", "fill", mr)
    k.task("scalars", "fill", er)
    k.task("shift", "sub_row_broadcast", mr, er, mr, params=(rows, cols))
    plan = autosync(k.finish())
    code = native_line(extract(plan).tasks[-1], 1)
    _, limit = re.search(r"for \(int (\w+) = 0; \w+ < (\d+);", code).groups()
    repeat, db, sb, bb, dr, sr, br = map(
        int,
        re.search(r"uint8_t\((\d+)\), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\);", code).groups(),
    )

    def offset(name, outer):
        term = re.search(name + r" \+ \(([^)]*)\)", code).group(1)
        return 0 if term == "0" else outer * int(term.split(" * ")[1]) // 4

    # Execute the 2201 normalized-mask repeat/block addressing contract,
    # independently of the compiler's physical-access frontier. A distinct
    # row scalar detects transposed repeats or an incorrect broadcast stride.
    seen = set()
    for outer in range(int(limit)):
        for rep in range(repeat):
            for block in range(8):
                for word in range(8):
                    dst = offset("matrix", outer) + rep * dr * 8 + block * db * 8 + word
                    src = offset("matrix", outer) + rep * sr * 8 + block * sb * 8 + word
                    scalar = offset("expanded", outer) + rep * br * 8 + block * bb * 8 + word
                    assert src == dst and scalar // 8 == dst // cols
                    assert 0 <= scalar < rows * 8 and dst not in seen
                    seen.add(dst)
    assert seen == set(range(rows * cols))
    assert int(limit) == min(rows, cols // 64)
