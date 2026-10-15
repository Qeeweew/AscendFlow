"""Hierarchical source addresses and parameterized region boundary summaries."""

import pytest
from xdsl.dialects import arith, scf

from examples.gemm.hierarchical import hierarchical_gemm
from examples.gemm.reduction import reduction_gemm
from examples.vector.independent_rings import independent_rings
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.ir import dump, parse
from xdsl_npu.loop_graph import Channel
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.region_summary import (
    Access,
    _event_reuse,
    summarize_local_events,
    summarize_reused_storage,
)
from xdsl_npu.source_ir import AllocationOp, PipelineOp, ProgramOp, SizeOp, ViewOp


def _value(value, bindings):
    if value in bindings:
        return bindings[value]
    op = value.owner
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    lhs, rhs = (_value(operand, bindings) for operand in op.operands)
    if isinstance(op, arith.AddiOp):
        return lhs + rhs
    if isinstance(op, arith.SubiOp):
        return lhs - rhs
    if isinstance(op, arith.MuliOp):
        return lhs * rhs
    if isinstance(op, (arith.DivUIOp, arith.RemUIOp, arith.CeilDivUIOp)):
        assert lhs >= 0 and rhs > 0
        if isinstance(op, arith.RemUIOp):
            return lhs % rhs
        return (lhs + rhs - 1) // rhs if isinstance(op, arith.CeilDivUIOp) else lhs // rhs
    if isinstance(op, arith.MinSIOp):
        return min(lhs, rhs)
    if isinstance(op, arith.MaxSIOp):
        return max(lhs, rhs)
    raise AssertionError(op.name)


def _address(view, bindings):
    # Independently derive row-major byte pitches from the allocation's axes.
    # No production normalization, address, or ownership proof is called.
    allocation = view.source.owner
    assert isinstance(allocation, AllocationOp)
    extents = [dimension.data for dimension in view.source.type.shape]
    for axis, dimension in zip(allocation.dynamic_axes, allocation.dimensions):
        extents[axis.data] = _value(dimension, bindings)
    linear = 0
    for origin, extent in zip(view.origins, extents):
        linear = linear * extent + _value(origin, bindings)
    return linear * {"f16": 2, "bf16": 2, "f32": 4}[view.source.type.dtype.data]


def test_parent_and_child_coordinates_and_axis_extents_are_retained_in_ir():
    module = parse(dump(hierarchical_gemm.to_ir()))
    program = module.body.block.first_op
    assert isinstance(program, ProgramOp)
    parents = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    children = [op for op in module.walk() if isinstance(op, PipelineOp)]
    assert len(parents) == len(children) == 1
    parent, child = parents[0], children[0]
    assert child.parent_op() is parent
    sizes = {op.sym_name.data: op.value for op in module.walk() if isinstance(op, SizeOp)}
    assert parent.ub is sizes["outputs"] and child.limit is sizes["panels"]
    views = [op for op in module.walk() if isinstance(op, ViewOp)]
    external = {
        op.source.owner.sym_name.data: op
        for op in views
        if isinstance(op.source.owner, AllocationOp) and op.source.type.memory.data == "GM"
    }
    coordinates = (program.body.block.args[0], parent.body.block.args[0], child.body.block.args[0])
    for name in ("A", "B"):
        view = external[name]
        assert tuple(view.origins[:3]) == coordinates
        assert tuple(axis.data for axis in view.axes) == (3, 4)
        assert tuple(view.source.owner.dimensions) == (sizes["outputs"], sizes["panels"])
    assert tuple(external["C"].origins[:2]) == coordinates[:2]
    assert tuple(axis.data for axis in external["C"].axes) == (2, 3)
    for outputs, panels in ((1, 1), (2, 7), (17, 1024), (1 << 20, (1 << 20) + 1)):
        for group, output, k in ((0, 0, 0), (19, outputs - 1, panels - 1)):
            bindings = {
                sizes["outputs"]: outputs,
                sizes["panels"]: panels,
                program.body.block.args[0]: group,
                parent.body.block.args[0]: output,
                child.body.block.args[0]: k,
            }
            linear = (group * outputs + output) * panels + k
            assert _address(external["A"], bindings) == linear * 32 * 64 * 2
            assert _address(external["B"], bindings) == linear * 64 * 64 * 2
            assert _address(external["C"], bindings) == (group * outputs + output) * 32 * 64 * 4
    compiled = compile(module)
    assert compiled.report["invocation_bound"] == "outputs"
    assert compiled.report["native_templates"] == 6


@pytest.mark.parametrize("bounds", ("panels", "1, outputs", "0, outputs, 2"))
def test_parent_axis_cannot_use_the_wrong_extent_or_traversal(bounds):
    definition = N.from_source(f"""
        def invalid(A: N.Tensor((outputs, panels, 64), N.f32),
                    outputs: N.int64, panels: N.int64):
            with N.Kernel():
                x = N.alloc_ub((64,))
                for output in N.serial({bounds}):
                    for k in N.Pipelined(panels, stages=1):
                        N.copy(A[output, k, :], x)
    """)
    with pytest.raises(CompileError, match="bounded by its tensor extent|zero-based unit-step"):
        lower_source(definition.to_ir())


def _members(access, count):
    if isinstance(access, Access):
        return {(access.position.evaluate(count), *access.task)}
    return {
        (g, *access.task)
        for g in range(access.first.evaluate(count), access.last.evaluate(count) + 1, access.step)
    }


def _native_cells(loop, roots, count):
    """Independent finite byte oracle: no production access_cells/frontiers."""
    ordered = sorted(
        (task.logical_phase, i, actor, task)
        for actor, data in loop.actors.items()
        for i, task in enumerate(data.program.tasks)
    )
    for g in range(count):
        for _, _, actor, task in ordered:
            if task.last and g != count - 1:
                continue
            touched = {}
            for number, (ref, mode) in enumerate(zip(task.refs, task.modes)):
                root = loop.root(actor, ref)
                if root not in roots:
                    continue
                if task.native == "mad_first" and g == 0 and number == 2:
                    mode = "W"
                # Independently evaluate the physical affine/modulo address.
                address = sum(o + s * ((g // d) % m if m else g // d) for o, s, m, d in ref.terms)
                for row in range(ref.rows):
                    for byte in range(
                        address + row * ref.row_stride, address + row * ref.row_stride + ref.nbytes
                    ):
                        touched.setdefault((root, byte), set()).update(mode)
            yield (g, actor, task.uid), touched


def _frontiers(loop, roots, count):
    # [reads before first writer, first writer, last writer, pending readers]
    result = {}
    for key, touched in _native_cells(loop, roots, count):
        for cell, mode in touched.items():
            state = result.setdefault(cell, [set(), None, None, set()])
            if "R" in mode:
                state[3].add(key)
                if state[1] is None:
                    state[0].add(key)
            if "W" in mode:
                state[1] = state[1] or key
                state[2], state[3] = key, set()
    return result


def _check_frontier(loop, roots, count, summary):
    expected = _frontiers(loop, roots, count)
    seen = set()
    for cell in summary.for_count(count):
        actual = [
            set().union(*(_members(r, count) for r in cell.reads_before_write)),
            next(iter(_members(cell.first_write, count))) if cell.first_write else None,
            next(iter(_members(cell.last_write, count))) if cell.last_write else None,
            set().union(*(_members(r, count) for r in cell.pending_reads)),
        ]
        for byte in range(cell.start, cell.stop):
            index = cell.storage, byte
            assert index not in seen  # Exactly one mutually exclusive N domain.
            seen.add(index)
            assert actual == expected.get(index, [set(), None, None, set()])
    assert expected.keys() <= seen


@pytest.fixture(scope="module")
def reduction_boundary():
    loop = extract_loop(
        lower_source(reduction_gemm.specialize(M=16, N=16, BK=16, GROUPS=1).to_ir())
    )
    roots = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref in task.refs
        if not ref.storage.external
    }
    return loop, roots, summarize_reused_storage(loop, roots)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 7, 11, 17])
def test_parameterized_child_frontier_matches_independent_byte_oracle(reduction_boundary, count):
    loop, roots, summary = reduction_boundary
    _check_frontier(loop, roots, count, summary)


def test_nested_reuse_requires_each_level_actual_last_reader(reduction_boundary):
    loop, _, summary = reduction_boundary
    tasks = {(actor, t.uid): t for actor, data in loop.actors.items() for t in data.program.tasks}
    for count in (1, 2, 3, 7, 1048576):
        war = [
            d
            for d in summary.invocation_dependencies()
            if d.hazard == "WAR" and d.domain.contains(count)
        ]
        assert {tasks[d.source.task].pipe for d in war if d.storage.endswith(("/a1", "/b1"))} == {
            "MTE1"
        }
        assert {tasks[d.source.task].pipe for d in war if d.storage.endswith(("/a0", "/b0"))} == {
            "M"
        }
        accum = [d for d in war if d.storage.endswith("/c0")]
        assert accum and all(tasks[d.source.task].pipe == "FIX" for d in accum)
        assert all(d.source.first.evaluate(count) == count - 1 for d in accum)
        assert all(d.destination.position.evaluate(count) == 0 for d in accum)
    assert not summary.for_count(0)
    assert summary.visits < 1000


@pytest.mark.parametrize("count", [1, 2, 3, 7, 17])
def test_summary_cross_invocation_hazards_match_two_invocations(reduction_boundary, count):
    loop, roots, summary = reduction_boundary
    old = _frontiers(loop, roots, count)
    expected = set()
    for destination, touched in _native_cells(loop, roots, count):
        for (root, byte), modes in touched.items():
            state = old.get((root, byte))
            if state is None:
                continue
            if "R" in modes and state[2] is not None:
                expected.add((root, byte, "RAW", state[2], destination))
            if "W" in modes:
                if state[2] is not None:
                    expected.add((root, byte, "WAW", state[2], destination))
                for reader in state[3]:
                    expected.add((root, byte, "WAR", reader, destination))
                del old[root, byte]  # Later accesses are intra-invocation.
    actual = set()
    for dep in summary.invocation_dependencies():
        if dep.domain.contains(count):
            for byte in range(dep.start, dep.stop):
                for a in _members(dep.source, count):
                    for b in _members(dep.destination, count):
                        actual.add((dep.storage, byte, dep.hazard, a, b))
    assert actual == expected


def test_terminal_only_writer_preserves_all_input_reader_generations():
    source = N.from_source(
        """
        def finish(X: N.Tensor((8,), N.f32, access="read_write"), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 8))
                for i in N.Pipelined(n, stages=3):
                    with N.Stage(0, name="read"):
                        value = N.copy(X, x[i % 2, :])
                    with N.Stage(2, name="write"):
                        if i == n - 1:
                            N.copy(value, X)
    """,
        disjoint_external=True,
    )
    loop = extract_loop(lower_source(source.to_ir()))
    summary = summarize_reused_storage(loop, {"X"})
    for count in (0, 1, 2, 7, 17):
        _check_frontier(loop, {"X"}, count, summary)
    (cell,) = summary.for_count(1048576)
    assert len(cell.reads_before_write) == 1
    (span,) = cell.reads_before_write
    assert span.first.evaluate(1048576) == 0 and span.last.evaluate(1048576) == 1048575
    assert cell.first_write.position.end


def test_summary_refuses_unproven_parent_gm_frame_ownership(reduction_boundary):
    loop, _, _ = reduction_boundary
    with pytest.raises(CompileError, match="parent ownership"):
        summarize_reused_storage(loop, {"A"})


@pytest.mark.parametrize(
    "period,residues,width,distance,terminal",
    [
        (1, (0,), 1, 0, False),
        (1, (0,), 3, 3, False),
        (3, (0, 2), 4, 2, False),
        (5, (1, 4), 3, 7, False),
        (1, (0,), 1, 0, True),
        (3, (1,), 1, 2, True),
    ],
)
def test_event_boundary_rotation_matches_exact_set_wait_pairing(
    period, residues, width, distance, terminal
):
    channel = Channel(
        "example",
        "local",
        (("aic", "send"),),
        (("aic", "receive"),),
        distance,
        period,
        residues,
        tuple(range(width)),
        terminal,
    )
    summaries = tuple(_event_reuse(channel, ("aic", "MTE2", "MTE1")))
    for count in range(80):
        ordinal, used = 0, {}
        for g in range(max(0, count - distance)):
            if g % period not in residues:
                continue
            if terminal and g != count - distance - 1:
                continue
            event_id = ordinal % width
            used.setdefault(event_id, []).append((g, g + distance))
            ordinal += 1
        active = [item for item in summaries if item.domain.contains(count)]
        assert len(active) == len(used)
        for item in active:
            assert item.republished.position.evaluate(count) == used[item.event_id][0][0]
            assert item.consumed.position.evaluate(count) == used[item.event_id][-1][1]


def test_actual_terminal_channel_needs_cross_call_wait_consumption():
    artifact = compile(reduction_gemm.specialize(M=16, N=16, BK=16, GROUPS=1))
    loop = extract_loop(artifact.plan)
    resources = summarize_local_events(loop)
    terminal = [r for r in resources if r.republished.position.end]
    assert len(terminal) == 1
    assert terminal[0].bank == ("aic", "M", "FIX")
    assert terminal[0].consumed.task[1].startswith("export_")
    for count in (1, 2, 4096, 1048576):
        assert terminal[0].domain.contains(count)
        assert terminal[0].consumed.position.evaluate(count) == count - 1
    assert not terminal[0].domain.contains(0)


def test_independent_storage_summaries_do_not_multiply_ring_periods():
    loop = extract_loop(lower_source(independent_rings.to_ir()))
    roots = {"aiv0/inputs", "aiv0/outputs"}
    summary = summarize_reused_storage(loop, roots)
    assert {cell.domain.period for cell in summary.cells} == {5, 7}
    for count in (0, 1, 4, 5, 6, 7, 8, 17, 35):
        _check_frontier(loop, roots, count, summary)


def test_summary_keeps_holes_and_multiple_reader_milestones():
    kernel = Kernel("subviews", disjoint_external=True)
    n = kernel.runtime_size("n")
    source = kernel.buffer("source", "GM", 64, external=True, initialized=True)
    out = kernel.buffer("out", "GM", 64, external=True)
    ub = kernel.buffer("ub", "UB", 512)
    kernel.loop(n)
    tile = kernel.view(ub, 32, rows=2, row_stride=128, index_terms=((256, 1, 2),))
    kernel.task("load", "copy_rows", kernel.view(source, 32, rows=2, row_stride=32), tile)
    kernel.scalar_load("last_word", kernel.view(ub, 4, offset=128, index_terms=((256, 1, 2),)))
    kernel.task("store", "copy_rows", tile, kernel.view(out, 32, rows=2, row_stride=32))
    loop = extract_loop(kernel.finish())
    summary = summarize_reused_storage(loop, {"aiv0/ub"})
    for count in (0, 1, 2, 7, 17):
        _check_frontier(loop, {"aiv0/ub"}, count, summary)
    for cell in summary.for_count(7):
        assert all(
            byte % 256 < 32 or 128 <= byte % 256 < 160 for byte in range(cell.start, cell.stop)
        )
        if cell.start % 256 == 128:
            assert {reader.task[1] for reader in cell.pending_reads} == {"last_word", "store"}


def test_summary_budget_is_on_static_structure(reduction_boundary, monkeypatch):
    import xdsl_npu.region_summary as summaries

    loop, roots, _ = reduction_boundary
    monkeypatch.setattr(summaries, "MAX_SUMMARY_VISITS", 1)
    with pytest.raises(CompileError, match="static visit budget"):
        summarize_reused_storage(loop, roots)


def test_dynamic_stride_cannot_hide_a_nonuniform_factor():
    module = hierarchical_gemm.to_ir()
    program = module.body.block.first_op
    allocation = next(
        op
        for op in program.body.block.ops
        if isinstance(op, AllocationOp) and op.sym_name.data == "A"
    )
    # A forged allocation axis would turn every earlier row-major pitch into
    # owner-dependent arithmetic. A block coordinate is not a uniform extent.
    product = arith.MuliOp(program.body.block.args[0], allocation.dimensions[1])
    program.body.block.insert_op_before(product, allocation)
    allocation.operands = (allocation.dimensions[0], product.result)
    module.verify()
    with pytest.raises(CompileError, match="GM extent requires a declared uniform scalar"):
        lower_source(module)


def test_constant_child_summary_cannot_be_rebound_to_another_length():
    # A static leaf has a smaller address-validation domain. Its boundary
    # certificate cannot silently be reinterpreted for arbitrary runtime N.
    kernel = Kernel("constant_child")
    x = kernel.buffer("x", "UB", 128)
    kernel.loop(2, symbolic=True)
    kernel.task("initialize", "fill", kernel.view(x, 64, stride=64, slots=2))
    summary = summarize_reused_storage(extract_loop(kernel.finish()), {"aiv0/x"})
    assert summary.for_count(2)
    assert all(cell.domain.contains(2) for cell in summary.cells)
    with pytest.raises(CompileError, match="different constant child length"):
        summary.for_count(3)


def test_physical_fragment_budget_precedes_row_residue_allocation(monkeypatch):
    from xdsl_npu.model import Ref

    kernel = Kernel("many_rows", disjoint_external=True)
    n = kernel.runtime_size("n")
    source = kernel.buffer("source", "GM", 4095 * 32, external=True, initialized=True)
    ub = kernel.buffer("ub", "UB", 192 * 1024)
    kernel.loop(n)
    a = kernel.view(source, 32, rows=4095, row_stride=32)
    b = kernel.view(ub, 32, rows=4095, row_stride=32, stride=32, slots=840)
    kernel.task("copy", "copy_rows", a, b)

    def no_materialization(*args):
        raise AssertionError("fragment allocation preceded budget check")

    monkeypatch.setattr(Ref, "segments", no_materialization)
    with pytest.raises(CompileError, match="access-fragment budget exceeded"):
        extract_loop(kernel.finish())
