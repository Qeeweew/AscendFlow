"""Explicit Cube operands, borrowed values, and independent finite hazards."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr

from examples.gemm.batched import batched_matmul
from tests.report_contract import semantic_report
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile, verify_plan
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import parse
from xdsl_npu.dialects.physical import BufferOp, LoopChannelOp, LoopSyncOp, TaskOp, ViewOp
from xdsl_npu.dialects.source import CopyOp, MatmulOp, PipelineOp
from xdsl_npu.dialects.source import ViewOp as SourceViewOp
from xdsl_npu.dialects.types import TensorType
from xdsl_npu.frontend import FrontendError
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.lowering import lower_source


@pytest.fixture(scope="module", params=[False, True])
def artifact(request):
    return compile(batched_matmul.specialize(TRANSPOSE_B=request.param))


def test_explicit_operands_are_not_allocated_or_copied_implicitly(artifact):
    loop = extract_loop(artifact.plan)
    tasks = loop.actors["aic"].program.tasks
    assert [t.native for t in tasks] in [
        ["nd2nz", "nd2nz", "load2d_a", "load2d_b_nn", "mad", "fix"],
        ["nd2nz", "nd2nz", "load2d_a", "load2d_b_nt", "mad", "fix"],
    ]
    assert {s.name for s in loop.actors["aic"].program.storages} == {
        "A",
        "B",
        "C",
        "a1",
        "b1",
        "a0",
        "b0",
        "c0",
    }
    assert artifact.report["storage_periods"]["aic/a1"] == 3
    assert artifact.report["storage_periods"]["aic/a0"] == 2
    assert semantic_report(verify_plan(parse(artifact.plan_ir))) == semantic_report(artifact.report)
    assert artifact.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert artifact.asc.count("asc_mmad(") == 1


def test_l1_l0_and_accumulator_releases_have_distinct_last_readers(artifact):
    loop = extract_loop(artifact.plan)
    tasks = {task.uid: task for task in loop.actors["aic"].program.tasks}
    war = [o for o in loop_dependencies(loop) if o.hazard == "WAR"]
    for storage, natives, distance in [
        ("a1", {"load2d_a"}, 3),
        ("b1", {"load2d_b_nn", "load2d_b_nt"}, 3),
        ("a0", {"mad"}, 2),
        ("b0", {"mad"}, 2),
        ("c0", {"fix"}, 2),
    ]:
        obligations = [o for o in war if o.storage == "aic/" + storage]
        assert obligations
        assert {o.distance for o in obligations} == {distance}
        assert {tasks[o.source[1]].native for o in obligations} <= natives
    plan = artifact.plan
    by_uid = {t.uid.data: t for t in plan.walk() if isinstance(t, TaskOp)}
    releases = [
        c
        for c in plan.walk()
        if isinstance(c, LoopChannelOp)
        and by_uid[c.sources.data[0].data.split("/")[1]].native.data.startswith("load2d")
        and by_uid[c.destinations.data[0].data.split("/")[1]].native.data == "nd2nz"
    ]
    assert releases
    for channel in releases:
        for op in list(plan.walk()):
            if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
                op.parent.erase_op(op)
        channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing loop WAR"):
        verify_plan(plan)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
def test_cube_raw_war_waw_against_finite_byte_and_hardware_oracles(artifact, count):
    _, (loop, channels, commands, _, _) = verify_loop(artifact.plan, _details=True)
    ids, reach = finite_hardware_dag(loop, commands, channels, count)
    native = {c.task.uid: c for c in commands["aic"] if c.kind == "task"}
    # Physical IR is in source effect order. Read actual offsets/rows/terms;
    # do not call production address or dependence routines for this oracle.
    tasks = [op for op in artifact.plan.walk() if isinstance(op, TaskOp)]
    writers, readers, conflicts = {}, {}, set()
    for g in range(count):
        for task in tasks:
            current = task.uid.data, g
            modes = ("R", "R", "W") if task.native.data == "mad" else ("R", "W")
            for value, mode in zip(task.refs, modes):
                view = value.owner
                assert isinstance(view, ViewOp)
                root = view.source.owner
                assert isinstance(root, BufferOp)
                address = view.offset.data
                for term in view.index_terms:
                    coefficient, divisor, modulus = (v.data for v in term)
                    index = g // divisor
                    address += coefficient * (index % modulus if modulus else index)
                # Test group0; production ownership checks validate all groups.
                for row in range(view.rows.data):
                    begin = address + row * view.row_stride.data
                    for byte in range(begin, begin + view.nbytes.data):
                        key = root.sym_name.data, byte
                        if mode == "R":
                            assert key in writers or root.initialized.data
                            if key in writers:
                                conflicts.add((writers[key], current))
                            readers.setdefault(key, set()).add(current)
                        else:
                            if key in writers:
                                conflicts.add((writers[key], current))
                            conflicts.update((reader, current) for reader in readers.get(key, ()))
                            writers[key], readers[key] = current, set()
    for (a, ag), (b, bg) in conflicts:
        if (a, ag) != (b, bg):
            assert reach[ids[native[a].index, ag, "end"]] & (1 << ids[native[b].index, bg, "begin"])


def test_explicit_accumulation_retains_two_mads_and_short_mad_completion():
    result = compile(batched_matmul.specialize(M=16, N=32, K=48, ACCUMULATE=True))
    loop = extract_loop(result.plan)
    mads = [t for t in loop.actors["aic"].program.tasks if t.native == "mad"]
    assert [t.params for t in mads] == [(16, 32, 48, 1), (16, 32, 48, 0)]
    assert all(not t.implicit_completion for t in mads)
    assert "asc_sync_pipe(PIPE_M)" in result.asc
    assert result.asc.count("asc_mmad(") == 2


def test_cube_bf16_uses_the_same_explicit_storage_semantics():
    result = compile(batched_matmul.specialize(DTYPE=N.bf16, TRANSPOSE_B=False))
    assert "__cbuf__ bfloat16_t" in result.asc
    assert result.report["native_templates"] == 6


def test_partial_fractal_view_is_not_mistaken_for_a_contiguous_row_slice():
    source = batched_matmul.to_ir()
    view = next(
        op
        for op in source.walk()
        if isinstance(op, SourceViewOp) and op.source.owner.sym_name.data == "a1"
    )
    # Keep the forged submatrix inside its logical allocation, so rejection
    # must come from the fractal complete-plane contract rather than bounds.
    allocation = view.source.owner
    typ = allocation.ref.type
    shape = [dimension.data for dimension in typ.shape]
    shape[-2] *= 2
    allocation.ref._type = TensorType(
        ArrayAttr(IntAttr(dimension) for dimension in shape), typ.dtype, typ.memory, typ.layout
    )
    row = arith.ConstantOp.from_int_and_width(4, IndexType())
    view.parent.insert_op_before(row, view)
    origins = list(view.origins)
    origins[-2] = row.result
    view.operands = (view.source, *origins)
    source.verify()
    with pytest.raises(CompileError, match="complete matrix plane"):
        lower_source(source)


def test_explicit_b_operand_tracks_its_own_numerical_writer():
    # Destroy the B logical producer identity while keeping same typed shape.
    # Use square A/B for a forged same-shape dependency, then let the physical
    # producer/consumer check reject it instead of trusting SSA alone.
    source = batched_matmul.specialize(M=64, N=64, K=64).to_ir()
    matmul = next(op for op in source.walk() if isinstance(op, MatmulOp))
    matmul.operands = (*matmul.operands[:4], matmul.dependencies[0], matmul.dependencies[0])
    with pytest.raises(CompileError, match="overwritten"):
        lower_source(source)


def test_ast_rejects_hidden_l0_recipe_and_runtime_conditional_expressions():
    source = batched_matmul.source.replace("T.matmul(aa, bb, accum=", "T.gemm(aa, bb, ")
    with pytest.raises(FrontendError, match="unknown DSL operation gemm"):
        replace(batched_matmul, source=source).to_ir()
    source = batched_matmul.source.replace("N if TRANSPOSE_B else K", "N if tiles > 0 else K")
    with pytest.raises(FrontendError, match="constexpr boolean"):
        replace(batched_matmul, source=source).to_ir()


def test_transpose_flag_does_not_license_other_dma_routes():
    source = batched_matmul.to_ir()
    copy = next(op for op in source.walk() if isinstance(op, CopyOp))
    copy.attributes["transpose"] = IntAttr(1)
    from xdsl.utils.exceptions import VerifyException

    with pytest.raises(VerifyException, match="L1→L0B"):
        source.verify()


def test_first_mad_cannot_accumulate_uninitialized_local_storage():
    source = batched_matmul.source.replace("initialize=True", "initialize=False")
    with pytest.raises(CompileError, match="undefined first read"):
        compile(replace(batched_matmul, source=source))


@pytest.mark.parametrize("sequence", [False, True])
def test_serialized_ast_cannot_bypass_explicit_operand_contract(sequence):
    source = batched_matmul.to_ir()
    if sequence:
        pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
        pipeline.parent.insert_op_after(pipeline.clone(), pipeline)
    matmul = next(op for op in source.walk() if isinstance(op, MatmulOp))
    a1 = next(
        op.ref
        for op in source.walk()
        if isinstance(op, SourceViewOp) and op.source.owner.sym_name.data == "a1"
    )
    b1 = next(
        op.ref
        for op in source.walk()
        if isinstance(op, SourceViewOp) and op.source.owner.sym_name.data == "b1"
    )
    matmul.operands = (a1, b1, *matmul.operands[2:])
    with pytest.raises(CompileError, match="implicit L1 recipes are legacy"):
        lower_source(source)
