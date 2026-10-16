"""Actual SSA accesses and literal issue order, without completion closures."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from examples.gemm.batched import batched_matmul
from examples.vector.prefix import scale_prefix
from examples.vector.scale import scale
from tests.fixtures.pipelines import cube_pipeline, mixed_pipeline
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis import native
from xdsl_npu.analysis.frontier import Clock, PrefixRank
from xdsl_npu.analysis.native import AccessMode, Register, extract_native_tasks
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.physical import BufferOp, FixConfigOp, Load3DConfigOp, TaskOp
from xdsl_npu.load3d import select_load3d
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import Program, Ref
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def byte_set(ref, generation, *, core=0):
    return {byte for lo, hi in ref.segments(generation, core) for byte in range(lo, hi)}


@pytest.fixture(scope="module")
def scale_effects():
    module = lower_source(scale.to_ir())
    loop = extract_loop(module)
    return loop, extract_native_tasks(loop, clock=Clock("scale physical step"))


def test_scale_effects_keep_actual_operands_roots_and_exact_double_buffer_bytes(scale_effects):
    loop, tasks = scale_effects
    data = next(iter(loop.actors.values()))
    actual = tuple(op for op in data.stream.walk() if isinstance(op, TaskOp))
    roots = {
        op.sym_name.data: op.ref for op in data.kernel.body.block.ops if isinstance(op, BufferOp)
    }
    assert [task.native for task in tasks] == ["copy", "muls", "copy"]
    assert [task.engine for task in tasks] == [Engine.MTE2, Engine.V, Engine.MTE3]
    assert all(task.actor is Actor.AIV0 for task in tasks)
    for index, task in enumerate(tasks):
        assert task.operation is actual[index]
        assert task.coordinate is data.stream.body.block.args[0]
        assert task.consumer_cut.operation is task.capture_cut.operation is actual[index]
        assert task.consumer_cut.milestone is Milestone.ADMISSION
        assert task.capture_cut.milestone is Milestone.NATIVE_END
        for number, access in enumerate(task.accesses):
            assert access.operand_index == number
            assert access.operand is actual[index].refs[number]
            assert access.ref is data.program.tasks[index].refs[number]
    for generation in (0, 1, 2, 7):
        external = set(range(generation * 256, (generation + 1) * 256))
        slot = set(range((generation % 2) * 256, (generation % 2 + 1) * 256))
        expected = (
            ((roots["X"], external, "R"), (roots["inputs"], slot, "W")),
            ((roots["inputs"], slot, "R"), (roots["outputs"], slot, "W")),
            ((roots["outputs"], slot, "R"), (roots["Y"], external, "W")),
        )
        for task, accesses in zip(tasks, expected, strict=True):
            for access, (root, addresses, mode) in zip(task.accesses, accesses, strict=True):
                assert access.root is root
                assert access.mode == mode
                assert byte_set(access.ref, generation) == addresses


@pytest.mark.parametrize("count", [0, 1, 10**9])
def test_cube_templates_do_not_expand_static_count_and_preserve_each_storage_level(count):
    loop = extract_loop(cube_pipeline(count=count))
    tasks = extract_native_tasks(loop, clock=Clock("cube physical step"))
    assert len(tasks) == 6
    assert [task.native for task in tasks] == ["copy"] * 4 + ["mad16", "fix16"]
    assert [task.engine for task in tasks] == [Engine.MTE2] * 2 + [Engine.MTE1] * 2 + [
        Engine.M,
        Engine.FIX,
    ]
    assert all(task.actor is Actor.AIC for task in tasks)
    mad, export = tasks[4:]
    assert not mad.implicit_completion
    assert mad.modes_for(2) == (AccessMode.READ, AccessMode.READ, AccessMode.WRITE)
    assert [access.ref.layout for access in mad.accesses] == ["ZZ", "ZN", "NZ"]
    assert [access.ref.total_nbytes for access in mad.accesses] == [512, 512, 1024]
    assert mad.accesses[2].root is export.accesses[0].root
    for generation in (0, 1, 2, 5):
        for task in tasks:
            for access in task.accesses:
                width = 1024 if access.ref.storage.bank == "L0C" else access.ref.nbytes
                offset = (
                    generation * width if access.ref.storage.external else generation % 2 * width
                )
                assert byte_set(access.ref, generation) == set(range(offset, offset + width))


def test_physical_rank_matches_literal_dispatch_and_binds_the_actual_operation():
    module = cube_pipeline(count=5)
    actual = tuple(op for op in module.walk() if isinstance(op, TaskOp))
    for index, operation in enumerate(actual):
        operation.attributes["uid"] = StringAttr(f"unrelated_name_{index}")
        operation.attributes["issue_order"] = IntAttr(100 - 7 * index)
    loop = extract_loop(module)
    clock = Clock("declared physical step")
    tasks = extract_native_tasks(loop, clock=clock)
    literal = sorted(
        (generation + operation.stage.data, operation.issue_order.data, index, operation)
        for generation in range(5)
        for index, operation in enumerate(actual)
    )
    ranked = sorted(
        (generation + task.rank.offset, task.rank.lexical, task.operation)
        for generation in range(5)
        for task in tasks
    )
    assert [item[-1] for item in ranked] == [item[-1] for item in literal]
    assert [task.operation for task in tasks] == list(actual)
    assert [task.lexical for task in tasks] == list(reversed(range(6)))
    # Source mad(g-2) precedes an operand overwrite. Substitute consumer
    # g=j-stage into the source rank explicitly; logical g is not global j.
    source, consumer, distance, physical_step = tasks[4], tasks[2], 2, 17
    required = source.cut(Milestone.NATIVE_END, generation_offset=-consumer.stage - distance).rank
    assert physical_step + required.offset == 16
    assert required == PrefixRank(clock, -1, source.lexical)


def test_native_end_is_an_anchor_and_cannot_manufacture_a_retirement_capture(scale_effects):
    _, (load, _, _) = scale_effects
    with pytest.raises(CompileError, match="cannot establish a retirement capture") as caught:
        load.cut(Milestone.RETIREMENT_CAPTURE)
    assert caught.value.kind is ErrorKind.UNSUPPORTED
    with pytest.raises(CompileError, match="cannot establish a retirement capture"):
        replace(load.capture_cut, milestone=Milestone.IMPLICIT_RETIREMENT)
    with pytest.raises(CompileError, match="across frontier clocks"):
        load.rank.compare(PrefixRank(Clock(load.clock.name), load.stage, load.lexical))


def test_one_decoder_contract_and_no_view_or_finite_replay_in_clock_binding(monkeypatch):
    module = lower_source(scale.to_ir())
    calls = []
    original = native.native_contract

    def contract(*arguments):
        calls.append(arguments)
        return original(*arguments)

    monkeypatch.setattr(native, "native_contract", contract)
    loop = extract_loop(module)
    assert len(calls) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("owned native effects cannot replay admission or finite instances")

    monkeypatch.setattr(native, "native_task_contract", forbidden)
    monkeypatch.setattr(Ref, "segments", forbidden)
    monkeypatch.setattr(Program, "instances", forbidden)
    first = extract_native_tasks(loop, clock=Clock("first explicit binding"))
    second = extract_native_tasks(loop, clock=Clock("second explicit binding"))
    assert len(calls) == 3
    assert all(left.accesses is right.accesses for left, right in zip(first, second, strict=True))
    assert first[0].operation is second[0].operation
    assert first[0].clock is not second[0].clock


def test_pitched_scratch_keeps_holes_period_divisor_and_actual_allocation_identity():
    kernel = Kernel("pitched_effects", disjoint_external=True)
    count = kernel.runtime_size("tiles")
    source = kernel.buffer(
        "source", "GM", 0, external=True, initialized=True, extent_count=count, count_scale=256
    )
    scratch = kernel.buffer("scratch", "UB", 512)
    kernel.loop(count)
    src = kernel.view(source, 32, stride=256, rows=2, row_stride=96)
    dst = kernel.view(
        scratch,
        32,
        stride=256,
        slots=2,
        divisor=2,
        index_terms=((32, 1, 2),),
        rows=2,
        row_stride=96,
    )
    kernel.task("pitched", "copy_rows", src, dst)
    loop = extract_loop(kernel.finish())
    (task,) = extract_native_tasks(loop, clock=Clock("pitched physical step"))
    access = task.accesses[1]
    assert access.root is scratch
    assert access.ref.terms == ((0, 256, 2, 2), (0, 32, 2, 1))
    assert (access.ref.rows, access.ref.nbytes, access.ref.row_stride) == (2, 32, 96)
    for generation in range(8):
        offset = 256 * ((generation // 2) % 2) + 32 * (generation % 2)
        expected = set(range(offset, offset + 32)) | set(range(offset + 96, offset + 128))
        assert byte_set(access.ref, generation) == expected
        assert offset + 64 not in byte_set(access.ref, generation)


def test_prefix_extents_and_owner_terms_remain_the_original_decoded_refs():
    loop = extract_loop(lower_source(scale_prefix.to_ir()))
    tasks = extract_native_tasks(loop, clock=Clock("prefix physical step"))
    data = next(iter(loop.actors.values()))
    prefixes = []
    for task, decoded in zip(tasks, data.program.tasks, strict=True):
        for access, ref in zip(task.accesses, decoded.refs, strict=True):
            assert access.ref is ref
            if ref.prefix_extent is not None:
                prefixes.append(ref)
    assert prefixes
    assert all(ref.prefix_elements > 0 for ref in prefixes)


def test_accumulator_rmw_is_kept_even_when_it_has_an_undefined_incoming_value():
    module = cube_pipeline(count=3)
    operation = next(
        op for op in module.walk() if isinstance(op, TaskOp) and op.native.data == "mad16"
    )
    operation.attributes["params"] = ArrayAttr([IntAttr(0)])
    loop = extract_loop(module)
    mad = next(
        task
        for task in extract_native_tasks(loop, clock=Clock("rmw"))
        if task.operation is operation
    )
    assert mad.accesses[2].mode is AccessMode.READ_WRITE
    assert mad.modes_for(0)[2] is AccessMode.READ_WRITE
    # Effect extraction does not initialize the accumulator or discharge RAW.
    from xdsl_npu.loop_model import loop_dependencies

    with pytest.raises(CompileError, match="undefined"):
        loop_dependencies(loop)


def test_explicit_spr_reads_keep_the_actual_initializer_and_separate_resources():
    source = batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1, TRANSPOSE_B=False)
    module = select_load3d(lower_source(source.to_ir()))
    loop = extract_loop(module)
    tasks = extract_native_tasks(loop, clock=Clock("load3d"))
    load3d = next(op for op in module.walk() if isinstance(op, Load3DConfigOp))
    fix = next(op for op in module.walk() if isinstance(op, FixConfigOp))
    for task in tasks:
        if task.native in {"load3d_a", "load3d_b_nn"}:
            specific = Register.FMATRIX_A if task.native == "load3d_a" else Register.FMATRIX_B
            assert {access.resource for access in task.registers} == {
                specific,
                Register.L13D_REPEAT,
                Register.L13D_PADDING,
                Register.L13D_SIZE,
            }
            assert all(access.configuration is load3d for access in task.registers)
        elif task.native == "fix":
            assert len(task.registers) == 1
            assert task.registers[0].resource is Register.ND_PARA
            assert task.registers[0].configuration is fix
        for access in task.registers:
            assert access.operation is task.operation
            assert access.actor is Actor.AIC
            assert access.mode is AccessMode.READ


def test_mask_setup_and_masked_partial_reads_keep_their_actual_state_and_byte_ranges():
    kernel = Kernel("masked_partials")
    source = kernel.buffer("partials", "UB", 64)
    destination = kernel.buffer("statistics", "UB", 32)
    kernel.loop(4, symbolic=True)
    kernel.task("init", "fill", kernel.view(source, 64))
    operation = kernel.task(
        "merge",
        "merge_sum_packed",
        kernel.view(source, 12, rows=2, row_stride=32),
        kernel.view(destination, 8),
        params=(2, 3),
    )
    loop = extract_loop(kernel.finish())
    merge = next(
        task
        for task in extract_native_tasks(loop, clock=Clock("mask"))
        if task.operation is operation
    )
    assert merge.accesses[0].ref.segments(0) == ((0, 12), (32, 44))
    assert {access.resource for access in merge.registers} == {
        Register.VECTOR_MASK,
        Register.VECTOR_MASK_MODE,
    }
    assert all(
        access.mode is AccessMode.WRITE
        and access.engine is Engine.S
        and access.configuration is None
        and access.operation is operation
        for access in merge.registers
    )


def test_external_physical_input_is_readmitted_before_bad_contracts_can_be_used():
    module = cube_pipeline(count=3)
    operation = next(
        op for op in module.walk() if isinstance(op, TaskOp) and op.native.data == "mad16"
    )
    extract_loop(module)
    operation.attributes["params"] = ArrayAttr([IntAttr(7)])
    with pytest.raises(CompileError, match="MAD v0.1"):
        extract_loop(module)


def test_nonlocal_and_finite_program_inputs_have_an_explicit_unsupported_diagnostic():
    mixed = extract_loop(mixed_pipeline(count=2))
    for value in (mixed, Program("finite", "aiv", 2, (), (), False)):
        with pytest.raises(CompileError, match="private local loop") as caught:
            extract_native_tasks(value, clock=Clock("unsupported"))
        assert caught.value.kind is ErrorKind.UNSUPPORTED
