"""Bitwise UB datamove uses real V effects and the ordinary final verifier."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import StringAttr

from tests.fixtures.ub_copy import ub_copy
from tests.report_contract import semantic_report
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp
from xdsl_npu.loop_graph import channels, commands
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.model import Ref, Storage
from xdsl_npu.sync import verify_plan
from xdsl_npu.target.native import native_contract


@pytest.fixture(scope="module")
def artifact():
    return compile(ub_copy)


def test_public_bitwise_copy_has_one_native_template_and_real_v_effects(artifact):
    _, (loop, channel_map, _, _, _) = verify_loop(artifact.plan, _details=True)
    (data,) = loop.actors.values()
    load, copy, store = data.program.tasks
    assert [task.native for task in (load, copy, store)] == ["copy"] * 3
    assert [task.pipe for task in (load, copy, store)] == ["MTE2", "V", "MTE3"]
    assert copy.modes == ("R", "W") and not copy.implicit_completion
    assert copy.refs[0].storage.name != copy.refs[1].storage.name
    assert [ref.total_nbytes for ref in copy.refs] == [32, 32]
    assert sorted(storage.nbytes for storage in data.program.storages if not storage.external) == [
        64,
        64,
    ]
    tasks = {
        (actor, task.uid): task
        for actor, actor_data in loop.actors.items()
        for task in actor_data.program.tasks
    }
    assert {
        (tasks[channel.sources[0]].pipe, tasks[channel.destinations[0]].pipe, channel.distance)
        for channel in channel_map.values()
    } == {("MTE2", "V", 0), ("V", "MTE3", 0), ("V", "MTE2", 2), ("MTE3", "V", 2)}
    code = artifact.asc
    assert code.count("asc_copy_ub2ub(") == 1
    assert next(line for line in code.splitlines() if "asc_copy_ub2ub(" in line).endswith(
        "uint32_t(32));"
    )
    assert "asc_copy_ub2ub_sync" not in code
    assert "asc_add" not in code and "asc_mul" not in code and "asc_copy(" not in code
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert semantic_report(verify_plan(artifact.plan)) == semantic_report(artifact.report)


def byte_conflicts(loop, count):
    """Literal byte writers/readers in source effect order, without frontiers."""
    actor, data = next(iter(loop.actors.items()))
    writers, readers, conflicts = {}, {}, set()
    for generation in range(count):
        for task in data.program.tasks:
            current = task.uid, generation
            for ref, mode in zip(task.refs, task.modes):
                for lo, hi in ref.segments(generation):
                    for byte in range(lo, hi):
                        key = loop.root(actor, ref), byte
                        if "R" in mode:
                            assert key in writers or ref.storage.initialized
                            if key in writers:
                                conflicts.add((writers[key], current))
                            readers.setdefault(key, set()).add(current)
                        if "W" in mode:
                            if key in writers:
                                conflicts.add((writers[key], current))
                            conflicts.update((reader, current) for reader in readers.get(key, ()))
                            writers[key], readers[key] = current, set()
    return {pair for pair in conflicts if pair[0] != pair[1]}


def check_actual_access_order(module, count):
    loop = extract_loop(module)
    channel_map = channels(loop)
    sequences, _, _ = commands(loop, channel_map)
    ids, reach = finite_hardware_dag(loop, sequences, channel_map, count)
    native = {
        command.task.uid: command
        for sequence in sequences.values()
        for command in sequence
        if command.kind == "task"
    }
    for (source, sg), (destination, dg) in byte_conflicts(loop, count):
        a = ids[native[source].index, sg, "end"]
        b = ids[native[destination].index, dg, "begin"]
        assert reach[a] & (1 << b), (source, sg, destination, dg)


@pytest.mark.parametrize("count", (0, 1, 2, 3, 7, 17))
def test_actual_commands_order_every_byte_reader_writer_and_slot_reuse(artifact, count):
    check_actual_access_order(artifact.plan, count)


@pytest.mark.parametrize(
    "source_pipe,destination_pipe,hazard",
    (("MTE2", "V", "RAW"), ("V", "MTE3", "RAW"), ("V", "MTE2", "WAR"), ("MTE3", "V", "WAR")),
)
def test_removing_actual_ready_or_release_is_rejected_by_two_checkers(
    artifact, source_pipe, destination_pipe, hazard
):
    module = artifact.plan
    _, (loop, channel_map, _, _, _) = verify_loop(module, _details=True)
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    (name,) = (
        name
        for name, channel in channel_map.items()
        if tasks[channel.sources[0]].pipe == source_pipe
        and tasks[channel.destinations[0]].pipe == destination_pipe
    )
    for op in tuple(module.walk()):
        if (
            isinstance(op, LoopSyncOp)
            and op.channel.data == name
            or isinstance(op, LoopChannelOp)
            and op.sym_name.data == name
        ):
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match=f"missing loop {hazard}"):
        verify_plan(module)
    with pytest.raises(AssertionError):
        check_actual_access_order(module, 7)


FILL = """
def fill_copy(O:N.Tensor((n,8),N.f32,access="write"),n:N.int64):
    with N.Kernel():
        source=N.alloc_ub((2,8),N.f32)
        destination=N.alloc_ub((2,8),N.f32)
        for tile in N.Pipelined(n,stages=3):
            with N.Stage(0,name="fill"):
                N.fill(source[tile%2,:],-0.0)
            with N.Stage(1,name="copy"):
                N.copy(source[tile%2,:],destination[tile%2,:])
            with N.Stage(2,name="store"):
                N.copy(destination[tile%2,:],O[tile,:])
"""


def test_public_fill_copy_store_needs_a_real_v_completion_fence():
    artifact = compile(N.from_source(FILL, disjoint_external=True))
    module = artifact.plan
    _, (loop, _, sequences, _, _) = verify_loop(module, _details=True)
    (data,) = loop.actors.values()
    assert [task.pipe for task in data.program.tasks] == ["V", "V", "MTE3"]
    barriers = [
        command
        for sequence in sequences.values()
        for command in sequence
        if command.kind == "barrier"
    ]
    assert barriers and all(command.task.pipe == "V" for command in barriers)
    assert "asc_copy_ub2ub(" in artifact.asc and "float(-0.0)" in artifact.asc
    acquisition = next(command for command in barriers if command.task.native == "copy")
    acquisition.operation.parent.erase_op(acquisition.operation)
    with pytest.raises(CompileError, match="missing loop RAW"):
        verify_plan(module)


def test_same_root_statically_disjoint_views_pass_normal_dsl_and_final_sync():
    text = (
        FILL.replace("source=N.alloc_ub((2,8),N.f32)", "source=N.alloc_ub((4,8),N.f32)")
        .replace("destination=N.alloc_ub((2,8),N.f32)", "")
        .replace("destination[tile%2,:]", "source[tile%2+2,:]")
    )
    artifact = compile(N.from_source(text, disjoint_external=True))
    loop = extract_loop(artifact.plan)
    task = next(
        task
        for data in loop.actors.values()
        for task in data.program.tasks
        if task.native == "copy" and task.pipe == "V"
    )
    assert task.refs[0].storage is task.refs[1].storage
    assert task.refs[1].address(0) - task.refs[0].address(0) == 64
    assert semantic_report(verify_plan(artifact.plan)) == semantic_report(artifact.report)
    assert "asc_copy_ub2ub(" in artifact.asc


@pytest.mark.parametrize("destination", ("0:16", "8:24"))
def test_exact_and_shifted_overlaps_are_refused_without_memmove_assumptions(destination):
    text = f"""
def overlap(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((32,),N.f32)
        for tile in N.Pipelined(n,stages=1):
            N.fill(x,1.0)
            N.copy(x[0:16],x[{destination}])
"""
    with pytest.raises(CompileError, match="UB-to-UB copy requires provably disjoint"):
        compile(N.from_source(text))


def test_generation_dependent_aliases_are_not_assumed_disjoint():
    text = """
def changing(n:N.int64):
    with N.Kernel():
        x=N.alloc_ub((3,8),N.f32)
        for tile in N.Pipelined(n,stages=1):
            N.fill(x,1.0)
            N.copy(x[tile%2,:],x[tile%3,:])
"""
    with pytest.raises(CompileError, match="overlapping or changing aliases"):
        compile(N.from_source(text))


@pytest.mark.parametrize("dtype,columns", (("f16", 16), ("bf16", 16), ("f32", 8)))
def test_public_copy_preserves_dtype_and_uses_bytes_for_all_supported_types(dtype, columns):
    text = (
        ub_copy.source.replace("N.f32", "N." + dtype)
        .replace("(n, 8)", f"(n, {columns})")
        .replace("(2, 8)", f"(2, {columns})")
    )
    artifact = compile(N.from_source(text, disjoint_external=True))
    loop = extract_loop(artifact.plan)
    task = next(
        task for data in loop.actors.values() for task in data.program.tasks if task.pipe == "V"
    )
    assert [ref.dtype for ref in task.refs] == [dtype, dtype]
    assert [ref.total_nbytes for ref in task.refs] == [32, 32]
    assert "uint32_t(32)" in next(
        line for line in artifact.asc.splitlines() if "asc_copy_ub2ub(" in line
    )
    assert "asc_add" not in artifact.asc and "asc_float2" not in artifact.asc


def physical_copy(*, length=32, source_offset=0, destination_offset=0):
    k = Kernel("physical_ub_copy")
    source = k.buffer("source", "UB", 128)
    destination = k.buffer("destination", "UB", 128)
    k.loop(1, symbolic=True)
    k.task("fill", "fill", k.view(source, 128), scalar=1.0)
    k.copy(
        "copy",
        k.view(source, length, offset=source_offset),
        k.view(destination, length, offset=destination_offset),
    )
    return k.finish()


@pytest.mark.parametrize("offsets", ((4, 0), (0, 4)))
def test_actual_ir_checks_both_native_origins_before_emission(offsets):
    with pytest.raises(CompileError, match="native origin alignment"):
        compile(physical_copy(source_offset=offsets[0], destination_offset=offsets[1]))


def test_partial_blocks_are_refused_instead_of_truncating_bytes():
    with pytest.raises(CompileError, match="32-byte aligned"):
        compile(physical_copy(length=16))


def test_burst_limit_and_equal_dtype_footprints_are_checked():
    storage = Storage("a", "UB", 65536 * 32, "f32", "linear", False, True)
    other = replace(storage, name="b")
    source = Ref(storage, ((0, 0, 0, 1),), 65535 * 32, "f32", "linear")
    destination = replace(source, storage=other)
    assert native_contract("copy", (source, destination), ()) == ("V", ("R", "W"))
    for refs, diagnostic in (
        (
            (replace(source, nbytes=65536 * 32), replace(destination, nbytes=65536 * 32)),
            "fit burst length",
        ),
        ((source, replace(destination, dtype="bf16")), "cannot silently convert dtype"),
        ((source, replace(destination, nbytes=32)), "equal native footprints"),
        ((source, replace(destination, tail_nbytes=32)), "fixed complete 32-byte blocks"),
        ((replace(source, tail_nbytes=32), destination), "fixed complete 32-byte blocks"),
    ):
        with pytest.raises(CompileError, match=diagnostic):
            native_contract("copy", refs, ())


def test_actual_ub_allocations_cannot_execute_on_an_aic(artifact):
    module = artifact.plan
    kernel = module.body.block.first_op
    kernel.attributes["core"] = StringAttr("aic")
    with pytest.raises(CompileError, match="UB capacity/owner violation"):
        verify_plan(module)
