"""Runtime prefix proofs tested against independently enumerated byte accesses."""

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntAttr

from examples.vector.prefix import scale_prefix
from tests.test_loop_sync import finite_hardware_dag
from xdsl_npu import compile, verify_plan
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import (
    BufferOp,
    LoopChannelOp,
    LoopSyncOp,
    RuntimeSizeOp,
    StreamOp,
    TaskOp,
    ViewOp,
)
from xdsl_npu.frontend import from_source
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.model import Program


@pytest.fixture(scope="module")
def artifact():
    return compile(scale_prefix)


def scalar(value, arguments):
    """Independent SSA interpreter for this fixture, not production scalar.py."""
    op = value.owner
    if isinstance(op, RuntimeSizeOp):
        return arguments[op.sym_name.data]
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    a, b = (scalar(operand, arguments) for operand in op.operands)
    if isinstance(op, arith.MinSIOp):
        return min(a, b)
    assert isinstance(op, arith.CeilDivUIOp)
    return (a + b - 1) // b


@pytest.mark.parametrize(
    "n,m",
    [
        (0, 0),
        (0, 80),
        (80, 0),
        (8, 65),
        (65, 8),
        (56, 64),
        (64, 64),
        (72, 80),
        (80, 72),
        (120, 129),
        (128, 136),
        (136, 145),
        (248, 256),
        (264, 280),
    ],
)
def test_exact_prefix_conflicts_have_real_hardware_completion_paths(artifact, n, m):
    plan = artifact.plan
    arguments = {"n": n, "m": m}
    stream = next(op for op in plan.walk() if isinstance(op, StreamOp))
    count = scalar(stream.limit, arguments)
    _, (loop, channel_map, commands, _, _) = verify_loop(plan, _details=True)
    ids, reach = finite_hardware_dag(loop, commands, channel_map, count)
    command_by_uid = {
        command.task.uid: command for command in commands["aiv0"] if command.kind == "task"
    }
    tasks = sorted(
        (op for op in stream.body.block.ops if isinstance(op, TaskOp)),
        key=lambda op: (op.logical_phase.data, op.issue_order.data),
    )
    writers, readers, conflicts = {}, {}, set()
    for generation in range(count):
        for task in tasks:
            current = (task.uid.data, generation)
            # Fixture native semantics are direct, audited DMA and Vector mul.
            assert task.native.data in {"copy", "muls"}
            for value, mode in zip(task.refs, ("R", "W")):
                view = value.owner
                assert isinstance(view, ViewOp)
                root = view.source.owner
                assert isinstance(root, BufferOp)
                origin = view.offset.data
                for term in view.index_terms:
                    coefficient, divisor, modulus = (item.data for item in term)
                    coordinate = generation // divisor
                    origin += coefficient * (coordinate % modulus if modulus else coordinate)
                valid = min(
                    view.prefix_elements.data,
                    scalar(view.extent_count, arguments) - generation * view.prefix_elements.data,
                )
                assert 0 < valid <= 64
                size = (
                    scalar(root.extent_count, arguments) * root.count_scale.data
                    if root.extent_count is not None
                    else root.nbytes.data
                )
                for byte in range(origin, origin + 4 * valid):
                    assert 0 <= byte < size, (root.sym_name.data, byte, size)
                    key = root.sym_name.data, byte
                    if mode == "R":
                        assert key in writers or root.initialized.data, (key, current)
                        if key in writers:
                            conflicts.add((writers[key], current, "RAW"))
                        readers.setdefault(key, set()).add(current)
                    else:
                        if key in writers:
                            conflicts.add((writers[key], current, "WAW"))
                        conflicts.update(
                            (reader, current, "WAR") for reader in readers.get(key, ())
                        )
                        writers[key] = current
                        readers[key] = set()
    for (a, ag), (b, bg), hazard in conflicts:
        if (a, ag) == (b, bg):
            continue
        source = ids[command_by_uid[a].index, ag, "end"]
        destination = ids[command_by_uid[b].index, bg, "begin"]
        assert reach[source] & (1 << destination), (n, m, a, ag, b, bg, hazard)
    if count >= 3:
        assert any(hazard == "WAR" and ag != bg for (_, ag), (_, bg), hazard in conflicts)


def test_full_reader_after_clipped_writer_is_rejected(artifact):
    plan = artifact.plan
    task = next(op for op in plan.walk() if isinstance(op, TaskOp) and op.native.data == "muls")
    task.refs[0].owner.attributes.pop("prefix_elements")
    with pytest.raises(CompileError, match="count-dependent|full and clipped"):
        verify_plan(plan)


def test_prefix_cannot_be_rebound_to_another_extent(artifact):
    plan = artifact.plan
    extent = next(
        op for op in plan.walk() if isinstance(op, RuntimeSizeOp) and op.sym_name.data == "n"
    )
    view = next(op for op in plan.walk() if isinstance(op, ViewOp))
    view.operands = (*view.operands[:2], extent.value)
    with pytest.raises(CompileError, match="disagrees|equal cell"):
        verify_plan(plan)


def test_offset_partial_cell_alias_cannot_reuse_whole_cell_proof(artifact):
    plan = artifact.plan
    view = next(
        op
        for op in plan.walk()
        if isinstance(op, ViewOp) and op.source.owner.sym_name.data == "inputs"
    )
    view.attributes["offset"] = IntAttr(32)
    with pytest.raises(CompileError, match="whole aligned cells"):
        verify_plan(plan)


def test_removing_release_channel_still_fails_for_clipped_accesses(artifact):
    plan = artifact.plan
    channel = next(
        op for op in plan.walk() if isinstance(op, LoopChannelOp) and op.distance.data == 2
    )
    for op in list(plan.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError):
        verify_plan(plan)


def test_runtime_sizes_do_not_expand_compilation(monkeypatch, artifact):
    def forbidden(*args, **kwargs):
        raise AssertionError("finite Program instantiated")

    monkeypatch.setattr(Program, "instances", forbidden)
    result = compile(scale_prefix)
    assert result.asc == artifact.asc
    assert result.report["runtime_parameters"] == ["n", "m"]
    assert result.report["native_templates"] == 3
    assert result.asc.count("asc_copy_gm2ub(") == result.asc.count("asc_copy_ub2gm(") == 1
    assert result.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert "exact DMA prefix alignment" in result.asc
    assert "UINTPTR_MAX - npu_bytes_" in result.asc


def test_incorrect_valid_expression_is_not_treated_as_padding():
    definition = from_source("""
        def invalid(X: N.Tensor((n,), N.f32), n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for t in N.Pipelined(N.ceildiv(n, 64), stages=1):
                    valid = N.min(64, n - t * 64 - 1)
                    N.copy(X[t * 64:(t + 1) * 64], x[t % 2, :], valid=valid)
    """)
    with pytest.raises(CompileError, match="valid must be"):
        compile(definition)


def test_uniform_scalar_codegen_has_no_exponential_ternary_expansion():
    from xdsl_npu.scalar import Expr, bindings, expression

    value = Expr("argument", ("a0",))
    for i in range(1, 32):
        value = expression("min", value, Expr("argument", (f"a{i}",)))
    names, lines = bindings([value])
    assert len(lines) == 31
    assert len("\n".join(lines)) < 6000
    assert len(names[value]) < 30


def test_equivalent_scalar_bound_does_not_change_tail_or_generated_code(artifact):
    source = scale_prefix.source.replace(
        "N.ceildiv(length, B)", "N.min(N.ceildiv(length, B), N.ceildiv(length, B))"
    )
    equivalent = from_source(source, disjoint_external=True)
    assert compile(equivalent).asc == artifact.asc


def test_prefix_does_not_add_waits_or_change_release_scope(artifact):
    from examples.vector.scale import scale

    regular = compile(scale)

    def synchronization(code):
        return [line.strip() for line in code.splitlines() if "asc_sync_" in line]

    assert synchronization(artifact.asc) == synchronization(regular.asc)
