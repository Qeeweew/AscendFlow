"""Cyclic work totals against literal owner/step enumeration, without devices."""

from collections import Counter

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.ir import Block, Region

from examples.vector.grouped import grouped_scale
from examples.vector.scale import scale
from tests.test_cyclic_domain import cyclic_program
from xdsl_npu import compile
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.physical import (
    BoundaryOp,
    BufferOp,
    CyclicCountOp,
    EndOp,
    KernelOp,
    RuntimeSizeOp,
    StreamOp,
    TaskOp,
)
from xdsl_npu.loop_sync import verify_loop
from xdsl_npu.performance import runtime_work
from xdsl_npu.scalar import INT64_MAX, Expr


def terminal_store(module, uid="store"):
    stream = next(op for op in module.walk() if isinstance(op, StreamOp))
    task = next(op for op in stream.walk() if isinstance(op, TaskOp) and op.uid.data == uid)
    stream.body.block.detach_op(task)
    boundary = BoundaryOp.create(
        operands=[stream.body.block.args[0]],
        attributes={"kind": StringAttr("last")},
        regions=[Region(Block([task, EndOp()]))],
    )
    stream.body.block.insert_op_before(boundary, stream.body.block.last_op)
    return module


def literal_syncs(sequences, counts):
    """Follow emitted step/stage guards; no arithmetic residue-count formula."""
    result = Counter(
        {
            f"{c.kind}:{c.task.pipe}": 0
            for stream in sequences.values()
            for c in stream
            if c.kind != "task"
        }
    )
    for count in counts:
        for stream in sequences.values():
            for step in range(count + max(command.task.stage for command in stream)):
                for command in stream:
                    g = step - command.task.stage
                    if command.kind == "task" or not 0 <= g < count:
                        continue
                    if not command.lower <= g < count - command.upper:
                        continue
                    if g % command.period not in command.residues:
                        continue
                    if command.last_gap is not None and g != count - 1 - command.last_gap:
                        continue
                    result[f"{command.kind}:{command.task.pipe}"] += 1
    return dict(result)


@pytest.mark.parametrize("groups", [1, 3, 8])
@pytest.mark.parametrize("terminal", [False, True])
def test_owner_totals_match_literal_small_domain_for_native_memory_vector_and_sync(
    groups, terminal
):
    module = cyclic_program(groups)
    artifact = compile(terminal_store(module) if terminal else module)
    _, (_, _, sequences, _, _) = verify_loop(artifact.plan, _details=True)
    for total in range(17):
        # Independent assignment by global task ordinal, including idle owners.
        counts = [sum(flat % groups == core for flat in range(total)) for core in range(groups)]
        stores = sum(count > 0 for count in counts) if terminal else total
        work = runtime_work(artifact.plan, {"total": total})
        assert work["tasks_per_group"] is None
        assert work["group_task_counts"] == [
            {"tasks": tasks, "groups": multiplicity}
            for tasks, multiplicity in sorted(Counter(counts).items())
        ]
        assert work["total_tasks"] == total
        assert work["native_invocations"] == {"copy": total + stores, "muls": total}
        assert work["vector_requested_elements"] == {"muls": 16 * total}
        assert work["memory_requested_bytes_by_pipe_bank_mode"] == {
            "MTE2:GM:R": 64 * total,
            "MTE2:UB:W": 64 * total,
            "V:UB:R": 64 * total,
            "V:UB:W": 64 * total,
            "MTE3:UB:R": 64 * stores,
            "MTE3:GM:W": 64 * stores,
        }
        assert work["transfer_requested_bytes"] == {"GM->UB": 64 * total, "UB->GM": 64 * stores}
        assert work["sync_invocations"] == literal_syncs(sequences, counts)
        assert work["declared_gm_working_set_bytes"] == 128 * total
        assert work["terminal_drains"] == groups  # Even the owners with no tasks return.
        assert work["initial_fix_handshakes"] == work["initial_load3d_handshakes"] == 0


def cube_reduction(groups):
    k = Kernel("cyclic_reduction", core="aic", blocks=groups, disjoint_external=True)
    total = k.runtime_size("total")
    count = CyclicCountOp.create(operands=[total], result_types=[IndexType()])
    k.block.add_op(count)
    x = k.buffer(
        "X",
        "GM",
        0,
        "f16",
        layout="NZ",
        external=True,
        initialized=True,
        extent_count=total,
        count_scale=512,
    )
    y = k.buffer(
        "Y",
        "GM",
        0,
        "f16",
        layout="NZ",
        external=True,
        initialized=True,
        extent_count=total,
        count_scale=512,
    )
    z = k.buffer("Z", "GM", 0, external=True, extent_count=total, count_scale=1024)
    a = k.buffer("a", "L1", 512, "f16", layout="NZ")
    b = k.buffer("b", "L1", 512, "f16", layout="NZ")
    aa = k.buffer("aa", "L0A", 512, "f16", layout="ZZ")
    bb = k.buffer("bb", "L0B", 512, "f16", layout="ZN")
    c = k.buffer("c", "L0C", 1024, layout="NZ")
    k.init_fixpipe()
    k.loop(count.count)
    xr, yr = (k.view(root, 512, stride=groups * 512, core_stride=512) for root in (x, y))
    zr = k.view(z, 1024, stride=groups * 1024, core_stride=1024)
    ar, br = k.view(a, 512), k.view(b, 512)
    aar, bbr, cr = k.view(aa, 512), k.view(bb, 512), k.view(c, 1024)
    k.copy("load_x", xr, ar)
    k.copy("load_y", yr, br)
    k.copy("operand_x", ar, aar, stage=1)
    k.copy("operand_y", br, bbr, stage=1)
    k.task("accumulate", "mad_first", aar, bbr, cr, params=(16, 16, 16), stage=2)
    k.fixpipe("store", cr, zr, stage=3)
    return terminal_store(k.finish())


def test_first_mad_read_and_terminal_fix_count_only_active_owners_but_configuration_uses_all():
    groups = 5
    artifact = compile(cube_reduction(groups))
    _, (_, _, sequences, _, _) = verify_loop(artifact.plan, _details=True)
    for total in (0, 1, 2, 5, 6, 12):
        counts = [sum(flat % groups == core for flat in range(total)) for core in range(groups)]
        active = sum(bool(count) for count in counts)
        work = runtime_work(artifact.plan, {"total": total})
        assert work["native_invocations"] == {
            "copy": 4 * total,
            "mad_first": total,
            "fix16": active,
        }
        assert (
            work["memory_requested_bytes_by_pipe_bank_mode"]["M:L0C:R"] == (total - active) * 1024
        )
        assert work["transfer_requested_bytes"]["L0C->GM"] == active * 1024
        assert work["sync_invocations"] == literal_syncs(sequences, counts)
        assert work["terminal_drains"] == work["initial_fix_handshakes"] == groups


def test_large_total_visits_the_same_templates_and_at_most_two_owner_classes(monkeypatch):
    artifact = compile(cyclic_program(8))
    evaluations = 0
    original = Expr.evaluate

    def counted(self, values):
        nonlocal evaluations
        evaluations += 1
        return original(self, values)

    monkeypatch.setattr(Expr, "evaluate", counted)
    visits = []
    for total in (19, (1 << 40) + 3):
        evaluations = 0
        work = runtime_work(artifact.plan, {"total": total})
        visits.append(evaluations)
        assert work["total_tasks"] == total
        assert len(work["group_task_counts"]) == 2
        assert work["native_invocations"] == {"copy": 2 * total, "muls": total}
        assert sum(isinstance(op, TaskOp) for op in artifact.plan.walk()) == 3
    assert visits[0] == visits[1]


@pytest.fixture(scope="module")
def grouped():
    return compile(grouped_scale)


@pytest.mark.parametrize("value", [0, 1, 17, -1, True, 16.0, 1 << 63])
def test_runtime_accounting_checks_actual_parameter_contracts(grouped, value):
    with pytest.raises(CompileError, match="argument|scalar"):
        runtime_work(grouped.plan, {"S": value})


def test_dynamic_extent_uses_positive_radix_facts_and_rejects_bad_arguments_before_dividing():
    module = cyclic_program()
    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    count = next(op for op in module.walk() if isinstance(op, CyclicCountOp))
    radix = RuntimeSizeOp.create(
        result_types=[IndexType()],
        attributes={"sym_name": StringAttr("radix"), "minimum": IntAttr(1), "multiple": IntAttr(1)},
    )
    quotient = arith.DivUIOp(count.total, radix.value)
    kernel.body.block.insert_ops_before([radix, quotient], count)
    count.operands = [quotient.result]
    for op in kernel.body.block.ops:
        if isinstance(op, BufferOp) and op.external.data:
            op.operands = [quotient.result]
    artifact = compile(module)
    work = runtime_work(artifact.plan, {"total": 41, "radix": 4})
    assert work["total_tasks"] == 10 and work["declared_gm_working_set_bytes"] == 1280
    with pytest.raises(CompileError, match="radix.*minimum/multiple"):
        runtime_work(artifact.plan, {"total": 41, "radix": 0})


def test_valid_int64_argument_cannot_overflow_a_real_cyclic_gm_extent():
    artifact = compile(cyclic_program())
    with pytest.raises(CompileError, match="GM extent byte overflow"):
        runtime_work(artifact.plan, {"total": 1 << 57})  # 64-byte tiles exceed int64.


def test_valid_int64_argument_cannot_overflow_the_emitted_loop_step_bound():
    k = Kernel("local_step_bound")
    total = k.runtime_size("total")
    a = k.buffer("a", "UB", 64)
    k.loop(total)
    k.task("fill", "fill", k.view(a, 64), scalar=1.0, stage=2)
    artifact = compile(k.finish())
    with pytest.raises(CompileError, match="loop step overflow"):
        runtime_work(artifact.plan, {"total": INT64_MAX})


def test_noncyclic_count_fields_keep_the_existing_interface():
    artifact = compile(scale)
    work = runtime_work(artifact.plan, {"tiles": 7})
    assert work["tasks_per_group"] == 7
    assert "group_task_counts" not in work and "total_tasks" not in work
