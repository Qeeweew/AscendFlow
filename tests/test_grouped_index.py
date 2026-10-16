"""Real AST runtime descriptors through native addresses and launcher guards."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr

from examples.vector.grouped import grouped_scale
from tests.test_runtime_contract import host_launcher
from xdsl_npu import compile, verify_plan
from xdsl_npu import language as T
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import BufferOp, LoopSyncOp
from xdsl_npu.dialects.source import ViewOp
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.lowering import lower_source


@pytest.fixture(scope="module")
def grouped():
    return compile(grouped_scale)


def test_runtime_descriptor_lowers_to_three_native_templates_and_affine_gm(grouped):
    assert grouped.report["native_templates"] == 3
    assert grouped.report["storage_periods"] == {
        "X": 1,
        "Y": 1,
        "aiv0/inputs": 2,
        "aiv0/outputs": 2,
    }
    assert grouped.report["channels"] == 4
    assert grouped.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    loop = extract_loop(parse(grouped.plan_ir))
    tasks = next(iter(loop.actors.values())).program.tasks
    gm = [ref for task in tasks for ref in task.refs if ref.storage.external]
    assert len(gm) == 2
    # Independent BNSD oracle, including more owners than work, uneven
    # division and a huge runtime extent without compiling another program.
    for length in (16, 48, 64, 4096, 1 << 30):
        tiles, total = length // 16, 6 * (length // 16)
        for core in range(20):
            owned = len(range(core, total, 20))
            assert loop.bound.evaluate({"S": length, "$core": core}) == owned
            for g in {0, owned - 1} if owned else set():
                flat = core + 20 * g
                batch, remainder = divmod(flat, 3 * tiles)
                head, row_tile = divmod(remainder, tiles)
                address = ((batch * 3 + head) * length + row_tile * 16) * 64 * 4
                assert all(ref.address(g, core) == address for ref in gm)
                assert all(ref.nbytes == 16 * 64 * 4 for ref in gm)
    # No runtime radix survives in a native task's GM pointer expression.
    copies = [line for line in grouped.asc.splitlines() if "asc_copy_" in line]
    assert all("npu_size_S" not in line and "npu_uniform" not in line for line in copies)
    assert all("g * 81920" in line and "core_id * 4096" in line for line in copies)


@pytest.mark.parametrize("groups", [1, 3, 20])
def test_owner_count_is_policy_not_an_input_length_specialization(groups):
    artifact = compile(grouped_scale.specialize(GROUPS=groups))
    assert artifact.report["native_templates"] == 3
    assert artifact.report["runtime_parameters"] == ["S"]


def test_actual_launcher_accepts_one_binary_for_different_lengths(grouped, tmp_path):
    library, launch = host_launcher(grouped, tmp_path)
    for length in (-1, 0, 1, 17, (1 << 63) - 1):
        assert launch(length) == -1
        assert library.dispatches() == 0
    for index, length in enumerate((16, 48, 64, 4096, 131072), 1):
        assert launch(length) == 0
        assert library.dispatches() == index


def test_source_axis_mutation_cannot_hide_inside_allocation_extent():
    source = grouped_scale.to_ir()
    view = next(
        op
        for op in source.walk()
        if isinstance(op, ViewOp) and op.source.owner.sym_name.data == "X"
    )
    # Carry one head past its legal axis: the resulting flattened address may
    # alias the next batch. Root byte bounds alone cannot certify this view.
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    changed = arith.AddiOp(view.origins[1], one.result)
    view.parent.insert_ops_before([one, changed], view)
    origins = list(view.origins)
    origins[1] = changed.result
    view.operands = (view.source, *origins, *((view.valid,) if view.valid is not None else ()))
    with pytest.raises(CompileError, match="logical axis"):
        lower_source(parse(dump(source)))


def test_runtime_radix_needs_positive_and_divisible_shape_contract():
    source = grouped_scale.source.replace("minimum=1, multiple_of=ROWS", "minimum=1")
    with pytest.raises(CompileError, match="provably positive|tensor extent"):
        lower_source(replace(grouped_scale, source=source).to_ir())


def test_cyclic_count_cannot_cancel_an_invalid_unsigned_dividend():
    source = grouped_scale.source.replace("total - group", "((total - group) * 3) // 3")
    with pytest.raises(CompileError, match="source unsigned division"):
        compile(replace(grouped_scale, source=source))


@pytest.mark.parametrize("cancel", ["bad - bad", "bad * 0"])
def test_source_division_contract_survives_frontend_constant_cancellation(cancel):
    definition = T.from_source(f"""
        def bad_index():
            with T.Kernel():
                x = T.alloc_ub((64,))
                for t in T.Pipelined(2, stages=1):
                    bad = (t - 1) // 2
                    offset = {cancel}
                    T.fill(x[offset:offset+64], 0.0)
    """)
    with pytest.raises(CompileError, match="source unsigned division"):
        compile(definition)


def test_final_verifier_rechecks_source_independent_allocation_and_events(grouped):
    plan = grouped.plan
    output = next(op for op in plan.walk() if isinstance(op, BufferOp) and op.sym_name.data == "Y")
    output.attributes["count_scale"] = IntAttr(output.count_scale.data - 4)
    with pytest.raises(CompileError, match="allocation must equal total"):
        verify_plan(plan)
    plan = grouped.plan
    wait = next(op for op in plan.walk() if isinstance(op, LoopSyncOp) and op.action.data == "wait")
    wait.parent.erase_op(wait)
    with pytest.raises(CompileError):
        verify_plan(plan)
