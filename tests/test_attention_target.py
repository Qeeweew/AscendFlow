"""The shared FA example and its explicitly unsupported continuous policy."""

import ast
from collections import Counter
from math import prod

import pytest

from tests.fixtures.attention_target import activation_trace, attention_target, target_source
from xdsl_npu.ir import BufferOp, RuntimeSizeOp, ScopeGuardOp, TaskOp, ViewOp
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CAPACITY, CompileError
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.source_ir import AllocationOp


def tree(mode):
    return ast.parse(target_source(continuous=mode))


def stages(module):
    for node in ast.walk(module):
        if isinstance(node, ast.With):
            calls = [item.context_expr for item in node.items]
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "Stage"
                and any(
                    k.arg == "name" and k.value.value in {"qk", "softmax", "pv", "update"}
                    for k in call.keywords
                )
                for call in calls
            ):
                yield ast.dump(node, include_attributes=False)


def test_both_organizations_share_actual_algorithm_and_explicit_operations():
    assert list(stages(tree(False))) == list(stages(tree(True)))
    for mode in (False, True):
        names = [
            node.func.attr
            for node in ast.walk(tree(mode))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "exp" in names and "reduce_sum" in names and "reduce_max" in names
        # Each of QK/PV has a default and a paired constexpr branch. Actual
        # retained native counts are checked after specialization below.
        assert names.count("matmul") == 4
        assert not set(names) & {"task", "autosync", "verify_plan", "set_flag", "wait_flag"}


def test_continuous_source_rejects_its_actual_unsupported_runtime_cycle():
    # Nested actual-IR synchronization failures are covered by serial
    # integration tests. This check fails during physical lowering, before
    # any full-FA planning or event search can run.
    with pytest.raises(CompileError, match="runtime-cycle predicates are unsupported"):
        lower_source(attention_target(continuous=True))


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"HQ": 7, "HKV": 2}, "query heads must be divisible by KV heads"),
        ({"BQ": 120}, "query tile must cover complete rows on both AIVs"),
        ({"BK": 1088}, "KV tile must contain complete Cube reduction tiles"),
        ({"CK": 64}, "shared QK/PV accumulator planes require CK equal to D"),
        ({"VR": 0}, "Vector row tile must be positive"),
    ],
)
def test_attention_source_rejects_truncated_or_incompatible_constexpr_tiles(parameters, message):
    with pytest.raises(CompileError, match=message):
        attention_target().specialize(**parameters).to_ir()


def test_declared_default_physical_allocations_fit_per_actor_not_combined_ub():
    import math

    function = tree(False).body[0]
    source = attention_target().to_ir()
    memory = Counter()
    for allocation in source.walk():
        if not isinstance(allocation, AllocationOp) or allocation.access.data != "local":
            continue
        tensor = allocation.ref.type
        size = 4 if tensor.dtype.data == "f32" else 2
        nbytes = math.prod(dim.data for dim in tensor.shape) * size
        memory[tensor.memory.data.lower()] += (nbytes + 31) // 32 * 32
    assert memory["l1"] == 224 * 1024
    assert memory["l0a"] == memory["l0b"] == 64 * 1024
    assert memory["l0c"] == 128 * 1024
    assert memory["ub"] <= 192 * 1024
    # No initialized input value is granted to GM workspace.
    for arg in function.args.args:
        if arg.arg in {"Score", "P", "PV"}:
            assert (
                next(k.value.value for k in arg.annotation.keywords if k.arg == "access") == "write"
            )


@pytest.mark.parametrize("queries,kv", [(0, 0), (0, 3), (2, 0), (1, 1), (2, 1), (2, 3), (3, 7)])
def test_each_real_logical_task_appears_once_per_stage(queries, kv):
    for mode in (False, True):
        trace = activation_trace(queries, kv, continuous=mode)
        counts = Counter(
            (row["stage"], row["query"], row["kv"]) for rows in trace.values() for row in rows
        )
        assert counts == Counter(
            {
                (stage, q, k): 1
                for stage in ("qk", "softmax", "pv", "update")
                for q in range(queries)
                for k in range(kv)
            }
        )


def test_continuous_query_boundary_exposes_actual_issue_overlap():
    a = activation_trace(2, 3, continuous=True)
    cube = [(row["stage"], row["query"], row["kv"]) for row in a["cube"]]
    assert cube.index(("qk", 1, 0)) < cube.index(("pv", 0, 2))
    vector = [(row["stage"], row["query"], row["kv"]) for row in a["vector"]]
    # This is a counterexample, not a supported schedule: S(q+1,0) resets
    # the single l buffer before U(q,last) reads its final denominator.
    assert vector.index(("softmax", 1, 0)) < vector.index(("update", 0, 2))
    nested = activation_trace(2, 3)
    cube = [(row["stage"], row["query"], row["kv"]) for row in nested["cube"]]
    assert cube.index(("pv", 0, 2)) < cube.index(("qk", 1, 0))
    vector = [(row["stage"], row["query"], row["kv"]) for row in nested["vector"]]
    assert vector.index(("update", 0, 2)) < vector.index(("softmax", 1, 0))
    assert vector.index(("softmax", 0, 1)) < vector.index(("update", 0, 0))


def test_trace_is_bounded_and_does_not_accept_fake_policy_or_lengths():
    with pytest.raises(ValueError, match="<=64"):
        activation_trace(1_000_000, 1_000_000)
    with pytest.raises(ValueError, match="boolean"):
        target_source(continuous="automatic")


@pytest.fixture(scope="module", params=(128, 256), ids=("default", "paired"))
def physical_policy(request):
    source = attention_target().specialize(BQ=request.param).to_ir()
    module = lower_source(source)
    return request.param, source, extract_scopes(module)


def actual_root(ref):
    owner = ref.owner
    while not isinstance(owner, BufferOp):
        owner = owner.source.owner
    return owner


def actual_tasks(leaf):
    return tuple(op for op in leaf.stream.walk() if isinstance(op, TaskOp))


def gm_l1_templates(program):
    result = {}
    for leaf in program.leaves.values():
        for task in actual_tasks(leaf):
            if task.native.data == "nd2nz":
                assert task.refs[0].type.bank.data == "GM"
                assert task.refs[1].type.bank.data == "L1"
                result.setdefault(actual_root(task.refs[0]).sym_name.data, []).append((leaf, task))
    return result


def actual_segments(ref, generation):
    """Small fixed-child addresses read directly from physical view attributes."""
    view = ref.owner
    assert isinstance(view, ViewOp)
    divisor = 1 if view.divisor is None else view.divisor.data
    coordinate = generation // divisor
    if view.modulus.data:
        coordinate %= view.modulus.data
    begin = view.offset.data + view.stride.data * coordinate
    for term in view.index_terms or ():
        coefficient, divisor, modulus = (item.data for item in term)
        coordinate = generation // divisor
        begin += coefficient * (coordinate % modulus if modulus else coordinate)
    rows = 1 if view.rows is None else view.rows.data
    pitch = 0 if view.row_stride is None else view.row_stride.data
    return tuple(
        (begin + row * pitch, begin + row * pitch + view.nbytes.data) for row in range(rows)
    )


def native_model(leaf, task):
    return next(
        native
        for data in leaf.loop.actors.values()
        for native in data.program.tasks
        if native.uid == task.uid.data
    )


def test_native_row_parts_share_one_k_and_v_load_without_runtime_expansion(physical_policy):
    bq, _, program = physical_policy
    parts, loads = bq // 128, gm_l1_templates(program)
    assert {name: len(tasks) for name, tasks in loads.items()} == {
        "Q": parts,
        "K": 1,
        "P": parts,
        "V": 1,
    }
    qk, pv = loads["K"][0][0], loads["V"][0][0]
    for leaf, mad, output in ((qk, "mad", "Score"), (pv, "mad_first", None)):
        tasks = actual_tasks(leaf)
        assert leaf.stream.count.data == 8
        assert sum(task.native.data == mad for task in tasks) == parts
        if output is not None:
            assert (
                sum(
                    task.native.data == "fix" and actual_root(task.refs[1]).sym_name.data == output
                    for task in tasks
                )
                == parts
            )
    assert (
        sum(
            task.native.data == "fix" and actual_root(task.refs[1]).sym_name.data == "PV"
            for leaf in program.leaves.values()
            for task in actual_tasks(leaf)
        )
        == parts
    )
    assert Counter(leaf.actor for leaf in program.leaves.values()) == {
        "aic": 4,
        "aiv0": 4,
        "aiv1": 4,
    }
    assert {op.sym_name.data for op in program.parameters} == {"Sq", "Sk"}
    assert all(isinstance(op, RuntimeSizeOp) for op in program.parameters)
    assert set(program.invocations) == {"aic", "aiv0", "aiv1"}
    assert all(
        parent.count is None and parent.limit is not None for parent in program.parents.values()
    )
    assert all(invocation.cyclic is not None for invocation in program.invocations.values())
    assert (
        program.invocations["aic"].cyclic.total.evaluate({"Sq": 4096, "Sk": 4096}) == 8 * 4096 // bq
    )
    assert program.bound.evaluate({"Sq": 4096, "Sk": 4096}) == 4
    # Only constexpr row parts are expanded; both runtime clocks remain in IR.
    assert sum(isinstance(op, TaskOp) for op in program.module.walk()) < 100


def test_actual_policy_shapes_and_exact_per_actor_capacity(physical_policy):
    bq, source, program = physical_policy
    loads = gm_l1_templates(program)
    allocations = {
        op.sym_name.data: op
        for op in source.walk()
        if isinstance(op, AllocationOp) and op.access.data == "local"
    }

    def shape(root):
        return tuple(dim.data for dim in allocations[root.sym_name.data].ref.type.shape)

    assert shape(actual_root(loads["Q"][0][1].refs[1])) == (
        (128, 128) if bq == 128 else (2, 128, 128)
    )
    assert shape(actual_root(loads["P"][0][1].refs[1])) == (
        (2, 128, 128) if bq == 128 else (2, 2, 128, 128)
    )
    mad = next(t for t in actual_tasks(loads["K"][0][0]) if t.native.data == "mad")
    assert shape(actual_root(mad.refs[2])) == (2, 128, 128)
    totals = {}
    for actor, kernel in program.kernels.items():
        totals[actor] = Counter()
        for op in kernel.body.block.ops:
            if isinstance(op, BufferOp) and not op.external.data:
                assert op.nbytes.data % 32 == 0
                totals[actor][op.ref.type.bank.data] += op.nbytes.data
        assert all(size <= CAPACITY[bank] for bank, size in totals[actor].items())
    assert totals == {
        "aic": {
            "L1": 224 * 1024 if bq == 128 else 320 * 1024,
            "L0A": 64 * 1024,
            "L0B": 64 * 1024,
            "L0C": 128 * 1024,
        },
        "aiv0": {"UB": 182400 if bq == 128 else 183424},
        "aiv1": {"UB": 182400 if bq == 128 else 183424},
    }
    source_ub = [op for op in allocations.values() if op.ref.type.memory.data == "UB"]
    assert sum(tuple(d.data for d in op.ref.type.shape) == (8, 1024) for op in source_ub) == (
        bq == 128
    )
    assert (
        sum(
            prod(d.data for d in op.ref.type.shape) * (4 if op.ref.type.dtype.data == "f32" else 2)
            for op in source_ub
        )
        == totals["aiv0"]["UB"]
    )


def test_score_slot_last_read_is_the_actual_inplace_cast(physical_policy):
    bq, _, program = physical_policy
    for actor in ("aiv0", "aiv1"):
        leaf, load = next(
            (leaf, task)
            for leaf in program.leaves.values()
            if leaf.actor == actor
            for task in actual_tasks(leaf)
            if task.native.data in {"copy", "copy_rows"}
            and actual_root(task.refs[0]).sym_name.data == "Score"
        )
        slot = actual_root(load.refs[1])
        readers = [
            task
            for task in actual_tasks(leaf)
            if any(
                actual_root(ref) is slot and "R" in mode
                for ref, mode in zip(task.refs, native_model(leaf, task).modes, strict=True)
            )
        ]
        assert readers[0].native.data == "muls"
        assert readers[-1].native.data == ("muls" if bq == 128 else "convert_rn")
        assert (actual_root(readers[0].refs[1]) is slot) == (bq == 256)
        assert actual_segments(load.refs[1], 0) != actual_segments(load.refs[1], 1)
        assert actual_segments(load.refs[1], 0) == actual_segments(load.refs[1], 2)
        if bq == 256:
            for generation in (0, 1, 2):
                assert actual_segments(readers[-1].refs[0], generation) == actual_segments(
                    load.refs[1], generation
                )


def test_paired_operand_and_accumulator_last_reads_remain_explicit(physical_policy):
    bq, _, program = physical_policy
    if bq == 128:
        return
    loads = gm_l1_templates(program)
    qk, pv = loads["K"][0][0], loads["V"][0][0]
    plane_refs = []
    for leaf, mad_name in ((qk, "mad"), (pv, "mad_first")):
        tasks = actual_tasks(leaf)
        mads = [task for task in tasks if task.native.data == mad_name]
        copies_a = [task for task in tasks if task.native.data == "load2d_a"]
        copy_b = next(task for task in tasks if task.native.data in {"load2d_b_nt", "load2d_b_nn"})
        assert len(mads) == len(copies_a) == 2
        assert actual_root(copies_a[0].refs[1]) is actual_root(copies_a[1].refs[1])
        assert actual_segments(copies_a[1].refs[1], 0) == actual_segments(mads[0].refs[0], 0)
        assert (
            tasks.index(copies_a[0])
            < tasks.index(mads[0])
            < tasks.index(copies_a[1])
            < tasks.index(mads[1])
        )
        for mad in mads:
            assert actual_root(copy_b.refs[1]) is actual_root(mad.refs[1])
            assert actual_segments(copy_b.refs[1], 1) == actual_segments(mad.refs[1], 0)
        assert [actual_segments(mad.refs[2], 0) for mad in mads] == [
            ((0, 65536),),
            ((65536, 131072),),
        ]
        assert [actual_segments(mad.refs[2], 1) for mad in mads] == [
            actual_segments(mad.refs[2], 0) for mad in mads
        ]
        plane_refs.append([mad.refs[2] for mad in mads])
        if mad_name == "mad_first":
            for mad in mads:
                native = native_model(leaf, mad)
                assert native.modes_for(0)[2] == "W"
                assert native.modes_for(1)[2] == "RW"
    assert all(
        actual_root(ref) is actual_root(plane_refs[0][0]) for refs in plane_refs for ref in refs
    )
    for output in ("Score", "PV"):
        fixes = [
            task
            for leaf in program.leaves.values()
            for task in actual_tasks(leaf)
            if task.native.data == "fix" and actual_root(task.refs[1]).sym_name.data == output
        ]
        assert len(fixes) == 2
        for fix, plane in zip(fixes, plane_refs[0], strict=True):
            assert actual_root(fix.refs[0]) is actual_root(plane)
            assert actual_segments(fix.refs[0], 0) == actual_segments(plane, 1)


def test_4k_l1_request_budget_comes_from_actual_dma_footprints_and_retained_counts(physical_policy):
    bq, _, program = physical_policy
    parameters = {"Sq": 4096, "Sk": 4096}
    invocation = program.invocations["aic"]
    queries = sum(
        invocation.bound.evaluate(parameters | {"$core": group}) for group in range(program.groups)
    )
    kv_tiles = program.bound.evaluate(parameters)
    requested = Counter()
    for name, templates in gm_l1_templates(program).items():
        for leaf, task in templates:
            guards = []
            ancestor = task.parent_op()
            while ancestor is not leaf.stream:
                if isinstance(ancestor, ScopeGuardOp):
                    guards.append(ancestor.kind.data)
                ancestor = ancestor.parent_op()
            assert guards == (["first"] if name == "Q" else [])
            source, destination = (ref.owner for ref in task.refs)
            assert isinstance(source, ViewOp) and isinstance(destination, ViewOp)
            source_bytes = source.nbytes.data * (1 if source.rows is None else source.rows.data)
            destination_bytes = destination.nbytes.data * (
                1 if destination.rows is None else destination.rows.data
            )
            assert source_bytes == destination_bytes == 32768
            assert tuple(param.data for param in task.params) == (128, 128)
            repetitions = leaf.stream.count.data * queries * (1 if guards else kv_tiles)
            requested[name] += source_bytes * repetitions
    mib = 1024 * 1024
    assert requested == {
        "Q": 8 * mib,
        "K": (256 if bq == 128 else 128) * mib,
        "V": (256 if bq == 128 else 128) * mib,
        "P": 256 * mib,
    }
    assert sum(requested.values()) == (776 if bq == 128 else 520) * mib
