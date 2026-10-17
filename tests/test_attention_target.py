"""The shared FA example and its explicitly unsupported continuous policy."""

import ast
from collections import Counter

import pytest

from ascendflow.diagnostics import CompileError
from ascendflow.dialects.source import AllocationOp
from tests.fixtures.attention_target import activation_trace, attention_target, target_source


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
