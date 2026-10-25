"""Task-scale changes preserve the examples' fixed worker scratch capacities."""

from math import prod

import pytest

from ascendflow.dialects.source import AllocationOp
from examples.flash_attention.attention import attention_target
from examples.flash_attention.simple import simple_attention
from examples.kda.prefill_gram_cross import prefill_gram_cross
from examples.kda.prefill_output import prefill_output
from examples.kda.prefill_prepare import prefill_prepare
from examples.kda.prefill_scan import prefill_scan
from examples.kda.prefill_solve import prefill_solve_merge
from examples.kda.prefill_wu import prefill_wu
from examples.sparse_attention.attention import sparse_mla


def capacities(definition):
    result = {}
    for allocation in definition.to_ir().walk():
        if not isinstance(allocation, AllocationOp):
            continue
        tensor = allocation.ref.type
        if tensor.memory.data != "GM" or allocation.access.data != "local":
            continue
        assert not allocation.dimensions
        nbytes = prod(n.data for n in tensor.shape) * (4 if tensor.dtype.data == "f32" else 2)
        result[allocation.sym_name.data] = (nbytes + 511) // 512 * 512
    return result


@pytest.mark.parametrize(
    "definition,small,large",
    [
        (attention_target, {"B": 1, "HQ": 8}, {"B": 8, "HQ": 32}),
        (simple_attention, {"H": 4, "S": 256}, {"H": 64, "S": 4096}),
        (sparse_mla, {"B": 1, "T": 4, "H": 64}, {"B": 4, "T": 128, "H": 128, "MAX_SEQ": 8192}),
        (prefill_prepare, {"CHUNKS": 2, "NC": 2, "H": 8}, {"CHUNKS": 64, "NC": 64, "H": 32}),
        (
            prefill_scan,
            {"CHUNKS": 2, "NC": 2, "H": 8},
            {"CHUNKS": 64, "NC": 16, "SEQUENCES": 4, "H": 32},
        ),
        (prefill_output, {"CHUNKS": 2, "NC": 2, "H": 8}, {"CHUNKS": 64, "NC": 64, "H": 32}),
        (prefill_gram_cross, {"CHUNKS": 2, "H": 2, "PACK": 2}, {"CHUNKS": 64, "H": 32, "PACK": 2}),
        (prefill_wu, {"CHUNKS": 2, "H": 2}, {"CHUNKS": 64, "H": 32}),
        (
            prefill_solve_merge,
            {"CHUNKS": 2, "H": 2, "WIDTH": 16},
            {"CHUNKS": 64, "H": 32, "WIDTH": 16},
        ),
        (
            prefill_solve_merge,
            {"CHUNKS": 2, "H": 2, "WIDTH": 32},
            {"CHUNKS": 64, "H": 32, "WIDTH": 32},
        ),
    ],
)
def test_private_workspace_capacity_is_independent_of_task_count(definition, small, large):
    first = capacities(definition.specialize(GROUPS=2, **small))
    second = capacities(definition.specialize(GROUPS=2, **large))
    assert first and first == second


def test_scan_uses_one_worker_plane_for_each_transient_value():
    sizes = capacities(prefill_scan.specialize(GROUPS=2))
    assert sizes == {
        "History": 2 * 128 * 128 * 4,
        "Projection": 2 * 64 * 128 * 4,
        "Delta": 2 * 128 * 128 * 4,
    }


def test_teaching_attention_retires_workspace_before_reusing_it_for_another_head():
    from ascendflow.passes.structured_sync import synchronize_structure
    from tests.models.structured_events import check_compiled_events

    plan = synchronize_structure(simple_attention.specialize(H=5, S=128, GROUPS=2))
    checked = check_compiled_events(plan, groups=(0, 1), invocations=2, max_commands=30000)
    assert checked.natives and checked.consumptions
