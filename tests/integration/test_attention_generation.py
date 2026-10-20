"""Actual default FA structure, one symbolic public source build.

These finite byte/event checks neither execute floating point arithmetic nor
claim CANN/device qualification. No constexpr policy or runtime length is
specialized for this regression.
"""

import re

import pytest

from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.simplify_scalars import simplify_scalars
from ascendflow.target.ascend910b import Actor
from benchmarks.attention.contract import AttentionPolicy
from benchmarks.attention.run import generated_contract, launcher_abi
from tests.models.structured_events import check_structured_events
from tests.support.attention_compilation import compile_attention


@pytest.fixture(scope="module", params=[128, 256])
def default_attention(tmp_path_factory, request):
    return compile_attention(tmp_path_factory, AttentionPolicy(query_tile=request.param))


def test_default_attention_artifact_matches_the_device_runner(default_attention):
    _, _, asc, directory, policy = default_attention
    contract = generated_contract(directory, policy)
    assert contract["local_bytes_by_actor"]["aiv0"]["bytes"]["UB"] <= 192 * 1024
    assert launcher_abi(asc, "attention_target", generated=True).keys == (
        "stream",
        "q",
        "k",
        "v",
        "o",
        "score",
        "p",
        "pv",
        "sq",
        "sk",
    )


def test_default_attention_keeps_shared_vector_sync_in_its_runtime_lane_body(default_attention):
    final, _, asc, _, _ = default_attention
    assert asc.count("if ASC_IS_AIV") == 1
    assert "asc_get_sub_block_id()" in asc and asc.count("npu_sid") > 1
    assert "npu_sid ==" not in asc
    for op in final.module.walk():
        if isinstance(op, NotificationOp) and op.participants.data[0].data != Actor.AIC.value:
            assert {actor.data for actor in op.participants} == {Actor.AIV0.value, Actor.AIV1.value}


def test_default_attention_has_compact_single_clock_output(default_attention):
    _, _, asc, _, _ = default_attention
    assert not re.search(r"NpuPipelinePhase|prologue|steady|epilogue|short|auto npu_stage_", asc)
    launcher = asc.split('extern "C" int launch_attention_target(', 1)[1]
    assert not re.search(r"\b(if|for|while)\b|npu_host_|npu_bytes|npu_ptrs|overflow", launcher)
    assert launcher.count("<<<") == 1
    # One KV clock and two child clocks in each of the AIC and shared AIV bodies.
    assert len(re.findall(r"for \(int64_t npu_step_", asc)) == 6


def test_default_attention_scalar_simplification_is_idempotent(default_attention):
    final, _, asc, _, _ = default_attention
    before = dump(final.module)
    module = parse(before)
    simplify_scalars(module)
    module.verify()
    assert dump(module) == before
    assert " / uint64_t(1)" not in asc and " % uint64_t(1)" not in asc
    assert " + 0;" not in asc and " - 0;" not in asc
    assert not re.search(r"\b0 \+ npu_", asc) and " * 1)" not in asc


@pytest.mark.parametrize(
    "sq,sk",
    [
        (256, 1024),
        (256, 2048),
        (256, 3072),
        (1024, 2048),
        (1, 1),
        (127, 129),
        (129, 127),
        (257, 513),
        (769, 385),
    ],
)
def test_default_attention_native_versions_and_real_event_consumption(default_attention, sq, sk):
    final, original, _, _, policy = default_attention
    execution = check_structured_events(
        final,
        {"Sq": sq, "Sk": sk},
        logical_reference=original,
        groups=(0, 19),
        max_commands=20000,
    )
    assert execution.natives and execution.consumptions
    assert execution.bytes_read and execution.bytes_written
    # Only a genuinely partial key tile enters the wide predicate packet
    # and its Vector fence. Aligned lengths skip the entire tail update.
    tail_updates = [
        command
        for command in execution.natives
        if command.occurrence.operation.native.data == "mask_fill"
    ]
    assert bool(tail_updates) == bool(sk % policy.kv_tile)
