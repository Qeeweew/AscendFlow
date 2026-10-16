"""Finite checks of the actual reference control, plus its numerical input contract.

The host shim traces C++ loop/guard/event statements. Random legal FIFO retirement
orders check pairing/progress, not arbitrary-N synchronization or tensor effects.
"""

import random
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import pytest

from benchmarks.references.attention import reference_source
from benchmarks.validation.attention import inputs_and_oracle
from tests.fixtures.attention_target import attention_target
from xdsl_npu.dialects.source import PipelineOp, StageOp


@pytest.fixture(scope="module")
def traced_reference(tmp_path_factory):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("host C++ compiler needed for actual reference control tracing")
    path = tmp_path_factory.mktemp("attention_control")
    src = reference_source(heads=1, kv_heads=1, groups=2)
    src = src.replace('#include "kernel_operator.h"', '#include "attention_trace.h"')
    src = src.split('extern "C" int launch_attention_reference')[0]
    src += """
int main(int argc,char** argv) {
 trace_group=std::atoi(argv[3]);
 for(trace_actor=0;trace_actor<3;++trace_actor)
  attention_reference(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                      std::atoll(argv[1]),std::atoll(argv[2]));
}
"""
    (path / "reference.cpp").write_text(src)
    subprocess.run(
        [
            cxx,
            "-std=c++20",
            "-O0",
            "-I",
            str(Path(__file__).parent / "support"),
            str(path / "reference.cpp"),
            "-o",
            str(path / "reference"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return path / "reference"


def retire_trace(lines, seed):
    program = [deque() for _ in range(3)]
    for line in lines.splitlines():
        actor, action, pipe, other, flag = line.split()
        program[int(actor)].append((action, pipe, other, int(flag)))
    queues, flags, counters = defaultdict(deque), set(), defaultdict(int)
    rng, steps, maximum = random.Random(seed), 0, 0
    while any(program) or any(queues.values()):
        enabled = []
        for actor, commands in enumerate(program):
            if not commands:
                continue
            action, pipe, other, flag = commands[0]
            if action == "recv":
                sources = (1, 2) if actor == 0 else (0,)
                if not all(counters[s, actor, flag] for s in sources):
                    continue
            if action == "drain" and any(q for (a, _), q in queues.items() if a == actor):
                continue
            enabled.append((True, actor, None))
        for (actor, pipe), queue in queues.items():
            if not queue:
                continue
            action, _, other, flag = queue[0]
            if action != "wait" or (actor, other, pipe, flag) in flags:
                enabled.append((False, actor, pipe))
        assert enabled, f"deadlock after {steps} transitions, seed={seed}"
        issue, actor, pipe = rng.choice(enabled)
        if issue:
            command = program[actor].popleft()
            action, pipe, _, flag = command
            if action == "recv":
                for source in (1, 2) if actor == 0 else (0,):
                    counters[source, actor, flag] -= 1
            elif action != "drain":
                queues[actor, pipe].append(command)
        else:
            action, pipe, other, flag = queues[actor, pipe].popleft()
            if action == "set":
                key = actor, pipe, other, flag
                assert key not in flags, f"binary Set overwrote unconsumed flag {key}"
                assert 0 <= flag < 8
                flags.add(key)
            elif action == "wait":
                flags.remove((actor, other, pipe, flag))
            elif action == "send":
                for destination in (1, 2) if actor == 0 else (0,):
                    counters[actor, destination, flag] += 1
                    maximum = max(maximum, counters[actor, destination, flag])
                    assert maximum <= 15
        steps += 1
    assert not flags and not any(counters.values()), "unconsumed final event/notification"
    return steps, maximum


@pytest.mark.parametrize(
    "sq,sk,group",
    [(0, 1024, 0), (128, 1024, 1), (128, 1024, 0), (512, 1024, 0), (640, 2048, 1), (896, 3072, 0)],
)
def test_actual_cpp_control_pairs_and_drains_across_queries(traced_reference, sq, sk, group):
    trace = subprocess.check_output(
        [str(traced_reference), str(sq), str(sk), str(group)], text=True, timeout=10
    )
    for seed in range(16):
        _, maximum = retire_trace(trace, seed)
        assert maximum <= 2


@pytest.mark.parametrize("remove", ["2 send MTE3 S 2", "0 set MTE2 MTE1 6"])
def test_traced_protocol_checker_rejects_missing_contribution_or_ready(traced_reference, remove):
    trace = subprocess.check_output(
        [str(traced_reference), "512", "2048", "0"], text=True, timeout=10
    )
    assert remove in trace
    trace = trace.replace(remove + "\n", "", 1)
    with pytest.raises(AssertionError, match="deadlock|unconsumed|overwrote"):
        retire_trace(trace, 0)


@pytest.fixture(scope="module")
def target_schedule():
    # Read the selected constexpr policy, including named issue-order tuples.
    # The finite operand model below remains independent of synchronization.
    return attention_target().to_ir()


def child_schedule(module, name, *, default=False):
    for node in module.walk():
        if not isinstance(node, PipelineOp):
            continue
        stages = {
            stage.label.data: stage.offset.data
            for stage in node.body.block.ops
            if isinstance(stage, StageOp)
        }
        if name not in stages:
            continue
        order = (
            tuple(stages)
            if default
            else tuple(label.data for label in node.issue_order.data["cube"])
        )
        return stages, order
    raise AssertionError("missing native child")


def check_single_operand(stages, order, producer, reader, count):
    current, consumed = None, True
    for step in range(count + max(stages.values())):
        for name in order:
            generation = step - stages[name]
            if not 0 <= generation < count:
                continue
            if name == producer:
                assert consumed, f"operand overwritten before MAD generation {current}"
                current, consumed = generation, False
            if name == reader:
                assert current == generation, "MAD reads wrong operand generation"
                consumed = True
    assert consumed


@pytest.mark.parametrize(
    "producer,reader", [("operand_k", "qk_native"), ("operand_pv", "pv_native")]
)
def test_target_one_slot_requires_explicit_inner_issue_order(target_schedule, producer, reader):
    for count in (0, 1, 2, 8, 17):
        stages, order = child_schedule(target_schedule, producer)
        check_single_operand(stages, order, producer, reader, count)
    stages, order = child_schedule(target_schedule, producer, default=True)
    with pytest.raises(AssertionError, match="overwritten before MAD"):
        check_single_operand(stages, order, producer, reader, 2)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_uniform_attention_oracle_and_declared_workspace(dtype):
    import numpy as np

    from benchmarks.validation.attention import decode, rounded

    policy = {"batch": 1, "heads": 2, "kv_heads": 1, "groups": 2, "dtype": dtype}
    raw, expected, selected, _ = inputs_and_oracle(policy, 128, 2048, data_case="uniform")
    v = decode(raw["v"], dtype, (1, 1, 2048, 128))
    mean = rounded(v.mean(axis=2, dtype=np.float32), dtype)
    assert np.array_equal(expected, np.broadcast_to(mean[:, :, None, :], expected.shape))
    assert len(selected) == 128
    assert len(raw["score"]) == 2 * 2 * 128 * 1024 * 4
    assert len(raw["p"]) == 2 * 2 * 128 * 1024 * 2
    assert len(raw["pv"]) == 2 * 2 * 128 * 128 * 4
