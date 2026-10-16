"""Real memory obligations and guarded cuts, without a synchronization solver."""

from dataclasses import replace

import pytest

from tests.fixtures.pipelines import cube_pipeline, vector_pipeline
from xdsl_npu.analysis.frontier import Clock, KnownPrefix, Partition, PrefixDomain, PrefixRank
from xdsl_npu.analysis.hazards import Activation, HazardKind, bind_hazards
from xdsl_npu.analysis.native import extract_native_tasks
from xdsl_npu.analysis.transitions import transition_layout
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.target.ascend910b import Milestone


def analyze(module):
    loop = extract_loop(module)
    tasks = extract_native_tasks(loop, clock=Clock("step"))
    return loop, tasks, bind_hazards(loop, tasks, loop_dependencies(loop))


def select(hazards, source, destination, kind):
    return tuple(
        h
        for h in hazards
        if h.producer.operation.uid.data == source
        and h.consumer.operation.uid.data == destination
        and h.kind == kind
    )


@pytest.mark.parametrize("stages", [(0, 1, 2), (0, 0, 0)])
def test_double_buffer_ready_release_and_no_fictional_endpoint(stages):
    _, _, hazards = analyze(vector_pipeline(count=9, stages=stages))
    expected = (
        ("load", "scale", "RAW", 0),
        ("scale", "load", "WAR", 2),
        ("postprocess", "store", "RAW", 0),
        ("store", "scale", "WAR", 2),
    )
    for source, destination, kind, distance in expected:
        matches = select(hazards, source, destination, kind)
        assert matches and all(h.activation.distance == distance for h in matches)
        for count in (0, 1, 2, 3, 9):
            consumer_generations = {
                g for h in matches for g in range(-2, count + 2) if h.activation.consumes(g, count)
            }
            producer_generations = {
                g for h in matches for g in range(-2, count + 2) if h.activation.produces(g, count)
            }
            assert consumer_generations == set(range(distance, count))
            assert producer_generations == set(range(max(0, count - distance)))


def test_last_actual_reader_is_retained_before_input_slot_reuse():
    _, _, hazards = analyze(vector_pipeline(late_reader=True))
    release = select(hazards, "last_read", "load", HazardKind.WAR)
    earlier = select(hazards, "scale", "load", HazardKind.WAR)
    assert release and earlier
    assert release[0].producer.operation is not earlier[0].producer.operation
    assert release[0].source_cut.rank.compare(earlier[0].source_cut.rank) > 0
    # Computation on the output continues after this read. It is not the
    # input's release anchor, and neither is the later GM store.
    assert not select(hazards, "postprocess", "load", HazardKind.WAR)
    assert not select(hazards, "store", "load", HazardKind.WAR)


def test_stage_binding_preserves_actual_end_and_consumer_admission():
    _, _, hazards = analyze(vector_pipeline(stages=(0, 2, 4), slots=3))
    hazard = select(hazards, "store", "scale", HazardKind.WAR)[0]
    source, consumer = hazard.physical_cuts()
    assert source.operation is hazard.producer.operation
    assert consumer.operation is hazard.consumer.operation
    assert source.milestone is Milestone.NATIVE_END
    assert consumer.milestone is Milestone.ADMISSION
    assert hazard.source_cut.rank.offset == 4 - 3
    assert source.rank.offset == 4 - 2 - 3
    assert consumer.rank.offset == 0
    for g in (3, 17, 10**12):
        step = g + hazard.consumer.stage
        assert step + source.rank.offset == g - 3 + hazard.producer.stage


def test_cube_releases_each_actual_input_layer_before_downstream_completion():
    _, _, hazards = analyze(cube_pipeline())
    assert select(hazards, "operand_a", "load_a", HazardKind.WAR)
    assert select(hazards, "operand_b", "load_b", HazardKind.WAR)
    assert select(hazards, "mad", "operand_a", HazardKind.WAR)
    assert select(hazards, "export", "mad", HazardKind.WAR)
    assert not select(hazards, "export", "load_a", HazardKind.WAR)
    assert not select(hazards, "mad", "load_a", HazardKind.WAR)


def test_short_mad_memory_need_cannot_use_unrelated_later_implicit_prefix():
    _, tasks, hazards = analyze(cube_pipeline())
    hazard = select(hazards, "mad", "export", HazardKind.RAW)[0]
    task = hazard.producer
    layout = transition_layout({task.actor: {t.engine for t in tasks}}, task.clock)
    partition = Partition()
    assert not task.implicit_completion
    assert len(hazard.completion_prefixes(partition)) == 1
    later_large = KnownPrefix(
        PrefixDomain(task.actor, task.engine, Milestone.IMPLICIT_RETIREMENT, task.clock),
        PrefixRank(task.clock, 100, 0),
    )
    state = layout.frontier.empty().record(later_large)
    assert not hazard.covered_by(state, partition)
    captured = state.record(*hazard.completion_prefixes(partition))
    assert hazard.covered_by(captured, partition)


def test_terminal_reader_prefix_and_source_existence_remain_separate():
    domain = Activation(2, 3, frozenset({0, 1, 2}), consumer_last=True)
    for count in range(8):
        assert {g for g in range(-2, count + 2) if domain.consumes(g, count)} == (
            {count - 1} if count > 2 else set()
        )
        assert {g for g in range(-2, count + 2) if domain.produces(g, count)} == (
            {count - 3} if count > 2 else set()
        )
    source_only = replace(domain, consumer_last=False, source_last=True)
    assert not any(source_only.consumes(g, 8) for g in range(8))


def test_large_fixed_count_does_not_change_native_or_hazard_structure():
    sizes = []
    for count in (2**10, 2**20):
        _, tasks, hazards = analyze(vector_pipeline(count=count))
        sizes.append(
            (
                len(tasks),
                tuple(
                    (
                        h.producer.operation.uid.data,
                        h.consumer.operation.uid.data,
                        h.activation,
                        h.kind,
                        h.memory.start,
                        h.memory.stop,
                    )
                    for h in hazards
                ),
            )
        )
    assert sizes[0] == sizes[1]


def test_adapter_does_not_recompute_accesses_or_merge_distinct_consumers(monkeypatch):
    loop = extract_loop(vector_pipeline(late_reader=True))
    tasks = extract_native_tasks(loop, clock=Clock("step"))
    dependencies = loop_dependencies(loop)

    def fail(*args, **kwargs):
        raise AssertionError("access scan repeated")

    monkeypatch.setattr("xdsl_npu.loop_model.loop_dependencies", fail)
    monkeypatch.setattr("xdsl_npu.loop_model.access_cells", fail)
    hazards = bind_hazards(loop, tasks, dependencies)
    assert len(hazards) == len(dependencies)
    first = select(hazards, "load", "scale", HazardKind.RAW)[0]
    last = select(hazards, "load", "last_read", HazardKind.RAW)[0]
    assert first.consumer_cut.operation is not last.consumer_cut.operation
    assert first.memory.root is last.memory.root


def test_binding_rejects_distinct_clock_declarations_even_with_same_spelling():
    loop = extract_loop(vector_pipeline())
    tasks = extract_native_tasks(loop, clock=Clock("step"))
    different = (*tasks[:-1], replace(tasks[-1], clock=Clock("step")))
    with pytest.raises(CompileError, match="one explicitly bound loop clock"):
        bind_hazards(loop, different, loop_dependencies(loop))
