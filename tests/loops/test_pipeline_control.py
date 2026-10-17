"""One retained body across PLE, with an independent finite dispatch oracle."""

from collections import Counter

import pytest
from xdsl.dialects import scf
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException

from ascendflow import language as N
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.io import create_context, dump, parse
from ascendflow.dialects.loops import LoopScheduleAttr, PhaseKind
from ascendflow.dialects.source import PipelineOp, StageOp
from ascendflow.passes.lower_pipeline import (
    SCHEDULE_ATTRIBUTE,
    LowerPipelinePass,
    normalize_loop_controls,
)
from examples.vector.scale import scale


@pytest.mark.parametrize("depth", (0, 1, 2, 4, 32))
def test_phases_preserve_each_stage_instance_and_physical_issue_order(depth):
    schedule = LoopScheduleAttr(IntAttr(depth))
    for count in sorted({0, 1, 2, depth, depth + 1, 2 * depth + 1}):
        # Independent oracle: place every logical operation on its physical
        # step, then sort by step and a deliberately non-stage-sorted policy.
        order = tuple(reversed(range(depth + 1)))
        expected = sorted(
            (
                (iteration + stage, rank, iteration, stage)
                for rank, stage in enumerate(order)
                for iteration in range(count)
            )
        )
        actual = []
        for phase in schedule.phases:
            interval = phase.interval(count)
            if interval is None:
                continue
            for step in range(*interval):
                for rank, stage in enumerate(order):
                    logical = step - stage
                    if 0 <= logical < count:
                        actual.append((step, rank, logical, stage))
        assert actual == expected
        assert len(actual) == count * (depth + 1)
        assert len(schedule.phases) <= 4


def test_depth_and_trip_count_equal_has_empty_steady_identity():
    schedule = LoopScheduleAttr(IntAttr(3))
    main = next(p for p in schedule.phases if p.kind == PhaseKind.LOOP)
    assert main.interval(3) == (3, 3)
    assert all(phase.interval(0) is None for phase in schedule.phases)


def test_stage_depth_does_not_remove_release_distance_boundary_guards():
    schedule = LoopScheduleAttr(IntAttr(0))
    for count in (0, 1, 2, 3, 8):
        generations = [
            g
            for phase in schedule.phases
            if (interval := phase.interval(count)) is not None
            for g in range(*interval)
        ]
        # Data reuse distance is two even though there is no stage filling.
        sends = [g for g in generations if g + 2 < count]
        waits = [g - 2 for g in generations if g >= 2]
        assert sends == waits == list(range(max(0, count - 2)))


def test_schedule_attribute_round_trips_and_no_source_body_is_cloned():
    module = scale.to_ir()
    before = tuple(module.walk())
    controls = normalize_loop_controls(module)
    assert tuple(module.walk()) == before
    assert controls.module is module
    assert len(controls.loops) == 1
    assert controls.loops[0].schedule.horizon.data == 2
    assert controls.statistics["body_clones"] == 0
    module.verify()
    text = dump(module)
    assert "#npu.loop_schedule<" in text
    assert dump(parse(text)) == text
    LowerPipelinePass().apply(create_context(), module)
    assert dump(module) == text


NESTED = """
def nested(n: N.int64, k: N.int64, PARTS: N.constexpr=3):
    with N.Kernel():
        x=N.alloc_ub((64,))
        N.fill(x,0.0)
        for outer in N.serial(n):
            N.fill(x,1.0)
            for middle in N.Pipelined(k,stages=3):
                with N.Stage(2,name="child"):
                    N.fill(x,2.0)
                    for inner in N.Pipelined(PARTS,stages=2):
                        with N.Stage(1,name="work"):
                            N.fill(x,3.0)
                    N.fill(x,4.0)
            N.fill(x,5.0)
        N.fill(x,6.0)
        for other in N.serial(2,11,3):
            N.fill(x,7.0)
"""


def test_arbitrary_loop_surrounding_work_and_ancestor_coordinates_remain_actual_ir():
    module = N.from_source(NESTED).to_ir()
    before = tuple(module.walk())
    controls = normalize_loop_controls(module)
    outer, middle, inner, sibling = controls.loops
    assert tuple(module.walk()) == before
    assert [loop.depth for loop in controls.loops] == [1, 2, 3, 1]
    assert outer.parent is sibling.parent is None
    assert middle.parent is outer and middle.parent_stage == 0
    assert inner.parent is middle and inner.parent_stage == 2
    assert inner.domain.upper == 3
    assert isinstance(outer.operation, scf.ForOp)
    assert isinstance(middle.operation, PipelineOp)
    assert sibling.domain.concrete({}) == (2, 11, 3)
    assert [sibling.source_iteration(step, 0, {}) for step in range(4)] == [2, 5, 8, None]
    assert inner.source_iteration(0, 1, {}) is None
    assert inner.source_iteration(1, 1, {}) == 0
    assert inner.source_iteration(4, 1, {}) is None
    # Both actual runtime operands survive; no constexpr specialization of n/k.
    assert outer.domain.trip_count({outer.domain.upper: 1_000_000}) == 1_000_000
    assert middle.domain.trip_count({middle.domain.upper: 0}) == 0
    module.verify()


def test_large_constant_loop_uses_identical_control_work():
    records = []
    for parts in (1024, 1_048_576):
        module = N.from_source(NESTED).specialize(PARTS=parts).to_ir()
        controls = normalize_loop_controls(module)
        records.append(controls.statistics)
        assert controls.loops[2].domain.upper == parts
    assert records[0] == records[1]


@pytest.mark.parametrize("depth", (1, 2, 4))
def test_nested_bodies_have_additive_phase_size_not_three_to_the_depth(depth):
    source = ["def nested(n: N.int64):", "    with N.Kernel():", "        x=N.alloc_ub((64,))"]
    for level in range(depth):
        indent = " " * (8 + 8 * level)
        source.append(indent + f"for i{level} in N.Pipelined(n,stages=3):")
        source.append(indent + "    with N.Stage(2,name='work'):")
    source.append(" " * (8 + 8 * depth) + "N.fill(x,0.0)")
    module = N.from_source("\n".join(source)).to_ir()
    controls = normalize_loop_controls(module)
    assert len(controls.loops) == depth
    assert controls.phase_descriptors == 4 * depth
    assert controls.statistics["maximum_depth"] == depth
    assert sum(op.name == "npu_source.element" for op in module.walk()) == 1
    assert Counter(type(op) for op in module.walk())[StageOp] == depth
    # Reading a retained contract never initiates a new normalization pass.
    stats = controls.statistics
    for _ in range(100):
        for control in controls.loops:
            assert controls.by_operation[control.operation] is control
    assert controls.statistics == stats


def test_scalar_carry_is_retained_as_control_not_silently_accepted_as_sync():
    module = N.from_source("""
        def carry(n: N.int64):
            with N.Kernel():
                total=0
                for i in N.serial(n):
                    total=total+i
    """).to_ir()
    (control,) = normalize_loop_controls(module).loops
    assert len(control.operation.iter_args) == len(control.operation.results) == 1
    assert len(control.operation.body.block.args) == 2
    module.verify()


def test_dynamic_step_diagnostic_before_partial_mutation():
    module = N.from_source(NESTED).to_ir()
    loops = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    first, last = loops
    last.operands = [last.lb, last.ub, first.ub]
    with pytest.raises(CompileError, match="positive constant step") as failure:
        normalize_loop_controls(module)
    assert failure.value.kind == ErrorKind.UNSUPPORTED
    assert failure.value.phase == "loop-control"
    assert all(SCHEDULE_ATTRIBUTE not in op.attributes for op in module.walk())


def test_stale_attribute_is_not_a_proof_and_invalid_depth_is_rejected():
    module = scale.to_ir()
    pipeline = next(op for op in module.walk() if isinstance(op, PipelineOp))
    pipeline.attributes[SCHEDULE_ATTRIBUTE] = LoopScheduleAttr(IntAttr(0))
    normalize_loop_controls(module)
    assert pipeline.attributes[SCHEDULE_ATTRIBUTE].horizon.data == 2
    with pytest.raises(VerifyException, match="stage offset"):
        LoopScheduleAttr(IntAttr(33))
