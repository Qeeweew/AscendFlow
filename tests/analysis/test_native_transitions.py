"""Native hardware-state counterexamples, without graph/path propagation."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr

from xdsl_npu._physical_builder import Kernel
from xdsl_npu.analysis.frontier import Clock, Handoff, KnownPrefix, PrefixDomain, PrefixRank
from xdsl_npu.analysis.regions import Port
from xdsl_npu.analysis.transitions import (
    InstructionSite,
    barrier_transfer,
    native_transfer,
    set_transfer,
    transition_layout,
    wait_transfer,
)
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def fact(site, milestone=Milestone.RETIREMENT_CAPTURE):
    engine = None if milestone is Milestone.ISSUE else site.engine
    return KnownPrefix(
        PrefixDomain(site.actor, engine, milestone, site.clock, site.partition), site.rank
    )


def port(site, milestone):
    return Port(site.actor, None if milestone is Milestone.ISSUE else site.engine, milestone)


def empty_inputs(layout):
    return {key: layout.frontier.empty() for key in layout.ports}


def scalar_operation():
    kernel = Kernel("scalar_transition")
    data = kernel.buffer("scalar", "UB", 32, initialized=True)
    kernel.loop(1, symbolic=True)
    operation = kernel.task("read", "scalar_load", kernel.view(data, 4))
    kernel.finish().verify()
    return operation


def aic_load_site(clock):
    kernel = Kernel("aic_read", core="aic", disjoint_external=True)
    source = kernel.buffer("X", "GM", 512, "f16", external=True, initialized=True)
    destination = kernel.buffer("a", "L1", 512, "f16")
    kernel.loop(1, symbolic=True)
    operation = kernel.copy("load", kernel.view(source, 512), kernel.view(destination, 512))
    kernel.finish().verify()
    return InstructionSite(operation, Actor.AIC, Engine.MTE2, PrefixRank(clock))


@pytest.fixture
def scale():
    kernel = Kernel("native_transition_scale", disjoint_external=True)
    source = kernel.buffer("X", "GM", 256, external=True, initialized=True)
    destination = kernel.buffer("Y", "GM", 256, external=True)
    a, b = kernel.buffer("a", "UB", 256), kernel.buffer("b", "UB", 256)
    kernel.loop(1, symbolic=True)
    load = kernel.copy("load", kernel.view(source, 256), kernel.view(a, 256))
    vector = kernel.scale("scale", kernel.view(a, 256), kernel.view(b, 256), 2.0)
    store = kernel.copy("store", kernel.view(b, 256), kernel.view(destination, 256))
    kernel.finish().verify()
    clock = Clock("physical-step")
    sites = tuple(
        InstructionSite(operation, Actor.AIV0, engine, PrefixRank(clock, 0, index))
        for index, (operation, engine) in enumerate(
            (
                (load, Engine.MTE2),
                (vector, Engine.V),
                (store, Engine.MTE3),
            )
        )
    )
    layout = transition_layout({Actor.AIV0: (Engine.MTE2, Engine.V, Engine.MTE3)}, clock)
    return layout, sites


def test_scale_requires_real_load_capture_and_acquire_before_vector_admission(scale):
    layout, (load, vector, _) = scale
    inputs = empty_inputs(layout)
    load_rule, vector_rule = native_transfer(layout, load), native_transfer(layout, vector)
    after_load = load_rule.boundary.apply(inputs)
    assert after_load[port(load, Milestone.RETIREMENT_CAPTURE)].covers(fact(load))
    # A future source retirement state is not already known to the Vector.
    assert not vector_rule.admission_state(after_load).covers(fact(load))
    handoff = Handoff("load_ready", load.clock)
    selected_set = set_transfer(layout, load, handoff)
    notification = selected_set.notification(after_load)
    after_set = selected_set.boundary.apply(after_load)
    assert notification.frontier.covers(fact(load))
    assert notification.frontier.covers(fact(load, Milestone.ADMISSION))
    assert not vector_rule.admission_state(after_set).covers(fact(load))
    selected_wait = wait_transfer(layout, vector, notification, handoff)
    after_wait = selected_wait.boundary.apply(after_set)
    assert vector_rule.admission_state(after_wait).covers(fact(load))
    assert not after_wait[port(vector, Milestone.ADMISSION)].covers(
        fact(vector, Milestone.ADMISSION)
    )
    after_vector = vector_rule.boundary.apply(after_wait)
    assert after_vector[port(vector, Milestone.ADMISSION)].covers(fact(vector, Milestone.ADMISSION))


def test_scale_store_gets_vector_retirement_only_through_selected_set_wait(scale):
    layout, (load, vector, store) = scale
    states = native_transfer(layout, load).boundary.apply(empty_inputs(layout))
    load_handoff = Handoff("load_ready", load.clock)
    load_set = set_transfer(layout, load, load_handoff)
    payload = load_set.notification(states)
    states = load_set.boundary.apply(states)
    states = wait_transfer(layout, vector, payload, load_handoff).boundary.apply(states)
    states = native_transfer(layout, vector).boundary.apply(states)
    store_rule = native_transfer(layout, store)
    assert not store_rule.admission_state(states).covers(fact(vector))
    vector_handoff = Handoff("vector_ready", vector.clock)
    vector_set = set_transfer(layout, vector, vector_handoff)
    payload = vector_set.notification(states)
    assert payload.frontier.covers(fact(vector))
    states = vector_set.boundary.apply(states)
    states = wait_transfer(layout, store, payload, vector_handoff).boundary.apply(states)
    assert store_rule.admission_state(states).covers(fact(vector))
    assert not store_rule.admission_state(states).covers(fact(store))


@pytest.fixture
def mixed_mad():
    kernel = Kernel("short_large_mad", core="aic")
    a = kernel.buffer("a", "L0A", 512, "f16", layout="ZZ", initialized=True)
    b = kernel.buffer("b", "L0B", 512, "f16", layout="ZN", initialized=True)
    c = kernel.buffer("c", "L0C", 1024, layout="NZ")
    large_a = kernel.buffer("large_a", "L0A", 2048, "f16", layout="ZZ", initialized=True)
    large_b = kernel.buffer("large_b", "L0B", 2048, "f16", layout="ZN", initialized=True)
    large_c = kernel.buffer("large_c", "L0C", 16384, layout="NZ")
    kernel.loop(1, symbolic=True)
    first = kernel.mad("short_a", kernel.view(a, 512), kernel.view(b, 512), kernel.view(c, 1024))
    second = kernel.task(
        "large_b",
        "mad",
        kernel.view(large_a, 2048),
        kernel.view(large_b, 2048),
        kernel.view(large_c, 16384),
        params=(64, 64, 16, 1),
    )
    third = kernel.mad(
        "later_a", kernel.view(a, 512), kernel.view(b, 512), kernel.view(c, 1024), clear=False
    )
    kernel.finish().verify()
    clock = Clock("mad-step")
    sites = tuple(
        InstructionSite(operation, Actor.AIC, Engine.M, PrefixRank(clock, 0, index))
        for index, operation in enumerate((first, second, third))
    )
    layout = transition_layout({Actor.AIC: (Engine.M,)}, clock)
    return layout, sites


def test_large_mad_implicit_end_does_not_retire_previous_unrelated_short_mad(mixed_mad):
    layout, (short, large, later) = mixed_mad
    states = native_transfer(layout, short).boundary.apply(empty_inputs(layout))
    assert not states[port(short, Milestone.IMPLICIT_RETIREMENT)].known
    large_rule = native_transfer(layout, large)
    assert not large_rule.admission_state(states).covers(fact(short))
    states = large_rule.boundary.apply(states)
    implicit = states[port(large, Milestone.IMPLICIT_RETIREMENT)]
    assert implicit.covers(fact(large, Milestone.IMPLICIT_RETIREMENT))
    assert not implicit.covers(fact(short))
    assert not implicit.covers(fact(large))
    assert not native_transfer(layout, later).admission_state(states).covers(fact(short))
    # A real fence joins prior retirement and the actual source's end. It
    # supplies the missing cumulative capture without a short-MAD FIFO claim.
    states = barrier_transfer(layout, large).boundary.apply(states)
    assert native_transfer(layout, later).admission_state(states).covers(fact(short))
    assert native_transfer(layout, later).admission_state(states).covers(fact(large))


def test_set_before_later_source_task_does_not_forward_its_effect_into_fifo_begin(mixed_mad):
    layout, (first, second, _) = mixed_mad
    states = native_transfer(layout, first).boundary.apply(empty_inputs(layout))
    handoff = Handoff("source_ready", first.clock)
    selected = set_transfer(layout, first, handoff)
    capture = selected.notification(states)
    assert capture.frontier.covers(fact(first))
    after_set = selected.boundary.apply(states)
    assert (
        after_set[port(first, Milestone.RETIREMENT_CAPTURE)]
        == (states[port(first, Milestone.RETIREMENT_CAPTURE)])
    )
    assert not capture.frontier.covers(fact(second))
    assert not native_transfer(layout, second).admission_state(after_set).covers(fact(first))
    assert not after_set[port(first, Milestone.IMPLICIT_RETIREMENT)].covers(fact(first))
    assert not after_set[port(first, Milestone.ISSUE)].covers(fact(first))


def test_scalar_native_end_advances_controller_issue_but_not_other_engines():
    clock = Clock("scalar-step")
    site = InstructionSite(scalar_operation(), Actor.AIV0, Engine.S, PrefixRank(clock))
    layout = transition_layout({Actor.AIV0: (Engine.S, Engine.V)}, clock)
    states = native_transfer(layout, site).boundary.apply(empty_inputs(layout))
    controller = states[port(site, Milestone.ISSUE)]
    assert controller.covers(fact(site, Milestone.ISSUE))
    assert controller.covers(fact(site, Milestone.IMPLICIT_RETIREMENT))
    assert not states[Port(Actor.AIV0, Engine.V, Milestone.ADMISSION)].known


@pytest.mark.parametrize("cross", (False, True))
def test_wait_updates_controller_only_for_cross_wait_or_scalar_wait(scale, cross):
    layout, (load, vector, _) = scale
    if cross:
        load = aic_load_site(load.clock)
        layout = transition_layout({Actor.AIC: (Engine.MTE2,), Actor.AIV0: (Engine.V,)}, load.clock)
    handoff = Handoff("ready", load.clock)
    states = native_transfer(layout, load).boundary.apply(empty_inputs(layout))
    capture = set_transfer(layout, load, handoff).notification(states)
    after = wait_transfer(layout, vector, capture, handoff, cross=cross).boundary.apply(states)
    assert after[port(vector, Milestone.ADMISSION)].covers(fact(load))
    assert after[port(vector, Milestone.ISSUE)].covers(fact(load)) is cross
    assert not after[port(vector, Milestone.RETIREMENT_CAPTURE)].covers(fact(load))
    assert not after[port(vector, Milestone.ISSUE)].covers(fact(vector, Milestone.ISSUE))


def test_scalar_wait_acquired_facts_reach_controller_after_actual_admission(scale):
    _, (load, _, _) = scale
    scalar_site = InstructionSite(
        scalar_operation(), Actor.AIV0, Engine.S, PrefixRank(load.clock, 0, 1)
    )
    layout = transition_layout({Actor.AIV0: (Engine.MTE2, Engine.S)}, load.clock)
    states = native_transfer(layout, load).boundary.apply(empty_inputs(layout))
    handoff = Handoff("scalar_ready", load.clock)
    capture = set_transfer(layout, load, handoff).notification(states)
    # This site is the real before-consumer anchor; native admission has not
    # happened. Wait's rule uses its pipe, never pretends that task completed.
    states = wait_transfer(layout, scalar_site, capture, handoff).boundary.apply(states)
    assert states[port(scalar_site, Milestone.ISSUE)].covers(fact(load))
    assert not states[port(scalar_site, Milestone.ADMISSION)].covers(
        fact(scalar_site, Milestone.ADMISSION)
    )


def test_native_end_anchor_alone_cannot_supply_a_notification(scale):
    layout, (load, _, _) = scale
    with pytest.raises(CompileError, match="only a real Set"):
        native_transfer(layout, load).notification(empty_inputs(layout))
    with pytest.raises(CompileError, match="actual TaskOp"):
        replace(load, operation=fact(load))


def test_wait_rejects_a_different_handoff_epoch(scale):
    layout, (load, vector, _) = scale
    states = native_transfer(layout, load).boundary.apply(empty_inputs(layout))
    handoff = Handoff("ready", load.clock)
    capture = set_transfer(layout, load, handoff).notification(states)
    with pytest.raises(CompileError, match="expected handoff"):
        wait_transfer(layout, vector, capture, replace(handoff, ordinal=1))


def test_wait_mechanism_must_match_the_actual_actor_and_engine_direction(scale):
    layout, (load, vector, _) = scale
    states = native_transfer(layout, load).boundary.apply(empty_inputs(layout))
    handoff = Handoff("ready", load.clock)
    capture = set_transfer(layout, load, handoff).notification(states)
    with pytest.raises(CompileError, match="actual source and destination actors"):
        wait_transfer(layout, vector, capture, handoff, cross=True)
    with pytest.raises(CompileError, match="different engines"):
        wait_transfer(layout, load, capture, handoff)


def test_bound_rank_is_used_directly_without_adding_the_operation_stage_again(scale):
    layout, (_, vector, _) = scale
    vector.operation.attributes["stage"] = IntAttr(7)
    bound = replace(vector, rank=PrefixRank(vector.clock, -3, 17))
    result = native_transfer(layout, bound).admission_state(empty_inputs(layout))
    assert result.covers(fact(bound, Milestone.ADMISSION))
    domain = fact(bound, Milestone.ADMISSION).domain
    assert result.known[domain] == PrefixRank(vector.clock, -3, 17)


def test_foreign_clock_or_undeclared_engine_never_falls_back_to_a_graph(scale):
    layout, (load, _, _) = scale
    with pytest.raises(CompileError, match="undeclared") as error:
        native_transfer(layout, replace(load, rank=PrefixRank(Clock(load.clock.name))))
    assert error.value.kind is ErrorKind.UNSUPPORTED
    with pytest.raises(CompileError, match="not bound") as error:
        set_transfer(layout, load, Handoff("ready", Clock(load.clock.name)))
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_transition_state_is_bounded_by_declared_hardware_ports_and_domains(scale):
    layout, (load, vector, _) = scale
    states = empty_inputs(layout)
    for index in range(100):
        source = replace(load, rank=PrefixRank(load.clock, index, 0))
        consumer = replace(vector, rank=PrefixRank(vector.clock, index, 1))
        handoff = Handoff("ready", load.clock, ordinal=index)
        states = native_transfer(layout, source).boundary.apply(states)
        selected = set_transfer(layout, source, handoff)
        capture = selected.notification(states)
        states = selected.boundary.apply(states)
        states = wait_transfer(layout, consumer, capture, handoff).boundary.apply(states)
        states = native_transfer(layout, consumer).boundary.apply(states)
        assert len(states) == len(layout.ports)
        assert sum(len(value) for value in states.values()) <= (
            len(layout.ports) * layout.frontier.capacity
        )
