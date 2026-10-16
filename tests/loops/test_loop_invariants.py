"""Explicit loop induction with no graph, trip-count expansion or fixpoint."""

from dataclasses import FrozenInstanceError, replace

import pytest
from xdsl.dialects.builtin import ModuleOp

from xdsl_npu.analysis.frontier import (
    Clock,
    Frontier,
    FrontierLayout,
    KnownPrefix,
    Partition,
    PrefixDomain,
    PrefixRank,
)
from xdsl_npu.analysis.loops import (
    LoopInvariantError,
    LoopWork,
    check_loop_invariant,
    zero_iterations,
)
from xdsl_npu.analysis.regions import BoundaryLayout, BoundaryTransfer, Output, Port, Requirement
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def prefix(key, offset=0, lexical=0):
    return KnownPrefix(key, PrefixRank(key.clock, offset, lexical))


def fixture(*, guard="all", external=False):
    clock = Clock("loop-step")
    port = Port(Actor.AIV0, Engine.V, Milestone.ADMISSION)
    key = PrefixDomain(
        Actor.AIV0, Engine.V, Milestone.RETIREMENT_CAPTURE, clock, Partition(guard=guard)
    )
    keys = [key]
    if external:
        keys.append(replace(key, clock=Clock("outer-fixed")))
    layout = BoundaryLayout(frozenset({port}), FrontierLayout(tuple(keys)))
    consumer = ModuleOp([])
    requirement = Requirement(frozenset({port}), prefix(key, -1), consumer)
    guarantee = Frontier(layout.frontier, (prefix(key),))
    body = BoundaryTransfer(layout, {port: Output(frozenset({port}), guarantee)}, (requirement,))
    initial = [prefix(key, -1)]
    if external:
        initial.append(prefix(keys[1], 7, 2))
    candidate = {port: Frontier(layout.frontier, initial)}
    return clock, port, key, layout, consumer, body, candidate


def test_candidate_is_checked_once_and_preserves_original_body_and_consumer_witnesses():
    clock, port, key, _, consumer, body, candidate = fixture()
    work = LoopWork()
    invariant = check_loop_invariant(body, clock, candidate, work=work)
    assert invariant.body is body
    assert invariant.requirements is body.requirements
    assert invariant.requirements[0].consumer is consumer
    assert invariant.candidate[port] is candidate[port]
    assert invariant.next_candidate[port].covers(prefix(key))
    assert invariant.after_one[port].covers(prefix(key))
    assert work.body_applications == work.rebound_prefixes == work.preservation_checks == 1
    assert invariant.check_entry(candidate, work=work)[port] is candidate[port]
    assert work.entry_checks == 1
    assert invariant.statistics["fixed_point_rounds"] == 0
    assert invariant.statistics["expanded_iterations"] == 0


def test_missing_carried_fact_reports_the_actual_native_consumer_before_applying_body():
    clock, port, _, layout, consumer, body, _ = fixture()
    work = LoopWork()
    with pytest.raises(LoopInvariantError, match="actual body consumers") as error:
        check_loop_invariant(body, clock, {port: layout.frontier.empty()}, work=work)
    assert error.value.kind is ErrorKind.UNSUPPORTED
    assert error.value.body is body
    assert error.value.missing_requirements[0] is body.requirements[0]
    assert error.value.missing_requirements[0].consumer is consumer
    assert work.body_applications == 0


def test_transparent_body_cannot_claim_retirement_progress_from_control_repetition():
    clock, port, key, layout, _, _, candidate = fixture()
    identity = BoundaryTransfer.identity(layout)
    work = LoopWork()
    with pytest.raises(LoopInvariantError, match="next-generation") as error:
        check_loop_invariant(identity, clock, candidate, work=work)
    assert error.value.missing_carry[0].port == port
    assert error.value.missing_carry[0].prefix == prefix(key)
    assert work.body_applications == 1


def test_initial_state_must_establish_candidate_and_is_not_assumed_from_step_preservation():
    clock, port, _, layout, _, body, candidate = fixture()
    invariant = check_loop_invariant(body, clock, candidate)
    with pytest.raises(LoopInvariantError, match="initial state") as error:
        invariant.check_entry({port: layout.frontier.empty()})
    assert error.value.body is body
    assert error.value.missing_carry


def test_external_clock_prefix_is_preserved_as_parameter_without_advancing_its_epoch():
    clock, port, key, _, _, body, candidate = fixture(external=True)
    external = next(domain for domain in candidate[port].known if domain.clock != clock)
    before = candidate[port].known[external]
    invariant = check_loop_invariant(body, clock, candidate)
    assert invariant.next_candidate[port].known[external] is before
    assert invariant.after_one[port].known[external] is before
    assert invariant.next_candidate[port].known[key].offset == 0
    assert invariant.next_candidate[port].known[external].offset == 7


def test_external_clock_fact_cannot_substitute_for_missing_loop_clock_progress():
    clock, port, key, layout, _, body, candidate = fixture(external=True)
    external = next(domain for domain in candidate[port].known if domain.clock != clock)
    wrong = BoundaryTransfer(
        layout,
        {port: Output(frozenset({port}), Frontier(layout.frontier, (prefix(external, 10**12),)))},
        body.requirements,
    )
    with pytest.raises(LoopInvariantError, match="next-generation") as error:
        check_loop_invariant(wrong, clock, candidate)
    assert tuple(item.prefix for item in error.value.missing_carry) == (prefix(key),)


def test_a_new_clock_with_the_same_name_is_not_the_body_loop_declaration():
    clock, _, _, _, _, body, candidate = fixture()
    with pytest.raises(CompileError, match="not the boundary's declared clock") as error:
        check_loop_invariant(body, Clock(clock.name), candidate)
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_zero_iterations_are_transparent_even_when_body_entry_requirements_are_missing():
    _, port, _, layout, _, body, _ = fixture()
    empty = {port: layout.frontier.empty()}
    assert body.missing(empty)
    zero = zero_iterations(layout, empty)
    assert zero[port] is empty[port]
    assert not zero[port].known
    assert len(body.requirements) == 1


def test_zero_iterations_keep_old_completion_instead_of_body_guarantees():
    _, port, key, layout, _, _, candidate = fixture()
    zero = zero_iterations(layout, candidate)
    assert zero[port] is candidate[port]
    assert zero[port].covers(prefix(key, -1))
    assert not zero[port].covers(prefix(key))


def test_one_exact_guard_partition_is_required_without_guard_implication():
    clock, _, _, _, _, body, candidate = fixture(guard="steady")
    check_loop_invariant(body, clock, candidate, guard="steady")
    with pytest.raises(CompileError, match="exact guard partition") as error:
        check_loop_invariant(body, clock, candidate)
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_or_alternative_ports_cover_body_requirement_but_both_carries_are_checked():
    clock, port, key, layout, consumer, _, candidate = fixture()
    other = Port(Actor.AIV0, Engine.MTE2, Milestone.ADMISSION)
    layout = BoundaryLayout(frozenset({port, other}), layout.frontier)
    need = Requirement(frozenset({port, other}), prefix(key, -1), consumer)
    body = BoundaryTransfer(
        layout,
        {
            port: Output(frozenset({port, other}), Frontier(layout.frontier, (prefix(key),))),
            other: Output(frozenset({other}), layout.frontier.empty()),
        },
        (need,),
    )
    candidate[other] = layout.frontier.empty()
    result = check_loop_invariant(body, clock, candidate)
    assert result.requirements[0] is need
    assert not result.next_candidate[other].known


def test_invariant_bindings_are_owned_immutable_snapshots_without_cloning_body_ir():
    clock, port, _, _, consumer, body, candidate = fixture()
    original_state = candidate[port]
    attributes = dict(consumer.attributes)
    invariant = check_loop_invariant(body, clock, candidate)
    candidate.clear()
    assert invariant.candidate[port] is original_state
    assert invariant.body is body
    assert dict(consumer.attributes) == attributes
    with pytest.raises(TypeError):
        invariant.candidate[port] = original_state
    with pytest.raises(FrozenInstanceError):
        invariant.body = body


def test_large_generation_and_trip_count_do_not_create_analysis_instances():
    clock, port, key, _, _, body, candidate = fixture()
    records = []
    for count in (1, 1024, 1_048_576, 10**12):
        work = LoopWork()
        result = check_loop_invariant(body, clock, candidate, work=work)
        records.append((result.statistics, work))
        # Independent arithmetic interpretation at an arbitrary final step.
        # No execution prefix of length count is constructed by this check.
        final_step = count - 1
        produced = result.after_one[port].known[key]
        required = result.next_candidate[port].known[key]
        assert (final_step + produced.offset, produced.lexical) >= (
            final_step + required.offset,
            required.lexical,
        )
    assert all(record == records[0] for record in records)


def test_empty_invariant_and_identity_body_need_no_artificial_progress():
    clock, port, _, layout, _, _, _ = fixture()
    empty = {port: layout.frontier.empty()}
    result = check_loop_invariant(BoundaryTransfer.identity(layout), clock, empty)
    assert not result.next_candidate[port].known


def test_structurally_wrong_port_bindings_are_invalid_input():
    clock, _, _, _, _, body, _ = fixture()
    with pytest.raises(CompileError, match="exactly its boundary ports") as error:
        check_loop_invariant(body, clock, {})
    assert error.value.kind is ErrorKind.INVALID
