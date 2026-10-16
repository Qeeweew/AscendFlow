"""Boundary composition checked against explicit finite prefix words."""

from dataclasses import replace
from random import Random

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from xdsl_npu.analysis.frontier import (
    Clock,
    FrontierLayout,
    KnownPrefix,
    PrefixDomain,
    PrefixRank,
)
from xdsl_npu.analysis.regions import (
    BoundaryLayout,
    BoundaryTransfer,
    Output,
    Port,
    RegionWork,
    Requirement,
    compose_sequence,
)
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone


def fixture():
    clock = Clock("physical-step")
    domains = tuple(
        PrefixDomain(Actor.AIV0, pipe, Milestone.RETIREMENT_CAPTURE, clock)
        for pipe in (Engine.MTE2, Engine.V, Engine.MTE3)
    )
    frontier = FrontierLayout(domains)
    ports = tuple(
        Port(Actor.AIV0, pipe, Milestone.ADMISSION) for pipe in (Engine.MTE2, Engine.V, Engine.MTE3)
    )
    return BoundaryLayout(frozenset(ports), frontier), ports, domains, clock


def prefix(domain, offset=0, lexical=0):
    return KnownPrefix(domain, PrefixRank(domain.clock, offset, lexical))


def consumer():
    # An actual IR identity is retained for diagnostics. These tests exercise
    # the transfer algebra, not a native lowering rule or a hardware proof.
    return arith.ConstantOp.from_int_and_width(1, IndexType())


def words(frontier, generation=4):
    """Independent finite ordered events (16 lexical positions per epoch)."""
    return {
        (domain, sequence)
        for domain, rank in frontier.known.items()
        for sequence in range((generation + rank.offset) * 16 + rank.lexical + 1)
    }


def interpret(transfer, inputs):
    for request in transfer.requirements:
        needed = words(transfer.layout.frontier.empty().record(request.prefix))
        supplied = set().union(*(inputs[source] for source in request.sources))
        if not needed <= supplied:
            return None
    return {
        port: words(output.guarantees).union(*(inputs[source] for source in output.sources))
        for port, output in transfer.outputs.items()
    }


def test_identity_is_transparent_and_cannot_complete_a_loop_body():
    layout, ports, domains, _ = fixture()
    incoming = {port: layout.frontier.empty() for port in ports}
    incoming[ports[0]] = incoming[ports[0]].record(prefix(domains[0], -1))
    identity = BoundaryTransfer.identity(layout)
    assert identity.apply(incoming) == incoming
    assert all(not row.guarantees.known for row in identity.outputs.values())
    need = Requirement(frozenset({ports[1]}), prefix(domains[1]), consumer())
    following = BoundaryTransfer(layout, identity.outputs, [need])
    combined = compose_sequence(layout, [identity, following])
    assert combined.missing(incoming) == (need,)
    with pytest.raises(CompileError, match="entry requirements"):
        combined.apply(incoming)


def test_local_guarantee_supplies_following_requirement_at_its_real_consumer():
    layout, ports, domains, _ = fixture()
    empty = layout.frontier.empty()
    identity = BoundaryTransfer.identity(layout)
    before = dict(identity.outputs)
    before[ports[0]] = Output(frozenset({ports[0]}), empty.record(prefix(domains[0], 0, 2)))
    first = BoundaryTransfer(layout, before)
    need = Requirement(frozenset({ports[0]}), prefix(domains[0], 0, 1), consumer())
    after = dict(identity.outputs)
    after[ports[1]] = Output(frozenset({ports[0], ports[1]}), empty)
    second = BoundaryTransfer(layout, after, [need])
    combined = compose_sequence(layout, [first, second])
    incoming = {port: empty for port in ports}
    assert not combined.requirements
    assert combined.apply(incoming) == second.apply(first.apply(incoming))
    assert combined.apply(incoming)[ports[1]].covers(need.prefix)
    # Acquiring MTE2 does not itself complete Vector or unrelated MTE3 work.
    assert not combined.apply(incoming)[ports[1]].covers(prefix(domains[1]))


def test_requirements_pull_back_as_alternatives_without_mixing_required_domains():
    layout, ports, domains, _ = fixture()
    empty = layout.frontier.empty()
    identity = BoundaryTransfer.identity(layout)
    wiring = dict(identity.outputs)
    wiring[ports[2]] = Output(frozenset({ports[0], ports[1]}), empty)
    before = BoundaryTransfer(layout, wiring)
    target = consumer()
    requests = [
        Requirement(frozenset({ports[2]}), prefix(domain), target) for domain in domains[:2]
    ]
    after = BoundaryTransfer(layout, identity.outputs, requests)
    combined = compose_sequence(layout, [before, after])
    assert len(combined.requirements) == 2
    assert all(
        r.sources == frozenset(ports[:2]) and r.consumer is target for r in combined.requirements
    )
    incoming = {port: empty for port in ports}
    incoming[ports[0]] = empty.record(prefix(domains[0]))
    assert len(combined.missing(incoming)) == 1
    incoming[ports[1]] = empty.record(prefix(domains[1]))
    assert not combined.missing(incoming)
    assert combined.apply(incoming) == after.apply(before.apply(incoming))


def test_constant_guarantees_do_not_discharge_earlier_entry_requirements():
    layout, ports, domains, _ = fixture()
    identity = BoundaryTransfer.identity(layout)
    need = Requirement(frozenset({ports[0]}), prefix(domains[0]), consumer())
    rows = dict(identity.outputs)
    rows[ports[0]] = Output(frozenset(), layout.frontier.empty().record(prefix(domains[0], 1)))
    # Read requires an old definition, then this region produces a new one.
    before = BoundaryTransfer(layout, rows, [need])
    after = BoundaryTransfer(layout, identity.outputs, [replace(need, consumer=consumer())])
    combined = compose_sequence(layout, [before, after])
    assert len(combined.requirements) == 1
    assert combined.requirements[0].consumer is need.consumer
    assert combined.missing({port: layout.frontier.empty() for port in ports})


def test_stronger_later_requirement_does_not_erase_the_earlier_insertion_cut():
    layout, ports, domains, _ = fixture()
    identity = BoundaryTransfer.identity(layout)
    first = Requirement(frozenset({ports[0]}), prefix(domains[0]), consumer())
    second = Requirement(frozenset({ports[0]}), prefix(domains[0], 0, 3), consumer())
    result = compose_sequence(
        layout,
        [
            BoundaryTransfer(layout, identity.outputs, [first]),
            BoundaryTransfer(layout, identity.outputs, [second]),
        ],
    )
    assert result.requirements == (first, second)


def test_sequential_composition_matches_finite_event_sets():
    layout, ports, domains, _ = fixture()
    random = Random(12)
    empty = layout.frontier.empty()
    for _ in range(20):
        chain = []
        for _ in range(4):
            rows = {}
            for port in ports:
                selected = frozenset(p for p in ports if random.randrange(2))
                guarantees = empty.record(
                    *(
                        prefix(d, random.randrange(-2, 2), random.randrange(4))
                        for d in domains
                        if random.randrange(2)
                    )
                )
                rows[port] = Output(selected, guarantees)
            requests = [
                Requirement(
                    frozenset({random.choice(ports)}),
                    prefix(random.choice(domains), -2),
                    consumer(),
                )
            ]
            chain.append(BoundaryTransfer(layout, rows, requests))
        combined = compose_sequence(layout, chain)
        incoming = {
            port: empty.record(*(prefix(d, -2, 2) for d in domains if random.randrange(2)))
            for port in ports
        }
        expected = {port: words(value) for port, value in incoming.items()}
        for step in chain:
            expected = interpret(step, expected)
            if expected is None:
                break
        assert bool(combined.missing(incoming)) == (expected is None)
        if expected is not None:
            assert {
                port: words(value) for port, value in combined.apply(incoming).items()
            } == expected


@pytest.mark.parametrize("templates", (32, 64, 128))
def test_boundary_state_is_bounded_and_sequence_work_grows_with_templates(templates):
    layout, ports, domains, _ = fixture()
    rows = dict(BoundaryTransfer.identity(layout).outputs)
    rows[ports[1]] = Output(frozenset(ports), layout.frontier.empty().record(prefix(domains[0])))
    body = BoundaryTransfer(layout, rows)
    work = RegionWork()
    result = compose_sequence(layout, (body for _ in range(templates)), work=work)
    assert result.statistics["output_wires"] <= len(ports) ** 2
    assert result.statistics["output_prefixes"] <= len(ports) * len(domains)
    assert work.output_rows == templates * len(ports)
    assert work.wire_visits == templates * 5
    assert not result.requirements


def test_entry_clause_count_never_multiplies_by_composition_paths():
    layout, ports, domains, _ = fixture()
    rows = {port: Output(frozenset(ports), layout.frontier.empty()) for port in ports}
    target = consumer()
    need = Requirement(frozenset({ports[0]}), prefix(domains[0]), target)
    body = BoundaryTransfer(layout, rows, [need])
    work = RegionWork()
    result = compose_sequence(layout, [body] * 100, work=work)
    assert work.requirement_visits == 100
    assert work.requirement_comparisons <= 100
    assert len(result.requirements) <= 2
    assert result.statistics["output_wires"] == len(ports) ** 2
