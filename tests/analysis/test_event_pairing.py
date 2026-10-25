"""Per-call activation pairing is independent of cross-call resource ranks."""

import pytest

from ascendflow.analysis.access_frontier import QueueEndpoint
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.indexing import arguments
from ascendflow.scalar import Expr
from ascendflow.symbols import Symbol, SymbolKind
from ascendflow.target.ascend910b import Engine
from tests.fixtures.nested import metadata_panels
from tests.support.structured import selected


@pytest.fixture(scope="module")
def panels():
    return selected(metadata_panels)


def test_metadata_activation_pairs_without_a_periodic_ancestor_window(panels, monkeypatch):
    def no_window(*args):
        pytest.fail("actual endpoint pairing must not need a dense event rank")

    monkeypatch.setattr(EventDomains, "_family", no_window)
    domains = EventDomains(panels, ResourceWork())
    assert domains.cases
    assert all(not case.windows for cases in domains.cases.values() for case in cases)
    assert any(domains.active.values())


def test_paired_forwarding_port_still_needs_a_consumption_lifetime(panels):
    domains = EventDomains(panels, ResourceWork()).for_lifetimes()
    channel = next(
        n
        for n in domains.notifications
        if any(need.kind is HazardKind.WAW for need in n.requirements)
        and isinstance(n.consumers[0].occurrence.endpoint, QueueEndpoint)
    )
    rule = HandbackRule(domains)
    lease = rule.lease(channel)
    (case,) = lease.cases
    assert lease.width == 1 and case.policy == "ordered-local-handback"
    assert case.inside and case.reentries and not case.rings
    assert all(r.delta is None for r in (*case.inside, *case.reentries))
    # The later task's metadata is an independent capture, not the old value
    # frozen across a sparse gap. Both may differ while the reuse is active.
    handback = case.reentries[0].handbacks[0]

    def metadata(point):
        return {
            name
            for test in point.guard.comparisons
            for name in arguments(test.lhs) | arguments(test.rhs)
            if isinstance(name, Symbol) and name.kind is SymbolKind.METADATA
        }

    (old,), (later,) = metadata(handback.consume), metadata(handback.next_set)
    assert old._identity == later._identity and old != later
    assert domains.arithmetic.queries.satisfies(
        (
            *handback.condition.comparisons,
            Comparison(Expr("argument", (old,)), "!=", Expr("argument", (later,))),
        )
    ).satisfied
    # Removing real return actions denies the lease even though pairing and
    # physical storage are unchanged. A channel cannot prove its own return.
    broken = HandbackRule(domains)
    broken.peers = tuple(
        n
        for n in broken.peers
        if (n.source_engine, n.destination_engine) != (Engine.MTE3, Engine.V)
    )
    with pytest.raises(CompileError, match="periodic membership"):
        broken.lease(channel)
