"""Capture bands and append-only rules preserve the original transfer relation."""

import pytest

from ascendflow.analysis.protocol_transfers import (
    CompletionOrigin,
    ConsumptionOrigin,
    ProtocolTransfers,
)
from ascendflow.analysis.smt import all_of
from tests.fixtures.nested import nested_copy, nested_cv
from tests.support.structured import prepared, selected
from tests.sync.test_completion_transfers import chain
from tests.sync.test_consumption_transfers import cube_chain


@pytest.mark.parametrize(
    "source", (nested_copy, nested_cv, chain(runtime=True, branch=True), cube_chain())
)
def test_new_cuts_inline_prefixes_without_new_causal_states(source):
    graph = ProtocolTransfers(prepared(selected(source)))
    assert not graph.rules and not graph.propositions
    requested = {i for i, node in enumerate(graph.nodes) if node.kind == "capture"}
    original_rules = {i: graph._rule(i) for i in requested}
    original_symbols = dict(graph.propositions)
    for episodes in graph.episodes.values():
        for episode in episodes:
            for point in episode.consumers:
                cut = graph.snapshot(point, episode.notification.destination_engine, strict=True)
                graph._proposition(cut)
    assert set(graph.rules) == requested
    assert all(graph.rules[i] is rule for i, rule in original_rules.items())
    assert all(graph.propositions[i] is symbol for i, symbol in original_symbols.items())
    assert graph.work.transfer_rules == len(requested)
    count = graph.work.transfer_rules
    for number in requested:
        graph._rule(number)
    assert graph.work.transfer_rules == count
    assert all(graph.nodes[i].kind != "union" for i in graph.rules)


@pytest.mark.parametrize(
    "source", (nested_copy, nested_cv, chain(runtime=True, branch=True), cube_chain())
)
def test_capture_bands_equal_exhaustive_route_activation_and_strict_cut_rules(source):
    domains = prepared(selected(source))
    graph = ProtocolTransfers(domains)
    for episodes in graph.episodes.values():
        for episode in episodes:
            notification = episode.notification
            for point in episode.consumers:
                for origin in (
                    CompletionOrigin(point, notification.destination_engine),
                    ConsumptionOrigin(point, notification),
                ):
                    for excluded in (frozenset(), frozenset((notification,))):
                        expected = {}
                        for number, node in enumerate(graph.nodes):
                            if (
                                node.kind == "capture"
                                and node.notification not in excluded
                                and graph._origin_matches(
                                    origin, node.point.actor, node.notification.source_engine
                                )
                            ):
                                premise = all_of(
                                    node.condition, graph._origin_order(origin, node.point)
                                )
                                if premise is not False:
                                    expected[number] = premise
                        seeds = tuple(graph._capture_seeds(origin, excluded))
                        assert {number for number, _ in seeds} == set(expected)
                        for number, actual in seeds:
                            premise = expected[number]
                            assumptions = tuple(
                                dict.fromkeys(
                                    (*point.guard.comparisons, *domains._input_ranges(point))
                                )
                            )
                            assert domains.arithmetic.queries.implies(
                                actual, (*assumptions, premise)
                            ).proved
                            assert domains.arithmetic.queries.implies(
                                premise, (*assumptions, actual)
                            ).proved
