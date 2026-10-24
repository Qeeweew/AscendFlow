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
def test_new_cuts_only_add_rules_without_rebuilding_established_relations(source):
    graph = ProtocolTransfers(prepared(selected(source)))
    original_rules, original_symbols = tuple(graph.rules), tuple(graph.propositions)
    for episodes in graph.episodes.values():
        for episode in episodes:
            for point in episode.consumers:
                graph.snapshot(point, episode.notification.destination_engine, strict=True)
    graph._compile_rules()
    assert len(graph.rules) == len(graph.propositions) == len(graph.nodes)
    assert all(a is b for a, b in zip(original_rules, graph.rules))
    assert all(a is b for a, b in zip(original_symbols, graph.propositions))
    assert graph.work.transfer_rules == len(graph.nodes)
    count = graph.work.transfer_rules
    graph._compile_rules()
    assert graph.work.transfer_rules == count


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
                                    expected[node] = number, premise
                        seeds = tuple(graph._capture_seeds(origin, excluded))
                        assert {rule.node for rule in seeds} == set(expected)
                        for rule in seeds:
                            number, premise = expected[rule.node]
                            assert rule.seed
                            assert rule.conclusion is graph.propositions[number]
                            assumptions = tuple(
                                dict.fromkeys(
                                    (*point.guard.comparisons, *domains._input_ranges(point))
                                )
                            )
                            assert domains.arithmetic.queries.implies(
                                rule.premise, (*assumptions, premise)
                            ).proved
                            assert domains.arithmetic.queries.implies(
                                premise, (*assumptions, rule.premise)
                            ).proved
