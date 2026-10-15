"""Conditional path composition, checked against finite independent reachability."""

import random
from collections import defaultdict
from dataclasses import replace
from math import lcm

import pytest

from xdsl_npu.completion import (
    CountDomain,
    Position,
    Site,
    covers,
    intersection,
    uncovered,
)
from xdsl_npu.completion_flow import CompletionFlow


def site(name, *, call=0, point="end"):
    return Site(("aic", name), Position(False, 0), point, region=name, call=call)


def prove(source, target, edges, domain=None):
    domain = CountDomain(1, 0, minimum=0) if domain is None else domain
    flow = CompletionFlow()
    values = {source, target, *edges}
    values.update(dst for items in edges.values() for dst, _ in items)
    nodes = {value: flow.node() for value in values}
    for value, items in edges.items():
        for dst, predicate in items:
            flow.edge(nodes[value], nodes[dst], predicate)
    flow.add_fact(nodes[source], domain, 1)
    flow.propagate()
    return covers(domain, [g for g, bits in flow.facts[nodes[target]].items() if bits & 1])


def test_disjoint_guards_cannot_be_composed_as_one_unconditional_path():
    a, b, c = (site(n) for n in "abc")
    even, odd = CountDomain(2, 0, minimum=0), CountDomain(2, 1, minimum=0)
    assert not prove(a, c, {a: [(b, even)], b: [(c, odd)]})
    assert prove(a, c, {a: [(b, even), (c, odd)], b: [(c, even)]})


def test_steady_path_does_not_cover_zero_or_large_exact_boundary():
    a, b = site("a"), site("b")
    assert not prove(a, b, {a: [(b, CountDomain(1, 0, minimum=1))]})
    edges = {a: [(b, CountDomain(1, 0, exact=0)), (b, CountDomain(1, 0, minimum=1))]}
    assert prove(a, b, edges)
    for n in (1024, 1048576):
        assert prove(a, b, {a: [(b, CountDomain(1, 0, exact=n))]}, CountDomain(1, 0, exact=n))
        assert not prove(
            a, b, {a: [(b, CountDomain(1, 0, minimum=n + 1))]}, CountDomain(1, 0, exact=n)
        )


def test_uncovered_interval_size_is_independent_of_its_numeric_width():
    required = CountDomain(1, 0, minimum=0)
    for n in (1024, 1048576, 10**12):
        missing = uncovered(required, (CountDomain(1, 0, minimum=n),))
        assert missing == (CountDomain(1, 0, minimum=0, maximum=n - 1),)
        assert not covers(required, missing)
        assert intersection(missing[0], CountDomain(1, 0, minimum=n)) is None


@pytest.mark.parametrize("seed", range(12))
def test_bounded_differences_match_independent_membership(seed):
    rng = random.Random(seed)
    for _ in range(30):
        domains = []
        for _ in range(5):
            p = rng.choice((1, 2, 3, 4))
            minimum = rng.randrange(8)
            domains.append(
                CountDomain(
                    p,
                    rng.randrange(p),
                    minimum=minimum,
                    maximum=rng.choice((None, minimum + 2 * p)),
                )
            )
        required, *guards = domains
        missing = uncovered(required, guards)
        outcomes = []
        for n in range(48):
            expected = required.contains(n) and not any(g.contains(n) for g in guards)
            outcomes.append(expected)
            assert any(d.contains(n) for d in missing) == expected
        assert covers(required, guards) == (not any(outcomes))


def test_continuous_cv_issue_relation_does_not_impose_serial_query_order():
    # Relation-level regression for the actual target policy, not a claim that
    # AST nested CV is executable: new-query QK is issued before old-query PV.
    qk = site("qk", call=1, point="begin")
    pv = site("pv", call=0, point="begin")
    domain = CountDomain(1, 0, minimum=1)
    edges = {qk: [(pv, domain)]}
    assert prove(qk, pv, edges, domain)
    assert not prove(pv, qk, edges, domain)
    assert not prove(replace(qk, point="end"), pv, edges, domain)
    assert not prove(replace(qk, call=0), pv, edges, domain)
    assert not prove(replace(qk, region="other"), pv, edges, domain)


@pytest.mark.parametrize("seed", range(12))
def test_guarded_cycles_and_path_unions_match_finite_graph(seed):
    rng = random.Random(seed)
    nodes = tuple(site(str(i)) for i in range(6))
    edges = defaultdict(list)
    for _ in range(20):
        period = rng.choice((1, 2, 3))
        edges[rng.choice(nodes)].append(
            (
                rng.choice(nodes),
                CountDomain(period, rng.randrange(period), minimum=rng.randrange(5)),
            )
        )
    # After all thresholds, this finite graph repeats modulo the predicate
    # periods. The oracle is explicit BFS per count, not the production solver.
    bound = 5 + lcm(*(g.period for items in edges.values() for _, g in items))
    outcomes = []
    for count in range(bound):
        seen, pending = {nodes[0]}, [nodes[0]]
        while pending:
            for target, guard in edges[pending.pop()]:
                if guard.contains(count) and target not in seen:
                    seen.add(target)
                    pending.append(target)
        expected = nodes[-1] in seen
        outcomes.append(expected)
        assert prove(nodes[0], nodes[-1], edges, CountDomain(1, 0, exact=count)) == expected
    assert prove(nodes[0], nodes[-1], edges) == all(outcomes)


def test_canonical_predicate_intersection_matches_independent_finite_membership():
    from itertools import product

    from xdsl_npu.completion import canonical, intersection
    from xdsl_npu.region_summary import CountDomain

    domains = [
        CountDomain(p, r, minimum=m) for p in (1, 2, 3, 5) for r in range(p) for m in (0, 1, 4)
    ]
    domains += [CountDomain(1, 0, exact=n) for n in (0, 1, 7)]
    for a, b in product(domains, repeat=2):
        result = intersection(canonical(a), canonical(b))
        for n in range(32):
            assert canonical(a).contains(n) == a.contains(n)
            assert bool(result is not None and result.contains(n)) == (
                a.contains(n) and b.contains(n)
            )
