"""Conditional path composition, checked against finite independent reachability."""

import random
from collections import defaultdict
from dataclasses import replace
from math import lcm

import pytest

from xdsl_npu.completion import CountDomain, Position, Site, covered_path


def site(name, *, call=0, point="end"):
    return Site(("aic", name), Position(False, 0), point, region=name, call=call)


def prove(source, target, edges, domain=None):
    domain = CountDomain(1, 0, minimum=0) if domain is None else domain
    return covered_path(source, target, domain, edges, lambda *_: (), lambda *_: True)


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
