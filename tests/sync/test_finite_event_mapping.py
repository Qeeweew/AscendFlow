"""Finite mappings agree with independent small enumeration of legal IDs."""

from itertools import combinations, product
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import NoEventAssignment, color_event_resources
from ascendflow.analysis.event_contracts import EventDemand, EventIdPools, ResourceWork
from ascendflow.analysis.event_frontier_contracts import FrozenEventRelations
from ascendflow.target.ascend910b import Actor, Engine

BANK = (Actor.AIV0, Engine.MTE2, Engine.V)


def problem(edges, size=5):
    return SimpleNamespace(
        demands=tuple(EventDemand((), 1, (BANK,)) for _ in range(size)),
        reservations=(),
        peers=(),
        counter_groups=(),
        counter_reuses=FrozenEventRelations(()),
        relations={pair: () for pair in combinations(range(size), 2) if pair not in edges},
        id_pools=EventIdPools(local=(1, 6)),
        work=ResourceWork(),
    )


def test_odd_cycle_has_no_mapping_in_the_actual_two_number_pool():
    numeric = problem({(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)})
    with pytest.raises(NoEventAssignment) as error:
        color_event_resources(numeric)
    assert error.value.witness


def test_all_five_vertex_graphs_match_an_independent_two_color_oracle():
    pairs = tuple(combinations(range(5), 2))
    for mask in range(1 << len(pairs)):
        edges = {pair for i, pair in enumerate(pairs) if mask & (1 << i)}
        expected = any(
            all(values[a] != values[b] for a, b in edges) for values in product((1, 6), repeat=5)
        )
        try:
            ids, option = color_event_resources(problem(edges))
        except NoEventAssignment:
            assert not expected
        else:
            assert expected and option == 0
            assert all(ids[a] != ids[b] for a, b in edges)
            assert all(values[0] in (1, 6) for values in ids)


def test_different_joint_bank_pools_keep_distinct_unused_id_choices():
    other = (Actor.AIV1, Engine.MTE2, Engine.V)
    numeric = problem({(0, 1)}, size=2)
    numeric.demands = (EventDemand((), 1, (BANK,)), EventDemand((), 1, (BANK, other)))
    numeric.id_pools = EventIdPools(banks=((BANK, (0, 1)), (other, (0,))))
    ids, _ = color_event_resources(numeric)
    assert ids == ((1,), (0,))


def test_protocol_and_ids_are_selected_by_one_joint_problem():
    preferred = problem({(0, 1), (1, 2), (0, 2)}, size=3)
    alternative = problem({(0, 1)}, size=3)
    ids, option = color_event_resources(preferred, alternatives=(alternative,))
    assert option == 1 and ids[0] != ids[1]
    assert preferred.work.allocation_queries == 1
    assert preferred.work.allocation_constraints > 0
    # A feasible preferred protocol wins even if another protocol also fits.
    ids, option = color_event_resources(alternative, alternatives=(preferred,))
    assert option == 0 and ids[0] != ids[1]
    assert alternative.work.allocation_queries == 1


@pytest.mark.parametrize("backlogs", [(1, 1, 1, 1), (7, 8, 7, 8), (9, 9, 1, 1)])
@pytest.mark.parametrize(
    "packs,compositions",
    [
        ((), ()),
        (((0, 1, 2, 3),), ()),
        (((0, 1), (2, 3)), (((0, 1), (2, 3)),)),
    ],
)
def test_counter_sets_match_independent_whole_group_enumeration(backlogs, packs, compositions):
    from ascendflow.analysis.event_frontier_contracts import CreditSeparation

    pairs = tuple(combinations(range(4), 2))
    for mask in range(1 << len(pairs)):
        separated = {pair for bit, pair in enumerate(pairs) if mask & (1 << bit)}
        rows = []
        for n in range(4):
            allowed = sum(1 << m for m in range(4) if (min(n, m), max(n, m)) in separated)
            rows.append(((allowed, CreditSeparation((), 1 << n, allowed)),) if allowed else ())
        numeric = problem(set(), size=4)
        numeric.demands = tuple(EventDemand((), 1, (), backlog=b) for b in backlogs)
        numeric.counter_reuses = FrozenEventRelations(rows)
        numeric.counter_groups = tuple(SimpleNamespace(resources=pack) for pack in packs)
        numeric.counter_compositions = tuple(
            SimpleNamespace(groups=groups) for groups in compositions
        )
        numeric.id_pools = EventIdPools(cross=(3, 9))

        def legal(values, separated=separated):
            for flag in (3, 9):
                members = {n for n, value in enumerate(values) if value == flag}
                if all(pair in separated for pair in combinations(sorted(members), 2)):
                    continue
                if (
                    any(members <= set(pack) for pack in packs)
                    and sum(backlogs[n] for n in members) <= 15
                ):
                    continue
                if any(
                    members <= set().union(*map(set, groups))
                    and all(
                        sum(backlogs[n] for n in members.intersection(child)) <= 15
                        for child in groups
                    )
                    for groups in compositions
                ):
                    continue
                return False
            return True

        expected = any(legal(values) for values in product((3, 9), repeat=4))
        try:
            ids, _ = color_event_resources(numeric)
        except NoEventAssignment:
            assert not expected, (backlogs, packs, separated)
        else:
            assert expected and legal(tuple(values[0] for values in ids))


def test_unknown_finite_choice_never_authorizes_an_allocation(monkeypatch):
    from z3 import unknown

    from ascendflow.analysis import smt
    from ascendflow.diagnostics import CompileError, ErrorKind

    class Undecided:
        def assert_and_track(self, *args):
            pass

        def set(self, **kwargs):
            assert kwargs["timeout"] <= 2000

        def check(self):
            return unknown

        def reason_unknown(self):
            return "test interruption"

    monkeypatch.setattr(smt, "Optimize", Undecided)
    with pytest.raises(CompileError, match="unknown.*no allocation permitted") as error:
        smt.finite_model((("pool", True),))
    assert error.value.kind is ErrorKind.ANALYSIS_LIMIT
