"""Frozen hardware subsets preserve actual numbers and private-bank intersections."""

from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_coloring import NoEventAssignment, color_event_resources
from ascendflow.analysis.event_contracts import EventDemand, EventIdPools
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.structured_resources import allocate_structured_events
from ascendflow.passes.generate_notifications import _complete_bindings
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.nested import nested_copy
from tests.models.structured_events import check_compiled_events
from tests.support.structured import selected

BANK0 = (Actor.AIV0, Engine.MTE2, Engine.V)
BANK1 = (Actor.AIV1, Engine.MTE2, Engine.V)


@pytest.mark.parametrize("values", [(8,), (-1,), (True,), (1.0,), (1, 1)])
def test_local_pool_cannot_change_the_hardware_namespace(values):
    with pytest.raises(ValueError):
        EventIdPools(local=values)


@pytest.mark.parametrize("values", [(16,), (-1,), (False,), (2.0,), (3, 3)])
def test_cross_pool_cannot_alias_or_expand_the_hardware_namespace(values):
    with pytest.raises(ValueError):
        EventIdPools(cross=values)


def test_pool_snapshots_mutable_inputs_and_every_private_bank():
    local, cross, override = [1, 6], [3, 15], [5, 7]
    banks = [(BANK1, override)]
    pools = EventIdPools(local=local, cross=cross, banks=banks)
    local.clear()
    cross.append(16)
    override.clear()
    banks.clear()
    assert pools.ids(BANK0) == (1, 6)
    assert pools.ids(BANK1) == (5, 7)
    assert pools.ids(None) == (3, 15)


def numeric(demands, pools, *, peers=(), relations=None, packs=()):
    return SimpleNamespace(
        demands=demands,
        id_pools=pools,
        peers=peers,
        reservations=(),
        relations={} if relations is None else relations,
        counter_reuses={},
        counter_groups=tuple(SimpleNamespace(resources=pack) for pack in packs),
    )


def test_peer_numbers_use_the_intersection_of_actual_private_pools():
    demands = (EventDemand((), 1, (BANK0,)), EventDemand((), 1, (BANK1,)))
    pools = EventIdPools(banks=((BANK0, (1, 5)), (BANK1, (3, 5))))
    assert color_event_resources(numeric(demands, pools, peers=((0, 1),)))[0] == ((5,), (5,))
    pools = EventIdPools(banks=((BANK0, (1,)), (BANK1, (3,))))
    with pytest.raises(NoEventAssignment):
        color_event_resources(numeric(demands, pools, peers=((0, 1),)))


def test_different_private_pools_do_not_bound_global_colors_by_one_bank_size():
    other = (Actor.AIC, Engine.MTE2, Engine.MTE1)
    demands = tuple(
        EventDemand((), 1, banks) for banks in ((BANK0, BANK1), (BANK1, other), (BANK0, other))
    )
    pools = EventIdPools(banks=((BANK0, (1, 5)), (BANK1, (1, 3)), (other, (3, 5))))
    assert color_event_resources(numeric(demands, pools))[0] == ((1,), (3,), (5,))


def test_exclusive_and_certified_word_paths_do_not_enumerate_reuse_pairs():
    class NoPairQueries:
        def get(self, *args):
            raise AssertionError("no residual conflict requires a reuse-pair lookup")

    pools = EventIdPools()
    demands = tuple(EventDemand((), 1, (BANK0,)) for _ in range(8))
    assert color_event_resources(numeric(demands, pools, relations=NoPairQueries()))[1] == 0
    demands = tuple(EventDemand((), 1, (), backlog=1) for _ in range(128))
    packs = tuple(tuple(range(i, min(i + 15, 128))) for i in range(0, 128, 15))
    ids, nodes = color_event_resources(
        numeric(demands, pools, relations=NoPairQueries(), packs=packs)
    )
    assert nodes == 0 and len({group[0] for group in ids}) == len(packs)


def test_real_nested_program_preserves_sparse_local_numbers_across_reentry():
    pools = EventIdPools(local=(0, 2, 3, 4, 5, 6))
    plan = synchronize_structure(nested_copy, id_pools=pools)
    assert all(set(a.ids) <= set(pools.local) for a in plan.resources.assignments)
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_allocator_keeps_its_frozen_pool_when_later_target_defaults_change(monkeypatch):
    from ascendflow.analysis import event_planning

    program = _complete_bindings(selected(nested_copy))
    pools = EventIdPools(local=(1, 2, 3, 4, 5, 7))
    problem = plan_event_resources(program, id_pools=pools)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation cannot obtain new target or source premises")

    monkeypatch.setattr(event_planning, "EventIdPools", forbidden)
    monkeypatch.setattr(type(program.domains), "_following_conditions", forbidden)
    plan = allocate_structured_events(problem)
    assert plan.id_pools is pools
    assert all(set(a.ids) <= set(pools.local) for a in plan.assignments)
