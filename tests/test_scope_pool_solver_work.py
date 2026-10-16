"""Resource lower bounds and query ownership across borrowing retries."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from xdsl_npu import scope_event_pools
from xdsl_npu.loop_graph import Command
from xdsl_npu.scope_domains import legacy_activation
from xdsl_npu.scope_event_borrow import ParentEventBankExhausted
from xdsl_npu.scope_guarded_events import allocate as allocate_guarded
from xdsl_npu.scope_sync import ScopeProtocol

PRIVATE = "aiv0", "MTE3", "MTE2"
FIRST_BORROW = "aiv0", "V", "MTE2"
SECOND_BORROW = "aiv0", "MTE2", "V"
CONDITIONAL = "aiv0", "M", "FIX"


@dataclass(frozen=True)
class _Task:
    pipe: str


def counted_proof(specifications):
    """Synthetic coverage spy for solver work; real event models live elsewhere."""
    events, protocols, ordinary, guarded = {}, [], [], []
    for index, (name, bank, conditional) in enumerate(specifications):
        distance = 1 if conditional else 0
        protocol = ScopeProtocol(
            name,
            "local",
            ((bank[0], name + "_source"),),
            ((bank[0], name + "_target"),),
            distance,
            1,
            (0,),
            activation=legacy_activation(distance) if conditional else None,
        )
        protocols.append(protocol)
        for side, action in enumerate(("set", "wait")):
            events[bank[0], name, action] = Command(
                2 * index + side,
                bank[0],
                _Task(bank[side + 1]),
                action,
                name,
                lower=distance if action == "wait" else 0,
                upper=distance if action == "set" else 0,
            )

    def cover(batches):
        def covered_many(queries):
            queries = tuple(queries)
            batches.append(queries)
            return [True] * len(queries)

        return covered_many

    proof = SimpleNamespace(
        graph=SimpleNamespace(period=1),
        events=events,
        covered_many=cover(ordinary),
        guarded=lambda: SimpleNamespace(covered_many=cover(guarded)),
    )
    return tuple(protocols), proof, ordinary, guarded


@pytest.mark.parametrize("conditional", (False, True), ids=("ordinary", "conditional"))
def test_zero_free_bank_uses_the_one_id_lower_bound_without_any_query(conditional):
    protocols, proof, ordinary, guarded = counted_proof((("full", PRIVATE, conditional),))
    statistics = {}
    allocate = allocate_guarded if conditional else scope_event_pools._allocate_private_pools
    with pytest.raises(ParentEventBankExhausted) as caught:
        allocate(protocols, proof, {PRIVATE: set(range(8))}, statistics=statistics)
    assert (caught.value.bank, caught.value.needed, caught.value.available) == (PRIVATE, 1, 0)
    assert ordinary == guarded == []
    assert statistics["zero_free_bank_rejections"] == 1
    assert statistics.get("self_queries", 0) == 0
    assert statistics.get("sharing_queries", 0) == 0
    assert statistics.get("conditional_self_queries", 0) == 0


def test_one_ledger_retains_self_pair_and_conditional_results_across_two_borrows(monkeypatch):
    protocols, proof, ordinary, guarded = counted_proof(
        (
            ("guarded", CONDITIONAL, True),
            ("private_a", PRIVATE, False),
            ("private_b", PRIVATE, False),
            ("borrow_a", FIRST_BORROW, False),
            ("borrow_b", SECOND_BORROW, False),
        )
    )
    reserved = {
        PRIVATE: set(range(7)),
        FIRST_BORROW: set(range(8)),
        SECOND_BORROW: set(range(8)),
    }
    borrowed = []

    def borrow(pending, owner, bank, *, statistics):
        assert owner is proof
        borrowed.append(bank)
        name = "borrow_a" if bank == FIRST_BORROW else "borrow_b"
        assert name in {p.name for p in pending}
        return {name: 0}

    monkeypatch.setattr(scope_event_pools, "allocate_borrows", borrow)
    # Two private channels, eight self widths, both pair orientations, and one
    # guarded width produce 16 + 32 + 1 actual queries across ALL attempts.
    monkeypatch.setattr(scope_event_pools, "MAX_POOL_QUERIES", 49)
    statistics = {}
    assigned, used = scope_event_pools.allocate_parent_pools(
        protocols, proof, reserved, statistics=statistics
    )
    assert borrowed == [FIRST_BORROW, SECOND_BORROW]
    assert tuple(p.ids for p in assigned) == ((0,), (7,), (7,), (0,), (0,))
    assert used[PRIVATE] == set(range(8)) and used[CONDITIONAL] == {0}
    assert reserved[PRIVATE] == set(range(7))
    assert statistics["self_queries"] == 16 and statistics["self_batches"] == 2
    assert statistics["sharing_queries"] == 32 and statistics["sharing_batches"] == 1
    assert statistics["conditional_self_queries"] == statistics["conditional_self_batches"] == 1
    assert statistics["zero_free_bank_rejections"] == 2
    assert len(ordinary) == 3 and len(guarded) == 1
    assert sum(map(len, ordinary)) + sum(map(len, guarded)) == 49
    assert all(
        source.channel.startswith("private_") and target.channel.startswith("private_")
        for batch in ordinary
        for source, _, target, _ in batch
    )


def test_lower_bound_does_not_claim_borrowing_is_available(monkeypatch):
    protocols, proof, ordinary, guarded = counted_proof((("full", PRIVATE, False),))
    banks = []

    def unavailable(pending, owner, bank, **kwargs):
        banks.append(bank)
        return {}

    monkeypatch.setattr(scope_event_pools, "allocate_borrows", unavailable)
    with pytest.raises(ParentEventBankExhausted):
        scope_event_pools.allocate_parent_pools(protocols, proof, {PRIVATE: set(range(8))})
    assert banks == [PRIVATE] and ordinary == guarded == []
