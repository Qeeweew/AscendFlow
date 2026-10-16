"""Real fixed-child event gaps, checked again from actual assigned IR."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.fixtures.event_borrow import vector_event_borrow
from tests.models.scoped_local import finite_check
from tests.test_scope_guarded_completion import command, edge, finite_covered, mark, singleton
from tests.test_scope_query_completion import FiniteQueries
from tests.test_scope_query_resources import actual_word, endpoint_coordinate
from xdsl_npu import compile, scope_event_pools
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.ir import ScopeChannelOp, ScopeSyncOp, dump
from xdsl_npu.loop_graph import Graph
from xdsl_npu.model import CompileError
from xdsl_npu.scope_anchored_completion import AnchoredCompletion
from xdsl_npu.scope_domains import shifted
from xdsl_npu.scope_event_borrow import (
    BorrowOption,
    ChildSlot,
    ParentEventBankExhausted,
    check_borrow_options,
    child_slots,
    prepare_borrow_options,
    verify_actual_borrows,
)
from xdsl_npu.scope_guarded_completion import GuardedCompletion
from xdsl_npu.scope_guarded_events import _bank
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_predicates import case_activation
from xdsl_npu.scope_query_completion import QueryCompletion
from xdsl_npu.scope_query_resources import (
    check_binary_resources,
    prepare_binary_resources,
)
from xdsl_npu.scope_sync import _leaf_banks, _verify_scope_parent


@pytest.fixture(scope="module")
def actual():
    observed = []
    original = scope_event_pools.allocate_parent_pools

    def allocate(protocols, proof, used, **kwargs):
        before = dump(proof.program.module), tuple(proof.graph.nodes), tuple(proof.graph.edges)
        result = original(protocols, proof, used, **kwargs)
        assert (
            dump(proof.program.module),
            tuple(proof.graph.nodes),
            tuple(proof.graph.edges),
        ) == before
        observed.append(True)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scope_event_pools, "allocate_parent_pools", allocate)
        artifact = compile(vector_event_borrow, timeout_s=10)
    _, checked, proof = _verify_scope_parent(extract_scopes(artifact.plan))
    reserved = _leaf_banks(checked.program, checked.analyses)
    borrowed = tuple(
        p
        for p in checked.channels.values()
        if p.kind == "local" and set(p.ids).intersection(reserved[_bank(p, proof)[0]])
    )
    assert observed and borrowed
    return artifact.plan, checked, proof, reserved, borrowed


def declaration(module, name):
    return next(
        op for op in module.walk() if isinstance(op, ScopeChannelOp) and op.sym_name.data == name
    )


def test_actual_source_uses_one_parent_per_existing_child_id(actual):
    _, checked, proof, reserved, borrowed = actual
    slots = child_slots(checked.program, checked.analyses)
    assert len(borrowed) == 6
    assert all(p.distance == len(p.ids) == 1 for p in borrowed)
    identities = [(_bank(p, proof)[0], p.ids[0]) for p in borrowed]
    assert len(set(identities)) == len(identities)
    assert all(key in slots for key in identities)
    assert any(slots[key].conditional for key in identities)
    assert verify_actual_borrows(checked.channels.values(), proof, reserved) == frozenset(
        p.name for p in borrowed
    )


@pytest.mark.parametrize("count", (0, 1, 2, 3, 5))
def test_actual_source_passes_independent_finite_instructions(actual, count):
    result = finite_check(actual[0], count)
    assert result.command_count >= 0


def test_private_fit_preserves_original_ids_and_skips_borrowing(actual, monkeypatch):
    _, _, proof, reserved, borrowed = actual

    def forbidden(*args, **kwargs):
        pytest.fail("a bank that fits must not ask any borrowing query")

    monkeypatch.setattr(scope_event_pools, "allocate_borrows", forbidden)
    statistics = {}
    protocols = tuple(replace(p, ids=()) for p in borrowed[:2])
    assigned, used = scope_event_pools.allocate_parent_pools(
        protocols, proof, reserved, statistics=statistics
    )
    assert tuple(p.ids for p in assigned) == ((6,), (7,))
    assert not any(key.startswith("borrow") for key in statistics)
    assert scope_event_pools.verify_parent_pools(assigned, proof, reserved) == used


def test_only_typed_bank_exhaustion_enters_borrowing(actual, monkeypatch):
    _, checked, proof, reserved, _ = actual
    protocols = tuple(replace(p, ids=()) for p in checked.channels.values())
    with pytest.raises(ParentEventBankExhausted) as caught:
        scope_event_pools._allocate_private_pools(protocols, proof, reserved)
    assert caught.value.bank == ("aiv0", "V", "MTE2")
    assert caught.value.needed > caught.value.available

    def untyped(*args, **kwargs):
        raise CompileError("conditional parent event bank exhausted")

    def forbidden(*args, **kwargs):
        pytest.fail("an error message cannot authorize borrowing")

    monkeypatch.setattr(scope_event_pools, "_allocate_private_pools", untyped)
    monkeypatch.setattr(scope_event_pools, "allocate_borrows", forbidden)
    with pytest.raises(CompileError, match="conditional parent event bank exhausted"):
        scope_event_pools.allocate_parent_pools(protocols, proof, reserved)


def test_actual_wrong_borrower_is_rejected_without_duplicate_parent_ids(actual):
    module, checked, proof, _, borrowed = actual
    slots = child_slots(checked.program, checked.analyses)
    options = prepare_borrow_options(borrowed, proof, slots)
    valid = {
        (option.channel, option.slot.event_id): result
        for option, result in zip(options, check_borrow_options(options, proof), strict=True)
    }
    first, second = next(
        (a, b) for a in borrowed for b in borrowed if a is not b and not valid[a.name, b.ids[0]]
    )
    changed = module.clone()
    a, b = declaration(changed, first.name), declaration(changed, second.name)
    a.attributes["event_ids"], b.attributes["event_ids"] = b.event_ids, a.event_ids
    with pytest.raises(CompileError, match="borrower can overwrite"):
        _verify_scope_parent(extract_scopes(changed))
    with pytest.raises(AssertionError, match="binary event"):
        finite_check(changed, 3)


def test_actual_second_parent_on_one_child_id_is_rejected(actual):
    module, _, _, _, borrowed = actual
    changed = module.clone()
    declaration(changed, borrowed[1].name).attributes["event_ids"] = declaration(
        changed, borrowed[0].name
    ).event_ids
    with pytest.raises(CompileError, match="multiple parent borrowers"):
        _verify_scope_parent(extract_scopes(changed))
    with pytest.raises(AssertionError, match="binary event overwrite"):
        finite_check(changed, 3)


@pytest.mark.parametrize("child", (False, True), ids=("parent", "child"))
def test_deleting_actual_wait_is_rejected(actual, child):
    from xdsl_npu.ir import LoopSyncOp

    changed = actual[0].clone()
    name = actual[4][0].name
    wait = next(
        op
        for op in changed.walk()
        if isinstance(op, LoopSyncOp if child else ScopeSyncOp)
        and op.action.data == "wait"
        and (child or op.channel.data == name)
    )
    wait.parent.erase_op(wait)
    reason = "unpaired symbolic endpoint" if child else "missing or changes.*Set/Wait placement"
    with pytest.raises(CompileError, match=reason):
        _verify_scope_parent(extract_scopes(changed))


def test_reserved_id_without_an_actual_child_owner_is_not_borrowable(actual):
    _, _, proof, reserved, borrowed = actual
    bank = _bank(borrowed[0], proof)[0]
    reserved = dict(reserved) | {bank: reserved[bank] | {7}}
    with pytest.raises(CompileError, match="configuration or has no actual child owner"):
        verify_actual_borrows((replace(borrowed[0], ids=(7,)),), proof, reserved)


def test_all_options_use_one_guarded_and_one_anchored_batch(actual, monkeypatch):
    _, checked, proof, _, borrowed = actual
    slots = child_slots(checked.program, checked.analyses)
    options = prepare_borrow_options(borrowed, proof, slots)
    calls = {"guarded": [], "anchored": []}
    for name, owner in (("guarded", proof.guarded()), ("anchored", proof.anchored())):
        original = owner.covered_many

        def record(queries, *, name=name, original=original):
            queries = tuple(queries)
            calls[name].append(queries)
            return original(queries)

        monkeypatch.setattr(owner, "covered_many", record)
    before = tuple(proof.graph.nodes), tuple(proof.graph.edges)
    statistics = {}
    assert any(check_borrow_options(options, proof, statistics=statistics))
    assert len(calls["guarded"]) == len(calls["anchored"]) == 1
    assert statistics["borrow_options"] == len(options)
    assert statistics["borrow_anchored_queries"] > 0
    assert before == (tuple(proof.graph.nodes), tuple(proof.graph.edges))


@pytest.mark.parametrize("budget", ("MAX_BORROW_OPTIONS", "MAX_BORROW_QUERIES"))
def test_borrowing_budgets_reject_explicitly(actual, monkeypatch, budget):
    _, checked, proof, _, borrowed = actual
    monkeypatch.setattr(f"xdsl_npu.scope_event_borrow.{budget}", 0)
    with pytest.raises(CompileError, match="borrowing (option|query) budget"):
        prepare_borrow_options(borrowed, proof, child_slots(checked.program, checked.analyses))


@pytest.mark.parametrize("count", (4, 5, 7))
@pytest.mark.parametrize("missing", ("left", "right", None))
def test_each_conditional_middle_hole_anchor_is_independently_necessary(count, missing):
    graph, mapping = Graph(1, count), {}
    old_wait, new_set, parent_set, parent_wait = (command(graph, i) for i in range(4))
    domain = CountDomain(1, 0, exact=count)
    activation = singleton(1, count=domain)
    mark(mapping, old_wait, case_activation("first"))
    mark(mapping, new_set, case_activation("last"))
    mark(mapping, parent_set, activation)
    mark(mapping, parent_wait, shifted(activation, 1))
    edge(graph, old_wait, new_set, count - 1)
    edge(graph, parent_set, parent_wait, 1)
    if missing != "left":
        edge(graph, old_wait, parent_set, 1)
    if missing != "right":
        edge(graph, parent_wait, new_set, count - 3)
    guarded = GuardedCompletion(graph, mapping)
    anchored = AnchoredCompletion(guarded)
    adjacent = (
        (old_wait[0], parent_set[0], 0, activation),
        (parent_wait[0], new_set[0], 0, shifted(activation, 1)),
    )
    anchors = (
        (old_wait[0], parent_set[0], Position(False, 0), None, activation),
        (parent_wait[0], new_set[0], None, Position(True, 0), shifted(activation, 1)),
    )
    # Both adjacent endpoints are absent. Child self-reuse nevertheless passes.
    assert guarded.covered_many(adjacent) == [True, True]
    assert finite_covered(
        graph, mapping, old_wait, new_set, count - 1, singleton(count=domain), count
    )
    # Literal finite BFS independently sees the two missing consumption edges.
    expected = (
        finite_covered(graph, mapping, old_wait, parent_set, 1, singleton(count=domain), count),
        finite_covered(
            graph, mapping, parent_wait, new_set, count - 3, singleton(2, count=domain), count
        ),
    )
    assert expected == (missing != "left", missing != "right")
    assert tuple(anchored.covered_many(anchors)) == expected
    option = BorrowOption(
        "hole", ChildSlot(("aic", "V", "MTE2"), 0, ("child", "c"), (), True), adjacent, anchors
    )
    proof = SimpleNamespace(guarded=lambda: guarded, anchored=lambda: anchored)
    assert check_borrow_options((option,), proof) == (missing is None,)


def test_actual_mixed_resources_keep_every_old_new_pair_including_self(actual):
    _, checked, proof, _, _ = actual
    prepared = prepare_binary_resources(checked, proof)
    assert any(check.kind == "mixed" for check in prepared.checks)
    query = QueryCompletion(proof)
    coverage = query.coverage_many((c.source, c.target, c.domain) for c in prepared.checks)
    for count in (1, 2, 3, 5):
        tokens = actual_word(checked, count)
        active = [c for c in prepared.checks if c.domain.contains(count)]
        for (bank, event_id, old_owner, _), old in tokens.items():
            for (other_bank, other_id, new_owner, _), new in tokens.items():
                if (bank, event_id) != (other_bank, other_id):
                    continue
                assert any(
                    c.bank == bank
                    and c.event_id == event_id
                    and c.source_owner == old_owner
                    and c.target_owner == new_owner
                    and endpoint_coordinate(c.source, count) == max(old["wait"])
                    and endpoint_coordinate(c.target, count) == min(new["set"])
                    for c in active
                )
        finite = FiniteQueries(proof, count)
        for check, result in zip(prepared.checks, coverage, strict=True):
            expected = check.domain.contains(count) and finite.covered(check.source, check.target)
            assert any(d.contains(count) for d in result) == expected
    assert all(check_binary_resources(prepared, query))
