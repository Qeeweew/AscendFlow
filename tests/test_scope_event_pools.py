"""Actual parent event pools preserve consumption through activation tails."""

from dataclasses import replace

import pytest

from tests.fixtures.nested import nested_copy
from tests.test_scope_events import planned_children
from tests.test_scope_protocol_review import fifo_proof
from xdsl_npu.ir import dump
from xdsl_npu.model import CompileError
from xdsl_npu.scope_access import analyze_scope_access
from xdsl_npu.scope_event_pools import allocate_parent_pools, verify_parent_pools
from xdsl_npu.scope_events import event_choices, resource_inputs
from xdsl_npu.scope_sync import ParentCompletion, ScopeProtocol

BANK = "aiv0", "MTE3", "MTE2"


@pytest.fixture(scope="module")
def copied():
    program, analyses = planned_children(nested_copy.to_ir())
    access = analyze_scope_access(program)
    choices = event_choices(program, analyses)
    first, second = program.leaves
    tasks = {
        (uid, task.uid): task
        for uid, analysis in analyses.items()
        for data in analysis[0].actors.values()
        for task in data.program.tasks
    }

    def owner(uid, pipe, position):
        return next(
            key
            for key, (call, site) in access.endpoints.items()
            if call == uid
            and site.position.offset == position
            and tasks[uid, site.task[1]].pipe == pipe
        )

    protocols = tuple(
        ScopeProtocol(
            "p" + str(index),
            "local",
            (owner(source, "MTE3", source_j),),
            (owner(destination, "MTE2", destination_j),),
            distance,
            1,
            (0,),
        )
        for index, (source, source_j, destination, destination_j, distance) in enumerate(
            (
                (first, 2, second, 0, 0),
                (first, 3, second, 1, 0),
                (second, 2, first, 0, 2),
                (second, 3, first, 1, 2),
            )
        )
    )

    def proof(period=1, selected=protocols):
        graph = ParentCompletion(
            program, analyses, access, period, resource_inputs=resource_inputs(choices)
        )
        for protocol in selected:
            graph.insert(protocol)
        graph.graph.potential()
        return graph

    return program, protocols, proof


@pytest.mark.parametrize("period", [1, 3])
def test_actual_copy_pools_expand_reverse_width_and_fit_eight_ids(copied, period):
    program, protocols, build = copied
    proof = build(period)
    before = dump(program.module), tuple(proof.graph.nodes), tuple(proof.graph.edges)
    reserved = {BANK: {0, 1, 2, 3}}
    statistics = {}
    allocated, used = allocate_parent_pools(protocols, proof, reserved, statistics=statistics)
    assert allocated[0].ids == allocated[2].ids == (4, 5)
    assert allocated[1].ids == allocated[3].ids == (6, 7)
    assert used[BANK] == set(range(8))
    assert reserved == {BANK: {0, 1, 2, 3}}
    assert statistics["allocated_ids"] == 4 and statistics["shared_pools"] == 2
    assert statistics["sharing_queries"] > 0
    checked = {}
    assert verify_parent_pools(allocated, proof, reserved, statistics=checked) == used
    assert checked["self_queries"] == 4 * period
    assert checked["self_batches"] == 1
    assert checked["sharing_queries"] == 8 * period
    assert (dump(program.module), tuple(proof.graph.nodes), tuple(proof.graph.edges)) == before


def test_banks_that_fit_skip_every_pair_query(copied):
    _, protocols, build = copied
    proof, statistics = build(), {}
    allocated, used = allocate_parent_pools(protocols, proof, {}, statistics=statistics)
    assert tuple(len(p.ids) for p in allocated) == (2, 2, 1, 1)
    assert len(used[BANK]) == 6
    assert statistics["sharing_queries"] == statistics["sharing_batches"] == 0
    assert statistics["merges"] == 0
    verified = {}
    assert verify_parent_pools(allocated, proof, {}, statistics=verified) == used
    assert verified["sharing_queries"] == verified["sharing_batches"] == 0


def test_checker_rejects_unproved_sharing_even_when_each_member_reuses_itself(copied):
    _, protocols, build = copied
    shared = tuple(replace(p, ids=(2, 6)) for p in protocols[:2])
    with pytest.raises(CompileError, match="no proved parent event pool sharing order"):
        verify_parent_pools(shared, build(), {})


def test_every_pair_in_a_shared_group_needs_a_proof(copied):
    _, protocols, build = copied
    # p0 can share with either reverse channel, but p2/p3 cannot share.
    shared = tuple(replace(protocols[index], ids=(0, 1)) for index in (0, 2, 3))
    with pytest.raises(CompileError, match="p2/p3: no proved.*sharing order"):
        verify_parent_pools(shared, build(), {})


@pytest.mark.parametrize("invalid", [(5, 6), (5, 4)])
def test_partial_overlap_and_different_tuple_order_are_rejected(copied, invalid):
    _, protocols, build = copied
    allocated, _ = allocate_parent_pools(protocols, build(), {BANK: {0, 1, 2, 3}})
    changed = (*allocated[:2], replace(allocated[2], ids=invalid), allocated[3])
    with pytest.raises(CompileError, match="partial overlap or different ID tuple order"):
        verify_parent_pools(changed, build(), {BANK: {0, 1, 2, 3}})


def test_non_arithmetic_but_identical_pool_tuples_are_valid(copied):
    _, protocols, build = copied
    shared = tuple(replace(protocols[index], ids=(7, 2)) for index in (0, 2))
    assert verify_parent_pools(shared, build(), {})[BANK] == {2, 7}


def test_matching_sparse_ordinal_streams_can_share_across_distance_and_wrap(copied):
    _, protocols, build = copied
    sparse = tuple(replace(protocol, period=3, residues=(0, 2)) for protocol in protocols)
    proof = build(3, selected=sparse)
    shared = tuple(replace(sparse[index], ids=(6, 4)) for index in (0, 2))
    assert verify_parent_pools(shared, proof, {})[BANK] == {4, 6}


def test_child_and_configuration_resources_cannot_be_used_by_parent_pool(copied):
    _, protocols, build = copied
    assigned = (replace(protocols[0], ids=(0, 1)),)
    with pytest.raises(CompileError, match="overlaps child/configuration resource"):
        verify_parent_pools(assigned, build(), {BANK: {0}})


def test_infeasible_current_pool_packing_is_rejected_without_mutating_input(copied):
    _, protocols, build = copied
    reserved = {BANK: set(range(7))}
    with pytest.raises(CompileError, match="packing needs 4 IDs, only 1 free"):
        allocate_parent_pools(protocols, build(), reserved)
    assert reserved == {BANK: set(range(7))}
    assert all(not p.ids for p in protocols)


def test_query_budget_counts_work_across_batches(copied, monkeypatch):
    _, protocols, build = copied
    statistics = {}
    monkeypatch.setattr("xdsl_npu.scope_event_pools.MAX_POOL_QUERIES", 5)
    with pytest.raises(CompileError, match="query budget exceeded"):
        allocate_parent_pools(protocols, build(), {}, statistics=statistics)
    # Width 1 needs four queries. The second batch's two queries exceed the
    # total limit even though each batch individually fits within five.
    assert statistics["self_queries"] == 4
    assert statistics["self_batches"] == 1


def test_merge_work_budget_includes_checks_inside_candidates(copied, monkeypatch):
    _, protocols, build = copied
    statistics = {}
    monkeypatch.setattr("xdsl_npu.scope_event_pools.MAX_POOL_MERGE_WORK", 1)
    with pytest.raises(CompileError, match="merge work budget exceeded"):
        allocate_parent_pools(protocols, build(), {BANK: {0, 1, 2, 3}}, statistics=statistics)
    assert statistics["merge_candidates"] == 1
    assert statistics["merge_checks"] == 2


def test_tail_absence_does_not_make_pair_proofs_a_replacement_for_self_reuse():
    proof = fifo_proof(2, 1)
    source, destination, _ = proof.access.endpoints
    active = ScopeProtocol("active", "local", (source,), (destination,), 0, 1, (0,), (0,))
    absent = ScopeProtocol("absent", "local", (source,), (destination,), 3, 1, (0,), (0,))
    for protocol in (active, absent):
        proof.insert(protocol)
    a_wait, b_wait = (proof.events[source[0], p.name, "wait"] for p in (active, absent))
    a_set, b_set = (proof.events[source[0], p.name, "set"] for p in (active, absent))
    # Both pair queries are vacuous because the distance-3 source is absent
    # from N=2. The active channel still has two notifications on ID 0.
    assert proof.graph.covered_many([(a_wait, 0, b_set, 0), (b_wait, 3, a_set, 1)]) == [True, True]
    with pytest.raises(CompileError, match="active:.*self reuse"):
        verify_parent_pools((active, absent), proof, {})


def test_different_sparse_source_activations_cannot_share_a_pool():
    proof = fifo_proof(4, 3)
    source, destination, _ = proof.access.endpoints
    protocols = tuple(
        ScopeProtocol(
            "sparse" + str(residue), "local", (source,), (destination,), 0, 3, (residue,), (0, 1)
        )
        for residue in (0, 2)
    )
    for protocol in protocols:
        proof.insert(protocol)
    with pytest.raises(CompileError, match="different period/residues"):
        verify_parent_pools(protocols, proof, {})


@pytest.mark.parametrize("attribute", ["last", "carry"])
def test_terminal_and_carried_channels_do_not_enter_static_parent_pools(copied, attribute):
    _, protocols, build = copied
    with pytest.raises(CompileError, match="nonterminal, non-carried"):
        allocate_parent_pools((replace(protocols[0], **{attribute: True}),), build(), {})
