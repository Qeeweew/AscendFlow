"""Finite prefix semantics and structural costs, without a completion graph."""

from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest

from xdsl_npu.analysis.frontier import (
    Clock,
    Frontier,
    FrontierError,
    FrontierLayout,
    FrontierWork,
    Handoff,
    KnownPrefix,
    Partition,
    PrefixDomain,
    PrefixRank,
)
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.target.ascend910b import Actor, Engine, Milestone

DEFAULT_PARTITION = Partition()


def domain(
    *,
    actor=Actor.AIV0,
    engine=Engine.V,
    milestone=Milestone.RETIREMENT_CAPTURE,
    clock=None,
    partition=DEFAULT_PARTITION,
):
    return PrefixDomain(
        actor, engine, milestone, Clock("step") if clock is None else clock, partition
    )


def prefix(key, offset=0, lexical=0):
    return KnownPrefix(key, PrefixRank(key.clock, offset, lexical))


def literal_prefixes(state, generation):
    """An explicit finite prefix word; contains no production order/join call."""
    return {
        (key, epoch, lexical)
        for key, rank in state.known.items()
        for epoch in range(generation - 2, generation + 3)
        for lexical in range(3)
        if (epoch, lexical) <= (generation + rank.offset, rank.lexical)
    }


def test_same_clock_affine_order_holds_at_every_generation():
    clock = Clock("physical_step")
    for (offset, lexical), (other_offset, other_lexical) in product(
        product(range(-2, 3), range(3)), repeat=2
    ):
        first, last = (
            PrefixRank(clock, offset, lexical),
            PrefixRank(clock, other_offset, other_lexical),
        )
        for generation in (0, 1, 8, 10**12):
            a, b = (generation + offset, lexical), (generation + other_offset, other_lexical)
            assert first.compare(last) == (a > b) - (a < b)


def test_acquire_and_branch_merge_match_literal_union_and_intersection():
    keys = (domain(), domain(engine=Engine.MTE2))
    layout = FrontierLayout(keys)
    states = tuple(
        Frontier(
            layout,
            (prefix(key, *rank) for key, rank in zip(keys, ranks, strict=True) if rank is not None),
        )
        for ranks in product((None, (-1, 1), (0, 0), (1, 2)), repeat=2)
    )
    for first, last in product(states, repeat=2):
        acquired = first.acquire(last)
        merged = Frontier.must_merge((first, last))
        for generation in (0, 5, 10**9):
            left, right = literal_prefixes(first, generation), literal_prefixes(last, generation)
            assert literal_prefixes(acquired, generation) == left | right
            assert literal_prefixes(merged, generation) == left & right
        assert first.acquire(last) == last.acquire(first)
        assert first.acquire(first) is first
        assert Frontier.must_merge((first, first)) is first


def test_join_and_must_merge_associate_without_path_records():
    key = domain()
    layout = FrontierLayout((key,))
    states = tuple(Frontier(layout, (prefix(key, value),)) for value in (-1, 0, 1))
    for first, middle, last in product(states, repeat=3):
        assert first.acquire(middle).acquire(last) == first.acquire(middle.acquire(last))
        assert Frontier.must_merge((Frontier.must_merge((first, middle)), last)) == (
            Frontier.must_merge((first, Frontier.must_merge((middle, last))))
        )
        assert first.acquire(Frontier.must_merge((middle, last))) == Frontier.must_merge(
            (
                first.acquire(middle),
                first.acquire(last),
            )
        )


def test_skipped_branch_inherits_input_and_cannot_contribute_new_facts():
    retired, queued = domain(), domain(milestone=Milestone.ADMISSION)
    layout = FrontierLayout((retired, queued))
    incoming = Frontier(layout, (prefix(retired, -1),))
    active = incoming.record(prefix(retired), prefix(queued))
    skipped = incoming.transparent()
    assert skipped is incoming
    result = Frontier.must_merge((active, skipped))
    assert result == incoming
    assert not result.covers(prefix(retired))
    assert not result.covers(prefix(queued))
    assert incoming.covers(prefix(retired, -1))


def test_guards_owners_and_slots_are_exact_declared_partitions():
    keys = tuple(
        domain(partition=Partition(owner=owner, slot=slot, guard=guard))
        for owner, slot, guard in (
            ("left", "slot0", "first"),
            ("left", "slot0", "last"),
            ("right", "slot0", "first"),
            ("left", "slot1", "first"),
        )
    )
    layout = FrontierLayout(keys)
    states = tuple(Frontier(layout, (prefix(key, 4),)) for key in keys)
    assert not Frontier.must_merge(states).known
    joined = layout.empty().acquire(*states)
    assert len(joined) == layout.capacity == 4
    assert all(joined.covers(prefix(key, 4)) for key in keys)
    unknown = domain(partition=Partition(owner="left", slot="slot0", guard="first or last"))
    with pytest.raises(FrontierError, match="undeclared") as error:
        joined.covers(prefix(unknown))
    assert error.value.kind is ErrorKind.UNSUPPORTED


@pytest.mark.parametrize("milestone", (Milestone.NATIVE_END, Milestone.SET_EFFECT))
def test_individual_end_and_set_effect_cannot_become_cumulative_prefixes(milestone):
    with pytest.raises(FrontierError, match="not cumulative.*real retirement capture") as error:
        domain(milestone=milestone)
    assert isinstance(error.value, CompileError)
    assert error.value.kind is ErrorKind.UNSUPPORTED


def test_set_capture_is_a_snapshot_and_does_not_retire_following_native():
    retired, queued = domain(), domain(milestone=Milestone.ADMISSION)
    layout = FrontierLayout((retired, queued))
    source = Frontier(layout, (prefix(retired, lexical=1),))
    capture = source.capture(Actor.AIV0, Engine.V)
    # Later dispatch/admission of another native contributes no retirement.
    source = source.record(prefix(queued, lexical=2))
    consumer = layout.empty().acquire(capture)
    assert consumer.covers(prefix(retired, lexical=1))
    assert not source.covers(prefix(retired, lexical=2))
    assert not consumer.covers(prefix(retired, lexical=2))
    assert not consumer.covers(prefix(queued, lexical=2))
    # Only a separately witnessed retirement capture may advance this prefix.
    later = source.record(prefix(retired, lexical=2))
    assert later.covers(prefix(retired, lexical=2))
    assert not capture.frontier.covers(prefix(retired, lexical=2))


def test_state_and_capture_payloads_are_immutable():
    key = domain()
    state = Frontier(FrontierLayout((key,)), (prefix(key),))
    with pytest.raises(TypeError):
        state.known[key] = PrefixRank(key.clock, 1)
    with pytest.raises(FrozenInstanceError):
        state.layout = FrontierLayout(())
    capture = state.capture(Actor.AIV0, Engine.V)
    with pytest.raises(FrozenInstanceError):
        capture.frontier = state.layout.empty()


def test_cross_clock_comparison_is_diagnostic_but_independent_clocks_coexist():
    first, last = Clock("outer"), Clock("inner")
    with pytest.raises(FrontierError, match="across frontier clocks") as error:
        PrefixRank(first).compare(PrefixRank(last))
    assert error.value.kind is ErrorKind.UNSUPPORTED
    key = domain(clock=first)
    with pytest.raises(FrontierError, match="clock binding") as error:
        KnownPrefix(key, PrefixRank(last))
    assert error.value.kind is ErrorKind.UNSUPPORTED
    other = domain(clock=last)
    layout = FrontierLayout((key, other))
    state = Frontier(layout, (prefix(key),)).acquire(Frontier(layout, (prefix(other),)))
    assert len(state) == 2


def test_same_spelling_does_not_bind_distinct_clock_declarations_or_layouts():
    first, second = Clock("i"), Clock("i")
    assert first is not second and first != second
    with pytest.raises(FrontierError, match="across frontier clocks"):
        PrefixRank(first).compare(PrefixRank(second))
    left, right = domain(clock=first), domain(clock=second)
    left_layout, right_layout = FrontierLayout((left,)), FrontierLayout((right,))
    with pytest.raises(FrontierError, match="same fixed layout"):
        left_layout.empty().acquire(right_layout.empty())


def test_same_spelling_does_not_pair_handoffs_in_distinct_clock_declarations():
    first, second = Clock("i"), Clock("i")
    key = domain(clock=first)
    layout = FrontierLayout((key,))
    expected = Handoff("ready", first)
    other = Handoff("ready", second)
    capture = Frontier(layout, (prefix(key),)).capture(Actor.AIV0, Engine.V, handoff=other)
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire(capture, expected=expected)
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire_quorum((Actor.AIV0,), (capture,), expected=expected)


def test_and_quorum_needs_all_actual_contributors_even_if_one_payload_knows_both():
    keys = tuple(domain(actor=actor, engine=Engine.MTE2) for actor in (Actor.AIV0, Actor.AIV1))
    layout = FrontierLayout(keys)
    handoff = Handoff("release", Clock("slot_epoch"), partition=Partition(owner="gm-slot"))
    complete_knowledge = Frontier(layout, tuple(prefix(key) for key in keys))
    first = complete_knowledge.capture(Actor.AIV0, Engine.MTE2, handoff=handoff)
    second = Frontier(layout, (prefix(keys[1]),)).capture(Actor.AIV1, Engine.MTE2, handoff=handoff)
    empty = layout.empty()
    # An ordinary acquired witness supplies its real payload once: OR paths
    # for the same fact need no duplicate origin or duplicate state entry.
    assert empty.acquire(first, first, expected=handoff) == complete_knowledge
    with pytest.raises(FrontierError, match="incomplete AND quorum.*aiv1"):
        empty.acquire_quorum((Actor.AIV0, Actor.AIV1), (first,), expected=handoff)
    with pytest.raises(FrontierError, match="repeats a contributor"):
        empty.acquire_quorum((Actor.AIV0, Actor.AIV1), (first, first), expected=handoff)
    assert (
        empty.acquire_quorum((Actor.AIV0, Actor.AIV1), (second, first), expected=handoff)
        == complete_knowledge
    )
    assert not empty.known


@pytest.mark.parametrize(
    "change",
    (
        {"protocol": "another_release"},
        {"clock": Clock("another_epoch")},
        {"ordinal": -1},
        {"partition": Partition(owner="another-owner", slot="slot0")},
        {"partition": Partition(owner="gm-slot", slot="slot1")},
        {"partition": Partition(owner="gm-slot", slot="slot0", guard="last")},
    ),
)
def test_quorum_rejects_mixed_protocol_epoch_clock_owner_slot_or_guard(change):
    keys = tuple(domain(actor=actor, engine=Engine.MTE2) for actor in (Actor.AIV0, Actor.AIV1))
    layout = FrontierLayout(keys)
    handoff = Handoff(
        "release", Clock("slot_epoch"), partition=Partition(owner="gm-slot", slot="slot0")
    )
    wrong = replace(handoff, **change)
    first = Frontier(layout, (prefix(keys[0]),)).capture(Actor.AIV0, Engine.MTE2, handoff=handoff)
    second = Frontier(layout, (prefix(keys[1]),)).capture(Actor.AIV1, Engine.MTE2, handoff=wrong)
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire_quorum((Actor.AIV0, Actor.AIV1), (first, second), expected=handoff)
    # Agreement between two wrong tokens still does not satisfy this acquire.
    first = replace(first, handoff=wrong)
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire_quorum((Actor.AIV0, Actor.AIV1), (first, second), expected=handoff)


def test_native_snapshot_cannot_supply_a_protocol_quorum_contribution():
    key = domain()
    layout = FrontierLayout((key,))
    native = Frontier(layout, (prefix(key),)).capture(Actor.AIV0, Engine.V)
    handoff = Handoff("release", Clock("slot_epoch"))
    assert layout.empty().acquire(native).covers(prefix(key))
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire_quorum((Actor.AIV0,), (native,), expected=handoff)


def test_single_producer_protocol_acquire_also_checks_the_expected_handoff():
    key = domain()
    layout = FrontierLayout((key,))
    handoff = Handoff("ready", Clock("tile_epoch"), partition=Partition(owner="tile"))
    capture = Frontier(layout, (prefix(key),)).capture(Actor.AIV0, Engine.V, handoff=handoff)
    with pytest.raises(FrontierError, match="requires an expected handoff"):
        layout.empty().acquire(capture)
    with pytest.raises(FrontierError, match="expected handoff"):
        layout.empty().acquire(capture, expected=replace(handoff, ordinal=1))
    assert layout.empty().acquire(capture, expected=handoff).covers(prefix(key))


def test_notification_identity_does_not_enter_or_expand_the_prefix_state():
    key = domain()
    layout = FrontierLayout((key,))
    state = Frontier(layout, (prefix(key),))
    original, work = state, FrontierWork()
    for epoch in range(1000):
        handoff = Handoff("ready", Clock("tile_epoch"), ordinal=epoch)
        capture = state.capture(Actor.AIV0, Engine.V, handoff=handoff)
        state = state.acquire(capture, expected=handoff, work=work)
    assert state is original
    assert len(state) == layout.capacity == 1
    assert work.entries == work.comparisons == 1000
    assert work.updates == work.copied_entries == 0


@pytest.mark.parametrize(
    "change",
    (
        {"protocol": ""},
        {"clock": "g"},
        {"ordinal": True},
        {"partition": "owner"},
    ),
)
def test_malformed_handoff_is_invalid_input(change):
    values = {"protocol": "ready", "clock": Clock("tile_epoch"), **change}
    with pytest.raises(FrontierError) as error:
        Handoff(**values)
    assert error.value.kind is ErrorKind.INVALID


def test_repeated_history_and_queries_cannot_grow_fixed_state_or_copy_on_idempotent_join():
    key = domain()
    layout = FrontierLayout((key,))
    state = layout.empty()
    for index in range(1000):
        state = state.record(prefix(key, offset=index))
        assert len(state) == layout.capacity == 1
    capture, work = state.capture(Actor.AIV0, Engine.V), FrontierWork()
    for _ in range(1000):
        assert state.acquire(capture, work=work) is state
        assert state.covers(prefix(key, offset=999))
    assert work.entries == work.comparisons == 1000
    assert work.updates == work.copied_entries == 0


@pytest.mark.parametrize("rings", ((2, 3), (2, 3, 5), (2, 3, 5, 7)))
def test_independent_component_join_and_branch_work_is_additive_not_period_product(rings):
    # Each independent ring declares its actual slot keys in its own clock.
    # No global LCM, origin vectors, paths, or Cartesian guard set is built.
    keys = tuple(
        domain(
            clock=Clock(f"ring{ring}"), partition=Partition(owner=f"ring{ring}", slot=f"slot{slot}")
        )
        for ring in rings
        for slot in range(ring)
    )
    layout = FrontierLayout(keys)
    components = tuple(Frontier(layout, (prefix(key),)) for key in keys)
    join_work = FrontierWork()
    combined = layout.empty().acquire(*components, work=join_work)
    assert len(combined) == layout.capacity == sum(rings)
    assert join_work.entries == join_work.updates == sum(rings)
    assert join_work.comparisons == join_work.copied_entries == 0
    branch_work = FrontierWork()
    later = combined.record(*(prefix(key, offset=1) for key in keys))
    assert Frontier.must_merge((combined, later), work=branch_work) is combined
    assert branch_work.entries == branch_work.comparisons == 2 * sum(rings)
    disjoint_work = FrontierWork()
    assert not Frontier.must_merge(components, work=disjoint_work).known
    assert disjoint_work.entries <= 2


def test_layout_is_fixed_and_independent_of_declaration_order():
    first, second = domain(), domain(engine=Engine.MTE2)
    with pytest.raises(FrontierError, match="duplicate domain"):
        FrontierLayout((first, first))
    left, right = FrontierLayout((first, second)), FrontierLayout((second, first))
    assert left == right
    state = Frontier(left, (prefix(first),)).acquire(Frontier(right, (prefix(second),)))
    assert len(state) == 2
    unknown = replace(first, partition=Partition(slot="undeclared"))
    with pytest.raises(FrontierError, match="undeclared") as error:
        state.record(prefix(unknown))
    assert error.value.kind is ErrorKind.UNSUPPORTED


@pytest.mark.parametrize(
    "change",
    (
        {"actor": "aiv0"},
        {"engine": Engine.M},
        {"milestone": "retirement_capture"},
        {"partition": "first"},
    ),
)
def test_malformed_target_domains_are_invalid_input(change):
    with pytest.raises(FrontierError) as error:
        domain(**change)
    assert error.value.kind is ErrorKind.INVALID


def test_controller_issue_is_distinct_from_engine_admission():
    issue = domain(engine=None, milestone=Milestone.ISSUE)
    admission = domain(milestone=Milestone.ADMISSION)
    layout = FrontierLayout((issue, admission))
    state = Frontier(layout, (prefix(issue),))
    assert state.covers(prefix(issue))
    assert not state.covers(prefix(admission))
    with pytest.raises(FrontierError, match="no pipeline engine"):
        domain(engine=Engine.V, milestone=Milestone.ISSUE)
