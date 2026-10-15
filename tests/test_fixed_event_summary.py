"""Fixed child event boundaries against independent chronological Set/Waits."""

import re
from dataclasses import replace
from itertools import combinations

import pytest

from examples.gemm.hierarchical import hierarchical_gemm
from tests.fixtures.cv import vector_source
from xdsl_npu import compile, region_summary
from xdsl_npu import language as N
from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.loop_graph import Channel, channels
from xdsl_npu.loop_model import extract_loop
from xdsl_npu.model import CompileError
from xdsl_npu.region_summary import Access, EventReuse, fixed_event_reuse, summarize_local_events

BANK = ("aiv0", "MTE2", "V")
IDS = (7, 2, 5, 1, 6, 0, 4, 3)


def channel(period=1, residues=(0,), width=1, distance=0, *, last=False):
    return Channel(
        "child",
        "local",
        (("aiv0", "publish"),),
        (("aiv0", "consume"),),
        distance,
        period,
        residues,
        IDS[:width],
        last=last,
    )


def oracle(protocol, bank, count):
    """Enumerate small real source epochs, independently of ordinal algebra."""
    used, ordinal = {}, 0
    for generation in range(count):
        if generation + protocol.distance >= count:
            continue
        if generation % protocol.period not in protocol.residues:
            continue
        if protocol.last and generation != count - protocol.distance - 1:
            continue
        event_id = protocol.ids[ordinal % len(protocol.ids)]
        wait = generation + protocol.distance
        if event_id not in used:
            used[event_id] = [generation, wait]
        else:
            used[event_id][1] = wait
        ordinal += 1
    return tuple(
        EventReuse(
            protocol.name,
            event_id,
            bank,
            CountDomain(1, 0, exact=count),
            Access(protocol.destinations[0], Position(False, last)),
            Access(protocol.sources[0], Position(False, first)),
        )
        for event_id, (first, last) in used.items()
    )


def residue_families(period):
    return (
        residues for size in range(1, period + 1) for residues in combinations(range(period), size)
    )


@pytest.mark.parametrize("period", range(1, 6))
@pytest.mark.parametrize("width", range(1, 9))
def test_fixed_boundaries_match_every_small_residue_family_and_notification_prefix(period, width):
    for residues in residue_families(period):
        for distance in (0, 1, 2, 5, 9):
            protocol = channel(period, residues, width, distance)
            counts = {
                *range(33),
                *(
                    distance + n
                    for n in (
                        period * width - 1,
                        period * width,
                        period * width + 1,
                        2 * period * width + 1,
                    )
                ),
            }
            for count in sorted(counts):
                assert fixed_event_reuse(protocol, BANK, count) == oracle(protocol, BANK, count), (
                    protocol,
                    count,
                )


@pytest.mark.parametrize("period", range(1, 7))
def test_terminal_boundaries_match_exact_final_set_and_wait_for_all_small_residues(period):
    for residues in residue_families(period):
        for distance in (0, 1, 2, 5, 9):
            protocol = channel(period, residues, distance=distance, last=True)
            for count in range(40):
                assert fixed_event_reuse(protocol, BANK, count) == oracle(protocol, BANK, count), (
                    protocol,
                    count,
                )


@pytest.mark.parametrize(
    "protocol,count",
    [
        (channel(width=8), 0),
        (channel(width=8, distance=5), 5),
        (channel(17, (8, 16), width=8), 8),
        (channel(17, (8, 16), distance=5, last=True), 13),
    ],
)
def test_no_real_notification_means_no_event_reuse_boundary(protocol, count):
    assert fixed_event_reuse(protocol, BANK, count) == ()


def test_only_ids_with_a_real_notification_are_retained_in_physical_rotation_order():
    protocol = channel(11, (3, 10), width=8, distance=7)
    events = fixed_event_reuse(protocol, BANK, 22)
    assert tuple(event.event_id for event in events) == IDS[:3]
    assert tuple(event.republished.position.offset for event in events) == (3, 10, 14)
    assert tuple(event.consumed.position.offset for event in events) == (10, 17, 21)
    assert all(event.domain == CountDomain(1, 0, exact=22) for event in events)


def test_huge_count_and_period_create_only_one_exact_domain_and_one_boundary_per_id(monkeypatch):
    domains = []

    def counted_domain(*args, **kwargs):
        result = CountDomain(*args, **kwargs)
        domains.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed event summary enumerated parametric count domains")

    def bounded_range(*args):
        result = range(*args)
        assert len(result) <= len(IDS), "fixed event summary traversed count or protocol period"
        return result

    monkeypatch.setattr(region_summary, "CountDomain", counted_domain)
    monkeypatch.setattr(region_summary, "_event_reuse", forbidden)
    monkeypatch.setattr(region_summary, "range", bounded_range, raising=False)
    # This period cannot be traversed. Both cases have the same eight output
    # records even though they complete very different numbers of ID rotations.
    period = (1 << 40) + 17
    protocol = channel(period, (3, period - 1), width=8, distance=17)
    for count in (6 * period, (1 << 63) - 33):
        domains.clear()
        events = fixed_event_reuse(protocol, BANK, count)
        assert len(domains) == 1 and domains[0] == CountDomain(1, 0, exact=count)
        assert len(events) == len(IDS)
        assert tuple(event.event_id for event in events) == IDS
        for index, event in enumerate(events):
            first = event.republished.position.evaluate(count)
            last = event.consumed.position.evaluate(count)
            assert 0 <= first <= last - protocol.distance < count - protocol.distance
            assert first % period in protocol.residues
            assert (last - protocol.distance) % period in protocol.residues
            # Independently rank source epochs by counting each residue's
            # arithmetic progression. The boundary is the last use of this ID.
            ordinal = (
                sum(
                    max(0, (last - protocol.distance - residue) // period + 1)
                    for residue in protocol.residues
                )
                - 1
            )
            assert ordinal % len(IDS) == index
            notifications = sum(
                max(0, (count - protocol.distance - 1 - residue) // period + 1)
                for residue in protocol.residues
            )
            assert ordinal + len(IDS) >= notifications
            assert (
                sum(max(0, (first - residue) // period + 1) for residue in protocol.residues) - 1
                == index
            )


def test_fixed_summarize_local_events_never_constructs_parametric_domains(monkeypatch):
    loop = extract_loop(compile(vector_source(5)).plan)
    actual_channels = channels(loop)
    assert actual_channels and loop.constant_count == 5

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed child called the runtime event-domain enumerator")

    monkeypatch.setattr(region_summary, "_event_reuse", forbidden)
    result = summarize_local_events(loop)
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    expected = tuple(
        event
        for protocol in actual_channels.values()
        for event in oracle(
            protocol,
            (
                protocol.sources[0][0],
                tasks[protocol.sources[0]].pipe,
                tasks[protocol.destinations[0]].pipe,
            ),
            5,
        )
    )
    assert result == expected
    assert all(event.domain.exact == 5 for event in result)


def test_runtime_summarize_local_events_keeps_the_original_domain_path(monkeypatch):
    loop = extract_loop(compile(vector_source(N.dynamic("tasks"))).plan)
    assert loop.constant_count is None
    expected = summarize_local_events(loop)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime child used the fixed-count event summarizer")

    monkeypatch.setattr(region_summary, "fixed_event_reuse", forbidden)
    assert summarize_local_events(loop) == expected
    assert expected and all(event.domain.exact is None for event in expected)


def test_fixed_invocation_credits_keep_their_existing_carried_contract(monkeypatch):
    source = (
        re.sub(r"\bN\b", "COLS", hierarchical_gemm.source)
        .replace("T.", "N.")
        .replace("outputs: N.int64,", "outputs: N.constexpr = OUTPUTS,")
        .replace("panels: N.int64,", "panels: N.constexpr = PANELS,")
    )
    loop = extract_loop(
        compile(
            N.from_source(source, constants={"OUTPUTS": 2, "PANELS": 7}, disjoint_external=True)
        ).plan
    )
    protocols = channels(loop)
    assert loop.constant_count == 7 and any(protocol.carry for protocol in protocols.values())
    original_runtime, original_fixed = region_summary._event_reuse, region_summary.fixed_event_reuse
    runtime_calls, fixed_calls = [], []

    def carried(protocol, bank):
        assert protocol.carry
        runtime_calls.append(protocol.name)
        return original_runtime(protocol, bank)

    def fixed(protocol, bank, count):
        assert not protocol.carry and count == 7
        fixed_calls.append(protocol.name)
        return original_fixed(protocol, bank, count)

    monkeypatch.setattr(region_summary, "_event_reuse", carried)
    monkeypatch.setattr(region_summary, "fixed_event_reuse", fixed)
    summaries = summarize_local_events(loop)
    assert runtime_calls and fixed_calls and summaries
    assert set(runtime_calls) == {p.name for p in protocols.values() if p.carry}
    assert set(fixed_calls) == {p.name for p in protocols.values() if not p.carry}
    for summary in summaries:
        assert summary.domain.contains(7)
        if not protocols[summary.channel].carry:
            assert summary.domain == CountDomain(1, 0, exact=7)


@pytest.mark.parametrize("count", [-1, True, None])
def test_fixed_event_count_must_be_a_nonnegative_integer(count):
    with pytest.raises(CompileError, match="fixed event count must be nonnegative"):
        fixed_event_reuse(channel(), BANK, count)


@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize(
    "protocol", [replace(channel(), kind="cross"), replace(channel(), carry=True)]
)
def test_cross_and_carried_channels_are_rejected_even_without_notifications(protocol, count):
    with pytest.raises(CompileError, match="participant analysis|cannot carry invocation credits"):
        fixed_event_reuse(protocol, BANK, count)


@pytest.mark.parametrize(
    "protocol,diagnostic",
    [
        (replace(channel(), ids=()), "IDs must be assigned"),
        (replace(channel(), ids=(2, 2)), "IDs must be distinct"),
        (channel(width=2, last=True), "terminal child channel needs one event ID"),
        (replace(channel(), period=0), "invalid fixed child event activation"),
        (replace(channel(), distance=-1), "invalid fixed child event activation"),
        (replace(channel(), residues=()), "invalid fixed child event activation"),
        (channel(3, (2, 1)), "residues must be sorted"),
        (channel(3, (1, 1)), "residues must be sorted"),
        (channel(3, (-1,)), "residues must be sorted"),
        (channel(3, (3,)), "residues must be sorted"),
        (
            replace(channel(), destinations=(("aic", "consume"),)),
            "one source and destination in one actor",
        ),
    ],
)
def test_invalid_event_metadata_is_rejected_before_ordinal_arithmetic(protocol, diagnostic):
    with pytest.raises(CompileError, match=diagnostic):
        fixed_event_reuse(protocol, BANK, 17)
