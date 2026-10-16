"""A parent boundary notification inside one fixed child event's idle gap.

Only distance-one, single-ID parent channels borrow, with at most one parent
per physical child ID. The child keeps its actual ID tuple and native cases.
These checks add no ordering edge; final checking reconstructs actual tuples.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from .budget import poll
from .completion import Position
from .diagnostics import CompileError, require
from .dialects.physical import LoopChannelOp
from .region_summary import summarize_local_events
from .scope_domains import legacy_activation, shifted
from .scope_events import resource_sites
from .scope_predicates import PARENT_CASES

MAX_BORROW_OPTIONS = 8192
MAX_BORROW_QUERIES = 262144


class ParentEventBankExhausted(CompileError):
    """A proved width cannot fit the currently reserved physical bank."""

    def __init__(self, bank, needed, available, *, conditional=False):
        self.bank, self.needed, self.available = bank, needed, available
        super().__init__(
            f"{bank}: conditional parent event bank exhausted"
            if conditional
            else f"{bank}: proved parent pool packing needs {needed} IDs, only {available} free"
        )


@dataclass(frozen=True)
class ChildSlot:
    bank: tuple[str, str, str]
    event_id: int
    owner: tuple[str, str]
    resources: tuple
    conditional: bool


@dataclass(frozen=True)
class BorrowOption:
    channel: str
    slot: ChildSlot
    guarded: tuple
    anchored: tuple


def child_slots(program, analyses):
    """Read actual fixed-child ID ownership, retaining each native case."""
    owners, resources, conditional = {}, defaultdict(dict), {}
    for uid, leaf in program.leaves.items():
        declarations = {
            op.sym_name.data: tuple(i.data for i in op.event_ids)
            for op in leaf.stream.walk()
            if isinstance(op, LoopChannelOp)
        }
        require(
            declarations == {name: p.ids for name, p in analyses[uid][1].items()},
            "child event borrowing needs the actual current ID tuples",
        )
        variants = (
            analyses.variants(uid) if hasattr(analyses, "variants") else ((None, analyses[uid]),)
        )
        for case, analysis in variants:
            require(case is None or case in PARENT_CASES, "unsupported child event borrowing case")
            count = analysis[0].constant_count
            for resource in summarize_local_events(analysis[0]):
                if not resource.domain.contains(count):
                    continue
                bank = leaf.actor, *resource.bank[1:]
                key, owner = (bank, resource.event_id), (uid, resource.channel)
                require(
                    resource.event_id in declarations.get(resource.channel, ())
                    and owners.get(key, owner) == owner,
                    "child event borrowing has no unique actual resource owner",
                )
                owners[key] = owner
                sites = resource_sites(uid, leaf.actor, count, resource)
                resources[key][uid, *sites] = None
                conditional[key] = bool(leaf.cases)
    return {
        key: ChildSlot(*key, owners[key], tuple(sites), conditional[key])
        for key, sites in resources.items()
    }


def _bank(protocol, proof):
    from .scope_guarded_events import _bank

    return _bank(protocol, proof)


def prepare_borrow_options(protocols, proof, slots):
    """Prepare all eligible parent/child-ID checks, bounded by eight IDs each.

    A distance-one source lies in [0,N-2]. If its child owner is absent at g,
    the only earlier case can be FIRST; if absent at g+1, its only later case
    can be LAST. The anchored clauses cover exactly those skipped middles.
    This relies on the validated single/first/middle/last native partition;
    arbitrary sparse child activation is outside this borrowing contract.
    """
    from .scope_guarded_events import self_reuse_queries

    result, work = [], 0
    for protocol in protocols:
        if (
            protocol.kind != "local"
            or protocol.distance != 1
            or len(protocol.ids) > 1
            or protocol.last
            or protocol.carry
        ):
            continue
        bank, send, wait = _bank(protocol, proof)
        activation = protocol.activation or legacy_activation(1, protocol.period, protocol.residues)
        self_queries = self_reuse_queries(replace(protocol, activation=activation), proof, 1)
        for slot in slots.values():
            if slot.bank != bank:
                continue
            poll("parent child event borrowing", channel=protocol.name, event_id=slot.event_id)
            require(
                len(result) < MAX_BORROW_OPTIONS, "parent child borrowing option budget exceeded"
            )
            direct, anchored = list(self_queries), []
            for uid, consumed, published in slot.resources:
                old_waits = proof.completion.commands_for(uid, consumed)
                new_sets = proof.completion.commands_for(uid, published)
                require(old_waits and new_sets, "child borrower lost an actual Set/Wait endpoint")
                direct.extend((old, send, 0, activation) for old in old_waits)
                direct.extend((wait, new, 0, shifted(activation, 1)) for new in new_sets)
                if slot.conditional:
                    anchored.extend(
                        (old, send, Position(False, 0), None, activation) for old in old_waits
                    )
                    anchored.extend(
                        (wait, new, None, Position(True, 0), shifted(activation, 1))
                        for new in new_sets
                    )
            work += len(direct) + len(anchored)
            require(work <= MAX_BORROW_QUERIES, "parent child borrowing query budget exceeded")
            result.append(BorrowOption(protocol.name, slot, tuple(direct), tuple(anchored)))
    return tuple(result)


def check_borrow_options(options, proof, *, statistics=None):
    """One guarded batch and one anchored batch; no native graph is rebuilt."""
    options = tuple(options)
    direct = proof.guarded().covered_many(q for option in options for q in option.guarded)
    anchors = tuple(q for option in options for q in option.anchored)
    anchored = proof.anchored().covered_many(anchors) if anchors else ()
    result, first, second = [], 0, 0
    for option in options:
        end, tail = first + len(option.guarded), second + len(option.anchored)
        result.append(all(direct[first:end]) and all(anchored[second:tail]))
        first, second = end, tail
    if statistics is not None:
        for key, count in (
            ("borrow_options", len(options)),
            ("borrow_guarded_queries", len(direct)),
            ("borrow_anchored_queries", len(anchored)),
        ):
            statistics[key] = statistics.get(key, 0) + count
    return tuple(result)


def allocate_borrows(protocols, proof, bank, *, statistics=None):
    slots = {
        key: slot
        for key, slot in child_slots(proof.program, proof.analyses).items()
        if slot.bank == bank
    }
    options = prepare_borrow_options(protocols, proof, slots)
    answers = check_borrow_options(options, proof, statistics=statistics)
    choices = defaultdict(list)
    for option, valid in zip(options, answers, strict=True):
        if valid:
            choices[option.channel].append(option.slot.event_id)
    assigned, occupied = {}, {}

    def match(channel, visited):
        for event_id in sorted(choices[channel]):
            if event_id in visited:
                continue
            visited.add(event_id)
            previous = occupied.get(event_id)
            if previous is None or match(previous, visited):
                occupied[event_id], assigned[channel] = channel, event_id
                return True
        return False

    for channel in choices:
        match(channel, set())
    if statistics is not None:
        statistics["borrowed_parent_channels"] = statistics.get(
            "borrowed_parent_channels", 0
        ) + len(assigned)
    return assigned


def verify_actual_borrows(protocols, proof, used_banks, *, statistics=None):
    """Classify actual overlaps and independently prove every borrowed ID."""
    slots, selected, borrowers = None, [], set()
    for protocol in protocols:
        if protocol.kind != "local":
            continue
        require(
            isinstance(protocol.ids, tuple)
            and 1 <= len(protocol.ids) <= 8
            and len(set(protocol.ids)) == len(protocol.ids)
            and all(type(i) is int and 0 <= i < 8 for i in protocol.ids),
            "invalid parent event pool IDs",
        )
        bank, _, _ = _bank(protocol, proof)
        if not set(protocol.ids).intersection(used_banks.get(bank, ())):
            continue
        require(
            len(protocol.ids) == 1
            and protocol.distance == 1
            and not protocol.last
            and not protocol.carry,
            "parent event pool overlaps child/configuration resource without a supported borrower",
        )
        if slots is None:
            slots = child_slots(proof.program, proof.analyses)
        key = bank, protocol.ids[0]
        require(key in slots, "parent event overlaps configuration or has no actual child owner")
        require(key not in borrowers, "multiple parent borrowers share one physical child event ID")
        borrowers.add(key)
        selected.extend(prepare_borrow_options((protocol,), proof, {key: slots[key]}))
    answers = check_borrow_options(selected, proof, statistics=statistics) if selected else ()
    for option, valid in zip(selected, answers, strict=True):
        require(
            valid,
            f"{option.channel}: parent child event borrower can overwrite an unconsumed notification",
        )
    return frozenset(option.channel for option in selected)
