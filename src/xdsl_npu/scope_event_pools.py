"""Static parent binary-event pools justified by existing consumption paths.

Pool allocation changes only ID tuples. It creates no graph edge or native
instruction. The checker reconstructs all proofs from actual assigned tuples;
it does not consume the allocator's merge decisions or proof cache.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations

from .budget import bounded, poll
from .model import require
from .scope_event_borrow import (
    ParentEventBankExhausted,
    allocate_borrows,
    verify_actual_borrows,
)

MAX_POOL_QUERIES = 1_000_000
MAX_POOL_MERGE_WORK = 1_000_000


@dataclass(frozen=True)
class _Local:
    protocol: object
    source: object
    destination: object
    bank: tuple[str, str, str]
    active: tuple[int, ...]


def _locals(protocols, proof):
    require(len({p.name for p in protocols}) == len(protocols), "duplicate parent pool channel")
    result = []
    for p in protocols:
        require(p.kind in {"local", "cross", "barrier"}, "unknown parent pool channel kind")
        if p.kind != "local":
            continue
        require(p.activation is None, "ordinary event pools cannot discard conditional activation")
        require(
            not p.last and not p.carry, "parent pools require nonterminal, non-carried channels"
        )
        require(
            len(p.sources) == len(p.destinations) == 1 and p.sources[0][0] == p.destinations[0][0],
            "parent local pool needs one actual actor",
        )
        require(
            type(p.distance) is int
            and p.distance >= 0
            and type(p.period) is int
            and p.period > 0
            and proof.graph.period % p.period == 0
            and p.residues
            and p.residues == tuple(sorted(set(p.residues)))
            and all(type(r) is int and 0 <= r < p.period for r in p.residues),
            "invalid parent pool activation",
        )
        actor = p.sources[0][0]
        require(
            (actor, p.name, "set") in proof.events and (actor, p.name, "wait") in proof.events,
            "parent pool has no actual Set/Wait commands",
        )
        source, destination = (
            proof.events[actor, p.name, "set"],
            proof.events[actor, p.name, "wait"],
        )
        require(
            source.actor == destination.actor == actor
            and source.kind == "set"
            and destination.kind == "wait"
            and source.channel == destination.channel == p.name
            and source.task.pipe != destination.task.pipe
            and (source.lower, source.upper, source.period, source.residues)
            == (0, p.distance, p.period, p.residues)
            and (destination.lower, destination.upper, destination.period, destination.residues)
            == (
                p.distance,
                0,
                p.period,
                tuple(sorted((r + p.distance) % p.period for r in p.residues)),
            ),
            "parent pool commands disagree with their actual activation/bank",
        )
        result.append(
            _Local(
                p,
                source,
                destination,
                (actor, source.task.pipe, destination.task.pipe),
                tuple(r for r in range(proof.graph.period) if r % p.period in p.residues),
            )
        )
    return tuple(result)


def _reserved(used_banks):
    result = defaultdict(set, {bank: set(ids) for bank, ids in used_banks.items()})
    require(
        all(type(i) is int and 0 <= i < 8 for ids in result.values() for i in ids),
        "invalid child/configuration event resource",
    )
    return result


def _activation(local):
    p = local.protocol
    return p.period, p.residues


def _following(local, rank, width, period):
    cycles, offset = divmod(rank + width, len(local.active))
    return local.active[offset] + cycles * period


class _Proofs:
    def __init__(self, proof, statistics):
        self.proof, self.facts = proof, {}
        self.queries = 0
        self.statistics = {} if statistics is None else statistics
        for kind in ("self", "sharing"):
            for metric in ("queries", "batches"):
                self.statistics.setdefault(kind + "_" + metric, 0)

    def _batch(self, groups, kind):
        queries, spans = [], {}
        for key, group in groups:
            if key in self.facts:
                continue
            if len(spans) % 128 == 0:
                poll("parent event pool " + kind, groups=len(spans), queries=len(queries))
            require(
                self.queries + len(queries) + len(group) <= MAX_POOL_QUERIES,
                "parent event pool query budget exceeded",
            )
            first = len(queries)
            queries.extend(group)
            spans[key] = first, len(queries)
        if not spans:
            return
        answers = (
            self.proof.covered_many(queries)
            if hasattr(self.proof, "covered_many")
            else self.proof.graph.covered_many(queries)
        )
        self.queries += len(queries)
        self.statistics[kind + "_queries"] += len(queries)
        self.statistics[kind + "_batches"] += 1
        self.facts.update((key, all(answers[first:last])) for key, (first, last) in spans.items())

    def selfs(self, locals_, widths):
        self.self_assignments((local, width) for local in locals_ for width in widths)

    def self_assignments(self, assignments):
        period = self.proof.graph.period
        self._batch(
            (
                (
                    ("self", local.protocol.name, width),
                    tuple(
                        (
                            local.destination,
                            r + local.protocol.distance,
                            local.source,
                            _following(local, rank, width, period),
                        )
                        for rank, r in enumerate(local.active)
                    ),
                )
                for local, width in assignments
            ),
            "self",
        )

    def self_ok(self, local, width):
        return self.facts["self", local.protocol.name, width]

    def pairs(self, locals_, widths):
        period = self.proof.graph.period

        def groups():
            for a, b in combinations(locals_, 2):
                if a.bank != b.bank or _activation(a) != _activation(b):
                    continue
                for width in widths:
                    if not self.self_ok(a, width) or not self.self_ok(b, width):
                        continue
                    for orientation, (first, last) in enumerate(((a, b), (b, a))):
                        yield (
                            ("pair", a.protocol.name, b.protocol.name, width, orientation),
                            tuple(
                                query
                                for rank, r in enumerate(first.active)
                                for query in (
                                    (
                                        first.destination,
                                        r + first.protocol.distance,
                                        last.source,
                                        r,
                                    ),
                                    (
                                        last.destination,
                                        r + last.protocol.distance,
                                        first.source,
                                        _following(first, rank, width, period),
                                    ),
                                )
                            ),
                        )

        self._batch(groups(), "sharing")

    def pair_ok(self, a, b, width):
        return any(
            self.facts.get(("pair", a.protocol.name, b.protocol.name, width, order), False)
            for order in (0, 1)
        )


class _MergeWork:
    def __init__(self, statistics):
        self.statistics = statistics
        self.work = self.candidates = 0
        statistics.update(merge_candidates=0, merge_checks=0)

    def tick(self, *, candidate=False):
        self.work += 1
        self.candidates += candidate
        self.statistics.update(merge_candidates=self.candidates, merge_checks=self.work)
        require(self.work <= MAX_POOL_MERGE_WORK, "parent event pool merge work budget exceeded")
        if self.work % 128 == 0 or candidate and self.candidates % 128 == 0:
            poll("parent event pool merge work", checks=self.work, candidates=self.candidates)

    def compatible(self, group, width, members, proofs):
        self.tick(candidate=True)
        for index in group:
            self.tick()
            if not proofs.self_ok(members[index], width):
                return False
        for a, b in combinations(group, 2):
            self.tick()
            if not proofs.pair_ok(members[a], members[b], width):
                return False
        return True


@bounded
def allocate_parent_pools(protocols, proof, used_banks, *, statistics=None):
    """Keep private pools when they fit; borrow only after typed exhaustion."""
    protocols = tuple(protocols)
    pending, borrowed, attempted = protocols, {}, set()
    while True:
        try:
            assigned, used = _allocate_private_pools(
                pending, proof, used_banks, statistics=statistics
            )
            by_name = {p.name: p for p in assigned}
            by_name.update(
                (p.name, replace(p, ids=(borrowed[p.name],)))
                for p in protocols
                if p.name in borrowed
            )
            return tuple(by_name[p.name] for p in protocols), used
        except ParentEventBankExhausted as failure:
            if failure.bank in attempted:
                raise
            attempted.add(failure.bank)
            choices = allocate_borrows(pending, proof, failure.bank, statistics=statistics)
            if not choices:
                raise
            borrowed.update(choices)
            pending = tuple(p for p in pending if p.name not in borrowed)


def _allocate_private_pools(protocols, proof, used_banks, *, statistics=None):
    """Pack local parent channels deterministically, without claiming optimality.

    Start with each channel's minimum self-reuse width. A bank that already
    fits performs no sharing queries. Otherwise, merge the compatible groups
    with the largest proved ID saving, breaking ties by width and input order.
    Nonlocal protocols and the input resource map remain unchanged.
    """
    protocols = tuple(protocols)
    if any(p.activation is not None for p in protocols):
        from .scope_guarded_events import allocate

        conditional = tuple(p for p in protocols if p.activation is not None)
        ordinary = tuple(p for p in protocols if p.activation is None)
        checked, reserved = allocate(conditional, proof, used_banks, statistics=statistics)
        checked_ordinary, used = _allocate_private_pools(
            ordinary, proof, reserved, statistics=statistics
        )
        assigned = {p.name: p for p in (*checked, *checked_ordinary)}
        return tuple(assigned[p.name] for p in protocols), used
    locals_ = _locals(protocols, proof)
    used, checks, assigned = _reserved(used_banks), _Proofs(proof, statistics), {}
    merge_work = _MergeWork(checks.statistics)
    banks = defaultdict(list)
    for local in locals_:
        banks[local.bank].append(local)
    # Banks share a completion topology even though their IDs are independent.
    # Register each required width across banks in one fact propagation; the
    # minimum widths and subsequent deterministic allocation stay unchanged.
    pending, minimum = list(locals_), {}
    for width in range(1, 9):
        checks.selfs(pending, (width,))
        for local in pending:
            if checks.self_ok(local, width):
                minimum[local.protocol.name] = width
        pending = [local for local in pending if local.protocol.name not in minimum]
        if not pending:
            break
    require(
        not pending,
        f"{pending[0].bank if pending else ''}: no proved parent self reuse width <=8",
    )
    merges, shared, allocated = 0, 0, 0
    for bank, members in banks.items():
        pools = [((index,), minimum[local.protocol.name]) for index, local in enumerate(members)]
        free = [i for i in range(8) if i not in used[bank]]
        cost = sum(width for _, width in pools)
        if cost > len(free):
            checks.selfs(members, range(1, 9))
            checks.pairs(members, range(1, 9))
            while cost > len(free):
                poll("parent event pool merge round", pools=len(pools), ids=cost, free=len(free))
                choices = []
                for i, (left, left_width) in enumerate(pools):
                    for j in range(i + 1, len(pools)):
                        right, right_width = pools[j]
                        group = tuple(sorted((*left, *right)))
                        for width in range(1, min(9, left_width + right_width)):
                            if merge_work.compatible(group, width, members, checks):
                                choices.append(
                                    (width - left_width - right_width, width, group, i, j)
                                )
                                break
                if not choices:
                    break
                saving, width, group, i, j = min(choices)
                cost += saving
                pools = [pool for k, pool in enumerate(pools) if k not in (i, j)] + [(group, width)]
                pools.sort(key=lambda pool: pool[0])
                merges += 1
        if cost > len(free):
            raise ParentEventBankExhausted(bank, cost, len(free))
        offset = 0
        for group, width in pools:
            ids = tuple(free[offset : offset + width])
            offset += width
            used[bank].update(ids)
            for index in group:
                assigned[members[index].protocol.name] = ids
            shared += len(group) > 1
        allocated += cost
    checks.statistics.update(merges=merges, shared_pools=shared, allocated_ids=allocated)
    return tuple(
        replace(p, ids=assigned[p.name]) if p.kind == "local" else p for p in protocols
    ), used


@bounded
def verify_parent_pools(protocols, proof, used_banks, *, statistics=None):
    """Reconstruct actual borrowing, then check the remaining private pools."""
    protocols = tuple(protocols)
    borrowed = verify_actual_borrows(protocols, proof, used_banks, statistics=statistics)
    return _verify_private_pools(
        tuple(p for p in protocols if p.name not in borrowed),
        proof,
        used_banks,
        statistics=statistics,
    )


def _verify_private_pools(protocols, proof, used_banks, *, statistics=None):
    """Independently verify actual complete ID tuples and all lifetime paths.

    Every source activation is a prefix of the same ordinal stream. When one
    member is absent at the tail, its slot is skipped using the remaining
    member's self-reuse path. Pair proofs alone only cover the intersection
    of activations and cannot replace these individual self-reuse checks.
    """
    protocols = tuple(protocols)
    if any(p.activation is not None for p in protocols):
        from .scope_guarded_events import allocate

        conditional = tuple(p for p in protocols if p.activation is not None)
        ordinary = tuple(p for p in protocols if p.activation is None)
        _, reserved = allocate(conditional, proof, used_banks, verify=True, statistics=statistics)
        return _verify_private_pools(ordinary, proof, reserved, statistics=statistics)
    locals_ = _locals(protocols, proof)
    used, checks, pools = _reserved(used_banks), _Proofs(proof, statistics), defaultdict(list)
    for local in locals_:
        ids = local.protocol.ids
        require(
            isinstance(ids, tuple)
            and 1 <= len(ids) <= 8
            and len(set(ids)) == len(ids)
            and all(type(i) is int and 0 <= i < 8 for i in ids),
            "invalid parent event pool IDs",
        )
        require(
            not used[local.bank].intersection(ids),
            "parent event pool overlaps child/configuration resource",
        )
        pools[local.bank, ids].append(local)
    keys = tuple(pools)
    for i, (bank, ids) in enumerate(keys):
        for other_bank, other_ids in keys[i + 1 :]:
            require(
                bank != other_bank or not set(ids).intersection(other_ids),
                "parent event pools have partial overlap or different ID tuple order",
            )
    for members in pools.values():
        if len(members) > 1:
            require(
                len({_activation(local) for local in members}) == 1,
                "shared parent pool has different period/residues",
            )
    checks.self_assignments((local, len(local.protocol.ids)) for local in locals_)
    for local in locals_:
        require(
            checks.self_ok(local, len(local.protocol.ids)),
            f"{local.protocol.name}: parent binary event self reuse can precede its old Wait",
        )
    shared = 0
    for (bank, ids), members in pools.items():
        if len(members) > 1:
            checks.pairs(members, (len(ids),))
            for a, b in combinations(members, 2):
                require(
                    checks.pair_ok(a, b, len(ids)),
                    f"{a.protocol.name}/{b.protocol.name}: no proved parent event pool sharing order",
                )
            shared += 1
        used[bank].update(ids)
    checks.statistics.update(shared_pools=shared, allocated_ids=sum(len(ids) for _, ids in pools))
    return used
