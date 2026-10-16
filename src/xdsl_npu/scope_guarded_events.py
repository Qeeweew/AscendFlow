"""Dedicated binary pools for count/first/last-conditional parent protocols.

Only ID assignment changes. Every reuse query uses the actual source predicate
and the same guarded completion proof as memory. These pools are deliberately
disjoint; the ordinary prefix-pool sharing theorem does not apply to them.
"""

from collections import defaultdict
from dataclasses import replace

from .budget import poll
from .diagnostics import require
from .scope_domains import Activation, affine, position, validate_activation
from .scope_event_borrow import ParentEventBankExhausted
from .symbolic_completion import add, le, meet


def _bank(protocol, proof):
    require(protocol.kind == "local", "conditional event allocation requires a local channel")
    require(
        len(protocol.sources) == len(protocol.destinations) == 1
        and protocol.sources[0][0] == protocol.destinations[0][0],
        "conditional local pool needs one actual actor",
    )
    actor = protocol.sources[0][0]
    send = proof.events[actor, protocol.name, "set"]
    wait = proof.events[actor, protocol.name, "wait"]
    return (actor, send.task.pipe, wait.task.pipe), send, wait


def self_reuse_queries(protocol, proof, width):
    """Last Wait precedes the following Set using that exact ordinal ID."""
    value = validate_activation(protocol.activation, distance=protocol.distance)
    _, send, wait = _bank(protocol, proof)
    active = tuple(r for r in range(proof.graph.period) if r % value.period in value.residues)
    queries = []
    for rank, residue in enumerate(active):
        poll("conditional parent event reuse", channel=protocol.name, width=width)
        cycles, index = divmod(rank + width, len(active))
        step = active[index] + cycles * proof.graph.period - residue
        first, last = affine(value.first), add(affine(value.last), -step)
        domain = meet(value.count, le(first, last))
        if domain is None:
            continue  # This invocation cannot publish the same ID twice.
        distance = protocol.distance
        required = Activation(
            domain,
            position(add(first, distance)),
            position(add(last, distance)),
            proof.graph.period,
            ((residue + distance) % proof.graph.period,),
        )
        queries.append((wait, send, step - distance, required))
    return tuple(queries)


def self_reuse(protocol, proof, width):
    return all(proof.guarded().covered_many(self_reuse_queries(protocol, proof, width)))


def allocate(protocols, proof, used_banks, *, verify=False, statistics=None, _proofs=None):
    """Reserve complete private pools; never infer ordering from ID equality."""
    from .scope_event_pools import _Proofs

    checks = _Proofs(proof, statistics) if _proofs is None else _proofs
    require(checks.proof is proof, "conditional allocation needs its actual completion owner")
    checks.statistics.setdefault("conditional_self_queries", 0)
    checks.statistics.setdefault("conditional_self_batches", 0)
    used = defaultdict(set, {key: set(ids) for key, ids in used_banks.items()})
    result, allocated = [], 0
    for protocol in protocols:
        if protocol.activation is None or protocol.kind == "barrier":
            result.append(protocol)
            continue
        bank, _, _ = _bank(protocol, proof)
        if verify:
            ids = protocol.ids
            require(
                1 <= len(ids) <= 8
                and len(set(ids)) == len(ids)
                and all(type(i) is int and 0 <= i < 8 for i in ids)
                and not used[bank].intersection(ids),
                "conditional parent event pool overlaps another resource or has invalid IDs",
            )
            require(
                checks.conditional_self(protocol, len(ids)),
                f"{protocol.name}: conditional parent event re-Set can precede its old Wait",
            )
        else:
            free = [i for i in range(8) if i not in used[bank]]
            if not free:
                checks.statistics["zero_free_bank_rejections"] = (
                    checks.statistics.get("zero_free_bank_rejections", 0) + 1
                )
                raise ParentEventBankExhausted(bank, 1, 0, conditional=True)
            width = None
            for candidate in range(1, 9):
                if checks.conditional_self(protocol, candidate):
                    width = candidate
                    break
            require(
                width is not None, f"{bank}: no conditional parent event width <=8 proves reuse"
            )
            if len(free) < width:
                raise ParentEventBankExhausted(bank, width, len(free), conditional=True)
            ids = tuple(free[:width])
        used[bank].update(ids)
        result.append(replace(protocol, ids=ids))
        allocated += len(ids)
    checks.statistics["conditional_allocated_ids"] = allocated
    return tuple(result), used
