"""Actual parent protocol epochs must exist at their native insertion sites.

Completion queries deliberately treat inactive endpoint combinations as
vacuous. This independent contract prevents a declared Set/Wait/fence from
borrowing that convention when ASC skips its conditional native wrapper.
"""

from bisect import bisect_left

from .budget import bounded, poll
from .completion import CountDomain, Position
from .diagnostics import require
from .periodic import bounded_period
from .scope_domains import Activation, affine, legacy_activation, shifted, validate_activation
from .scope_predicates import PARENT_CASES, case_activation
from .symbolic_completion import add, choice, le, meet


def _intersects(required, missing, count):
    """Some original N and aligned epoch belong to both activation domains."""
    domain = meet(required.count, missing.count)
    if count is not None:
        domain = meet(domain, CountDomain(1, 0, exact=count))
    if domain is None:
        return False
    require(
        missing.period == 1 and missing.residues == (0,),
        "native existence cases must use the fixed four-case partition",
    )
    period = required.period
    bounded_period(period, "native protocol activation")
    for count_residue in range(period):
        if count_residue % 128 == 0:
            poll("native protocol activation intersection", residues=count_residue)
        active = meet(domain, CountDomain(period, count_residue, minimum=0))
        if active is None:
            continue
        for first, lower_domain in choice(
            affine(required.first), affine(missing.first), active, maximum=True
        ):
            # N modulo the epoch period fixes this affine endpoint's residue.
            # The next allowed epoch uses a binary search over static residues,
            # never a traversal of the potentially unbounded parent interval.
            residue = (first[0] * count_residue + first[1]) % period
            index = bisect_left(required.residues, residue)
            following = (
                required.residues[index]
                if index < len(required.residues)
                else required.residues[0] + period
            )
            aligned = add(first, following - residue)
            for last, intersection in choice(
                affine(required.last), affine(missing.last), lower_domain, maximum=False
            ):
                if meet(intersection, le(aligned, last)) is not None:
                    return True
    return False


def _present(leaf, site):
    require(
        set(leaf.cases) == set(PARENT_CASES),
        "native activation needs all four actual parent cases, including empty cases",
    )
    present = set()
    for label in PARENT_CASES:
        case = leaf.cases[label]
        require(
            case.label == label and case.activation == case_activation(label),
            "native activation case metadata disagrees with its actual parent partition",
        )
        if case.loop is None:
            continue
        count = case.loop.constant_count
        require(type(count) is int and count >= 0, "native activation needs a fixed child domain")
        generation = site.position.evaluate(count)
        tasks = [
            task
            for data in case.loop.actors.values()
            for task in data.program.tasks
            if task.uid == site.task[1]
        ]
        require(len(tasks) <= 1, "native activation endpoint has ambiguous actual child tasks")
        # These are verified symbolic native templates. Task.upper=-1 is the
        # symbolic marker, not an ordinary finite Task.active() bound.
        if tasks and 0 <= generation < count and (not tasks[0].last or generation == count - 1):
            present.add(label)
    return present


def _endpoint_activation(protocol, action):
    if protocol.activation is not None:
        activation = validate_activation(protocol.activation, distance=protocol.distance)
        require(
            (protocol.period, protocol.residues) == (activation.period, activation.residues),
            "scope protocol activation residues disagree with its actual declaration",
        )
        return shifted(activation, 0 if action == "set" else protocol.distance)
    require(protocol.distance >= 0, "legacy native protocol distance must be nonnegative")
    if action == "barrier":
        # Legacy barriers already declare DESTINATION residues, unlike local
        # event channels whose residues identify the source publication.
        return Activation(
            first=Position(False, protocol.distance),
            period=protocol.period,
            residues=protocol.residues,
        )
    activation = legacy_activation(protocol.distance, protocol.period, protocol.residues)
    return activation if action == "set" else shifted(activation, protocol.distance)


@bounded
def validate_scope_protocol(program, access, protocol):
    """Every declared endpoint epoch lies inside actual native existence.

    Explicit activation is always in the SOURCE epoch; Wait and barrier sites
    execute at source+d. Unconditional leaves retain the existing fast path.
    The caller remains responsible for validating physical banks, matching
    actual ScopeSyncOps, and proving completion and resource lifetimes.
    """
    require(protocol.kind in {"local", "cross", "barrier"}, "unknown native scope protocol kind")
    placements = (
        [(owner, "barrier") for owner in protocol.destinations]
        if protocol.kind == "barrier"
        else [(owner, "set") for owner in protocol.sources]
        + [(owner, "wait") for owner in protocol.destinations]
    )
    for owner, action in placements:
        require(owner in access.endpoints, "scope protocol has an unknown actual native endpoint")
        uid, site = access.endpoints[owner]
        require(uid in program.leaves, "scope protocol has an unknown actual child")
        leaf = program.leaves[uid]
        if not leaf.cases:
            continue
        require(
            site.task[0] == leaf.actor == owner[0],
            "scope protocol loses its actual native actor identity",
        )
        activation = _endpoint_activation(protocol, action)
        bounded_period(activation.period, "native protocol activation")
        require(
            activation.residues
            and activation.residues == tuple(sorted(set(activation.residues)))
            and all(type(r) is int and 0 <= r < activation.period for r in activation.residues),
            "invalid native protocol activation residues",
        )
        present = _present(leaf, site)
        for label in PARENT_CASES:
            if label in present:
                continue
            require(
                not _intersects(activation, case_activation(label), program.count),
                f"{protocol.name}: {action} activation exceeds actual native activation "
                f"at {uid}/{site.task[1]} in parent case {label}",
            )
