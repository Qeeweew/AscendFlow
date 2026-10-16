"""Prepare real cross-query byte obligations for batched completion queries.

A reader span has no implicit last-reader completion. Its source witness is
the actual retirement prefix at its tail, only after every reader reaches it.
The corresponding target witness must precede every actual reader begin.
This module supplies no synchronization and grants no query verification gate.
"""

from dataclasses import dataclass, replace

from .budget import bounded, poll
from .completion import CountDomain, Position, coalesce, covers
from .model import require
from .region_patterns import HazardPattern
from .region_sequence import Obligation
from .region_summary import Access, ReadSpan
from .scope_domains import Activation, affine, validate_activation
from .scope_guarded_completion import _activation
from .scope_query_completion import QuerySite
from .scope_query_summary import QueryPatternSummary, QuerySummary
from .symbolic_completion import holds, meet

MAX_QUERY_MEMORY_CHECKS = 262_144


@dataclass(frozen=True)
class QueryMemoryCheck:
    source: QuerySite
    target: QuerySite
    domain: CountDomain
    obligations: tuple[Obligation, ...]
    patterns: tuple[HazardPattern, ...] = ()


def _reason(obligation, access):
    if isinstance(obligation, HazardPattern):
        storage, start, stop = obligation.first_range()
    else:
        storage, start, stop = obligation.storage, obligation.start, obligation.stop
    names = [name for name, root in access.origins.items() if root is storage]
    root = "/".join(names) if names else "unknown allocation"
    return f"{obligation.hazard} {root}[{start}:{stop}]"


def memory_check_reason(check, access):
    """Name a real physical hazard without expanding its other byte witnesses."""
    require(
        isinstance(check, QueryMemoryCheck) and (check.obligations or check.patterns),
        "query memory check has no physical hazard witness",
    )
    return _reason(check.patterns[0] if check.patterns else check.obligations[0], access)


def _validate_patterns(patterns, roots, seen):
    for pattern in patterns:
        require(
            isinstance(pattern, HazardPattern) and pattern.coverage,
            "query memory check lost its actual allocation identity",
        )
        for coverage in pattern.coverage:
            require(coverage.ranges, "query memory check has no physical coverage")
            identity = id(coverage.ranges)
            if identity in seen:
                continue
            seen.add(identity)
            for ranges in coverage.ranges:
                require(
                    id(ranges.storage) in roots
                    and roots[id(ranges.storage)] is ranges.storage
                    and ranges.spans
                    and type(ranges.shift) is int,
                    "query memory check lost its actual allocation identity",
                )
                require(
                    all(
                        type(lo) is int
                        and type(hi) is int
                        and 0 <= lo < hi
                        and type(order) is int
                        and order >= 0
                        for lo, hi, order in ranges.pieces()
                    ),
                    "query memory check has invalid physical byte coverage",
                )


def _required(domain, parent):
    if parent.graph.maximum is None:
        return domain
    return meet(domain, CountDomain(1, 0, exact=parent.graph.maximum))


def _singleton(span, domain):
    first, last = affine(span.first), affine(span.last)
    return holds(first, last, domain) and holds(last, first, domain)


def _span_activations(span, domain, period, reason):
    require(
        type(span.step) is int and span.step > 0 and period % span.step == 0,
        f"{reason}: reader span step must divide the parent graph period",
    )
    first = affine(span.first)
    if not first[0]:
        domains = ((domain, first[1] % span.step),)
    elif domain.exact is not None:
        domains = ((domain, (domain.exact + first[1]) % span.step),)
    elif domain.period % span.step == 0:
        domains = ((domain, (domain.residue + first[1]) % span.step),)
    else:
        # An N-relative first position may change residue with N. Retain each
        # original count predicate instead of sampling one representative N.
        domains = tuple(
            (active, (residue + first[1]) % span.step)
            for residue in range(span.step)
            if (active := meet(domain, CountDomain(span.step, residue, minimum=0))) is not None
        )
    result = []
    for active, residue in domains:
        activation = Activation(active, span.first, span.last, span.step, (residue,))
        validate_activation(activation)
        _activation(activation, period)
        result.append(activation)
    return tuple(result)


def _exists(parent, uid, site, position: Position, domain, reason):
    """An anchored solver's inactive case must not invent a real consumer."""
    guarded = parent.guarded()
    point = affine(position)
    guards = []
    commands = parent.completion.commands_for(uid, site)
    for command in commands:
        for node in guarded._endpoint(command, None, source=True):
            guard = guarded._point_guard(node, point)
            if guard is not None:
                guards.append(guard)
    require(
        covers(domain, coalesce(guards)),
        f"{reason}: query memory endpoint does not exist over its count domain "
        f"({uid}, {site.task[1]}, {site.point}, {position})",
    )
    return commands


@bounded
def prepare_memory_checks(checked, parent) -> tuple[QueryMemoryCheck, ...]:
    """Derive exact endpoint checks and prove all span witnesses in one batch.

    This preparation must be repeated from actual IR by a final verifier. It
    rejects an unproved span compression; an absent cross-query path itself is
    left for ``check_memory`` so a planner can add a real protocol.
    """
    summary = checked.query_summary
    require(
        isinstance(summary, (QuerySummary, QueryPatternSummary)),
        "query memory needs an actual query summary",
    )
    require(
        parent.access is checked.access and parent.program is checked.program,
        "query memory preparation changed its actual parent/access identities",
    )
    compact = isinstance(summary, QueryPatternSummary)
    hazards = summary.patterns if compact else summary.obligations
    require(len(hazards) <= MAX_QUERY_MEMORY_CHECKS, "query memory obligation budget exceeded")
    roots = {id(root): root for root in checked.access.origins.values()}
    if compact:
        _validate_patterns(hazards, roots, set())
    guarded = parent.guarded()
    guarded._sync()
    witnesses, endpoints, checks = {}, {}, {}

    def endpoint(value, domain, *, source, obligation):
        key = value, domain, source
        if key in endpoints:
            return endpoints[key]
        reason = _reason(obligation, checked.access)
        require(
            isinstance(value.access, (Access, ReadSpan))
            and value.access.task in checked.access.endpoints,
            f"{reason}: query memory has no actual native endpoint",
        )
        require(
            value.region.isdecimal()
            and str(int(value.region)) == value.region
            and (int(value.region) == 0 if source else 0 < int(value.region) < len(summary.parts)),
            f"{reason}: query memory has an invalid static query identity",
        )
        require(
            value.constant_count == checked.program.count,
            f"{reason}: query memory changed its parent KV clock",
        )
        uid, actual = checked.access.endpoints[value.access.task]
        native = value.access
        if isinstance(native, ReadSpan):
            require(
                type(native.step) is int and native.step > 0,
                f"{reason}: invalid query reader span step",
            )
        if isinstance(native, Access):
            position = native.position
            site = replace(actual, point="end" if source else "begin")
            validate_activation(Activation(domain, position, position))
            _exists(parent, uid, site, position, domain, reason)
        elif _singleton(native, domain):
            position = native.first
            site = replace(actual, point="end" if source else "begin")
            validate_activation(Activation(domain, position, position))
            _exists(parent, uid, site, position, domain, reason)
        else:
            activations = _span_activations(native, domain, parent.period, reason)
            position = native.last if source else native.first
            site = replace(actual, point="prefix" if source else "begin")
            anchors = _exists(parent, uid, site, position, domain, reason)
            family = parent.completion.commands_for(
                uid, replace(actual, point="end" if source else "begin")
            )
            for activation in activations:
                for first in family if source else anchors:
                    for last in anchors if source else family:
                        query = (
                            first,
                            last,
                            None if source else native.first,
                            native.last if source else None,
                            activation,
                        )
                        witnesses.setdefault(query, obligation)
            require(
                len(witnesses) <= MAX_QUERY_MEMORY_CHECKS,
                "query memory span witness budget exceeded",
            )
        result = QuerySite(int(value.region), uid, site, position)
        endpoints[key] = result
        return result

    for index, obligation in enumerate(hazards):
        if index % 128 == 0:
            poll("query memory preparation", obligations=index, witnesses=len(witnesses))
        if not compact:
            require(
                id(obligation.storage) in roots
                and roots[id(obligation.storage)] is obligation.storage,
                "query memory obligation lost its actual allocation identity",
            )
        domain = _required(obligation.domain, parent)
        require(domain is not None, "query memory obligation has no actual parent count")
        source = endpoint(obligation.source, domain, source=True, obligation=obligation)
        target = endpoint(obligation.destination, domain, source=False, obligation=obligation)
        bucket = checks.setdefault((source, target, obligation.domain), ([], []))
        bucket[1 if compact else 0].append(obligation)

    # Register every origin before allowing either forward or reverse flow to
    # propagate. Byte-cell duplicates do not create another witness query.
    answers = parent.anchored().covered_many(tuple(witnesses))
    for (query, obligation), covered in zip(witnesses.items(), answers, strict=True):
        require(
            covered,
            f"{_reason(obligation, checked.access)}: unproved query reader span witness "
            f"({query[2]}, {query[3]}, {query[4]})",
        )
    return tuple(
        QueryMemoryCheck(source, target, domain, tuple(obligations), tuple(patterns))
        for (source, target, domain), (obligations, patterns) in checks.items()
    )


@bounded
def check_memory(checks, query_completion) -> tuple[bool, ...]:
    """Check one prepared batch, retaining missing paths for protocol planning."""
    checks = tuple(checks)
    roots = {id(root): root for root in query_completion.parent.access.origins.values()}
    seen = set()
    for check in checks:
        require(
            isinstance(check, QueryMemoryCheck)
            and (check.obligations or check.patterns)
            and all(
                id(obligation.storage) in roots
                and roots[id(obligation.storage)] is obligation.storage
                for obligation in check.obligations
            ),
            "query memory check lost its actual allocation identity",
        )
        _validate_patterns(check.patterns, roots, seen)
    return tuple(
        query_completion.covered_many(
            (check.source, check.target, check.domain) for check in checks
        )
    )
