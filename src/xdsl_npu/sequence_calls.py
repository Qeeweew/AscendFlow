"""Compose actual region milestones across adjacent serial invocations.

The two call coordinates below are symbolic p and p+1, never runtime expansion.
Leaf progress and lexical forward channels give a base case; carried channels
only depend on an earlier call. Actual native prefixes discharge memory, and
separate Wait-to-Set obligations discharge finite event resources.
"""

from dataclasses import replace

from .completion import CountDomain, Requirement, Site
from .model import require
from .region_completion import RegionCompletion, _data, _point, _task, mark, position
from .region_summary import Access, summarize_local_events
from .sequence_sync import Bridge, _domain_row


def memory_requirements(sequence):
    result = []
    for dep in sequence.invocation_dependencies():
        # Invocation GM frames are distinct; these are actual reused local bytes.
        require(not dep.storage.owner.external.data, "call transfer includes unpartitioned GM")
        source = _point(sequence, dep.source, dep.domain, source=True)
        destination = _point(sequence, dep.destination, dep.domain, source=False)
        access = dep.source.access
        point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
        result.append(
            Requirement(
                mark(sequence, source, point),
                mark(sequence, destination, "begin", 1),
                dep.domain,
                f"{dep.hazard} {dep.storage.owner.sym_name.data}[{dep.start}:{dep.stop}]",
            )
        )
    return tuple(result)


def event_requirements(sequence):
    result = []
    for name, loop in sequence.regions.items():
        for event in summarize_local_events(loop):
            # A fixed-count leaf is active independently of the common dynamic N.
            domain = event.domain if loop.constant_count is None else CountDomain(1, 0, minimum=0)
            left = Site(
                event.consumed.task,
                position(sequence, name, event.consumed.position),
                "begin",
                event.channel,
                "wait",
            )
            right = Site(
                event.republished.task,
                position(sequence, name, event.republished.position),
                "end",
                event.channel,
                "set",
            )
            result.append(
                Requirement(
                    replace(left, region=name),
                    replace(right, region=name, call=1),
                    domain,
                    f"event {name}/{event.channel}:{event.event_id} Wait-before-Set",
                )
            )
    return tuple(result)


def boundary_requirements(sequence, bridges):
    result = []
    for bridge in bridges:
        if bridge.event_id < 0:
            continue
        if bridge.carried and sequence.invocation.bound.constant == 2:
            continue  # One send/one receive; no second publication of this ID.
        for domain in bridge.domains:
            # Consumer begin follows the real boundary Wait; source prefix is
            # captured by the real Set. These are sufficient, not equivalent,
            # resource endpoints. They never retire a buffer on Wait alone.
            result.append(
                Requirement(
                    mark(sequence, bridge.destination, "begin"),
                    mark(sequence, bridge.source, "prefix", 0 if bridge.carried else 1),
                    domain,
                    f"event {bridge.name} Wait-before-Set",
                )
            )
    return tuple(result)


def verify_calls(sequence, analyses, bridges, *, native=None):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return {}
    memory = memory_requirements(sequence)
    events = event_requirements(sequence) + boundary_requirements(sequence, bridges)
    proof = RegionCompletion(sequence, analyses, memory + events, native=native)
    for requirement in memory + events:
        require(
            proof.covered(requirement, bridges),
            f"missing sequence invocation {requirement.reason}: {requirement.source} -> {requirement.destination}, {requirement.domain}",
        )
    return {
        "invocation_bound": sequence.invocation.bound.text(),
        "invocation_frames": len(sequence.invocation.frames),
        "invocation_memory_obligations": len(memory),
        "invocation_event_domains": len(events),
        "invocation_proof_queries": proof.queries,
        "invocation_path_visits": proof.visits,
    }


def plan_calls(sequence, analyses, forward, banks, *, native=None):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return []
    memory, events = memory_requirements(sequence), event_requirements(sequence)
    proof = RegionCompletion(sequence, analyses, memory + events, native=native)
    proposals = []
    # A late reader release often covers an earlier writer and several input
    # rings. Examine real cross-pipe WARs before manufacturing same-pipe fences.
    ordered = sorted(
        memory,
        key=lambda r: (
            not r.reason.startswith("WAR"),
            _task(sequence, r.source.location).pipe == _task(sequence, r.destination.location).pipe,
            -proof.order[r.source.region],
            proof.order[r.destination.region],
            repr(r),
        ),
    )
    for requirement in ordered:
        if proof.covered(requirement, [*forward, *proposals]):
            continue
        source, destination = requirement.source.location, requirement.destination.location
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        proposals.append(
            Bridge(
                source,
                destination,
                (requirement.domain,),
                -1 if sp == dp else 0,
                f"call_c{len(proposals)}",
                True,
            )
        )
    grouped = {}
    for bridge in proposals:
        key = bridge.source, bridge.destination, bridge.event_id
        grouped.setdefault(key, set()).update(bridge.domains)
    proposals = [
        Bridge(s, d, tuple(sorted(domains, key=_domain_row)), event_id, f"call_c{i}", True)
        for i, ((s, d, event_id), domains) in enumerate(grouped.items())
    ]
    for candidate in reversed(tuple(proposals)):
        remaining = [b for b in proposals if b is not candidate]
        if all(proof.covered(r, [*forward, *remaining]) for r in memory + events):
            proposals = remaining
    actor = next(iter(next(iter(sequence.regions.values())).actors))
    assigned = []
    for bridge in proposals:
        if bridge.event_id >= 0:
            sp, dp = _task(sequence, bridge.source).pipe, _task(sequence, bridge.destination).pipe
            key = actor, sp, dp
            reserved = int(
                (sp, dp) == ("S", "FIX") and _data(sequence, bridge.source.region).program.config
            )
            event_id = max(banks[key], reserved)
            require(event_id < 8, f"{key}: invocation event bank exhausted; no drain fallback")
            banks[key] = event_id + 1
            bridge = replace(bridge, event_id=event_id)
        assigned.append(bridge)
    return assigned
