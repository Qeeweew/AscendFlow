"""Compose actual region milestones across adjacent serial invocations.

The two call coordinates below are symbolic p and p+1, never runtime expansion.
Leaf progress and lexical forward channels give a base case; carried channels
only depend on an earlier call. Actual native prefixes discharge memory, and
separate Wait-to-Set obligations discharge finite event resources.
"""

from dataclasses import dataclass, replace

from .completion import CountDomain, Requirement, Site, covers
from .model import require
from .region_completion import RegionCompletion, _point, _task, mark, position
from .region_summary import Access, summarize_local_events
from .sequence_plan import BridgeSelection, Candidate, select_bridges


def memory_requirements(sequence):
    result = []
    for dep in sequence.invocation_dependencies():
        # Invocation GM frames are distinct; these are actual reused local bytes.
        require(not dep.storage.owner.external.data, "call transfer includes unpartitioned GM")
        source = _point(sequence.regions, dep.source, dep.domain, source=True)
        destination = _point(sequence.regions, dep.destination, dep.domain, source=False)
        access = dep.source.access
        point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
        result.append(
            Requirement(
                mark(sequence.regions, source, point),
                mark(sequence.regions, destination, "begin", 1),
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
                position(sequence.regions, name, event.consumed.position),
                "begin",
                event.channel,
                "wait",
            )
            right = Site(
                event.republished.task,
                position(sequence.regions, name, event.republished.position),
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
                    mark(sequence.regions, bridge.destination, "begin"),
                    mark(sequence.regions, bridge.source, "prefix", 0 if bridge.carried else 1),
                    domain,
                    f"event {bridge.name} Wait-before-Set",
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class CallRequirements:
    memory: tuple[Requirement, ...]
    events: tuple[Requirement, ...]
    reserved: tuple[Requirement, ...]


def prepare_call_requirements(sequence, forward):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return None
    memory, events = memory_requirements(sequence), event_requirements(sequence)
    # Reserve source bits and exact native cuts before projecting fixed
    # children. These are future resource QUESTIONS, not completion edges.
    reserved = tuple(
        Requirement(
            replace(request.destination, call=0),
            replace(request.source, point="prefix", call=call),
            request.domain,
            "potential boundary event reuse",
        )
        for requests, call in ((forward, 1), (memory, 0))
        for request in requests
    )
    reserved += tuple(
        replace(
            request,
            source=replace(request.source, call=1),
            destination=replace(request.destination, call=1),
        )
        for request in forward
    )
    return CallRequirements(memory, events, reserved)


@dataclass(frozen=True)
class SequenceCallPlan:
    sequence: object
    selection: BridgeSelection
    memory: tuple[Requirement, ...]
    events: tuple[Requirement, ...]
    coverage: tuple[tuple[CountDomain, ...], ...]

    @property
    def bridges(self):
        return self.selection.bridges

    @property
    def proof(self):
        return self.selection.proof

    def report(self):
        return _report(self.sequence, self.memory, self.events, self.proof)


def _report(sequence, memory, events, proof):
    return {
        "invocation_bound": sequence.invocation.bound.text(),
        "invocation_frames": len(sequence.invocation.frames),
        "invocation_memory_obligations": len(memory),
        "invocation_event_domains": len(events),
        "invocation_proof_queries": proof.queries,
        "invocation_path_visits": 0,
        "invocation_completion_sites": len(proof.flow.edges),
        "invocation_child_projection_statistics": proof.projection_statistics,
        "invocation_completion_facts": len(proof.sources),
        "invocation_completion_transfers": proof.flow.transfers,
        "invocation_completion_edges": proof.flow.edge_count,
    }


def verify_calls(sequence, analyses, bridges):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return {}
    memory = memory_requirements(sequence)
    events = event_requirements(sequence) + boundary_requirements(sequence, bridges)
    proof = RegionCompletion(sequence.regions, analyses, memory + events, bridges=bridges)
    for requirement in memory + events:
        require(
            proof.covered(requirement),
            f"missing sequence invocation {requirement.reason}: {requirement.source} -> {requirement.destination}, {requirement.domain}",
        )
    return _report(sequence, memory, events, proof)


def plan_calls(sequence, analyses, forward, banks, *, proof=None, requirements=None):
    if sequence.invocation is None or sequence.invocation.bound.constant in {0, 1}:
        return None
    if requirements is None:
        questions = tuple(
            Requirement(
                mark(sequence.regions, bridge.source, "prefix"),
                mark(sequence.regions, bridge.destination, "begin"),
                domain,
                "forward boundary native cuts",
            )
            for bridge in forward
            for domain in bridge.domains
        )
        requirements = prepare_call_requirements(sequence, questions)
    memory, events = requirements.memory, requirements.events
    if proof is None:
        proof = RegionCompletion(
            sequence.regions,
            analyses,
            memory + events + requirements.reserved,
            bridges=forward,
        )
    # A late reader release often covers an earlier writer and several input
    # rings. Examine real cross-pipe WARs before manufacturing same-pipe fences.
    ordered = sorted(
        memory,
        key=lambda r: (
            not r.reason.startswith("WAR"),
            _task(sequence.regions, r.source.location).pipe
            == _task(sequence.regions, r.destination.location).pipe,
            -proof.order[r.source.region],
            proof.order[r.destination.region],
            repr(r),
        ),
    )
    candidates = [Candidate(r, r.source.location, r.destination.location) for r in ordered]
    selection = select_bridges(sequence, proof, candidates, banks, carried=True)
    events = events + boundary_requirements(sequence, (*forward, *selection.bridges))
    coverage = tuple(tuple(proof.coverage(request)) for request in events)
    for request, covered in zip(events, coverage, strict=True):
        require(
            covers(request.domain, covered),
            f"missing sequence invocation {request.reason}: {request.source} -> {request.destination}, {request.domain}",
        )
    return SequenceCallPlan(sequence, selection, memory, events, coverage)
