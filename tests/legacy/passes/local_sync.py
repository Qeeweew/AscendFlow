"""Historical M2 local protocol selection retained as a test oracle.

This is the M2 single-clock entry. A dependency scan has already
removed only requirements with a retained read-carry certificate. Selection
uses actual native order, cumulative source captures, and earlier Waits on
the same consumer queue. It never inserts a candidate and then deletes it.
Event leases and global progress are separate contracts, established after
this selection and before materialization.
"""

from collections import defaultdict
from dataclasses import dataclass

from tests.legacy.analysis.protocols import LocalMechanism, LocalProtocol
from xdsl_npu.analysis.frontier import PrefixRank
from xdsl_npu.analysis.hazards import Activation, NativeHazard
from xdsl_npu.analysis.native import NativeTask
from xdsl_npu.diagnostics import ErrorKind, require
from xdsl_npu.periodic import union
from xdsl_npu.target.ascend910b import Engine
from xdsl_npu.target.packets import audited_packet


@dataclass(frozen=True, slots=True)
class Coverage:
    requirement: NativeHazard
    rule: str
    protocol: LocalProtocol | None = None


@dataclass(slots=True)
class SelectionWork:
    requirements: int = 0
    groups: int = 0
    prefix_comparisons: int = 0
    queue_candidates: int = 0
    queue_partitions: int = 0
    queue_updates: int = 0


@dataclass(frozen=True, slots=True)
class LocalSelection:
    tasks: tuple[NativeTask, ...]
    requirements: tuple[NativeHazard, ...]
    protocols: tuple[LocalProtocol, ...]
    coverage: tuple[Coverage, ...]


def _native_order(hazard):
    source, destination = hazard.physical_cuts()
    if source.rank.compare(destination.rank) >= 0:
        return None
    if hazard.producer.engine is Engine.S:
        return "blocking scalar end before controller dispatch"
    if hazard.producer.engine is hazard.consumer.engine and hazard.producer.implicit_completion:
        return "qualified implicit end before later native admission"
    return None


def _contains(first: Activation, second: Activation):
    """First consumer domain contains second, without any clock substitution."""
    return (
        not first.consumer_last
        and not first.source_last
        and not second.consumer_last
        and not second.source_last
        and first.distance <= second.distance
        and second.period % first.period == 0
        and all(r % first.period in first.residues for r in second.residues)
    )


def _source_rank(task, distance):
    return PrefixRank(task.clock, task.stage - distance, task.lexical)


def select_local_protocols(
    tasks: tuple[NativeTask, ...],
    requirements: tuple[NativeHazard, ...],
    *,
    work: SelectionWork | None = None,
) -> LocalSelection:
    """Select missing local relations in fixed dispatch order.

    Endpoint fragments first unite their own residue sets. Only requirements
    of one actual consumer and one equal activation may share the latest
    source-engine retirement capture. This cannot defer an earlier consumer
    or wait for an unrelated later source. Earlier queue acquisitions are
    reused only with the same stage and a proved containing activation.
    Closed actor issue packets retain their internal configuration order.
    External register writes and terminal-only control remain separate duties.
    """
    tasks, requirements = tuple(tasks), tuple(requirements)
    if not tasks:
        require(not requirements, "empty local body has memory requirements")
        return LocalSelection((), (), (), ())
    actor, clock = tasks[0].actor, tasks[0].clock
    require(
        all(task.actor is actor and task.clock is clock for task in tasks),
        "local selection requires one actor and explicitly bound clock",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-sync",
    )
    require(
        not any(task.last for task in tasks),
        "terminal-only natives need a separately bound boundary contract",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-sync",
    )
    require(
        all(
            task.packet is not None
            and task.packet is audited_packet(task.native)
            and task.packet.native == task.native == task.operation.native.data
            and task.packet.closed_mask
            for task in tasks
        ),
        "local native needs an audited closed actor issue packet",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-sync",
    )
    require(
        not any(access.mode.writes for task in tasks for access in task.registers),
        "external native register setup needs its actual dispatch contract",
        kind=ErrorKind.UNSUPPORTED,
        phase="local-sync",
    )
    actual = {task.operation for task in tasks}
    # The owned native decoder already established scalar capture/dispatch
    # lifetime and physical step identity. Selection preserves those natives.
    coverage, fragments = [], {}
    for hazard in requirements:
        if work is not None:
            work.requirements += 1
        require(
            hazard.producer.operation in actual and hazard.consumer.operation in actual,
            "memory requirement does not refer to the owned local body",
        )
        if rule := _native_order(hazard):
            coverage.append(Coverage(hazard, rule))
            continue
        activation = hazard.activation
        require(
            not activation.consumer_last and not activation.source_last,
            "terminal synchronization needs an exact boundary binding",
            kind=ErrorKind.UNSUPPORTED,
            phase="local-sync",
        )
        key = hazard.producer.operation, hazard.consumer.operation, activation.distance
        if key not in fragments:
            fragments[key] = (hazard.producer, hazard.consumer, defaultdict(set), [])
        _, _, families, members = fragments[key]
        families[activation.period].update(activation.residues)
        members.append(hazard)

    groups = {}
    for (_, _, distance), (source, destination, families, members) in fragments.items():
        period, residues = union(families, "one local capture/acquire activation")
        activation = Activation(distance, period, frozenset(residues))
        key = destination.operation, source.engine, activation
        previous = groups.get(key)
        if previous is None:
            groups[key] = (source, destination, activation, list(members))
        else:
            old_source, _, _, old_members = previous
            if work is not None:
                work.prefix_comparisons += 1
            if _source_rank(source, distance).compare(_source_rank(old_source, distance)) > 0:
                old_source = source
            old_members.extend(members)
            groups[key] = old_source, destination, activation, old_members

    # The key universe is fixed by actual consumer activations, not by the
    # number/history of selected notifications. Each key keeps one strongest
    # conditional acquisition and its actual placement witness.
    selected, queues = [], defaultdict(dict)
    for source, destination, activation, _ in groups.values():
        queues[destination.engine, destination.stage, source.engine][activation] = None
    controller_targets = defaultdict(list)
    for (_, stage, engine), entries in queues.items():
        controller_targets[stage, engine].append(entries)
    if work is not None:
        work.queue_partitions = sum(map(len, queues.values()))
    ordered = sorted(
        groups.values(),
        key=lambda item: (item[1].lexical, item[2].distance, item[0].engine.value, item[0].lexical),
    )
    for source, destination, activation, members in ordered:
        if work is not None:
            work.groups += 1
        needed = _source_rank(source, activation.distance)
        queue_key = destination.engine, destination.stage, source.engine
        witness = queues[queue_key][activation]
        if work is not None:
            work.prefix_comparisons += witness is not None
        if (
            witness is not None
            and _source_rank(witness.source, witness.distance).compare(needed) >= 0
        ):
            coverage.extend(
                Coverage(
                    h, "earlier acquisition on the native queue or blocking controller", witness
                )
                for h in members
            )
            continue
        mechanism = (
            LocalMechanism.FENCE if source.engine is destination.engine else LocalMechanism.EVENT
        )
        if mechanism is LocalMechanism.FENCE:
            require(
                needed.compare(destination.rank) < 0,
                f"{destination.operation.uid.data}: same-engine memory source "
                f"{source.operation.uid.data} is not earlier in physical issue order",
                phase="local-sync",
            )
        protocol = LocalProtocol(
            f"local_{len(selected)}", source, destination, activation, mechanism, tuple(members)
        )
        selected.append(protocol)
        destinations = (
            controller_targets[destination.stage, source.engine]
            if destination.engine is Engine.S
            else (queues[queue_key],)
        )
        for entries in destinations:
            for domain, previous in entries.items():
                if work is not None:
                    work.queue_candidates += 1
                if not _contains(activation, domain):
                    continue
                if previous is not None:
                    if work is not None:
                        work.prefix_comparisons += 1
                    if _source_rank(previous.source, previous.distance).compare(needed) >= 0:
                        continue
                entries[domain] = protocol
                if work is not None:
                    work.queue_updates += 1
        coverage.extend(Coverage(h, "actual cumulative source capture", protocol) for h in members)
    return LocalSelection(tasks, requirements, tuple(selected), tuple(coverage))
