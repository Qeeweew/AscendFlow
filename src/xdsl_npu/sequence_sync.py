"""Local sequence completion: conditional leaf proofs plus real boundary events.

A boundary notification captures the actual producer-pipe retirement prefix.
Its wait gates the real first consumer on its pipe. Each region is dispatched
in lexical order, so forward-only boundary dependencies preserve leaf progress.
No block exit is treated as completion; every event is paired on its actual N
activation and has a distinct bank/ID lifetime in this initial allocator.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import poll
from .completion import Point, intersection
from .ir import LoopSyncOp, RegionChannelOp, RegionSyncOp, TaskOp
from .loop_boundary import stream_operations
from .loop_sync import _plan_leaf, _verify_leaf
from .model import optional_int, require
from .periodic import bounded_period
from .region_completion import (
    RegionCompletion,
    _before,
    _data,
    _point,
    _task,
    _valid_point,
    memory_requirement,
)
from .region_summary import CountDomain, Position, allocation
from .sequence_model import extract_sequence


@dataclass(frozen=True)
class Bridge:
    source: Point
    destination: Point
    domains: tuple[CountDomain, ...]
    event_id: int
    name: str = ""
    carried: bool = False


@dataclass(frozen=True)
class CheckedSequence:
    sequence: object
    analyses: dict
    bridges: tuple[Bridge, ...]


def _domain_row(d):
    return (
        (1, 0, d.exact, 0)
        if d.exact is not None
        else (d.period, d.residue, -1, d.period if d.minimum is None else d.minimum)
    )


def _requirements(sequence):
    result = []
    for obligation in sequence.summary.obligations:
        source = _point(sequence, obligation.source, obligation.domain, source=True)
        destination = _point(sequence, obligation.destination, obligation.domain, source=False)
        require(
            _valid_point(sequence, source, obligation.domain)
            and _valid_point(sequence, destination, obligation.domain),
            "memory frontier points outside its actual region activation",
        )
        result.append((source, destination, obligation.domain, obligation))
    return result


def _plan_boundaries(sequence, analyses, banks):
    requirements = _requirements(sequence)
    queries = tuple(memory_requirement(sequence, *r) for r in requirements)
    proof = RegionCompletion(sequence, analyses, queries, calls=(0,))
    proposals = []
    for source, destination, domain, _ in _requirements(sequence):
        poll("sequence boundary proposals", proposals=len(proposals))
        if proof.leaves.inherited(source, destination, domain):
            continue
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        found = False
        for i, (old_source, old_dest, old_domain) in enumerate(proposals):
            if (
                old_domain != domain
                or old_dest.region != destination.region
                or _task(sequence, old_dest).pipe != dp
            ):
                continue
            if old_source != source and not (sp == dp == _task(sequence, old_source).pipe):
                continue
            if _before(sequence, old_dest, destination, domain):
                found = True
                break
            if _before(sequence, destination, old_dest, domain):
                proposals[i] = (source, destination, domain)
                found = True
                break
        if not found:
            proposals.append((source, destination, domain))
    grouped = {}
    for source, destination, domain in proposals:
        grouped.setdefault((source, destination), set()).add(domain)
    result = []
    for (source, destination), domains in grouped.items():
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        result.append(
            Bridge(
                source,
                destination,
                tuple(sorted(domains, key=_domain_row)),
                -1 if sp == dp else 0,
                f"region_c{len(result)}",
            )
        )
    requirements = _requirements(sequence)
    for candidate in reversed(tuple(result)):
        remaining = [b for b in result if b is not candidate]
        if all(
            proof.leaves.inherited(s, d, domain)
            or proof.covered(memory_requirement(sequence, s, d, domain, obligation), remaining)
            for s, d, domain, obligation in requirements
        ):
            result = remaining
    # Zero iterations are a structural path, not one sample of the steady
    # state. A notification needed only on that path must not gate nonempty
    # work. Split this one boundary symbolically; never enumerate N or widen a
    # buffer. Every removal is checked against all original memory obligations.
    positive = CountDomain(1, 0, minimum=1)
    for candidate in tuple(result):
        pieces = []
        for domain in candidate.domains:
            if domain.exact is None and domain.contains(0):
                pieces.append(CountDomain(1, 0, exact=0))
                rest = intersection(domain, positive)
                if rest is not None:
                    pieces.append(rest)
            else:
                pieces.append(domain)
        if tuple(pieces) == candidate.domains:
            continue
        changed = False
        for piece in reversed(tuple(pieces)):
            trimmed = tuple(p for p in pieces if p != piece)
            replacement = [replace(candidate, domains=trimmed)] if trimmed else []
            remaining = [b for b in result if b is not candidate] + replacement
            if all(
                proof.leaves.inherited(s, d, domain)
                or proof.covered(memory_requirement(sequence, s, d, domain, obligation), remaining)
                for s, d, domain, obligation in requirements
            ):
                pieces = list(trimmed)
                changed = True
        if changed:
            result = [b for b in result if b is not candidate]
            if pieces:
                result.append(replace(candidate, domains=tuple(sorted(pieces, key=_domain_row))))
    actor = next(iter(next(iter(sequence.regions.values())).actors))
    assigned = []
    for bridge in result:
        if bridge.event_id >= 0:
            sp, dp = _task(sequence, bridge.source).pipe, _task(sequence, bridge.destination).pipe
            key = actor, sp, dp
            reserved = int(
                (sp, dp) == ("S", "FIX") and _data(sequence, bridge.source.region).program.config
            )
            event_id = max(banks[key], reserved)
            require(
                event_id < 8,
                f"{key}: sequential event bank exhausted; no whole-region fence fallback",
            )
            banks[key] = event_id + 1
            bridge = replace(bridge, event_id=event_id)
        assigned.append(bridge)
    return assigned, proof.leaves


def _install(sequence, bridges):
    kernel = sequence.module.body.block.first_op
    first = (
        sequence.invocation.op
        if sequence.invocation is not None
        else next(iter(sequence.regions.values())).region
    )
    for bridge in bridges:
        s, d = bridge.source, bridge.destination
        declaration = RegionChannelOp.create(
            attributes={
                "sym_name": StringAttr(bridge.name),
                "source_region": StringAttr(s.region),
                "source_task": StringAttr(s.task),
                "source_end": IntAttr(int(s.position.end)),
                "source_offset": IntAttr(s.position.offset),
                "destination_region": StringAttr(d.region),
                "destination_task": StringAttr(d.task),
                "destination_end": IntAttr(int(d.position.end)),
                "destination_offset": IntAttr(d.position.offset),
                "domains": ArrayAttr(
                    ArrayAttr(IntAttr(n) for n in _domain_row(g)) for g in bridge.domains
                ),
                "event_id": IntAttr(bridge.event_id),
                **({"carried": IntAttr(1)} if bridge.carried else {}),
            }
        )
        kernel.body.block.insert_op_before(declaration, first)
        for action, point in (
            (("set", s), ("wait", d)) if bridge.event_id >= 0 else (("barrier", d),)
        ):
            data = _data(sequence, point.region)
            operation = RegionSyncOp.create(
                operands=[data.stream.body.block.args[0]],
                attributes={"action": StringAttr(action), "channel": StringAttr(bridge.name)},
            )
            task = data.operations[point.task]
            if action == "set":
                task.parent.insert_op_after(operation, task)
            else:
                task.parent.insert_op_before(operation, task)


def _read_bridges(sequence, analyses):
    kernel = sequence.module.body.block.first_op
    bridges, names = [], set()
    usage = defaultdict(set)
    actor = next(iter(next(iter(sequence.regions.values())).actors))
    if _data(sequence, next(iter(sequence.regions))).program.config:
        usage["S", "FIX"].add(0)
    for analysis in analyses.values():
        loop, channel_map, _, _, _ = analysis
        data = next(iter(loop.actors.values()))
        tasks = {t.uid: t for t in data.program.tasks}
        for channel in channel_map.values():
            require(
                channel.kind == "local" and not channel.carry,
                "sequence needs local non-carried leaf event resources",
            )
            key = tasks[channel.sources[0][1]].pipe, tasks[channel.destinations[0][1]].pipe
            require(
                not usage[key].intersection(channel.ids),
                "event IDs overlap across sequence regions",
            )
            usage[key].update(channel.ids)
    for op in kernel.body.block.ops:
        if not isinstance(op, RegionChannelOp):
            continue
        require(op.sym_name.data not in names, "duplicate region channel")
        names.add(op.sym_name.data)
        points = []
        for region, task, end, offset in (
            (op.source_region, op.source_task, op.source_end, op.source_offset),
            (op.destination_region, op.destination_task, op.destination_end, op.destination_offset),
        ):
            require(
                region.data in sequence.regions
                and task.data in _data(sequence, region.data).operations
                and end.data in (0, 1)
                and offset.data >= 0,
                "invalid region endpoint",
            )
            points.append(Point(region.data, task.data, Position(bool(end.data), offset.data)))
        source, destination = points
        carried = optional_int(op.carried, 0)
        require(
            carried in {0, 1} and (not carried or sequence.invocation is not None),
            "carried region channel requires its actual serial parent",
        )
        require(
            carried
            or list(sequence.regions).index(source.region)
            < list(sequence.regions).index(destination.region),
            "region protocols must follow actual lexical dispatch order",
        )
        domains = []
        for row in op.domains:
            require(len(row) == 4, "region count predicate needs four integers")
            p, r, exact, minimum = (v.data for v in row)
            bounded_period(p, "region count predicate")
            require(
                0 <= r < p
                and exact >= -1
                and minimum >= 0
                and (exact == -1 or (p, r, minimum) == (1, 0, 0)),
                "invalid region count predicate",
            )
            domains.append(CountDomain(p, r, exact if exact >= 0 else None, minimum))
        require(domains and len(set(domains)) == len(domains), "empty/duplicate region activation")
        require(
            all(_valid_point(sequence, point, domain) for point in points for domain in domains),
            "region event has no actual producer/consumer for its activation",
        )
        sp, dp = _task(sequence, source).pipe, _task(sequence, destination).pipe
        event_id = op.event_id.data
        if sp == dp:
            require(event_id == -1, "same-pipe region synchronization requires a native fence")
        else:
            require(
                0 <= event_id < 8 and event_id not in usage[sp, dp],
                "region event IDs overlap/exceed their bank",
            )
            usage[sp, dp].add(event_id)
        bridges.append(
            Bridge(source, destination, tuple(domains), event_id, op.sym_name.data, bool(carried))
        )
    actual = {}
    for name, loop in sequence.regions.items():
        data = next(iter(loop.actors.values()))
        previous, pending, parent = None, [], None
        leaf_sync_since_task = False
        for op, _ in stream_operations(data.stream):
            if op.parent is not parent:
                require(not pending, "region wait escapes its physical block")
                previous, parent = None, op.parent
                leaf_sync_since_task = False
            if isinstance(op, TaskOp):
                for wait in pending:
                    key = wait.channel.data, wait.action.data
                    require(key not in actual, "duplicate region sync endpoint")
                    actual[key] = name, op.uid.data
                previous, pending = op.uid.data, []
                leaf_sync_since_task = False
            elif isinstance(op, LoopSyncOp):
                require(
                    not pending,
                    "boundary Wait must follow leaf synchronization before its consumer",
                )
                leaf_sync_since_task = True
            elif isinstance(op, RegionSyncOp):
                require(
                    op.channel.data in names and op.generation is data.stream.body.block.args[0],
                    "region synchronization uses the wrong resource or coordinate",
                )
                require(op.action.data in {"set", "wait", "barrier"}, "invalid region sync action")
                if op.action.data == "set":
                    key = op.channel.data, "set"
                    require(
                        previous is not None
                        and not pending
                        and key not in actual
                        and not leaf_sync_since_task,
                        "region Set must follow its real producer",
                    )
                    actual[key] = name, previous
                else:
                    pending.append(op)
        require(not pending, "trailing region waits are not a drain")
    expected = {}
    for bridge in bridges:
        for action, p in (
            (("set", bridge.source), ("wait", bridge.destination))
            if bridge.event_id >= 0
            else (("barrier", bridge.destination),)
        ):
            expected[bridge.name, action] = p.region, p.task
    require(
        actual == expected, "region channels are unpaired or placed at the wrong native endpoint"
    )
    return tuple(bridges), {f"{actor}:{a}:{b}": sorted(ids) for (a, b), ids in usage.items()}


def verify_sequence(module, *, _details=False):
    sequence = extract_sequence(module)
    analyses, reports = {}, {}
    for name, loop in sequence.regions.items():
        reports[name], analyses[name] = _verify_leaf(
            loop, live_in=sequence.live_in[name], _details=True
        )
    bridges, banks = _read_bridges(sequence, analyses)
    forward = tuple(b for b in bridges if not b.carried)
    requirements = _requirements(sequence)
    queries = tuple(memory_requirement(sequence, *r) for r in requirements)
    proof = RegionCompletion(sequence, analyses, queries, calls=(0,))
    inherited = 0
    for source, destination, domain, obligation in _requirements(sequence):
        poll("sequence boundary coverage", obligations=len(sequence.summary.obligations))
        if proof.leaves.inherited(source, destination, domain):
            inherited += 1
            continue
        require(
            proof.covered(
                memory_requirement(sequence, source, destination, domain, obligation), forward
            ),
            f"missing sequence {obligation.hazard}: {source} -> {destination}, "
            f"{allocation(obligation.storage).sym_name.data}[{obligation.start}:{obligation.stop}]",
        )
    report = {
        "kernel": module.body.block.first_op.sym_name.data,
        "verification": "conditional leaf proofs plus exact sequential byte frontiers and native completion/event boundaries",
        "regions": reports,
        "native_templates": sum(r["native_templates"] for r in reports.values()),
        "sequence_memory_obligations": len(sequence.summary.obligations),
        "sequence_inherited_paths": inherited,
        "sequence_channels": len(bridges),
        "sequence_summary_visits": sequence.summary.visits,
        "completion_node_families": sum(len(a[3].nodes) for a in analyses.values()),
        "completion_arc_families": sum(len(a[3].edges) for a in analyses.values()),
        "proof_domain_states": sum(a[3].domain_states for a in analyses.values()),
        "proof_path_queries": sum(a[3].path_queries for a in analyses.values()),
        "sequence_boundary_coverage_queries": proof.queries,
        "event_ids": banks,
        "runtime_parameters": [
            p.sym_name.data for p in next(iter(sequence.regions.values())).parameters
        ],
        "terminal_drain": "one kernel drain; region exits preserve actual outstanding native work",
    }
    if sequence.invocation is not None:
        from .sequence_calls import verify_calls

        report.update(verify_calls(sequence, analyses, bridges, native=proof.leaves))
        report["proof_domain_states"] = sum(a[3].domain_states for a in analyses.values())
        report["proof_path_queries"] = sum(a[3].path_queries for a in analyses.values())
    checked = CheckedSequence(sequence, analyses, bridges)
    return (report, checked) if _details else report


def autosync_sequence(module, *, _capture=None):
    from .sync import _finish

    result = module.clone()
    require(
        not any(isinstance(op, (RegionChannelOp, RegionSyncOp)) for op in result.walk()),
        "AutoSync requires unsynchronized region boundaries",
    )
    sequence = extract_sequence(result)
    banks = defaultdict(int)
    for name, loop in sequence.regions.items():
        _plan_leaf(loop, live_in=sequence.live_in[name], banks=banks)
    analyses = {
        name: _verify_leaf(loop, live_in=sequence.live_in[name], _details=True)[1]
        for name, loop in sequence.regions.items()
    }
    forward, native = _plan_boundaries(sequence, analyses, banks)
    if sequence.invocation is not None:
        from .sequence_calls import plan_calls

        carried = plan_calls(sequence, analyses, forward, banks, native=native)
    else:
        carried = []
    _install(sequence, [*forward, *carried])
    return _finish(result, _capture=_capture)
