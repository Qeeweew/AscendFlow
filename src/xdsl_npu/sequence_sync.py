"""Local sequence completion: conditional leaf proofs plus real boundary events.

A boundary notification captures the actual producer-pipe retirement prefix.
Its wait gates the real first consumer on its pipe. Each region is dispatched
in lexical order, so forward-only boundary dependencies preserve leaf progress.
No block exit is treated as completion; every event is paired on its actual N
activation and has a distinct bank/ID lifetime in this initial allocator.
"""

from collections import defaultdict
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import poll
from .completion import Point
from .ir import LoopSyncOp, RegionChannelOp, RegionSyncOp, TaskOp
from .loop_boundary import stream_operations
from .loop_sync import _plan_leaf, _verify_leaf
from .model import optional_int, require
from .periodic import bounded_period
from .region_completion import (
    RegionCompletion,
    _data,
    _point,
    _task,
    _valid_point,
    memory_requirement,
)
from .region_summary import CountDomain, Position, allocation
from .sequence_model import extract_sequence
from .sequence_plan import Bridge, Candidate, domain_row, select_bridges


@dataclass(frozen=True)
class CheckedSequence:
    sequence: object
    analyses: dict
    bridges: tuple[Bridge, ...]


def _requirements(sequence):
    result = []
    for obligation in sequence.summary.obligations:
        source = _point(sequence.regions, obligation.source, obligation.domain, source=True)
        destination = _point(
            sequence.regions, obligation.destination, obligation.domain, source=False
        )
        require(
            _valid_point(sequence.regions, source, obligation.domain)
            and _valid_point(sequence.regions, destination, obligation.domain),
            "memory frontier points outside its actual region activation",
        )
        result.append((source, destination, obligation.domain, obligation))
    return result


def _plan_boundaries(sequence, analyses, banks):
    requirements = _requirements(sequence)
    queries = tuple(memory_requirement(sequence, *r) for r in requirements)
    proof = RegionCompletion(sequence.regions, analyses, queries, calls=(0,))
    # Select actual releases before ready edges targeting the same region.
    # Earlier destination regions establish the context for later transfers.
    # Only already selected protocols may justify coverage: no circular
    # reliance on other candidates that might themselves be omitted.
    order = proof.order

    def consumer_issue(point):
        task = _task(sequence.regions, point)
        generation = -1 - point.position.offset if point.position.end else point.position.offset
        tasks = _data(sequence.regions, point.region).program.tasks
        return point.position.end, generation + task.stage, task.issue_order, tasks.index(task)

    def fixed_producer_issue(point):
        # A later capture on the same producer pipe includes its earlier real
        # retirements. Try it first for one consumer so no add-then-delete proof
        # loop is needed. Runtime endpoint ordering retains its existing domain
        # policy; a head and a tail cannot be ranked by textual offset alone.
        count = sequence.regions[point.region].constant_count
        if count is None:
            return (0, 0, 0)
        task = _task(sequence.regions, point)
        tasks = _data(sequence.regions, point.region).program.tasks
        return (
            -(point.position.evaluate(count) + task.stage),
            -task.issue_order,
            -tasks.index(task),
        )

    ordered = sorted(
        requirements,
        key=lambda r: (
            order[r[1].region],
            consumer_issue(r[1]),
            r[3].hazard != "WAR",
            -order[r[0].region],
            _task(sequence.regions, r[0]).pipe == _task(sequence.regions, r[1]).pipe,
            fixed_producer_issue(r[0]),
            repr(r[:3]),
        ),
    )
    candidates = [
        Candidate(
            memory_requirement(sequence, source, destination, domain, obligation),
            source,
            destination,
        )
        for source, destination, domain, obligation in ordered
    ]
    return select_bridges(sequence, proof, candidates, banks)


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
                    ArrayAttr(IntAttr(n) for n in domain_row(g)) for g in bridge.domains
                ),
                "event_id": IntAttr(bridge.event_id),
                **({"carried": IntAttr(1)} if bridge.carried else {}),
            }
        )
        kernel.body.block.insert_op_before(declaration, first)
        for action, point in (
            (("set", s), ("wait", d)) if bridge.event_id >= 0 else (("barrier", d),)
        ):
            data = _data(sequence.regions, point.region)
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
    if _data(sequence.regions, next(iter(sequence.regions))).program.config:
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
                and task.data in _data(sequence.regions, region.data).operations
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
            require(len(row) in (4, 5), "region count predicate needs four or five integers")
            p, r, exact, minimum = (v.data for v in row.data[:4])
            maximum = row.data[4].data if len(row) == 5 else None
            bounded_period(p, "region count predicate")
            require(
                0 <= r < p
                and exact >= -1
                and minimum >= 0
                and (exact == -1 or (p, r, minimum) == (1, 0, 0))
                and (maximum is None or exact == -1 and maximum >= minimum),
                "invalid region count predicate",
            )
            domain = CountDomain(p, r, exact if exact >= 0 else None, minimum, maximum)
            require(
                maximum is None or minimum + (r - minimum) % p <= maximum,
                "empty bounded region predicate",
            )
            domains.append(domain)
        require(domains and len(set(domains)) == len(domains), "empty/duplicate region activation")
        require(
            all(
                _valid_point(sequence.regions, point, domain)
                for point in points
                for domain in domains
            ),
            "region event has no actual producer/consumer for its activation",
        )
        sp, dp = _task(sequence.regions, source).pipe, _task(sequence.regions, destination).pipe
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
    proof = RegionCompletion(sequence.regions, analyses, queries, calls=(0,), bridges=forward)
    inherited = 0
    for source, destination, domain, obligation in requirements:
        poll("sequence boundary coverage", obligations=len(sequence.summary.obligations))
        if proof.inherited(memory_requirement(sequence, source, destination, domain, obligation)):
            inherited += 1
            continue
        require(
            proof.covered(memory_requirement(sequence, source, destination, domain, obligation)),
            f"missing sequence {obligation.hazard}: {source} -> {destination}, "
            f"{allocation(obligation.storage).sym_name.data}[{obligation.start}:{obligation.stop}]",
        )
    local_statistics = [analysis[3].completion_statistics() for analysis in analyses.values()]
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
        "completion_statistics": {
            key: sum(counts[key] for counts in local_statistics) for key in local_statistics[0]
        },
        "sequence_boundary_statistics": proof.flow.statistics(),
        "sequence_child_projection_statistics": proof.projection_statistics,
        "proof_path_queries": sum(a[3].path_queries for a in analyses.values()),
        "sequence_boundary_coverage_queries": proof.queries,
        "sequence_completion_sites": len(proof.flow.edges),
        "sequence_completion_facts": len(proof.sources),
        "sequence_completion_transfers": proof.flow.transfers,
        "sequence_completion_edges": proof.flow.edge_count,
        "sequence_native_relation_checks": 0,
        "event_ids": banks,
        "runtime_parameters": [
            p.sym_name.data for p in next(iter(sequence.regions.values())).parameters
        ],
        "terminal_drain": "one kernel drain; region exits preserve actual outstanding native work",
    }
    if sequence.invocation is not None:
        from .sequence_calls import verify_calls

        report.update(verify_calls(sequence, analyses, bridges))
        local_statistics = [analysis[3].completion_statistics() for analysis in analyses.values()]
        report["completion_statistics"] = {
            key: sum(counts[key] for counts in local_statistics) for key in local_statistics[0]
        }
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
    forward = _plan_boundaries(sequence, analyses, banks)
    if sequence.invocation is not None:
        from .sequence_calls import plan_calls

        carried = plan_calls(sequence, analyses, forward, banks)
    else:
        carried = []
    _install(sequence, [*forward, *carried])
    return _finish(result, _capture=_capture)
