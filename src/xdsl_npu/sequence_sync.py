"""Local sequence completion: conditional leaf proofs plus real boundary events.

A boundary notification captures the actual producer-pipe retirement prefix.
Its wait gates the real first consumer on its pipe. Each region is dispatched
in lexical order, so forward-only boundary dependencies preserve leaf progress.
No block exit is treated as completion; every event is paired on its actual N
activation and has a distinct bank/ID lifetime in this initial allocator.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from math import lcm

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr

from .budget import poll
from .invocation_sync import Inside, Site, _first
from .ir import LoopSyncOp, RegionChannelOp, RegionSyncOp, TaskOp
from .loop_boundary import stream_operations
from .loop_sync import _plan_leaf, _verify_leaf
from .model import optional_int, require
from .periodic import bounded_period
from .region_summary import Access, CountDomain, Position, allocation
from .sequence_model import extract_sequence


@dataclass(frozen=True)
class Point:
    region: str
    task: str
    position: Position


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


def _data(sequence, name):
    return next(iter(sequence.regions[name].actors.values()))


def _task(sequence, point):
    return next(t for t in _data(sequence, point.region).program.tasks if t.uid == point.task)


def _expression(sequence, point):
    count = sequence.regions[point.region].constant_count
    p = point.position
    if count is not None:
        return 0, p.evaluate(count)
    return (1, -1 - p.offset) if p.end else (0, p.offset)


def _nonnegative(a, b, domain):
    return a * _first(domain) + b >= 0 and (domain.exact is not None or a >= 0)


def _valid_point(sequence, point, domain):
    task = _task(sequence, point)
    a, b = _expression(sequence, point)
    count = sequence.regions[point.region].constant_count
    ca, cb = (1, 0) if count is None else (0, count)
    return (
        _nonnegative(a, b, domain)
        and _nonnegative(ca - a, cb - b - 1, domain)
        and (
            not task.last
            or a == ca
            and b == cb - 1
            or domain.exact is not None
            and a * domain.exact + b == ca * domain.exact + cb - 1
        )
    )


def _before(sequence, left, right, domain):
    """Same-region, same-pipe issue order, for the entire parameter domain."""
    if left.region != right.region:
        names = list(sequence.regions)
        return names.index(left.region) < names.index(right.region)
    tasks = _data(sequence, left.region).program.tasks
    lt, rt = _task(sequence, left), _task(sequence, right)
    a, b = _expression(sequence, left)
    c, d = _expression(sequence, right)
    delta = (c - a) * _first(domain) + d + rt.stage - b - lt.stage
    return (
        delta >= 0
        and (domain.exact is not None or c - a >= 0)
        and (delta > 0 or (lt.issue_order, tasks.index(lt)) <= (rt.issue_order, tasks.index(rt)))
    )


def _point(sequence, endpoint, domain, *, source):
    access = endpoint.access
    p = access.position if isinstance(access, Access) else access.last if source else access.first
    count = endpoint.constant_count
    task = next(
        t for t in _data(sequence, endpoint.region).program.tasks if t.uid == access.task[1]
    )
    if count is not None:
        p = Position(False, p.evaluate(count))
    elif domain.exact is not None and (source or task.last):
        p = Position(True, domain.exact - 1 - p.evaluate(domain.exact))
    return Point(endpoint.region, access.task[1], p)


def _domain_row(d):
    return (
        (1, 0, d.exact, 0)
        if d.exact is not None
        else (d.period, d.residue, -1, d.period if d.minimum is None else d.minimum)
    )


def _covers(required, guards):
    guards = tuple(guards)
    if not guards:
        return False
    if required in guards:
        return True
    if required.exact is not None:
        return any(g.contains(required.exact) for g in guards)
    # Membership is constant per residue between threshold/exact boundaries.
    # Never enumerate up to a large integer boundary such as one million.
    period = required.period
    cuts = {_first(required)}
    for guard in guards:
        if guard.exact is None:
            period = bounded_period(lcm(period, guard.period), "region activation coverage")
            cuts.add(guard.period if guard.minimum is None else guard.minimum)
        else:
            cuts.update((guard.exact, guard.exact + 1))
    cuts = sorted(c for c in cuts if c >= _first(required))
    for i, lower in enumerate(cuts):
        upper = cuts[i + 1] if i + 1 < len(cuts) else None
        for r in range(period):
            n = lower + (r - lower) % period
            if (
                (upper is None or n < upper)
                and required.contains(n)
                and not any(g.contains(n) for g in guards)
            ):
                return False
    return True


class BoundaryProof:
    def __init__(self, sequence, analyses):
        self.sequence, self.analyses = sequence, analyses
        self.inside = {}
        self.cache = {}
        self.site_cache = {}
        self.coverage_queries = 0
        for name, analysis in analyses.items():
            loop, _, commands, graph, _ = analysis
            native = {
                (actor, c.task.uid): c
                for actor, items in commands.items()
                for c in items
                if c.kind == "task"
            }
            endpoints = {
                (actor, c.channel, c.kind): c
                for actor, items in commands.items()
                for c in items
                if c.kind in {"set", "wait"}
            }
            self.inside[name] = Inside(loop, graph, native, endpoints)

    def inherited(self, source, destination, domain):
        key = source, destination, domain
        if key not in self.cache:
            self.cache[key] = self._inherited(source, destination, domain)
        return self.cache[key]

    def local_path(self, left, right, domain, source_point, destination_point):
        if left.region != right.region:
            return False
        if not _valid_point(self.sequence, left, domain) or not _valid_point(
            self.sequence, right, domain
        ):
            return False
        actor = next(iter(self.sequence.regions[left.region].actors))
        return self.site_path(
            left.region,
            Site((actor, left.task), left.position, source_point),
            Site((actor, right.task), right.position, destination_point),
            domain,
        )

    def site_path(self, region, left, right, domain):
        constant = self.sequence.regions[region].constant_count
        # A fixed-count leaf's boundary proof is independent of the parent's
        # dynamic count family; cache before repeatedly normalizing its sites.
        key = region, left, right, domain if constant is None else None
        if key not in self.site_cache:
            self.site_cache[key] = self._site_path(region, left, right, domain)
        return self.site_cache[key]

    def _site_path(self, region, left, right, domain):
        inside = self.inside[region]
        loop, _, commands, _, _ = self.analyses[region]
        positions = [left.position, right.position]
        if loop.constant_count is not None:
            count = loop.constant_count
            positions = [
                Position(True, count - 1 - p.evaluate(count))
                if p.evaluate(count) > count // 2
                else Position(False, p.evaluate(count))
                for p in positions
            ]
            head = max([0] + [p.offset for p in positions if not p.end])
            tail = max([0] + [p.offset for p in positions if p.end])
            window = max(
                [head + tail + 1, inside.graph.period]
                + [c.lower + c.upper + 1 for items in commands.values() for c in items]
                + [
                    p.offset
                    + (inside.command(site).lower if p.end else inside.command(site).upper)
                    + 1
                    for p, site in zip(positions, (left, right))
                ]
            )
            domain = CountDomain(
                inside.graph.period, count % inside.graph.period, minimum=min(count, window)
            )
        return inside.prove(
            replace(left, position=positions[0]),
            replace(right, position=positions[1]),
            domain,
        )

    def covered(self, source, destination, domain, obligation, bridges):
        self.coverage_queries += 1
        poll("sequence boundary paths", queries=self.coverage_queries, candidates=len(bridges))
        require(self.coverage_queries <= 20000, "sequence boundary query budget exceeded")
        seq = self.sequence
        sp, dp = _task(seq, source).pipe, _task(seq, destination).pipe
        access = obligation.source.access
        point = "end" if isinstance(access, Access) or access.first == access.last else "prefix"
        covered = []
        for bridge in bridges:
            if bridge.destination.region != destination.region:
                continue
            bp = _task(seq, bridge.destination).pipe
            target = (
                bp == dp and _before(seq, bridge.destination, destination, domain)
            ) or self.local_path(bridge.destination, destination, domain, "begin", "begin")
            if not target:
                continue
            if bridge.event_id == -1:
                valid = sp == bp  # Fence captures earlier regions on this pipe.
            else:
                valid = (
                    sp == _task(seq, bridge.source).pipe
                    and _before(seq, source, bridge.source, domain)
                ) or self.local_path(source, bridge.source, domain, point, "prefix")
            if valid:
                covered.extend(bridge.domains)
        return _covers(domain, covered)

    def _inherited(self, source, destination, domain, *, capture_only=False):
        seq = self.sequence
        st, dt = _task(seq, source), _task(seq, destination)
        if not capture_only and (st.pipe == "S" or st.pipe == dt.pipe and st.implicit_completion):
            return True
        # Any real Set/barrier in the receiving region on the source pipe also
        # captures preceding regions' native work. Only use an actual completed
        # path from that capture to this first consumer; source order is not one.
        loop, _, commands, _, _ = self.analyses[destination.region]
        inside = self.inside[destination.region]
        local_domain = domain
        target_position = destination.position
        if loop.constant_count is not None:
            # A stronger static boundary family contains this fixed count.
            # Never build an N-step history just to prove an entry gate.
            if dt.last:
                target_position = Position(True, 0)
            window = max(
                [1, inside.graph.period, 0 if target_position.end else target_position.offset + 1]
                + [c.lower + c.upper + 1 for items in commands.values() for c in items]
            )
            local_domain = CountDomain(
                inside.graph.period,
                loop.constant_count % inside.graph.period,
                minimum=min(loop.constant_count, window),
            )
        for actor, items in commands.items():
            for command in items:
                if command.kind not in {"set", "barrier"} or command.task.pipe != st.pipe:
                    continue
                if command.last_gap is not None:
                    positions = [Position(True, command.last_gap)]
                else:
                    positions = [
                        Position(False, command.lower + (r - command.lower) % command.period)
                        for r in command.residues
                    ]
                for position in positions:
                    point = Point(destination.region, command.task.uid, position)
                    if not _valid_point(seq, point, domain):
                        continue
                    ca, cb = _expression(seq, point)
                    n = _first(domain)
                    g = ca * n + cb
                    length = n if loop.constant_count is None else loop.constant_count
                    if not command.lower <= g < length - command.upper:
                        continue
                    if ca and domain.exact is None and domain.period % command.period:
                        continue
                    if g % command.period not in command.residues:
                        continue
                    # For a nonterminal fixed head, lower bound must also ensure
                    # the upper activation gap for every N, not just one sample.
                    if loop.constant_count is None and not _nonnegative(
                        1 - ca, -cb - command.upper - 1, domain
                    ):
                        continue
                    actor_key = actor, destination.task
                    if command.kind == "set":
                        start = Site(
                            (actor, command.task.uid), position, "end", command.channel, "set"
                        )
                        if inside.prove(
                            start, Site(actor_key, target_position, "begin"), local_domain
                        ):
                            return True
                    else:
                        start = Site((actor, command.task.uid), position, "begin", direct=command)
                        if inside.prove(
                            start, Site(actor_key, target_position, "begin"), local_domain
                        ):
                            return True
        return False


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
    proof = BoundaryProof(sequence, analyses)
    proposals = []
    for source, destination, domain, _ in _requirements(sequence):
        poll("sequence boundary proposals", proposals=len(proposals))
        if proof.inherited(source, destination, domain):
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
            proof.inherited(s, d, domain) or proof.covered(s, d, domain, obligation, remaining)
            for s, d, domain, obligation in requirements
        ):
            result = remaining
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
    return assigned


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
    proof = BoundaryProof(sequence, analyses)
    inherited = 0
    for source, destination, domain, obligation in _requirements(sequence):
        poll("sequence boundary coverage", obligations=len(sequence.summary.obligations))
        if proof.inherited(source, destination, domain):
            inherited += 1
            continue
        require(
            proof.covered(source, destination, domain, obligation, forward),
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
        "sequence_boundary_coverage_queries": proof.coverage_queries,
        "event_ids": banks,
        "runtime_parameters": [
            p.sym_name.data for p in next(iter(sequence.regions.values())).parameters
        ],
        "terminal_drain": "one kernel drain; region exits preserve actual outstanding native work",
    }
    if sequence.invocation is not None:
        from .sequence_calls import verify_calls

        report.update(verify_calls(sequence, analyses, bridges))
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
    forward = _plan_boundaries(sequence, analyses, banks)
    if sequence.invocation is not None:
        from .sequence_calls import plan_calls

        carried = plan_calls(sequence, analyses, forward, banks)
    else:
        carried = []
    _install(sequence, [*forward, *carried])
    return _finish(result, _capture=_capture)
