"""Binary-event reuse across serial queries, from actual assigned resources.

Parent IDs follow the emitter's source-epoch ordinal, including explicit
activation intervals. A shared pool contributes every old/new member pair,
including self pairs. Child boundaries retain their actual fixed inner sites
and each nonempty parent case block. No completion edge is added here.

These checks cover declared parent/child binary channels. Cross-counter
backlog, configuration recipes, and newly introduced query channels have
separate contracts and are reported separately; this is no verification gate.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import gcd, lcm

from .budget import bounded, poll
from .completion import CountDomain, Position, canonical, coalesce
from .diagnostics import require
from .dialects.physical import FixConfigOp, Load3DConfigOp
from .periodic import bounded_period, normalize
from .region_summary import summarize_local_events
from .scope_domains import affine, legacy_activation, position, validate_activation
from .scope_events import resource_sites
from .scope_predicates import case_activation
from .scope_query_completion import QueryEvent, QuerySite
from .scope_resources import invocation_resources
from .symbolic_completion import add, le, meet

MAX_BINARY_BOUNDARIES = 65_536
MAX_BINARY_CHECKS = 65_536
MAX_BINARY_PAIR_WORK = 1_000_000
POSITIVE = CountDomain(1, 0, minimum=1)


@dataclass(frozen=True)
class QueryBinaryCheck:
    source: QueryEvent | QuerySite
    target: QueryEvent | QuerySite
    domain: CountDomain
    bank: tuple[str, str, str]
    event_id: int
    source_owner: str
    target_owner: str
    kind: str


@dataclass(frozen=True)
class QueryBinaryResources:
    checks: tuple[QueryBinaryCheck, ...]
    parent_resources: int
    child_resources: int
    child_status: str
    cross_counter_status: str
    configuration_status: str
    query_channel_status: str = "not_checked"


@dataclass(frozen=True)
class _Boundary:
    source: QueryEvent | QuerySite
    target: QueryEvent | QuerySite
    domain: CountDomain
    bank: tuple[str, str, str]
    event_id: int
    owner: str
    kind: str


def _required(parent, domain):
    domain = meet(domain, POSITIVE)
    if parent.graph.maximum is not None:
        domain = meet(domain, CountDomain(1, 0, exact=parent.graph.maximum))
    return domain


def _ordinal_boundaries(parent, protocol):
    ids = protocol.ids
    require(
        isinstance(ids, tuple)
        and 1 <= len(ids) <= 8
        and len(set(ids)) == len(ids)
        and all(type(value) is int and 0 <= value < 8 for value in ids),
        f"{protocol.name}: invalid assigned event IDs",
    )
    activation = (
        legacy_activation(protocol.distance, protocol.period, protocol.residues)
        if protocol.activation is None
        else validate_activation(protocol.activation, distance=protocol.distance)
    )
    require(
        (activation.period, activation.residues) == (protocol.period, protocol.residues),
        f"{protocol.name}: binary activation differs from its actual source word",
    )
    first, last = affine(activation.first), affine(activation.last)
    require(
        first == last or first[0] == 0 and last[0] == 1,
        f"{protocol.name}: query binary activation needs a singleton or constant-to-end interval",
    )
    domain = _required(parent, activation.count)
    if domain is None:
        return
    # Normalizing the periodic source word preserves its ordinal from epoch
    # zero. Activation.first does not restart the emitted ID rotation.
    period, residues = normalize(protocol.period, protocol.residues)
    width, ranks = len(ids), len(residues)
    rotation = bounded_period(
        period * width // gcd(ranks, width), f"{protocol.name}: query binary ID rotation"
    )
    words = defaultdict(list)
    for cycle in range(rotation // period):
        for rank, residue in enumerate(residues):
            words[ids[(cycle * ranks + rank) % width]].append(cycle * period + residue)
    for event_id, word in words.items():
        clock, active = normalize(rotation, word)
        bounded_period(
            lcm(clock, canonical(domain).period), f"{protocol.name}: query binary count predicate"
        )
        count_residues = range(clock) if domain.exact is None else (domain.exact % clock,)
        for residue in count_residues:
            poll("query parent binary boundary", channel=protocol.name, event_id=event_id)
            needed = meet(domain, CountDomain(clock, residue, minimum=1))
            if needed is None:
                continue
            head = add(
                first,
                min((r - first[0] * residue - first[1]) % clock for r in active),
            )
            tail = add(
                last,
                -min((last[0] * residue + last[1] - r) % clock for r in active),
            )
            needed = meet(needed, le(head, tail))
            if needed is not None:
                yield event_id, head, tail, needed


def _parent_boundaries(parent, protocol):
    actor = protocol.sources[0][0]
    send = parent.events[actor, protocol.name, "set"]
    wait = parent.events[actor, protocol.name, "wait"]
    bank = actor, send.task.pipe, wait.task.pipe
    for event_id, head, tail, needed in _ordinal_boundaries(parent, protocol):
        yield _Boundary(
            QueryEvent(0, actor, protocol.name, "wait", position(add(tail, protocol.distance))),
            QueryEvent(1, actor, protocol.name, "set", position(head)),
            needed,
            bank,
            event_id,
            protocol.name,
            "parent",
        )


def _child_boundaries(checked, parent):
    for uid, leaf in checked.program.leaves.items():
        variants = (
            checked.analyses.variants(uid)
            if hasattr(checked.analyses, "variants")
            else ((None, checked.analyses[uid]),)
        )
        for label, analysis in variants:
            # A fixed child resource's exact domain is its INNER count, never
            # the outer KV N. Empty cases have no verified child analysis.
            activation = None if label is None else case_activation(label)
            domain = _required(parent, POSITIVE if activation is None else activation.count)
            if domain is None:
                continue
            first = Position(False, 0) if activation is None else activation.first
            last = Position(True, 0) if activation is None else activation.last
            count = analysis[0].constant_count
            require(
                type(count) is int and count >= 0,
                f"{uid}: query child resource needs its actual fixed inner count",
            )
            for resource in summarize_local_events(analysis[0]):
                poll("query child binary boundary", call=uid, channel=resource.channel)
                require(
                    resource.domain.contains(count),
                    f"{uid}/{resource.channel}: child resource escaped its fixed count",
                )
                consumed, published = resource_sites(uid, leaf.actor, count, resource)
                parent.completion.commands_for(uid, consumed)
                parent.completion.commands_for(uid, published)
                yield _Boundary(
                    QuerySite(0, uid, consumed, last),
                    QuerySite(1, uid, published, first),
                    domain,
                    (leaf.actor, *resource.bank[1:]),
                    resource.event_id,
                    f"{uid}/{resource.channel}",
                    "child",
                )


@bounded
def prepare_binary_resources(checked, parent, *, resources=None) -> QueryBinaryResources:
    """Reconstruct binary boundaries after the invocation-local verifier.

    Every active case block's tail/head is kept. Requiring all old/new blocks
    includes the actual last/first occurrence even when middle calls omit the
    channel. This avoids borrowing inactive endpoints as a successful proof.
    The final verifier must reconstruct this input from actual checked IR.
    """
    resources = invocation_resources(checked, parent, resources)
    return _prepare_binary_resources(checked, parent, resources)


def _prepare_binary_resources(checked, parent, resources) -> QueryBinaryResources:
    """Compose boundaries from a resource result checked at this call's entry."""
    groups, boundaries, physical = defaultdict(list), 0, defaultdict(set)

    def register(boundary):
        nonlocal boundaries
        boundaries += 1
        require(boundaries <= MAX_BINARY_BOUNDARIES, "query binary boundary budget exceeded")
        groups[boundary.bank, boundary.event_id].append(boundary)
        physical[boundary.kind].add((boundary.bank, boundary.event_id))

    for protocol in checked.channels.values():
        if protocol.kind == "local":
            for boundary in _parent_boundaries(parent, protocol):
                register(boundary)
    for boundary in _child_boundaries(checked, parent):
        register(boundary)
    pending, work = defaultdict(list), 0
    for (bank, event_id), members in groups.items():
        for source in members:
            for target in members:
                work += 1
                require(work <= MAX_BINARY_PAIR_WORK, "query binary pool pair budget exceeded")
                if work % 128 == 0:
                    poll("query binary pool pairs", work=work, checks=len(pending))
                # The invocation result already proved every actual child
                # borrower. Keep all mixed old/new boundary pairs too.
                domain = meet(source.domain, target.domain)
                if domain is None:
                    continue
                key = (
                    source.source,
                    target.target,
                    bank,
                    event_id,
                    source.owner,
                    target.owner,
                    source.kind if source.kind == target.kind else "mixed",
                )
                pending[key].append(domain)
                require(len(pending) <= MAX_BINARY_CHECKS, "query binary check budget exceeded")
    checks = []
    for (source, target, bank, event_id, old_owner, new_owner, kind), domains in pending.items():
        for domain in coalesce(domains):
            checks.append(
                QueryBinaryCheck(source, target, domain, bank, event_id, old_owner, new_owner, kind)
            )
            require(len(checks) <= MAX_BINARY_CHECKS, "query binary check budget exceeded")
    return QueryBinaryResources(
        tuple(checks),
        len(physical["parent"]),
        len(physical["child"]),
        "complete",
        "not_checked" if any(p.kind == "cross" for p in checked.channels.values()) else "absent",
        "reserved_not_checked" if resources.configurations else "absent",
    )


@bounded
def check_binary_resources(prepared, query_completion) -> tuple[bool, ...]:
    """Check all prepared binary resources in one shared completion pass."""
    require(
        isinstance(prepared, QueryBinaryResources)
        and all(
            isinstance(check, QueryBinaryCheck)
            and check.kind in {"parent", "child", "mixed"}
            and isinstance(check.source, (QueryEvent, QuerySite))
            and isinstance(check.target, (QueryEvent, QuerySite))
            and check.source.query == 0
            and check.target.query == 1
            and type(check.event_id) is int
            and 0 <= check.event_id < 8
            for check in prepared.checks
        ),
        "query binary check needs reconstructed parent/child resource boundaries",
    )
    return tuple(
        query_completion.covered_many(
            (check.source, check.target, check.domain) for check in prepared.checks
        )
    )


def _configuration_checks(checked):
    """Validate the emitter's one-shot kernel-entry handshake resources.

    emit_loop emits FIX and Load3D configuration, including their matching
    Set/Wait, before calling emit_scope_actor. Only direct invariant kernel
    declarations can use this rule; configuration inside a query is rejected.
    """
    program, seen, rows = checked.program, set(), []
    for actor, kernel in program.kernels.items():
        invocation = program.invocations.get(actor)
        execution = program.parents[actor] if invocation is None else invocation.op
        operations = tuple(kernel.body.block.ops)
        require(execution in operations, "configuration has no actual actor execution scope")
        configs = [op for op in operations if isinstance(op, (FixConfigOp, Load3DConfigOp))]
        require(
            len({type(op) for op in configs}) == len(configs),
            f"{actor}: configuration must be unique at kernel entry",
        )
        fix, load3d = False, None
        for op in configs:
            require(
                kernel.core.data == "aic" and operations.index(op) < operations.index(execution),
                f"{actor}: configuration must execute once before the serial query loop",
            )
            seen.add(op)
            if isinstance(op, FixConfigOp):
                require(not op.attributes, f"{actor}: unsupported conditional FIX configuration")
                fix, destination = True, "FIX"
            else:
                require(
                    op.attributes.keys() == {"a_width", "b_width"},
                    f"{actor}: unsupported conditional Load3D configuration",
                )
                load3d = op.a_width.data, op.b_width.data
                require(
                    all(
                        type(width) is int and 16 <= width <= 256 and width % 16 == 0
                        for width in load3d
                    ),
                    f"{actor}: unsupported invariant Load3D configuration",
                )
                destination = "MTE1"
            rows.append(
                {
                    "kind": "configuration",
                    "actor": actor,
                    "bank": [actor, "S", destination],
                    "event_id": 0,
                    "entry_once": True,
                    "before_serial_query": True,
                    "matching_entry_wait": True,
                    "covered": True,
                }
            )
        descriptions = [checked.access.loop.actors[actor].program]
        for uid, leaf in program.leaves.items():
            if leaf.actor != actor:
                continue
            analyses = [checked.analyses[uid]]
            if hasattr(checked.analyses, "variants"):
                analyses.extend(analysis for _, analysis in checked.analyses.variants(uid))
            descriptions.extend(
                data.program for analysis in analyses for data in analysis[0].actors.values()
            )
        require(
            all((data.config, data.load3d) == (fix, load3d) for data in descriptions),
            f"{actor}: configuration descriptor differs across actual query/child cases",
        )
    require(
        seen
        == {op for op in program.module.walk() if isinstance(op, (FixConfigOp, Load3DConfigOp))},
        "configuration inside a query/child needs an explicit resource contract",
    )
    return rows


def _maximum_queries(program):
    invocation = next(iter(program.invocations.values()))
    if invocation.cyclic is not None and invocation.cyclic.total.constant is not None:
        total, groups = invocation.cyclic.total.constant, invocation.cyclic.groups
        return (total + groups - 1) // groups
    return invocation.bound.constant


def _query_bank(checked, protocol):
    tasks = []
    for endpoint in (protocol.source, protocol.target):
        leaf = checked.program.leaves[endpoint.call]
        matches = [
            task
            for data in leaf.loop.actors.values()
            for task in data.program.tasks
            if task.uid == endpoint.site.task[1]
        ]
        require(len(matches) == 1, "query event has no unique actual native bank")
        tasks.append(matches[0])
    return protocol.source.site.task[0], tasks[0].pipe, tasks[1].pipe


@bounded
def verify_query_resources(
    checked, parent, query_completion, *, resources=None, query_plan=None
) -> dict:
    established_here = resources is None and getattr(checked, "resource_proof", None) is None
    resources = invocation_resources(checked, parent, resources)
    if query_plan is not None:
        from .scope_query_planning import QueryMemoryPlan

        require(isinstance(query_plan, QueryMemoryPlan), "query resources need an owned query plan")
        query_plan._check_owned(checked, parent, resources)
    report = _verify_query_resources(checked, parent, query_completion, resources, query_plan)
    if established_here:
        report["invocation_resources"].update(
            source="rebuilt from actual IR by this entry", new_invocation_resource_solver_calls=3
        )
    return report


def _verify_query_resources(checked, parent, query_completion, resources, query_plan=None) -> dict:
    """Rebuild and discharge every supported actual query resource contract.

    Cross counters use a sufficient boundary theorem: each destination's last
    old Wait precedes every source's first new Set. Together with the rebuilt
    invocation-local <=15 proof, this prevents backlog accumulation at query
    boundaries. A missing feedback path is a rejection, never a new drain.
    """
    from .scope_query_completion import QueryCompletion
    from .scope_query_protocols import _actual_domains, read_query_protocols

    require(
        isinstance(query_completion, QueryCompletion) and query_completion.parent is parent,
        "query resource verification needs its actual query completion snapshot",
    )
    _actual_domains(checked.program, checked.access)
    query_completion._check_snapshot()
    configurations = [dict(row) for row in resources.configurations]
    prepared = _prepare_binary_resources(checked, parent, resources)
    protocols = read_query_protocols(checked.program, checked.access)
    used = resources.copy_used_banks()
    maximum = _maximum_queries(checked.program)
    adjacent = maximum is None or maximum >= 2
    requests, rows, reasons, pending_rows = [], [], [], []
    reuse = {} if query_plan is None else {p.protocol.name: p for p in query_plan.reuse_proofs}
    require(
        query_plan is None or len(reuse) == len(query_plan.reuse_proofs),
        "query resource plan has duplicate event rotation results",
    )

    def register(source, target, domain, metadata, reason, established=None):
        require(len(rows) < MAX_BINARY_CHECKS, "query resource check budget exceeded")
        rows.append(
            metadata
            | {"source": asdict(source), "target": asdict(target), "domain": asdict(domain)}
        )
        if established is None:
            requests.append((source, target, domain))
            pending_rows.append(rows[-1])
            reasons.append(reason)
        else:
            require(
                established.request == (source, target, domain),
                "query resource allocation result changed its actual rotation endpoints",
            )
            rows[-1].update(covered=True, established_in="query event allocation")

    if adjacent:
        for check in prepared.checks:
            register(
                check.source,
                check.target,
                check.domain,
                {
                    "kind": check.kind + "_binary",
                    "bank": list(check.bank),
                    "event_id": check.event_id,
                    "source_owner": check.source_owner,
                    "target_owner": check.target_owner,
                },
                f"{check.source_owner}/{check.target_owner}: binary event {check.event_id} re-Set can precede old Wait",
            )
    if adjacent:
        for protocol in checked.channels.values():
            if protocol.kind != "cross":
                continue
            require(
                protocol.activation is None
                and not protocol.last
                and not protocol.carry
                and type(protocol.distance) is int
                and protocol.distance >= 0
                and len(protocol.ids) == 1,
                f"{protocol.name}: cross counter boundary feedback needs an ordinary allocated source word",
            )
            for event_id, head, tail, domain in _ordinal_boundaries(parent, protocol):
                for destination, _ in protocol.destinations:
                    for source_actor, _ in protocol.sources:
                        register(
                            QueryEvent(
                                0,
                                destination,
                                protocol.name,
                                "wait",
                                position(add(tail, protocol.distance)),
                            ),
                            QueryEvent(1, source_actor, protocol.name, "set", position(head)),
                            domain,
                            {
                                "kind": "cross_counter_boundary",
                                "channel": protocol.name,
                                "event_id": event_id,
                                "internal_backlog_limit": 15,
                            },
                            f"{protocol.name}: cross counter boundary feedback is unproved",
                        )
    for protocol in protocols:
        bank, width, distance = (
            _query_bank(checked, protocol),
            len(protocol.ids),
            protocol.target.query,
        )
        require(
            not used[bank].intersection(protocol.ids),
            f"{protocol.name}: query event ID overlaps child/configuration/parent/query resource at {bank}",
        )
        used[bank].update(protocol.ids)
        # Every actual Set at query q has its matching Wait at q+d. A second
        # publication of one ID is possible only when M > width+d.
        if maximum is not None and maximum <= width + distance:
            continue
        established = reuse.get(protocol.name)
        if established is None:
            require(
                query_completion.copies >= max(distance, width) + 1,
                f"{protocol.name}: query resource proof window omits old Wait or re-Set",
            )
        domain = _required(parent, protocol.domain)
        if domain is None:
            continue
        actor = bank[0]
        if established is not None:
            established.check(parent, protocol, protocols)
        register(
            QueryEvent(distance, actor, protocol.name, "wait", protocol.target.parent),
            QueryEvent(width, actor, protocol.name, "set", protocol.source.parent),
            domain,
            {
                "kind": "query_binary",
                "channel": protocol.name,
                "bank": list(bank),
                "ids": list(protocol.ids),
                "event_id": protocol.ids[0],
                "distance": distance,
                "width": width,
                "all_ids_by_query_translation": True,
            },
            f"{protocol.name}: query event re-Set can precede old Wait",
            established,
        )
    answers = query_completion.covered_many(requests)
    for row, answer, reason in zip(pending_rows, answers, reasons, strict=True):
        row["covered"] = answer
        require(answer, f"{reason}; domain {row['domain']}; {row['source']} -> {row['target']}")
    checks = rows + configurations
    return {
        "verified": True,
        "checks": checks,
        "required_checks": len(checks),
        "parent_binary_checks": sum(row["kind"] == "parent_binary" for row in rows),
        "child_binary_checks": sum(row["kind"] == "child_binary" for row in rows),
        "mixed_binary_checks": sum(row["kind"] == "mixed_binary" for row in rows),
        "cross_counter_boundary_checks": sum(
            row["kind"] == "cross_counter_boundary" for row in rows
        ),
        "query_event_reuse_checks": sum(row["kind"] == "query_binary" for row in rows),
        "configuration_checks": len(configurations),
        "new_completion_checks": len(requests),
        "query_event_reuse_checks_reused": sum("established_in" in row for row in rows),
        "parent_resources": prepared.parent_resources,
        "child_resources": prepared.child_resources,
        "cross_counter_ids": sorted(resources.cross_counter_ids),
        "invocation_resources": {
            "source": "composed owned invocation result",
            "child_lifetime_queries_reused": resources.child_lifetime_queries,
            "pool_statistics_reused": dict(resources.pool_statistics),
            "counter_backlog_reused": True,
            "new_invocation_resource_solver_calls": 0,
        },
        "internal_counter_backlog_limit": 15,
        "counter_boundary_rule": "last old Wait before first new Set; invocation-local backlog <=15",
        "maximum_static_queries": maximum,
    }
