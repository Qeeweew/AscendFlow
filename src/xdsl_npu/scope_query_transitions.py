"""Actual local query Set/Wait transitions at retained native admission cuts.

The static proof window is not the kernel's query count. Its last fragment can
contain an outgoing Set whose matching Wait lies outside this window. Such a
Set is conditional on the real outer bound admitting that consumer; it grants
no incoming completion and no native retirement. Kernel pairing and binary
ID reuse are separate obligations, not consequences of this topology.
"""

from dataclasses import replace
from types import SimpleNamespace

from .completion import coalesce, covers
from .model import require
from .native_order import instruction_points, prefix_inputs
from .scope_domains import affine
from .symbolic_completion import Family, le, meet


def install_query_transitions(query):
    """Import only protocols reconstructed from actual QueryChannel/Sync IR."""
    from .scope_query_protocols import read_query_protocols

    parent = query.parent
    protocols = (
        read_query_protocols(parent.program, parent.access) if parent.program.invocations else ()
    )
    query.query_protocols = {protocol.name: protocol for protocol in protocols}
    query.query_events = {}
    forwarding = {}
    by_site = {
        (uid, replace(site, point="begin")): owner
        for owner, (uid, site) in parent.access.endpoints.items()
    }
    tasks = {
        (actor, task.uid): task
        for actor, data in parent.access.loop.actors.items()
        for task in data.program.tasks
    }

    def site_bindings(endpoint, point):
        return query._bindings(replace(endpoint, site=replace(endpoint.site, point=point)))

    def parent_forwarding(endpoint, owner, action, kind):
        """Existing parent syncs precede query syncs at this actual native."""
        bindings = []
        point = affine(endpoint.parent)
        for residue in range(parent.period):
            key = owner, residue
            prior = (
                parent.sends.get(key)
                if action == "set" and kind == "queue"
                else parent.waits.get(key, {}).get(kind)
                if action == "wait"
                else None
            )
            if prior is None:
                continue
            node, delta = prior
            require(delta == 0, "query sync needs an actual native-local parent insertion cut")
            guard = query.guarded._point_guard(node, point)
            if guard is not None:
                bindings.append((query._node(endpoint.query, node), point, guard))
        return tuple(bindings)

    def link(left, right):
        for source, source_point, source_guard in left:
            for target, target_point, target_guard in right:
                query._edge(
                    source,
                    target,
                    source_point=source_point,
                    target_point=target_point,
                    guard=meet(source_guard, target_guard),
                )

    before_edges = query.topology.edge_count
    for protocol in protocols:
        distance = protocol.target.query
        require(distance < query.copies, "query proof window omits a protocol consumer")
        owners = {
            action: by_site[endpoint.call, replace(endpoint.site, point="begin")]
            for action, endpoint in (("set", protocol.source), ("wait", protocol.target))
        }
        for index in range(query.copies):
            for action, endpoint in (("set", protocol.source), ("wait", protocol.target)):
                if action == "wait" and index < distance:
                    continue
                endpoint = replace(endpoint, query=index)
                owner, position = owners[action], affine(endpoint.parent)
                task = tasks[owner]
                # A genuine native must exist across the whole KV count guard.
                native = site_bindings(endpoint, "prefix" if action == "set" else "begin")
                required = query._required(protocol.domain)
                require(
                    required is not None
                    and covers(required, coalesce(guard for _, _, guard in native)),
                    "query synchronization endpoint is absent on part of its KV count domain",
                )
                before = (
                    {
                        "issue": site_bindings(
                            endpoint, "end" if task.pipe == "S" else "admit_issue"
                        ),
                        "queue": site_bindings(endpoint, "begin"),
                        "retire": native,
                    }
                    if action == "set"
                    else {
                        "issue": site_bindings(endpoint, "issue"),
                        "queue": site_bindings(endpoint, "queue"),
                    }
                )
                key = index, owner, action
                for kind in before:
                    before[kind] += parent_forwarding(endpoint, owner, action, kind)
                    # Different FIRST/LAST expressions can denote the same
                    # instruction at short N. Keep their actual lexical order
                    # exactly on that equality domain, without widening it.
                    for prior_position, binding in forwarding.get((key, kind), ()):
                        same = meet(le(position, prior_position), le(prior_position, position))
                        before[kind] += tuple(
                            (node, point, meet(guard, same))
                            for node, point, guard in binding
                            if meet(guard, same) is not None
                        )

                def node(_point, position=position, domain=protocol.domain):
                    number = query.topology.node(Family(0, (1, -1), last=position))
                    return ((number, position, domain),)

                # Use the same hardware transition definitions as leaf/parent
                # synchronization. Set completion never feeds a native prefix.
                command = SimpleNamespace(kind=action, task=task)
                points = instruction_points(command, before, node, link)
                query.query_events[index, owner[0], protocol.name, action] = points[
                    "end" if action == "set" else "begin"
                ]
                for kind, binding in prefix_inputs(command, points).items():
                    forwarding.setdefault((key, kind), []).append((position, binding))
                    following = "after_queue" if action == "set" else "admit_" + kind
                    link(binding, site_bindings(endpoint, following))

        for index in range(query.copies - distance):
            link(
                query.query_events[index, owners["set"][0], protocol.name, "set"],
                query.query_events[index + distance, owners["wait"][0], protocol.name, "wait"],
            )
    query.query_protocol_edges = query.topology.edge_count - before_edges


def query_event_bindings(query, endpoint):
    protocol = query.query_protocols[endpoint.channel]
    actual = protocol.source if endpoint.action == "set" else protocol.target
    require(
        endpoint.actor == actual.site.task[0] and endpoint.parent == actual.parent,
        "query event changed its actual actor or parent clock position",
    )
    return query.query_events.get(
        (endpoint.query, endpoint.actor, endpoint.channel, endpoint.action), ()
    )
