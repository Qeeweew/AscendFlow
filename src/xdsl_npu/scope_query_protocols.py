"""Actual local events across independent serial-query and KV clocks.

These contracts locate and reconstruct instructions. They do not allocate an
event, establish completion, prove binary-event reuse, or open a public gate.
The final proof must consume the actual installed Set/Wait transitions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntAttr, StringAttr

from .budget import bounded, poll
from .completion import CountDomain, Position
from .diagnostics import require
from .dialects.physical import QueryChannelOp, QuerySyncOp, ScopeChannelOp, ScopeSyncOp, TaskOp
from .scalar import INT64_MAX, facts_from_sizes
from .scope_domains import Activation, validate_activation
from .scope_variants import operation_cases, scope_operations

if TYPE_CHECKING:
    from .scope_query_completion import QuerySite


@dataclass(frozen=True)
class QueryProtocol:
    name: str
    source: QuerySite
    target: QuerySite
    domain: CountDomain
    ids: tuple[int, ...] = ()


def _ids(ids, *, actual=False):
    require(
        isinstance(ids, tuple)
        and (bool(ids) or not actual)
        and all(type(i) is int and 0 <= i < 8 for i in ids)
        and len(ids) == len(set(ids)),
        "query local event needs distinct IDs in [0,8); installed events need an ID",
    )


def _actual_domains(program, access):
    """Recheck the live loop domains without cloning any native template."""
    from .scope_invocation import extract_scope_invocation, verify_query_domains
    from .scope_model import _uniforms

    require(
        program.module.body.block.first_op is program.root and access.loop.op is program.root,
        "query protocol has a stale actual program/access identity",
    )
    require(
        program.invocations
        and program.invocations.keys() == program.parents.keys() == program.kernels.keys(),
        "query protocol requires actual serial invocations for every actor",
    )
    facts = facts_from_sizes(program.parameters)
    uniforms = _uniforms(tuple(program.root.body.block.ops), facts, allow_local=True)
    for actor, kernel in program.kernels.items():
        invocation, parent = extract_scope_invocation(kernel, uniforms, facts, program.groups)
        require(
            invocation is not None
            and invocation == program.invocations[actor]
            and parent is program.parents[actor],
            "query protocol has a stale actual query/parent domain",
        )
        require(
            len(parent.body.block.args) == 1
            and (
                parent.limit in uniforms and uniforms[parent.limit] == program.bound
                if parent.limit is not None
                else program.bound is None and parent.count.data == program.count
            ),
            "query protocol has a stale actual KV count",
        )
    verify_query_domains(program.invocations, program.kernels)


def _endpoint(program, access, endpoint, *, source, domain):
    from .scope_native_activation import _intersects
    from .scope_predicates import PARENT_CASES, case_activation, guard_active
    from .scope_query_access import _native_modes
    from .scope_query_completion import QuerySite
    from .symbolic_completion import meet

    require(
        isinstance(endpoint, QuerySite)
        and endpoint.call in program.leaves
        and endpoint.site.point in ({"end", "prefix"} if source else {"begin"}),
        "query protocol needs an actual native retirement source and begin target",
    )
    leaf = program.leaves[endpoint.call]
    site = endpoint.site
    require(
        site.task[0] == leaf.actor
        and site.region == endpoint.call
        and not site.channel
        and not site.action,
        "query protocol changes its actual native actor/call identity",
    )
    owners = [
        owner
        for owner, (uid, actual) in access.endpoints.items()
        if uid == endpoint.call and replace(site, point=actual.point) == actual
    ]
    require(len(owners) == 1, "query protocol has no unique actual child native endpoint")
    parent = program.parents[leaf.actor]
    require(
        leaf.call.parent is parent.body.block
        and leaf.call.generation is parent.body.block.args[0]
        and leaf.stream.parent is leaf.call.body.block
        and leaf.stream.count is not None
        and leaf.stream.count.data == leaf.loop.constant_count,
        "query protocol changes its actual KV/child clock or child count",
    )
    _native_modes(leaf)  # A cached descriptor cannot hide a changed native or view.
    choices = [
        (op, guards)
        for op, guards in scope_operations(leaf.stream)
        if isinstance(op, TaskOp) and op.uid.data == site.task[1]
    ]
    require(len(choices) == 1, "query protocol has no unique actual native instruction")
    native, guards = choices[0]
    task = next(
        task
        for data in leaf.loop.actors.values()
        for task in data.program.tasks
        if task.uid == site.task[1]
    )
    inner = site.position.evaluate(leaf.stream.count.data)
    require(
        0 <= inner < leaf.stream.count.data
        and task.lower <= inner
        and (task.upper < 0 or inner < task.upper)
        and task.phase <= inner % task.period < task.phase + task.phase_width
        and (not task.last or inner == leaf.stream.count.data - 1),
        "query protocol has no active fixed child position",
    )
    validate_activation(Activation(domain, endpoint.parent, endpoint.parent))
    active = domain
    if program.count is not None:
        active = meet(domain, CountDomain(1, 0, exact=program.count))
        require(active is not None, "query protocol is inactive at its actual fixed KV count")
    activation = Activation(active, endpoint.parent, endpoint.parent)
    cases = set(operation_cases(native)) & {
        case for case in PARENT_CASES if all(guard_active(kind, case) for kind in guards)
    }
    for case in PARENT_CASES:
        if case not in cases:
            require(
                not _intersects(activation, case_activation(case), program.count),
                "query protocol endpoint activation exceeds its actual native activation",
            )
    return owners[0], leaf, native, task, inner


@bounded
def validate_query_protocol(program, access, protocol):
    """Check real native existence; candidates may still have no allocated ID."""
    from .scope_query_completion import QuerySite

    require(isinstance(protocol, QueryProtocol), "expected a local QueryProtocol")
    require(
        isinstance(protocol.name, str) and re.fullmatch(r"[A-Za-z_]\w*", protocol.name),
        "invalid query protocol name",
    )
    require(
        isinstance(protocol.source, QuerySite)
        and isinstance(protocol.target, QuerySite)
        and type(protocol.source.query) is int
        and protocol.source.query == 0
        and type(protocol.target.query) is int
        and 0 < protocol.target.query <= INT64_MAX,
        "query protocol needs source query 0 and a positive query distance",
    )
    require(
        isinstance(protocol.domain, CountDomain)
        and all(
            value is None or type(value) is int and 0 <= value <= INT64_MAX
            for value in (
                protocol.domain.period,
                protocol.domain.residue,
                protocol.domain.exact,
                protocol.domain.minimum,
                protocol.domain.maximum,
            )
        ),
        "query protocol needs an int64 KV CountDomain",
    )
    _ids(protocol.ids)
    _actual_domains(program, access)
    first = _endpoint(program, access, protocol.source, source=True, domain=protocol.domain)
    last = _endpoint(program, access, protocol.target, source=False, domain=protocol.domain)
    require(
        first[1].actor == last[1].actor and first[3].pipe != last[3].pipe,
        "query protocols support only local events between distinct pipes of one actor",
    )


def _encode(access, owner):
    uid, site = access.endpoints[owner]
    require(not site.position.end, "query channel needs a fixed actual child position")
    return "/".join((owner[0], uid, site.task[1], str(site.position.offset)))


def _position(value):
    return DictionaryAttr({"end": IntAttr(int(value.end)), "offset": IntAttr(value.offset)})


def _count(value):
    fields = {"period": value.period, "residue": value.residue}
    fields.update(
        (name, number)
        for name in ("exact", "minimum", "maximum")
        if (number := getattr(value, name)) is not None
    )
    return DictionaryAttr({key: IntAttr(number) for key, number in fields.items()})


def _decode_position(attr):
    require(
        isinstance(attr, DictionaryAttr)
        and attr.data.keys() == {"end", "offset"}
        and all(isinstance(v, IntAttr) for v in attr.data.values())
        and attr.data["end"].data in (0, 1),
        "invalid query KV position fields",
    )
    return Position(bool(attr.data["end"].data), attr.data["offset"].data)


def _decode_count(attr):
    require(
        isinstance(attr, DictionaryAttr)
        and {"period", "residue"}
        <= attr.data.keys()
        <= {"period", "residue", "exact", "minimum", "maximum"}
        and all(isinstance(v, IntAttr) for v in attr.data.values()),
        "invalid query KV count fields",
    )
    return CountDomain(
        attr.data["period"].data,
        attr.data["residue"].data,
        **{key: attr.data[key].data for key in ("exact", "minimum", "maximum") if key in attr.data},
    )


@bounded
def install_query_protocols(program, access, protocols):
    """Install beside the native, after same-action ScopeSyncs, inside child syncs."""
    protocols = tuple(protocols)
    if not protocols:
        return
    names = {
        op.sym_name.data
        for op in program.root.body.block.ops
        if isinstance(op, (ScopeChannelOp, QueryChannelOp))
    }
    prepared = []
    for protocol in protocols:
        poll("actual query protocol installation")
        validate_query_protocol(program, access, protocol)
        _ids(protocol.ids, actual=True)
        require(protocol.name not in names, "duplicate query/parent protocol name")
        names.add(protocol.name)
        prepared.append(
            (
                protocol,
                _endpoint(program, access, protocol.source, source=True, domain=protocol.domain),
                _endpoint(program, access, protocol.target, source=False, domain=protocol.domain),
            )
        )
    first = (
        next(iter(program.kernels.values()))
        if len(program.kernels) > 1
        else next(iter(program.invocations.values())).op
    )
    for protocol, source, target in prepared:
        declaration = QueryChannelOp.create(
            attributes={
                "sym_name": StringAttr(protocol.name),
                "source": StringAttr(_encode(access, source[0])),
                "target": StringAttr(_encode(access, target[0])),
                "distance": IntAttr(protocol.target.query),
                "source_parent": _position(protocol.source.parent),
                "target_parent": _position(protocol.target.parent),
                "count": _count(protocol.domain),
                "source_point": StringAttr(protocol.source.site.point),
                "event_ids": ArrayAttr(IntAttr(i) for i in protocol.ids),
            }
        )
        program.root.body.block.insert_op_before(declaration, first)
        for action, (_, leaf, native, _, inner) in (("set", source), ("wait", target)):
            sync = QuerySyncOp.create(
                operands=[
                    program.invocations[leaf.actor].op.body.block.args[0],
                    leaf.call.generation,
                    leaf.stream.body.block.args[0],
                ],
                attributes={
                    "action": StringAttr(action),
                    "channel": StringAttr(protocol.name),
                    "position": IntAttr(inner),
                },
            )
            if action == "set":
                anchor = native
                while (
                    isinstance(anchor.next_op, (ScopeSyncOp, QuerySyncOp))
                    and anchor.next_op.action.data == "set"
                ):
                    anchor = anchor.next_op
                native.parent.insert_op_after(sync, anchor)
            else:
                native.parent.insert_op_before(sync, native)


@bounded
def read_query_protocols(program, access) -> tuple[QueryProtocol, ...]:
    """Reconstruct declarations, clocks, native adjacency and mixed sync order."""
    from .scope_query_completion import QuerySite

    declarations = [op for op in program.module.walk() if isinstance(op, QueryChannelOp)]
    by_encoding = {_encode(access, owner): owner for owner in access.endpoints}
    protocols, expected = {}, {}
    expected_order, actual_order = defaultdict(list), defaultdict(list)
    parent_names = {
        op.sym_name.data for op in program.module.walk() if isinstance(op, ScopeChannelOp)
    }
    for op in declarations:
        poll("actual query protocol extraction")
        require(op.parent_op() is program.root, "query channel must belong to its actual root")
        require(
            op.attributes.keys()
            == {
                "sym_name",
                "source",
                "target",
                "distance",
                "source_parent",
                "target_parent",
                "count",
                "source_point",
                "event_ids",
            },
            "query channels support only the declared local event fields",
        )
        name = op.sym_name.data
        require(
            name not in protocols and name not in parent_names,
            "duplicate query/parent protocol name",
        )
        require(
            op.source.data in by_encoding and op.target.data in by_encoding,
            "query channel has no actual child native endpoint",
        )
        endpoints = []
        for source, encoded, query, parent, point in (
            (True, op.source.data, 0, op.source_parent, op.source_point.data),
            (False, op.target.data, op.distance.data, op.target_parent, "begin"),
        ):
            owner = by_encoding[encoded]
            uid, site = access.endpoints[owner]
            endpoint = QuerySite(query, uid, replace(site, point=point), _decode_position(parent))
            endpoints.append(endpoint)
            action = "set" if source else "wait"
            key = name, owner[0], action
            expected[key] = uid, site.task[1], site.position.offset
            expected_order[uid, site.task[1], source].append((name, action, site.position.offset))
        protocol = QueryProtocol(
            name, *endpoints, _decode_count(op.count), tuple(i.data for i in op.event_ids)
        )
        validate_query_protocol(program, access, protocol)
        _ids(protocol.ids, actual=True)
        protocols[name] = protocol
    actual, observed = {}, set()
    for uid, leaf in program.leaves.items():
        pending, previous, block, query_set = [], None, None, False
        for op, _ in scope_operations(leaf.stream):
            if op.parent is not block:
                require(not pending, "query acquire escaped its actual native region")
                previous, block, query_set = None, op.parent, False
            if isinstance(op, TaskOp):
                query_seen = False
                for sync in pending:
                    if isinstance(sync, ScopeSyncOp):
                        require(not query_seen, "ScopeSync acquire must precede QuerySync acquire")
                        continue
                    query_seen = True
                    key = sync.channel.data, leaf.actor, "wait"
                    require(key not in actual, "duplicate query synchronization")
                    actual[key] = uid, op.uid.data, sync.position.data
                    actual_order[uid, op.uid.data, False].append(
                        (sync.channel.data, "wait", sync.position.data)
                    )
                pending, previous, query_set = [], op, False
            elif isinstance(op, (ScopeSyncOp, QuerySyncOp)):
                action = op.action.data
                if isinstance(op, QuerySyncOp):
                    observed.add(op)
                    require(
                        op.attributes.keys() == {"action", "channel", "position"},
                        "query synchronization supports only local Set/Wait fields",
                    )
                    require(
                        op.channel.data in protocols and action in {"set", "wait"},
                        "query synchronization supports only a declared local Set/Wait",
                    )
                    require(
                        op.query_generation is program.invocations[leaf.actor].op.body.block.args[0]
                        and op.parent_generation is leaf.call.generation
                        and op.child_generation is leaf.stream.body.block.args[0],
                        "query synchronization uses the wrong query/KV/child clock",
                    )
                    require(
                        "scope_cases" not in op.attributes, "query sync cannot hide parent cases"
                    )
                else:
                    require(action in {"set", "wait", "barrier"}, "unknown parent sync action")
                if action == "set":
                    require(
                        previous is not None and not pending,
                        "query/parent Set must follow its actual native",
                    )
                    if isinstance(op, QuerySyncOp):
                        query_set = True
                        key = op.channel.data, leaf.actor, action
                        require(key not in actual, "duplicate query synchronization")
                        actual[key] = uid, previous.uid.data, op.position.data
                        actual_order[uid, previous.uid.data, True].append(
                            (op.channel.data, action, op.position.data)
                        )
                    else:
                        require(not query_set, "ScopeSync Set must precede QuerySync Set")
                else:
                    pending.append(op)
            else:
                require(not pending, "query/parent acquire must be adjacent to its actual native")
                previous, query_set = None, False
        require(not pending, "unattached query/parent acquire")
    require(
        actual == expected,
        "query protocol is missing or changes an actual native Set/Wait placement",
    )
    require(
        actual_order == expected_order, "query synchronization order differs from declaration order"
    )
    require(
        observed == {op for op in program.module.walk() if isinstance(op, QuerySyncOp)},
        "query synchronization escaped its actual child scope",
    )
    return tuple(protocols.values())
