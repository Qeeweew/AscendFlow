"""Directed virtual native endpoints on an owned scope boundary graph."""

from .model import require
from .scope_child_transfer import VirtualCut, VirtualNode
from .scope_child_transfer import point_guard as native_point_guard
from .symbolic_completion import Family, le, meet


def revision(graph):
    manager = getattr(graph, "child_transfer", None)
    return 0 if manager is None else manager.transfer_revision


def prepare(graph, endpoints):
    manager = getattr(graph, "child_transfer", None)
    cuts = tuple(endpoint for endpoint in endpoints if isinstance(endpoint, VirtualCut))
    if cuts:
        require(manager is not None, "virtual native cut has no owned scope transfer")
        manager.prepare_virtual(cuts)


def routes(graph, node, *, source, reverse=False, since=None):
    if not isinstance(node, VirtualNode):
        return (node,) if since is None else ()
    manager = getattr(graph, "child_transfer", None)
    require(manager is not None, "virtual native cut has no owned scope transfer")
    inject = source != reverse
    return (manager.inject if inject else manager.extract)(node, since=since)


def family(node, families, *, count_residue=None):
    if not isinstance(node, VirtualNode):
        return families[node]
    value = node.family
    if count_residue is None:
        return value
    require(value.horizon[0] == 1, "anchored reversal needs a parent-count horizon")
    return Family(
        -1 - value.horizon[1],
        (1, -1 - value.lower),
        value.period,
        (count_residue - 1 - value.residue) % value.period,
        None if value.last is None else (1 - value.last[0], -1 - value.last[1]),
        value.monotone,
    )


def condition(node, conditions):
    return node.condition if isinstance(node, VirtualNode) else conditions[node]


def point_guard(node, point, families, conditions, *, count_residue=None):
    if node is None:
        return None
    if isinstance(node, VirtualNode) and count_residue is None:
        return native_point_guard(node, point)
    value = family(node, families, count_residue=count_residue)
    guard = meet(condition(node, conditions), value.residue_guard(point))
    guard = meet(guard, meet(le((0, value.lower), point), le(point, value.horizon)))
    if value.last is not None:
        guard = meet(guard, meet(le(point, value.last), le(value.last, point)))
    return guard


def direct(graph, source, target, *, reverse=False):
    if not isinstance(source, VirtualNode) or not isinstance(target, VirtualNode):
        return False
    manager = getattr(graph, "child_transfer", None)
    require(manager is not None, "virtual native cut has no owned scope transfer")
    return manager.direct(target, source) if reverse else manager.direct(source, target)
