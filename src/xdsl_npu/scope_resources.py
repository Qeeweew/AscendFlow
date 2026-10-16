"""Owned invocation resource results consumed by serial-query composition.

The allocator and independent checker establish the same resource contract.
Their results stay inside one compilation; they are neither IR attributes nor
serialized verification flags. Query insertion preserves invocation facts and
adds separate old/new-boundary obligations.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from .budget import bounded
from .diagnostics import require
from .dialects.physical import QueryChannelOp, QuerySyncOp

Bank = tuple[str, str, str]
_AUTHORITY = object()


def _banks(used):
    return MappingProxyType({bank: frozenset(ids) for bank, ids in used.items()})


def invocation_snapshot(program):
    """Linear materialization correspondence at an ownership-transfer boundary.

    Query declarations/instructions are the only permitted additions. Retain
    operands, immutable attributes, block membership and order so a mutable
    capture, native, guard, clock or synchronization cannot reuse old results.
    This does not extract accesses, build a graph, or replay resource queries.
    """
    ignored = (QueryChannelOp, QuerySyncOp)
    return tuple(
        (
            op,
            op.parent,
            tuple(op.attributes.items()),
            tuple(op.properties.items()),
            tuple(op.operands),
            tuple(result.type for result in op.results),
            tuple(
                (
                    block,
                    tuple(argument.type for argument in block.args),
                    tuple(child for child in block.ops if not isinstance(child, ignored)),
                )
                for region in op.regions
                for block in region.blocks
            ),
        )
        for op in program.module.walk()
        if not isinstance(op, ignored)
    )


@dataclass(frozen=True)
class ChildResourceProof:
    parent: object
    pairs: tuple
    answers: tuple[bool, ...]
    lifetime_queries: int
    _authority: object = field(repr=False, compare=False)


def capture_child_resources(parent, answers) -> ChildResourceProof:
    """Capture the allocator's completed batch for its actual selected pairs.

    This is an internal transfer of owned query results, not an import API.
    The independent entry point always calls ``check_invocation_children``.
    """
    pairs, answers = tuple(parent.resource_pairs), tuple(answers)
    require(
        len(answers) == len(pairs) and all(type(answer) is bool and answer for answer in answers),
        "child resource result does not cover its actual selected resource pairs",
    )
    return ChildResourceProof(parent, pairs, answers, len(pairs) * parent.graph.period, _AUTHORITY)


def check_invocation_children(parent) -> ChildResourceProof:
    """Establish actual child reuse once, including guarded FIRST/LAST holes."""
    # Keep the checker's diagnostics and its independently reconstructed pairs.
    parent.check_child_resources()
    return capture_child_resources(parent, (True for _ in parent.resource_pairs))


@dataclass(frozen=True)
class ParentPoolResources:
    parent: object
    protocols: tuple
    used_banks: Mapping[Bank, frozenset[int]]
    statistics: Mapping[str, int]
    _authority: object = field(repr=False, compare=False)


def capture_parent_pools(parent, protocols, used_banks, statistics) -> ParentPoolResources:
    """Internal allocator/checker receipt; callers do not supply proof flags."""
    return ParentPoolResources(
        parent,
        tuple(protocols),
        _banks(used_banks),
        MappingProxyType(dict(statistics)),
        _AUTHORITY,
    )


@dataclass(frozen=True)
class InvocationResourceProof:
    parent: object
    program: object
    access: object
    analyses: object
    protocols: tuple
    children: ChildResourceProof
    pools: ParentPoolResources
    cross_counter_ids: frozenset[int]
    configurations: tuple[Mapping, ...]
    _snapshot: tuple = field(repr=False)
    _authority: object = field(repr=False, compare=False)

    @property
    def child_lifetime_queries(self):
        return self.children.lifetime_queries

    @property
    def pool_statistics(self):
        return self.pools.statistics

    @property
    def used_banks(self):
        return self.pools.used_banks

    def copy_used_banks(self):
        return defaultdict(set, {bank: set(ids) for bank, ids in self.used_banks.items()})

    def check(self, checked, parent):
        _actual_parent(checked, parent)
        require(
            self._authority is _AUTHORITY
            and self.parent is parent
            and self.program is checked.program
            and self.access is checked.access
            and self.analyses is checked.analyses
            and self.protocols == tuple(checked.channels.values())
            and self.children.pairs == tuple(parent.resource_pairs)
            and self._snapshot == invocation_snapshot(checked.program),
            "invocation resource snapshot changed; establish resources from actual IR",
        )
        return self


def _actual_parent(checked, parent):
    from .scope_sync import ParentCompletion

    require(
        isinstance(parent, ParentCompletion)
        and parent.program is checked.program
        and parent.access is checked.access
        and tuple(parent.protocols) == tuple(checked.channels.values()),
        "query binary preparation needs the actual checked parent/protocol identities",
    )


@bounded
def establish_invocation_resources(
    checked, parent, *, children=None, pools=None
) -> InvocationResourceProof:
    """Establish missing facts once, or compose the allocator's owned results.

    Counter backlog is new here in either mode. A normal allocator supplies
    child and pool receipts; imported actual IR supplies neither and executes
    the independent resource checker. No report boolean is accepted.
    """
    from .loop_sync import _lifetime
    from .scope_event_pools import verify_parent_pools
    from .scope_query_resources import _configuration_checks
    from .scope_sync import _leaf_banks

    _actual_parent(checked, parent)
    if children is None:
        children = check_invocation_children(parent)
    require(
        isinstance(children, ChildResourceProof)
        and children._authority is _AUTHORITY
        and children.parent is parent
        and children.pairs == tuple(parent.resource_pairs),
        "invocation resources need the actual owned child resource result",
    )
    if pools is None:
        captured = []
        verify_parent_pools(
            checked.channels.values(),
            parent,
            _leaf_banks(checked.program, checked.analyses),
            _capture=captured.append,
        )
        pools = captured[0]
    require(
        isinstance(pools, ParentPoolResources)
        and pools._authority is _AUTHORITY
        and pools.parent is parent
        and tuple(replace(p, ids=()) if p.kind != "local" else p for p in pools.protocols)
        == tuple(replace(p, ids=()) if p.kind != "local" else p for p in checked.channels.values()),
        "invocation resources need the actual owned parent pool result",
    )
    _, counters, _ = _lifetime(
        checked.access.loop,
        checked.channels,
        parent if parent.node_activations else parent.graph,
        parent.events,
        local=False,
    )
    return InvocationResourceProof(
        parent,
        checked.program,
        checked.access,
        checked.analyses,
        tuple(checked.channels.values()),
        children,
        pools,
        frozenset(counters),
        tuple(MappingProxyType(dict(row)) for row in _configuration_checks(checked)),
        invocation_snapshot(checked.program),
        _AUTHORITY,
    )


def invocation_resources(checked, parent, resources=None) -> InvocationResourceProof:
    """Consume a same-snapshot result, establishing one for standalone calls."""
    if resources is None:
        resources = getattr(checked, "resource_proof", None)
    if resources is None:
        return establish_invocation_resources(checked, parent)
    require(
        isinstance(resources, InvocationResourceProof),
        "query composition needs an invocation resource proof, not a report flag",
    )
    return resources.check(checked, parent)
