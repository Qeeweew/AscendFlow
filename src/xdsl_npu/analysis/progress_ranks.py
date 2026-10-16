"""Sufficient local rank and cycle-exclusion proofs for progress regions.

The caller establishes endpoints and relations. One invocation shares its
activation owner with this rank evaluator; numeric rank attempts,
uniform SCC branches and finite exclusions do not remap endpoint identity.
"""

from __future__ import annotations

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from .guard_cover import opposite
from .progress_activations import ProgressActivations
from .progress_regions import ExcludedCycle, GuardedRank, port_label
from .structured_schedule import ActionGuard


def _guard(tests):
    return ActionGuard(tuple(dict.fromkeys(tests)))


class ProgressRanks:
    MAX_CONDITIONAL_RANKS = 64

    def __init__(self, domains, activations: ProgressActivations):
        self.domains = domains
        self.activations = activations
        self.control = domains.control
        self.root = domains.memory.module.body.block.first_op

    def fail(self, region, message, witness=(), *, kind=ErrorKind.UNSUPPORTED):
        raise CompileError(
            message,
            kind=kind,
            phase="structured-event-progress",
            location=str(region.location),
            witness=witness,
        )

    @staticmethod
    def _rank_attempt(count, arcs, stride):
        offsets, predecessor = [0] * count, [None] * count
        by_index = dict(arcs)
        for _ in range(count):
            poll("progress rank relaxation", nodes=count, relations=len(by_index), iteration=_)
            changed = None
            for index, arc in arcs:
                required = offsets[arc.source] + 1 - stride * arc.steps
                if required > offsets[arc.destination]:
                    offsets[arc.destination] = required
                    predecessor[arc.destination] = index
                    changed = arc.destination
            if changed is None:
                return tuple(offsets), ()
        if not count:
            return (), ()
        cursor = changed
        for _ in range(count):
            cursor = by_index[predecessor[cursor]].source
        first, cycle = cursor, []
        while not cycle or cursor != first:
            cycle.append(predecessor[cursor])
            arc = by_index[predecessor[cursor]]
            cursor = arc.source
        return None, tuple(cycle)

    @staticmethod
    def _components(count, arcs):
        adjacency = [[] for _ in range(count)]
        for _, arc in arcs:
            adjacency[arc.source].append(arc.destination)
        counter, stack, onstack, ids, low, groups = 0, [], set(), {}, {}, []

        def visit(node):
            nonlocal counter
            poll("progress rank SCC", nodes=count, visited=counter)
            ids[node] = low[node] = counter
            counter += 1
            stack.append(node)
            onstack.add(node)
            for destination in adjacency[node]:
                if destination not in ids:
                    visit(destination)
                    low[node] = min(low[node], low[destination])
                elif destination in onstack:
                    low[node] = min(low[node], ids[destination])
            if low[node] == ids[node]:
                group = []
                while True:
                    member = stack.pop()
                    onstack.remove(member)
                    group.append(member)
                    if member == node:
                        break
                groups.append(tuple(group))

        for node in range(count):
            if node not in ids:
                visit(node)
        return tuple(groups)

    def _conditional_proof(self, region, ports, arcs, *, stream):
        """Split uniform predicates only inside a structurally cyclic SCC.

        Each SCC independently covers the complete parameter domain. There is
        no Cartesian product across notifications or SCCs, and a stream never
        splits on a predicate that can change along its iterations.
        """
        count, stride = len(ports), len(ports) + 1
        indexed = tuple(enumerate(arcs))
        rank, cycle = self._rank_attempt(count, indexed, stride)
        if not cycle:
            return stride, rank, (), ()
        scope = self.activations.region(region)
        uniform = scope.map_guards(tuple(arc.activation for arc in arcs), self._uniform_guard)
        certificates, excluded = [], []

        def internal(group, candidates):
            nodes = set(group)
            return tuple(
                (index, arc)
                for index, arc in candidates
                if arc.source in nodes and arc.destination in nodes
            )

        def cyclic(group, candidates):
            return len(group) > 1 or any(arc.source == arc.destination for _, arc in candidates)

        def finite_exclusions(condition, candidates):
            result = []
            for group in self._components(count, candidates):
                edges = internal(group, candidates)
                if not cyclic(group, edges):
                    continue
                if any(
                    sum(arc.source == node for _, arc in edges) != 1
                    or sum(arc.destination == node for _, arc in edges) != 1
                    for node in group
                ):
                    return None
                activation = scope.guard(condition)
                for _, arc in edges:
                    activation = scope.all_of(activation, arc.activation)
                if not scope.refuted(activation):
                    return None
                result.append(ExcludedCycle(tuple(index for index, _ in edges), activation))
            return result

        for group in self._components(count, indexed):
            edges = internal(group, indexed)
            if not cyclic(group, edges):
                continue
            work = 0

            def establish(condition, group=group, edges=edges):
                nonlocal work
                poll(
                    "conditional progress rank",
                    nodes=len(group),
                    relations=len(edges),
                    branches=work,
                )
                scope.charge("conditional progress rank branch")
                work += 1
                self.domains.work.progress_case_branches += 1
                if work > self.MAX_CONDITIONAL_RANKS:
                    self.fail(
                        region,
                        "uniform-case SCC progress exceeds its explicit branch budget",
                        kind=ErrorKind.ANALYSIS_LIMIT,
                    )
                if scope.refuted(scope.true, condition):
                    return
                active = tuple(
                    (index, arc)
                    for index, arc in edges
                    if not scope.refuted(uniform[index], condition)
                )
                local_rank, local_cycle = self._rank_attempt(count, active, stride)
                if not local_cycle:
                    certificates.append(
                        GuardedRank(
                            condition,
                            tuple(group),
                            tuple(index for index, _ in active),
                            stride,
                            tuple(local_rank[node] for node in group),
                        )
                    )
                    return
                if not stream and (impossible := finite_exclusions(condition, active)) is not None:
                    excluded.extend(impossible)
                    return
                # Follow each witness's existing predicate order. A predicate
                # is split only after all its earlier uniform prerequisites
                # have already been established on this proof branch.
                split = None
                for index in local_cycle:
                    split = scope.split(uniform[index], condition)
                    if split is not None:
                        break
                if split is not None:
                    establish(_guard(condition.comparisons + (split,)))
                    establish(_guard(condition.comparisons + (opposite(split),)))
                    return
                if stream:
                    message = "selected structured notifications have no increasing blocking-interface rank"
                else:
                    active_groups = self._components(count, active)
                    branched = any(
                        cyclic(part, (local := internal(part, active)))
                        and any(
                            sum(arc.source == node for _, arc in local) != 1
                            or sum(arc.destination == node for _, arc in local) != 1
                            for node in part
                        )
                        for part in active_groups
                    )
                    message = (
                        "conditional region SCC needs an exact joint interface projection"
                        if branched
                        else "selected finite region has a possible blocking cycle under its shared activation condition"
                        if all(arcs[index].exact for index in local_cycle)
                        else "child blocking relation cannot be projected exactly for a possible parent cycle"
                    )
                self.fail(
                    region,
                    message,
                    tuple(
                        f"{port_label(ports[arcs[index].source])} -> {port_label(ports[arcs[index].destination])}, "
                        f"clock step {arcs[index].steps}: {arcs[index].reason}"
                        for index in local_cycle
                    ),
                )

            establish(ActionGuard())
        return stride, (), tuple(certificates), tuple(excluded)

    def stream(self, region, ports, arcs):
        stride, rank, conditional, _ = self._conditional_proof(region, ports, arcs, stream=True)
        return stride, rank, conditional

    def finite(self, region, ports, arcs):
        _, _, conditional, excluded = self._conditional_proof(region, ports, arcs, stream=False)
        return excluded, conditional

    def _uniform_guard(self, guard):
        return _guard(
            tuple(
                test
                for test in guard.comparisons
                if self.control.invariant_over(test.lhs, self.root)
                and self.control.invariant_over(test.rhs, self.root)
            )
        )
