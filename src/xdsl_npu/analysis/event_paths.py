"""Guarded paths through real selected queue/controller interfaces."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from .event_contracts import LeaseStep
from .guard_cover import CoverWork, GuardCover, opposite, prove_guard_cover
from .structured_protocols import BoundPlacement, LogicalNotification
from .structured_schedule import ActionGuard, Comparison

if TYPE_CHECKING:
    from .event_domains import EventDomains

EventPathCandidate = tuple[
    LogicalNotification, tuple[BoundPlacement, ...], tuple[BoundPlacement, ...]
]


@dataclass(frozen=True, slots=True)
class EventPath:
    condition: ActionGuard
    steps: tuple[LeaseStep, ...]


@dataclass(frozen=True, slots=True)
class EventPathCover:
    assumptions: tuple[Comparison, ...]
    paths: tuple[EventPath, ...]
    cover: GuardCover


@dataclass(frozen=True, slots=True)
class _InterfaceEdge:
    target: int
    condition: ActionGuard
    step: LeaseStep


class _EndpointGraph:
    """One query's canonical endpoints and possible real notification effects.

    Candidate storage retains the original 4096 limit. Only nodes with a
    possible path to the goal enter the 256-state graph budget. Reverse
    reachability ignores conjunctions between edges, so an unknown condition
    can retain a node but can never establish a path cover by itself.
    """

    def __init__(self, domains, notification, old, goal, extend, limit, max_states, max_edges):
        self.domains, self.notification, self.goal = domains, notification, goal
        self.extend, self.limit = extend, limit
        self.max_states, self.max_edges = max_states, max_edges
        self.nodes = [(notification, old)]
        self.identities = {(notification, old): 0}
        self.candidates, self.incoming, self.orders, self.effects = [], {}, {}, {}
        self.terminals, self.successors = {}, {}
        self.reachable, self.counted, self.edge_keys = set(), {0}, set()
        self.domains.work.path_states += 1

    def _intern(self, peer, consume):
        key = peer, consume
        if key not in self.identities:
            self.identities[key] = len(self.nodes)
            self.nodes.append(key)
        return self.identities[key]

    def _following(self, node, peer, source):
        current, consume = self.nodes[node]
        if consume.actor is not source.actor or (
            not self.domains._gate(current) and current.destination_engine is not peer.source_engine
        ):
            return ()
        key = node, peer, source
        if key not in self.orders:
            self.domains.work.path_edge_queries += 1
            poll(
                "event path graph",
                states=len(self.counted),
                edges=len(self.edge_keys),
                queries=self.domains.work.path_edge_queries,
            )
            self.orders[key] = tuple(
                self.domains._following_conditions(current, consume, peer, source)
            )
        return self.orders[key]

    def final_conditions(self, node):
        if node not in self.terminals:
            conditions = []
            for clause in self._following(node, self.notification, self.goal):
                condition = self.extend(ActionGuard(), clause.comparisons)
                if condition is not None and condition not in conditions:
                    conditions.append(condition)
            self.terminals[node] = tuple(conditions)
        return self.terminals[node]

    def _effect(self, index):
        """Query each canonical consume -> peer Set boundary only once."""
        if index in self.effects:
            yield from self.effects[index]
            return
        peer, sources, _, activation = self.candidates[index]
        edges = []
        seen = set()
        for node, (current, consume) in enumerate(self.nodes):
            for source in sources:
                for clause in self._following(node, peer, source):
                    condition = self.extend(activation, clause.comparisons)
                    if condition is None:
                        continue
                    step = LeaseStep(consume, source, peer, self.domains._gate(current))
                    edge = node, condition, step
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(edge)
                        # The caller charges graph nodes/edges before another
                        # incoming alternative can be queried or retained.
                        yield edge
        self.effects[index] = tuple(edges)

    def build_to_goal(self, candidates):
        owned = set()
        for count, candidate in enumerate(candidates):
            poll("event path candidates", candidates=count)
            self.limit(count < 4096, "candidate interface count exceeds its explicit budget")
            peer, sources, consumers = candidate
            if peer is self.notification or candidate in owned:
                continue
            owned.add(candidate)
            activation = self.extend(
                ActionGuard(),
                tuple(test for point in sources + consumers for test in point.guard.comparisons),
            )
            if activation is None:
                continue
            index = len(self.candidates)
            self.candidates.append((peer, sources, consumers, activation))
            for consume in consumers:
                node = self._intern(peer, consume)
                self.incoming.setdefault(node, []).append(index)

        pending = deque()

        def reach(node):
            if node in self.reachable:
                return
            self.reachable.add(node)
            pending.append(node)
            if node not in self.counted:
                self.counted.add(node)
                self.domains.work.path_states += 1
                self.limit(
                    len(self.counted) <= self.max_states,
                    "conditional interface states exceed their explicit budget",
                )

        for node in range(len(self.nodes)):
            if self.final_conditions(node):
                reach(node)
        while pending:
            target = pending.popleft()
            for index in self.incoming.get(target, ()):
                for source, condition, step in self._effect(index):
                    edge = _InterfaceEdge(target, condition, step)
                    key = source, edge
                    if key in self.edge_keys:
                        continue
                    self.edge_keys.add(key)
                    self.domains.work.path_edges += 1
                    self.limit(
                        len(self.edge_keys) <= self.max_edges,
                        "conditional interface edges exceed their explicit budget",
                    )
                    self.successors.setdefault(source, []).append(edge)
                    reach(source)


def prove_event_path_cover(
    domains: EventDomains,
    notification: LogicalNotification,
    old: BoundPlacement,
    goal: BoundPlacement,
    assumptions: tuple[Comparison, ...],
    candidates: Iterable[EventPathCandidate],
) -> EventPathCover | None:
    """Cover one consumption/reuse obligation using only existing effects.

    Candidates are canonical endpoint instances owned by EventDomains; their
    ordered activation and coordinate-binding contracts precede this stage.
    This function is not an admission API for external endpoint records.
    One owned finite graph retains the full notification/placement identity.
    Reverse reachability removes only nodes without any possible path to the
    goal. Forward coverage still retains each complete path condition; neither
    graph membership nor an unconditional visited set proves activation.
    """
    max_states, max_edges, max_paths, max_steps = 256, 8192, 128, 128
    domains.work.lease_searches += 1
    visited, paths = {}, []
    guard_states = 0
    work = CoverWork()
    revision, checked_revision = 0, -1

    def limit(condition, message):
        if not condition:
            raise CompileError(
                f"{notification.name}: {message}",
                kind=ErrorKind.ANALYSIS_LIMIT,
                phase="event-path-cover",
            )

    def extend(guard, tests):
        result = list(guard.comparisons)
        for test in tests:
            premises = assumptions + tuple(result)
            if domains.arithmetic.prove(test, premises):
                continue
            if domains.arithmetic.prove(opposite(test), premises):
                return None
            result.append(test)
        return ActionGuard(tuple(dict.fromkeys(result)))

    def record(path):
        nonlocal revision
        limit(
            len(path.steps) <= max_steps,
            "conditional interface path length exceeds its explicit budget",
        )
        tests = frozenset(path.condition.comparisons)
        if any(frozenset(p.condition.comparisons) <= tests for p in paths):
            return
        paths[:] = [p for p in paths if not tests < frozenset(p.condition.comparisons)]
        paths.append(path)
        revision += 1
        limit(len(paths) <= max_paths, "conditional path alternatives exceed their explicit budget")

    def check_cover():
        nodes, queries = work.nodes, work.queries
        try:
            return prove_guard_cover(
                domains.arithmetic, assumptions, tuple(p.condition for p in paths), work=work
            )
        finally:
            domains.work.cover_nodes += work.nodes - nodes
            domains.work.cover_queries += work.queries - queries

    graph = _EndpointGraph(domains, notification, old, goal, extend, limit, max_states, max_edges)

    def admit(node, guard):
        nonlocal guard_states
        tests = frozenset(guard.comparisons)
        prior = visited.setdefault(node, [])
        if any(known <= tests for known in prior):
            return False
        prior[:] = [known for known in prior if not tests < known]
        prior.append(tests)
        guard_states += 1
        domains.work.path_guard_states += 1
        limit(
            guard_states <= max_states,
            "conditional path guard states exceed their explicit budget",
        )
        return True

    def finish(node, guard, steps):
        current, consume = graph.nodes[node]
        complete = False
        for clause in graph.final_conditions(node):
            condition = extend(guard, clause.comparisons)
            if condition is None:
                continue
            final = LeaseStep(consume, goal, None, domains._gate(current))
            record(EventPath(condition, steps + (final,)))
            complete |= condition == guard
        return complete

    def covered():
        nonlocal checked_revision
        if revision != checked_revision:
            cover = check_cover()
            checked_revision = revision
            if cover is not None:
                domains.work.path_covers += 1
                domains.work.witness_steps += sum(len(path.steps) for path in paths)
                return EventPathCover(assumptions, tuple(paths), cover)
        return None

    # A complete direct cover needs no candidate graph at all. A partial
    # direct cover remains an alternative while peer paths cover its remainder.
    admit(0, ActionGuard())
    finish(0, ActionGuard(), ())
    proof = covered()
    if proof is not None:
        return proof
    graph.build_to_goal(candidates)
    if 0 not in graph.reachable:
        return None

    queue = [(0, ActionGuard(), ())]
    while queue:
        pending = []
        for node, guard, steps in queue:
            poll(
                "event path cover",
                states=len(graph.counted),
                edges=len(graph.edge_keys),
                paths=len(paths),
                queries=domains.work.path_edge_queries,
                guard_states=guard_states,
            )
            tests = frozenset(guard.comparisons)
            if tests not in visited[node]:
                continue
            if not finish(node, guard, steps):
                pending.append((node, guard, steps))

        # Reachability is complete once these existing paths cover the actual
        # reuse obligation. Check before constructing another interface layer;
        # unrelated paths cannot strengthen an already universal proof.
        proof = covered()
        if proof is not None:
            return proof
        if not pending:
            return None
        queue = []
        for node, guard, steps in pending:
            limit(
                len(steps) < max_steps,
                "conditional interface path length exceeds its explicit budget",
            )
            for edge in graph.successors.get(node, ()):
                condition = extend(guard, edge.condition.comparisons)
                if condition is not None and admit(edge.target, condition):
                    queue.append((edge.target, condition, steps + (edge.step,)))
    return None
