"""Guarded paths through real selected queue/controller interfaces."""

from __future__ import annotations

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
    Every reached placement retains its complete path condition. Syntactic
    weaker-condition absorption is valid only at an identical actual endpoint;
    an unconditional visited set would lose alternative phase-dependent paths.
    """
    max_states, max_edges, max_paths, max_steps = 256, 8192, 128, 128
    domains.work.lease_searches += 1
    queue = [(notification, old, ActionGuard(), ())]
    visited, paths = {}, []
    states = edges = 0
    owned_candidates = None
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

    while queue:
        pending = []
        for current, consume, guard, steps in queue:
            poll("event path cover", states=states, edges=edges, paths=len(paths))
            key, tests = (current, consume), frozenset(guard.comparisons)
            prior = visited.setdefault(key, [])
            if any(known <= tests for known in prior):
                continue
            prior[:] = [known for known in prior if not tests < known]
            prior.append(tests)
            states += 1
            domains.work.path_states += 1
            limit(states <= max_states, "conditional interface states exceed their explicit budget")
            finishes = False
            for clause in domains._following_conditions(current, consume, notification, goal):
                condition = extend(guard, clause.comparisons)
                if condition is None:
                    continue
                final = LeaseStep(consume, goal, None, domains._gate(current))
                record(EventPath(condition, steps + (final,)))
                finishes |= condition == guard
            if not finishes:
                pending.append((current, consume, guard, steps))

        # Reachability is complete once these existing paths cover the actual
        # reuse obligation. Check before constructing another interface layer;
        # unrelated paths cannot strengthen an already universal proof.
        if revision != checked_revision:
            cover = check_cover()
            checked_revision = revision
            if cover is not None:
                domains.work.path_covers += 1
                domains.work.witness_steps += sum(len(path.steps) for path in paths)
                return EventPathCover(assumptions, tuple(paths), cover)
        if not pending:
            return None
        if owned_candidates is None:
            owned_candidates = []
            for candidate in candidates:
                limit(
                    len(owned_candidates) < 4096,
                    "candidate interface count exceeds its explicit budget",
                )
                owned_candidates.append(candidate)
        queue = []
        for current, consume, guard, steps in pending:
            limit(
                len(steps) < max_steps,
                "conditional interface path length exceeds its explicit budget",
            )
            for peer, sources, consumers in owned_candidates:
                if peer is notification:
                    continue
                if (
                    not domains._gate(current)
                    and current.destination_engine is not peer.source_engine
                ):
                    continue
                for source in sources:
                    if source.actor is not consume.actor:
                        continue
                    edges += 1
                    domains.work.path_edges += 1
                    limit(
                        edges <= max_edges,
                        "conditional interface edges exceed their explicit budget",
                    )
                    activation = tuple(
                        test for p in sources + consumers for test in p.guard.comparisons
                    )
                    condition = extend(guard, activation)
                    if condition is None:
                        continue
                    for clause in domains._following_conditions(current, consume, peer, source):
                        ordered = extend(condition, clause.comparisons)
                        if ordered is None:
                            continue
                        step = LeaseStep(consume, source, peer, domains._gate(current))
                        queue.extend(
                            (peer, destination, ordered, steps + (step,))
                            for destination in consumers
                        )
        # New layers keep distinct guards for the same endpoint. A partial
        # direct path must not hide a longer path needed on the remaining domain.
    return None
