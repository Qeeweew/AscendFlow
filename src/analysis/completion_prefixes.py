"""Construct retained synchronization by admitting actual support frontiers."""

from dataclasses import dataclass, replace

from ..diagnostics import require
from .protocol_transfers import ProtocolTransfers
from .structured_protocols import NotificationKind


@dataclass(frozen=True, slots=True)
class CompletionObligation:
    bindings: tuple
    source: object
    consumer: object
    case: object
    proof: object
    answer: object


@dataclass(frozen=True, slots=True)
class CompletionCoverage:
    """Original demand endpoints and their retained-action derivations.

    Coverage belongs to each RequirementBinding, independently of the
    candidate notification that happened to group these demands. Only the
    actual supporting actions remain in the proof.
    """

    dispatch: object
    cases: tuple
    obligations: tuple[CompletionObligation, ...]

    def covers(self, binding):
        return all(
            any(
                item.case == case and any(original is binding for original in item.bindings)
                for item in self.obligations
            )
            for case in self.cases
        )

    def supports(self):
        seen, protocols, pending = set(), set(), [item.proof for item in self.obligations]
        while pending:
            proof = pending.pop()
            if proof in seen:
                continue
            seen.add(proof)
            if proof.node.notification is not None and proof.node.notification not in protocols:
                protocols.add(proof.node.notification)
                yield proof.node.notification
            pending.extend(proof.inputs)


def select_completion_frontiers(domains):
    """Admit a complete support frontier before publishing its coverage.

    The candidate graph proposes real actions, not acquired facts. A proposal
    excludes both its own candidate and every previously covered candidate.
    When a successful proposal uses new actions, check whether the already
    admitted actions suffice instead. A solver core must not needlessly admit
    alternatives. Admit the chosen support together, then publish its coverage.
    Static consumer order is not dynamic acquire order across loop instances.
    """
    graph = ProtocolTransfers(domains)

    bindings = {}
    for item in domains.selection.coverage:
        if item.notification is not None:
            bindings.setdefault(item.notification, []).append(item.binding)

    def cover(original, episodes, excluded):
        obligations = []
        for episode in episodes:
            pairs = {}
            for binding in original:
                source = domains._instance(binding.completion_source, {}, case=episode.case)
                consumer = domains._instance(binding.consumer, {}, case=episode.case)
                pairs.setdefault((source, consumer), []).append(binding)
            for (source, consumer), owners in pairs.items():
                # Each demand keeps its own activation. Unrelated writers'
                # guards must not be conjoined into a vacuous obligation.
                assumptions = tuple(
                    dict.fromkeys(
                        test
                        for point in (source, consumer)
                        for test in (*point.guard.comparisons, *domains._input_ranges(point))
                    )
                )
                proof = graph.completion(
                    source,
                    domains.dispatch.site(source).engine,
                    consumer,
                    domains.dispatch.site(consumer).engine,
                    excluded=excluded,
                    assumptions=assumptions,
                )
                if proof is None:
                    return None
                answer = domains.arithmetic.queries.implies(proof.condition, assumptions)
                if not answer.proved:
                    return None
                obligations.append(
                    CompletionObligation(
                        tuple(owners), source, consumer, episode.case, proof, answer
                    )
                )
        return CompletionCoverage(
            domains.dispatch,
            tuple(item.case for item in episodes),
            tuple(obligations),
        )

    covered, retained = {}, set()
    for notification in domains.notifications:
        if notification in retained:
            continue
        original = bindings.get(notification, ())
        # Hand-assembled internal selections without original endpoint facts
        # retain their protocols and all existing checks.
        if not original or any(item is None for item in original):
            retained.add(notification)
            continue
        episodes = graph.episodes.get(notification, ())
        expected = (
            1
            if notification.mechanism is NotificationKind.PIPE_FENCE
            else sum(domains.active[notification, case] for case in domains.cases[notification])
        )
        if not episodes or len(episodes) != expected:
            retained.add(notification)
            continue
        excluded = frozenset((*covered, notification))
        coverage = cover(original, episodes, excluded)
        if coverage is not None:
            support = frozenset(coverage.supports())
            require(not support & excluded, "completion frontier depends on an absent protocol")
            if support - retained:
                unavailable = frozenset(n for n in domains.notifications if n not in retained)
                existing = cover(original, episodes, unavailable)
                if existing is not None:
                    coverage = existing
                    support = frozenset(coverage.supports())
                    require(support <= retained, "existing frontier proposes a new protocol")
            retained.update(support)
            covered[notification] = coverage
            continue
        retained.add(notification)
    if not covered:
        return domains
    selection = domains.selection
    selection = replace(
        selection,
        notifications=tuple(item for item in selection.notifications if item in retained),
        coverage=tuple(
            replace(
                item,
                notification=None,
                completion=covered[item.notification],
                rule="already acquired queue completion frontier",
            )
            if item.notification in covered
            else item
            for item in selection.coverage
        ),
    )
    domains.work.completion_covered += len(covered)
    return domains.retain(selection)
