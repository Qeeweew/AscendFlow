"""Completion coverage from frozen real queue captures and matching gates."""

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
    """Original ownership and the independent retained-action derivations.

    The removed notification is a description of the original obligation,
    not a leaf of any proof. Its origins are never transferred to a relay.
    """

    notification: object
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


def reduce_completion_prefixes(domains):
    """Freeze captures once, then remove only independently covered actions.

    A protocol supporting an accepted deletion is pinned. No later deletion
    can invalidate it, and every query excludes both its candidate and all
    already removed protocols. Resource and consumption facts are established
    afterwards on the final retained set.
    """
    graph = ProtocolTransfers(domains)
    bindings = {}
    for item in domains.selection.coverage:
        if item.notification is not None:
            bindings.setdefault(item.notification, []).append(item.binding)
    removed, pinned, visited = {}, set(), set()
    for notification in domains.notifications:
        if notification in pinned:
            continue
        original = bindings.get(notification, ())
        # Hand-assembled internal selections without original endpoint facts
        # retain their protocols and all existing checks.
        if not original or any(item is None for item in original):
            continue
        episodes = graph.episodes.get(notification, ())
        expected = (
            1
            if notification.mechanism is NotificationKind.PIPE_FENCE
            else sum(domains.active[notification, case] for case in domains.cases[notification])
        )
        if not episodes or len(episodes) != expected:
            continue
        obligations = []
        excluded = frozenset((*removed, notification))
        for episode in episodes:
            pairs = {}
            for binding in original:
                source = domains._instance(binding.source, {}, case=episode.case)
                consumer = domains._instance(binding.consumer, {}, case=episode.case)
                pairs.setdefault((source, consumer), []).append(binding)
            for (source, consumer), owners in pairs.items():
                # Each demand has its own activation. Conjoining unrelated
                # writers' guards could turn two real obligations into an
                # empty domain and incorrectly delete their common protocol.
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
                    break
                answer = domains.arithmetic.queries.implies(proof.condition, assumptions)
                if not answer.proved:
                    break
                obligations.append(
                    CompletionObligation(
                        tuple(owners), source, consumer, episode.case, proof, answer
                    )
                )
            else:
                continue
            break
        else:
            coverage = CompletionCoverage(
                notification,
                domains.dispatch,
                tuple(item.case for item in episodes),
                tuple(obligations),
            )
            pending = [item.proof for item in obligations]
            while pending:
                proof = pending.pop()
                if proof in visited:
                    continue
                visited.add(proof)
                if proof.node.notification is not None:
                    require(
                        proof.node.notification not in excluded,
                        "completion proof depends on a deleted protocol",
                    )
                    pinned.add(proof.node.notification)
                pending.extend(proof.inputs)
            removed[notification] = coverage
    if not removed:
        return domains, graph
    selection = domains.selection
    selection = replace(
        selection,
        notifications=tuple(item for item in selection.notifications if item not in removed),
        coverage=tuple(
            replace(
                item,
                notification=None,
                completion=removed[item.notification],
                rule="composed actual queue completion prefixes",
            )
            if item.notification in removed
            else item
            for item in selection.coverage
        ),
    )
    domains.work.completion_eliminated += len(removed)
    return domains.retain(selection), graph
