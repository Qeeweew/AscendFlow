"""Established queue-admission consumption facts, indexed by two real cuts.

Each entry retains its actual Wait, admission cut, complete premises and owned
mathematical answer. A query may advance along the same destination queue and
use an older consumed prefix of the same blocked queue/controller. No Set
effect FIFO, data readiness, or chosen integer event ID is assumed.
"""

from dataclasses import dataclass
from itertools import pairwise

from ..diagnostics import require
from .position_words import position_word
from .protocol_transfers import CausalProof, TransferNode
from .smt import all_of, any_of


@dataclass(frozen=True, slots=True)
class AcquiredConsumption:
    origin: object
    target: object
    engine: object
    premises: tuple
    proof: CausalProof
    answer: object
    queue_composition: bool = False
    after_action: bool = False


class _FrontierWord:
    def __init__(self, fact, positions, maximum):
        self.example, self.positions = fact, positions
        self.path = positions.position(fact.origin.point)
        self.premises = all_of(*fact.premises)
        self.answers = {}
        self.size = 1 << maximum.bit_length()
        self.best = {}

    def add(self, fact):
        source = self.positions.position(fact.origin.point)[-1][2]
        target = self.positions.position(fact.target)[-1][2]
        assert 0 <= source < self.size
        node = self.size + source
        while node:
            previous = self.best.get(node)
            if previous is None or target < previous[0]:
                self.best[node] = target, fact
            node //= 2

    def candidates(self, right):
        left = self.path
        depth = len(left) - 1
        cuts = {0, self.size}
        if len(right) > depth and left[-1][0] is right[depth][0]:
            threshold = right[depth][2]
            cuts.update((max(0, min(self.size, threshold)), max(0, min(self.size, threshold + 1))))
        bounds = sorted(cuts)
        for start, end in pairwise(bounds):
            start, end = start + self.size, end + self.size
            best = None
            while start < end:
                if start & 1:
                    candidate = self.best.get(start)
                    if candidate is not None and (best is None or candidate[0] < best[0]):
                        best = candidate
                    start += 1
                if end & 1:
                    end -= 1
                    candidate = self.best.get(end)
                    if candidate is not None and (best is None or candidate[0] < best[0]):
                        best = candidate
                start, end = start // 2, end // 2
            if best is not None:
                yield best[1]


class ConsumptionFrontiers:
    """Sparse established frontiers; arbitrary activation classes stay apart."""

    def __init__(self, domains, positions):
        self.domains, self.positions = domains, positions
        self.routes, self.words, self.maximum = {}, {}, {}
        dispatch = domains.dispatch
        for task in dispatch.tasks.values():
            self.maximum[task.scope] = max(self.maximum.get(task.scope, 0), task.lexical * 3 + 2)
        for scope, _, _, end in dispatch.boundaries.values():
            self.maximum[scope] = max(self.maximum.get(scope, 0), end * 3 + 2)
        for scope in dispatch.scopes.values():
            if scope.parent is not None:
                self.maximum[scope.parent] = max(
                    self.maximum.get(scope.parent, 0), scope.parent_lexical * 3 + 2
                )

    def add(self, fact):
        require(
            self.domains.arithmetic.queries.owns_implication(
                fact.answer, fact.proof.condition, fact.premises
            ),
            "consumption frontier lacks its exact owned mathematical implication",
        )
        origin, target = fact.origin, fact.target
        if self.positions.position(origin.point) is None or self.positions.position(target) is None:
            return
        route = origin.point.actor, target.actor, fact.engine
        key = (
            route,
            None
            if self.domains._gate(origin.notification)
            else origin.notification.destination_engine,
            fact.queue_composition,
            fact.after_action,
            fact.premises,
            position_word((origin.point,), self.positions)[0],
            position_word((target,), self.positions)[0],
        )
        if key not in self.words:
            scope = self.positions.position(origin.point)[-1][0]
            word = self.words[key] = _FrontierWord(fact, self.positions, self.maximum[scope])
            self.routes.setdefault(route, []).append(word)
            self.domains.work.consumption_frontier_words += 1
        self.words[key].add(fact)

    def add_acquisitions(self, origin, premises, proof, *, queue_composition):
        """Keep the acquired facts already established throughout this proof.

        A proof's inputs precede its conclusion; the conclusion cannot be
        backdated to them. Each acquire therefore exports only its own input
        derivation, and only where that derivation covers the complete domain.
        Captures and factoring vertices export no acquired fact.
        """
        d = self.domains
        pending, seen, acquisitions = [proof], set(), {}
        while pending:
            candidate = pending.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            pending.extend(candidate.inputs)
            node = candidate.node
            if node.kind == "acquire":
                acquisitions.setdefault(node, []).append(candidate)
        for node, alternatives in acquisitions.items():
            candidate = alternatives[0]
            if len(alternatives) > 1:
                candidate = CausalProof(
                    node,
                    tuple(dict.fromkeys(p for proof in alternatives for p in proof.inputs)),
                    any_of(*(proof.condition for proof in alternatives)),
                )
            answer = d.arithmetic.queries.implies(candidate.condition, premises)
            if answer.proved:
                engine = (
                    None if d._gate(node.notification) else node.notification.destination_engine
                )
                self.add(
                    AcquiredConsumption(
                        origin,
                        node.point,
                        engine,
                        premises,
                        candidate,
                        answer,
                        queue_composition=queue_composition,
                        after_action=True,
                    )
                )
                d.work.consumption_acquired_boundaries += 1

    def find(self, origin, target, engine, assumptions, *, queue_composition=True):
        d = self.domains
        words = tuple(
            word
            for route in dict.fromkeys(
                (
                    (origin.point.actor, target.actor, engine),
                    (origin.point.actor, target.actor, None),
                )
            )
            for word in self.routes.get(route, ())
        )
        if not words:
            return None
        context = d.arithmetic._context(assumptions)
        path = self.positions.position(origin.point)
        seen = set()
        for word in words:
            example = word.example
            if example.queue_composition and not queue_composition:
                continue
            if not d._gate(origin.notification) and (
                d._gate(example.origin.notification)
                or origin.notification.destination_engine
                is not example.origin.notification.destination_engine
            ):
                continue
            for fact in word.candidates(path):
                if id(fact) in seen:
                    continue
                seen.add(id(fact))
                before = self.positions.order(
                    origin.point,
                    fact.origin.point,
                    strict=origin.notification is not fact.origin.notification,
                )
                after = self.positions.order(fact.target, target, strict=fact.after_action)
                if before is False or after is False:
                    continue
                # One word has exactly the same complete premise sequence.
                # Its lexical bands can ask the same mathematical question at
                # many real cuts. Cache that answer, never the causal proof:
                # the latter must still use this candidate's actual Wait/cut.
                key = context, before, after
                established = word.answers.get(key)
                if established is None:
                    condition = all_of(word.premises, before, after)
                    d.work.consumption_frontier_queries += 1
                    answer = d.arithmetic.queries.implies(condition, assumptions)
                    word.answers[key] = condition, answer
                else:
                    condition, answer = established
                    d.work.consumption_frontier_answer_reuses += 1
                if not answer.proved:
                    continue
                # The established implication premises => fact.proof.condition
                # remains an explicit input, with its owned answer. The new
                # question checks only the changed cuts and activation domain.
                node = TransferNode("consumed-prefix", None, target, (), condition, fact)
                proof = CausalProof(node, (fact.proof,), condition)
                d.work.consumption_frontier_hits += 1
                return proof, answer
        return None
