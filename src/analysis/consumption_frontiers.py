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
from .smt import all_of


@dataclass(frozen=True, slots=True)
class AcquiredConsumption:
    origin: object
    target: object
    engine: object
    premises: tuple
    proof: CausalProof
    answer: object
    after_action: bool = False


class _FrontierWord:
    def __init__(self, fact, positions, maximum):
        self.example, self.positions = fact, positions
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

    def candidates(self, origin):
        left = self.positions.position(self.example.origin.point)
        right = self.positions.position(origin.point)
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

    def add(self, origin, target, engine, premises, proof, answer, *, after_action=False):
        require(
            self.domains.arithmetic.queries.owns_implication(answer, proof.condition, premises),
            "consumption frontier lacks its exact owned mathematical implication",
        )
        if self.positions.position(origin.point) is None or self.positions.position(target) is None:
            return
        fact = AcquiredConsumption(origin, target, engine, premises, proof, answer, after_action)
        route = origin.point.actor, target.actor, engine
        key = (
            route,
            origin.notification.destination_engine,
            after_action,
            premises,
            position_word((origin.point,), self.positions)[0],
            position_word((target,), self.positions)[0],
        )
        if key not in self.words:
            scope = self.positions.position(origin.point)[-1][0]
            word = self.words[key] = _FrontierWord(fact, self.positions, self.maximum[scope])
            self.routes.setdefault(route, []).append(word)
            self.domains.work.consumption_frontier_words += 1
        self.words[key].add(fact)

    def add_boundaries(self, origin, target, engine, premises, proof):
        """Export the real acquired cuts, not just a later query endpoint.

        Only union factoring is transparent here. An actual acquire already
        carries this origin; a composed fact of another origin is not opened.
        Each exported cut keeps its own complete activation implication.
        """
        d = self.domains
        pending, seen = [proof], set()
        while pending:
            candidate = pending.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            node = candidate.node
            if node.kind == "union":
                pending.extend(candidate.inputs)
            elif (
                node.kind == "acquire"
                and node.point.actor is target.actor
                and (d._gate(node.notification) or node.notification.destination_engine is engine)
            ):
                answer = d.arithmetic.queries.implies(candidate.condition, premises)
                if answer.proved:
                    self.add(
                        origin, node.point, engine, premises, candidate, answer, after_action=True
                    )
                    d.work.consumption_acquired_boundaries += 1

    def find(self, origin, target, engine, assumptions):
        d = self.domains
        route = origin.point.actor, target.actor, engine
        seen = set()
        for word in self.routes.get(route, ()):
            example = word.example
            if (
                not d._gate(origin.notification)
                and origin.notification.destination_engine
                is not example.origin.notification.destination_engine
            ):
                continue
            for fact in word.candidates(origin):
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
                condition = all_of(*fact.premises, before, after)
                d.work.consumption_frontier_queries += 1
                answer = d.arithmetic.queries.implies(condition, assumptions)
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
