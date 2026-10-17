"""Invocation-owned mathematical conjunctions for admitted progress endpoints.

Endpoint admission must already establish ordered arithmetic definedness. The
canonical predicates here serve only mathematical queries; they never replace
an ActionGuard's ordered execution witness or its original expressions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .integer_facts import IntegerFacts
from .structured_schedule import Comparison

_OPPOSITE = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}


@dataclass(frozen=True, slots=True, eq=False)
class Condition:
    """One interned mathematical set, with identity local to its owner.

    This record carries no admission or arithmetic proof certificate. Original
    expressions are retained, even when integer normalization could cancel them.
    """

    members: frozenset[Comparison]
    predicates: tuple[Comparison, ...]


class ProgressConditions:
    """Own sufficient inconsistency facts for one immutable integer environment.

    False means that this proof did not establish inconsistency. It supplies no
    satisfying assignment or safety fact. Conditions imported from another
    owner are interned anew and queried under this owner's arithmetic facts.
    """

    def __init__(self, arithmetic: IntegerFacts):
        self._arithmetic = arithmetic
        self._conditions: dict[frozenset[Comparison], Condition] = {}
        self._answers: dict[Condition, bool] = {}
        self._proof_queries = 0
        self._cache_hits = 0

    @property
    def condition_count(self) -> int:
        return len(self._conditions)

    @property
    def proof_queries(self) -> int:
        """Calls this owner made to arithmetic.prove, including failed attempts."""
        return self._proof_queries

    @property
    def cache_hits(self) -> int:
        """impossible calls reusing an established True or False answer."""
        return self._cache_hits

    def intern(self, tests: Iterable[Comparison] | Condition) -> Condition:
        members = tests.members if isinstance(tests, Condition) else frozenset(tests)
        condition = self._conditions.get(members)
        if condition is None:
            predicates = tuple(
                sorted(
                    members,
                    key=lambda test: (test.lhs.order_key, test.predicate, test.rhs.order_key),
                )
            )
            condition = self._conditions[members] = Condition(members, predicates)
        return condition

    def impossible(self, tests: Iterable[Comparison] | Condition) -> bool:
        condition = self.intern(tests)
        if condition in self._answers:
            self._cache_hits += 1
            return self._answers[condition]
        predicates = condition.predicates
        answer = False
        for index, test in enumerate(predicates):
            opposite = Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs)
            self._proof_queries += 1
            if self._arithmetic.prove(opposite, predicates[:index] + predicates[index + 1 :]):
                answer = True
                break
        self._answers[condition] = answer
        return answer
