"""Prove one obligation is covered by ordered conditional alternatives.

Integer trichotomies and finite residue partitions are proof branches only.
They never become a kernel schedule, duplicate runtime iterations, or alter a
notification's invocation-uniform domain. A missing proof stays uncovered.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from ..indexing import arguments
from ..scalar import INT64_MAX, Expr, expression, literal
from .integer_facts import IntegerFacts
from .modular import ModularAtom, ModularPredicate, modular_predicate, specialize_residues
from .structured_schedule import ActionGuard, Comparison

_OPPOSITE = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def opposite(test):
    return Comparison(test.lhs, _OPPOSITE[test.predicate], test.rhs)


@dataclass(frozen=True, slots=True)
class CoverLeaf:
    alternative: int | None
    contradiction: Comparison | ModularAtom | None = None


@dataclass(frozen=True, slots=True)
class IntegerSplit:
    value: Expr
    point: int
    children: tuple[CoverNode, CoverNode, CoverNode]
    established: tuple[Comparison, ...] = ()


@dataclass(frozen=True, slots=True)
class ResidueSplit:
    atom: ModularAtom
    children: tuple[tuple[frozenset[int], CoverNode], ...]
    established: tuple[Comparison, ...] = ()


CoverNode = CoverLeaf | IntegerSplit | ResidueSplit


@dataclass(slots=True)
class CoverWork:
    nodes: int = 0
    queries: int = 0


@dataclass(frozen=True, slots=True)
class GuardCover:
    assumptions: tuple[Comparison, ...]
    alternatives: tuple[ActionGuard, ...]
    root: CoverNode
    work: CoverWork


class _Coverage:
    def __init__(self, arithmetic, assumptions, alternatives, max_nodes, max_queries, work):
        self.arithmetic, self.assumptions, self.alternatives = arithmetic, assumptions, alternatives
        self.max_nodes, self.max_queries = max_nodes, max_queries
        self.work = work
        self.modular, self.memo, self.specializations = {}, {}, {}

    def fail(self, message, *, kind=ErrorKind.UNSUPPORTED, witness=()):
        raise CompileError(message, kind=kind, phase="event-guard-cover", witness=witness)

    def charge(self):
        self.work.queries += 1
        if self.work.queries > self.max_queries:
            self.fail(
                "guard coverage predicate work exceeds its explicit budget",
                kind=ErrorKind.ANALYSIS_LIMIT,
            )

    def prove(self, test, assumptions):
        self.charge()
        return self.arithmetic.prove(test, assumptions)

    def bindings(self, values, premises):
        names = set().union(*(arguments(value) for value in values))
        bindings = []
        for name in sorted(names):
            self.charge()
            value = Expr("argument", (name,))
            low, high = self.arithmetic.bounds(value, premises)
            if low is not None and low == high:
                bindings.append((value, low))
                continue
            # A relational premise can pin an endpoint even when ordinary
            # interval propagation has not produced the opposite bound.
            for endpoint in (low, high):
                if endpoint is None or not -INT64_MAX - 1 <= endpoint <= INT64_MAX:
                    continue
                if self.prove(Comparison(value, "==", literal(endpoint)), premises):
                    bindings.append((value, endpoint))
                    break
        return tuple(bindings)

    def residue(self, test, premises):
        self.charge()
        bindings = self.bindings((test.lhs, test.rhs), premises)
        key = test, bindings
        if key not in self.modular:
            self.modular[key] = modular_predicate(test, bindings=dict(bindings))
        return self.modular[key]

    def normalized_state(self, state, premises):
        result = {}
        for atom, values in state.items():
            predicate = ModularPredicate(atom, values)
            bindings = self.bindings(
                tuple(factor for term, _ in atom.terms for factor in term), premises
            )
            key = predicate, bindings
            if key not in self.specializations:
                self.charge()
                self.specializations[key] = specialize_residues(predicate, dict(bindings))
            normalized = self.specializations[key]
            if not normalized.residues:
                return None, atom
            if normalized.atom.modulus == 1:
                continue
            previous = result.get(normalized.atom, frozenset(range(normalized.atom.modulus)))
            narrowed = previous & normalized.residues
            if not narrowed:
                return None, normalized.atom
            result[normalized.atom] = narrowed
        return result, None

    def defined(self, tests):
        # Charge the complete ordered input before its audit. IntegerFacts
        # owns the bounded scalar reasoning behind each requested premise.
        for _ in tests:
            self.charge()
        missing = self.arithmetic.ordered_definedness(tests)
        if missing is not None:
            required, prefix = missing
            self.fail(
                "guard coverage lacks an ordered arithmetic prefix proof",
                witness=(
                    f"required: {required.lhs.text()} {required.predicate} {required.rhs.text()}",
                    "prefix: "
                    + "; ".join(f"{t.lhs.text()} {t.predicate} {t.rhs.text()}" for t in prefix),
                ),
            )

    def allowed(self, atom, state, premises):
        state, _ = self.normalized_state(state, premises)
        if state is None:
            return frozenset()
        values = state.get(atom, frozenset(range(atom.modulus)))
        # Premise congruences are exact facts, never independent assignments
        # to their component loop ordinals or uniform coefficients.
        for test in premises:
            predicate = self.residue(test, premises)
            if predicate is not None and predicate.atom == atom:
                values &= predicate.residues
        return values

    def path_status(self, guard, premises, state):
        established = []
        for test in guard.comparisons:
            current = premises + tuple(established)
            if self.prove(test, current):
                established.append(test)
                continue
            if self.prove(opposite(test), current):
                return False, None
            predicate = self.residue(test, current)
            if predicate is not None:
                allowed = self.allowed(predicate.atom, state, current)
                if not allowed or allowed <= predicate.residues:
                    established.append(test)
                    continue
                if not allowed & predicate.residues:
                    return False, None
                return None, (test, predicate, allowed, tuple(established))
            return None, (test, None, None, tuple(established))
        return True, None

    def solve(self, premises, state):
        poll("event guard cover", nodes=self.work.nodes, queries=self.work.queries)
        state, contradiction = self.normalized_state(state, premises)
        if state is None:
            return CoverLeaf(None, contradiction)
        key = premises, frozenset(state.items())
        if key in self.memo:
            return self.memo[key]
        self.work.nodes += 1
        if self.work.nodes > self.max_nodes:
            self.fail(
                "guard coverage decision nodes exceed their explicit budget",
                kind=ErrorKind.ANALYSIS_LIMIT,
            )
        for index, test in enumerate(premises):
            if self.prove(opposite(test), premises[:index] + premises[index + 1 :]):
                result = CoverLeaf(None, test)
                self.memo[key] = result
                return result
        for atom in state:
            if not self.allowed(atom, state, premises):
                result = CoverLeaf(None, atom)
                self.memo[key] = result
                return result
        choice = None
        for index, guard in enumerate(self.alternatives):
            active, pending = self.path_status(guard, premises, state)
            if active:
                result = CoverLeaf(index)
                self.memo[key] = result
                return result
            if pending is not None and choice is None:
                choice = pending
        if choice is None:
            self.memo[key] = None
            return None
        test, predicate, allowed, established = choice
        extended = tuple(dict.fromkeys(premises + established))
        if predicate is not None:
            choices = (allowed & predicate.residues, allowed - predicate.residues)
            children = []
            for values in choices:
                child = self.solve(extended, state | {predicate.atom: values})
                if child is None:
                    self.memo[key] = None
                    return None
                children.append((values, child))
            result = ResidueSplit(predicate.atom, tuple(children), established)
        else:
            value, point = (
                (test.lhs, test.rhs.constant)
                if test.rhs.constant is not None
                else (test.rhs, test.lhs.constant)
                if test.lhs.constant is not None
                else (self.arithmetic.bounded(expression("sub", test.lhs, test.rhs)), 0)
            )
            children = []
            for relation in ("<", "==", ">"):
                branch = Comparison(value, relation, literal(point))
                # The three integer relations partition the current domain.
                # A branch may establish a fact about an iteration, but does
                # not specialize or duplicate that iteration in production IR.
                child = self.solve(tuple(dict.fromkeys(extended + (branch,))), state)
                if child is None:
                    self.memo[key] = None
                    return None
                children.append(child)
            result = IntegerSplit(value, point, tuple(children), established)
        self.memo[key] = result
        return result

    def build(self):
        self.defined(self.assumptions)
        for guard in self.alternatives:
            self.defined(self.assumptions + guard.comparisons)
        root = self.solve(self.assumptions, {})
        return (
            None
            if root is None
            else GuardCover(self.assumptions, self.alternatives, root, self.work)
        )


def prove_guard_cover(
    arithmetic: IntegerFacts,
    assumptions: tuple[Comparison, ...],
    alternatives: tuple[ActionGuard, ...],
    *,
    max_nodes=128,
    max_queries=4096,
    work: CoverWork | None = None,
) -> GuardCover | None:
    """Return complete local coverage evidence, or None for an uncovered domain.

    A caller-owned work budget records failed proofs as well as successful
    ones. Reusing it shares the same limits across dependent coverage queries.
    """
    return _Coverage(
        arithmetic,
        assumptions,
        alternatives,
        max_nodes,
        max_queries,
        CoverWork() if work is None else work,
    ).build()
