"""Shared, factored Boolean activations for one admitted progress invocation.

Guard leaves retain their complete ordered execution witnesses. The Boolean
circuit serves mathematical queries only; interning, mapping and refutation
neither admit unsigned arithmetic nor establish exact interface projection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..budget import poll
from ..diagnostics import CompileError, ErrorKind
from .event_contracts import ResourceWork
from .guard_cover import opposite
from .progress_conditions import ProgressConditions
from .structured_schedule import ActionGuard, Comparison

_EMPTY_GUARD = ActionGuard()


@dataclass(frozen=True, slots=True, eq=False)
class Activation:
    """An immutable circuit node with invocation-local identity."""

    owner: ProgressActivations
    kind: str
    guard: ActionGuard | None = None
    operands: tuple[Activation, ...] = ()

    def evaluate(self, values) -> bool:
        """Evaluate a small independent oracle, preserving leaf short circuiting."""
        answers: dict[Activation, bool] = {}
        pending = [(self, 0)]
        while pending:
            node, stage = pending.pop()
            if node in answers:
                continue
            if node.kind in {"true", "false"}:
                answers[node] = node.kind == "true"
            elif node.kind == "guard":
                assert node.guard is not None
                answers[node] = node.guard.evaluate(values)
            elif node.kind in {"and", "or"}:
                left, right = node.operands
                if stage == 0:
                    pending.extend(((node, 1), (left, 0)))
                elif stage == 1:
                    if answers[left] == (node.kind == "or"):
                        answers[node] = answers[left]
                    else:
                        pending.extend(((node, 2), (right, 0)))
                else:
                    answers[node] = answers[right]
            else:
                raise ValueError("unknown progress activation kind")
        return answers[self]


class ProgressActivations:
    """One invocation's circuit identities and completed mathematical answers."""

    MAX_REGION_WORK = 100_000
    MAX_QUERY_BRANCHES = 64

    def __init__(self, conditions: ProgressConditions, work: ResourceWork):
        self.conditions = conditions
        self.work = work
        self._nodes: dict[tuple, Activation] = {}
        self._indices: dict[Activation, int] = {}
        self._scopes: dict[int, ActivationScope] = {}
        self._refutations: dict[tuple[Activation, ActionGuard], bool] = {}
        self._splits: dict[tuple[Activation, ActionGuard], Comparison | None] = {}

    def region(self, region) -> ActivationScope:
        scope = self._scopes.get(id(region))
        if scope is None:
            scope = self._scopes[id(region)] = ActivationScope(self, region)
        scope.charge("region access")
        return scope

    def _intern(self, kind, guard=None, operands=()) -> Activation:
        key = kind, guard, operands
        node = self._nodes.get(key)
        if node is None:
            node = self._nodes[key] = Activation(self, kind, guard, operands)
            self._indices[node] = len(self._indices)
            self.work.progress_formula_nodes += 1
        return node


class ActivationScope:
    """Charge all circuit work to the region that requested it, including hits."""

    def __init__(self, owner: ProgressActivations, region):
        self._owner = owner
        self._region = region
        self._work = 0

    def _fail(self, message):
        location = getattr(self._region, "location", None)
        raise CompileError(
            message,
            kind=ErrorKind.ANALYSIS_LIMIT,
            phase="structured-event-progress",
            location=None if location is None else str(location),
        )

    def charge(self, detail, amount=1):
        if type(amount) is not int or amount < 0:
            raise ValueError("activation work amount must be a nonnegative integer")
        self._work += amount
        self._owner.work.progress_formula_operations += 1
        self._owner.work.progress_formula_work += amount
        try:
            poll(f"progress activation {detail}", work=self._work)
        except CompileError as error:
            if error.phase is not None:
                raise
            self._fail(error.message)
        if self._work > self._owner.MAX_REGION_WORK:
            self._fail("region activation work exceeds its explicit budget")

    def _check(self, root: Activation):
        if not isinstance(root, Activation) or root.owner is not self._owner:
            raise ValueError("progress activation belongs to another owner")

    @property
    def true(self) -> Activation:
        self.charge("true construction")
        return self._owner._intern("true")

    @property
    def false(self) -> Activation:
        self.charge("false construction")
        return self._owner._intern("false")

    def guard(self, guard: ActionGuard) -> Activation:
        if not isinstance(guard, ActionGuard):
            raise TypeError("progress activation leaf needs an ordered ActionGuard")
        self.charge("guard construction", 1 + len(guard.comparisons))
        return self._owner._intern("guard", guard) if guard.comparisons else self.true

    def all_of(self, left: Activation, right: Activation) -> Activation:
        return self._binary("and", left, right)

    def any_of(self, left: Activation, right: Activation) -> Activation:
        return self._binary("or", left, right)

    def _binary(self, kind, left, right):
        self._check(left)
        self._check(right)
        self.charge(f"{kind} construction")
        absorbing, identity = ("false", "true") if kind == "and" else ("true", "false")
        if left.kind == absorbing:
            return left
        if right.kind == absorbing:
            return right
        if left.kind == identity:
            return right
        if right.kind == identity or left is right:
            return left
        dual = "or" if kind == "and" else "and"
        if right.kind == dual and left in right.operands:
            return left
        if left.kind == dual and right in left.operands:
            return right
        if self._owner._indices[left] > self._owner._indices[right]:
            left, right = right, left
        return self._owner._intern(kind, operands=(left, right))

    def map_guards(
        self, roots: Iterable[Activation], transform: Callable[[ActionGuard], ActionGuard]
    ) -> tuple[Activation, ...]:
        """Map one certified transform across a forest with one shared traversal."""
        self.charge("mapping forest")
        selected = []
        for root in roots:
            self._check(root)
            self.charge("mapping root")
            selected.append(root)
        mapped: dict[Activation, Activation] = {}
        pending = [(root, False) for root in reversed(selected)]
        while pending:
            node, ready = pending.pop()
            self.charge("mapping visit")
            if node in mapped:
                continue
            if node.kind in {"and", "or"}:
                if not ready:
                    pending.append((node, True))
                    pending.extend((child, False) for child in reversed(node.operands))
                    continue
                left, right = (mapped[child] for child in node.operands)
                mapped[node] = self._binary(node.kind, left, right)
            elif node.kind == "guard":
                assert node.guard is not None
                mapped[node] = self.guard(transform(node.guard))
            else:
                mapped[node] = self.true if node.kind == "true" else self.false
        self.charge("mapping results", len(selected))
        return tuple(mapped[root] for root in selected)

    def refuted(self, root: Activation, premises: ActionGuard = _EMPTY_GUARD) -> bool:
        self._check(root)
        self._query(premises, "refutation query")
        key = root, premises
        if key in self._owner._refutations:
            self._owner.work.progress_formula_query_hits += 1
            return self._owner._refutations[key]
        answer = _Query(self).solve((root,), premises.comparisons)
        self._owner._refutations[key] = answer
        return answer

    def split(self, root: Activation, premises: ActionGuard = _EMPTY_GUARD) -> Comparison | None:
        self._check(root)
        self._query(premises, "split query")
        key = root, premises
        if key in self._owner._splits:
            self._owner.work.progress_formula_query_hits += 1
            return self._owner._splits[key]
        query = _Query(self)
        if key in self._owner._refutations:
            self.charge("refutation cache visit")
            self._owner.work.progress_formula_query_hits += 1
            refuted = self._owner._refutations[key]
        else:
            refuted = query.solve((root,), premises.comparisons)
        answer = None if refuted else query.split(root, premises.comparisons)
        # A failed query never leaves a completed answer, including a split
        # whose refutation phase finished before a later limit was reached.
        self._owner._refutations[key] = refuted
        self._owner._splits[key] = answer
        return answer

    def _query(self, premises, detail):
        if not isinstance(premises, ActionGuard):
            raise TypeError("progress activation premises need an ordered ActionGuard")
        self.charge(detail, 1 + len(premises.comparisons))
        self._owner.work.progress_formula_queries += 1


class _Query:
    """Refute pending conjunctions, branching only on a required Boolean OR."""

    def __init__(self, scope: ActivationScope):
        self.scope = scope
        self.owner = scope._owner
        self.branches = 0

    def _branch(self):
        self.scope.charge("query branch")
        self.branches += 1
        self.owner.work.progress_formula_query_states += 1
        if self.branches > self.owner.MAX_QUERY_BRANCHES:
            self.scope._fail("activation query branches exceed their explicit budget")

    def _impossible(self, tests):
        self.scope.charge("conjunction query", 1 + len(tests))
        return self.owner.conditions.impossible(tests)

    def solve(self, pending: tuple[Activation, ...], premises: tuple[Comparison, ...]) -> bool:
        self._branch()
        self.scope.charge("pending conjunction", len(pending))
        todo, seen, choices, mandatory = list(pending), set(), [], []
        while todo:
            node = todo.pop()
            self.scope.charge("query visit")
            if node in seen:
                self.owner.work.progress_formula_query_hits += 1
                continue
            seen.add(node)
            if node.kind == "false":
                return True
            if node.kind == "guard":
                assert node.guard is not None
                mandatory.extend(node.guard.comparisons)
            elif node.kind == "and":
                todo.extend(reversed(node.operands))
            elif node.kind == "or":
                choices.append(node)
        if mandatory:
            self.scope.charge("mandatory predicates", len(premises) + len(mandatory))
            premises = tuple(dict.fromkeys((*premises, *mandatory)))
        # Mandatory guards are checked before any OR branch. In particular a
        # shared conjunct can refute a deep circuit without enumerating routes.
        if self._impossible(premises):
            return True
        if not choices:
            return False
        first, *remaining = choices
        rest = tuple(remaining)
        return all(self.solve((*rest, child), premises) for child in first.operands)

    def _guard_status(self, guard, premises):
        if self._impossible(premises + guard.comparisons):
            return False, None
        established = []
        for test in guard.comparisons:
            self.scope.charge("ordered split predicate")
            current = premises + tuple(established)
            if self._impossible(current + (opposite(test),)):
                established.append(test)
                continue
            # Earlier original predicates have all been established before
            # this comparison can become a split, including unsigned domains.
            return None, test
        return True, None

    def split(self, root, premises):
        if not self.branches:
            self._branch()
        statuses: dict[Activation, tuple[bool | None, Comparison | None]] = {}
        pending = [(root, 0)]
        while pending:
            node, stage = pending.pop()
            self.scope.charge("split visit")
            if node in statuses:
                self.owner.work.progress_formula_query_hits += 1
                continue
            if node.kind in {"true", "false"}:
                statuses[node] = node.kind == "true", None
            elif node.kind == "guard":
                assert node.guard is not None
                statuses[node] = self._guard_status(node.guard, premises)
            else:
                left, right = node.operands
                if stage == 0:
                    pending.extend(((node, 1), (left, 0)))
                    continue
                left_status, left_split = statuses[left]
                if stage == 1:
                    if left_status is (node.kind == "or"):
                        statuses[node] = left_status, None
                    else:
                        pending.extend(((node, 2), (right, 0)))
                    continue
                right_status, right_split = statuses[right]
                absorbing = node.kind == "or"
                if right_status is absorbing:
                    statuses[node] = right_status, None
                elif left_status is not None and right_status is not None:
                    statuses[node] = not absorbing, None
                else:
                    statuses[node] = None, left_split or right_split
        return statuses[root][1]
