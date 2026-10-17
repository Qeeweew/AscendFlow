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
from .progress_graphs import GraphTopology, PathTree, Projection
from .structured_schedule import ActionGuard, Comparison

_EMPTY_GUARD = ActionGuard()


@dataclass(frozen=True, slots=True, eq=False)
class ActivationGraph:
    """One immutable topology and its completely bound edge activations."""

    owner: ProgressActivations
    topology: GraphTopology
    inputs: tuple[Activation, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Activation:
    """An immutable circuit node with invocation-local identity."""

    owner: ProgressActivations
    kind: str
    guard: ActionGuard | None = None
    operands: tuple[Activation, ...] = ()
    path: tuple[ActivationGraph, frozenset[int], int, int] | None = None

    def evaluate(self, values) -> bool:
        """Evaluate a small independent oracle, preserving leaf short circuiting."""
        answers: dict[Activation, bool] = {}
        graph_masks: dict[ActivationGraph, int] = {}
        path_trees: dict[tuple[ActivationGraph, frozenset[int], int, int], PathTree] = {}
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
            elif node.kind == "path":
                assert node.path is not None
                graph, hidden, source, destination = node.path
                mask = graph_masks.get(graph, (1 << len(graph.inputs)) - 1)
                key = graph, hidden, source, mask
                tree = path_trees.get(key)
                if tree is None:
                    tree = path_trees[key] = graph.topology.search(
                        source, hidden, mask, lambda *args: None
                    )
                witness = tree.path(destination, lambda *args: None)
                if witness is None:
                    answers[node] = False
                    continue
                for index in witness:
                    child = graph.inputs[index]
                    if child not in answers:
                        pending.extend(((node, 1), (child, 0)))
                        break
                    if not answers[child]:
                        graph_masks[graph] = mask & ~(1 << index)
                        pending.append((node, 0))
                        break
                else:
                    answers[node] = True
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


@dataclass(frozen=True, slots=True, eq=False)
class _Premises:
    tests: tuple[Comparison, ...]
    members: frozenset[Comparison]


@dataclass(frozen=True, slots=True)
class _GraphState:
    may: int
    must: int
    inputs: tuple[tuple[bool | None, Comparison | None], ...]


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
        self._graphs: dict[tuple[GraphTopology, tuple[Activation, ...]], ActivationGraph] = {}
        self._hidden: dict[tuple[GraphTopology, int], frozenset[int]] = {}
        self._contexts: dict[tuple[Comparison, ...], _Premises] = {}
        self._statuses: dict[
            tuple[Activation, _Premises], tuple[bool | None, Comparison | None]
        ] = {}
        self._graph_states: dict[tuple[ActivationGraph, _Premises], _GraphState] = {}
        self._projections: dict[tuple[ActivationGraph, frozenset[int], int], Projection] = {}
        self._path_trees: dict[tuple[ActivationGraph, frozenset[int], int, int], PathTree] = {}
        self._mandatory: dict[Activation, tuple[Comparison, ...]] = {}

    def region(self, region) -> ActivationScope:
        scope = self._scopes.get(id(region))
        if scope is None:
            scope = self._scopes[id(region)] = ActivationScope(self, region)
        scope.charge("region access")
        return scope

    def _intern(self, kind, guard=None, operands=(), path=None) -> Activation:
        key = kind, guard, operands, path
        node = self._nodes.get(key)
        if node is None:
            node = self._nodes[key] = Activation(self, kind, guard, operands, path)
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
            self._fail(
                "region activation work exceeds its explicit budget "
                f"(work={self._work}, limit={self._owner.MAX_REGION_WORK}, detail={detail})"
            )

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

    def graph(self, topology: GraphTopology, inputs: tuple[Activation, ...]) -> ActivationGraph:
        if not isinstance(topology, GraphTopology) or not isinstance(inputs, tuple):
            raise TypeError("activation graph needs a topology and immutable input tuple")
        self.charge("graph input registration", 1 + len(topology.edges))
        if len(inputs) != len(topology.edges):
            raise ValueError("activation graph inputs differ from its topology edges")
        for root in inputs:
            self._check(root)
        key = topology, inputs
        graph = self._owner._graphs.get(key)
        if graph is None:
            graph = self._owner._graphs[key] = ActivationGraph(self._owner, topology, inputs)
        return graph

    def path(
        self, graph: ActivationGraph, hidden: frozenset[int], source: int, destination: int
    ) -> Activation:
        self.charge("path construction")
        if not isinstance(graph, ActivationGraph) or graph.owner is not self._owner:
            raise ValueError("activation graph belongs to another owner")
        self._validate_hidden(graph.topology, hidden)
        if any(
            type(node) is not int or not 0 <= node < graph.topology.size or node in hidden
            for node in (source, destination)
        ):
            raise ValueError("path endpoints must be retained topology vertices")
        return self._owner._intern("path", path=(graph, hidden, source, destination))

    def _validate_hidden(self, topology, hidden):
        if not isinstance(hidden, frozenset):
            raise TypeError("path hidden vertices need an immutable frozenset")
        key = topology, id(hidden)
        if key not in self._owner._hidden:
            self.charge("hidden vertex validation", len(hidden))
            if any(type(node) is not int or not 0 <= node < topology.size for node in hidden):
                raise ValueError("path hidden vertex is out of range")
            self._owner._hidden[key] = hidden

    def project(self, graph: ActivationGraph, hidden: frozenset[int], mask: int) -> Projection:
        """Share completed topology facts, independently of conditional proofs."""
        self.charge("projection cache visit")
        if not isinstance(graph, ActivationGraph) or graph.owner is not self._owner:
            raise ValueError("activation graph belongs to another owner")
        self._validate_hidden(graph.topology, hidden)
        if type(mask) is not int or mask < 0 or mask >> len(graph.inputs):
            raise ValueError("projection mask is outside its topology edges")
        key = graph, hidden, mask
        projection = self._owner._projections.get(key)
        if projection is None:
            projection = graph.topology.project(hidden, mask, self.charge)
            self._owner._projections[key] = projection
        else:
            self._owner.work.progress_formula_query_hits += 1
        return projection

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
        graphs: dict[ActivationGraph, ActivationGraph] = {}
        pending = [(root, False) for root in reversed(selected)]
        while pending:
            node, ready = pending.pop()
            self.charge("mapping visit")
            if node in mapped:
                continue
            if node.kind == "path":
                assert node.path is not None
                graph, hidden, source, destination = node.path
                if graph not in graphs:
                    if not ready:
                        self.charge("mapping graph inputs", len(graph.inputs))
                        pending.append((node, True))
                        pending.extend((child, False) for child in reversed(graph.inputs))
                        continue
                    graphs[graph] = self.graph(
                        graph.topology, tuple(mapped[child] for child in graph.inputs)
                    )
                mapped[node] = self.path(graphs[graph], hidden, source, destination)
            elif node.kind in {"and", "or"}:
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
        query = _Query(self)
        answer = query.solve(root, premises.comparisons)
        query.commit()
        self._owner._refutations[key] = answer
        return answer

    def split(self, root: Activation, premises: ActionGuard = _EMPTY_GUARD) -> Comparison | None:
        self._check(root)
        self._query(premises, "split query")
        key = root, premises
        if key in self._owner._splits:
            self._owner.work.progress_formula_query_hits += 1
            return self._owner._splits[key]
        if self._owner._refutations.get(key) is True:
            self.charge("refutation cache visit")
            self._owner.work.progress_formula_query_hits += 1
            self._owner._splits[key] = None
            return None
        query = _Query(self)
        answer = query.split(root, premises.comparisons)
        query.commit()
        self._owner._splits[key] = answer
        return answer

    def _query(self, premises, detail):
        if not isinstance(premises, ActionGuard):
            raise TypeError("progress activation premises need an ordered ActionGuard")
        self.charge(detail, 1 + len(premises.comparisons))
        self._owner.work.progress_formula_queries += 1


class _Query:
    """One bounded joint Boolean query under complete, shared premises."""

    def __init__(self, scope: ActivationScope):
        self.scope = scope
        self.owner = scope._owner
        self.branches = 0
        self.contexts = {}
        self.statuses = {}
        self.graph_states = {}
        self.path_trees = {}
        self.mandatory = {}

    def _hit(self):
        self.owner.work.progress_formula_query_hits += 1

    def _branch(self):
        self.scope.charge("query branch")
        self.branches += 1
        self.owner.work.progress_formula_query_states += 1
        if self.branches > self.owner.MAX_QUERY_BRANCHES:
            self.scope._fail("activation query branches exceed their explicit budget")

    def _context(self, tests):
        self.scope.charge("premise context", 1 + len(tests))
        context = self.contexts.get(tests) or self.owner._contexts.get(tests)
        if context is None:
            context = self.contexts[tests] = _Premises(tests, frozenset(tests))
        else:
            self._hit()
        return context

    def _impossible(self, tests):
        self.scope.charge("conjunction query", 1 + len(tests))
        return self.owner.conditions.impossible(tests)

    def _extend(self, context, additions):
        if not additions:
            self.scope.charge("empty conjunction extension")
            return context.tests
        self.scope.charge("conjunction extension", len(context.tests) + len(additions))
        members, extra = set(context.members), []
        for test in additions:
            if test not in members:
                members.add(test)
                extra.append(test)
        return context.tests + tuple(extra)

    def _mandatory(self, root):
        self.scope.charge("mandatory cache visit")
        cached = self.mandatory.get(root)
        if cached is None:
            cached = self.owner._mandatory.get(root)
        if cached is not None:
            self._hit()
            return cached
        pending, seen, members, tests = [root], set(), set(), []
        while pending:
            node = pending.pop()
            self.scope.charge("mandatory visit")
            if node in seen:
                self._hit()
                continue
            seen.add(node)
            if node.kind == "guard":
                assert node.guard is not None
                self.scope.charge("mandatory guard predicates", len(node.guard.comparisons))
                for test in node.guard.comparisons:
                    if test not in members:
                        members.add(test)
                        tests.append(test)
            elif node.kind == "and":
                pending.extend(reversed(node.operands))
        result = self.mandatory[root] = tuple(tests)
        return result

    def _lookup(self, node, context):
        self.scope.charge("status cache visit")
        key = node, context
        answer = self.statuses.get(key)
        if answer is None:
            answer = self.owner._statuses.get(key)
        if answer is not None:
            self._hit()
        return answer

    def _guard_status(self, guard, context):
        if self._impossible(context.tests + guard.comparisons):
            return False, None
        established = []
        for test in guard.comparisons:
            self.scope.charge("ordered split predicate")
            inverse = opposite(test)
            if inverse in context.members:
                return False, None
            if test in context.members or self._impossible(
                context.tests + tuple(established) + (inverse,)
            ):
                established.append(test)
                continue
            # No later predicate can become a split until this original
            # comparison and its unsigned prerequisites have been established.
            return None, test
        return True, None

    def _graph_state(self, graph, context, *, build=False):
        self.scope.charge("graph state cache visit")
        key = graph, context
        state = self.graph_states.get(key) or self.owner._graph_states.get(key)
        if state is not None:
            self._hit()
            return state
        if not build:
            return None
        may, must, inputs = 0, 0, []
        for index, root in enumerate(graph.inputs):
            self.scope.charge("graph input state")
            answer = self._lookup(root, context)
            assert answer is not None
            inputs.append(answer)
            if answer[0] is not False:
                may |= 1 << index
            if answer[0] is True:
                must |= 1 << index
        state = self.graph_states[key] = _GraphState(may, must, tuple(inputs))
        return state

    def _path_witness(self, graph, hidden, source, destination, mask):
        self.scope.charge("path tree cache visit")
        key = graph, hidden, source, mask
        tree = self.path_trees.get(key) or self.owner._path_trees.get(key)
        if tree is None:
            tree = self.path_trees[key] = graph.topology.search(
                source, hidden, mask, self.scope.charge
            )
        else:
            self._hit()
        return tree.path(destination, self.scope.charge)

    def _path_status(self, node, context, state):
        assert node.path is not None
        graph, hidden, source, destination = node.path
        bit = 1 << destination
        if not self.scope.project(graph, hidden, state.may).rows[source] & bit:
            return False, None
        if self.scope.project(graph, hidden, state.must).rows[source] & bit:
            return True, None
        witness = self._path_witness(graph, hidden, source, destination, state.may)
        assert witness is not None, "may projection has no nonempty path witness"
        for index in witness:
            self.scope.charge("path split edge")
            value, split = state.inputs[index]
            if value is None and split is not None:
                return None, split
        return None, None

    def status(self, root, context):
        pending = [(root, 0)]
        while pending:
            node, stage = pending.pop()
            self.scope.charge("query visit")
            if self._lookup(node, context) is not None:
                continue
            key = node, context
            if node.kind in {"true", "false"}:
                self.statuses[key] = node.kind == "true", None
            elif node.kind == "guard":
                assert node.guard is not None
                self.statuses[key] = self._guard_status(node.guard, context)
            elif node.kind == "path":
                assert node.path is not None
                graph = node.path[0]
                state = self._graph_state(graph, context)
                if state is None:
                    if stage == 0:
                        self.scope.charge("graph dependencies", len(graph.inputs))
                        pending.append((node, 1))
                        pending.extend((child, 0) for child in reversed(graph.inputs))
                        continue
                    state = self._graph_state(graph, context, build=True)
                self.statuses[key] = self._path_status(node, context, state)
            else:
                left, right = node.operands
                if stage == 0:
                    pending.extend(((node, 1), (left, 0)))
                    continue
                left_answer = self._lookup(left, context)
                assert left_answer is not None
                left_status, left_split = left_answer
                absorbing = node.kind == "or"
                if stage == 1:
                    if left_status is absorbing:
                        self.statuses[key] = left_status, None
                    else:
                        pending.extend(((node, 2), (right, 0)))
                    continue
                right_answer = self._lookup(right, context)
                assert right_answer is not None
                right_status, right_split = right_answer
                if right_status is absorbing:
                    self.statuses[key] = right_status, None
                elif left_status is not None and right_status is not None:
                    self.statuses[key] = not absorbing, None
                else:
                    self.statuses[key] = None, left_split or right_split
        answer = self._lookup(root, context)
        assert answer is not None
        return answer

    def _witness_term(self, root, context):
        """Collect one joint conjunction sufficient for this formula."""
        pending, seen, members, tests = [root], set(), set(), []
        while pending:
            node = pending.pop()
            self.scope.charge("joint witness visit")
            if node in seen:
                self._hit()
                continue
            seen.add(node)
            answer = self._lookup(node, context)
            assert answer is not None and answer[0] is not False
            if answer[0] is True:
                continue
            if node.kind == "guard":
                assert node.guard is not None
                self.scope.charge("joint witness predicates", len(node.guard.comparisons))
                for test in node.guard.comparisons:
                    if test not in members:
                        members.add(test)
                        tests.append(test)
            elif node.kind == "and":
                pending.extend(reversed(node.operands))
            elif node.kind == "or":
                for child in node.operands:
                    status = self._lookup(child, context)
                    if status is None:
                        status = self.status(child, context)
                    if status[0] is not False:
                        pending.append(child)
                        break
                else:
                    raise AssertionError("possible OR has no possible witness")
            elif node.kind == "path":
                assert node.path is not None
                graph, hidden, source, destination = node.path
                state = self._graph_state(graph, context)
                assert state is not None
                witness = self._path_witness(graph, hidden, source, destination, state.may)
                assert witness is not None
                self.scope.charge("joint witness edges", len(witness))
                pending.extend(graph.inputs[index] for index in reversed(witness))
            else:
                raise AssertionError("unknown activation has no guard or path witness")
        return tuple(tests)

    def solve(self, root, premises):
        self._branch()
        original = self._context(premises)
        tests = self._extend(original, self._mandatory(root))
        context = original if tests is original.tests else self._context(tests)
        # All mandatory ordered guards participate in one contradiction query,
        # before any reachability or conditional branch work is requested.
        if self._impossible(context.tests):
            return True
        value, split = self.status(root, context)
        if value is False:
            return True
        if value is True:
            return False
        term = self._witness_term(root, context)
        if not self._impossible(self._extend(context, term)):
            # A sufficient conjunction was not refuted. This is an unproved
            # contradiction, never a satisfying assignment or exact witness.
            return False
        if split is None:
            return False
        return self.solve(root, context.tests + (split,)) and self.solve(
            root, context.tests + (opposite(split),)
        )

    def split(self, root, premises):
        self._branch()
        context = self._context(premises)
        if self._impossible(self._extend(context, self._mandatory(root))):
            return None
        # Case selection uses the original premises, not the mandatory facts
        # assumed only while refuting an active formula.
        return self.status(root, context)[1]

    def commit(self):
        caches = (
            (self.owner._contexts, self.contexts),
            (self.owner._statuses, self.statuses),
            (self.owner._graph_states, self.graph_states),
            (self.owner._path_trees, self.path_trees),
            (self.owner._mandatory, self.mandatory),
        )
        self.scope.charge("completed query caches", 1 + sum(len(local) for _, local in caches))
        for target, local in caches:
            target.update(local)
