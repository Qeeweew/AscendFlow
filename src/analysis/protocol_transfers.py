"""Guarded queue-prefix snapshots at actual capture and acquire actions.

The immutable transfer graph records two kinds of real causal step: a matching
capture/acquire, and an earlier Wait blocking the queue/controller of a later
capture. Range vertices only factor queue prefixes; they are not instructions.
An origin is learned at a real capture, never at native issue. Facts about
native completion and event consumption use this same positional skeleton but
have distinct origins and queries.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType

from ..budget import poll
from ..scalar import literal, substitute
from .event_domains import EventCase
from .guard_cover import opposite
from .smt import all_of, any_of
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard


class _QueryFacts:
    """Elementary guard facts for pruning, never a new causal permission."""

    def __init__(self, arithmetic, assumptions):
        self.arithmetic = arithmetic
        self.tests, self.equalities, self.cache = set(), {}, {}
        intervals = {}
        for test in assumptions:
            lhs, rhs = arithmetic.normalize(test.lhs), arithmetic.normalize(test.rhs)
            self.tests.add(type(test)(lhs, test.predicate, rhs))
            if test.predicate == "==":
                if rhs.constant is not None:
                    self.equalities[lhs] = rhs.constant
                if lhs.constant is not None:
                    self.equalities[rhs] = lhs.constant
            if rhs.constant is not None:
                low, high = intervals.get(lhs, (None, None))
                value = rhs.constant
                if test.predicate in {">", ">=", "=="}:
                    bound = value + int(test.predicate == ">")
                    low = bound if low is None else max(low, bound)
                if test.predicate in {"<", "<=", "=="}:
                    bound = value - int(test.predicate == "<")
                    high = bound if high is None else min(high, bound)
                intervals[lhs] = low, high
            elif lhs.constant is not None:
                # Keep the same scalar identity when the literal is on the left.
                flip = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "==": "==", "!=": "!="}
                low, high = intervals.get(rhs, (None, None))
                relation, value = flip[test.predicate], lhs.constant
                if relation in {">", ">=", "=="}:
                    bound = value + int(relation == ">")
                    low = bound if low is None else max(low, bound)
                if relation in {"<", "<=", "=="}:
                    bound = value - int(relation == "<")
                    high = bound if high is None else min(high, bound)
                intervals[rhs] = low, high
        for value, (low, high) in intervals.items():
            if low is not None and low == high:
                self.equalities[value] = low
        self.atoms = {
            value: literal(number)
            for value, number in self.equalities.items()
            if value.kind == "argument"
        }

    def normalize(self, value):
        return (
            self.arithmetic.normalize(substitute(value, self.atoms))
            if self.atoms
            else self.arithmetic.normalize(value)
        )

    def truth(self, condition):
        if type(condition) is bool:
            return condition
        if condition in self.cache:
            return self.cache[condition]
        if hasattr(condition, "kind"):
            parts = tuple(self.truth(part) for part in condition.operands)
            if condition.kind == "and":
                answer = (
                    False if False in parts else True if all(p is True for p in parts) else None
                )
            else:
                answer = (
                    True if True in parts else False if all(p is False for p in parts) else None
                )
        else:
            lhs, rhs = self.normalize(condition.lhs), self.normalize(condition.rhs)
            test = type(condition)(lhs, condition.predicate, rhs)
            left = lhs.constant if lhs.constant is not None else self.equalities.get(lhs)
            right = rhs.constant if rhs.constant is not None else self.equalities.get(rhs)
            if left is not None and right is not None:
                answer = {
                    "==": left == right,
                    "!=": left != right,
                    "<": left < right,
                    "<=": left <= right,
                    ">": left > right,
                    ">=": left >= right,
                }[test.predicate]
            elif test in self.tests:
                answer = True
            elif opposite(test) in self.tests:
                answer = False
            else:
                answer = None
        self.cache[condition] = answer
        return answer


@dataclass(frozen=True, slots=True)
class ProtocolEpisode:
    notification: object
    case: object
    sources: tuple
    captures: tuple
    consumers: tuple
    publications: tuple[int, ...]
    acquisitions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransferNode:
    kind: str
    notification: object | None
    point: object | None
    inputs: tuple[int, ...]
    condition: object = True


@dataclass(frozen=True, eq=False, slots=True)
class CausalProof:
    """A shared acyclic derivation whose leaves name real retained actions."""

    node: TransferNode
    inputs: tuple[CausalProof, ...]
    condition: object


class _Positions:
    def __init__(self, domains):
        self.domains, self.dispatch = domains, domains.dispatch

    def position(self, point):
        return self.domains._positions(self.dispatch.site(point), point.coordinates)


class _QueueWord:
    """A region's lexical sequence with shared closed-child exits.

    A query enters only the child tied with its actual position. Earlier and
    later siblings are whole ranges whose condition uses the common parent
    clock. Guards stay on each acquire, so an empty child supplies no facts.
    """

    def __init__(self, graph, entries, depth=0):
        self.graph, self.depth = graph, depth
        self.scope = graph.positions.position(entries[0][0])[depth][0]
        graph.work.transfer_words += 1
        ranks = {}
        for point, node in entries:
            path = graph.positions.position(point)
            ranks.setdefault(path[depth][2], []).append((point, node))
        self.keys = sorted(ranks)
        self.examples = {rank: group[0][0] for rank, group in ranks.items()}
        self.children, self.leaves = {}, {}
        for rank, group in ranks.items():
            children, leaves = {}, {}
            for point, node in group:
                path = graph.positions.position(point)
                if len(path) > depth + 1:
                    children.setdefault(path[depth + 1][:2], []).append((point, node))
                else:
                    leaves.setdefault((point.operation, point.placement), []).append((point, node))
            self.children[rank] = tuple(
                _QueueWord(graph, values, depth + 1) for values in children.values()
            )
            self.leaves[rank] = tuple(
                (values[0][0], graph.merge(node for _, node in values))
                for values in leaves.values()
            )
        self.size = 1 << (len(self.keys) - 1).bit_length()
        self.tree = [0] * (2 * self.size)
        for index, rank in enumerate(self.keys):
            self.tree[self.size + index] = graph.merge(
                (
                    *[word.tree[1] for word in self.children[rank]],
                    *[node for _, node in self.leaves[rank]],
                )
            )
        for index in range(self.size - 1, 0, -1):
            self.tree[index] = graph.merge(self.tree[2 * index : 2 * index + 2])

    def interval(self, start, end):
        start, end = start + self.size, end + self.size
        while start < end:
            if start & 1:
                yield self.tree[start]
                start += 1
            if end & 1:
                end -= 1
                yield self.tree[end]
            start, end = start // 2, end // 2

    def before(self, point, *, strict):
        graph = self.graph
        path = graph.positions.position(point)
        if len(path) <= self.depth or path[self.depth][0] is not self.scope:
            # The target and this child have only their parent in common.
            # Every interior endpoint has that same complete order formula.
            graph.work.transfer_order_bands += 1
            condition = graph.order(self.examples[self.keys[0]], point, strict=strict)
            if condition is not False:
                graph.work.transfer_range_vertices += 1
                yield graph.merge((self.tree[1],), condition=condition)
            return
        rank = path[self.depth][2]
        first, last = bisect_left(self.keys, rank), bisect_right(self.keys, rank)
        for start, end in ((0, first), (last, len(self.keys))):
            if start == end:
                continue
            graph.work.transfer_order_bands += 1
            condition = graph.order(self.examples[self.keys[start]], point, strict=strict)
            if condition is False:
                continue
            for node in self.interval(start, end):
                graph.work.transfer_range_vertices += 1
                yield graph.merge((node,), condition=condition)
        if first != last:
            for word in self.children[rank]:
                yield from word.before(point, strict=strict)
            for representative, node in self.leaves[rank]:
                graph.work.transfer_order_bands += 1
                condition = graph.order(representative, point, strict=strict)
                if condition is not False:
                    graph.work.transfer_range_vertices += 1
                    yield graph.merge((node,), condition=condition)


class ProtocolTransfers:
    """One owned set of capture snapshots, independent of integer event IDs.

    Each source word is indexed once. A capture reads an immutable prefix of
    actual earlier acquires, so a later Wait cannot backdate that capture.
    Parameter cases and coordinates come from the established EventDomains.
    No runtime iteration, global LCM, or candidate ID participates here.
    """

    def __init__(self, domains, episodes=None):
        self.domains, self.work = domains, domains.work
        self.positions = _Positions(domains)
        self.nodes = [TransferNode("union", None, None, (), False)]
        self.episodes = {}
        self._unions, self._snapshots, self._orders = {}, {}, {}
        words = {}
        if episodes is None:
            episodes = self._episodes()
        for notification, case, sources, consumers in episodes:
            captures = (
                consumers if notification.mechanism is NotificationKind.PIPE_FENCE else sources
            )
            if any(
                domains._inactive(point.guard) or self.positions.position(point) is None
                for point in captures + consumers
            ):
                continue
            publications = tuple(self.reserve() for _ in captures)
            acquisitions = tuple(self.reserve() for _ in consumers)
            episode = ProtocolEpisode(
                notification, case, sources, captures, consumers, publications, acquisitions
            )
            self.episodes.setdefault(notification, []).append(episode)
            guard = all_of(
                *(test for point in captures + consumers for test in point.guard.comparisons)
            )
            for point, node in zip(consumers, acquisitions, strict=True):
                self.nodes[node] = TransferNode("acquire", notification, point, publications, guard)
                engine = None if domains._gate(notification) else notification.destination_engine
                key = point.actor, engine, self.positions.position(point)[0][:2]
                words.setdefault(key, []).append((point, node))
        self.words = {}
        for (actor, engine, _), entries in words.items():
            self.words.setdefault((actor, engine), []).append(_QueueWord(self, entries))
        for group in self.episodes.values():
            for episode in group:
                for point, node in zip(episode.captures, episode.publications, strict=True):
                    before = self.snapshot(point, episode.notification.source_engine, strict=True)
                    self.nodes[node] = TransferNode(
                        "capture",
                        episode.notification,
                        point,
                        (before,),
                        all_of(*point.guard.comparisons),
                    )
        self.work.transfer_nodes += len(self.nodes)
        self.work.transfer_edges += sum(len(node.inputs) for node in self.nodes)
        self._build_may_prefixes()
        self.ready = True

    def _episodes(self):
        domains = self.domains
        fence = EventCase(ActionGuard(), (), (), ())
        for notification in domains.notifications:
            for case in domains.cases.get(notification, (fence,)):
                if case.empty or not domains.active.get((notification, case), True):
                    continue
                yield (
                    notification,
                    case,
                    tuple(
                        domains._instance(point, {}, case=case) for point in notification.sources
                    ),
                    tuple(
                        domains._instance(point, {}, case=case) for point in notification.consumers
                    ),
                )

    def reserve(self):
        self.nodes.append(None)
        return len(self.nodes) - 1

    def merge(self, inputs, *, condition=True):
        inputs = tuple(dict.fromkeys(node for node in inputs if node))
        if not inputs or condition is False:
            return 0
        if len(inputs) == 1 and condition is True:
            return inputs[0]
        key = inputs, condition
        if key not in self._unions:
            self._unions[key] = len(self.nodes)
            self.nodes.append(TransferNode("union", None, None, inputs, condition))
            if getattr(self, "ready", False):
                self.work.transfer_nodes += 1
                self.work.transfer_edges += len(inputs)
                self._may.append(None)
        return self._unions[key]

    def _join_may(self, result, other):
        """Upper bounds only: a later possible capture subsumes an older one.

        Guards and protocol availability are deliberately ignored. This may
        retain too much, but cannot authorize a completion. It only permits
        skipping a subtree which has no possible covering origin capture.
        """
        for key, points in other.items():
            frontier = result.setdefault(key, [])
            for point in points:
                if any(self.order(point, prior, strict=False) is True for prior in frontier):
                    continue
                frontier[:] = [
                    prior
                    for prior in frontier
                    if self.order(prior, point, strict=False) is not True
                ]
                frontier.append(point)

    def _build_may_prefixes(self):
        """Factor cyclic positional dependencies without using them as proof."""
        reverse = [[] for _ in self.nodes]
        for number, node in enumerate(self.nodes):
            for child in node.inputs:
                reverse[child].append(number)
        visited, finished = set(), []
        for start in range(len(self.nodes)):
            if start in visited:
                continue
            stack = [(start, False)]
            while stack:
                number, done = stack.pop()
                if done:
                    finished.append(number)
                elif number not in visited:
                    visited.add(number)
                    stack.append((number, True))
                    stack.extend((child, False) for child in reversed(self.nodes[number].inputs))
        component, members = {}, []
        for start in reversed(finished):
            if start in component:
                continue
            group, pending = [], [start]
            while pending:
                number = pending.pop()
                if number in component:
                    continue
                component[number] = len(members)
                group.append(number)
                pending.extend(reverse[number])
            members.append(group)
        dependencies = [set() for _ in members]
        parents = [set() for _ in members]
        for number, node in enumerate(self.nodes):
            own = component[number]
            for child in node.inputs:
                other = component[child]
                if own != other:
                    dependencies[own].add(other)
                    parents[other].add(own)
        ready = deque(i for i, values in enumerate(dependencies) if not values)
        waiting = [len(values) for values in dependencies]
        self._may = [None] * len(self.nodes)
        while ready:
            own = ready.popleft()
            frontier = {}
            for other in dependencies[own]:
                self._join_may(frontier, self._may[members[other][0]])
            for number in members[own]:
                node = self.nodes[number]
                if node.kind == "capture":
                    self._join_may(
                        frontier,
                        {(node.point.actor, node.notification.source_engine): (node.point,)},
                    )
            summary = MappingProxyType({key: tuple(points) for key, points in frontier.items()})
            for number in members[own]:
                self._may[number] = summary
            self.work.transfer_may_points += sum(map(len, summary.values()))
            for parent in parents[own]:
                waiting[parent] -= 1
                if waiting[parent] == 0:
                    ready.append(parent)
        self.work.transfer_components += len(members)

    def _may_complete(self, number, source, engine, cache):
        if self._may[number] is None:
            # New query unions only point at already frozen capture snapshots.
            pending = [(number, False)]
            while pending:
                current, done = pending.pop()
                if self._may[current] is not None:
                    continue
                if done:
                    frontier = {}
                    for child in self.nodes[current].inputs:
                        self._join_may(frontier, self._may[child])
                    self._may[current] = MappingProxyType(
                        {key: tuple(points) for key, points in frontier.items()}
                    )
                else:
                    pending.append((current, True))
                    pending.extend(
                        (child, False)
                        for child in self.nodes[current].inputs
                        if self._may[child] is None
                    )
        summary = self._may[number]
        key = id(summary)
        if key not in cache:
            cache[key] = any(
                self.order(source, capture, strict=False) is not False
                for capture in summary.get((source.actor, engine), ())
            )
        return cache[key]

    def order(self, before, after, *, strict):
        key = before, after, strict
        if key not in self._orders:
            left, right = self.positions.position(before), self.positions.position(after)
            guards = ()
            if before.actor is after.actor and left is not None and right is not None:
                common = 0
                while common < min(len(left), len(right)) and left[common][0] is right[common][0]:
                    common += 1
                # Equal cuts denote the same prefix only at the same real
                # operation/phase. Distinct actions at tied anchors acquire no
                # order, including endpoints in different child invocations.
                equal_cut = (
                    not strict
                    and common == len(left) == len(right)
                    and before.operation is after.operation
                    and before.placement is after.placement
                )
                word = tuple(
                    (
                        left[i][1],
                        right[i][1],
                        i + 1 == common
                        and (right[i][2] > left[i][2] or equal_cut and right[i][2] == left[i][2]),
                    )
                    for i in range(common)
                )
                guards = self.domains._order_word(word)
            self._orders[key] = any_of(*(all_of(*guard.comparisons) for guard in guards))
        return self._orders[key]

    def snapshot(self, point, engine, *, strict):
        key = point, engine, strict
        if key not in self._snapshots:
            nodes = tuple(
                node
                for route in ((point.actor, engine), (point.actor, None))
                for word in self.words.get(route, ())
                for node in word.before(point, strict=strict)
            )
            self._snapshots[key] = self.merge(nodes)
        return self._snapshots[key]

    def completion(self, source, engine, target, destination, *, excluded, assumptions=()):
        """Derive one origin queue prefix at a real consumer admission.

        Excluded protocols cannot support themselves or another deletion.
        Gray backedges supply no facts; resulting witnesses are immutable
        acyclic derivations rooted in actual retained capture/acquire actions.
        """
        self.work.completion_queries += 1
        facts = _QueryFacts(self.domains.arithmetic, assumptions)
        if any(facts.truth(test) is False for test in assumptions):
            # The caller still proves this empty activation with the shared
            # mathematical service. No real action is pinned by an empty case.
            return CausalProof(self.nodes[0], (), False)
        if self.positions.position(source) is None or self.positions.position(target) is None:
            return None
        root = self.snapshot(target, destination, strict=False)
        results, possible, visiting, stack = {}, {}, set(), [(root, 0)]
        while stack:
            poll("completion frontier transfer")
            number, next_input = stack[-1]
            node = self.nodes[number]
            if node.notification in excluded:
                results[number] = None
                stack.pop()
                continue
            if next_input == 0 and number not in visiting:
                if facts.truth(node.condition) is False or not self._may_complete(
                    number, source, engine, possible
                ):
                    results[number] = None
                    self.work.completion_pruned += 1
                    stack.pop()
                    continue
                if (
                    node.kind == "capture"
                    and node.point.actor is source.actor
                    and node.notification.source_engine is engine
                ):
                    direct = self.order(source, node.point, strict=False)
                    if facts.truth(direct) is True:
                        results[number] = CausalProof(node, (), all_of(node.condition, direct))
                        self.work.completion_nodes += 1
                        stack.pop()
                        continue
            visiting.add(number)
            if next_input:
                previous = results.get(node.inputs[next_input - 1])
                if previous is not None and facts.truth(previous.condition) is True:
                    # One acquired frontier already covers this query's whole
                    # domain. Preserve its actual proof and the parent's guard;
                    # inspecting the rest of the union cannot enlarge that
                    # guarded fact under these same assumptions.
                    results[number] = CausalProof(
                        node, (previous,), all_of(node.condition, previous.condition)
                    )
                    self.work.completion_nodes += 1
                    self.work.completion_shortcuts += 1
                    visiting.remove(number)
                    stack.pop()
                    continue
            if next_input < len(node.inputs):
                child = node.inputs[next_input]
                stack[-1] = number, next_input + 1
                if child in visiting:
                    self.work.completion_cycles += 1
                    continue
                if child not in results:
                    stack.append((child, 0))
                continue
            inputs = tuple(results[i] for i in node.inputs if results.get(i) is not None)
            condition = all_of(node.condition, any_of(*(proof.condition for proof in inputs)))
            if (
                node.kind == "capture"
                and node.point.actor is source.actor
                and node.notification.source_engine is engine
            ):
                direct = all_of(node.condition, self.order(source, node.point, strict=False))
                if direct is not False:
                    leaf = CausalProof(node, (), direct)
                    inputs = (leaf, *inputs)
                    condition = any_of(direct, condition)
            results[number] = None if condition is False else CausalProof(node, inputs, condition)
            self.work.completion_nodes += 1
            visiting.remove(number)
            stack.pop()
        return results.get(root)
