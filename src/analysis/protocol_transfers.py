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
from dataclasses import dataclass, field

from ..budget import poll
from ..scalar import Expr, literal
from ..symbols import Symbol, SymbolKind
from .event_domains import EventCase
from .position_words import PositionWord, position_word
from .smt import all_of, any_of, negate
from .structured_protocols import NotificationKind
from .structured_schedule import ActionGuard, Comparison


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
    evidence: object | None = None


@dataclass(frozen=True, eq=False, slots=True)
class CausalProof:
    """An established cut, with immutable real-action support and deduction."""

    node: TransferNode
    inputs: tuple[CausalProof, ...]
    condition: object
    deduction: TransferDeduction | None = None


@dataclass(frozen=True, slots=True)
class TransferRule:
    """One actual capture seed or existing guarded transfer, never a fact."""

    node: TransferNode
    premise: object
    conclusion: object
    seed: bool = False
    predicate: object = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "predicate", any_of(negate(self.premise), self.conclusion))


@dataclass(frozen=True, slots=True)
class TransferDeduction:
    origin: object
    premises: tuple
    rules: tuple[TransferRule, ...]
    conclusion: object
    answer: object


@dataclass(frozen=True, slots=True)
class CompletionOrigin:
    point: object
    engine: object


@dataclass(frozen=True, slots=True)
class ConsumptionOrigin:
    """A real matched Wait, distinct from a native-completion prefix."""

    point: object
    notification: object


class ProtocolPositions:
    def __init__(self, domains):
        self.domains = domains
        self._orders = {}

    @property
    def dispatch(self):
        return self.domains.dispatch

    def position(self, point):
        return self.domains._positions(self.dispatch.site(point), point.coordinates)

    def order(self, before, after, *, strict):
        key = before, after, strict
        if key not in self._orders:
            left, right = self.position(before), self.position(after)
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
        left, right = [], []
        while start < end:
            if start & 1:
                left.append(self.tree[start])
                start += 1
            if end & 1:
                end -= 1
                right.append(self.tree[end])
            start, end = start // 2, end // 2
        yield from left
        yield from reversed(right)

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
        # The query first consumes the closed prefix entering this region.
        # Newer facts inside the tied child remain available for requirements
        # that the entry state does not cover. Inputs are visited in reverse.
        for start, end, child in (
            (last, len(self.keys), False),
            (first, last, True),
            (0, first, False),
        ):
            if start == end:
                continue
            if child:
                for word in self.children[rank]:
                    yield from word.before(point, strict=strict)
                for representative, node in self.leaves[rank]:
                    graph.work.transfer_order_bands += 1
                    condition = graph.order(representative, point, strict=strict)
                    if condition is not False:
                        graph.work.transfer_range_vertices += 1
                        yield graph.merge((node,), condition=condition)
                continue
            graph.work.transfer_order_bands += 1
            condition = graph.order(self.examples[self.keys[start]], point, strict=strict)
            if condition is False:
                continue
            for node in self.interval(start, end):
                graph.work.transfer_range_vertices += 1
                yield graph.merge((node,), condition=condition)


class _CaptureWord:
    def __init__(self, entries, positions):
        ranks, examples = {}, {}
        for number, node in entries:
            rank = positions.position(node.point)[-1][2]
            ranks.setdefault(rank, []).append(number)
            examples[rank] = (node.point,)
        self.positions = PositionWord(examples.items(), positions)
        self.members = tuple(tuple(ranks[rank]) for rank in self.positions.keys)

    def seeds(self, graph, origin):
        for start, end, (point,) in self.positions.bands((origin.point,)):
            graph.work.transfer_capture_bands += 1
            condition = graph._origin_order(origin, point)
            if condition is not False:
                for group in self.members[start:end]:
                    for number in group:
                        yield number, condition


class ProtocolTransfers:
    """One owned set of capture snapshots, independent of integer event IDs.

    Each source word is indexed once. A capture reads an immutable prefix of
    actual earlier acquires, so a later Wait cannot backdate that capture.
    Parameter cases and coordinates come from the established EventDomains.
    No runtime iteration, global LCM, or candidate ID participates here.
    """

    def __init__(self, domains, episodes=None, *, positions=None, native_composition=False):
        self.domains, self.work = domains, domains.work
        self.positions = ProtocolPositions(domains) if positions is None else positions
        self.nodes = [TransferNode("union", None, None, (), False)]
        self.episodes = {}
        self._unions, self._snapshots = {}, {}
        self.native_acquires = {}
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
                self.native_acquires.setdefault((point.actor, engine, point.operation), []).append(
                    (point, node)
                )
                key = point.actor, engine, self.positions.position(point)[0][:2]
                words.setdefault(key, []).append((point, node))
        self.words = {}
        for (actor, engine, _), entries in words.items():
            self.words.setdefault((actor, engine), []).append(_QueueWord(self, entries))
        for group in self.episodes.values():
            for episode in group:
                for point, node in zip(episode.captures, episode.publications, strict=True):
                    before = self.snapshot(
                        point,
                        episode.notification.source_engine,
                        strict=True,
                        native_only=native_composition,
                    )
                    self.nodes[node] = TransferNode(
                        "capture",
                        episode.notification,
                        point,
                        (before,),
                        all_of(*point.guard.comparisons),
                    )
        self.work.transfer_nodes += len(self.nodes)
        self.work.transfer_edges += sum(len(node.inputs) for node in self.nodes)
        self.capture_words = {}
        words = {}
        for number, node in enumerate(self.nodes):
            if node.kind == "capture":
                route = node.point.actor, node.notification.source_engine
                key = route, position_word((node.point,), self.positions)[0]
                words.setdefault(key, []).append((number, node))
        for (route, _), entries in words.items():
            self.capture_words.setdefault(route, []).append(_CaptureWord(entries, self.positions))
        self.work.transfer_capture_words += len(words)
        self.propositions, self.rules = [], []
        self._rule_system = None
        self._compile_rules()
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
        return self._unions[key]

    def _compile_rules(self):
        first = len(self.propositions)
        for number in range(first, len(self.nodes)):
            symbol = Symbol(f"reached_{number}", SymbolKind.SSA, object())
            self.propositions.append(Comparison(Expr("argument", (symbol,)), "==", literal(0)))
        for number in range(first, len(self.nodes)):
            node = self.nodes[number]
            premise = all_of(node.condition, any_of(*(self.propositions[i] for i in node.inputs)))
            self.rules.append(TransferRule(node, premise, self.propositions[number]))
        self.work.transfer_rules += len(self.nodes) - first

    def _origin_matches(self, origin, actor, engine):
        if actor is not origin.point.actor:
            return False
        if isinstance(origin, CompletionOrigin):
            return engine is origin.engine
        return (
            self.domains._gate(origin.notification)
            or engine is origin.notification.destination_engine
        )

    def _origin_order(self, origin, capture):
        return self.order(origin.point, capture, strict=isinstance(origin, ConsumptionOrigin))

    def _capture_seeds(self, origin, excluded):
        for (actor, engine), words in self.capture_words.items():
            if self._origin_matches(origin, actor, engine):
                for word in words:
                    for number, order in word.seeds(self, origin):
                        node = self.nodes[number]
                        if node.notification not in excluded:
                            premise = all_of(node.condition, order)
                            if premise is not False:
                                yield TransferRule(node, premise, self.propositions[number], True)

    def order(self, before, after, *, strict):
        return self.positions.order(before, after, strict=strict)

    def snapshot(self, point, engine, *, strict, native_only=False):
        key = point, engine, strict, native_only
        if key not in self._snapshots:
            if native_only:
                nodes = tuple(
                    self.merge((node,), condition=self.order(prior, point, strict=strict))
                    for route in ((point.actor, engine), (point.actor, None))
                    for prior, node in self.native_acquires.get((*route, point.operation), ())
                )
            else:
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
        The deduction retains the real capture/acquire rules supporting the
        conclusion under these complete original premises.
        """
        return self._derive(
            CompletionOrigin(source, engine),
            target,
            destination,
            excluded=excluded,
            assumptions=assumptions,
            strict_target=False,
        )

    def consumption(self, notification, old, target, engine, *, assumptions=()):
        """Carry an actual Wait's consumption through retained real gates.

        The origin blocks only its specified queue, or its actual controller
        for Scalar/cross Wait. A later Set acquires that fact only at a strict
        capture cut; native readiness alone never creates this origin.
        """
        return self._derive(
            ConsumptionOrigin(old, notification),
            target,
            engine,
            excluded=frozenset((notification,)),
            assumptions=assumptions,
            strict_target=True,
        )

    def _derive(
        self,
        origin,
        target,
        destination,
        *,
        excluded,
        assumptions,
        strict_target,
    ):
        """Prove the required cut under its complete original premises.

        Existing guarded transfer rules are shared; only the real origin's
        capture seeds belong to this query. Exported acquires are proved
        separately in the same temporary context.
        """
        kind = "consumption" if isinstance(origin, ConsumptionOrigin) else "completion"
        name = kind + "_queries"
        setattr(self.work, name, getattr(self.work, name) + 1)
        if self.positions.position(origin.point) is None or self.positions.position(target) is None:
            return None
        root = self.snapshot(target, destination, strict=strict_target)
        self._compile_rules()
        predicates = tuple(
            rule.predicate
            for rule in self.rules[
                0 if self._rule_system is None else len(self._rule_system.rules) :
            ]
        )
        if self._rule_system is None:
            self._rule_system = self.domains.arithmetic.queries.rule_system(predicates)
        else:
            self._rule_system.extend(predicates)
        enabled = tuple(
            i for i, rule in enumerate(self.rules) if rule.node.notification not in excluded
        )
        seeds = tuple(self._capture_seeds(origin, excluded))
        rules = (*self.rules, *seeds)
        name = kind + "_rule_instances"
        setattr(self.work, name, getattr(self.work, name) + len(enabled) + len(seeds))
        leaves = {}

        def establish(number, inference):
            support = tuple(rules[index] for index in inference.support)
            inputs = []
            for index in inference.support:
                if index not in leaves:
                    rule = rules[index]
                    node = rule.node
                    leaf = TransferNode(
                        node.kind + "-rule", node.notification, node.point, (), rule.predicate, rule
                    )
                    leaves[index] = CausalProof(leaf, (), rule.predicate)
                inputs.append(leaves[index])
            deduction = TransferDeduction(
                origin,
                assumptions,
                support,
                self.propositions[number],
                inference.answer,
            )
            name = kind + "_conclusions"
            setattr(self.work, name, getattr(self.work, name) + 1)
            return CausalProof(self.nodes[number], tuple(inputs), inference.condition, deduction)

        with self._rule_system.assuming(
            assumptions, extra=tuple(rule.predicate for rule in seeds), enabled=enabled
        ) as context:
            result = context.prove(self.propositions[root])
            if not result.answer.proved:
                return None
            proof = establish(root, result)
            if isinstance(origin, CompletionOrigin):
                return proof
            # A rule is not an acquired fact. Only independently established
            # real acquires may become inputs for future boundary composition.
            supported = {rules[index].node for index in result.support}
            acquired = []
            for number, node in enumerate(self.nodes):
                if number != root and node.kind == "acquire" and node in supported:
                    poll("consumption acquired cut", vertex=number)
                    answer = context.prove(self.propositions[number])
                    if answer.answer.proved:
                        acquired.append(establish(number, answer))
            return CausalProof(
                proof.node, (*proof.inputs, *acquired), proof.condition, proof.deduction
            )
