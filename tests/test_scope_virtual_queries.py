"""Virtual boundary adapters compared with independently expanded rich graphs."""

from dataclasses import dataclass

import pytest

from tests.test_scope_anchored_completion import finite_covered as anchored_finite
from tests.test_scope_guarded_completion import command, edge, finite_covered, singleton
from xdsl_npu.completion import CountDomain, Position, Site
from xdsl_npu.scope_anchored_completion import AnchoredCompletion
from xdsl_npu.scope_child_transfer import VirtualCut, direct
from xdsl_npu.scope_completion import _ClockSlot
from xdsl_npu.scope_domains import Activation
from xdsl_npu.scope_guarded_completion import GuardedCompletion

FIRST, LAST = Position(False, 0), Position(True, 0)


class _Native:
    """Finite local DAG oracle; contains no production completion machinery."""

    def __init__(self, labels, arcs):
        self.following = {label: [] for label in labels}
        for source, target in arcs:
            self.following[source].append(target)

    def covered(self, source, target):
        seen, pending = {source.task[1]}, [source.task[1]]
        while pending:
            for label in self.following[pending.pop()]:
                if label not in seen:
                    seen.add(label)
                    pending.append(label)
        return target.task[1] in seen


@dataclass
class _Body:
    key: str
    uid: str
    case_label: str | None
    activation: Activation | None
    native: _Native
    sites: dict
    cuts: dict


class _Network:
    def __init__(self, period):
        from xdsl_npu.loop_graph import Graph

        self.graph, self.rich = Graph(period, None), Graph(period, None)
        self.mapping, self.rich_mapping = {}, {}
        self.points, self.rich_points, self.bodies, self.records = {}, {}, {}, {}
        self.transfer_revision, self.virtual_index, self.prepare_calls = 0, -1, 0
        self.graph.child_transfer = self

    def _node(self, graph, mapping, activation=None, prefix=False):
        value = command(
            graph,
            len(graph.nodes) // graph.period,
            kind="scope_prefix" if prefix else "scope_component",
            point="queue" if prefix else "component",
            guard=(0, 0) if prefix else None,
        )
        if activation is not None:
            mapping.update((node, activation) for node in value[1])
        return value

    def body(self, key, labels, arcs, selected=(), activation=None, uid=None, case_label=None):
        native = _Native(labels, arcs)
        body = self.bodies[key] = _Body(
            key, key if uid is None else uid, case_label, activation, native, {}, {}
        )
        self.records[key] = []
        for label in labels:
            body.sites[label] = Site(("aic", label), FIRST, "begin", region=body.uid)
            self.rich_points[key, label] = self._node(self.rich, self.rich_mapping, activation)
        for source, target in arcs:
            edge(self.rich, self.rich_points[key, source], self.rich_points[key, target])
        for label in selected:
            self._expose(body, label, 0)
        return body

    def _expose(self, body, label, revision):
        point = self.points[body.key, label] = self._node(
            self.graph, self.mapping, body.activation
        )
        for other, nodes, _ in self.records[body.key]:
            if body.native.covered(body.sites[other], body.sites[label]):
                edge(self.graph, nodes, point)
            if body.native.covered(body.sites[label], body.sites[other]):
                edge(self.graph, point, nodes)
        self.records[body.key].append((label, point, revision))

    def promote(self, key, label):
        self.transfer_revision += 1
        self._expose(self.bodies[key], label, self.transfer_revision)

    def cut(self, key, label):
        body = self.bodies[key]
        if label not in body.cuts:
            body.cuts[label] = VirtualCut(
                self.virtual_index,
                "aic",
                _ClockSlot(body.uid, 0),
                "scope_virtual",
                uid=body.uid,
                site=body.sites[label],
                case_label=body.case_label,
                activation=body.activation,
                transfer=body.native,
                parent_period=self.graph.period,
            )
            self.virtual_index -= 1
        return body.cuts[label]

    def external(self, label, activation=None, prefix=False):
        key = "external", label
        self.points[key] = self._node(self.graph, self.mapping, activation, prefix)
        self.rich_points[key] = self._node(self.rich, self.rich_mapping, activation, prefix)
        if prefix:
            self.connect(key, key, 1)
        return key

    def connect(self, source, target, shift=0):
        edge(self.graph, self.points[source], self.points[target], shift)
        edge(self.rich, self.rich_points[source], self.rich_points[target], shift)

    def prepare_virtual(self, cuts):
        tuple(cuts)
        self.prepare_calls += 1

    def _routes(self, node, source, since):
        body = next(body for body in self.bodies.values() if body.native is node.cut.transfer)
        result = []
        for label, nodes, revision in self.records[body.key]:
            if since is not None and revision <= since:
                continue
            first, last = (
                (node.cut.site, body.sites[label])
                if source
                else (body.sites[label], node.cut.site)
            )
            if body.native.covered(first, last):
                result.append(nodes[1][node.residue])
        return tuple(result)

    def inject(self, node, *, since=None):
        return self._routes(node, True, since)

    def extract(self, node, *, since=None):
        return self._routes(node, False, since)

    @staticmethod
    def direct(first, last):
        return direct(first, last)


def _two_bodies(period):
    network = _Network(period)
    network.body("A", ("S", "O1", "O2"), (("S", "O1"), ("S", "O2")), ("O1", "O2"))
    network.body("B", ("I1", "I2", "T"), (("I1", "T"), ("I2", "T")), ("I1", "I2"))
    return network


@pytest.mark.parametrize("period", (1, 3))
def test_multi_boundary_routes_are_or_and_share_one_source_origin(period):
    network = _two_bodies(period)
    network.connect(("A", "O1"), ("B", "I1"))
    proof = GuardedCompletion(network.graph, network.mapping)
    activation = singleton(count=CountDomain(1, 0, minimum=1))
    assert proof.covered(network.cut("A", "S"), network.cut("B", "T"), 0, activation)
    assert proof.origin_count == 1  # Two injections and extractions are not four obligations.
    for count in range(1, 8):
        assert finite_covered(
            network.rich, network.rich_mapping,
            network.rich_points["A", "S"], network.rich_points["B", "T"], 0,
            activation, count,
        )


@pytest.mark.parametrize("period", (1, 3))
def test_disjoint_count_paths_join_before_target_missing_is_computed(period):
    network = _two_bodies(period)
    even = network.external("even", Activation(CountDomain(2, 0, minimum=0)))
    odd = network.external("odd", Activation(CountDomain(2, 1, minimum=0)))
    network.connect(("A", "O1"), even)
    network.connect(even, ("B", "I1"))
    network.connect(("A", "O2"), odd)
    network.connect(odd, ("B", "I2"))
    proof = GuardedCompletion(network.graph, network.mapping)
    activation = Activation()
    coverage = proof.coverage(network.cut("A", "S"), network.cut("B", "T"), 0, activation)
    for count in range(9):
        expected = finite_covered(
            network.rich, network.rich_mapping,
            network.rich_points["A", "S"], network.rich_points["B", "T"], 0,
            activation, count,
        )
        assert any(domain.contains(count) for domain in coverage) == expected
    assert proof.covered(network.cut("A", "S"), network.cut("B", "T"), 0, activation)


@pytest.mark.parametrize("period", (1, 3))
def test_direct_local_path_needs_no_boundary_and_does_not_cross_epochs(period):
    network = _Network(period)
    network.body("A", ("S", "T"), (("S", "T"),))
    source, target = network.cut("A", "S"), network.cut("A", "T")
    guarded = GuardedCompletion(network.graph, network.mapping)
    assert guarded.covered(source, target, 0, Activation())
    assert guarded.origin_count == 0 and len(network.graph.nodes) == 0
    assert not guarded.covered(source, target, 1, Activation(CountDomain(1, 0, minimum=2)))
    anchored = AnchoredCompletion(guarded)
    for first, last in ((FIRST, LAST), (FIRST, None), (None, LAST)):
        activation = Activation()
        coverage = anchored.coverage(source, target, first, last, activation)
        for count in range(9):
            expected = anchored_finite(
                network.rich, network.rich_mapping,
                network.rich_points["A", "S"], network.rich_points["A", "T"],
                first, last, activation, count,
            )
            assert any(domain.contains(count) for domain in coverage) == expected


def test_promotion_reinjects_existing_virtual_source_bit_into_new_boundary():
    network = _Network(1)
    network.body("A", ("S", "X"), (("S", "X"),))
    network.body("B", ("I", "T"), (("I", "T"),), ("I",))
    source, target = network.cut("A", "S"), network.cut("B", "T")
    proof = GuardedCompletion(network.graph, network.mapping)
    activation = singleton(count=CountDomain(1, 0, exact=3))
    assert not proof.covered(source, target, 0, activation)
    flows, origins = dict(proof.flows), proof.origin_count
    network.promote("A", "X")
    network.connect(("A", "X"), ("B", "I"))
    assert proof.covered(source, target, 0, activation)
    assert proof.origin_count == origins
    assert all(proof.flows[key] is value for key, value in flows.items())


@pytest.mark.parametrize("period", (1, 3))
def test_reverse_promotion_reinjects_old_destination_origin_without_rebuilding(period):
    network = _Network(period)
    network.body("A", ("S", "O"), (("S", "O"),), ("O",))
    network.body("B", ("X", "T"), (("X", "T"),))
    source, target = network.cut("A", "S"), network.cut("B", "T")
    proof = AnchoredCompletion(GuardedCompletion(network.graph, network.mapping))
    activation = singleton(end=True, count=CountDomain(1, 0, minimum=1))
    assert not proof.covered(source, target, None, LAST, activation)
    states, origins = dict(proof.reverse), proof.reverse_origins
    network.promote("B", "X")
    network.connect(("A", "O"), ("B", "X"))
    assert proof.covered(source, target, None, LAST, activation)
    assert proof.reverse_origins == origins
    assert all(proof.reverse[key] is value for key, value in states.items())


@pytest.mark.parametrize("period", (1, 3))
def test_span_unions_all_boundary_facts_without_filling_a_middle_hole(period):
    network = _Network(period)
    network.body("A", ("S", "O"), (("S", "O"),), ("O",))
    network.body("B", ("I1", "I2", "T"), (("I1", "T"), ("I2", "T")), ("I1", "I2"))
    prefix = network.external("prefix", prefix=True)
    tail = network.external("tail", singleton(end=True))
    network.connect(("A", "O"), ("B", "I1"))
    network.connect(("A", "O"), prefix)
    network.connect(prefix, tail)
    network.connect(tail, ("B", "I2"))
    proof = AnchoredCompletion(GuardedCompletion(network.graph, network.mapping))
    source, target = network.cut("A", "S"), network.cut("B", "T")
    query = source, target, FIRST, None, Activation()
    for enriched in (False, True):
        if enriched:
            middle = network.external(
                "middle", Activation(CountDomain(2, 0, minimum=0), Position(False, 1), Position(True, 1))
            )
            network.connect(prefix, middle)
            network.connect(middle, ("B", "I2"))
        coverage = proof.coverage(*query)
        for count in range(9):
            expected = anchored_finite(
                network.rich, network.rich_mapping,
                network.rich_points["A", "S"], network.rich_points["B", "T"],
                FIRST, None, query[-1], count,
            )
            assert expected == (count <= 2 or enriched and count % 2 == 0)
            assert any(domain.contains(count) for domain in coverage) == expected


@pytest.mark.parametrize("mode", ("guarded", "point", "span"))
def test_shared_physical_target_is_read_once_for_all_virtual_targets_and_origins(mode):
    network = _Network(1)
    network.body("A", ("S1", "S2", "O"), (("S1", "O"), ("S2", "O")), ("O",))
    network.body(
        "B", ("I", "T1", "T2", "T3"),
        (("I", "T1"), ("I", "T2"), ("I", "T3")), ("I",),
    )
    if mode == "span":
        prefix = network.external("prefix", prefix=True)
        network.connect(("A", "O"), prefix)
        network.connect(prefix, ("B", "I"))
    else:
        network.connect(("A", "O"), ("B", "I"))
    guarded = GuardedCompletion(network.graph, network.mapping)
    pairs = tuple(
        (network.cut("A", source), network.cut("B", target))
        for source in ("S1", "S2") for target in ("T1", "T2", "T3")
    )
    if mode == "guarded":
        results = guarded.covered_many((*pair, 0, singleton()) for pair in pairs)
    else:
        proof = AnchoredCompletion(guarded)
        results = proof.covered_many(
            (*pair, FIRST, None if mode == "span" else FIRST, Activation()) for pair in pairs
        )
    assert results == [True] * 6
    counts = guarded.statistics()
    assert counts["origins"] == 2
    assert counts["coverage_batches"] == 1
    assert counts["coverage_requests"] == 2
    assert counts["coverage_fact_scans"] == 1
