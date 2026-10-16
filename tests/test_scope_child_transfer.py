"""Virtual native cuts retain exact local and composed finite reachability."""

from collections import defaultdict
from dataclasses import replace

import pytest

from tests.test_child_completion import checked
from tests.test_scope_completion import Parent
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.child_completion import project_fixed_child
from xdsl_npu.completion import CountDomain, Position, Site
from xdsl_npu.model import CompileError
from xdsl_npu.scope_child_transfer import (
    ConditionalChildTransfer,
    FixedChildTransfer,
    TransferCase,
)
from xdsl_npu.scope_completion import VirtualCut, compose_calls, compose_transfers
from xdsl_npu.scope_domains import Activation
from xdsl_npu.symbolic_completion import native_family


@pytest.fixture(scope="module")
def child():
    kernel = Kernel("transfer_copy_then_vector", disjoint_external=True)
    source = kernel.buffer("X", "GM", 64, external=True, initialized=True)
    first = kernel.buffer("a", "UB", 64)
    second = kernel.buffer("b", "UB", 64)
    kernel.loop(2, symbolic=True)
    kernel.copy("load", kernel.view(source, 64), kernel.view(first, 64))
    kernel.scale("scale", kernel.view(first, 64), kernel.view(second, 64), 2.0, stage=1)
    return checked(kernel.finish())


def sites(analysis, uid):
    actor = next(iter(analysis[0].actors))
    return tuple(Site((actor, task.uid), Position(False, generation), point, region=uid)
                 for task in analysis[0].actors[actor].program.tasks
                 for generation in (0, 1)
                 for point in ("begin", "end", "prefix"))


def finite_graph(completion, count):
    """Literal parent generations; independent of symbolic flow propagation."""
    graph, active = completion.graph, set()
    for index, meta in enumerate(graph.nodes):
        family = native_family(meta, graph.period)
        maximum = family.horizon[0] * count + family.horizon[1]
        for generation in range(family.lower, maximum + 1):
            if generation % family.period != family.residue:
                continue
            if family.last is not None and generation != family.last[0] * count + family.last[1]:
                continue
            activation = completion.node_activations.get(index)
            if activation is not None and not (
                activation.count.contains(count)
                and activation.first.evaluate(count) <= generation <= activation.last.evaluate(count)
                and generation % activation.period in activation.residues
            ):
                continue
            active.add((index, generation))
    edges = defaultdict(set)
    for source, target, cycles in graph.edges:
        delta = cycles * graph.period + graph.nodes[target].residue - graph.nodes[source].residue
        for node, generation in active:
            if node == source and (target, generation + delta) in active:
                edges[node, generation].add((target, generation + delta))
    return active, edges


def closure(edges, source):
    reached, pending = {source}, [source]
    while pending:
        for target in edges.get(pending.pop(), ()):
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def test_virtual_points_do_not_enter_parent_and_local_transfer_matches_rich_projection(child):
    plan, analysis = child
    actor = next(iter(analysis[0].actors))
    transfer = FixedChildTransfer(analysis, actor)
    parent = Parent()
    parent.add("child", plan, transfer)
    completion = compose_transfers(parent.calls, period=1, count=None)
    endpoints = sites(analysis, "child")
    before = len(completion.graph.nodes), len(completion.graph.edges)
    cuts = tuple(completion.commands_for("child", site)[0] for site in endpoints)
    assert all(isinstance(cut, VirtualCut) for cut in cuts)
    completion.prepare_virtual(cuts)
    rich = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    for source in cuts:
        assert completion.commands_for("child", source.site)[0] is source
        assert completion.direct(source.resolve(0), source.resolve(0))
        completion.inject(source.resolve(0))
        completion.extract(source.resolve(0))
        for target in cuts:
            assert completion.direct(source.resolve(0), target.resolve(0)) == (
                source.site == target.site or rich.reaches(source.site, target.site)
            )
    assert (len(completion.graph.nodes), len(completion.graph.edges)) == before
    assert completion.transfer_revision == 0


def test_routes_keep_native_antichains_and_reuse_incremental_maps(child):
    plan, analysis = child
    actor = next(iter(analysis[0].actors))
    transfer = FixedChildTransfer(analysis, actor)
    parent = Parent(3)
    parent.add("child", plan, transfer)
    completion = compose_transfers(parent.calls, period=2, count=3)
    endpoints = sites(analysis, "child")
    rich = project_fixed_child(analysis, endpoints, insertion_cuts=True)
    cuts = tuple(completion.commands_for("child", site)[0] for site in endpoints)
    virtual = tuple(cut.resolve(residue) for cut in cuts for residue in range(2))
    completion.prepare_virtual(cuts)
    records = completion.transfer_state.boundaries["child", None]
    local_maps = 2 * len({transfer.binding(site) for site in endpoints})
    omitted = 0

    def native(first, last):
        return first == last or rich.reaches(first, last)

    def check_routes(since):
        nonlocal omitted
        active, edges = finite_graph(completion, 3)
        reached = {point: closure(edges, point) for point in active}
        for node in virtual:
            for source, route in ((True, completion.inject), (False, completion.extract)):
                result = route(node)
                frontier = [record for record in records if record.nodes[node.residue] in result]
                assert route(node, since=since) == tuple(sorted(
                    record.nodes[node.residue] for record in frontier if record.revision > since
                ))
                assert all(not native(first.point, last.point)
                           for first in frontier for last in frontier if first is not last)
                eligible = [record for record in records if (
                    native(node.cut.site, record.point) if source
                    else native(record.point, node.cut.site)
                )]
                omitted += len(eligible) - len(frontier)
                for record in eligible:
                    for generation in range(node.residue, 3, 2):
                        boundary = record.nodes[node.residue], generation
                        if boundary not in active:
                            continue
                        # Every omitted local alternative remains reachable
                        # through actual retained parent arcs in this direction.
                        assert any(
                            boundary in reached.get((value, generation), ()) if source
                            else (value, generation) in reached[boundary]
                            for value in result
                        )

    check_routes(0)
    previous_records = len(records)
    previous_checks = transfer.statistics()["route_boundary_checks"]
    previous_revision = completion.transfer_revision
    # These insertion cuts become real vertices, with native macro arcs to
    # the old retained interface. Query reads must examine only these additions.
    for site in (endpoints[0], endpoints[-1]):
        completion.transfer_state.promote("child", site)
    assert len(records) > previous_records
    completion.graph.potential()
    check_routes(previous_revision)
    assert omitted
    assert transfer.statistics()["route_boundary_checks"] - previous_checks == (
        len(records) - previous_records
    ) * local_maps

    previous_requests = transfer.reach.statistics()["local_requests"]
    previous_checks = transfer.statistics()["route_boundary_checks"]
    for node in virtual:
        for route in (completion.inject, completion.extract):
            for since in (None, -1, 0, previous_revision, completion.transfer_revision):
                route(node, since=since)
                route(node, since=since)
    assert transfer.reach.statistics()["local_requests"] == previous_requests
    assert transfer.statistics()["route_boundary_checks"] == previous_checks


@pytest.mark.parametrize("count", (1, 2, 3))
def test_virtual_injection_extraction_and_promotion_match_literal_rich_graph(child, count):
    plan, analysis = child
    actor = next(iter(analysis[0].actors))
    parent = Parent(count)
    modern, legacy = {}, {}
    for order, uid in enumerate(("first", "second")):
        transfer = FixedChildTransfer(analysis, actor)
        call = parent.add(uid, plan, transfer, stage=order, order=order)
        modern[uid] = call, transfer
        legacy[uid] = call, project_fixed_child(analysis, sites(analysis, uid), insertion_cuts=True)
    reduced = compose_transfers(modern, period=1, count=count)
    rich = compose_calls(legacy, period=1, count=count)
    assert len(reduced.graph.nodes) < len(rich.graph.nodes)
    virtual = tuple(reduced.commands_for(uid, site)[0].resolve(0)
                    for uid in modern for site in sites(analysis, uid))
    reduced.prepare_virtual(node.cut for node in virtual)
    full_active, full_edges = finite_graph(rich, count)

    def compare():
        active, edges = finite_graph(reduced, count)
        for source in virtual:
            for generation in range(count):
                old_source = rich.at(source.cut.uid, source.cut.site, generation)[0], generation
                expected = closure(full_edges, old_source)
                reached = set().union(*(closure(edges, (node, generation))
                                        for node in reduced.inject(source)
                                        if (node, generation) in active))
                for target in virtual:
                    for last in range(count):
                        old_target = rich.at(target.cut.uid, target.cut.site, last)[0], last
                        observed = (
                            generation == last and reduced.direct(source, target)
                            or any((node, last) in reached for node in reduced.extract(target))
                        )
                        assert bool(observed) == (old_source in full_active and old_target in expected)

    compare()
    old = reduced.transfer_revision
    source = virtual[0]
    # Promotion is reserved for an actual synchronization attachment.
    reduced.transfer_state.promote(source.cut.uid, source.cut.site)
    assert reduced.transfer_revision > old
    assert reduced.inject(source, since=old)
    assert reduced.extract(source, since=old)
    reduced.graph.potential()
    compare()


def test_case_identity_and_first_last_existence_do_not_mix(child):
    plan, analysis = child
    actor = next(iter(analysis[0].actors))
    transfer = FixedChildTransfer(analysis, actor)
    positive = CountDomain(1, 0, minimum=2)
    first = Activation(positive, Position(False, 0), Position(False, 0))
    last = Activation(positive, Position(True, 0), Position(True, 0))
    variant = ConditionalChildTransfer(transfer.count, transfer.native_templates,
                                       (TransferCase("first", first, transfer),
                                        TransferCase("last", last, transfer)))
    parent = Parent()
    parent.add("child", plan, variant)
    completion = compose_transfers(parent.calls, period=1, count=None)
    selected = sites(analysis, "child")[0]
    cuts = completion.commands_for("child", selected)
    assert tuple(cut.case_label for cut in cuts) == ("first", "last")
    assert cuts[0].resolve(0).family.last == (0, 0)
    assert cuts[1].resolve(0).family.last == (1, -1)
    assert not completion.direct(cuts[0].resolve(0), cuts[1].resolve(0))
    assert set(completion.inject(cuts[0].resolve(0))).isdisjoint(
        completion.inject(cuts[1].resolve(0)))
    for change in ({"region": "other"}, {"position": Position(False, 1.0)}):
        with pytest.raises(CompileError, match="lexical|inner position"):
            completion.commands_for("child", replace(selected, **change))
