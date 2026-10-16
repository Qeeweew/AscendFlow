"""Sparse fixed projections preserve independent finite native reachability."""

from collections import defaultdict
from dataclasses import replace
from random import Random

import pytest

from xdsl_npu.child_completion import ChildPort, FixedChildProjection, _fixed_relations
from xdsl_npu.completion import Position, Site
from xdsl_npu.completion_flow import ALL
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.symbolic_completion import Family, SymbolicCompletion


def synthetic(edges, endpoints, *, inputs=(), outputs=(), aliases=()):
    """Project a small explicit native graph; the oracle retains its raw edges."""
    vertices = {*endpoints, *inputs, *outputs} | {node for edge in edges for node in edge}
    assert not any(target in inputs or source in outputs for source, target in edges)
    flow = SymbolicCompletion()
    nodes = {node: flow.node(Family(0, (0, 0))) for node in sorted(vertices)}
    for source, target in edges:
        flow.edge(nodes[source], nodes[target])
    sites = {node: Site(("a", str(node)), Position(False, 0), "begin") for node in endpoints}
    before = tuple(ChildPort(str(node), None, "issue", "input") for node in inputs)
    after = tuple(ChildPort(str(node), None, "issue", "output") for node in outputs)
    handles = {
        **dict(zip(before, inputs, strict=True)),
        **{site: node for node, site in sites.items()},
        **dict(zip(after, outputs, strict=True)),
    }
    for node in aliases:
        handles[replace(sites[node], position=Position(True, 0))] = node
    marks = tuple(point for point in handles if isinstance(point, Site))
    bindings = {point: (nodes[node], (0, 0)) for point, node in handles.items()}
    origins = {}
    for source in (*before, *marks):
        origins.setdefault(bindings[source], source)
    sources = tuple(origins.values())
    for bit, source in enumerate(sources):
        flow.seed(*bindings[source], ALL, 1 << bit)
    flow.propagate()
    relations = _fixed_relations(flow, sources, (*marks, *after), bindings)
    projection = FixedChildProjection(1, before, after, marks, relations, 0, 0, 0, 0, 0, ())
    return projection, handles


def check_native_paths(projection, handles, edges):
    following = defaultdict(set)
    for source, target in edges:
        following[source].add(target)
    for source, native in handles.items():
        seen, pending = {native}, [native]
        while pending:
            for target in following[pending.pop()]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        for target, destination in handles.items():
            assert projection.reaches(source, target) == (source != target and destination in seen)
    assert all(source != target for source, target in projection.relations)


@pytest.mark.parametrize(
    "edges,endpoints,aliases",
    [
        (((0, 1), (0, 2), (1, 3), (2, 3)), (3, 0, 2, 1), (0, 3)),
        ((), (4, 3, 2, 1, 0), (1, 4)),
        (((0, 1), (1, 2), (2, 1), (2, 3)), (2, 0, 3, 1), (1, 3)),
        (((0, 1), (1, 2), (2, 3)), (3, 0), (3,)),
        (((0, 1), (1, 2), (2, 0)), (1, 2, 0), (0, 2)),
    ],
)
def test_sparse_diamonds_antichains_equivalences_and_binding_aliases(edges, endpoints, aliases):
    projection, handles = synthetic(edges, endpoints, aliases=aliases)
    check_native_paths(projection, handles, edges)


def test_equal_predecessors_do_not_merge_distinct_output_ports():
    edges = ((0, 1), (1, 2), (1, 3))
    projection, handles = synthetic(edges, (1,), inputs=(0,), outputs=(2, 3), aliases=(1,))
    check_native_paths(projection, handles, edges)
    left, right = projection.outputs
    assert not projection.reaches(left, right)
    assert not projection.reaches(right, left)
    assert all(projection.reaches(projection.endpoints[0], port) for port in (left, right))


def test_admission_and_fifo_exit_do_not_acquire_reverse_native_completion():
    # issue -> admission -> begin -> end -> retirement; begin -> FIFO exit.
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7), (7, 8))
    projection, handles = synthetic(edges, (1, 2, 3, 4, 5, 6, 7), inputs=(0,), outputs=(8,))
    check_native_paths(projection, handles, edges)
    points = {native: point for point, native in handles.items()}
    assert not projection.reaches(points[2], points[1])
    assert not projection.reaches(points[4], points[6])
    assert projection.reaches(points[6], points[7])


def test_sparse_graphs_match_finite_bfs_with_permuted_origins():
    random = Random(910)
    for _ in range(40):
        endpoints = list(range(9))
        random.shuffle(endpoints)
        edges = tuple(
            (source, target)
            for source in range(9)
            for target in range(9)
            if source != target and random.random() < 0.13
        )
        projection, handles = synthetic(edges, endpoints, aliases=(endpoints[0], endpoints[-1]))
        check_native_paths(projection, handles, edges)


@pytest.mark.parametrize("chain", (False, True))
def test_large_projection_does_not_materialize_potential_or_actual_dense_closure(chain):
    size = 1800
    edges = tuple((node, node + 1) for node in range(size - 1)) if chain else ()
    projection, _ = synthetic(edges, range(size))
    assert len(projection.relations) == (size - 1 if chain else 0)
    first, last = projection.endpoints[0], projection.endpoints[-1]
    assert projection.reaches(first, last) == chain
    assert not projection.reaches(last, first)
    assert not projection.reaches(first, first)


def test_mask_word_budget_is_enforced_without_expanding_pairs(monkeypatch):
    import xdsl_npu.child_completion as child

    monkeypatch.setattr(child, "MAX_PROJECTION_RELATIONS", 100)
    with pytest.raises(CompileError, match="mask budget"):
        synthetic((), range(60))


def test_enumeration_work_budget_is_enforced_on_sparse_graphs(monkeypatch):
    import xdsl_npu.child_completion as child

    monkeypatch.setattr(child, "MAX_PROJECTION_RELATIONS", 70)
    with pytest.raises(CompileError, match="work budget"):
        synthetic(tuple((node, node + 1) for node in range(11)), range(12))


def test_structural_reaches_cache_is_local_to_projection_replacement():
    projection, _ = synthetic(((0, 1),), (0, 1), aliases=(1,))
    first, second, alias = projection.endpoints
    assert projection.reaches(first, second) and projection.reaches(alias, second)
    replacement = replace(projection, relations=frozenset())
    assert not replacement.reaches(first, second)
    assert projection.reaches(first, alias)
