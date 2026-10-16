"""Uniform alternatives remain ORs in progress interfaces and certificates.

The small graph oracles enumerate parameter values and independently check
each admitted numeric inequality. A real DSL stream additionally exercises
EventCase identity and matching through the public progress composition.
"""

from dataclasses import replace

import pytest

from tests.support.structured import prepared, selected
from xdsl_npu.analysis.event_domains import EventCase, EventDomains
from xdsl_npu.analysis.event_progress import _Composer, prove_progress
from xdsl_npu.analysis.progress_regions import (
    ChannelProjection,
    GuardedTransfer,
    ProgressPort,
    QueueCut,
    port_key,
)
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.scalar import Expr, expression, literal
from xdsl_npu.target.ascend910b import Actor, Engine

MODE = Expr("argument", ("mode",))
N = Expr("argument", ("n",))


@pytest.fixture
def composer():
    domains = prepared(
        selected("""
    def anchors(mode:N.int64,n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for i in N.Pipelined(n,stages=1):
                N.fill(x,1.0)
    """)
    )
    return _Composer(domains)


def nodes(composer, count):
    resources = (
        (Engine.V, "input"),
        (Engine.V, "output"),
        (Engine.MTE2, "input"),
        (Engine.MTE2, "output"),
    )
    return tuple(
        QueueCut(composer.root, Actor.AIV0, engine, phase, ())
        for engine, phase in resources[:count]
    )


def edge(composer, source, destination, *tests):
    activation = composer.activations.region(composer.root).guard(ActionGuard(tests))
    return GuardedTransfer(source, destination, 0, activation, "test queue relation")


def alternatives():
    return Comparison(MODE, "==", literal(0)), Comparison(MODE, "!=", literal(0))


def assert_certificate_cover(certificates, arcs, node_group, contexts):
    for context in contexts:
        applicable = [
            item
            for item in certificates
            if set(item.nodes) == set(node_group) and item.condition.evaluate(context)
        ]
        assert applicable
        for certificate in applicable:
            rank = dict(zip(certificate.nodes, certificate.rank, strict=True))
            for index, arc in enumerate(arcs):
                if arc.source not in rank or arc.destination not in rank:
                    continue
                if arc.activation.evaluate(context):
                    assert index in certificate.relations
            for index in certificate.relations:
                arc = arcs[index]
                assert (
                    rank[arc.source] + 1 <= certificate.stride * arc.steps + rank[arc.destination]
                )


def test_alternative_routes_are_or_and_keep_each_ordered_witness(composer):
    a, b = alternatives()
    graph = ChannelProjection(
        composer.root,
        nodes(composer, 4),
        (
            edge(composer, 0, 1, a),
            edge(composer, 1, 3),
            edge(composer, 0, 2, b),
            edge(composer, 2, 3),
        ),
        (),
    )
    paths, _ = composer._paths(graph)
    paths.eliminate((1, 2))
    activation = paths.relation(0, 3)
    assert activation.kind == "or"
    assert {node.guard.comparisons for node in activation.operands} == {(a,), (b,)}
    for mode in range(4):
        assert activation.evaluate({"mode": mode})

    prefix = Comparison(N, ">", literal(0))
    residue = Comparison(
        expression("rem", expression("sub", N, literal(1)), literal(2)), "==", literal(0)
    )
    ordered = replace(
        graph, relations=(edge(composer, 0, 1, prefix, residue), edge(composer, 1, 3))
    )
    paths, _ = composer._paths(ordered)
    paths.eliminate((1,))
    activation = paths.relation(0, 3)
    assert activation.guard.comparisons == (prefix, residue)
    for count in range(5):
        assert activation.evaluate({"n": count}) == (count > 0 and (count - 1) % 2 == 0)


def test_exclusive_uniform_cycle_has_conditional_stream_and_finite_certificates(composer):
    a, b = alternatives()
    ports = nodes(composer, 2)
    arcs = (edge(composer, 0, 1, a), edge(composer, 1, 0, b))
    _, rank, certificates = composer.ranks.stream(composer.root, ports, arcs)
    assert not rank and len(certificates) == 2
    contexts = [{"mode": mode} for mode in range(4)]
    assert_certificate_cover(certificates, arcs, (0, 1), contexts)
    excluded, finite = composer.ranks.finite(composer.root, ports, arcs)
    assert excluded and not finite
    assert all(not item.condition.evaluate(context) for item in excluded for context in contexts)
    branched = (
        edge(composer, 0, 1, a),
        edge(composer, 1, 0, b),
        edge(composer, 0, 2, b),
        edge(composer, 2, 0, a),
    )
    excluded, finite = composer.ranks.finite(composer.root, nodes(composer, 3), branched)
    assert not excluded
    assert_certificate_cover(finite, branched, (0, 1, 2), contexts)


def test_a_real_cycle_in_one_uniform_alternative_is_not_anded_with_another(composer):
    a, b = alternatives()
    ports = nodes(composer, 2)
    arcs = (edge(composer, 0, 1, a), edge(composer, 1, 0, b), edge(composer, 1, 0, a))
    # mode=0 has both directed edges; mode=1 alone does not. Requiring all
    # alternatives simultaneously would incorrectly erase the real cycle.
    assert arcs[0].activation.evaluate({"mode": 0}) and arcs[2].activation.evaluate({"mode": 0})
    with pytest.raises(CompileError, match="increasing blocking-interface rank"):
        composer.ranks.stream(composer.root, ports, arcs)
    with pytest.raises(CompileError, match="blocking cycle"):
        composer.ranks.finite(composer.root, ports, arcs)


def test_iteration_predicates_do_not_become_uniform_stream_cases(composer):
    loop = next(iter(composer.control.loops.values()))
    arcs = (
        edge(composer, 0, 1, Comparison(loop.ordinal, "==", literal(0))),
        edge(composer, 1, 0, Comparison(loop.ordinal, "!=", literal(0))),
    )
    ports = nodes(composer, 2)
    with pytest.raises(CompileError, match="increasing blocking-interface rank"):
        composer.ranks.stream(loop.operation, ports, arcs)
    excluded, _ = composer.ranks.finite(loop.operation, ports, arcs)
    assert excluded


def test_independent_sccs_keep_local_cases_without_a_global_case_product(composer):
    a, b = alternatives()
    c, d = Comparison(N, "==", literal(0)), Comparison(N, "!=", literal(0))
    arcs = (
        edge(composer, 0, 1, a),
        edge(composer, 1, 0, b),
        edge(composer, 2, 3, c),
        edge(composer, 3, 2, d),
    )
    _, rank, certificates = composer.ranks.stream(composer.root, nodes(composer, 4), arcs)
    assert not rank and len(certificates) == 4
    contexts = [{"mode": mode, "n": n} for mode in range(3) for n in range(3)]
    for group in ((0, 1), (2, 3)):
        assert_certificate_cover(certificates, arcs, group, contexts)


def test_explicit_region_and_branch_budgets_reject_at_the_proof_boundary(composer):
    a, b = alternatives()
    graph = ChannelProjection(
        composer.root,
        nodes(composer, 4),
        (
            edge(composer, 0, 1, a),
            edge(composer, 1, 3),
            edge(composer, 0, 2, b),
            edge(composer, 2, 3),
        ),
        (),
    )
    paths, _ = composer._paths(graph)
    composer.activations.MAX_REGION_WORK = composer.domains.work.progress_formula_work
    with pytest.raises(CompileError, match="region activation work.*budget") as failure:
        paths.eliminate((1, 2))
    assert failure.value.phase == "structured-event-progress"
    assert failure.value.kind is ErrorKind.ANALYSIS_LIMIT
    composer.activations.MAX_REGION_WORK = 100_000
    composer.ranks.MAX_CONDITIONAL_RANKS = 1
    with pytest.raises(CompileError, match="branch budget"):
        composer.ranks.stream(
            composer.root, nodes(composer, 2), (edge(composer, 0, 1, a), edge(composer, 1, 0, b))
        )


def test_real_stream_keeps_case_identity_in_ports_pairing_and_bound_facts(monkeypatch):
    original = EventDomains._family
    parity = expression("rem", MODE, literal(2))
    empty = EventCase(ActionGuard((Comparison(MODE, "<", literal(0)),)), (), (), (), True)

    def split(self, notification):
        result = []
        for case in original(self, notification):
            for predicate in ("==", "!="):
                result.append(
                    replace(
                        case,
                        condition=ActionGuard(
                            case.condition.comparisons
                            + (Comparison(parity, predicate, literal(0)),)
                        ),
                    )
                )
        return tuple(result) + (empty,)

    monkeypatch.setattr(EventDomains, "_family", split)
    domains = prepared(
        selected("""
    def stream(X:N.Tensor((n,64),N.f32),mode:N.int64,n:N.int64):
        with N.Kernel():
            x=N.alloc_ub((64,))
            for i in N.Pipelined(n,stages=2):
                with N.Stage(0):
                    value=N.copy(X[i,:],x)
                with N.Stage(1):
                    N.mul(value,2.0,out=x)
    """)
    )
    composer = _Composer(domains)
    proof = prove_progress(domains)
    stream = proof.region.children[0].stream
    assert stream is not None
    assert all(port.case is not empty for port in stream.ports)
    for notification in domains.notifications:
        ports = [
            port
            for port in stream.ports
            if port.notification is notification and port.phase == "set"
        ]
        assert len(ports) == 2 and ports[0].case is not ports[1].case
        assert port_key(ports[0]) != port_key(ports[1])
        assert composer.endpoints.instance(ports[0]) is not composer.endpoints.instance(ports[1])
        for port in ports:
            guard = composer.endpoints.instance(port).guard.comparisons
            assert guard[: len(port.case.condition.comparisons)] == port.case.condition.comparisons
    for arc in stream.relations:
        a, b = stream.nodes[arc.source], stream.nodes[arc.destination]
        if (
            isinstance(a, ProgressPort)
            and isinstance(b, ProgressPort)
            and a.phase == "set"
            and b.phase == "wait"
            and a.notification is b.notification
        ):
            assert a.case is b.case
