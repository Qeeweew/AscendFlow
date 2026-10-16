"""Case-local lease proofs and one physical allocation per notification."""

from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from types import SimpleNamespace

import pytest
from xdsl.dialects import arith
from xdsl.ir import Operation

from xdsl_npu.analysis.event_contracts import LeaseKind, ResourceWork
from xdsl_npu.analysis.event_domains import EventCase
from xdsl_npu.analysis.event_lifetimes import establish_lease, reentry, roundtrip
from xdsl_npu.analysis.event_paths import prove_event_path_cover
from xdsl_npu.analysis.event_windows import EventWindow
from xdsl_npu.analysis.integer_facts import IntegerFacts
from xdsl_npu.analysis.structured_event_sharing import (
    EventSharingContext,
    prove_finite_event_sharing,
)
from xdsl_npu.analysis.structured_protocols import (
    BoundPlacement,
    LogicalNotification,
    NotificationKind,
)
from xdsl_npu.analysis.structured_resources import _EventBanks
from xdsl_npu.analysis.structured_schedule import ActionGuard, Comparison, Placement
from xdsl_npu.scalar import Expr, expression, literal, substitute
from xdsl_npu.target.ascend910b import Actor, Engine, SyncKind

MODE = Expr("argument", ("mode",))
G = Expr("argument", ("g",))
H = Expr("argument", ("h",))


@dataclass(frozen=True)
class _Occurrence:
    operation: Operation
    participant: Actor = Actor.AIV0


def _notification(name, coordinates=(), guard=None):
    def point(placement):
        return BoundPlacement(
            occurrence=_Occurrence(arith.ConstantOp.from_int_and_width(0, 64)),
            placement=placement,
            guard=ActionGuard() if guard is None else guard,
            substitutions=(),
            coordinate_guards=(),
            families=(),
            coordinates=coordinates,
        )

    return LogicalNotification(
        name,
        (point(Placement.AFTER),),
        (point(Placement.BEFORE),),
        Engine.MTE2,
        Engine.V,
        NotificationKind.LOCAL_EVENT,
        (),
        coordinates[0][0] if coordinates else None,
        coordinates[0][1] if coordinates else literal(0),
        1,
        0,
    )


def _window(loop, ordinal, count, *, period=1, residues=(0,), upper=None):
    tests = (
        Comparison(ordinal, ">=", literal(0)),
        Comparison(ordinal, "<", count),
    ) + tuple(
        Comparison(expression("rem", ordinal, literal(period)), "!=", literal(residue))
        for residue in range(period)
        if residue not in residues
    )
    return EventWindow(
        loop,
        literal(min(residues)),
        count,
        period,
        tuple(literal(value) for value in residues),
        upper,
        ordinal=ordinal,
        lower=literal(0),
        membership=ActionGuard(tests),
        interval=ActionGuard(tests),
        invariants=ActionGuard(),
        nonempty=ActionGuard((Comparison(count, ">", literal(min(residues))),)),
    )


def _case(mode, windows=(), *, empty=False):
    return EventCase(
        condition=ActionGuard((Comparison(MODE, "==", literal(mode)),)),
        substitutions=((MODE, literal(mode)),),
        windows=windows,
        projections=tuple(window.project_exists() for window in reversed(windows)),
        empty=empty,
    )


class _Domains:
    """The established-domain protocol, with explicit test queue relations."""

    def __init__(self, cases, following=lambda *args: ()):
        self.cases = cases
        self.notifications = tuple(cases)
        self.selection = SimpleNamespace(notifications=self.notifications)
        self.work = ResourceWork()
        self.control = SimpleNamespace(
            loops={
                window.loop: SimpleNamespace(ordinal=window.ordinal)
                for alternatives in cases.values()
                for case in alternatives
                for window in case.windows
            }
        )
        self.arithmetic = IntegerFacts({})
        self.following = following
        self.pair_checks = []

    def _instance(self, placement, bindings, *, case):
        def value(expr):
            return substitute(case.specialize(expr), bindings)

        return replace(
            placement,
            coordinates=tuple((loop, value(expr)) for loop, expr in placement.coordinates),
            guard=ActionGuard(
                case.condition.comparisons
                + tuple(
                    Comparison(value(test.lhs), test.predicate, value(test.rhs))
                    for test in placement.guard.comparisons
                )
            ),
            substitutions=(),
        )

    def _instances(self, notification, case, loop, offset, *, terminal=False):
        ordinal = self.control.loops[loop].ordinal
        bindings = {ordinal: expression("add", ordinal, literal(offset))}
        yield (
            tuple(self._instance(point, bindings, case=case) for point in notification.sources),
            tuple(self._instance(point, bindings, case=case) for point in notification.consumers),
        )

    def _following_conditions(self, current, consume, following, source):
        self.work.lifetime_interfaces += 1
        return self.following(current, consume, following, source)

    def _gate(self, notification):
        return False

    def _paired_id(self, assignment, bound, event, *, case):
        self.pair_checks.append((assignment, bound, event, case))

    def fail(self, notification, detail):
        pytest.fail(f"unexpected lease failure for {notification.name}: {detail}")


def test_case_rings_reserve_the_maximum_once_and_emit_their_own_modulus(monkeypatch):
    outer, inner = (arith.ConstantOp.from_int_and_width(0, 64) for _ in range(2))
    original_guard = ActionGuard((Comparison(G, ">=", literal(0)), Comparison(H, ">=", literal(0))))
    notification = _notification("conditional", ((outer, G), (inner, H)), original_guard)
    first = _case(
        0,
        (
            _window(outer, G, literal(2), upper=2),
            _window(inner, H, literal(2), upper=2),
        ),
    )
    second = _case(
        1,
        (
            _window(outer, G, literal(1), upper=1),
            _window(inner, H, literal(3), upper=3),
        ),
    )
    domains = _Domains({notification: (first, second, _case(2, empty=True))})
    # Isolate finite-use admission: this fixture has no selected return path.
    monkeypatch.setattr("xdsl_npu.analysis.event_lifetimes.roundtrip", lambda *args: None)
    monkeypatch.setattr("xdsl_npu.analysis.event_lifetimes.reentry", lambda *args: None)
    lease = establish_lease(domains, notification)
    assert lease.width == 4
    assert [case.width for case in lease.cases] == [4, 3]
    assert [[radix for _, radix in case.rings] for case in lease.cases] == [[2, 2], [1, 3]]
    assert all(witness.kind is LeaseKind.FINITE for case in lease.cases for witness in case.inside)
    banks = _EventBanks(domains)
    assignment = banks._assign(notification, lease)
    banks._place(notification, assignment)
    assert assignment.ids == (0, 1, 2, 3)
    assert assignment.cases is lease.cases
    assert domains.work.assigned_local_ids == 4
    assert all(
        occupants == [notification]
        for pool in banks.occupied.values()
        for occupants in pool.values()
    )
    assert len(banks.actions) == 4 and len(domains.pair_checks) == 4
    for action, (_, _, _, case) in zip(banks.actions, domains.pair_checks, strict=True):
        assert action.guard.comparisons == case.condition.comparisons + original_guard.comparisons
        assert action.event is not None and action.event.kind == "rem"
        assert action.event.args[1].constant == (4 if case is first else 3)
    for mode, g, h in product(range(3), range(4), range(4)):
        active = [
            action.active_event({"mode": mode, "g": g, "h": h})
            for action in banks.actions
            if action.kind is SyncKind.LOCAL_SET
        ]
        active = [event for event in active if event is not None]
        assert active == ([] if mode == 2 else [(g % 2) * 2 + h % 2] if mode == 0 else [h % 3])


def test_all_empty_cases_have_no_ids_or_actions():
    notification = _notification("empty")
    domains = _Domains({notification: (_case(0, empty=True),)})
    lease = establish_lease(domains, notification)
    banks = _EventBanks(domains)
    assignment = banks._assign(notification, lease)
    banks._place(notification, assignment)
    assert lease.width == 0 and assignment.ids == () and assignment.cases == ()
    assert not banks.actions and domains.work.assigned_local_ids == 0


@pytest.mark.parametrize("prove", [roundtrip, reentry])
def test_peer_paths_must_cover_the_entire_target_domain(prove):
    loop = arith.ConstantOp.from_int_and_width(0, 64)
    window = _window(loop, G, literal(20))
    target = _notification("target", ((loop, G),))
    peer = replace(
        _notification("peer", ((loop, G),)),
        source_engine=Engine.V,
        destination_engine=Engine.MTE2,
    )
    target_case = _case(0, (window,))
    broad_target = replace(target_case, condition=ActionGuard(), substitutions=())
    nonzero_case = replace(
        target_case,
        condition=ActionGuard((Comparison(MODE, "!=", literal(0)),)),
        substitutions=(),
    )
    zero_case = _case(0, (window,))

    def following(current, consume, following, source):
        return (ActionGuard(),) if current is peer or following is peer else ()

    domains = _Domains({target: (broad_target,), peer: (nonzero_case,)}, following)
    assert prove(domains, target, broad_target, window, 1) is None
    domains.cases[peer] = (nonzero_case, zero_case)
    witnesses = prove(domains, target, broad_target, window, 1)
    assert witnesses and all(witness.path_cover is not None for witness in witnesses)
    for witness in witnesses:
        for mode in (0, 1):
            active = [
                path
                for path in witness.path_cover.paths
                if path.condition.evaluate({"mode": mode, "g": 0})
            ]
            assert active and all(path.steps[0].notification is peer for path in active)
    domains.cases[peer] = (nonzero_case,)
    domains.cases[target] = (target_case,)
    assert prove(domains, target, target_case, window, 1) is None
    domains.cases[peer] = (nonzero_case, zero_case)
    witnesses = prove(domains, target, target_case, window, 1)
    assert witnesses and all(witness.path_cover is not None for witness in witnesses)
    assert all(
        target_case.condition.comparisons[0] in witness.condition.comparisons
        for witness in witnesses
    )


@pytest.mark.parametrize("prove", [roundtrip, reentry])
def test_sparse_reuse_checks_the_nearest_actual_same_id_in_every_phase(prove):
    loop = arith.ConstantOp.from_int_and_width(0, 64)
    window = _window(loop, G, literal(20), period=5, residues=(0, 2))
    notification = _notification("sparse", ((loop, G),))
    case = _case(0, (window,))
    arithmetic = IntegerFacts({})

    def following(current, consume, following, source):
        old, new = dict(consume.coordinates)[loop], dict(source.coordinates)[loop]
        distance = arithmetic.normalize(expression("sub", new, old)).constant
        return (ActionGuard(),) if distance is not None and distance >= 3 else ()

    domains = _Domains({notification: (case,)}, following)
    # Members 0 and 2 are consecutive. A period-5 jump misses the unsafe reuse.
    assert prove(domains, notification, case, window, 1) is None
    witnesses = prove(domains, notification, case, window, 2)
    assert witnesses and {witness.delta for witness in witnesses} == {5}
    assert len(witnesses) == 2
    for generation in (0, 2):
        assert (
            sum(witness.condition.evaluate({"mode": 0, "g": generation}) for witness in witnesses)
            == 1
        )


def test_finite_sharing_accepts_one_case_and_rejects_multiple_cases():
    left, right = _notification("left"), _notification("right")
    finite = _case(0)
    domains = _Domains(
        {left: (finite,), right: (finite,)}, following=lambda *args: (ActionGuard(),)
    )
    root = arith.ConstantOp.from_int_and_width(0, 64)
    control = SimpleNamespace(
        module=SimpleNamespace(body=SimpleNamespace(block=SimpleNamespace(first_op=root))),
        frames={
            point.operation: SimpleNamespace(loops=lambda: ())
            for n in (left, right)
            for point in n.sources + n.consumers
        },
        invariant_over=lambda value, scope: True,
    )
    domains.selection.dependencies = SimpleNamespace(memory=SimpleNamespace(control=control))

    context = EventSharingContext(
        domains.selection,
        domains.cases,
        domains._instance,
        partial(prove_event_path_cover, domains),
    )
    witness = prove_finite_event_sharing(left, right, context=context)
    assert witness is not None and witness.path_cover.paths
    domains.cases[right] = (finite, _case(1, empty=True))

    def forbidden_path_cover(*args):
        pytest.fail("multiple cases reached the finite sharing path proof")

    context = replace(context, prove_path_cover=forbidden_path_cover)
    assert prove_finite_event_sharing(left, right, context=context) is None


def test_complementary_direct_paths_finish_before_candidate_construction():
    target = _notification("target")
    conditions = tuple(
        ActionGuard((Comparison(MODE, relation, literal(0)),)) for relation in ("==", "!=")
    )
    domains = _Domains({target: (_case(0),)}, following=lambda *args: conditions)

    def unused_candidates():
        pytest.fail("complete direct coverage must not enumerate unrelated interfaces")
        yield

    proof = prove_event_path_cover(
        domains, target, target.consumers[0], target.sources[0], (), unused_candidates()
    )
    assert proof is not None and len(proof.paths) == 2
    assert domains.work.path_states == 1 and domains.work.path_edges == 0
    assert all(len(path.steps) == 1 for path in proof.paths)


def test_partial_direct_path_keeps_the_peer_path_for_its_uncovered_domain():
    target = _notification("target")
    peer = replace(_notification("peer"), source_engine=Engine.V, destination_engine=Engine.MTE2)

    def following(current, consume, next_notification, source):
        if current is peer:
            return (ActionGuard(),)
        predicate = "==" if next_notification is target else "!="
        return (ActionGuard((Comparison(MODE, predicate, literal(0)),)),)

    domains = _Domains({target: (_case(0),), peer: (_case(0),)}, following)
    proof = prove_event_path_cover(
        domains,
        target,
        target.consumers[0],
        target.sources[0],
        (),
        ((peer, peer.sources, peer.consumers),),
    )
    assert proof is not None
    for mode in (0, 1):
        active = [path for path in proof.paths if path.condition.evaluate({"mode": mode})]
        assert active and all(len(path.steps) == mode + 1 for path in active)
