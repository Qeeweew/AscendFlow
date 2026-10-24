"""Actual blocking routes constrain the composition of consumed prefixes."""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.consumption_frontiers import ConsumptionFrontiers
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.analysis.protocol_transfers import CausalProof, ConsumptionOrigin, TransferNode
from ascendflow.analysis.smt import all_of, any_of
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import Expr, literal
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.counter_epochs import counter_epochs
from tests.models.structured_events import check_compiled_events
from tests.sync.test_consumption_transfers import cube_chain


def mixed_prefixes():
    return N.from_source(
        """
def prefixes(A:N.Tensor((16,16),N.bf16),B:N.Tensor((16,16),N.bf16),
             X:N.Tensor((64,),N.f32),Work:N.Tensor((16,16),N.f32,access="write"),
             Out:N.Tensor((2,64),N.f32,access="write",write_disjoint=True)):
    with N.Kernel(1,kind="mixed"):
        a1=N.alloc_l1((16,16),N.bf16)
        b1=N.alloc_l1((16,16),N.bf16)
        a0=N.alloc_l0a((16,16),N.bf16)
        b0=N.alloc_l0b((16,16),N.bf16)
        c0=N.alloc_l0c((16,16),N.f32)
        x=N.alloc_ub((64,),N.f32)
        y=N.alloc_ub((8,16),N.f32)
        z=N.alloc_ub((64,),N.f32)
        with N.Cube():
            N.copy(A,a1)
            N.copy(B,b1)
            N.copy(a1,a0)
            N.copy(b1,b0)
            N.matmul(a0,b0,accum=c0,initialize=True)
            N.copy(c0,Work)
        with N.Vector() as lane:
            N.fill(x,1.0)
            N.copy(X,x)
            N.copy(Work[lane*8:lane*8+8,:],y)
            N.fill(z,2.0)
            N.copy(X,z)
            N.copy(z,Out[lane,:])
""",
        disjoint_external=True,
    )


def test_later_controller_wait_does_not_consume_an_earlier_pipe_wait():
    plan = synchronize_structure(mixed_prefixes())
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.mechanism is NotificationKind.LOCAL_EVENT
        and n.source_engine is Engine.V
        and n.destination_engine is Engine.MTE2
        and n.sources[0].actor is Actor.AIV0
    ]
    earlier, later = messages
    controller = next(
        n
        for n in domains.notifications
        if n.mechanism is NotificationKind.CROSS_EVENT and n.destination_engine is Engine.MTE2
    )
    old = next(p for p in controller.consumers if p.actor is Actor.AIV0)
    target = later.sources[0]
    assumptions = old.guard.comparisons + target.guard.comparisons
    rule = HandbackRule(domains)
    assert rule.establish(controller, old, later, target, assumptions, loop=None)
    assert rule.frontiers.find(ConsumptionOrigin(old, controller), target, Engine.V, assumptions)

    # The independent model executes the real local/cross Wait APIs. A later
    # controller gate releases dispatch without consuming the earlier MTE2
    # queue's pending local Wait, even though both destination enums say MTE2.
    model = check_compiled_events(plan)

    def use(notification):
        return next(
            m
            for m in model.consumptions
            if m.wait.occurrence.actor is Actor.AIV0
            and notification.name in m.source.occurrence.operation.protocol.data.split(", ")
        )

    first_use, controller_use, next_use = map(use, (earlier, controller, later))
    assert model.happens_before(controller_use.wait.begin, next_use.source.end)
    assert not model.happens_before(first_use.wait.begin, next_use.source.end)
    assert (
        rule.frontiers.find(
            ConsumptionOrigin(earlier.consumers[0], earlier), target, Engine.V, assumptions
        )
        is None
    )
    assert (
        rule.establish(
            earlier,
            earlier.consumers[0],
            later,
            target,
            assumptions,
            loop=None,
        )
        is None
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_one_handback_exports_its_actual_intermediate_acquisitions(mixed):
    source = (
        replace(
            counter_epochs,
            source=counter_epochs.source.replace("N.serial(n)", "N.unroll(1)"),
        ).specialize(PHASES=2)
        if mixed
        else cube_chain()
    )
    plan = synchronize_structure(source)
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is (Engine.FIX if mixed else Engine.MTE2)
        and n.destination_engine is (Engine.MTE2 if mixed else Engine.MTE1)
        and n.sources[0].actor is Actor.AIC
    ]
    earlier, later = messages[0], messages[-1]
    old, target = earlier.consumers[0], later.sources[0]
    assumptions = old.guard.comparisons + target.guard.comparisons
    rule = HandbackRule(domains)
    assert rule.establish(earlier, old, later, target, assumptions, loop=None)
    queries = domains.work.consumption_queries
    model = check_compiled_events(plan, {"n": 1} if mixed else {})
    old_wait = next(
        m.wait
        for m in model.consumptions
        if m.wait.occurrence.actor is old.actor
        and earlier.name in m.source.occurrence.operation.protocol.data.split(", ")
    )
    facts = {
        id(fact): fact
        for word in rule.frontiers.words.values()
        for _, fact in word.best.values()
        if fact.after_action
    }.values()
    for fact in facts:
        acquired = fact.proof.node.notification
        acquired_wait = next(
            m.wait
            for m in model.consumptions
            if m.wait.occurrence.actor is fact.target.actor
            and acquired.name in m.source.occurrence.operation.protocol.data.split(", ")
        )
        assert model.happens_before(old_wait.begin, acquired_wait.begin)
    if mixed:
        assert any(fact.engine is None and fact.target.actor is Actor.AIC for fact in facts)
    # These cuts precede the first query's target. The intermediate acquire
    # proofs, rather than the final conclusion, must justify their facts.
    relays = [
        n
        for n in domains.notifications
        if n.source_engine in ({Engine.MTE2, Engine.M} if mixed else {Engine.M, Engine.FIX})
        and n.sources[0].actor is Actor.AIC
        and rule.positions.order(n.sources[0], target, strict=True) is True
        and model.happens_before(
            old_wait.begin,
            next(
                m.source.end
                for m in model.consumptions
                if m.source.occurrence.actor is Actor.AIC
                and n.name in m.source.occurrence.operation.protocol.data.split(", ")
            ),
        )
    ]
    assert {n.source_engine for n in relays} == (
        {Engine.MTE2, Engine.M} if mixed else {Engine.M, Engine.FIX}
    )
    for relay in relays:
        assert rule.establish(
            earlier,
            old,
            relay,
            relay.sources[0],
            assumptions,
            loop=None,
        )
        assert domains.work.consumption_queries == queries


def test_queue_relay_facts_do_not_change_the_native_only_policy():
    plan = synchronize_structure(cube_chain())
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[0], messages[-1]
    old, target = earlier.consumers[0], later.sources[0]
    assumptions = old.guard.comparisons + target.guard.comparisons
    rule = HandbackRule(domains)
    assert rule.establish(earlier, old, later, target, assumptions, loop=None)
    # A native-only derivation may independently prove the same relation.
    # Its policy cannot be supplied by a stored queue-composition derivation.
    origin = ConsumptionOrigin(old, earlier)
    for queue_composition in (False, True, False):
        result = rule.frontiers.find(
            origin,
            target,
            later.source_engine,
            assumptions,
            queue_composition=queue_composition,
        )
        assert (result is not None) == queue_composition
    check_compiled_events(plan)


@pytest.mark.parametrize("complete", [False, True])
def test_acquire_exports_the_union_of_its_proved_domains(complete):
    plan = synchronize_structure(cube_chain())
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[0], messages[-1]
    old, target = earlier.consumers[0], later.sources[0]
    rule = HandbackRule(domains)
    assert rule.establish(earlier, old, later, target, (), loop=None)
    fact = min(
        (
            word.example
            for word in rule.frontiers.words.values()
            if word.example.engine is Engine.M and word.example.after_action
        ),
        key=lambda fact: rule.positions.position(fact.target)[-1][2],
    )
    target = next(
        n.sources[0]
        for n in domains.notifications
        if n.source_engine is Engine.M and n.sources[0].operation is fact.target.operation
    )
    # Restrict an actual acquired proof to two disjoint mathematical domains.
    # Each restriction remains sound, but only their union is unconditional.
    selector = Expr("argument", ("acquire_case",))
    parts = tuple(
        replace(
            fact.proof,
            condition=all_of(fact.proof.condition, Comparison(selector, predicate, literal(0))),
        )
        for predicate in (("==", "!=") if complete else ("==",))
    )
    condition = any_of(*(proof.condition for proof in parts))
    proof = CausalProof(TransferNode("union", None, None, (), condition), parts, condition)
    frontiers = ConsumptionFrontiers(domains, rule.positions)
    frontiers.add_acquisitions(fact.origin, fact.premises, proof, queue_composition=True)
    result = frontiers.find(fact.origin, target, Engine.M, fact.premises)
    assert (result is not None) == complete
    check_compiled_events(plan)
