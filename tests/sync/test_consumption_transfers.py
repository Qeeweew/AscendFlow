"""Real Wait consumption crosses Cube queues through the common transfer DAG."""

import pytest

from ascendflow import language as N
from ascendflow.analysis.event_coloring import NoEventAssignment
from ascendflow.analysis.notification_lifetimes import HandbackRule
from ascendflow.analysis.protocol_transfers import ProtocolTransfers
from ascendflow.analysis.structured_schedule import Comparison
from ascendflow.dialects.structured import NotificationOp
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import Expr, literal
from ascendflow.target.ascend910b import Engine
from tests.models.structured_events import EventModelError, check_compiled_events


def cube_chain(phases=3, *, feedback=True):
    return N.from_source(
        f"""
def consumption_chain(A:N.Tensor((16,16),N.bf16),B:N.Tensor((16,16),N.bf16),
                      Work:N.Tensor(({phases},16,16),N.bf16,access="write")):
    with N.Kernel(kind="cube"):
        a1=N.alloc_l1((16,16),N.bf16,slots={phases})
        b1=N.alloc_l1((16,16),N.bf16,slots={phases})
        a0=N.alloc_l0a((16,16),N.bf16,slots={phases})
        b0=N.alloc_l0b((16,16),N.bf16,slots={phases})
        c0=N.alloc_l0c((16,16),N.f32,slots={phases})
        for phase in N.unroll({phases}):
            if phase>0:
                av=N.copy({"Work[phase-1,:,:]" if feedback else "A"},a1.at(phase))
            else:
                av=N.copy(A,a1.at(phase))
            bv=N.copy(B,b1.at(phase))
            aa=N.copy(av,a0.at(phase))
            bb=N.copy(bv,b0.at(phase),transpose=True)
            N.matmul(aa,bb,accum=c0.at(phase),initialize=True)
            N.copy(c0.at(phase),Work[phase,:,:])
""",
        disjoint_external=True,
    )


def proof_protocols(proof):
    seen, protocols, pending = set(), set(), [proof]
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        if item.node.notification is not None:
            protocols.add(item.node.notification)
        pending.extend(item.inputs)
    return protocols


def test_three_real_queue_handoffs_carry_consumption_without_a_relay_depth_limit():
    plan = synchronize_structure(cube_chain())
    domains = plan.generation.domains
    graph = ProtocolTransfers(domains)
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[0], messages[-1]
    old, target = earlier.consumers[0], later.sources[0]
    assumptions = tuple(
        test for p in (old, target) for test in (*p.guard.comparisons, *domains._input_ranges(p))
    )
    proof = graph.consumption(earlier, old, target, later.source_engine, assumptions=assumptions)
    assert proof is not None
    assert domains.arithmetic.queries.implies(proof.condition, assumptions).proved
    assert {n.source_engine for n in proof_protocols(proof)} >= {Engine.MTE1, Engine.M, Engine.FIX}
    assert earlier not in proof_protocols(proof)
    model = check_compiled_events(plan)
    uses = [
        m
        for m in model.consumptions
        if m.source.source is Engine.MTE2 and m.source.destination is Engine.MTE1
    ]
    pairs = [(a.wait, b.source) for a in uses for b in uses if a.wait.issue < b.source.issue]
    assert pairs
    assert all(model.happens_before(wait.begin, send.end) for wait, send in pairs)


def test_nine_cube_phases_reuse_binary_ids_through_three_real_handoffs():
    plan = synchronize_structure(cube_chain(9))
    assert plan.resources.sharing
    assert plan.generation.domains.work.consumption_queries
    assert plan.generation.domains.work.consumption_regions
    check_compiled_events(plan, invocations=2)


def test_consumption_words_reuse_only_exact_mathematical_questions():
    plan = synchronize_structure(cube_chain())
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[0], messages[-1]
    old, target = earlier.consumers[0], later.sources[0]
    # Add an explicit conditional domain to a real established handback. Its
    # proof cannot supply an unconditional conclusion when the domain weakens.
    selector = Expr("argument", ("frontier_selector",))
    active = Comparison(selector, "==", literal(1))
    assumptions = (active,)
    rule = HandbackRule(domains)
    assert rule.establish(earlier, old, later, target, assumptions, loop=None)
    origin = next(iter(rule.frontiers.words.values())).example.origin
    work = domains.work
    for context, expected in ((assumptions, True), ((), False), (assumptions, True)):
        result = rule.frontiers.find(origin, target, later.source_engine, context)
        assert (result is not None) is expected
        queries = work.consumption_frontier_queries
        for _ in range(16):
            repeated = rule.frontiers.find(origin, target, later.source_engine, context)
            assert (repeated is not None) is expected
        assert work.consumption_frontier_queries == queries
        if result is not None:
            proof, answer = result
            assert domains.arithmetic.queries.owns_implication(answer, proof.condition, context)
            assert earlier not in proof_protocols(proof)
    assert work.consumption_frontier_answer_reuses >= 48
    assert rule.frontiers.find(origin, earlier.sources[0], later.source_engine, assumptions) is None
    check_compiled_events(plan, invocations=2)


def test_three_handoff_consumption_requires_the_actual_feedback_gate():
    plan = synchronize_structure(cube_chain(feedback=False))
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[0], messages[2]
    old, target = earlier.consumers[0], later.sources[0]
    assumptions = old.guard.comparisons + target.guard.comparisons
    proof = ProtocolTransfers(domains).consumption(
        earlier, old, target, later.source_engine, assumptions=assumptions
    )
    assert (
        proof is None or not domains.arithmetic.queries.implies(proof.condition, assumptions).proved
    )
    model = check_compiled_events(plan)
    messages = [
        m
        for m in model.consumptions
        if m.source.source is Engine.MTE2 and m.source.destination is Engine.MTE1
    ]
    assert not model.happens_before(messages[0].wait.begin, messages[2].source.end)
    with pytest.raises(NoEventAssignment, match="capacity 8"):
        synchronize_structure(cube_chain(9, feedback=False))


def test_capture_before_the_old_wait_cannot_return_its_later_consumption():
    plan = synchronize_structure(cube_chain())
    domains = plan.generation.domains
    messages = [
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE2 and n.destination_engine is Engine.MTE1
    ]
    earlier, later = messages[1], messages[2]
    old, target = earlier.consumers[0], later.sources[0]
    relay = next(
        n
        for n in domains.notifications
        if n.source_engine is Engine.MTE1
        and n.destination_engine is Engine.M
        and any(p.operation is old.operation for p in n.sources)
    )
    original = ProtocolTransfers(domains)
    graph = ProtocolTransfers(
        domains, (episode for episode in original._episodes() if episode[0] is not relay)
    )
    assumptions = old.guard.comparisons + target.guard.comparisons
    proof = graph.consumption(earlier, old, target, later.source_engine, assumptions=assumptions)
    assert (
        proof is None or not domains.arithmetic.queries.implies(proof.condition, assumptions).proved
    )
    check_compiled_events(plan)
    damaged = [
        operation
        for operation in plan.module.walk()
        if isinstance(operation, NotificationOp)
        and relay.name in operation.protocol.data.split(", ")
    ]
    assert len(damaged) == 2
    for operation in damaged:
        operation.parent.erase_op(operation)
    with pytest.raises(EventModelError):
        check_compiled_events(plan)
