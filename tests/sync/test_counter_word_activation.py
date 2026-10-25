"""A common actual FIFO word is filtered by each message's proved activation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ascendflow.analysis.event_contracts import CounterFIFOGroup, EventDemand, EventIdPools
from ascendflow.analysis.event_counter_groups import _certificate
from ascendflow.analysis.event_planning import plan_event_resources
from ascendflow.analysis.resource_report import event_resource_report
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.analysis.structured_resources import allocate_event_ids
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison
from ascendflow.diagnostics import CompileError
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine
from tests.fixtures.counter_epochs import counter_epochs
from tests.fixtures.fifo_panels import conditional_panels
from tests.models.structured_events import check_compiled_events
from tests.support.structured import prepared, selected


def test_common_word_with_distinct_runtime_guards_fits_one_real_counter():
    plan = synchronize_structure(conditional_panels, id_pools=EventIdPools(cross=(3,)))
    assert plan.resources.counter_groups
    assert any(isinstance(group, CounterFIFOGroup) for group in plan.resources.sharing)
    for group in plan.resources.counter_groups:
        assert group.event == 3
        certificate = group.certificate
        assert all(word == certificate.words[0][2] for _, _, word in certificate.words)
        assert len({n.sources[0].guard for _, _, n in certificate.messages}) > 1
    report = event_resource_report(plan.resources)
    assert report["counter_groups"] and report["sharing"]
    for count in (0, 1, 2, 4, 5):
        model = check_compiled_events(plan, {"n": count})
        assert count == 0 or model.bytes_read and model.bytes_written


def test_numbering_rejects_a_word_missing_one_selected_member():
    pools = EventIdPools(cross=(3,))
    plan = synchronize_structure(conditional_panels, id_pools=pools)
    problem = plan_event_resources(plan.generation, id_pools=pools)
    object.__setattr__(
        problem,
        "counter_groups",
        tuple(
            replace(c, words=tuple((phase, actor, word[:-1]) for phase, actor, word in c.words))
            for c in problem.counter_groups
        ),
    )
    with pytest.raises(CompileError, match="common complete-credit message sequence"):
        allocate_event_ids(problem)


@pytest.mark.parametrize("members,overlapping", [(8, False), (32, False), (64, False), (8, True)])
def test_tied_word_activation_has_linear_size_and_no_invented_action_order(
    monkeypatch, members, overlapping
):
    # This isolates the word premise on real native endpoints. It deliberately
    # omits source coverage and does not authorize executable compilation.
    selection = selected(counter_epochs.specialize(PHASES=2))
    original = next(
        n
        for n in selection.notifications
        if n.mechanism is NotificationKind.CROSS_EVENT and n.source_engine is Engine.FIX
    )
    original_domains = prepared(selection)
    loop = original_domains.clocks[original][0]
    count = original_domains.control.loops[loop].count
    alternatives = []
    for index in range(members):
        test = Comparison(count, ">=" if overlapping else "==", literal(index + 1))

        def guarded(point, test=test):
            return replace(point, guard=ActionGuard(point.guard.comparisons + (test,)))

        alternatives.append(
            replace(
                original,
                name=f"word_case_{index}",
                requirements=(),
                anchors=(),
                sources=tuple(map(guarded, original.sources)),
                consumers=tuple(map(guarded, original.consumers)),
            )
        )
    selection = replace(
        selection,
        dependencies=replace(selection.dependencies, required=()),
        notifications=tuple(alternatives),
        coverage=(),
        bindings=(),
    )
    domains = prepared(selection)
    demands = tuple(EventDemand(tuple(alternatives[part::2]), 1, (), backlog=1) for part in (0, 1))
    queries = domains.arithmetic.queries
    before = queries.statistics.solver_checks

    def already_paired(*args, **kwargs):
        pytest.fail("FIFO word rebuilt an already paired endpoint activation")

    with monkeypatch.context() as patch:
        patch.setattr(type(domains), "_instance", already_paired)
        certificate = _certificate(
            SimpleNamespace(domains=domains, order=object()), demands, (0, 1)
        )
    assert (certificate is None) == overlapping
    assert queries.statistics.solver_checks - before <= 3
    if certificate is None:
        return
    assert len(certificate.exclusions) == 3
    assert {actor for _, actor, _, _, _ in certificate.exclusions} == {
        Actor.AIC,
        Actor.AIV0,
        Actor.AIV1,
    }
    for _, _, word, overlap, answer in certificate.exclusions:
        assert len(word) == members
        assert queries.owns_implication(answer, False, (overlap,))
        pending, nodes = [overlap], set()
        while pending:
            node = pending.pop()
            if node in nodes:
                continue
            nodes.add(node)
            pending.extend(getattr(node, "operands", ()))
        assert len(nodes) <= 8 * members
