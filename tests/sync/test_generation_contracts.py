"""Rule premises and finite counterexamples, independent of old solvers."""

from dataclasses import replace

import pytest

from ascendflow.analysis.queue_order import establish_queue_order
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_protocols import NotificationKind
from ascendflow.analysis.structured_schedule import Placement, SyncAction
from ascendflow.diagnostics import CompileError
from ascendflow.passes.generate_notifications import _complete_bindings
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import literal
from ascendflow.target.ascend910b import Actor, Engine, SyncKind
from tests.fixtures.nested import partitioned_cv
from tests.fixtures.notification_cycles import (
    PARENT_REUSE,
    actual_port_cycle,
    alternating_queues,
    child_to_parent,
    collective_children,
    finite_ports,
    local_child_route,
    mode2_participants,
    nested_channel_cycle,
    prologue_stream_epilogue,
    sibling_ports,
    translated_child_route,
    translated_commands,
)
from tests.models.structured_events import (
    EventModelError,
    _causality,
    check_compiled_events,
    check_structured_events,
)
from tests.support.structured import prepared, selected


def order(selection):
    return establish_queue_order(prepared(selection))


@pytest.mark.parametrize("source", [nested_channel_cycle, alternating_queues])
def test_nested_or_alternating_queue_cycles_cannot_be_dropped(source):
    with pytest.raises(CompileError):
        order(source())


def test_pairing_does_not_admit_both_controllers_waiting_before_their_sets():
    selection = mode2_participants(reverse=False)
    (forward,) = selection.notifications
    reverse = replace(
        forward,
        name="reverse",
        sources=tuple(replace(p, placement=Placement.AFTER) for p in forward.consumers),
        consumers=tuple(replace(p, placement=Placement.BEFORE) for p in forward.sources),
        source_engine=forward.destination_engine,
        destination_engine=forward.source_engine,
    )
    selection = replace(selection, notifications=(forward, reverse))
    domains = prepared(selection)  # Counts, domains and complete quorum pair.
    with pytest.raises(CompileError, match="cycle"):
        establish_queue_order(domains)
    actions = []
    for number, channel in enumerate(selection.notifications):
        for bounds, kind in (
            (channel.sources, SyncKind.CROSS_SET),
            (channel.consumers, SyncKind.CROSS_WAIT),
        ):
            for point in bounds:
                actions.append(
                    SyncAction(
                        point.operation,
                        point.placement,
                        (point.actor,),
                        kind,
                        channel.source_engine,
                        channel.destination_engine,
                        literal(number),
                    )
                )
    materialize_notifications(domains.memory, actions)
    with pytest.raises(EventModelError, match="controller command deadlock"):
        check_structured_events(domains.memory, {"n": 1})


def test_closed_children_keep_opposite_controller_entry_orders():
    with pytest.raises(CompileError, match="cycle"):
        order(collective_children(reverse_vector=True))
    accepted = order(collective_children(reverse_vector=False))
    assert accepted.root.children


@pytest.mark.parametrize("scalar", [True, False])
def test_only_a_controller_wait_blocks_unrelated_parent_queues(scalar):
    if scalar:
        with pytest.raises(CompileError):
            order(child_to_parent(scalar=True))
    else:
        assert order(child_to_parent(scalar=False)).root.children
        assert order(local_child_route(vector_publication=False)).root.children
    with pytest.raises(CompileError):
        order(local_child_route(vector_publication=True))


def test_symbolic_last_and_short_sibling_loops_retain_actual_boundaries():
    assert not actual_port_cycle(finite_ports(), 1)
    assert actual_port_cycle(finite_ports(), 2)
    for fixture in (finite_ports, sibling_ports):
        with pytest.raises(CompileError):
            order(fixture())
        assert order(fixture(at_most_one=True)).root.children


def test_startup_and_epilogue_compose_without_inventing_loop_completion():
    assert order(prologue_stream_epilogue()).root.children
    with pytest.raises(CompileError):
        order(prologue_stream_epilogue(cyclic=True))


@pytest.mark.parametrize("count", [1, 2])
def test_distinct_parent_invocations_match_independent_command_causality(count):
    selection, parent, child = translated_child_route(count)
    commands, pairs = translated_commands(selection, parent, child, count)
    assert len(pairs["previous"]) == count - 1
    if count == 1:
        _causality(commands)
        order(selection)
    else:
        with pytest.raises(EventModelError, match="deadlock"):
            _causality(commands)
        with pytest.raises(CompileError):
            order(selection)


@pytest.fixture(scope="module")
def parent_credit():
    from ascendflow import language as N

    return synchronize_structure(N.from_source(PARENT_REUSE, disjoint_external=True))


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_physical_cell_credit_spans_child_and_parent_invocations(parent_credit, count):
    assert parent_credit.generation.channels
    assert not parent_credit.generation.order.root.ports
    result = check_structured_events(
        collect_native_memory(parent_credit.module), {"n": count}, invocations=2
    )
    assert len(result.natives) == count * 8


def test_allocation_rejects_coverage_only_instead_of_trusting_a_label():
    from ascendflow.analysis.event_planning import plan_event_resources
    from ascendflow.analysis.structured_resources import allocate_structured_events
    from examples.vector.scale import scale

    selection = selected(scale)
    with pytest.raises(CompileError, match="virtual event resource problem"):
        allocate_structured_events(selection)
    with pytest.raises(CompileError, match="generated notification lifetimes"):
        plan_event_resources(selection)
    assert allocate_structured_events(
        plan_event_resources(_complete_bindings(selection))
    ).assignments


def test_generated_counter_quorum_retains_two_private_actors():
    from tests.fixtures.nested import nested_cv

    program = _complete_bindings(selected(nested_cv))
    channels = [
        c for c in program.channels if c.members[0].mechanism is NotificationKind.CROSS_EVENT
    ]
    assert channels
    for channel in channels:
        item = channel.members[0]
        vector = item.sources if len(item.sources) == 2 else item.consumers
        assert {p.actor for p in vector} == {Actor.AIV0, Actor.AIV1}
        assert vector[0].operation is vector[1].operation


def test_partitioned_cube_prefix_and_acquire_keep_all_real_reader_versions():
    plan = synchronize_structure(partitioned_cv)
    for count in (0, 1, 2, 3, 5):
        execution = check_compiled_events(plan, {"n": count}, groups=(0,))
        assert count == 0 or execution.bytes_read and execution.bytes_written


def finite_independent_cells(*, scalar):
    from ascendflow import language as N

    declarations = "\n".join(f"        x{i}=N.alloc_ub((64,))" for i in range(9))
    body = "\n".join(
        f"        N.fill(x{i},1.0)\n"
        + (f"        N.load_scalar(x{i}[:1])" if scalar else f"        N.copy(x{i},O[{i},:])")
        for i in range(9)
    )
    return N.from_source(
        'def finite(O:N.Tensor((9,64),N.f32,access="write")):\n'
        + "    with N.Kernel():\n"
        + declarations
        + "\n"
        + body
        + "\n",
        disjoint_external=True,
    )


def test_new_id_sharing_requires_a_real_consumption_handback():
    # Scalar Wait blocks its controller before the next Vector Set. Distinct
    # storage credits can therefore share one ID via a *new* finite constraint.
    plan = synchronize_structure(finite_independent_cells(scalar=True))
    assert plan.resources.sharing
    check_compiled_events(plan)
    # A queued MTE3 Wait has no corresponding handback to V. Lexical source
    # order and equal flag counts cannot justify overwriting its binary flag.
    with pytest.raises(CompileError, match="capacity 8"):
        synchronize_structure(finite_independent_cells(scalar=False))


def test_optional_sharing_knowledge_does_not_change_an_unpressured_kernel(monkeypatch):
    from ascendflow import compile
    from ascendflow import language as N
    from ascendflow.analysis import event_planning

    source = N.from_source(
        """
def exclusive():
    with N.Kernel(1,kind="vector"):
        cells=N.alloc_ub((3,64),N.f32)
        for panel in N.unroll(3):
            N.fill(cells[panel,:],1.0)
            N.load_scalar(cells[panel,0:1])
"""
    )
    baseline = compile(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("optional sharing must not participate without resource pressure")

    monkeypatch.setattr(event_planning, "share_events", forbidden)
    monkeypatch.setattr(event_planning, "plan_counter_groups", forbidden)
    current = compile(source)
    assert current.asc == baseline.asc
    assert current.report["analysis_work"]["resources"]["sharing_pairs"] == 0


def test_optional_queue_handbacks_do_not_choose_a_smaller_unpressured_ring(monkeypatch):
    from ascendflow import compile
    from ascendflow.analysis.notification_lifetimes import HandbackRule
    from examples.gemm.hierarchical import hierarchical_gemm

    source = hierarchical_gemm.specialize(M=16, N=16, BK=16, GROUPS=1)
    baseline = compile(source)
    assert any(s["width"] == 2 for s in baseline.report["event_resources"]["streams"])

    def forbidden(*args, **kwargs):
        raise AssertionError("optional protocol proofs cannot choose an unpressured ring")

    monkeypatch.setattr(HandbackRule, "smaller_lease", forbidden)
    current = compile(source)
    assert current.asc == baseline.asc
    selection = current.report["event_resources"]["protocol_selection"]
    assert selection == {
        "candidates": 1,
        "selected_option": 0,
        "changed_streams": [],
        "preferred_changes": [],
    }


def test_finite_fragment_sharing_also_checks_its_later_loop():
    from ascendflow import language as N

    source = finite_independent_cells(scalar=True).source
    source = source.replace(
        'def finite(O:N.Tensor((9,64),N.f32,access="write")):',
        'def finite(O:N.Tensor((9,64),N.f32,access="write"),n:N.i64):',
    )
    source += "        for j in N.Pipelined(n):\n            N.fill(x0,2.0)\n            N.load_scalar(x0[:1])\n"
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    channel = next(c for c in plan.generation.channels if len(c.members) > 1)
    ids = {
        i for a in plan.resources.assignments if a.notification in channel.members for i in a.ids
    }
    assert len(ids) == 1
    from tests.support.structured import shared_message_pairs

    shared = set(shared_message_pairs(plan))
    for assignment in plan.resources.assignments:
        if assignment.notification not in channel.members and ids.intersection(assignment.ids):
            # A proof for the startup member cannot license the later loop.
            assert all(
                frozenset((member, assignment.notification)) in shared for member in channel.members
            )
    for count in (0, 1, 3):
        check_compiled_events(plan, {"n": count}, invocations=2)


def test_a_proved_absent_domain_does_not_consume_finite_event_capacity():
    from ascendflow import language as N

    source = finite_independent_cells(scalar=False).source
    source = source.replace(
        'def finite(O:N.Tensor((9,64),N.f32,access="write")):',
        'def finite(O:N.Tensor((9,64),N.f32,access="write"),n:N.i64):',
    )
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "N.fill" in line)
    source = (
        "\n".join(lines[:start] + ["        if n < 0:"] + ["    " + line for line in lines[start:]])
        + "\n"
    )
    plan = synchronize_structure(N.from_source(source, disjoint_external=True))
    assert not any(a.ids for a in plan.resources.assignments)
    assert not check_compiled_events(plan, {"n": 3}).natives


def test_allocation_consumes_the_existing_child_boundary_membership(monkeypatch):
    from ascendflow.analysis import event_families
    from ascendflow.analysis.event_planning import plan_event_resources
    from ascendflow.analysis.structured_resources import allocate_structured_events
    from tests.fixtures.nested import nested_copy

    program = _complete_bindings(selected(nested_copy))
    assert program.handbacks.boundary_cases
    established = dict(program.handbacks.boundary_cases)
    original = event_families.build_event_family

    def only_new_premises(domains, notification):
        assert notification not in established, "unchanged child membership was rebuilt"
        return original(domains, notification)

    monkeypatch.setattr(event_families, "build_event_family", only_new_premises)
    from ascendflow.analysis.notification_lifetimes import HandbackRule

    def no_reconstruction(*args, **kwargs):
        raise AssertionError("resource planning must retain the established consumption analysis")

    monkeypatch.setattr(HandbackRule, "__init__", no_reconstruction)
    problem = plan_event_resources(program)

    def no_queries(*args, **kwargs):
        raise AssertionError("numeric assignment must consume frozen resource premises")

    monkeypatch.setattr(event_families, "build_event_family", no_queries)
    monkeypatch.setattr(type(program.domains), "_following_conditions", no_queries)
    allocate_structured_events(problem)


def test_configuration_and_final_drain_have_explicit_unborrowed_bank_entries():
    from ascendflow.analysis.event_planning import plan_event_resources
    from ascendflow.analysis.structured_resources import allocate_structured_events
    from examples.gemm.batched import batched_matmul

    program = _complete_bindings(selected(batched_matmul.specialize(M=16, N=16, K=16, GROUPS=1)))
    plan = allocate_structured_events(plan_event_resources(program))
    configuration = [r for r in plan.reservations if r.bank[1] is Engine.S]
    drains = [r for r in plan.reservations if r.bank[2] is Engine.S]
    assert configuration and drains
    for reservation in configuration + drains:
        assert reservation.event == (0 if reservation in configuration else 7)
        assert all(
            reservation.event not in a.ids for a in plan.assignments if reservation.bank in a.banks
        )
    for reservation in drains:
        actions = [
            a
            for a in plan.actions
            if a.anchor is reservation.anchor
            and a.placement is Placement.EXIT
            and a.event == literal(reservation.event)
            and reservation.bank[0] in a.participants
            and (a.source, a.destination) == reservation.bank[1:]
        ]
        assert {a.kind for a in actions} == {SyncKind.LOCAL_SET, SyncKind.LOCAL_WAIT}
