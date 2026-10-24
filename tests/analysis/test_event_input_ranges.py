"""Endpoint input facts stay local to the real load and captured invocation."""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_frontier import AccessEndpoint
from ascendflow.analysis.access_order import build_access_order
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_protocols import BoundPlacement, SynchronizationSelection
from ascendflow.analysis.structured_regions import AccessOccurrence, LoopBinding
from ascendflow.analysis.structured_schedule import ActionGuard, Comparison, Placement
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import expression, literal, substitute
from ascendflow.target.ascend910b import Actor


@pytest.fixture(scope="module")
def inputs():
    source = N.from_source("""
def input_scopes(First:N.Tensor((n,),N.i64),Second:N.Tensor((n,8),N.i64),n:N.i64):
    with N.Kernel(1,kind="vector"):
        x=N.alloc_ub((64,),N.f32)
        for i in N.serial(n):
            size=N.load_index(First[i:i+1],minimum=1,maximum=8)
            for row in N.serial(size):
                late=N.load_index(Second[i,row:row+1],minimum=2,maximum=3)
                N.fill(x,0.0)
""")
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_access_order(memory)
    selection = SynchronizationSelection(replace(dependencies, required=()), (), ())
    domains = EventDomains(selection, ResourceWork())
    loads = [t for t in memory.templates if t.spec.native == "index_load"]
    fill = next(t for t in memory.templates if t.spec.native == "fill")
    return domains, loads, fill


def point(domains, template, placement, shift=0, actor=Actor.AIV0):
    frame = template.frame
    coordinates = tuple((loop.operation, loop.ordinal) for loop in frame.loops())
    if shift:
        outer = coordinates[0][1]
        coordinates = tuple(
            (loop, substitute(value, {outer: expression("add", outer, literal(shift))}))
            for loop, value in coordinates
        )
    occurrence = AccessOccurrence(
        AccessEndpoint.from_template(template, actor),
        tuple(LoopBinding(loop, value) for loop, value in coordinates),
    )
    return BoundPlacement(occurrence, placement, ActionGuard(), (), (), (), coordinates)


def test_load_range_is_available_after_the_load_and_not_before_it(inputs):
    domains, loads, _ = inputs
    first, second = loads
    assert domains._input_ranges(point(domains, first, Placement.BEFORE)) == ()
    after_first = domains._input_ranges(point(domains, first, Placement.AFTER))
    before_second = domains._input_ranges(point(domains, second, Placement.BEFORE))
    after_second = domains._input_ranges(point(domains, second, Placement.AFTER))
    assert after_first == before_second
    assert len(after_second) > len(before_second)
    assert all(not name.captures for name in domains.facts)


def test_next_invocation_keeps_an_independent_loaded_value(inputs):
    domains, loads, fill = inputs
    current = point(domains, fill, Placement.BEFORE)
    following = point(domains, fill, Placement.BEFORE, shift=1)
    known = domains._input_ranges(current) + domains._input_ranges(following)
    value = domains.control.environment.expression(loads[0].operation.value)
    parent = domains.control.loops[current.coordinates[0][0]].ordinal
    left = substitute(value, {domains.control.sid: literal(0)})
    right = substitute(
        value, {parent: expression("add", parent, literal(1)), domains.control.sid: literal(0)}
    )
    # Both loads obey [1,8], but two generations need not read equal data.
    assert domains.arithmetic.queries.satisfies(
        (Comparison(left, "==", literal(1)), Comparison(right, "==", literal(8))), known
    ).satisfied
    assert left != right


def test_two_aivs_keep_independent_metadata_values():
    source = N.from_source("""
def participant_inputs(First:N.Tensor((2,),N.i64)):
    with N.Kernel(1,kind="mixed"):
        x=N.alloc_ub((64,),N.f32)
        with N.Vector() as sid:
            value=N.load_index(First[sid:sid+1],minimum=1,maximum=8)
            N.fill(x,0.0)
""")
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_access_order(memory)
    selection = SynchronizationSelection(replace(dependencies, required=()), (), ())
    domains = EventDomains(selection, ResourceWork())
    load = next(t for t in memory.templates if t.spec.native == "index_load")
    fill = next(t for t in memory.templates if t.spec.native == "fill")
    points = [
        point(domains, fill, Placement.BEFORE, actor=actor) for actor in (Actor.AIV0, Actor.AIV1)
    ]
    value = domains.control.environment.expression(load.operation.value)
    left, right = [substitute(value, {domains.control.sid: literal(sid)}) for sid in (0, 1)]
    known = tuple(test for p in points for test in domains._input_ranges(p))
    assert domains.arithmetic.queries.satisfies(
        (Comparison(left, "==", literal(1)), Comparison(right, "==", literal(8))), known
    ).satisfied


def test_positive_loaded_length_survives_selected_child_handback():
    source = N.from_source(
        """
def positive(Lengths:N.Tensor((n,),N.i64),X:N.Tensor((n,64),N.f32),
             Out:N.Tensor((n,64),N.f32,access="write",write_disjoint=True),n:N.i64):
    with N.Kernel(1,kind="vector"):
        x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
        for i in N.serial(n):
            valid=N.load_index(Lengths[i:i+1],minimum=1,maximum=64)
            for row in N.serial(N.ceildiv(valid,64)):
                N.copy(X[i,:],x)
                N.mul(x,2.0,out=y)
            N.copy(y,Out[i,:])
""",
        disjoint_external=True,
    )
    plan = synchronize_structure(source)
    assert any(record.discharged for record in plan.generation.domains._activations.values())
    assert all(not name.captures for name in plan.generation.domains.facts)
