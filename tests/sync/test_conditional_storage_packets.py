"""Independent scratch invocations retain ownership through skipped slots.

The finite oracle reads actual addresses, publications, waits and byte
versions. It does not use the packet partition or its closure witness.
"""

from dataclasses import replace

import pytest

from ascendflow import language as N
from ascendflow.analysis.access_frontier import QueueEndpoint
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.analysis.structured_resources import allocate_structured_events
from ascendflow.passes.generate_notifications import _complete_bindings
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.materialize_structure import materialize_notifications
from ascendflow.passes.notification_binding import bind_notifications
from ascendflow.passes.structured_sync import synchronize_structure
from ascendflow.scalar import literal
from tests.models.structured_events import (
    EventModelError,
    check_compiled_events,
    check_structured_events,
)
from tests.support.structured import native_memory

SCRATCH_CALLS = """
def scratch_calls(X:N.Tensor((n,2,4,64),N.f32),
                  Out:N.Tensor((n,2,4,64),N.f32,access="write",write_disjoint=True),
                  n:N.int64):
    with N.Kernel(1,kind="vector"):
        scratch=N.alloc_ub((2,64),N.f32)
        with N.Vector():
            for parent in N.Pipelined(n,issue_order=("first","second")):
                with N.Stage(0,name="first"):
                    for inner in N.Pipelined(2):
                        with N.Stage(0,name="load_first"):
                            N.copy(X[parent,0,inner,:],scratch[inner%2,:])
                        with N.Stage(1,name="store_first"):
                            N.copy(scratch[inner%2,:],Out[parent,0,inner,:])
                with N.Stage(1,name="second"):
                    for inner in N.Pipelined(parent%4):
                        with N.Stage(0,name="load_second"):
                            N.copy(X[parent,1,inner,:],scratch[inner%2,:])
                        with N.Stage(1,name="store_second"):
                            N.copy(scratch[inner%2,:],Out[parent,1,inner,:])
"""


@pytest.fixture(scope="module", params=("parent%4", "(parent//3)%3"))
def compiled_calls(request):
    # The second case leaves both slots unvisited for consecutive calls.
    source = SCRATCH_CALLS.replace("parent%4", request.param)
    return synchronize_structure(N.from_source(source, disjoint_external=True))


@pytest.mark.parametrize("n", (0, 1, 2, 3, 4, 7, 10))
def test_closed_calls_keep_real_prefixes_across_empty_and_short_children(compiled_calls, n):
    packets = compiled_calls.dependencies.order.packet_partitions
    assert any(packet.closed_invocations for partition in packets for packet in partition.packets)
    assert len(compiled_calls.generation.credits) == 2
    check_compiled_events(compiled_calls, {"n": n}, invocations=2)


def test_scratch_value_escaping_a_skipped_child_prevents_packet_reordering():
    source = SCRATCH_CALLS.replace(
        "n:N.int64):", 'Tail:N.Tensor((64,),N.f32,access="write"), n:N.int64):'
    )
    source += "\n            N.copy(scratch[1,:],Tail[:])\n"
    memory = collect_native_memory(lower_native_structure(N.from_source(source)).module)
    dependencies = build_structured_dependencies(memory)
    # The last observable value depends on which optional invocation ran.
    # Its cell must retain source-order requirements, even though every
    # nonempty child itself starts with a complete write.
    partitions = dependencies.order.packet_partitions
    assert not any(partition.cell.start == 256 for partition in partitions)


def test_closed_prefixes_do_not_license_one_binary_id_for_a_whole_cell():
    # Empty prefixes forward queue history without reading the newly written
    # value. Unlike a native reader, that forwarding does not itself return
    # the consumption of an earlier flag. Remove only this lifetime protection
    # by coalescing the actual ready instructions for one physical cell.
    memory = native_memory(N.from_source(SCRATCH_CALLS, disjoint_external=True))
    dependencies = build_structured_dependencies(memory)
    # Exercise the general forwarding-prefix rule without taking the new
    # optional closed-round precision. All original requirements remain.
    dependencies = replace(dependencies, order=replace(dependencies.order, cell_lifetimes=()))
    selection = bind_notifications(dependencies, dependencies.dispatch)
    resources = allocate_structured_events(_complete_bindings(selection))
    groups = {}
    for assignment in resources.assignments:
        item = assignment.notification
        if not item.origins:
            continue
        key = (
            frozenset(origin.ownership_key for origin in item.origins),
            item.source_engine,
            item.destination_engine,
        )
        groups.setdefault(key, []).append(assignment)
    assert any(
        isinstance(point.occurrence.endpoint, QueueEndpoint)
        for items in groups.values()
        for assignment in items
        for point in assignment.notification.sources + assignment.notification.consumers
    )
    ids, banks = {}, {}
    for items in groups.values():
        bank = items[0].banks
        event = banks.get(bank, 0)
        banks[bank] = event + 1
        assert event < 8
        ids.update((assignment.notification.name, event) for assignment in items)
    memory = selection.dependencies.memory
    original = collect_native_memory(memory.module.clone())
    actions = tuple(
        replace(action, event=literal(ids[action.protocols[0]]))
        if len(action.protocols) == 1 and action.protocols[0] in ids
        else action
        for action in resources.actions
    )
    materialize_notifications(memory, actions)
    with pytest.raises(EventModelError, match="binary event overwritten"):
        check_structured_events(
            collect_native_memory(memory.module),
            {"n": 10},
            logical_reference=original,
            invocations=2,
        )
