"""Shared setup for notification-selection and event-domain tests.

Runs the real frontend/native lowering, memory collection, dependency analysis
and notification selection; prepared additionally constructs event domains.
These helpers run neither complete memory admission nor executable compilation
and are not independent correctness oracles. Importing creates no fixture.
"""

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from ascendflow import language as N
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.notification_binding import bind_notifications


def index_constant(block, value):
    operation = arith.ConstantOp.from_int_and_width(value, IndexType())
    block.add_op(operation)
    return operation.result


def native_memory(source):
    return collect_native_memory(lower_native_structure(source).module)


def selected(source):
    memory = native_memory(N.from_source(source) if isinstance(source, str) else source)
    dependencies = build_structured_dependencies(memory)
    result = bind_notifications(dependencies, dependencies.dispatch)
    result.require_closed()
    return result


def prepared(selected):
    return EventDomains(selected, ResourceWork(notifications=len(selected.notifications)))


def allocated(program):
    """Use both production stages; coverage alone cannot bypass generation."""
    from ascendflow.analysis.event_planning import plan_event_resources
    from ascendflow.analysis.structured_resources import allocate_structured_events

    return allocate_structured_events(plan_event_resources(program))


def shared_message_pairs(plan):
    """Test-only expansion of frozen local rectangles to real message pairs."""
    from ascendflow.analysis.event_frontier_contracts import LocalEventReuse

    for witness in plan.resources.sharing:
        if not isinstance(witness, LocalEventReuse):
            yield frozenset((witness.earlier, witness.later))
            continue
        credits = dict(witness.credits)
        for rectangle in witness.rectangles:
            left = [r for r in credits if rectangle.left & (1 << r)]
            right = [r for r in credits if rectangle.right & (1 << r)]
            for a in left:
                for b in right:
                    for x in credits[a]:
                        for y in credits[b]:
                            assert any(
                                x in p.left.members
                                and y in p.right.members
                                or y in p.left.members
                                and x in p.right.members
                                for p in rectangle.certificate.interfaces
                            ), "compressed credit certificate omitted an actual message pair"
                            yield frozenset((x, y))
