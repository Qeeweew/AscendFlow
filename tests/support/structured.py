"""Shared setup for notification-selection and event-domain tests.

Runs the real frontend/native lowering, memory collection, dependency analysis
and notification selection; prepared additionally constructs event domains.
These helpers run neither complete memory admission nor executable compilation
and are not independent correctness oracles. Importing creates no fixture.
"""

from ascendflow import language as N
from ascendflow.analysis.event_contracts import ResourceWork
from ascendflow.analysis.event_domains import EventDomains
from ascendflow.analysis.structured_dependencies import build_structured_dependencies
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.passes.notification_binding import bind_notifications


def selected(source):
    structure = lower_native_structure(N.from_source(source) if isinstance(source, str) else source)
    memory = collect_native_memory(structure.module)
    dependencies = build_structured_dependencies(memory)
    result = bind_notifications(dependencies, dependencies.dispatch)
    result.require_closed()
    return result


def prepared(selected):
    return EventDomains(selected, ResourceWork(notifications=len(selected.notifications)))
