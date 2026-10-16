"""Shared setup for notification-selection and event-domain tests.

Runs the real frontend/native lowering, memory collection, dependency analysis
and notification selection; prepared additionally constructs event domains.
These helpers run neither complete memory admission nor executable compilation
and are not independent correctness oracles. Importing creates no fixture.
"""

from xdsl_npu import language as N
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.structured_dependencies import build_structured_dependencies
from xdsl_npu.analysis.structured_memory import collect_native_memory
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.passes.select_notifications import select_notifications


def selected(source):
    structure = lower_native_structure(N.from_source(source) if isinstance(source, str) else source)
    memory = collect_native_memory(structure.module)
    dependencies = build_structured_dependencies(memory)
    result = select_notifications(dependencies, dependencies.dispatch)
    result.require_closed()
    return result


def prepared(selected):
    return EventDomains(selected, ResourceWork(notifications=len(selected.notifications)))
