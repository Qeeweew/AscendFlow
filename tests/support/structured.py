"""Shared construction of production structured-analysis inputs for tests.

These helpers run the real frontend, native admission, dependency analysis,
notification selection and event-domain construction. They are setup helpers,
not independent correctness oracles; importing this module creates no fixture.
"""

from xdsl_npu import language as N
from xdsl_npu.analysis.event_contracts import ResourceWork
from xdsl_npu.analysis.event_domains import EventDomains
from xdsl_npu.analysis.structured_dependencies import build_structured_dependencies
from xdsl_npu.analysis.structured_dispatch import bind_structured_dispatch
from xdsl_npu.analysis.structured_memory import collect_native_memory
from xdsl_npu.passes.lower_native import lower_native_structure
from xdsl_npu.passes.select_notifications import select_notifications


def selected(source):
    structure = lower_native_structure(N.from_source(source) if isinstance(source, str) else source)
    memory = collect_native_memory(structure.module)
    dependencies = build_structured_dependencies(memory)
    dispatch = bind_structured_dispatch(memory)
    result = select_notifications(dependencies, dispatch)
    result.require_closed()
    return result


def selection(source):
    source = N.from_source(source) if isinstance(source, str) else source
    memory = collect_native_memory(lower_native_structure(source).module)
    dependencies = build_structured_dependencies(memory)
    dependencies.require_closed()
    result = select_notifications(dependencies, bind_structured_dispatch(memory))
    result.require_closed()
    return result


def prepared(selected):
    return EventDomains(selected, ResourceWork(notifications=len(selected.notifications)))
