"""One owned DSL -> native facts -> generated credits -> synchronized IR chain.

Generation checks its pairing/phase premises and constructs physical-cell or
queue-handback lifetimes. Allocation consumes them and establishes bank/ID
constraints. Materialization preserves native identities; the backend prints
only actual IR, with no second action-placement interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import ModuleOp

from ..analysis.smt import query_scope, query_statistics
from ..analysis.structured_admission import StructuredAdmission, admit_structured_memory
from ..analysis.structured_bounds import admit_memory_bounds
from ..analysis.structured_dependencies import StructuredDependencies, build_structured_dependencies
from ..analysis.structured_dispatch import bind_structured_dispatch
from ..analysis.structured_memory import collect_native_memory
from ..analysis.structured_protocols import SynchronizationSelection
from ..diagnostics import require
from ..dialects.io import dump
from .executable import ExecutablePlan, NativeExecution
from .lower_execution import lower_execution
from .lower_native import NativeStructure, lower_native_structure
from .materialize_structure import materialize_notifications
from .simplify_conditions import ConditionWork, simplify_conditions
from .simplify_scalars import ScalarFacts, simplify_scalars

if TYPE_CHECKING:
    from ..analysis.structured_resources import StructuredEventPlan
    from .generate_notifications import NotificationProgram


@dataclass(frozen=True, init=False)
class StructuredSynchronization:
    """Terminal execution and retained logical diagnostics on separate modules.

    Structure/admission/dependency/resource records refer to the synchronized
    logical module. Terminal lowering owns a distinct module and explicitly
    maps its native identities and scalar ranges once. The backend receives
    only executable, never these logical diagnostics. The independent pre-sync
    snapshot precedes notification insertion.
    No external verified flag can authorize this result.
    """

    structure: NativeStructure
    admission: StructuredAdmission
    dependencies: StructuredDependencies
    selection: SynchronizationSelection
    resources: StructuredEventPlan
    before_ir: str
    generation: NotificationProgram
    executable: ExecutablePlan
    condition_work: ConditionWork
    timings: tuple[tuple[str, float], ...]
    integer_queries: tuple[dict, ...]

    def __init__(self, *args, **kwargs):
        raise TypeError("use synchronize_structure to establish the executable contracts")

    @property
    def module(self) -> ModuleOp:
        return self.executable.module

    @property
    def bounds(self):
        return self.admission.bounds


def _materialize_owned(structure, admission, dependencies, generation, resources, timings):
    selection = generation.selection
    require(
        admission.structure is structure
        and dependencies.memory is admission.memory
        and selection.dependencies is dependencies
        and resources.selection is selection,
        "synchronization stages belong to different owned structures",
    )
    started = perf_counter()
    before_ir = dump(structure.module)
    frozen = perf_counter()
    logical = materialize_notifications(admission.memory, resources.actions)
    require(logical is structure.module, "notification insertion changed its owned physical module")
    materialized = perf_counter()
    admitted = ScalarFacts.from_control(admission.memory.control, admission.host_expressions)
    values = {}
    module = logical.clone(value_mapper=values)
    operations = dict(zip(logical.walk(), module.walk(), strict=True))
    natives = tuple(
        NativeExecution.from_template(template, operations[template.operation])
        for template in admission.memory.templates
    )
    scalar_facts = ScalarFacts(
        dict(admitted.limits), {values[v]: bounds for v, bounds in admitted.ranges.items()}
    )
    projected = perf_counter()
    schedules = {
        (operations[loop], family): schedule.project(operations)
        for (loop, family), schedule in selection.dispatch.schedules.items()
    }
    lower_execution(module, natives, scalar_facts, schedules)
    lowered = perf_counter()
    simplify_scalars(module, facts=scalar_facts, finalize=False)
    simplified = perf_counter()
    conditions = simplify_conditions(module, facts=scalar_facts)
    executable = ExecutablePlan(module, natives, scalar_facts.ranges)
    merged = perf_counter()
    timings += (
        ("before_snapshot", frozen - started),
        ("materialize", materialized - frozen),
        ("execution_projection", projected - materialized),
        ("execution_lowering", lowered - projected),
        ("scalar_simplification", simplified - lowered),
        ("condition_simplification", merged - simplified),
    )
    result = object.__new__(StructuredSynchronization)
    for name, value in (
        ("structure", structure),
        ("admission", admission),
        ("dependencies", dependencies),
        ("selection", selection),
        ("resources", resources),
        ("before_ir", before_ir),
        ("generation", generation),
        ("executable", executable),
        ("condition_work", conditions),
        ("timings", timings),
        ("integer_queries", query_statistics()),
    ):
        object.__setattr__(result, name, value)
    return result


@query_scope()
def synchronize_structure(source) -> StructuredSynchronization:
    """Run the current analysis stages, then materialize the selected program.

    Imported physical snapshots are not accepted by this source entry. It
    reports an unsupported contract when a required fact cannot be established.
    """
    from ..analysis.event_planning import plan_event_resources
    from ..analysis.structured_resources import allocate_structured_events
    from .generate_notifications import generate_notifications

    timings = []

    def stage(name, function, *args, **kwargs):
        started = perf_counter()
        result = function(*args, **kwargs)
        timings.append((name, perf_counter() - started))
        return result

    structure = stage("native_lowering", lower_native_structure, source)
    memory = stage("memory_collection", collect_native_memory, structure.module)
    bounds = stage("bounds", admit_memory_bounds, memory)
    admission = stage("admission", admit_structured_memory, structure, memory, bounds)
    dispatch = stage("dispatch", bind_structured_dispatch, memory)
    dependencies = stage(
        "dependencies",
        build_structured_dependencies,
        memory,
        admission=admission,
        dispatch=dispatch,
    )
    program = generate_notifications(dependencies, dispatch)
    timings.extend(program.timings)
    problem = stage("event_constraints", plan_event_resources, program)
    resources = stage("event_banks", allocate_structured_events, problem)
    return _materialize_owned(
        structure, admission, dependencies, program, resources, tuple(timings)
    )
