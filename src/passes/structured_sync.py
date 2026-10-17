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

from ..analysis.structured_admission import StructuredAdmission, admit_structured_memory
from ..analysis.structured_bounds import admit_memory_bounds
from ..analysis.structured_dependencies import StructuredDependencies, build_structured_dependencies
from ..analysis.structured_dispatch import bind_structured_dispatch
from ..analysis.structured_memory import collect_native_memory
from ..analysis.structured_protocols import SynchronizationSelection
from ..diagnostics import require
from ..dialects.io import dump
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
    """Final owned physical IR plus the facts established for its native body.

    Logical facts are established before terminal clock lowering. Native
    identities survive projection; address/control SSA is explicitly remapped.
    The independent pre-sync snapshot precedes notification insertion.
    No external verified flag can authorize this result.
    """

    structure: NativeStructure
    admission: StructuredAdmission
    dependencies: StructuredDependencies
    selection: SynchronizationSelection
    resources: StructuredEventPlan
    before_ir: str
    generation: NotificationProgram
    condition_work: ConditionWork
    scalar_facts: ScalarFacts
    timings: tuple[tuple[str, float], ...]

    def __init__(self, *args, **kwargs):
        raise TypeError("use synchronize_structure to establish the executable contracts")

    @property
    def module(self) -> ModuleOp:
        return self.structure.module

    @property
    def memory(self):
        return self.admission.memory

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
    module = materialize_notifications(admission.memory, resources.actions)
    require(module is structure.module, "notification insertion changed its owned physical module")
    # Terminal lowering preserves the admitted actor instruction trace and
    # remaps control/address SSA. Subsequent passes use these actual domains;
    # original logical facts remain analysis input, not final control handles.
    materialized = perf_counter()
    scalar_facts = ScalarFacts.from_control(admission.memory.control, admission.host_expressions)
    lower_execution(admission.memory, scalar_facts)
    lowered = perf_counter()
    simplify_scalars(module, facts=scalar_facts, finalize=False)
    simplified = perf_counter()
    conditions = simplify_conditions(module, facts=scalar_facts)
    merged = perf_counter()
    timings += (
        ("before_snapshot", frozen - started),
        ("materialize", materialized - frozen),
        ("execution_lowering", lowered - materialized),
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
        ("condition_work", conditions),
        ("scalar_facts", scalar_facts),
        ("timings", timings),
    ):
        object.__setattr__(result, name, value)
    return result


def synchronize_structure(source) -> StructuredSynchronization:
    """Run the current analysis stages, then materialize the selected program.

    Imported physical snapshots are not accepted by this source entry. It
    reports an unsupported contract when a required fact cannot be established.
    """
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
    resources = stage("event_banks", allocate_structured_events, program)
    return _materialize_owned(
        structure, admission, dependencies, program, resources, tuple(timings)
    )
