"""One owned source-to-synchronized-IR pipeline.

All memory requirements use the same logical notification representation.
Selection establishes capture/acquire coverage; one resource stage establishes
pairing, reuse and progress across both finite and repeated invocations. Only
then does materialization insert real notification operations in physical IR.
The backend consumes that IR, never a second action-placement interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .lower_native import NativeStructure, lower_native_structure
from .materialize_structure import materialize_notifications

if TYPE_CHECKING:
    from ..analysis.structured_resources import StructuredEventPlan


@dataclass(frozen=True, init=False)
class StructuredSynchronization:
    """Final owned physical IR plus the facts established for its native body.

    Controlled notification insertion preserves original native operations,
    address operands and loop identities. The pre-sync snapshot is frozen
    before insertion. No external verified flag can authorize this result.
    """

    structure: NativeStructure
    admission: StructuredAdmission
    dependencies: StructuredDependencies
    selection: SynchronizationSelection
    resources: StructuredEventPlan
    before_ir: str

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


def _materialize_owned(structure, admission, dependencies, selection, resources):
    require(
        admission.structure is structure
        and dependencies.memory is admission.memory
        and selection.dependencies is dependencies
        and resources.selection is selection,
        "synchronization stages belong to different owned structures",
    )
    before_ir = dump(structure.module)
    module = materialize_notifications(admission.memory, resources.actions)
    require(module is structure.module, "notification insertion changed its owned physical module")
    result = object.__new__(StructuredSynchronization)
    for name, value in (
        ("structure", structure),
        ("admission", admission),
        ("dependencies", dependencies),
        ("selection", selection),
        ("resources", resources),
        ("before_ir", before_ir),
    ):
        object.__setattr__(result, name, value)
    return result


def synchronize_structure(source) -> StructuredSynchronization:
    """Establish each stage's facts once and materialize the selected program.

    Imported physical snapshots are not accepted by this source entry. It
    reports an unsupported contract when a required fact cannot be established.
    """
    from ..analysis.structured_resources import allocate_structured_events
    from .select_notifications import select_notifications

    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    admission = admit_structured_memory(structure, memory, bounds)
    dependencies = build_structured_dependencies(memory, admission=admission)
    dispatch = bind_structured_dispatch(memory)
    selection = select_notifications(dependencies, dispatch)
    resources = allocate_structured_events(selection)
    return _materialize_owned(structure, admission, dependencies, selection, resources)
