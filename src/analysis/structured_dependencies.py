"""Join immutable logical values with dispatch-owned storage retirement.

Source traversal selects read versions exactly once. Physical storage analysis
then connects adjacent admitted lifetimes. Neither stage allocates events or
supplies asynchronous completion merely from a logical value edge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..diagnostics import require
from .logical_valueflow import LogicalValueFlow, build_logical_valueflow
from .storage_lifetimes import StorageLifetimes, build_storage_lifetimes
from .structured_dispatch import StructuredDispatch, bind_structured_dispatch
from .structured_memory import NativeMemoryFacts
from .structured_regions import AccessOccurrence, DependencyWork, StructuredNeed

if TYPE_CHECKING:
    from .structured_admission import StructuredAdmission


@dataclass(frozen=True)
class StructuredDependencies:
    memory: NativeMemoryFacts
    admission: StructuredAdmission | None
    dispatch: StructuredDispatch
    values: LogicalValueFlow
    storage: StorageLifetimes
    needs: tuple[StructuredNeed, ...]
    required: tuple[StructuredNeed, ...]
    work: DependencyWork

    @property
    def closed(self):
        return self.values.closed and self.storage.closed

    def require_closed(self):
        self.values.require_closed()
        self.storage.require_closed()


def build_structured_dependencies(
    memory: NativeMemoryFacts,
    *,
    admission: StructuredAdmission | None = None,
    dispatch: StructuredDispatch | None = None,
) -> StructuredDependencies:
    dispatch = bind_structured_dispatch(memory) if dispatch is None else dispatch
    require(dispatch.memory is memory, "storage dispatch belongs to another memory snapshot")
    values = build_logical_valueflow(memory, admission=admission)
    storage = build_storage_lifetimes(values, dispatch)
    reads = tuple(need for need in values.reads if isinstance(need.source, AccessOccurrence))
    needs = reads + storage.needs
    required = reads + storage.required
    work = replace(
        values.work,
        needs=len(needs),
        loop_carries=sum(n.loop is not None and n.distance not in (None, 0) for n in needs),
    )
    return StructuredDependencies(
        memory, admission, dispatch, values, storage, needs, required, work
    )
