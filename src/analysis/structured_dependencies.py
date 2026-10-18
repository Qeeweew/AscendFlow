"""One may-conflict order interface for native, region and storage obligations.

Source values and initialization belong to source/admission facts. Sync does
not require whole-program exact valueflow closure. Closed private packets are
local optional precision; all other states retain conservative source order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..diagnostics import require
from .access_order import AccessOrder, build_access_order
from .structured_dispatch import StructuredDispatch, bind_structured_dispatch
from .structured_memory import NativeMemoryFacts
from .structured_regions import DependencyWork, StructuredNeed

if TYPE_CHECKING:
    from .structured_admission import StructuredAdmission


@dataclass(frozen=True)
class StructuredDependencies:
    memory: NativeMemoryFacts
    admission: StructuredAdmission | None
    dispatch: StructuredDispatch
    order: AccessOrder
    needs: tuple[StructuredNeed, ...]
    required: tuple[StructuredNeed, ...]
    work: DependencyWork


def build_structured_dependencies(memory, *, admission=None, dispatch=None):
    dispatch = bind_structured_dispatch(memory) if dispatch is None else dispatch
    require(dispatch.memory is memory, "storage dispatch belongs to another memory snapshot")
    order = build_access_order(memory, dispatch, admission=admission)
    return StructuredDependencies(
        memory, admission, dispatch, order, order.needs, order.required, order.work
    )
