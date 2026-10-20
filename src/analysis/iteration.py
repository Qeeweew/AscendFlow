"""Shared source-loop domains and immutable handles to retained xDSL control.

The schedule pass and arithmetic admission use the same actual lower/upper/
stride identity. These data types do not depend on a transformation or planner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from ..diagnostics import CompileError, ErrorKind, require
from ..dialects.source import IterationOp

Bound = int | SSAValue


def _constant(value: SSAValue) -> int | None:
    owner = value.owner
    return owner.value.value.data if isinstance(owner, arith.ConstantOp) else None


@dataclass(frozen=True, slots=True)
class IterationDomain:
    lower: Bound
    upper: Bound
    stride: int

    def __post_init__(self):
        require(type(self.stride) is int and self.stride > 0, "loop step must be positive")

    def concrete(self, values: Mapping[SSAValue, int]) -> tuple[int, int, int]:
        """Explicit finite interpretation only; no compiler phase enumerates it."""

        def bind(value):
            result = value if type(value) is int else values[value]
            require(type(result) is int, "loop bound must be an integer")
            return result

        return bind(self.lower), bind(self.upper), self.stride

    def trip_count(self, values: Mapping[SSAValue, int]) -> int:
        lower, upper, stride = self.concrete(values)
        size = max(upper - lower, 0)
        return size // stride + (size % stride != 0)


@dataclass(frozen=True, slots=True)
class LoopControl:
    operation: IterationOp | scf.ForOp
    domain: IterationDomain
    parent: LoopControl | None
    depth: int

    @property
    def coordinate(self) -> SSAValue:
        return self.operation.body.block.args[0]

    def source_iteration(self, ordinal, values):
        """Bind one logical ordinal, without choosing any actor's clock."""
        count = self.domain.trip_count(values)
        if not 0 <= ordinal < count:
            return None
        lower, _, stride = self.domain.concrete(values)
        return lower + ordinal * stride


@dataclass(frozen=True)
class LoopControls:
    """Owned references to existing IR, not another mutable program tree."""

    module: ModuleOp
    loops: tuple[LoopControl, ...]
    by_operation: Mapping[Operation, LoopControl]
    operations_visited: int

    @property
    def statistics(self):
        return {
            "operations_visited": self.operations_visited,
            "loop_templates": len(self.loops),
            "body_clones": 0,
            "expanded_iterations": 0,
            "maximum_depth": max((loop.depth for loop in self.loops), default=0),
        }


def iteration_domain(operation: IterationOp | scf.ForOp) -> IterationDomain:
    if isinstance(operation, IterationOp):
        upper = operation.limit if operation.limit is not None else operation.count.data
        return IterationDomain(0, upper, 1)
    require(isinstance(operation, scf.ForOp), "expected a structured loop")
    stride = _constant(operation.step)
    if stride is None or stride <= 0:
        raise CompileError(
            "serial loop requires a positive constant step",
            kind=ErrorKind.UNSUPPORTED,
            phase="loop-control",
            location=str(operation.location),
        )
    lower, upper = operation.lb, operation.ub
    lower_value, upper_value = _constant(lower), _constant(upper)
    return IterationDomain(
        lower if lower_value is None else lower_value,
        upper if upper_value is None else upper_value,
        stride,
    )
