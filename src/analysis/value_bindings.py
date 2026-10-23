"""Finite source Tile provenance, without value or completion admission.

Stage and Domain results forward the yielded binding. A Pipeline result
selects its last logical iteration, and is undefined when that loop is empty.
Collecting this selection does not establish non-emptiness, a reaching writer,
initialized data, ownership, or asynchronous completion. Those obligations
must be discharged at actual uses before executable code can be authorized.

Each source SSA value is resolved once. Aliases share their binding; nested
last selections share their child node. No iteration, history, completion
graph, or copy of a value's storage is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from ..diagnostics import ErrorKind, require
from ..dialects.physical import TaskOp
from ..dialects.source import CopyGroupOp, DomainOp, IterationOp, StageOp, TileSliceOp, YieldOp
from ..dialects.structured import PipelineLoopOp
from ..dialects.types import TileType

if TYPE_CHECKING:
    from ..passes.native_recipes import RecipeResult


@dataclass(frozen=True, slots=True)
class TileDefinition:
    """An actual native writer/operand anchor, not a reaching-def certificate."""

    operation: TaskOp
    operand: int


@dataclass(frozen=True, slots=True)
class LastTile:
    """Conditional last-logical-iteration selection on this actual loop.

    The later consumer must establish a nonempty execution domain and bind
    the selected loop's ordinal/iv. Other enclosing clocks retain identity.
    This node neither evaluates N-1 nor acquires the selected writer's end.
    """

    loop: PipelineLoopOp
    value: TileDefinition | LastTile


@dataclass(frozen=True, slots=True)
class TileUse:
    expected: TileDefinition | LastTile
    reader: TaskOp
    operand: int
    source: SSAValue
    source_input: int


@dataclass(frozen=True, slots=True, eq=False)
class ValueScope:
    """An actual shared lexical scope; parent links are not execution history.

    A later value admission binds common logical clocks across Stage scopes,
    exact branch activation, and explicit LastTile selections. Merely sharing
    a scope or leaving one establishes no native readiness.
    """

    parent: ValueScope | None
    operation: Operation
    branch: int | None = None


def _supported(condition, message, operation=None):
    require(
        condition,
        message,
        kind=ErrorKind.UNSUPPORTED,
        phase="value-bindings",
        location=str(operation.location) if isinstance(operation, Operation) else None,
    )


def bind_values(
    source: ModuleOp,
    recipes: Mapping[Operation, RecipeResult],
    loops: Mapping[Operation, Operation],
) -> tuple[dict[SSAValue, TileDefinition | LastTile], tuple[TileUse, ...]]:
    """Collect unresolved writer selections and each actual native read use.

    Inputs belong to the caller's admitted source/structure snapshot. No
    native contracts, scalar/Tile legality, or completion proof are rerun.
    RecipeResult is imported only for static typing: analysis has no runtime
    dependency on transformation passes.
    """
    values: dict[SSAValue, TileDefinition | LastTile] = {}

    def resolve(value):
        # Yield provenance has one successor per SSA value. Iterative unwind
        # avoids a Python recursion limit for otherwise finite region nesting.
        path, active = [], set()
        current = value
        while current not in values:
            owner = current.owner
            _supported(current not in active, "cyclic logical Tile yield provenance", owner)
            active.add(current)
            if isinstance(owner, TileSliceOp):
                path.append((current, owner))
                current = owner.source
            elif isinstance(owner, (StageOp, DomainOp, IterationOp, CopyGroupOp)):
                terminator = owner.body.block.last_op
                _supported(
                    isinstance(terminator, YieldOp) and current.index < len(terminator.values),
                    "logical Tile region lost its corresponding yield operand",
                    owner,
                )
                path.append((current, owner))
                current = terminator.values[current.index]
            else:
                recipe = recipes.get(owner)
                _supported(
                    recipe is not None
                    and isinstance(recipe.definition, TaskOp)
                    and recipe.definition_ref is not None,
                    "logical Tile has no actual native writer",
                    owner,
                )
                values[current] = TileDefinition(recipe.definition, recipe.definition_ref)
        result = values[current]
        for alias, owner in reversed(path):
            if isinstance(owner, IterationOp):
                loop = loops.get(owner)
                _supported(
                    isinstance(loop, PipelineLoopOp),
                    "logical Tile last selection lost its actual physical pipeline",
                    owner,
                )
                result = LastTile(loop, result)
            values[alias] = result
        return values[value]

    uses = []
    for operation, recipe in recipes.items():
        for value in operation.results:
            if isinstance(value.type, TileType):
                resolve(value)
        readers = {}
        for reader in recipe.readers:
            readers.setdefault(reader.source_input, []).append(reader)
        for dependency, source_index in zip(
            getattr(operation, "dependencies", ()),
            getattr(operation, "dependency_inputs", ()),
            strict=True,
        ):
            origin = resolve(dependency)
            matching = readers.get(source_index.data, ())
            require(matching, "logical Tile dependency lost its actual native reader")
            uses.extend(
                TileUse(origin, reader.operation, reader.ref_index, dependency, source_index.data)
                for reader in matching
            )
    # The frontend also exports unused results. Empty loops can retain such
    # unresolved LastTile nodes without inventing a defined numerical value.
    for operation in source.walk():
        if isinstance(operation, (StageOp, DomainOp, IterationOp, CopyGroupOp, TileSliceOp)):
            for value in operation.results:
                if isinstance(value.type, TileType):
                    resolve(value)
    return values, tuple(uses)
