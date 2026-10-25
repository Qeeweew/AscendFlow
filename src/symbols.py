"""Scalar identity and role, independent of its diagnostic spelling.

Compiler symbols bind an actual SSA value or control operation. Two captures
with the same printed name remain distinct; renaming one binding changes no
identity. Unbound mathematical symbols use an explicit name and role.
"""

from dataclasses import dataclass, field
from enum import Enum

from xdsl.ir import Block, Operation, SSAValue

from .diagnostics import require


class SymbolKind(str, Enum):
    PARAMETER = "parameter"
    GROUP = "group"
    PARTICIPANT = "participant"
    ORDINAL = "ordinal"
    METADATA = "metadata"
    CARRY = "carry"
    SSA = "ssa"
    PREDICATE = "predicate"
    SELECTION = "selection"


def _binding_order(origin):
    """Retain a binding's structural order before terminal IR mutation."""
    suffix = ()
    if isinstance(origin, SSAValue):
        owner = origin.owner
        suffix = (int(isinstance(owner, Block)), origin.index)
        operation = owner.parent_op() if isinstance(owner, Block) else owner
    else:
        operation = origin
    path = []
    while isinstance(operation, Operation) and operation.parent is not None:
        block = operation.parent
        region = block.parent
        parent = region.parent if region is not None else None
        path.append(
            (
                tuple(parent.regions).index(region) if parent is not None else 0,
                region.get_block_index(block) if region is not None else 0,
                block.get_operation_index(operation),
            )
        )
        operation = parent
    # The final identity only breaks ties between detached/different modules;
    # structural positions uniquely order bindings in an owned compilation.
    return tuple(reversed(path)), suffix, id(origin)


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str = field(compare=False)
    kind: SymbolKind
    origin: object | None = field(default=None, compare=False, repr=False)
    captures: tuple = ()
    _identity: tuple = field(init=False, repr=False)
    _order: tuple = field(init=False, compare=False, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        require(type(self.name) is str and bool(self.name), "scalar symbol needs a diagnostic name")
        require(isinstance(self.kind, SymbolKind), "scalar symbol needs an explicit role")
        require(isinstance(self.captures, tuple), "scalar capture coordinates must be immutable")
        # Keep origin alive: its id cannot be reused during this symbol's life.
        identity = (0, self.name) if self.origin is None else (1, id(self.origin))
        object.__setattr__(self, "_identity", identity)
        order = (0, self.name) if self.origin is None else (1, _binding_order(self.origin))
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_hash", hash((self.kind, self.captures, identity)))

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self.name

    def with_captures(self, captures):
        """Substitute coordinates while retaining the original SSA binding order.

        The binding and its lexical position do not change when a notification
        selects another iteration. Rewalking the IR here is both redundant and
        would observe a different order after terminal mutation.
        """
        require(isinstance(captures, tuple), "scalar capture coordinates must be immutable")
        if captures == self.captures:
            return self
        result = object.__new__(Symbol)
        for name in ("name", "kind", "origin", "_identity", "_order"):
            object.__setattr__(result, name, getattr(self, name))
        object.__setattr__(result, "captures", captures)
        object.__setattr__(result, "_hash", hash((self.kind, captures, self._identity)))
        return result


def symbol_name(symbol: str | Symbol) -> str:
    return symbol.name if isinstance(symbol, Symbol) else symbol


def symbol_order(symbol: str | Symbol) -> tuple:
    if isinstance(symbol, Symbol):
        return (
            symbol.kind.value,
            symbol._order,
            tuple(value.order_key for value in symbol.captures),
        )
    return SymbolKind.PARAMETER.value, (0, symbol), ()


def is_coordinate(symbol: str | Symbol) -> bool:
    """Only explicit roles distinguish task coordinates from parameters."""
    return isinstance(symbol, Symbol) and symbol.kind is not SymbolKind.PARAMETER
