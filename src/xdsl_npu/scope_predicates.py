"""Narrow native first/last predicates on one actual retained parent clock.

This admits equality to zero or that clock's own bound minus one. A runtime
cycle such as task % kv_tiles is deliberately not a parent-boundary predicate.
Four fixed analysis cases classify native effects without expanding runtime N.
"""

from xdsl.dialects import arith, scf

from .completion import CountDomain, Position
from .model import require
from .scalar import expression, from_ssa, literal
from .scope_domains import Activation
from .source_ir import PipelineOp

PARENT_CASES = ("single", "first", "middle", "last")


def case_activation(label: str) -> Activation:
    require(label in PARENT_CASES, "unknown native parent guard case")
    if label == "single":
        return Activation(CountDomain(1, 0, exact=1), Position(False, 0), Position(False, 0))
    if label == "first":
        return Activation(CountDomain(1, 0, minimum=2), Position(False, 0), Position(False, 0))
    if label == "middle":
        return Activation(CountDomain(1, 0, minimum=3), Position(False, 1), Position(True, 1))
    return Activation(CountDomain(1, 0, minimum=2), Position(True, 0), Position(True, 0))


def guard_active(kind: str, label: str) -> bool:
    require(kind in {"first", "last"}, "unknown native parent guard kind")
    require(label in PARENT_CASES, "unknown native parent guard case")
    return label == "single" or label == kind


def classify_guard(op: scf.IfOp, parent: PipelineOp) -> str | None:
    """Return parent first/last, or None for a validated child-local last.

    All source guards, including effect-free ones, must be checked. Identity
    of the compared block argument establishes the clock; address equivalence
    and a remainder derived from that argument cannot authorize another clock.
    """
    ancestors = []
    current = op.parent_op()
    while current is not None:
        ancestors.append(current)
        current = current.parent_op()
    require(parent in ancestors, "source guard does not belong to its actual parent clock")
    require(
        not any(isinstance(ancestor, scf.IfOp) for ancestor in ancestors),
        "nested source guards need hierarchical predicate analysis",
    )
    require(
        not op.results
        and not list(op.false_region.block.ops)[:-1]
        and isinstance(op.cond.owner, arith.CmpiOp)
        and op.cond.owner.predicate.value.data == 0,
        "native scope guards require equality with no else or results",
    )
    clocks = [ancestor for ancestor in ancestors if isinstance(ancestor, (PipelineOp, scf.ForOp))]
    require(
        clocks and isinstance(clocks[0], PipelineOp), "source guard needs an actual pipeline clock"
    )
    nearest = clocks[0]
    coordinates = {clock.body.block.args[0]: f"$guard{i}" for i, clock in enumerate(clocks)}
    operands = tuple(op.cond.owner.operands)
    for clock in dict.fromkeys((parent, nearest)):
        coordinate = clock.body.block.args[0]
        if operands[0] is coordinate:
            other = operands[1]
        elif operands[1] is coordinate:
            other = operands[0]
        else:
            continue
        value = from_ssa(other, coordinates=coordinates)
        bound = (
            from_ssa(clock.limit, coordinates=coordinates)
            if clock.limit is not None
            else literal(clock.count.data)
        )
        if clock is parent and value == literal(0):
            return "first"
        require(
            value == expression("sub", bound, literal(1)),
            "native scope guard requires actual parent coordinate == 0 or its bound - 1; "
            "child predicates remain last-only",
        )
        return "last" if clock is parent else None
    require(
        False,
        "native scope guard must compare an actual clock coordinate; runtime-cycle predicates are unsupported",
    )
    return None
