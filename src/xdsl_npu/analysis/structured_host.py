"""Host integer-safety contracts over the owned physical SSA structure.

This stage derives launch-time signed extrema. It consumes the same actual
control and allocation facts as address admission; it neither selects events
nor reinterprets native memory effects. This extraction preserves the existing
proof algorithm, without caching or performance-specific changes.
"""

from xdsl.dialects.builtin import IndexType

from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import arguments, interval
from ..scalar import Expr, expression, literal, positive


def _require(condition, message, operation, *, kind=ErrorKind.UNSUPPORTED, witness=()):
    require(
        condition,
        message,
        kind=kind,
        phase="structured-host",
        location=str(operation.location),
        witness=witness,
    )


def _host_interval(value, domains, facts):
    """Signed host extrema; unsigned operations retain their nonnegative domain.

    Independent endpoint combinations can describe inactive owners as well.
    Clipping an unsigned dividend's extrema to zero keeps those empty-domain
    endpoints evaluable; source arithmetic and active address proofs establish
    the real unsigned dividend premise separately.
    """
    try:
        return interval(value, domains, facts)
    except CompileError as error:
        if error.message not in {
            "index product has an unproven sign",
            "unsigned index division has an unproven nonnegative dividend",
        }:
            raise
    if value.kind in {"constant", "argument"}:
        return value, value
    a, b = value.args
    al, ah = _host_interval(a, domains, facts)
    bl, bh = _host_interval(b, domains, facts)
    if value.kind in {"add", "min", "max"}:
        return expression(value.kind, al, bl), expression(value.kind, ah, bh)
    if value.kind == "mul":
        products = tuple(expression("mul", x, y) for x in (al, ah) for y in (bl, bh))
        low = high = products[0]
        for product in products[1:]:
            low, high = expression("min", low, product), expression("max", high, product)
        return low, high
    require(
        value.kind in {"floordiv", "ceildiv", "rem"} and bl == bh and positive(b, facts),
        "host extrema require a positive uniform divisor",
        kind=ErrorKind.UNSUPPORTED,
    )
    if value.kind == "rem":
        return literal(0), expression("sub", b, literal(1))
    return tuple(
        expression(value.kind, expression("max", literal(0), bound), b) for bound in (al, ah)
    )


def build_host_expressions(memory, bounds):
    """Uniform signed extrema of every actual retained index SSA intermediate."""
    environment, facts = memory.control.environment, memory.control.environment.facts
    result = {}

    def include(value, frame, operation):
        lower = upper = value
        domains = frame.domains()
        for _ in range(len(domains) + 2):
            if not any(name.startswith("$") for name in arguments(lower) | arguments(upper)):
                result[lower] = result[upper] = None
                return
            next_lower = _host_interval(lower, domains, facts)[0]
            next_upper = _host_interval(upper, domains, facts)[1]
            if (next_lower, next_upper) == (lower, upper):
                break
            lower, upper = next_lower, next_upper
        _require(
            False,
            "host integer safety lacks a uniform scalar range",
            operation,
            witness=(f"value={value.text()}",),
        )

    for root, capacity in bounds.allocations.items():
        include(capacity, memory.control.frames[root.owner], root.owner)
    for operation, frame in memory.control.frames.items():
        if frame.statically_empty:
            continue
        for value in operation.results:
            if isinstance(value.type, IndexType):
                include(environment.expression(value), frame, operation)
    for loop in memory.control.loops.values():
        frame = memory.control.frames[loop.operation]
        if frame.parent is not None and frame.parent.statically_empty:
            continue
        parent = frame.parent or frame
        include(loop.lower, parent, loop.operation)
        include(loop.upper, parent, loop.operation)
        include(loop.count, parent, loop.operation)
        horizon = getattr(loop.operation, "schedule", None)
        extent = 0 if horizon is None else horizon.horizon.data
        include(expression("add", loop.count, literal(extent)), parent, loop.operation)
        if not frame.statically_empty:
            include(loop.induction, frame, loop.operation)
    pending, seen = list(result), set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        if value.constant is not None:
            _require(
                -(1 << 63) <= value.constant < 1 << 63,
                "static host index exceeds signed int64",
                memory.module,
                kind=ErrorKind.INVALID,
                witness=(value.text(),),
            )
        else:
            pending.extend(child for child in value.args if isinstance(child, Expr))
    return tuple(result)
