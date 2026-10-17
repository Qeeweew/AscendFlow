"""Compile-time uniform extrema over the owned physical SSA structure.

These extrema describe legal index evaluations without emitting input checks.
This stage consumes the same actual
control and allocation facts as address admission; it neither selects events
nor reinterprets native memory effects. Each scope constructs its scalar-range
DAG once. Uniform subexpressions need no loop elimination or polynomial query.
"""

from xdsl.dialects.builtin import IndexType

from ..diagnostics import ErrorKind, require
from ..indexing import _joint_affine_interval, arguments, nonnegative
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


class _HostRanges:
    """Owned scalar-range facts for one immutable lexical domain.

    Independent endpoint combinations can describe inactive owners as well.
    Clipping an unsigned dividend's extrema to zero keeps those empty-domain
    endpoints evaluable; source arithmetic and active address proofs establish
    the real unsigned dividend premise separately. Exact joint affine ranges
    preserve cyclic owner/iteration correlation rather than treating them as
    independent intervals. This stage does not authorize source arithmetic.
    """

    def __init__(self, domains, facts):
        self.domains, self.facts, self.ranges = domains, facts, {}
        for coordinate, lower, upper in domains:
            self.ranges.setdefault(coordinate, (lower, upper))

    def visit(self, value):
        if value not in self.ranges:
            self.ranges[value] = self._construct(value)
        return self.ranges[value]

    def _construct(self, value):
        if not any(name.startswith("$") for name in arguments(value)):
            return value, value
        if value.kind in {"constant", "argument"}:
            return value, value  # Unknown scalar carries remain unbounded.
        joint = _joint_affine_interval(value, self.domains, self.facts, atomic=True)
        if joint is not None:
            return joint
        a, b = value.args
        (al, ah), (bl, bh) = self.visit(a), self.visit(b)
        if value.kind in {"add", "min", "max"}:
            return expression(value.kind, al, bl), expression(value.kind, ah, bh)
        if value.kind == "mul":
            for constant, low, high in ((a, bl, bh), (b, al, ah)):
                if constant.constant is not None:
                    if constant.constant < 0:
                        low, high = high, low
                    return expression("mul", constant, low), expression("mul", constant, high)
            if nonnegative(al, self.facts) and nonnegative(bl, self.facts):
                return expression("mul", al, bl), expression("mul", ah, bh)
            products = tuple(expression("mul", x, y) for x in (al, ah) for y in (bl, bh))
            low = high = products[0]
            for product in products[1:]:
                low, high = expression("min", low, product), expression("max", high, product)
            return low, high
        require(
            value.kind in {"floordiv", "ceildiv", "rem"}
            and b == bl == bh
            and positive(b, self.facts),
            "host extrema require a positive uniform divisor",
            kind=ErrorKind.UNSUPPORTED,
        )
        if value.kind == "rem":
            return literal(0), expression("sub", b, literal(1))
        return tuple(
            expression(
                value.kind,
                bound if nonnegative(bound, self.facts) else expression("max", literal(0), bound),
                b,
            )
            for bound in (al, ah)
        )


def build_host_expressions(memory, bounds):
    """Uniform signed extrema for the compile-time index-range hypotheses."""
    environment, facts = memory.control.environment, memory.control.environment.facts
    result = {}
    scopes = {}

    def include(value, frame, operation):
        lower = upper = value
        domains = frame.domains()
        if domains not in scopes:
            scopes[domains] = _HostRanges(domains, facts)
        ranges = scopes[domains]
        for _ in range(len(domains) + 2):
            if not any(name.startswith("$") for name in arguments(lower) | arguments(upper)):
                result[lower] = result[upper] = None
                return
            next_lower = ranges.visit(lower)[0]
            next_upper = ranges.visit(upper)[1]
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
