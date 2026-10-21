"""Optional mathematical views of executable integer SSA.

Expression identity contains no range or evaluation certificate. Projections
may mention inactive/undefined coordinates; source static diagnostics and
new machine evaluations are checked separately. Unknown facts keep the SSA.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from math import gcd

from xdsl.dialects import arith

from .diagnostics import CompileError, ErrorKind, require
from .dialects.physical import RuntimeSizeOp
from .symbols import Symbol, SymbolKind, symbol_name, symbol_order

INT64_MAX = (1 << 63) - 1
DIVISION = {"floordiv", "rem", "ceildiv"}
BINARY = {"add", "mul", "min", "max", *DIVISION}
_expression_nodes = ContextVar("ascendflow_expression_nodes", default=None)


@contextmanager
def expression_scope():
    """Own raw syntax nodes for one compilation, never range or proof facts."""
    nodes = {}
    token = _expression_nodes.set(nodes)
    try:
        yield
    finally:
        _expression_nodes.reset(token)
        nodes.clear()


class _ExprType(type):
    def __call__(cls, kind, args):
        nodes = _expression_nodes.get()
        if nodes is None or cls is not Expr:
            return super().__call__(kind, args)
        # Validate the key's shape before lookup: Python otherwise conflates
        # invalid True/1.0 constants with a cached, valid integer 1. Failed
        # constructions retain their original diagnostics and never enter the
        # arena. A hit shares syntax, not the domain of its original SSA use.
        if type(kind) is not str or type(args) is not tuple:
            return super().__call__(kind, args)
        if kind == "constant":
            valid = len(args) == 1 and type(args[0]) is int
        elif kind == "argument":
            valid = len(args) == 1 and type(args[0]) in {str, Symbol}
        else:
            valid = (
                kind in BINARY
                and len(args) == 2
                and type(args[0]) is Expr
                and type(args[1]) is Expr
            )
        if not valid:
            return super().__call__(kind, args)
        key = kind, args
        node = nodes.get(key)
        if node is None:
            node = super().__call__(kind, args)
            nodes[key] = node
        return node


@dataclass(frozen=True)
class Fact:
    """A nonnegative integer range whose values are multiples of ``multiple``.

    Bounds describe mathematical integers. Derived products may exceed int64;
    legal kernel inputs must keep actual index evaluations representable.
    """

    minimum: int = 0
    multiple: int = 1
    maximum: int | None = None

    def __post_init__(self):
        require(type(self.minimum) is int and self.minimum >= 0, "invalid scalar minimum")
        require(type(self.multiple) is int and self.multiple > 0, "invalid scalar multiple")
        require(
            self.maximum is None or type(self.maximum) is int and self.maximum >= 0,
            "invalid scalar maximum",
        )
        minimum = ((self.minimum + self.multiple - 1) // self.multiple) * self.multiple
        maximum = None if self.maximum is None else self.maximum // self.multiple * self.multiple
        require(maximum is None or minimum <= maximum, "scalar range has no feasible value")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


def facts_from_sizes(operations):
    """Collect Size/RuntimeSize contracts from an explicitly chosen IR scope."""
    from .dialects.source import SizeOp

    facts, contracts = {}, {}
    for op in operations:
        if not isinstance(op, (SizeOp, RuntimeSizeOp)):
            continue
        name, minimum, multiple = op.sym_name.data, op.minimum.data, op.multiple.data
        require(
            name.isidentifier() and 0 <= minimum <= INT64_MAX and 0 < multiple <= INT64_MAX,
            "invalid uniform scalar contract",
        )
        fact = Fact(minimum, multiple, INT64_MAX)
        require(
            name not in contracts or contracts[name] == fact,
            f"uniform scalar contract disagrees: {name}",
        )
        contracts[name] = fact
        name = Symbol(name, SymbolKind.PARAMETER, op.value)
        require(
            name not in facts or facts[name] == fact, f"uniform scalar contract disagrees: {name}"
        )
        facts[name] = fact
    return facts


def _facts(names_or_facts):
    if isinstance(names_or_facts, Mapping):
        require(
            all(isinstance(fact, Fact) for fact in names_or_facts.values()),
            "uniform scalar facts must contain Fact values",
        )
        return names_or_facts
    return {name: Fact(maximum=INT64_MAX) for name in names_or_facts}


@dataclass(frozen=True, slots=True)
class Expr(metaclass=_ExprType):
    kind: str
    args: tuple
    weight: int = field(init=False, compare=False, repr=False)
    depth: int = field(init=False, compare=False, repr=False)
    order_key: tuple = field(init=False, compare=False, repr=False)
    _hash: int = field(init=False, compare=False, repr=False)
    variables: frozenset[str | Symbol] = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        require(
            isinstance(self.args, tuple), "scalar expression arguments must be an immutable tuple"
        )
        require(
            self.kind in {"constant", "argument", *BINARY},
            f"unsupported scalar expression kind: {self.kind}",
        )
        if self.kind == "constant":
            require(len(self.args) == 1 and type(self.args[0]) is int, "invalid scalar constant")
        elif self.kind == "argument":
            require(
                len(self.args) == 1 and isinstance(self.args[0], (str, Symbol)),
                "invalid scalar argument",
            )
            if isinstance(self.args[0], Symbol):
                require(
                    all(isinstance(value, Expr) for value in self.args[0].captures),
                    "scalar captures need original integer coordinates",
                )
        else:
            require(
                len(self.args) == 2 and all(isinstance(arg, Expr) for arg in self.args),
                "binary scalar expression needs two expressions",
            )
        children = [arg for arg in self.args if isinstance(arg, Expr)]
        weight = 1 + sum(arg.weight for arg in children)
        depth = 1 + max((arg.depth for arg in children), default=0)
        require(
            weight <= 4096 and depth <= 128,
            "scalar expression proof budget exceeded (4096 terms / depth 128)",
            kind=ErrorKind.ANALYSIS_LIMIT,
        )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "_hash", hash((self.kind, self.args)))
        key = (
            (symbol_order(self.args[0]),)
            if self.kind == "argument"
            else tuple(arg.order_key for arg in children)
            if children
            else self.args
        )
        object.__setattr__(self, "order_key", (self.kind, key))
        names = (
            frozenset((self.args[0],))
            if self.kind == "argument"
            else frozenset().union(*(arg.variables for arg in children))
        )
        if self.kind == "argument" and isinstance(self.args[0], Symbol):
            names |= frozenset().union(*(value.variables for value in self.args[0].captures))
        object.__setattr__(self, "variables", names)

    def __hash__(self):
        return self._hash

    @property
    def constant(self):
        return self.args[0] if self.kind == "constant" else None

    def evaluate(self, values):
        if self.kind == "constant":
            return self.args[0]
        if self.kind == "argument":
            symbol = self.args[0]
            return values[symbol] if symbol in values else values[symbol_name(symbol)]
        a, b = (arg.evaluate(values) for arg in self.args)
        if self.kind in DIVISION:
            require(
                a >= 0 and b > 0,
                "scalar division needs a nonnegative numerator and positive divisor",
            )
        operations = {
            "add": lambda: a + b,
            "mul": lambda: a * b,
            "min": lambda: min(a, b),
            "max": lambda: max(a, b),
            "floordiv": lambda: a // b,
            "rem": lambda: a % b,
            "ceildiv": lambda: a // b + (a % b != 0),
        }
        if self.kind not in operations:
            raise CompileError(f"unsupported scalar expression kind: {self.kind}")
        return operations[self.kind]()

    def text(self):
        if self.kind == "constant":
            return str(self.args[0])
        if self.kind == "argument":
            return symbol_name(self.args[0])
        return f"{self.kind}({', '.join(arg.text() for arg in self.args)})"

    def c(self, names=None):
        if names is not None and self in names:
            return names[self]
        if self.kind == "constant":
            return str(self.args[0])
        if self.kind == "argument":
            return "g" if self.args[0] == "$g" else "npu_size_" + self.args[0]
        a, b = (arg.c(names) for arg in self.args)
        if self.kind in {"add", "mul"}:
            return f"({a} {'+' if self.kind == 'add' else '*'} {b})"
        if self.kind in {"min", "max"}:
            return f"({a} {'<' if self.kind == 'min' else '>'} {b} ? {a} : {b})"
        if self.kind in {"floordiv", "ceildiv"} and self.args[1].constant == 1:
            # Keep the division in the proof DAG: its unsigned numerator still
            # needs validation. Rendering its identity preserves every checked
            # numerator operation without importing i64 wrapping arithmetic.
            return a
        if self.kind == "floordiv":
            return f"({a} / {b})"
        if self.kind == "rem":
            return f"({a} % {b})"
        if self.kind == "ceildiv":
            # Quotient/remainder spelling avoids the n + divisor - 1 overflow.
            return f"({a} / {b} + ({a} % {b} != 0))"
        raise CompileError(f"unsupported scalar expression kind: {self.kind}")


_UNIT_LITERALS = tuple(Expr("constant", (value,)) for value in (-1, 0, 1))


def literal(value):
    if type(value) is int and -1 <= value <= 1:
        return _UNIT_LITERALS[value + 1]
    return Expr("constant", (value,))


def substitute(value: Expr, bindings: Mapping[Expr, Expr]) -> Expr:
    """Simultaneous coordinate substitution, without assumptions about its domain.

    Replacements are not recursively substituted. Callers retain the actual
    loop/guard scope that makes any division or selected occurrence valid.
    This shared operation supplies no alias or completion proof.
    """
    if not bindings:
        return value
    memo = {}

    def visit(current):
        identity = id(current)
        if identity in memo:
            return memo[identity]
        if current in bindings:
            result = bindings[current]
        elif current.kind == "argument" and isinstance(current.args[0], Symbol):
            symbol = current.args[0]
            captures = tuple(visit(capture) for capture in symbol.captures)
            result = (
                current
                if captures == symbol.captures
                else Expr("argument", (replace(symbol, captures=captures),))
            )
        elif current.kind in {"constant", "argument"}:
            result = current
        else:
            children = tuple(visit(child) for child in current.args)
            result = (
                current
                if all(old is new for old, new in zip(current.args, children, strict=True))
                else expression(current.kind, *children)
            )
        memo[identity] = result
        return result

    return visit(value)


def expression(kind, a, b):
    if kind == "sub":
        return expression("add", a, expression("mul", literal(-1), b))
    if a.constant is not None and b.constant is not None:
        require(kind in BINARY, f"unsupported scalar expression kind: {kind}")
        x, y = a.constant, b.constant
        if kind in DIVISION and (x < 0 or y <= 0):
            # A mathematical projection can mention an inactive coordinate.
            # Keep its raw DAG; actual source/static and newly emitted machine
            # evaluations have their own definedness checks.
            return Expr(kind, (a, b))
        if kind == "add":
            return literal(x + y)
        if kind == "mul":
            return literal(x * y)
        if kind == "min":
            return literal(min(x, y))
        if kind == "max":
            return literal(max(x, y))
        if kind == "floordiv":
            return literal(x // y)
        if kind == "ceildiv":
            return literal(x // y + int(x % y != 0))
        return literal(x % y)
    if kind == "add":
        if a.constant == 0:
            return b
        if b.constant == 0:
            return a
    if kind == "mul":
        # Normalize -(tile * g) to (-tile) * g as emitted by affine indices.
        if a.constant is not None and b.kind == "mul":
            for i in (0, 1):
                if b.args[i].constant is not None:
                    return expression(
                        "mul", literal(a.constant * b.args[i].constant), b.args[1 - i]
                    )
        if b.constant is not None and a.kind == "mul":
            return expression("mul", b, a)
        if a.constant == 0 or b.constant == 0:
            return literal(0)
        if a.constant == 1:
            return b
        if b.constant == 1:
            return a
    if kind in {"min", "max"} and a == b:
        return a
    if kind in {"add", "mul", "min", "max"} and a.order_key > b.order_key:
        a, b = b, a
    return Expr(kind, (a, b))


ARITHMETIC = {
    arith.AddiOp: "add",
    arith.SubiOp: "sub",
    arith.MuliOp: "mul",
    arith.MinSIOp: "min",
    arith.MaxSIOp: "max",
    arith.DivUIOp: "floordiv",
    arith.RemUIOp: "rem",
    arith.CeilDivUIOp: "ceildiv",
}


def from_ssa(value, *, coordinates=None, memo=None):
    """Optional mathematical view of executable SSA, without input admission.

    Unmodelled operations retain their SSA identity and operand captures. This
    is not a range, invariance, or source-definedness certificate.
    """
    from .dialects.source import SizeOp

    memo = {} if memo is None else memo

    def visit(value):
        if value in memo:
            return memo[value]
        if coordinates and value in coordinates:
            result = Expr("argument", (coordinates[value],))
        elif isinstance(value.owner, (SizeOp, RuntimeSizeOp)):
            result = Expr(
                "argument", (Symbol(value.owner.sym_name.data, SymbolKind.PARAMETER, value),)
            )
        elif isinstance(value.owner, arith.ConstantOp):
            result = literal(value.owner.value.value.data)
        else:
            op = value.owner
            operands = tuple(visit(v) for v in getattr(op, "operands", ()))
            if (
                type(op) in ARITHMETIC
                and 1 + sum(v.weight for v in operands) <= 4096
                and max(v.depth for v in operands) < 128
            ):
                result = expression(ARITHMETIC[type(op)], *operands)
            else:
                result = Expr("argument", (Symbol("$ssa", SymbolKind.SSA, value, operands),))
        memo[value] = result
        return result

    return visit(value)


def infer(expr, facts):
    """Infer a sufficient nonnegative range; unknown is ``None``, never assumed.

    Facts can also describe coordinates for range queries. Such an environment
    must not be passed as the uniform-input scope to ``uniform``/``from_ssa``.
    """
    environment, memo = _facts(facts), {}

    def visit(expr):
        if expr in memo:
            return memo[expr]
        if expr.kind == "constant":
            value = expr.constant
            result = Fact(value, max(1, value), value) if 0 <= value <= INT64_MAX else None
        elif expr.kind == "argument":
            result = environment.get(expr.args[0])
        else:
            lhs, rhs = expr.args
            a, b = visit(lhs), visit(rhs)
            result = None
            if expr.kind == "max" and (a is None) != (b is None):
                # max(known nonnegative, unknown signed) keeps the known
                # lower bound; no upper bound or divisibility is implied.
                known = b if a is None else a
                result = Fact(known.minimum)
            if expr.kind == "rem" and b is not None and b.minimum > 0 and a is None:
                # Euclidean remainder has this range even when its dividend's
                # interval is unknown. Original unsigned definedness is still
                # checked independently at every actual SSA use.
                result = Fact(0, 1, None if b.maximum is None else b.maximum - 1)
            if a is not None and b is not None:
                maximum = None
                if expr.kind == "add":
                    if a.maximum is not None and b.maximum is not None:
                        maximum = a.maximum + b.maximum
                    result = Fact(a.minimum + b.minimum, gcd(a.multiple, b.multiple), maximum)
                elif expr.kind == "mul":
                    if a.maximum is not None and b.maximum is not None:
                        maximum = a.maximum * b.maximum
                    result = Fact(a.minimum * b.minimum, a.multiple * b.multiple, maximum)
                elif expr.kind in {"min", "max"}:
                    operation = min if expr.kind == "min" else max
                    if a.maximum is not None and b.maximum is not None:
                        maximum = operation(a.maximum, b.maximum)
                    elif expr.kind == "min":
                        maximum = a.maximum if a.maximum is not None else b.maximum
                    result = Fact(
                        operation(a.minimum, b.minimum), gcd(a.multiple, b.multiple), maximum
                    )
                elif expr.kind in DIVISION and b.minimum > 0:
                    if lhs == rhs:
                        value = 0 if expr.kind == "rem" else 1
                        result = Fact(value, 1, value)
                    elif expr.kind == "rem":
                        bounds = [
                            n
                            for n in (a.maximum, None if b.maximum is None else b.maximum - 1)
                            if n is not None
                        ]
                        maximum = min(bounds) if bounds else None
                        result = Fact(0, gcd(a.multiple, b.multiple), maximum)
                    else:
                        multiple = (
                            a.multiple // rhs.constant
                            if rhs.constant is not None and a.multiple % rhs.constant == 0
                            else 1
                        )
                        if expr.kind == "floordiv":
                            # A lower bound on the denominator cannot establish
                            # a lower bound on the quotient.
                            minimum = 0 if b.maximum is None else a.minimum // b.maximum
                            if a.maximum is not None:
                                maximum = a.maximum // b.minimum
                        else:
                            minimum = (
                                int(a.minimum > 0)
                                if b.maximum is None
                                else a.minimum // b.maximum + (a.minimum % b.maximum != 0)
                            )
                            if a.maximum is not None:
                                maximum = a.maximum // b.minimum + (a.maximum % b.minimum != 0)
                        result = Fact(minimum, multiple, maximum)
        memo[expr] = result
        return result

    return visit(expr)


def positive(expr, facts):
    fact = infer(expr, facts)
    return fact is not None and fact.minimum > 0


def uniform(expr, names_or_facts):
    """Validate the nonnegative grammar in a declared uniform-input scope.

    A set of names supplies no positivity information. A facts mapping
    additionally permits dynamic divisors proven positive by their ranges.
    """
    return infer(expr, names_or_facts) is not None


def bounded_by(lhs, rhs):
    """Sufficient nonnegative-domain bound; unknown is false, never assumed."""
    if lhs == rhs:
        return True
    if lhs.kind == "min":
        return any(bounded_by(arg, rhs) for arg in lhs.args)
    if rhs.kind == "max":
        return any(bounded_by(lhs, arg) for arg in rhs.args)
    return lhs.constant is not None and rhs.constant is not None and lhs.constant <= rhs.constant


def prefix(extent, tile):
    remaining = expression(
        "add", extent, expression("mul", literal(-tile), Expr("argument", ("$g",)))
    )
    return expression("min", literal(tile), remaining)


def bindings(expressions, *, checked=False):
    """Emit each uniform DAG node once; nested min/max never duplicate C text."""
    names, lines = {}, []
    definitions = 0

    def visit(expr):
        nonlocal definitions
        if expr in names:
            return
        if expr.kind in {"argument", "constant"}:
            names[expr] = expr.c()
            return
        for arg in expr.args:
            visit(arg)
        name = f"npu_uniform_{definitions}"
        definitions += 1
        if checked and expr.kind == "mul":
            a, b = (arg.c(names) for arg in expr.args)
            lines.append(
                f"  if ({b} != 0 && {a} > INT64_MAX / {b}) return -1; // uniform product overflow"
            )
        if checked and expr.kind == "add":
            a, b = (arg.c(names) for arg in expr.args)
            lines.append(f"  if ({a} > INT64_MAX - {b}) return -1; // uniform sum overflow")
        if checked and expr.kind in DIVISION:
            divisor = expr.args[1].c(names)
            lines.append(f"  if ({divisor} <= 0) return -1; // uniform divisor must be positive")
        lines.append(f"  const int64_t {name} = {expr.c(names)};")
        names[expr] = name

    for expr in expressions:
        if expr is not None:
            visit(expr)
    return names, lines
