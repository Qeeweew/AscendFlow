"""Small, explicit index expressions shared by access proofs and ASC emission.

This is not a general integer solver. Uniform index expressions use nonnegative
integers and provably positive divisors. Facts belong to a proof environment,
not expression identity. Affine expressions additionally describe a task's valid
prefix; unsupported domains fail explicitly.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import gcd

from xdsl.dialects import arith

from .diagnostics import CompileError, require
from .dialects.physical import RuntimeSizeOp

INT64_MAX = (1 << 63) - 1
DIVISION = {"floordiv", "rem", "ceildiv"}
BINARY = {"add", "mul", "min", "max", *DIVISION}


@dataclass(frozen=True)
class Fact:
    """A nonnegative integer range whose values are multiples of ``multiple``.

    Bounds describe mathematical integers. Derived products may exceed int64;
    the checked launcher rejects overflowing evaluations before dispatch.
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

    facts = {}
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


@dataclass(frozen=True)
class Expr:
    kind: str
    args: tuple
    weight: int = field(init=False, compare=False, repr=False)
    depth: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        require(
            self.kind in {"constant", "argument", *BINARY},
            f"unsupported scalar expression kind: {self.kind}",
        )
        if self.kind == "constant":
            require(len(self.args) == 1 and type(self.args[0]) is int, "invalid scalar constant")
        elif self.kind == "argument":
            require(
                len(self.args) == 1 and isinstance(self.args[0], str), "invalid scalar argument"
            )
        else:
            require(
                len(self.args) == 2 and all(isinstance(arg, Expr) for arg in self.args),
                "binary scalar expression needs two expressions",
            )
        if self.kind in DIVISION:
            numerator, divisor = self.args
            require(
                divisor.constant is None or divisor.constant > 0,
                "scalar divisor must be positive",
            )
            require(
                numerator.constant is None or numerator.constant >= 0,
                "scalar division numerator must be nonnegative",
            )
        children = [arg for arg in self.args if isinstance(arg, Expr)]
        weight = 1 + sum(arg.weight for arg in children)
        depth = 1 + max((arg.depth for arg in children), default=0)
        require(
            weight <= 4096 and depth <= 128,
            "scalar expression proof budget exceeded (4096 terms / depth 128)",
        )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "depth", depth)

    @property
    def constant(self):
        return self.args[0] if self.kind == "constant" else None

    def evaluate(self, values):
        if self.kind == "constant":
            return self.args[0]
        if self.kind == "argument":
            return values[self.args[0]]
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
            return self.args[0]
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


def literal(value):
    return Expr("constant", (value,))


def substitute(value: Expr, bindings: Mapping[Expr, Expr]) -> Expr:
    """Simultaneous coordinate substitution, without assumptions about its domain.

    Replacements are not recursively substituted. Callers retain the actual
    loop/guard scope that makes any division or selected occurrence valid.
    This shared operation supplies no alias or completion proof.
    """
    if value in bindings:
        return bindings[value]
    if value.kind in {"constant", "argument"}:
        return value
    return expression(value.kind, *(substitute(child, bindings) for child in value.args))


def expression(kind, a, b):
    if kind == "sub":
        return expression("add", a, expression("mul", literal(-1), b))
    if a.constant is not None and b.constant is not None:
        return literal(Expr(kind, (a, b)).evaluate({}))
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
    if kind in {"add", "mul", "min", "max"} and repr(a) > repr(b):
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


def from_ssa(value, *, coordinates=None, memo=None, facts=None):
    """Extract an expression, validating every divisor in its uniform scope.

    Coordinate names may occur in numerators. ``facts`` only declares uniform
    inputs; coordinate range facts belong to a separate call to ``infer``.
    """
    from .dialects.source import SizeOp

    memo = {} if memo is None else memo
    environment = {} if facts is None else _facts(facts)

    def visit(value):
        if value in memo:
            return memo[value]
        if coordinates and value in coordinates:
            result = Expr("argument", (coordinates[value],))
        elif isinstance(value.owner, (SizeOp, RuntimeSizeOp)):
            result = Expr("argument", (value.owner.sym_name.data,))
        elif isinstance(value.owner, arith.ConstantOp):
            result = literal(value.owner.value.value.data)
        else:
            op = value.owner
            require(
                type(op) in ARITHMETIC, "unsupported scalar expression in access/bound contract"
            )
            a, b = (visit(v) for v in op.operands)
            result = expression(ARITHMETIC[type(op)], a, b)
        memo[value] = result
        return result

    result = visit(value)
    seen = set()

    def validate(expr):
        if expr in seen:
            return
        seen.add(expr)
        if expr.kind in DIVISION:
            divisor = expr.args[1]
            require(
                uniform(divisor, environment) and positive(divisor, environment),
                "scalar divisor must be uniform and provably positive",
            )
        for arg in expr.args:
            if isinstance(arg, Expr):
                validate(arg)

    # Validate cached expressions too: an extraction cache is not a certificate
    # that a divisor remains positive in another proof environment.
    validate(result)
    return result


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
