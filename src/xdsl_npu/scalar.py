"""Small, explicit index expressions shared by access proofs and ASC emission.

This is not a general integer solver. Uniform bounds currently admit nonnegative
arguments, min/max and constant-divisor ceildiv. Affine expressions additionally
describe a task's valid prefix; unsupported domains fail explicitly.
"""

from dataclasses import dataclass, field

from xdsl.dialects import arith

from .ir import RuntimeSizeOp
from .model import require


@dataclass(frozen=True)
class Expr:
    kind: str
    args: tuple
    weight: int = field(init=False, compare=False, repr=False)
    depth: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
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
        return {
            "add": lambda: a + b,
            "mul": lambda: a * b,
            "min": lambda: min(a, b),
            "max": lambda: max(a, b),
            "ceildiv": lambda: (a + b - 1) // b,
        }[self.kind]()

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
        # Quotient/remainder spelling avoids the n + divisor - 1 overflow.
        return f"({a} / {b} + ({a} % {b} != 0))"


def literal(value):
    return Expr("constant", (value,))


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
    arith.CeilDivUIOp: "ceildiv",
}


def from_ssa(value, *, coordinates=None, memo=None):
    from .source_ir import SizeOp

    memo = {} if memo is None else memo
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
        require(type(op) in ARITHMETIC, "unsupported scalar expression in access/bound contract")
        a, b = (from_ssa(v, coordinates=coordinates, memo=memo) for v in op.operands)
        if ARITHMETIC[type(op)] == "ceildiv":
            require(
                b.constant is not None and b.constant > 0,
                "ceildiv divisor must be positive and constant",
            )
        result = expression(ARITHMETIC[type(op)], a, b)
    memo[value] = result
    return result


def uniform(expr, names):
    if expr.kind == "constant":
        return 0 <= expr.constant <= (1 << 63) - 1
    if expr.kind == "argument":
        return expr.args[0] in names
    if expr.kind in {"min", "max"}:
        return all(uniform(arg, names) for arg in expr.args)
    if expr.kind == "mul":
        return all(uniform(arg, names) for arg in expr.args)
    return (
        expr.kind == "ceildiv"
        and uniform(expr.args[0], names)
        and expr.args[1].constant is not None
        and expr.args[1].constant > 0
    )


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
        lines.append(f"  const int64_t {name} = {expr.c(names)};")
        names[expr] = name

    for expr in expressions:
        if expr is not None:
            visit(expr)
    return names, lines
