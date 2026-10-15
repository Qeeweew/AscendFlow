"""Use xDSL's standard passes for pure integer expressions during ASC emission.

One scope belongs to adjacent commands with the same activation. Only arith operations enter
the temporary SSA function; asynchronous calls never enter canonicalize/CSE.
Placeholders connect its results to C operands, without parsing C expressions.
All divisors are positive constants, including in unselected event branches.
"""

import re
from collections import Counter

from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import Builtin, ModuleOp, i1, i64
from xdsl.ir import Block, Region, SSAValue
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

from .model import require


class ScalarScope:
    _token = re.compile(r"@npu_scalar_(\d+)@")

    def __init__(self):
        self.block = Block()
        self.arguments = {}
        self.outputs = []

    def value(self, value):
        if isinstance(value, SSAValue):
            return value
        if isinstance(value, int):
            return self.op(arith.ConstantOp.from_int_and_width(value, i64))
        if value == "0":
            return self.value(0)
        require(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z_]\w*", value),
            "ASC scalar input must be an integer, SSA value or identifier",
        )
        if value not in self.arguments:
            self.arguments[value] = self.block.insert_arg(i64, len(self.arguments))
        return self.arguments[value]

    def op(self, op):
        self.block.add_op(op)
        return op.results[0]

    def add(self, a, b):
        return self.op(arith.AddiOp(self.value(a), self.value(b)))

    def sub(self, a, b):
        return self.op(arith.SubiOp(self.value(a), self.value(b)))

    def mul(self, a, b):
        return self.op(arith.MuliOp(self.value(a), self.value(b)))

    def div(self, a, divisor):
        require(type(divisor) is int and divisor > 0, "ASC scalar divisor must be positive")
        return self.op(arith.DivSIOp(self.value(a), self.value(divisor)))

    def rem(self, a, divisor):
        require(type(divisor) is int and divisor > 0, "ASC scalar divisor must be positive")
        return self.op(arith.RemSIOp(self.value(a), self.value(divisor)))

    def select_lt(self, a, b, lhs, rhs):
        cond = self.op(arith.CmpiOp(self.value(a), self.value(b), "slt"))
        return self.op(arith.SelectOp(cond, self.value(lhs), self.value(rhs)))

    def select_eq(self, a, b, lhs, rhs):
        cond = self.op(arith.CmpiOp(self.value(a), self.value(b), "eq"))
        return self.op(arith.SelectOp(cond, self.value(lhs), self.value(rhs)))

    def use(self, value):
        token = f"@npu_scalar_{len(self.outputs)}@"
        self.outputs.append(self.value(value))
        return token

    def finish(self, lines):
        """Optimize once, then materialize shared SSA nodes as local constants."""
        text = "\n".join(lines)
        occurrences = Counter(int(m[1]) for m in self._token.finditer(text))
        if not occurrences:
            return lines
        indices = sorted(occurrences)
        ret = func.ReturnOp(*(self.outputs[i] for i in indices))
        self.block.add_op(ret)
        function = func.FuncOp(
            "asc_scalars",
            (self.block.arg_types, ret.arguments.types),
            Region(self.block),
        )
        module = ModuleOp([function])
        ctx = Context()
        for dialect in (Builtin, arith.Arith, func.Func):
            ctx.load_dialect(dialect)
        CanonicalizePass().apply(ctx, module)
        CommonSubexpressionElimination().apply(ctx, module)
        CanonicalizePass().apply(ctx, module)

        references = Counter()
        for i, value in zip(indices, ret.arguments):
            references[value] += occurrences[i]
        for op in self.block.ops:
            if op is not ret:
                references.update(op.operands)
        names = {value: name for name, value in self.arguments.items()}
        occupied = set(re.findall(r"\b[A-Za-z_]\w*\b", self._token.sub("", text))) | set(
            self.arguments
        )
        declarations = []
        serial = 0
        binary = {
            arith.AddiOp: "+",
            arith.SubiOp: "-",
            arith.MuliOp: "*",
            arith.DivSIOp: "/",
            arith.RemSIOp: "%",
        }
        for op in self.block.ops:
            if op is ret:
                break
            result = op.results[0]
            if isinstance(op, arith.ConstantOp):
                value = op.value.value.data
                names[result] = (
                    ("true" if value else "false")
                    if result.type == i1
                    else ("INT64_MIN" if value == -(1 << 63) else str(value))
                )
                continue
            operands = [names[v] for v in op.operands]
            if type(op) in binary:
                a, b = operands
                expression = f"({a} {binary[type(op)]} {b})"
            elif isinstance(op, arith.CmpiOp):
                predicate = ("==", "!=", "<", "<=", ">", ">=")[op.predicate.value.data]
                expression = f"({operands[0]} {predicate} {operands[1]})"
            elif isinstance(op, arith.SelectOp):
                expression = f"({operands[0]} ? {operands[1]} : {operands[2]})"
            else:
                raise ValueError(
                    f"unexpected ASC scalar operation after standard passes: {op.name}"
                )
            if references[result] > 1:
                while (name := f"npu_scalar_{serial}") in occupied:
                    serial += 1
                serial += 1
                occupied.add(name)
                ctype = "bool" if result.type == i1 else "int64_t"
                declarations.append(f"const {ctype} {name} = {expression};")
                names[result] = name
            else:
                names[result] = expression
        replacements = {i: names[v] for i, v in zip(indices, ret.arguments)}
        return declarations + [
            self._token.sub(lambda m: replacements[int(m[1])], line) for line in lines
        ]
