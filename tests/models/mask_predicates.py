"""Independent source predicate truth values and scalar mask-word expectations."""

from __future__ import annotations

from itertools import product

from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr

from ascendflow.dialects.io import parse
from ascendflow.dialects.source import MaskOp, SizeOp


def _scalar(value, environment, memo):
    if value in memo:
        return memo[value]
    owner = value.owner
    assert isinstance(
        owner,
        (
            arith.ConstantOp,
            arith.AddiOp,
            arith.SubiOp,
            arith.MuliOp,
            arith.DivUIOp,
            arith.RemUIOp,
            arith.MinSIOp,
            arith.MaxSIOp,
            arith.CeilDivUIOp,
            arith.CmpiOp,
            SizeOp,
        ),
    ), type(owner)
    if isinstance(owner, SizeOp):
        result = environment[owner.sym_name.data]
    elif isinstance(owner, arith.ConstantOp):
        assert isinstance(owner.value, IntegerAttr)
        result = owner.value.value.data
    else:
        operands = [_scalar(item, environment, memo) for item in owner.operands]
        if isinstance(owner, arith.AddiOp):
            result = operands[0] + operands[1]
        elif isinstance(owner, arith.SubiOp):
            result = operands[0] - operands[1]
        elif isinstance(owner, arith.MuliOp):
            result = operands[0] * operands[1]
        elif isinstance(owner, arith.DivUIOp):
            result = operands[0] // operands[1]
        elif isinstance(owner, arith.RemUIOp):
            result = operands[0] % operands[1]
        elif isinstance(owner, arith.MinSIOp):
            result = min(operands)
        elif isinstance(owner, arith.MaxSIOp):
            result = max(operands)
        elif isinstance(owner, arith.CeilDivUIOp):
            result = -(-operands[0] // operands[1])
        else:
            assert isinstance(owner, arith.CmpiOp)
            predicates = (
                lambda a, b: a == b,
                lambda a, b: a != b,
                lambda a, b: a < b,
                lambda a, b: a <= b,
                lambda a, b: a > b,
                lambda a, b: a >= b,
            )
            result = predicates[owner.predicate.value.data](*operands)
    memo[value] = result
    return result


def _evaluate(program, mask, index, environment, memo):
    scalar_memo = {}
    values = []
    for node, stored in zip(program.nodes, program.node_shapes, strict=True):
        args = node.args
        if node.op == 0:  # const
            values.append(args[0])
        elif node.op == 1:  # scalar
            values.append(_scalar(mask.scalars[args[0]], environment, scalar_memo))
        elif node.op == 2:  # lane
            target = len(program.shape) - len(stored) + args[0]
            values.append(index[target] if stored[args[0]] > 1 else 0)
        elif node.op == 3:
            values.append(values[args[0]] + values[args[1]])
        elif node.op == 4:
            values.append(values[args[0]] - values[args[1]])
        elif node.op == 5:
            values.append(values[args[0]] * args[1])
        elif node.op == 6:
            values.append(values[args[0]] // args[1])
        elif node.op == 7:
            values.append(values[args[0]] % args[1])
        elif node.op == 8:
            values.append(-values[args[0]])
        elif node.op == 9:
            left, right, kind = values[args[0]], values[args[1]], args[2]
            values.append(
                (
                    left == right,
                    left != right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                )[kind]
            )
        elif node.op == 10:
            values.append(values[args[0]] and values[args[1]])
        elif node.op == 11:
            values.append(values[args[0]] or values[args[1]])
        elif node.op == 12:
            values.append(not values[args[0]])
        else:
            assert node.op == 13
            values.append(values[args[0]])
    return values[-1]


def mask_values(result, environment):
    """All source predicate programs with their exact per-lane truth values."""
    module = parse(result.source_ir)
    truth = []
    for operation in module.walk():
        if isinstance(operation, MaskOp):
            program = operation.program()
            lanes = {
                index: _evaluate(program, operation, index, environment, {})
                for index in product(*[range(size) for size in program.shape])
            }
            truth.append((program.shape, lanes))
    return truth


def _expected_word(a, b, kind):
    ops = (
        lambda value: value == 0,
        lambda value: value != 0,
        lambda value: value < 0,
        lambda value: value <= 0,
        lambda value: value > 0,
        lambda value: value >= 0,
    )[kind]
    word = 0
    for lane in range(64):
        if ops(a * lane + b):
            word |= 1 << lane
    return word
