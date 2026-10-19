"""Independent source-coordinate integer evaluation, without hardware scheduling claims."""

from xdsl.dialects import arith


def evaluate(value, bindings):
    """Independent integer interpretation used only to inspect source coordinates."""
    if value in bindings:
        return bindings[value]
    op = value.owner
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    lhs, rhs = (evaluate(operand, bindings) for operand in op.operands)
    if isinstance(op, arith.AddiOp):
        return lhs + rhs
    if isinstance(op, arith.SubiOp):
        return lhs - rhs
    if isinstance(op, arith.MuliOp):
        return lhs * rhs
    if isinstance(op, arith.DivUIOp):
        assert lhs >= 0 and rhs > 0
        return lhs // rhs
    if isinstance(op, arith.RemUIOp):
        assert lhs >= 0 and rhs > 0
        return lhs % rhs
    if isinstance(op, arith.CeilDivUIOp):
        assert lhs >= 0 and rhs > 0
        return (lhs + rhs - 1) // rhs
    if isinstance(op, arith.MaxSIOp):
        return max(lhs, rhs)
    if isinstance(op, arith.MinSIOp):
        return min(lhs, rhs)
    raise AssertionError(op.name)
