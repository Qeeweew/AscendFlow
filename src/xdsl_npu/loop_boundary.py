"""Visit retained physical boundary regions without instantiating iterations."""

from .diagnostics import require
from .dialects.physical import BoundaryOp, EndOp


def stream_operations(stream):
    for op in stream.body.block.ops:
        if not isinstance(op, BoundaryOp):
            yield op, False
            continue
        require(
            op.kind.data == "last"
            and op.generation is stream.body.block.args[0]
            and not op.body.block.args
            and isinstance(op.body.block.last_op, EndOp),
            "boundary requires last in the enclosing stream clock and a terminated body",
        )
        for child in op.body.block.ops:
            require(
                not isinstance(child, BoundaryOp), "nested boundary predicates need region analysis"
            )
            yield child, True
