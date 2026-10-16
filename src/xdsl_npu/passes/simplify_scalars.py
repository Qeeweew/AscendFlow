"""Local stock canonicalization of pure integer/index SSA arithmetic.

Only scalar arith roots with the Pure trait reach the stock patterns and DCE.
Physical unsigned division/remainder operations, floating-point arithmetic and all
control/native/address/notification operations remain outside that boundary.
No reassociation, code motion or cross-block commoning is performed.
"""

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerType, ModuleOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import Pure
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern

_ARITH_OPERATIONS = tuple(arith.Arith.operations)


class _PureIntegerArithmetic(RewritePattern):
    def __init__(self):
        self.pattern = GreedyRewritePatternApplier([CanonicalizationRewritePattern()])

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, _ARITH_OPERATIONS) or not op.has_trait(Pure):
            return
        if not all(
            isinstance(value.type, (IntegerType, IndexType))
            for value in (*op.operands, *op.results)
        ):
            return
        self.pattern.match_and_rewrite(op, rewriter)


def simplify_scalars(module: ModuleOp) -> None:
    """Simplify owned arithmetic in place without revisiting execution facts."""
    PatternRewriteWalker(_PureIntegerArithmetic()).rewrite_module(module)
