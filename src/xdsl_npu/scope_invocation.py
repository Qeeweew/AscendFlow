"""Actual outer query clocks around retained KV parent streams.

This describes execution and address domains only. It supplies no completion,
cross-query storage lifetime, event reuse, or invocation-frame contract.
"""

from dataclasses import dataclass

from xdsl.dialects import scf
from xdsl.dialects.builtin import IndexType

from .cyclic import CyclicDomain
from .ir import CyclicCountOp, StreamOp
from .model import require
from .scalar import Expr, expression, literal, uniform


@dataclass(frozen=True)
class ScopeInvocation:
    op: scf.ForOp
    bound: Expr
    cyclic: CyclicDomain | None = None


def extract_scope_invocation(kernel, uniforms, facts, groups):
    """Return the real serial query and its unique parent Stream, if present."""
    serial = [op for op in kernel.body.block.ops if isinstance(op, scf.ForOp)]
    streams = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
    counts = [op for op in kernel.body.block.ops if isinstance(op, CyclicCountOp)]
    if not serial:
        require(not counts, "cyclic query count requires its actual serial query loop")
        require(len(streams) == 1, "nested actor requires one actual parent stream")
        return None, streams[0]
    require(len(serial) == 1 and not streams, "nested actor requires one serial query scope")
    query = serial[0]
    require(
        not query.iter_args
        and not query.results
        and len(query.body.block.args) == 1
        and isinstance(query.body.block.args[0].type, IndexType)
        and uniforms.get(query.lb) == literal(0)
        and uniforms.get(query.step) == literal(1),
        "serial query requires actual zero/unit bounds, one clock and no carries",
    )
    if isinstance(query.ub.owner, CyclicCountOp):
        count = query.ub.owner
        require(
            counts == [count]
            and count.parent_op() is kernel
            and query.ub is count.count
            and count.total in uniforms,
            "cyclic query count must bind an actual root-shared uniform total",
        )
        total = uniforms[count.total]
        require(uniform(total, facts), "cyclic query total must be nonnegative and uniform")
        cyclic = CyclicDomain(total, groups)
        bound = cyclic.local_bound
    else:
        require(
            not counts and query.ub in uniforms,
            "serial query bound must be an actual uniform or cyclic query count",
        )
        cyclic, bound = None, uniforms[query.ub]
    operations = tuple(query.body.block.ops)
    require(
        len(operations) == 2
        and isinstance(operations[0], StreamOp)
        and isinstance(operations[1], scf.YieldOp)
        and not operations[1].arguments,
        "serial query body must retain exactly its parent stream and empty yield",
    )
    return ScopeInvocation(query, bound, cyclic), operations[0]


def query_domains(invocation):
    """Inclusive active-query ranges, with the joint cyclic owner coordinate."""
    query = Expr("argument", ("$parent1",))
    result = []
    if invocation.cyclic is not None:
        domain = invocation.cyclic
        flat = expression(
            "add",
            Expr("argument", ("$core",)),
            expression("mul", query, literal(domain.groups)),
        )
        result.append((flat, literal(0), expression("sub", domain.total, literal(1))))
    result.append((query, literal(0), expression("sub", invocation.bound, literal(1))))
    return result


def verify_query_domains(invocations, kernels):
    require(
        not invocations or invocations.keys() == kernels.keys(),
        "nested actors disagree on the actual serial query scope",
    )
    signatures = {(value.bound, value.cyclic) for value in invocations.values()}
    require(len(signatures) <= 1, "nested actors disagree on the query domain or cyclic ownership")
