"""Physical serial invocation and explicit per-owner GM frame contracts."""

from dataclasses import dataclass

from xdsl.dialects import scf

from .ir import BufferOp, FrameOp, OnceOp, StreamOp
from .model import Storage, optional_int, require
from .scalar import Expr, expression, literal


def factors(expr):
    """Canonical nonnegative product; no distribution or nonlinear solver."""
    if expr.kind == "mul":
        left, right = (factors(arg) for arg in expr.args)
        return left[0] * right[0], tuple(sorted(left[1] + right[1], key=repr))
    if expr.constant is not None:
        return expr.constant, ()
    return 1, (expr,)


@dataclass(frozen=True)
class Frame:
    op: FrameOp
    root: BufferOp
    name: str
    extent: Expr
    scale: int


@dataclass(frozen=True)
class Invocation:
    op: scf.ForOp
    bound: Expr
    frames: tuple[Frame, ...]


def extract_invocation(kernel, uniform_values, *, sequence=False):
    parents = [op for op in kernel.body.block.ops if isinstance(op, scf.ForOp)]
    if not parents:
        return None
    require(len(parents) == 1, "one serial invocation scope currently required")
    parent = parents[0]
    require(
        not parent.iter_args
        and not parent.results
        and parent.lb in uniform_values
        and uniform_values[parent.lb] == literal(0)
        and parent.step in uniform_values
        and uniform_values[parent.step] == literal(1)
        and parent.ub in uniform_values
        and len(parent.body.block.args) == 1,
        "invocations require an immutable zero-based unit-step bound and no scalar carry",
    )
    bound = uniform_values[parent.ub]
    frames, used = [], set()
    streams = []
    for op in parent.body.block.ops:
        if isinstance(op, FrameOp):
            root = op.source.owner
            require(
                isinstance(root, BufferOp)
                and root.parent_op() is kernel
                and root.external.data == 1
                and root.ref.type == op.ref.type
                and op.invocation is parent.body.block.args[0]
                and op.extent in uniform_values
                and op.scale.data > 0
                and op.scale.data % 32 == 0
                and root not in used
                and not streams,
                "GM frame must bind its actual root, invocation coordinate and aligned extent",
            )
            extent = uniform_values[op.extent]
            actual = (
                literal(root.nbytes.data)
                if root.extent_count is None
                else expression(
                    "mul", uniform_values[root.extent_count], literal(root.count_scale.data)
                )
            )
            expected = expression(
                "mul",
                bound,
                expression("mul", extent, literal(optional_int(kernel.blocks, 1) * op.scale.data)),
            )
            require(
                factors(actual) == factors(expected),
                "GM frame does not partition its declared root extent",
            )
            frames.append(Frame(op, root, "npu_frame_" + root.sym_name.data, extent, op.scale.data))
            used.add(root)
        elif isinstance(op, StreamOp) or sequence and isinstance(op, OnceOp):
            streams.append(op)
        else:
            require(
                isinstance(op, scf.YieldOp) and not op.arguments,
                f"unsupported invocation effect {op.name}",
            )
    require(
        bool(streams) if sequence else len(streams) == 1,
        "each invocation requires its actual retained execution regions",
    )
    roots = {op for op in kernel.body.block.ops if isinstance(op, BufferOp) and op.external.data}
    require(used == roots, "every invocation GM root requires an explicit frame")
    return Invocation(parent, bound, tuple(frames))


def frame_storage(frame):
    typ = frame.root.ref.type
    constant = frame.extent.constant
    return Storage(
        frame.name,
        "GM",
        0 if constant is None else constant * frame.scale,
        typ.dtype.data,
        typ.layout.data,
        True,
        bool(frame.root.initialized.data),
        frame.scale if constant is None else 0,
        frame.extent if constant is None else None,
    )
