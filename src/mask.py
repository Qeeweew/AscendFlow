"""Exact symbolic boolean predicates over logical lane coordinates.

A predicate program is a small postfix integer expression DAG with three leaf
kinds: compile-time constants, uniform index SSA operands and logical lane
coordinates of one tile axis. Comparisons produce booleans; ``and``/``or``/
``not`` combine them. No floating point index table, whole-program value
flow, or arbitrary predicate-overlap proof is involved.

The same encoding is retained on the source op, the selected native task and
the final emitter. Node values are bounded so the emitted ``__int128``
evaluation is exact for every signed int64 operand; expressions beyond that
contract are rejected instead of silently wrapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr

CONST = 0
SCALAR = 1
LANE = 2
ADD = 3
SUB = 4
MULC = 5
DIVC = 6
MODC = 7
NEG = 8
CMP = 9
AND = 10
OR = 11
NOT = 12
BROADCAST = 13

VALUE_OPS = frozenset({CONST, SCALAR, LANE, ADD, SUB, MULC, DIVC, MODC, NEG})
BOOLEAN_OPS = frozenset({CMP, AND, OR, NOT, BROADCAST})
PERIODIC_OPS = frozenset({DIVC, MODC})
CMP_KINDS = ("eq", "ne", "lt", "le", "gt", "ge")

MAX_LANES = 1 << 22
# Every node value stays within +-2**125, and a selected comparison's two
# operands differ by less than 2**126, so signed 128-bit evaluation is exact
# for every int64 operand and bounded lane coordinate.
MAX_VALUE = 1 << 125
MAX_PAIR = 1 << 126
# Lane coefficients select the affine word strategy; the difference of two
# coefficients stays below 2**63 so the emitted int64 coefficient is exact
# and negation is defined.
MAX_COEFFICIENT = 1 << 61
# A periodic divisor is emitted as an exact signed int64 literal.
MAX_DIVISOR = (1 << 63) - 1
MASK_REPEAT_POLICY = "full_runs_255"
WORD_AFFINE = "affine64"
WORD_LANE = "lane64"
# How the element native walks its packed chunks. `rows` replays one
# unit-monotone comparison per logical row as clipped spans; `chunks` keeps the
# general per-64-lane replay.
ITERATION_CHUNKS = "chunks"
ITERATION_ROWS = "rows"
ITERATION_VALUES = (ITERATION_CHUNKS, ITERATION_ROWS)


class MaskError(ValueError):
    """An invalid, unbounded, or unsupported predicate program."""


@dataclass(frozen=True, slots=True)
class Node:
    op: int
    args: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Program:
    """One admitted predicate over a static logical lane domain."""

    shape: tuple[int, ...]
    node_shapes: tuple[tuple[int, ...], ...]
    nodes: tuple[Node, ...]

    @property
    def lanes(self):
        return _product(self.shape)

    @property
    def root(self):
        return self.nodes[-1]

    @property
    def is_periodic(self):
        return any(node.op in PERIODIC_OPS for node in self.nodes)

    def encode(self):
        return {
            "shape": ArrayAttr(IntAttr(size) for size in self.shape),
            "node_shapes": ArrayAttr(
                ArrayAttr(IntAttr(size) for size in shape) for shape in self.node_shapes
            ),
            "nodes": ArrayAttr(
                ArrayAttr(IntAttr(value) for value in (node.op, *node.args)) for node in self.nodes
            ),
        }

    @classmethod
    def decode(cls, attributes):
        shape = tuple(value.data for value in attributes["shape"])
        shapes = tuple(tuple(value.data for value in row) for row in attributes["node_shapes"])
        nodes = tuple(
            Node(row.data[0].data, tuple(value.data for value in row.data[1:]))
            for row in attributes["nodes"]
        )
        validate(nodes, shapes, shape)
        return cls(shape, shapes, nodes)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def broadcast_shapes(*shapes):
    """NumPy trailing-axis broadcast of positive static extents."""
    rank = max((len(shape) for shape in shapes), default=0)
    result = []
    for axis in range(rank):
        sizes = {
            shape[len(shape) - rank + axis]
            for shape in shapes
            if len(shape) - rank + axis >= 0 and shape[len(shape) - rank + axis] > 1
        }
        if len(sizes) > 1:
            return None
        result.append(sizes.pop() if sizes else 1)
    return tuple(result)


def node_shape(node, shapes):
    """Result shape of a validated node, given operand shapes."""
    if node.op in {CONST, SCALAR}:
        return ()
    if node.op in {ADD, SUB}:
        result = broadcast_shapes(shapes[node.args[0]], shapes[node.args[1]])
        if result is None:
            raise MaskError("coordinate operands do not broadcast")
        return result
    if node.op in {MULC, DIVC, MODC, NEG}:
        return shapes[node.args[0]]
    if node.op == CMP:
        result = broadcast_shapes(shapes[node.args[0]], shapes[node.args[1]])
        if result is None:
            raise MaskError("compared coordinates do not broadcast")
        return result
    if node.op in {AND, OR}:
        result = broadcast_shapes(shapes[node.args[0]], shapes[node.args[1]])
        if result is None:
            raise MaskError("combined predicates do not broadcast")
        return result
    return shapes[node.args[0]]


def children(node):
    """Actual intra-program node references; immediate fields are not nodes."""
    if node.op in {CONST, SCALAR, LANE}:
        return ()
    if node.op in {MULC, DIVC, MODC, NEG, NOT, BROADCAST}:
        return (node.args[0],)
    if node.op == CMP:
        return (node.args[0], node.args[1])
    return tuple(node.args)


def _scalar_free(nodes, index, memo=None):
    memo = {} if memo is None else memo
    if index in memo:
        return memo[index]
    node = nodes[index]
    result = node.op != SCALAR and all(_scalar_free(nodes, child, memo) for child in children(node))
    memo[index] = result
    return result


def _magnitude(bound):
    weight, constant, lane = bound
    return constant + weight * (1 << 63) + lane * MAX_LANES


def _bounds(nodes):
    """Per-node (scalar weight, constant magnitude, lane coefficient) bounds."""
    bounds, memo = [], {}
    for index, node in enumerate(nodes):
        op = node.op
        if op == CONST:
            bound = (0, abs(node.args[0]), 0)
        elif op == SCALAR:
            bound = (1, 0, 0)
        elif op == LANE:
            bound = (0, 0, 1)
        elif op in {ADD, SUB}:
            a, b = bounds[node.args[0]], bounds[node.args[1]]
            bound = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        elif op == NEG:
            bound = bounds[node.args[0]]
        elif op == MULC:
            factor = abs(node.args[1])
            if factor > (1 << 62):
                raise MaskError("coordinate multiplier exceeds the exact int64 range")
            a = bounds[node.args[0]]
            bound = (a[0] * factor, a[1] * factor, a[2] * factor)
        elif op in {DIVC, MODC}:
            if not _scalar_free(nodes, node.args[0], memo):
                raise MaskError(
                    "coordinate division and modulo require a scalar-free lane expression"
                )
            if not 0 < node.args[1] <= MAX_DIVISOR:
                raise MaskError("coordinate division and modulo need a positive int64 divisor")
            a = bounds[node.args[0]]
            bound = (0, a[1] + a[2] * MAX_LANES, 0)
        elif op == CMP:
            a, b = bounds[node.args[0]], bounds[node.args[1]]
            if _magnitude(a) + _magnitude(b) > MAX_PAIR:
                raise MaskError("compared coordinate values exceed exact signed 128-bit range")
            bound = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        else:
            a = bounds[node.args[0]]
            bound = (
                a
                if op in {NOT, BROADCAST}
                else tuple(x + y for x, y in zip(a, bounds[node.args[1]], strict=True))
            )
        if _magnitude(bound) > MAX_VALUE:
            raise MaskError("coordinate predicate exceeds exact signed 128-bit evaluation")
        if bound[2] > MAX_COEFFICIENT:
            raise MaskError(
                "coordinate lane coefficient exceeds the exact signed int64 strategy range"
            )
        bounds.append(bound)
    return bounds


def affine_axes(program):
    """Per value-node lane coefficients, or None where the node is not affine.

    One authoritative coefficient computation: the word recipe, the runtime
    tile bounds and the row-iteration selection all read this.
    """
    rank = len(program.shape)
    axes = [None] * len(program.nodes)
    for index, node in enumerate(program.nodes):
        op, args = node.op, node.args
        if op in {CONST, SCALAR}:
            axes[index] = (0,) * rank
        elif op == LANE:
            target = rank - len(program.node_shapes[index]) + args[0]
            axes[index] = tuple(1 if axis == target else 0 for axis in range(rank))
        elif op == NEG:
            base = axes[args[0]]
            axes[index] = None if base is None else tuple(-value for value in base)
        elif op == MULC:
            base = axes[args[0]]
            axes[index] = None if base is None else tuple(value * args[1] for value in base)
        elif op in {ADD, SUB}:
            left, right = axes[args[0]], axes[args[1]]
            sign = 1 if op == ADD else -1
            axes[index] = (
                None
                if left is None or right is None
                else tuple(a + sign * b for a, b in zip(left, right, strict=True))
            )
        else:
            axes[index] = None
    return axes


def comparison_root(program):
    """The compared node under shape-only broadcast wrappers, or None."""
    index = len(program.nodes) - 1
    node = program.nodes[index]
    while node.op == BROADCAST:
        index = node.args[0]
        node = program.nodes[index]
    return node if node.op == CMP else None


def iteration_strategy(program):
    """Selected iteration recipe; total over admitted programs.

    `rows` needs a single comparison (through shape-only broadcasts) whose two
    operands are affine and whose last-axis lane coefficient difference is +-1,
    with a whole-64 last axis. Every other axis is then fixed within a row and
    contributes an exact per-row base. Everything else keeps `chunks`.
    """
    if program.is_periodic or program.shape[-1] % 64:
        return ITERATION_CHUNKS
    node = comparison_root(program)
    if node is None:
        return ITERATION_CHUNKS
    axes = affine_axes(program)
    left, right = axes[node.args[0]], axes[node.args[1]]
    if left is None or right is None:
        return ITERATION_CHUNKS
    if left[-1] - right[-1] not in {1, -1}:
        return ITERATION_CHUNKS
    return ITERATION_ROWS


def word_strategy(program):
    """Selected mask-word recipe; a total function over admitted programs."""
    if program.is_periodic:
        return WORD_LANE
    for axis in range(len(program.shape) - 1):
        stride = 1
        for size in program.shape[axis + 1 :]:
            stride *= size
        if stride % 64:
            return WORD_LANE
    return WORD_AFFINE


def validate(nodes, shapes, shape):
    """Check one program's structure, typing, shapes and exact-integer range."""
    if not nodes:
        raise MaskError("predicate program needs at least one node")
    if len(nodes) != len(shapes):
        raise MaskError("predicate program lost its per-node shapes")
    if not shape or any(type(size) is not int or not 0 < size <= MAX_LANES for size in shape):
        raise MaskError("predicate needs positive bounded static lane extents")
    expected = []
    for index, node in enumerate(nodes):
        args = node.args
        if node.op == CONST:
            if len(args) != 1 or not -(1 << 63) <= args[0] < (1 << 63):
                raise MaskError("coordinate constant exceeds signed int64")
            expected.append(())
        elif node.op == SCALAR:
            if len(args) != 1 or args[0] < 0:
                raise MaskError("predicate scalar atom needs an operand index")
            expected.append(())
        elif node.op == LANE:
            stored = tuple(shapes[index])
            if (
                len(args) != 1
                or not 0 <= args[0] < len(stored)
                or len(stored) > len(shape)
                or stored[args[0]] <= 1
                or any(size != 1 for axis, size in enumerate(stored) if axis != args[0])
            ):
                raise MaskError("lane coordinate needs one non-singleton axis")
            target = len(shape) - len(stored) + args[0]
            if shape[target] != stored[args[0]]:
                raise MaskError("lane coordinate exceeds its lane domain")
            expected.append(stored)
        elif node.op in {ADD, SUB}:
            if len(args) != 2 or any(arg >= index for arg in args):
                raise MaskError("coordinate arithmetic needs earlier operands")
            expected.append(node_shape(node, expected))
        elif node.op in {MULC, DIVC, MODC}:
            if len(args) != 2 or args[0] >= index or args[1] == 0:
                raise MaskError("coordinate constant operator needs an earlier operand")
            expected.append(expected[args[0]])
        elif node.op == NEG:
            if len(args) != 1 or args[0] >= index:
                raise MaskError("coordinate negation needs an earlier operand")
            expected.append(expected[args[0]])
        elif node.op == CMP:
            if (
                len(args) != 3
                or any(arg >= index for arg in args[:2])
                or not 0 <= args[2] < len(CMP_KINDS)
                or nodes[args[0]].op in BOOLEAN_OPS
                or nodes[args[1]].op in BOOLEAN_OPS
            ):
                raise MaskError("coordinate comparison needs two values and one kind")
            expected.append(node_shape(node, expected))
        elif node.op in {AND, OR}:
            if (
                len(args) != 2
                or any(arg >= index for arg in args)
                or any(nodes[arg].op not in BOOLEAN_OPS for arg in args)
            ):
                raise MaskError("boolean combination needs earlier predicates")
            expected.append(node_shape(node, expected))
        elif node.op == NOT:
            if len(args) != 1 or args[0] >= index or nodes[args[0]].op not in BOOLEAN_OPS:
                raise MaskError("predicate negation needs an earlier predicate")
            expected.append(expected[args[0]])
        elif node.op == BROADCAST:
            if len(args) != 1 or args[0] >= index or nodes[args[0]].op not in BOOLEAN_OPS:
                raise MaskError("predicate broadcast needs an earlier predicate")
            stored = tuple(shapes[index])
            if broadcast_shapes(expected[args[0]], stored) != stored:
                raise MaskError("predicate broadcast shape is not a singleton extension")
            expected.append(stored)
        else:
            raise MaskError("unknown predicate node")
        if tuple(shapes[index]) != expected[-1]:
            raise MaskError(
                f"predicate node {index} {node.op} {node.args} shape {tuple(shapes[index])} "
                f"does not match operands {expected[-1]}"
            )
    if nodes[-1].op not in BOOLEAN_OPS:
        raise MaskError("predicate result must be a boolean node")
    if tuple(shape) != expected[-1]:
        raise MaskError("predicate lane domain does not match its result shape")
    _bounds(nodes)
    return tuple(expected)


def scalar_operand_count(nodes, *, limit=256):
    count = max((node.args[0] + 1 for node in nodes if node.op == SCALAR), default=0)
    if count > limit:
        raise MaskError("predicate references too many scalar atoms")
    return count


def shift(nodes, offset):
    """Offset every intra-program node reference by an earlier program size."""
    result = []
    for node in nodes:
        args = node.args
        if node.op in {CONST, SCALAR, LANE}:
            shifted = args
        elif node.op in {MULC, DIVC, MODC}:
            shifted = (args[0] + offset, args[1])
        elif node.op in {NEG, NOT, BROADCAST}:
            shifted = (args[0] + offset,)
        elif node.op == CMP:
            shifted = (args[0] + offset, args[1] + offset, args[2])
        else:
            shifted = tuple(arg + offset for arg in args)
        result.append(Node(node.op, shifted))
    return tuple(result)


def rebase(nodes, mapping):
    """Rewrite scalar atom indexes through an explicit old-to-new mapping."""
    result = []
    for node in nodes:
        args = node.args
        if node.op == SCALAR:
            args = (mapping[args[0]],)
        result.append(Node(node.op, args))
    return tuple(result)


def append_boolean(nodes, shapes, op, operands):
    """Append one boolean node over earlier node indexes."""
    if op == NOT:
        result_shape = shapes[operands[0]]
    else:
        result_shape = broadcast_shapes(shapes[operands[0]], shapes[operands[1]])
        if result_shape is None:
            raise MaskError("combined predicates do not broadcast")
    return (
        (*nodes, Node(op, tuple(operands))),
        (*shapes, result_shape),
    )
