"""Merge ordered conditional packets in the final, owned physical SSA tree.

Loop/branch premises describe integer domains, never runtime phase flags. A
notification packet is rewritten only when its scalar evaluations are total at
the packet entry. Mutually exclusive alternatives may share an instruction and
an event expression. Consecutive equal pipe fences share one completion prefix
and may also merge overlapping alternatives; event contributions keep their
original multiplicity. Native packets and loops end the notification packet.
"""

from dataclasses import dataclass
from functools import reduce
from itertools import product
from math import gcd, inf

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, StringAttr, i1
from xdsl.ir import Block, Region

from ..dialects.physical import BufferOp, CoreIdOp, RuntimeSizeOp
from ..dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from ..target.ascend910b import SyncKind
from .simplify_scalars import (
    _UNSIGNED,
    ScalarFacts,
    _affine,
    _bounded_interval,
    _constant,
    _fold,
    _interval,
    _key,
    _limits,
    _linear_operation,
    _modulo,
    _place_invariant,
    _replace,
    _scalar_arithmetic,
    _Scope,
    _seed,
    cleanup_scalar_syntax,
)


@dataclass(frozen=True)
class _Ranges:
    intervals: tuple = ((-inf, inf),)

    def intersect(self, other):
        return _Ranges(
            tuple(
                (max(a, c), min(b, d))
                for a, b in self.intervals
                for c, d in other.intervals
                if max(a, c) <= min(b, d)
            )
        )

    def union(self, other):
        result = []
        for low, high in sorted((*self.intervals, *other.intervals)):
            if result and low <= result[-1][1] + 1:
                result[-1] = result[-1][0], max(high, result[-1][1])
            else:
                result.append((low, high))
        return _Ranges(tuple(result))

    def contains(self, other):
        return self.intersect(other) == other

    def complement(self):
        result, low = [], -inf
        for a, b in self.intervals:
            if low < a:
                result.append((low, a - 1))
            low = b + 1
        if low < inf:
            result.append((low, inf))
        return _Ranges(tuple(result))


_FULL = _Ranges()
_EMPTY = _Ranges(())


class _Unchanged(Exception):
    """A scalar cannot be evaluated in the common packet entry domain."""


def _empty_region(region):
    return all(
        isinstance(op, scf.YieldOp) and not op.arguments
        for block in region.blocks
        for op in block.ops
    )


@dataclass
class ConditionWork:
    packets: int = 0
    input_notifications: int = 0
    output_notifications: int = 0
    shared_blocks: int = 0
    removed_fences: int = 0


class _Domains:
    """Local interval unions with the actual loop difference constraints.

    A cube is a conjunction of ranges of affine SSA quantities. Difference
    closure retains g<N without separating g and N. Other affine quantities
    use conservative intervals. No integer values or phase products are
    enumerated; residue reasoning is bounded by a real event bank's width.
    """

    def __init__(self, scope, ranks):
        self.scope = scope
        self.ranks = ranks
        self.affines, self.quantities, self.tests = {}, {}, {}
        self.bounds_cache = {}
        self.intervals = {}
        self.modulos = {}
        self.normalized = {}
        self.unions = {}
        self.closures = {}
        self.residue_pairs = {}
        self.resolving = set()
        self.base = {}
        visible = set()
        current = scope
        while current is not None:
            visible.update(current.defined)
            for (a, predicate, b), truth in current.relations.items():
                if predicate < 6:
                    q, allowed = self.comparison(a, predicate, b, truth)
                    self.base[q] = self.base.get(q, _FULL).intersect(allowed)
            current = current.parent
        self.remainders = sorted((v for v in visible if self.modulo(v) is not None), key=self.rank)

    def interval(self, value):
        if value not in self.intervals:
            self.intervals[value] = self.scope.interval(value)
        return self.intervals[value]

    def modulo(self, value):
        if value not in self.modulos:
            self.modulos[value] = _modulo(value)
        return self.modulos[value]

    def rank(self, value):
        if value not in self.ranks:
            self.ranks[value] = len(self.ranks)
        return self.ranks[value]

    def affine(self, value):
        return _affine(value, self.scope)

    def comparison(self, lhs, predicate, rhs, truth=True):
        left, a = self.affine(lhs)
        right, b = self.affine(rhs)
        terms = dict(left)
        for v, c in right.items():
            terms[v] = terms.get(v, 0) - c
        terms = {v: c for v, c in terms.items() if c}
        point = b - a
        allowed = (
            _Ranges(((point, point),)),
            _Ranges(((-inf, point - 1), (point + 1, inf))),
            _Ranges(((-inf, point - 1),)),
            _Ranges(((-inf, point),)),
            _Ranges(((point + 1, inf),)),
            _Ranges(((point, inf),)),
        )[predicate]
        if not truth:
            allowed = allowed.complement()
        # Stable order is the lexical discovery order of actual SSA atoms.
        for v in terms:
            self.quantities.setdefault(((v, 1),), len(self.quantities))
        items = sorted(terms.items(), key=lambda t: self.rank(t[0]))
        scale = reduce(gcd, (abs(c) for _, c in items), 0) or 1
        sign = -1 if items and items[0][1] < 0 else 1
        q = tuple((v, c // scale * sign) for v, c in items)
        intervals = tuple(
            (low if low == -inf else -(-low // scale), high if high == inf else high // scale)
            for low, high in allowed.intervals
        )
        intervals = tuple((a, b) for a, b in intervals if a <= b)
        if sign < 0:
            intervals = tuple((-b, -a) for a, b in reversed(intervals))
        self.quantities.setdefault(q, len(self.quantities))
        return q, _Ranges(intervals)

    def literal(self, value, truth):
        constant = _constant(value)
        if constant is not None:
            return {} if bool(constant) == truth else None
        op = value.owner
        if isinstance(op, arith.CmpiOp):
            predicate = op.predicate.value.data
            if predicate >= 6:
                if min(self.interval(op.lhs)[0], self.interval(op.rhs)[0]) < 0:
                    raise _Unchanged
                predicate -= 4
            q, allowed = self.comparison(op.lhs, predicate, op.rhs, truth)
        else:
            raise _Unchanged
        self.tests[q, allowed] = value, truth
        return {q: allowed}

    def _closure(self, cube):
        key = frozenset(cube.items())
        if key not in self.closures:
            self.closures[key] = self._close(cube)
        return self.closures[key]

    def _close(self, cube):
        constraints = dict(self.base)
        for q, allowed in cube.items():
            constraints[q] = constraints.get(q, _FULL).intersect(allowed)
        variables = dict.fromkeys(v for q in constraints for v, _ in q)
        values = [None, *variables]
        index = {v: i for i, v in enumerate(values)}
        matrix = [[0 if a == b else inf for b in values] for a in values]
        for v in variables:
            low, high = self.interval(v)
            matrix[index[v]][0], matrix[0][index[v]] = high, -low
        aliases = {}
        for v in variables:
            modulo = self.modulo(v)
            if modulo is None:
                continue
            period, numerator = modulo
            if period & (period - 1) and self.interval(numerator)[0] < 0:
                continue
            terms, bias = self.affine(numerator)
            signature = (
                period,
                frozenset((a, c % period) for a, c in terms.items() if c % period),
                bias % period,
            )
            if signature in aliases:
                a, b = index[v], index[aliases[signature]]
                matrix[a][b] = matrix[b][a] = 0
            else:
                aliases[signature] = v
        for q, allowed in constraints.items():
            if not allowed.intervals:
                return None, constraints
            low, high = allowed.intervals[0][0], allowed.intervals[-1][1]
            if not q:
                if not allowed.contains(_Ranges(((0, 0),))):
                    return None, constraints
                continue
            if len(q) == 1 and q[0][1] == 1:
                a, b = index[q[0][0]], 0
            elif len(q) == 2 and {c for _, c in q} == {-1, 1}:
                a = index[next(v for v, c in q if c == 1)]
                b = index[next(v for v, c in q if c == -1)]
            else:
                continue
            matrix[a][b], matrix[b][a] = min(matrix[a][b], high), min(matrix[b][a], -low)
        for k in range(len(values)):
            for i in range(len(values)):
                if matrix[i][k] == inf:
                    continue
                for j in range(len(values)):
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
        if any(matrix[i][i] < 0 for i in range(len(values))):
            return None, constraints
        # Exact difference equalities can turn a general affine equality
        # into an impossible integer residue, e.g. x-y=0 and x+y=1.
        for q, allowed in constraints.items():
            reduced, bias = {}, 0
            for v, coefficient in q:
                a = index[v]
                root = next(
                    b
                    for b in range(len(values))
                    if matrix[a][b] < inf and matrix[a][b] == -matrix[b][a]
                )
                bias += coefficient * matrix[a][root]
                if root:
                    reduced[root] = reduced.get(root, 0) + coefficient
            divisor = reduce(gcd, (abs(c) for c in reduced.values() if c), 0)
            if not divisor:
                possible = allowed.contains(_Ranges(((bias, bias),)))
            else:
                possible = any(
                    low == -inf
                    or high == inf
                    or -(-(low - bias) // divisor) <= (high - bias) // divisor
                    for low, high in allowed.intervals
                )
            if not possible:
                return None, constraints
        # Preserve the finite, correlated choices of two event-slot residues.
        # Taking only the hull of x-y != 0 loses the fact that two bits must
        # then satisfy x+y == 1. Enumerate bank residues, never loop iterations
        # or combinations of pipeline phases. Other variables are deliberately
        # projected out, so rejecting an empty pair projection is conservative.
        pairs = dict.fromkeys(
            tuple(v for v, _ in q)
            for q in constraints
            if len(q) == 2
            and all((m := self.modulo(v)) is not None and 1 < m[0] <= 16 for v, _ in q)
        )
        for pair in pairs:
            projected = tuple(
                (q, allowed) for q, allowed in constraints.items() if all(v in pair for v, _ in q)
            )
            a, b = (index[v] for v in pair)
            bounds = tuple(
                (max(0, int(-matrix[0][i])), min(self.modulo(v)[0] - 1, int(matrix[i][0])))
                for v, i in zip(pair, (a, b), strict=True)
            )
            differences = -matrix[b][a], matrix[a][b]
            key = pair, projected, bounds, differences
            if key not in self.residue_pairs:
                self.residue_pairs[key] = any(
                    differences[0] <= left - right <= differences[1]
                    and all(
                        allowed.contains(
                            _Ranges(((total := sum(c * values[v] for v, c in q), total),))
                        )
                        for q, allowed in projected
                    )
                    for left, right in product(*(range(low, high + 1) for low, high in bounds))
                    for values in ({pair[0]: left, pair[1]: right},)
                )
            if not self.residue_pairs[key]:
                return None, constraints
        return (index, matrix), constraints

    def bounds(self, q, cube):
        key = q, frozenset(cube.items())
        if key in self.bounds_cache:
            return self.bounds_cache[key]
        closure, constraints = self._closure(cube)
        if closure is None:
            result = _EMPTY
        else:
            index, matrix = closure
            if len(q) == 2 and {c for _, c in q} == {-1, 1} and all(v in index for v, _ in q):
                a = index[next(v for v, c in q if c == 1)]
                b = index[next(v for v, c in q if c == -1)]
                low, high = -matrix[b][a], matrix[a][b]
            else:
                low = high = 0
                for v, c in q:
                    vl, vh = (
                        (-matrix[0][index[v]], matrix[index[v]][0])
                        if v in index
                        else self.interval(v)
                    )
                    if self.modulo(v) is not None and (residues := self.residues(v, cube, closure)):
                        vl, vh = max(vl, min(residues)), min(vh, max(residues))
                    a, b = c * vl, c * vh
                    low, high = low + min(a, b), high + max(a, b)
            result = _Ranges(((low, high),)).intersect(constraints.get(q, _FULL))
        self.bounds_cache[key] = result
        return result

    def residues(self, value, cube, closure):
        period, numerator = self.modulo(value)
        if period is None or not 1 < period <= 16:
            return None
        key = value, frozenset(cube.items())
        if key in self.resolving:
            return set(range(period))
        self.resolving.add(key)
        try:
            terms, bias = self.affine(numerator)
            terms = dict(terms)
            # Powers of two also respect unsigned fixed-width bit patterns.
            # Other moduli need the original nonnegative numerator domain.
            if period & (period - 1) and self.interval(numerator)[0] < 0:
                return set(range(period))
            replaced = {}
            for atom, coefficient in terms.items():
                replacement = atom
                for remainder in self.remainders:
                    inner_period, inner_numerator = self.modulo(remainder)
                    if remainder is value or inner_period != period:
                        continue
                    if period & (period - 1) and self.interval(inner_numerator)[0] < 0:
                        continue
                    basis, offset = self.affine(inner_numerator)
                    if basis == {atom: 1} and offset == 0:
                        replacement = remainder
                        break
                replaced[replacement] = replaced.get(replacement, 0) + coefficient
            terms = replaced
            index, matrix = closure
            reduced = {}
            for atom, coefficient in terms.items():
                root, offset = atom, 0
                if atom in index:
                    a = index[atom]
                    equivalents = [
                        v
                        for v, b in index.items()
                        if v is not value and matrix[a][b] < inf and matrix[a][b] == -matrix[b][a]
                    ]
                    if equivalents:
                        root = min(equivalents, key=lambda v: -1 if v is None else self.rank(v))
                        offset = matrix[a][index[root]]
                bias += coefficient * offset
                if root is not None:
                    reduced[root] = (reduced.get(root, 0) + coefficient) % period
            reduced = {v: c for v, c in reduced.items() if c}
            if not reduced:
                return {int(bias) % period}
            if value in reduced:
                return set(range(period))
            q = tuple(sorted(reduced.items(), key=lambda t: self.rank(t[0])))
            bounds = self.bounds(q, cube)
            result = set()
            for low, high in bounds.intervals:
                if high - low >= period or low == -inf or high == inf:
                    return set(range(period))
                result.update(n % period for n in range(int(low + bias), int(high + bias) + 1))
            return result
        finally:
            self.resolving.remove(key)

    def normalize(self, cube):
        key = frozenset(cube.items())
        if key in self.normalized:
            result = self.normalized[key]
            return None if result is None else dict(result)
        if self._closure(cube)[0] is None:
            self.normalized[key] = None
            return None
        result = dict(cube)
        for q, allowed in tuple(result.items()):
            other = result.copy()
            del other[q]
            valid = self.bounds(q, other)
            narrowed = allowed.intersect(valid)
            if not narrowed.intervals:
                self.normalized[key] = None
                return None
            if allowed.contains(valid):
                del result[q]
            else:
                result[q] = narrowed
        self.normalized[key] = tuple(result.items())
        return result

    def intersect(self, a, b):
        result = dict(a)
        for q, allowed in b.items():
            result[q] = result.get(q, _FULL).intersect(allowed)
        return self.normalize(result)

    def covers(self, a, b):
        return all(allowed.contains(self.bounds(q, b)) for q, allowed in a.items())

    def covered(self, candidate, alternatives):
        """Subtract ordered cubes; empty remainder is exact local coverage.

        Splits use only existing predicate endpoints, never loop iterations.
        If a fragmented union is too large for this local optimization, retain
        its original predicates. No correctness check or protocol is bypassed.
        """
        pending = [candidate]
        for alternative in alternatives:
            remaining = []
            for cube in pending:
                if self.covers(alternative, cube):
                    continue
                if self.intersect(cube, alternative) is None:
                    remaining.append(cube)
                    continue
                inside = cube
                for q, allowed in alternative.items():
                    outside = self.intersect(inside, {q: allowed.complement()})
                    if outside is not None:
                        remaining.append(outside)
                    inside = self.intersect(inside, {q: allowed})
                    if inside is None:
                        break
                if len(remaining) > 128:
                    return False
            pending = remaining
            if not pending:
                return True
        return not pending

    def union(self, cubes):
        key = tuple(frozenset(cube.items()) for cube in cubes)
        if key in self.unions:
            return [dict(cube) for cube in self.unions[key]]
        result = []
        for cube in cubes:
            cube = self.normalize(cube)
            if cube is not None and not any(self.covers(prior, cube) for prior in result):
                result = [prior for prior in result if not self.covers(cube, prior)] + [cube]
        # Each replacement strictly broadens an existing cube, or removes one.
        # The finite input endpoints bound this reduction; no case tree grows.
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(result):
                for j, b in enumerate(result[:i]):
                    for q in dict.fromkeys((*a, *b)):
                        candidate = {}
                        for variable in dict.fromkeys((*a, *b)):
                            left, right = a.get(variable, _FULL), b.get(variable, _FULL)
                            candidate[variable] = (
                                left.union(right) if variable == q else left.intersect(right)
                            )
                        candidate = self.normalize(candidate)
                        if candidate is None:
                            continue
                        indices = [k for k, c in enumerate(result) if self.covers(candidate, c)]
                        if not indices or any(self.covers(c, candidate) for c in result):
                            continue
                        result = [c for k, c in enumerate(result) if k not in indices] + [candidate]
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break
        changed = True
        while changed:
            changed = False
            for i, cube in enumerate(result):
                for q in cube:
                    candidate = dict(cube)
                    del candidate[q]
                    if self.covered(candidate, result):
                        result[i] = candidate
                        result = [
                            c
                            for j, c in enumerate(result)
                            if j == i or not self.covers(candidate, c)
                        ]
                        changed = True
                        break
                if changed:
                    break
        # Remove a redundant component of a disjoint range (A || B)&C when
        # another cube already covers B. This exposes A&C without repeating B.
        for i, cube in enumerate(result):
            others = result[:i] + result[i + 1 :]
            for q, allowed in tuple(cube.items()):
                if len(allowed.intervals) > 1:
                    kept = tuple(
                        interval
                        for interval in allowed.intervals
                        if not self.covered(cube | {q: _Ranges((interval,))}, others)
                    )
                    if kept:
                        cube[q] = _Ranges(kept)
        self.unions[key] = tuple(tuple(cube.items()) for cube in result)
        return result


class _Build:
    def __init__(self, domains):
        self.domains = domains
        self.block = Block()
        self.available = {}
        self.values = {}
        self.scope = domains.scope.child()

    def operation(self, op):
        normalized = _linear_operation(op, self.scope)
        if normalized is not op:
            op.erase()
            op = normalized
        folded = _fold(op, self.scope)
        if isinstance(folded, int):
            typ = op.results[0].type
            op.erase()
            return self.constant(folded, typ)
        if folded is not None:
            op.erase()
            return folded
        key = _key(op)
        if key in self.available:
            op.erase()
            return self.available[key]
        self.block.add_op(op)
        self.available[key] = op.results[0]
        self.scope.defined.update(op.results)
        if not isinstance(op, arith.ConstantOp):
            self.scope.ranges[op.results[0]] = _interval(op, self.scope)
        return op.results[0]

    def constant(self, value, typ=None):
        typ = IndexType() if typ is None else typ
        return self.operation(arith.ConstantOp.from_int_and_width(value, typ))

    def quantity(self, q):
        if q in self.values:
            return self.values[q]
        if len(q) == 1 and q[0][1] == 1:
            return q[0][0]
        result, low, high = None, 0, 0
        for value, coefficient in q:
            minimum, maximum = _limits(value)
            vl, vh = self.domains.scope.interval(value)
            a, b = sorted((coefficient * vl, coefficient * vh))
            if not minimum <= a <= b <= maximum or not minimum <= low + a <= high + b <= maximum:
                raise _Unchanged
            operand = (
                value
                if coefficient == 1
                else self.operation(arith.MuliOp(value, self.constant(coefficient, value.type)))
            )
            result = operand if result is None else self.operation(arith.AddiOp(result, operand))
            low, high = low + a, high + b
        result = self.constant(0) if result is None else result
        self.values[q] = result
        return result

    def test(self, q, allowed):
        domain = self.domains.bounds(q, {})
        if allowed.contains(domain):
            return self.constant(1, i1)
        if (known := self.domains.tests.get((q, allowed))) is not None:
            value, truth = known
            if truth:
                return value
            return self.operation(arith.CmpiOp(value, self.constant(0, i1), "eq"))
        value, alternatives = self.quantity(q), []
        for low, high in allowed.intervals:
            tests = []
            if low == high:
                tests.append(
                    self.operation(arith.CmpiOp(value, self.constant(int(low), value.type), "eq"))
                )
            else:
                if low != -inf and low > domain.intervals[0][0]:
                    tests.append(
                        self.operation(
                            arith.CmpiOp(value, self.constant(int(low), value.type), "sge")
                        )
                    )
                if high != inf and high < domain.intervals[-1][1]:
                    tests.append(
                        self.operation(
                            arith.CmpiOp(value, self.constant(int(high), value.type), "sle")
                        )
                    )
            alternatives.append(self.logical(tests, arith.AndIOp, True))
        return self.logical(alternatives, arith.OrIOp, False)

    def logical(self, values, operation, empty):
        filtered = []
        for value in values:
            constant = _constant(value)
            if constant is not None:
                if bool(constant) != empty:
                    return self.constant(int(not empty), i1)
            else:
                filtered.append(value)
        values = filtered
        if not values:
            return self.constant(int(empty), i1)
        return reduce(lambda a, b: self.operation(operation(a, b)), dict.fromkeys(values))

    def guard(self, cubes):
        return self.logical(
            [
                self.logical([self.test(q, r) for q, r in cube.items()], arith.AndIOp, True)
                for cube in cubes
            ],
            arith.OrIOp,
            False,
        )

    def path(self, cubes):
        common = {q: r for q, r in cubes[0].items() if all(c.get(q) == r for c in cubes[1:])}
        residual = [{q: r for q, r in c.items() if q not in common} for c in cubes]
        result = [self.test(q, r) for q, r in common.items()]
        result = [value for value in result if _constant(value) in (None, 0)]
        if all(residual):
            result.append(self.guard(residual))
        return result

    def discard(self):
        for op in reversed(tuple(self.block.ops)):
            self.block.erase_op(op, safe_erase=False)

    def event(self, cases):
        events = [op.event for _, op in cases]
        if (
            all(v is events[0] for v in events)
            or all(_constant(v) == _constant(events[0]) for v in events if v is not None)
            and _constant(events[0]) is not None
        ):
            return events[0]
        # A finite selector of constant IDs is one indexed event bank. Prefer
        # an already evaluated physical slot to an unrelated bounded scalar.
        candidates = list(dict.fromkeys(v for q in self.domains.quantities for v, _ in q))
        candidates.sort(key=lambda v: _modulo(v) is None)
        for value in candidates:
            mapping = {}
            for cube, instruction in cases:
                event = _constant(instruction.event)
                if event is None:
                    break
                ranges = self.domains.bounds(((value, 1),), cube)
                if not ranges.intervals or any(
                    not -(1 << 63) <= a <= b < (1 << 63) or b - a > 15 for a, b in ranges.intervals
                ):
                    break
                residues = [r for a, b in ranges.intervals for r in range(int(a), int(b) + 1)]
                if len(residues) > 16:
                    break
                if any(r in mapping and mapping[r] != event for r in residues):
                    break
                mapping.update((r, event) for r in residues)
            else:
                points = sorted(mapping)
                if len(points) >= 2:
                    first, second = points[:2]
                    delta = second - first
                    if (mapping[second] - mapping[first]) % delta:
                        continue
                    stride = (mapping[second] - mapping[first]) // delta
                    base = mapping[first] - stride * first
                    if all(event == base + stride * r for r, event in mapping.items()):
                        vl, vh = self.domains.scope.interval(value)
                        low, high = sorted((vl * stride, vh * stride))
                        minimum, maximum = _limits(value)
                        if not minimum <= low <= high <= maximum or not (
                            minimum <= low + base <= high + base <= maximum
                        ):
                            continue
                        result = (
                            value
                            if stride == 1
                            else self.operation(
                                arith.MuliOp(value, self.constant(stride, value.type))
                            )
                        )
                        return (
                            result
                            if base == 0
                            else self.operation(
                                arith.AddiOp(result, self.constant(base, value.type))
                            )
                        )
        by_event = {}
        for cube, instruction in cases:
            event = instruction.event
            key = event if _constant(event) is None else (event.type, _constant(event))
            by_event.setdefault(key, (event, []))[1].append(cube)
        alternatives = [(event, self.domains.union(cubes)) for event, cubes in by_event.values()]
        result = alternatives[-1][0]
        for event, cubes in reversed(alternatives[:-1]):
            result = self.operation(arith.SelectOp(self.guard(cubes), event, result))
        return result


def _instruction(op):
    return op.kind, op.source, op.destination, op.participants


def _packet(operations, scope, work, ranks, changed):
    movable, cases = [], []
    local = scope.child()
    if scope.facts is not None:
        # Proposal-only domains must not borrow a result's old guarded bound
        # after moving its evaluation into the common packet entry.
        local.facts = ScalarFacts(scope.facts.limits, dict(scope.facts.ranges))

    def collect(op, guard):
        if _scalar_arithmetic(op):
            if not all(local.has_definition(v) or _constant(v) is not None for v in op.operands):
                raise _Unchanged
            if isinstance(op, _UNSIGNED) and (
                local.interval(op.lhs)[0] < 0 or local.interval(op.rhs)[0] <= 0
            ):
                raise _Unchanged
            interval = _bounded_interval(op, local)
            if not isinstance(op, (arith.CmpiOp, arith.ConstantOp)) and interval is None:
                raise _Unchanged
            movable.append(op)
            local.defined.update(op.results)
            if not isinstance(op, arith.ConstantOp):
                local.ranges[op.results[0]] = interval or _limits(op.results[0])
                if local.facts is not None:
                    local.facts.ranges[op.results[0]] = local.ranges[op.results[0]]
        elif isinstance(op, scf.IfOp) and not op.results:
            constant = _constant(op.cond)
            for truth, region in ((True, op.true_region), (False, op.false_region)):
                if constant is not None and bool(constant) != truth:
                    continue
                for block in region.blocks:
                    for child in block.ops:
                        collect(
                            child, guard + (() if constant is not None else ((op.cond, truth),))
                        )
        elif isinstance(op, NotificationOp):
            if (
                op.event is not None
                and not local.has_definition(op.event)
                and _constant(op.event) is None
            ):
                raise _Unchanged
            cases.append((guard, op))
        elif not isinstance(op, scf.YieldOp):
            raise _Unchanged

    for op in operations:
        collect(op, ())
    domains = _Domains(local, ranks)
    normalized = []
    for guard, op in cases:
        cube = {}
        for value, truth in guard:
            literal = domains.literal(value, truth)
            if literal is None:
                cube = None
                break
            cube = domains.intersect(cube, literal)
            if cube is None:
                break
        if cube is not None:
            normalized.append((cube, op))
    build, groups = _Build(domains), []
    for cube, op in normalized:
        if (
            groups
            and _instruction(groups[-1][0][1]) == _instruction(op)
            and (
                # collect admits only pure integer SSA, branches and notifications.
                # Adjacent equal fences therefore have the same actor/pipe and
                # no new work or other synchronization between their captures.
                # F;F = F on every active path, so their existence is the union
                # of guards. This does not apply to Set/Wait contributions.
                op.kind.data == SyncKind.FENCE.value
                or all(domains.intersect(cube, prior) is None for prior, _ in groups[-1])
            )
        ):
            groups[-1].append((cube, op))
        else:
            groups.append([(cube, op)])
    pending = []
    frequency = {}
    try:
        for group in groups:
            cubes = domains.union([cube for cube, _ in group])
            path = build.path(cubes)
            common = {q: r for q, r in cubes[0].items() if all(c.get(q) == r for c in cubes[1:])}
            protocols = tuple(dict.fromkeys(p.protocol.data for _, p in group))
            pending.append((path, group, common, protocols))
            for value in path:
                frequency[value] = frequency.get(value, 0) + 1
        for path, _, _, _ in pending:
            path.sort(key=lambda value: -frequency[value])
        builder = build.block
        prefixes, blocks, contexts = [], [builder], [build.scope]
        replacements = []
        for path, group, assumptions, protocols in pending:
            common = 0
            while common < min(len(prefixes), len(path)) and prefixes[common] is path[common]:
                common += 1
            while len(prefixes) > common:
                blocks.pop().add_op(scf.YieldOp())
                contexts.pop()
                prefixes.pop()
            for condition in path[common:]:
                child = Block()
                blocks[-1].add_op(scf.IfOp(condition, [], Region(child), Region()))
                context = contexts[-1].child()
                context.assume(condition, True)
                contexts.append(context)
                prefixes.append(condition)
                blocks.append(child)
            original = group[0][1]
            event = original.event
            event_builder = None
            if event is not None and len(group) > 1:
                nested = _Domains(contexts[-1], ranks)
                nested.tests.update(domains.tests)
                nested.quantities.update(domains.quantities)
                for q, allowed in assumptions.items():
                    nested.base[q] = nested.base.get(q, _FULL).intersect(allowed)
                relative = [
                    (cube, instruction)
                    for prior, instruction in group
                    if (cube := nested.normalize(prior)) is not None
                ]
                event_builder = _Build(nested)
                try:
                    event = event_builder.event(relative)
                except _Unchanged:
                    event_builder.discard()
                    raise
                for scalar in tuple(event_builder.block.ops):
                    event_builder.block.detach_op(scalar)
                    blocks[-1].add_op(scalar)
                    contexts[-1].defined.update(scalar.results)
            placeholder = original.clone()
            if event is not None:
                placeholder.operands = (event,)
            placeholder.attributes["protocol"] = StringAttr(", ".join(protocols))
            blocks[-1].add_op(placeholder)
            replacements.append((original, placeholder))
        while len(blocks) > 1:
            blocks.pop().add_op(scf.YieldOp())
    except _Unchanged:
        build.discard()
        raise
    # New selectors are total in their actual guarded domains. Commit only
    # after all rewrites succeed, preserving native/notification identities.
    for original, placeholder in replacements:
        original.parent.detach_op(original)
        original.operands = tuple(placeholder.operands)
        original.attributes["protocol"] = placeholder.protocol
        block = placeholder.parent
        block.insert_op_before(original, placeholder)
        block.erase_op(placeholder)
    for op in movable:
        op.parent.detach_op(op)
        if scope.facts is not None and op.results[0] in local.facts.ranges:
            scope.facts.ranges[op.results[0]] = local.facts.ranges[op.results[0]]
    prefix = list(movable)
    for scalar in tuple(builder.ops):
        if _scalar_arithmetic(scalar):
            _place_invariant(scalar, build.scope, _key(scalar))
    result = prefix + list(builder.ops)
    changed.update(op for root in result for op in root.walk() if _scalar_arithmetic(op))
    for op in tuple(builder.ops):
        builder.detach_op(op)
    work.packets += 1
    work.input_notifications += len(cases)
    work.output_notifications += len(groups)
    work.removed_fences += sum(
        len(group) - 1 for group in groups if group[0][1].kind.data == SyncKind.FENCE.value
    )
    return result


def simplify_conditions(module, *, index_extrema=(), facts=None):
    """Normalize domains and share consecutive guarded packets, in place."""
    facts = ScalarFacts.from_control(index_extrema=index_extrema) if facts is None else facts
    work, limits = ConditionWork(), facts.limits
    # Notification/control-only unsigned remainders by powers of two have an
    # exact, total bit-mask form. This permits legal common-domain evaluation
    # even when the original guarded numerator could be negative outside its
    # prefix. Original DSL definedness was admitted before this pass. Address,
    # loop-header and storage expressions retain their admitted representation.
    protected = set()
    affected = set()

    def protect(value):
        if value in protected:
            return
        protected.add(value)
        if _scalar_arithmetic(value.owner):
            for operand in value.owner.operands:
                protect(operand)

    for op in module.walk():
        if isinstance(op, (AddressOp, PipelineLoopOp, scf.ForOp, BufferOp, scf.YieldOp)):
            for value in op.operands:
                protect(value)
    for op in tuple(module.walk()):
        if isinstance(op, arith.RemUIOp) and op.result not in protected:
            period = _constant(op.rhs)
            if period is not None and period > 0 and period & (period - 1) == 0:
                mask = arith.ConstantOp.from_int_and_width(period - 1, op.result.type)
                masked = arith.AndIOp(op.lhs, mask)
                affected.update((mask, masked))
                pending = [use.operation for use in op.result.uses]
                while pending:
                    user = pending.pop()
                    if user in affected or not _scalar_arithmetic(user):
                        continue
                    affected.add(user)
                    pending.extend(use.operation for result in user.results for use in result.uses)
                op.parent.insert_ops_before((mask, masked), op)
                op.result.replace_all_uses_with(masked.result)
                op.parent.erase_op(op)
    ranks = {}
    for op in module.walk():
        for region in op.regions:
            for block in region.blocks:
                for value in block.args:
                    ranks[value] = len(ranks)
        for value in op.results:
            ranks[value] = len(ranks)

    packet_shapes = {}

    def packet_shape(op):
        if op not in packet_shapes:
            packet_shapes[op] = (
                _scalar_arithmetic(op)
                or isinstance(op, (NotificationOp, scf.YieldOp))
                or isinstance(op, scf.IfOp)
                and not op.results
                and all(packet_shape(child) for region in op.regions for child in region.walk())
            )
        return packet_shapes[op]

    def visit(block, scope):
        _seed(block, scope)
        operations, i = tuple(block.ops), 0
        while i < len(operations):
            op = operations[i]
            if isinstance(op, (scf.IfOp, NotificationOp)) and not op.results and packet_shape(op):
                previous = op.prev_op
                end = i
                while (
                    end < len(operations)
                    and not isinstance(operations[end], scf.YieldOp)
                    and packet_shape(operations[end])
                ):
                    end += 1
                # A control/native barrier inside an If ends this packet. Try
                # each consecutive root until the maximal legal prefix is known.
                replacement, stop = None, i
                for candidate in range(end, i, -1):
                    try:
                        replacement = _packet(operations[i:candidate], scope, work, ranks, affected)
                    except _Unchanged:
                        continue
                    stop = candidate
                    break
                if replacement is not None:
                    anchor = block.first_op if previous is None else previous.next_op
                    if anchor is None:
                        block.add_ops(replacement)
                    else:
                        block.insert_ops_before(replacement, anchor)
                    for old in operations[i:stop]:
                        if old.parent is block and old not in replacement:
                            block.erase_op(old)
                    for new in replacement:
                        if _scalar_arithmetic(new):
                            scope.defined.update(new.results)
                            if not isinstance(new, arith.ConstantOp):
                                scope.ranges[new.results[0]] = _interval(new, scope)
                    i = stop
                    continue
            if isinstance(op, RuntimeSizeOp):
                multiple = op.multiple.data
                scope.ranges[op.value] = (
                    (op.minimum.data + multiple - 1) // multiple * multiple,
                    limits.get(op.sym_name.data, _limits(op.value)[1]),
                )
            elif isinstance(op, CoreIdOp):
                root = module.body.block.first_op
                scope.ranges[op.value] = (0, (1 if root.blocks is None else root.blocks.data) - 1)
            elif _scalar_arithmetic(op):
                if op in affected:
                    folded = _fold(op, scope)
                    if isinstance(folded, int):
                        constant = arith.ConstantOp.from_int_and_width(folded, op.results[0].type)
                        op.parent.insert_op_before(constant, op)
                        folded = constant.result
                    if folded is not None:
                        _replace(op, folded)
                        i += 1
                        continue
                    if _place_invariant(op, scope, _key(op)):
                        i += 1
                        continue
                if not isinstance(op, arith.ConstantOp):
                    scope.ranges[op.results[0]] = _interval(op, scope)
                scope.available[_key(op)] = op.results[0]
            for branch, region in enumerate(op.regions):
                for child in region.blocks:
                    nested = scope.child()
                    if (
                        isinstance(op, (scf.ForOp, scf.IfOp, PipelineLoopOp))
                        or op.name == "npu.actor_domain"
                    ):
                        nested.placements += ((op, scope),)
                    if isinstance(op, scf.IfOp):
                        nested.assume(op.cond, branch == 0)
                    visit(child, nested)
            scope.defined.update(op.results)
            i += 1
        # General consecutive blocks, including native and memory operations,
        # share identical SSA conditions without moving any instruction.
        previous, pending = None, []
        for op in tuple(block.ops):
            if isinstance(op, scf.IfOp) and not op.results and _empty_region(op.false_region):
                if previous is not None and previous.cond is op.cond:
                    for scalar in pending:
                        block.detach_op(scalar)
                        block.insert_op_before(scalar, previous)
                    pending.clear()
                    destination = previous.true_region.block
                    terminator = destination.last_op
                    for child in tuple(op.true_region.block.ops):
                        if not isinstance(child, scf.YieldOp):
                            op.true_region.block.detach_op(child)
                            destination.insert_op_before(child, terminator)
                    block.erase_op(op)
                    work.shared_blocks += 1
                    continue
                previous = op
                pending.clear()
            elif _scalar_arithmetic(op) and (
                isinstance(
                    op, (arith.ConstantOp, arith.CmpiOp, arith.AndIOp, arith.OrIOp, arith.SelectOp)
                )
                or _bounded_interval(op, scope) is not None
            ):
                pending.append(op)
            else:
                previous, pending = None, []

    for block in module.body.blocks:
        visit(block, _Scope(facts=facts))
    cleanup_scalar_syntax(module, facts=facts, changed=affected)
    return work
