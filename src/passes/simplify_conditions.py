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
from math import gcd, inf

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, StringAttr, i1
from xdsl.ir import Block, Region

from ..analysis.smt import all_of, any_of, query_service
from ..analysis.structured_schedule import Comparison
from ..dialects.physical import BufferOp, CoreIdOp, RuntimeSizeOp
from ..dialects.structured import AddressOp, NotificationOp, PipelineLoopOp
from ..scalar import Expr, literal
from ..symbols import Symbol, SymbolKind
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
    """Compact guard syntax with one shared mathematical query service.

    Quantities expand only arithmetic already known not to wrap in the packet
    entry scope. Atoms retain actual SSA identity and their admitted entry
    bounds. Modulo links preserve correlations without enumerating phase or
    residue products. This view never authorizes a new runtime evaluation.
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
        self.base = {}
        self.queries = query_service({})
        self.atoms, self.constraints, self.predicates, self.cubes = {}, {}, {}, {}
        self.fixed_constraints = None
        current = scope
        while current is not None:
            for (a, predicate, b), truth in current.relations.items():
                if predicate < 6:
                    q, allowed = self.comparison(a, predicate, b, truth)
                    self.base[q] = self.base.get(q, _FULL).intersect(allowed)
            current = current.parent

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

    def _atom(self, value):
        if value in self.atoms:
            return self.atoms[value]
        symbol = Symbol(f"ssa{self.rank(value)}", SymbolKind.SSA, value)
        result = Expr("argument", (symbol,))
        self.atoms[value] = result
        low, high = self.interval(value)
        constraints = [
            Comparison(result, ">=", literal(low)),
            Comparison(result, "<=", literal(high)),
        ]
        self.constraints[value] = constraints
        self.fixed_constraints = None
        modulo = self.modulo(value)
        if modulo is not None:
            period, numerator = modulo
            # Low-bit masks and unsigned power-of-two remainders equal
            # mathematical modulo for every bit pattern. Other divisors need
            # the nonnegative dividend premise from this actual entry scope.
            if period & (period - 1) == 0 or self.interval(numerator)[0] >= 0:
                terms, bias = self.affine(numerator)
                dividend = self._quantity(tuple(terms.items()), bias)
                constraints.append(
                    Comparison(result, "==", Expr("rem", (dividend, literal(period))))
                )
        return result

    def _quantity(self, q, bias=0):
        result = literal(bias)
        for value, coefficient in q:
            term = self._atom(value)
            if coefficient != 1:
                term = Expr("mul", (literal(coefficient), term))
            result = term if result.constant == 0 else Expr("add", (result, term))
        return result

    def _predicate(self, q, allowed):
        key = q, allowed
        if key not in self.predicates:
            value = self._quantity(q)
            self.predicates[key] = any_of(
                *(
                    Comparison(value, "==", literal(low))
                    if low == high
                    else all_of(
                        True if low == -inf else Comparison(value, ">=", literal(low)),
                        True if high == inf else Comparison(value, "<=", literal(high)),
                    )
                    for low, high in allowed.intervals
                )
            )
        return self.predicates[key]

    def _formula(self, cube):
        key = frozenset(cube.items())
        if key not in self.cubes:
            self.cubes[key] = all_of(*(self._predicate(q, r) for q, r in cube.items()))
        return self.cubes[key]

    def _premises(self, cube):
        # Encode first: a newly encountered atom may contribute its own bounds
        # and a modulo link. Every query key then contains the complete snapshot.
        base, current = self._formula(self.base), self._formula(cube)
        if self.fixed_constraints is None:
            self.fixed_constraints = all_of(
                *(test for constraints in self.constraints.values() for test in constraints)
            )
        return self.fixed_constraints, base, current

    def _bounds(self, q, cube):
        low = high = 0
        for value, coefficient in q:
            a, b = (coefficient * bound for bound in self.interval(value))
            low, high = low + min(a, b), high + max(a, b)
        return (
            _Ranges(((low, high),)).intersect(self.base.get(q, _FULL)).intersect(cube.get(q, _FULL))
        )

    def bounds(self, q, cube):
        key = q, frozenset(cube.items())
        if key not in self.bounds_cache:
            result = self._bounds(q, cube)
            # Only event-selector output needs individual finite ID values.
            # This projects at most 16 values, not pairs of residues or loops.
            if sum(high - low + 1 for low, high in result.intervals) <= 16:
                value = self._quantity(q)
                premises = self._premises(cube)
                result = _EMPTY
                for low, high in self._bounds(q, cube).intervals:
                    for point in range(low, high + 1):
                        possible = self.queries.satisfies(
                            (Comparison(value, "==", literal(point)),), premises
                        )
                        if not possible.impossible:
                            result = result.union(_Ranges(((point, point),)))
            self.bounds_cache[key] = result
        return self.bounds_cache[key]

    def normalize(self, cube):
        if not cube:
            return {}
        key = frozenset(cube.items())
        if key in self.normalized:
            result = self.normalized[key]
            return None if result is None else dict(result)
        formula = self._formula(cube)
        if self.queries.satisfies((formula,), self._premises({})).impossible:
            self.normalized[key] = None
            return None
        result = dict(cube)
        for q, allowed in tuple(result.items()):
            other = result.copy()
            del other[q]
            valid = self._bounds(q, other)
            narrowed = allowed.intersect(valid)
            if not narrowed.intervals:
                self.normalized[key] = None
                return None
            if (
                allowed.contains(valid)
                or self.queries.implies(self._predicate(q, allowed), self._premises(other)).proved
            ):
                del result[q]
            else:
                result[q] = narrowed
        self.normalized[key] = tuple(result.items())
        return result

    def intersect(self, a, b):
        result = dict(a)
        for q, allowed in b.items():
            result[q] = result.get(q, _FULL).intersect(allowed)
        return result

    def disjoint(self, a, b):
        # Mutual exclusion needs one coexistence question, not a normalized
        # intersection with every surviving predicate proved again.
        cube = self.intersect(a, b)
        return (
            any(not allowed.intervals for allowed in cube.values())
            or self.queries.satisfies((self._formula(cube),), self._premises({})).impossible
        )

    @staticmethod
    def _contains(a, b):
        """Syntactic absorption only; absence is no mathematical refutation."""
        return all(allowed.contains(b.get(q, _FULL)) for q, allowed in a.items())

    def covered(self, candidate, alternatives):
        """One exact union query; unknown retains the original guards."""
        formula = any_of(*(self._formula(cube) for cube in alternatives))
        return self.queries.implies(formula, self._premises(candidate)).proved

    def union(self, cubes):
        key = tuple(frozenset(cube.items()) for cube in cubes)
        if key in self.unions:
            return [dict(cube) for cube in self.unions[key]]
        result = []
        for cube in cubes:
            cube = self.normalize(cube)
            if cube is not None and not any(self._contains(prior, cube) for prior in result):
                result = [prior for prior in result if not self._contains(cube, prior)] + [cube]
        # Aligned rectangles merge by syntax. Do not ask the solver about
        # every pair and every guessed dimension: the complete union below
        # decides correlated coverage once for each actual simplification.
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(result):
                for j, b in enumerate(result[:i]):
                    differences = [q for q in a.keys() | b.keys() if a.get(q) != b.get(q)]
                    if len(differences) != 1:
                        continue
                    q = differences[0]
                    candidate = dict(a)
                    candidate[q] = a.get(q, _FULL).union(b.get(q, _FULL))
                    candidate = self.normalize(candidate)
                    result = [c for k, c in enumerate(result) if k not in (i, j)]
                    if candidate is not None:
                        result.append(candidate)
                    changed = True
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
                            if j == i or not self._contains(candidate, c)
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
            self.scope.bind_range(op.results[0], _interval(op, self.scope))
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

    def static_binary_event(self, cases):
        """Encode two constant mode-2 IDs as one branch, preserving the word.

        Variable wait_flag_dev/config operands are costly on 2201. The packet
        already established mutual exclusion and a covering outer guard; its
        first alternative therefore selects one of exactly these two IDs.
        This changes only the terminal encoding, never allocation or sharing.
        Larger/variable selectors retain the ordinary scalar representation.
        """
        alternatives = {}
        for cube, op in cases:
            event = _constant(op.event)
            if event is None:
                return None
            alternatives.setdefault(event, []).append((cube, op))
        if len(alternatives) != 2:
            return None
        left, right = alternatives.values()
        condition = self.guard(self.domains.union([cube for cube, _ in left]))
        return condition, (left, right)

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
    # Instruction sharing depends only on route and mutually exclusive
    # execution domains, never numeric IDs or virtual credit identity.
    # Allocation may simplify the resulting scalar selector, but cannot
    # decide which notifications are grouped or change their contributions.
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
                local.bind_range(op.results[0], interval or _limits(op.results[0]))
                if local.facts is not None:
                    local.facts.bind_range(op.results[0], local.ranges[op.results[0]])
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
        # collect already admitted every moved scalar at this packet entry.
        # Build the complete mathematical guard first, then normalize once;
        # partial conjunctions are not separate optimization obligations.
        if cube is not None and (cube := domains.normalize(cube)) is not None:
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
                or all(domains.disjoint(cube, prior) for prior, _ in groups[-1])
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
            static_event = None
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
                    if original.kind.data in {
                        SyncKind.CROSS_SET.value, SyncKind.CROSS_WAIT.value
                    }:
                        static_event = event_builder.static_binary_event(relative)
                    if static_event is None:
                        event = event_builder.event(relative)
                except _Unchanged:
                    event_builder.discard()
                    raise
                for scalar in tuple(event_builder.block.ops):
                    event_builder.block.detach_op(scalar)
                    blocks[-1].add_op(scalar)
                    contexts[-1].defined.update(scalar.results)
            if static_event is not None:
                condition, alternatives = static_event
                branches = (Block(), Block())
                blocks[-1].add_op(scf.IfOp(condition, [], *(Region(b) for b in branches)))
                for branch, members in zip(branches, alternatives, strict=True):
                    selected = members[0][1]
                    placeholder = selected.clone()
                    placeholder.attributes["protocol"] = StringAttr(
                        ", ".join(dict.fromkeys(op.protocol.data for _, op in members))
                    )
                    branch.add_ops((placeholder, scf.YieldOp()))
                    replacements.append((selected, placeholder))
                continue
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
            scope.facts.bind_range(op.results[0], local.facts.ranges[op.results[0]])
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
    work.output_notifications += len(replacements)
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
                                scope.bind_range(new.results[0], _interval(new, scope))
                    i = stop
                    continue
            if isinstance(op, RuntimeSizeOp):
                multiple = op.multiple.data
                scope.bind_range(
                    op.value,
                    (
                        (op.minimum.data + multiple - 1) // multiple * multiple,
                        limits.get(op.sym_name.data, _limits(op.value)[1]),
                    ),
                )
            elif isinstance(op, CoreIdOp):
                root = module.body.block.first_op
                scope.bind_range(
                    op.value, (0, (1 if root.blocks is None else root.blocks.data) - 1)
                )
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
                    scope.bind_range(op.results[0], _interval(op, scope))
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
