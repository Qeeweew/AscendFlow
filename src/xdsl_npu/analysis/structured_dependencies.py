"""Recursive, symbolic memory obligations on the retained physical tree.

The result describes required native-end -> admission ordering. It supplies no
completion, event allocation, initialization or ownership proof. Logical source
order is authoritative; issue_order remains a later protocol-placement input.
Runtime loop counts are never enumerated. Finite address images are partitioned
by actual storage slots/fragments, and loop boundaries use guarded selectors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING

from xdsl.dialects import arith, scf
from xdsl.ir import Operation

from ..diagnostics import CompileError, ErrorKind, require
from ..indexing import equivalent, normalize
from ..scalar import Expr, expression, infer, literal, substitute
from .access_frontier import AccessEndpoint, LoopResidueSelection, ResiduePosition
from .hazards import HazardKind
from .structured_access import PartitionedAccesses, partition_accesses
from .structured_memory import NativeMemoryFacts
from .structured_regions import (
    AccessOccurrence,
    BoundaryKind,
    BoundaryPort,
    BranchGuard,
    CellSummary,
    CountRelation,
    DependencyIssue,
    DependencyIssueKind,
    DependencyWork,
    ExitObligation,
    GuardedCellSummary,
    LoopBinding,
    LoopCarry,
    LoopCountGuard,
    LoopDomain,
    PendingReader,
    ScopeSummary,
    StructuredNeed,
    StructuredReadCarry,
)

if TYPE_CHECKING:
    from .structured_admission import StructuredAdmission


@dataclass(frozen=True)
class StructuredDependencies:
    memory: NativeMemoryFacts
    admission: StructuredAdmission | None
    accesses: PartitionedAccesses
    summaries: Mapping[Operation, ScopeSummary]
    root: ScopeSummary
    needs: tuple[StructuredNeed, ...]
    required: tuple[StructuredNeed, ...]
    read_carries: tuple[StructuredReadCarry, ...]
    loop_carries: tuple[LoopCarry, ...]
    entries: tuple[StructuredNeed, ...]
    exits: tuple[ExitObligation, ...]
    unresolved: tuple[DependencyIssue, ...]
    work: DependencyWork

    @property
    def closed(self):
        return not self.unresolved

    def require_closed(self):
        if self.unresolved:
            issue = self.unresolved[0]
            raise CompileError(
                issue.message,
                kind=ErrorKind.UNSUPPORTED,
                phase="structured-dependencies",
                location=str(issue.operation.location),
                witness=(f"classification={issue.kind.value}",),
            )


def build_structured_dependencies(
    memory: NativeMemoryFacts, *, admission: StructuredAdmission | None = None
) -> StructuredDependencies:
    """Build actual recursive obligations; callers must reject unresolved results."""
    if admission is not None:
        require(
            admission.memory is memory, "dependency admission belongs to another memory snapshot"
        )
    return _Builder(memory, admission).build()


def _normalize(value, facts):
    # Declared uniform multiples can give the proof polynomial rational
    # coefficients (e.g. Sq/128). Keep its original exact quotient DAG when
    # that polynomial cannot be reconstructed with integer coefficients.
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


class _Builder:
    # Limits bound proof representation, never supply a synchronization fallback.
    MAX_CASES = 128

    def __init__(self, memory, admission):
        self.memory, self.admission = memory, admission
        self.facts = memory.control.environment.facts
        access = self.accesses = partition_accesses(memory, admission)
        self.work = access.work
        self.needs, self.carries, self.loop_carries = [], [], []
        self.issues = list(access.issues)
        self.summaries = {}
        self.effects, self.patterns = access.effects, access.patterns
        self._issue_keys = {(issue.kind, issue.operation, issue.cell) for issue in access.issues}

    def issue(self, kind, operation, message, cell=None):
        key = (kind, operation, cell)
        if key not in self._issue_keys:
            self._issue_keys.add(key)
            self.issues.append(DependencyIssue(kind, operation, message, cell))

    def _pattern(self, cell, loop):
        patterns = self.patterns.get((cell, loop), {(1, 0)})
        if len(patterns) != 1:
            self.issue(
                DependencyIssueKind.MULTI_RESIDUE,
                loop,
                "one cell has multiple activation residues; finite word closure is not implemented",
                cell,
            )
        return max(patterns, key=repr)

    def _domains(self, cell, occurrence):
        return tuple(
            LoopDomain(loop.operation, loop.count, *self._pattern(cell, loop.operation))
            for loop in occurrence.frame.loops()
            if any(
                binding.loop is loop.operation and binding.ordinal == loop.ordinal
                for binding in occurrence.bindings
            )
        )

    def _guards(self, *groups):
        guards = []
        for guard in (guard for group in groups for guard in group):
            if isinstance(guard, LoopCountGuard):
                count = _normalize(guard.count, self.facts)
                bounds = infer(count, self.facts)
                threshold = (
                    literal(guard.threshold) if type(guard.threshold) is int else guard.threshold
                )
                threshold_bounds = infer(threshold, self.facts)
                if threshold.kind == "rem" and threshold.args[1].constant is not None:
                    from ..scalar import Fact

                    threshold_bounds = Fact(0, 1, threshold.args[1].constant - 1)
                value = count.constant
                minimum = None if bounds is None else bounds.minimum
                maximum = None if bounds is None else bounds.maximum
                for prior in guards:
                    if (
                        isinstance(prior, LoopCountGuard)
                        and prior.count == guard.count
                        and type(prior.threshold) is int
                    ):
                        if prior.relation is CountRelation.MORE:
                            minimum = max(minimum or 0, prior.threshold + 1)
                        else:
                            maximum = (
                                prior.threshold
                                if maximum is None
                                else min(maximum, prior.threshold)
                            )
                if value is None and threshold_bounds is not None:
                    if (
                        minimum is not None
                        and threshold_bounds.maximum is not None
                        and minimum > threshold_bounds.maximum
                    ):
                        value = minimum
                        threshold = literal(threshold_bounds.maximum)
                    elif maximum is not None and maximum <= threshold_bounds.minimum:
                        value = maximum
                        threshold = literal(threshold_bounds.minimum)
                if value is None and bounds is not None and threshold.constant is not None:
                    if bounds.minimum > threshold.constant:
                        value = bounds.minimum
                    elif bounds.maximum is not None and bounds.maximum <= threshold.constant:
                        value = bounds.maximum
                if value is not None and threshold.constant is not None:
                    active = value > threshold.constant
                    if active != (guard.relation is CountRelation.MORE):
                        return None
                    continue
                for prior in guards:
                    if (
                        isinstance(prior, LoopCountGuard)
                        and prior.loop is guard.loop
                        and prior.count == guard.count
                        and prior.relation is not guard.relation
                        and type(prior.threshold) is int
                        and type(guard.threshold) is int
                    ):
                        more, at_most = (
                            (prior, guard)
                            if prior.relation is CountRelation.MORE
                            else (guard, prior)
                        )
                        if more.threshold >= at_most.threshold:
                            return None
            elif any(
                isinstance(prior, BranchGuard)
                and prior.operation is guard.operation
                and prior.branch != guard.branch
                for prior in guards
            ):
                return None
            if guard not in guards:
                guards.append(guard)
        return tuple(guards)

    def _need(
        self,
        cell,
        kind,
        source,
        consumer,
        scope,
        guards,
        *,
        loop=None,
        distance=None,
        period=None,
        residue=None,
        emit=True,
        first=0,
        limit=None,
    ):
        require(source != consumer, "distinct access occurrences need explicit loop bindings")
        domains = self._domains(cell, consumer)
        if loop is None:
            if consumer.bindings and isinstance(consumer.bindings[-1].ordinal, Expr):
                candidate = consumer.bindings[-1]
                if isinstance(source, BoundaryPort) or (
                    source.bindings
                    and source.bindings[-1].loop is candidate.loop
                    and isinstance(source.bindings[-1].ordinal, Expr)
                ):
                    loop = candidate.loop
                    if candidate.ordinal.constant is not None and limit is None:
                        first = max(first, candidate.ordinal.constant)
                        limit = expression(
                            "min",
                            self.memory.control.loops[loop].count,
                            literal(candidate.ordinal.constant + 1),
                        )
                    if distance is None:
                        distance = (
                            0
                            if isinstance(source, BoundaryPort)
                            else _normalize(
                                expression("sub", candidate.ordinal, source.bindings[-1].ordinal),
                                self.facts,
                            ).constant
                        )
            elif not consumer.bindings and (
                isinstance(source, BoundaryPort) or not source.bindings
            ):
                distance = 0
        if loop is not None:
            default_period, default_residue = self._pattern(cell, loop)
            period = default_period if period is None else period
            residue = default_residue if residue is None else residue
            domains = tuple(
                replace(
                    domain,
                    count=domain.count if limit is None else limit,
                    distance=distance or 0,
                    first=first,
                )
                if domain.loop is loop
                else domain
                for domain in domains
            )
            if not any(domain.loop is loop for domain in domains) and (first or limit is not None):
                domains += (
                    LoopDomain(
                        loop,
                        self.memory.control.loops[loop].count if limit is None else limit,
                        period,
                        residue,
                        distance or 0,
                        first,
                    ),
                )
        need = StructuredNeed(
            cell,
            kind,
            source,
            consumer,
            scope,
            domains,
            guards,
            loop,
            distance,
            period or 1,
            residue or 0,
        )
        if emit and isinstance(source, AccessOccurrence):
            self.needs.append(need)
            self.work.needs += 1
        return need

    def _native(self, operation):
        cells = {}
        for cell, accesses in self.effects.get(operation, {}).items():
            cases = []
            for template, participant, mode in accesses:
                occurrence = AccessOccurrence(
                    AccessEndpoint.from_template(template, participant),
                    tuple(
                        LoopBinding(loop.operation, loop.ordinal) for loop in template.frame.loops()
                    ),
                )
                reads, carries = (), ()
                writer = BoundaryPort(cell, BoundaryKind.WRITER)
                if mode.reads:
                    raw = self._need(
                        cell,
                        HazardKind.RAW,
                        writer,
                        occurrence,
                        operation,
                        cell.conditions,
                        emit=False,
                    )
                    reads = (PendingReader(occurrence, writer, raw),)
                write = occurrence if mode.writes else None
                if mode.reads and mode.writes:
                    waw = self._need(
                        cell,
                        HazardKind.WAW,
                        writer,
                        occurrence,
                        operation,
                        cell.conditions,
                        emit=False,
                    )
                    carries = (StructuredReadCarry(waw, reads[0].raw, None),)
                cases.append(
                    GuardedCellSummary(
                        cell.conditions,
                        CellSummary(
                            cell, reads, write, write, () if mode.writes else reads, carries
                        ),
                    )
                )
            # Multiple participant reads execute together, rather than being
            # alternative branches of the shared native template.
            summary = CellSummary(cell)
            for case in cases:
                summary = self._compose(summary, case.transfer, operation, cell.conditions)
            cells[cell] = (GuardedCellSummary(cell.conditions, summary),)
        return cells

    def _compose(self, left, right, scope, guards, **clock):
        self.work.boundary_visits += len(right.head_reads) + len(left.tail_reads)
        bound, raw_bindings = [], {}
        for read in right.head_reads:
            raw = self._need(
                left.cell, HazardKind.RAW, left.writer, read.reader, scope, guards, **clock
            )
            bound.append(PendingReader(read.reader, left.writer, raw))
            raw_bindings[id(read.raw)] = raw
        entry_carries = list(left.entry_carries)
        if right.first_write is not None:
            waw = self._need(
                left.cell, HazardKind.WAW, left.writer, right.first_write, scope, guards, **clock
            )
            for read in left.tail_reads:
                war = self._need(
                    left.cell,
                    HazardKind.WAR,
                    read.reader,
                    right.first_write,
                    scope,
                    guards,
                    **clock,
                )
                if read.writer == waw.source:
                    carry = StructuredReadCarry(waw, read.raw, war)
                    if isinstance(waw.source, BoundaryPort):
                        entry_carries.append(carry)
                    else:
                        self.carries.append(carry)
            for carry in right.entry_carries:
                rebound = StructuredReadCarry(waw, raw_bindings[id(carry.raw)], carry.war)
                if isinstance(waw.source, BoundaryPort):
                    entry_carries.append(rebound)
                else:
                    self.carries.append(rebound)
        head = left.head_reads + tuple(bound) if left.first_write is None else left.head_reads
        tail = right.tail_reads if right.first_write is not None else left.tail_reads + tuple(bound)
        return CellSummary(
            left.cell,
            head,
            left.first_write or right.first_write,
            right.last_write or left.last_write,
            tail,
            tuple(entry_carries),
        )

    def _sequence(self, left, right, scope):
        if not right:
            return left
        if not left:
            return right
        result = dict(left)
        for cell, right_cases in right.items():
            left_cases = result.get(cell, (GuardedCellSummary((), CellSummary(cell)),))
            cases = []
            for a, b in product(left_cases, right_cases):
                guards = self._guards(a.guards, b.guards)
                if guards is not None:
                    cases.append(
                        GuardedCellSummary(
                            guards, self._compose(a.transfer, b.transfer, scope, guards)
                        )
                    )
            if len(cases) > self.MAX_CASES:
                self.issue(
                    DependencyIssueKind.BUDGET,
                    scope,
                    "conditional boundary alternatives exceed dependency proof budget",
                    cell,
                )
                cases = cases[: self.MAX_CASES]
            result[cell] = tuple(cases)
        return result

    def _bind(self, occurrence, loop, ordinal, *, family=None):
        if occurrence is None:
            return None
        logical = self.memory.control.loops[loop]
        value = ordinal
        if isinstance(ordinal, LoopResidueSelection):
            residue = literal(ordinal.residue) if type(ordinal.residue) is int else ordinal.residue
            value = (
                residue
                if ordinal.position is ResiduePosition.FIRST
                else expression(
                    "sub",
                    expression("sub", logical.count, literal(1)),
                    expression(
                        "rem",
                        expression("sub", expression("sub", logical.count, literal(1)), residue),
                        literal(ordinal.period),
                    ),
                )
            )
        substitution = {logical.ordinal: value}

        def map_ordinal(other):
            if isinstance(other, Expr):
                return substitute(other, substitution)
            if isinstance(other.residue, Expr):
                selected = substitute(other.residue, substitution)
                return replace(
                    other, residue=selected.constant if selected.constant is not None else selected
                )
            return other

        bindings = tuple(
            replace(
                binding, ordinal=ordinal if binding.loop is loop else map_ordinal(binding.ordinal)
            )
            for binding in occurrence.bindings
        )
        families = tuple(
            replace(
                domain,
                count=substitute(domain.count, substitution),
                residue=substitute(domain.residue, substitution)
                if isinstance(domain.residue, Expr)
                else domain.residue,
            )
            for domain in occurrence.families
        ) + (() if family is None else (family,))
        return replace(occurrence, bindings=bindings, families=families)

    def _map_transfer(self, transfer, transform, scope, guards):
        bindings = {}

        def occurrence(value):
            return transform(value) if isinstance(value, AccessOccurrence) else value

        def need(value):
            key = id(value)
            if key not in bindings:
                bindings[key] = replace(
                    value,
                    source=occurrence(value.source),
                    consumer=occurrence(value.consumer),
                    scope=scope,
                    guards=guards,
                    instance_of=value.instance_of or value,
                )
            return bindings[key]

        def read(value):
            return PendingReader(
                occurrence(value.reader), occurrence(value.writer), need(value.raw)
            )

        return CellSummary(
            transfer.cell,
            tuple(read(r) for r in transfer.head_reads),
            transform(transfer.first_write),
            transform(transfer.last_write),
            tuple(read(r) for r in transfer.tail_reads),
            tuple(
                StructuredReadCarry(
                    need(c.waw), need(c.raw), None if c.war is None else need(c.war)
                )
                for c in transfer.entry_carries
            ),
        )

    def _guard_depends(self, guard, loop):
        if isinstance(guard, LoopCountGuard):
            return (
                not self.memory.control.invariant_over(guard.count, loop.operation)
                or isinstance(guard.threshold, Expr)
                and not self.memory.control.invariant_over(guard.threshold, loop.operation)
            )
        condition = guard.operation.cond.owner
        for operand in condition.operands:
            try:
                if not self.memory.control.invariant_over(
                    self.memory.control.environment.expression(operand), loop.operation
                ):
                    return True
            except CompileError:
                return True
        return False

    def _first_branch(self, guard, logical):
        if not isinstance(guard, BranchGuard) or not isinstance(
            guard.operation.cond.owner, arith.CmpiOp
        ):
            return None
        comparison = guard.operation.cond.owner
        if comparison.predicate.value.data not in {0, 1}:
            return None
        try:
            lhs, rhs = (self.memory.control.environment.expression(v) for v in comparison.operands)
        except CompileError:
            return None
        first = any(
            equivalent(a, logical.induction, self.facts)
            and equivalent(b, logical.lower, self.facts)
            or equivalent(a, logical.ordinal, self.facts)
            and equivalent(b, literal(0), self.facts)
            for a, b in ((lhs, rhs), (rhs, lhs))
        )
        if not first:
            return None
        first_branch = 0 if comparison.predicate.value.data == 0 else 1
        return guard.branch == first_branch

    def _last_branch(self, guard, logical):
        if not isinstance(guard, BranchGuard) or not isinstance(
            guard.operation.cond.owner, arith.CmpiOp
        ):
            return None
        comparison = guard.operation.cond.owner
        if comparison.predicate.value.data not in {0, 1}:
            return None
        try:
            lhs, rhs = (self.memory.control.environment.expression(v) for v in comparison.operands)
        except CompileError:
            return None
        last_ordinal = expression("sub", logical.count, literal(1))
        last_induction = expression(
            "add", logical.lower, expression("mul", last_ordinal, literal(logical.step))
        )
        candidates = (last_induction,)
        if logical.step == 1:
            candidates += (expression("sub", logical.upper, literal(1)),)
        last = any(
            equivalent(a, logical.induction, self.facts)
            and any(equivalent(b, last, self.facts) for last in candidates)
            or equivalent(a, logical.ordinal, self.facts)
            and equivalent(b, last_ordinal, self.facts)
            for a, b in ((lhs, rhs), (rhs, lhs))
        )
        if not last:
            return None
        last_branch = 0 if comparison.predicate.value.data == 0 else 1
        return guard.branch == last_branch

    def _phase_cases(self, cases, logical):
        phases = {"single": [], "first": [], "middle": [], "last": []}
        has_first = has_last = False
        for case in cases:
            constraints, guards = [], []
            for guard in case.guards:
                first = self._first_branch(guard, logical)
                last = self._last_branch(guard, logical) if first is None else None
                if first is not None:
                    has_first = True
                    constraints.append(("first", first))
                elif last is not None:
                    has_last = True
                    constraints.append(("last", last))
                elif self._guard_depends(guard, logical):
                    return None
                else:
                    guards.append(guard)
            for phase, table in (
                ("single", {"first": True, "last": True}),
                ("first", {"first": True, "last": False}),
                ("middle", {"first": False, "last": False}),
                ("last", {"first": False, "last": True}),
            ):
                if all(table[kind] == truth for kind, truth in constraints):
                    phases[phase].append(GuardedCellSummary(tuple(guards), case.transfer))
        return phases, has_first, has_last

    def _repeat_window(self, operation, transfer, lower, upper, guards, has_last):
        logical = self.memory.control.loops[operation]
        if transfer.first_write is None or operation in transfer.cell.streaming:
            family = LoopDomain(operation, upper, first=lower)
            return self._map_transfer(
                transfer,
                lambda value: self._bind(value, operation, logical.ordinal, family=family),
                operation,
                guards,
            )
        repeated = self._guards(
            guards,
            (
                LoopCountGuard(
                    operation, logical.count, CountRelation.MORE, lower + int(has_last) + 1
                ),
            ),
        )
        if repeated is not None:
            previous = self._map_transfer(
                transfer,
                lambda value: self._bind(
                    value, operation, expression("sub", logical.ordinal, literal(1))
                ),
                operation,
                repeated,
            )
            current = self._map_transfer(
                transfer,
                lambda value: self._bind(value, operation, logical.ordinal),
                operation,
                repeated,
            )
            before = len(self.needs)
            self._compose(
                previous,
                current,
                operation,
                repeated,
                loop=operation,
                distance=1,
                first=lower + 1,
                limit=upper,
            )
            self._record_loop_needs(operation, logical, 1, 0, before)
        head = self._map_transfer(
            transfer, lambda value: self._bind(value, operation, literal(lower)), operation, guards
        )
        tail = self._map_transfer(
            transfer,
            lambda value: self._bind(value, operation, expression("sub", upper, literal(1))),
            operation,
            guards,
        )
        return CellSummary(
            transfer.cell,
            head.head_reads,
            head.first_write,
            tail.last_write,
            tail.tail_reads,
            head.entry_carries,
        )

    def _close_phases(self, operation, cell, cases):
        logical = self.memory.control.loops[operation]
        phase_cases = self._phase_cases(cases, logical)
        if phase_cases is None or self._pattern(cell, operation) != (1, 0):
            return None
        phases, has_first, has_last = phase_cases
        if not has_first and not has_last:
            return None
        result = []
        for case in phases["single"]:
            empty = self._guards(
                case.guards, (LoopCountGuard(operation, logical.count, CountRelation.AT_MOST, 0),)
            )
            one = self._guards(
                case.guards,
                (
                    LoopCountGuard(operation, logical.count, CountRelation.MORE, 0),
                    LoopCountGuard(operation, logical.count, CountRelation.AT_MOST, 1),
                ),
            )
            if empty is not None:
                result.append(GuardedCellSummary(empty, CellSummary(cell)))
            if one is not None:
                result.append(
                    GuardedCellSummary(
                        one,
                        self._map_transfer(
                            case.transfer,
                            lambda value: self._bind(value, operation, literal(0)),
                            operation,
                            one,
                        ),
                    )
                )
        identity = GuardedCellSummary((), CellSummary(cell))
        first_cases = phases["first"] if has_first else (identity,)
        last_cases = phases["last"] if has_last else (identity,)
        lower = int(has_first)
        upper = expression("sub", logical.count, literal(int(has_last)))
        for first, middle, last in product(first_cases, phases["middle"], last_cases):
            guards = self._guards(
                first.guards,
                middle.guards,
                last.guards,
                (LoopCountGuard(operation, logical.count, CountRelation.MORE, 1),),
            )
            if guards is None:
                continue
            for populated in (False, True):
                live = self._guards(
                    guards,
                    (
                        LoopCountGuard(
                            operation,
                            logical.count,
                            CountRelation.MORE if populated else CountRelation.AT_MOST,
                            lower + int(has_last),
                        ),
                    ),
                )
                if live is None:
                    continue
                transfer = CellSummary(cell)
                if has_first:
                    transfer = self._map_transfer(
                        first.transfer,
                        lambda value: self._bind(value, operation, literal(0)),
                        operation,
                        live,
                    )
                if populated:
                    repeat = self._repeat_window(
                        operation, middle.transfer, lower, upper, live, has_last
                    )
                    transfer = self._compose(transfer, repeat, operation, live, first=lower)
                if has_last:
                    suffix = self._map_transfer(
                        last.transfer,
                        lambda value: self._bind(
                            value, operation, expression("sub", logical.count, literal(1))
                        ),
                        operation,
                        live,
                    )
                    transfer = self._compose(transfer, suffix, operation, live)
                result.append(GuardedCellSummary(live, transfer))
        return tuple(result)

    def _record_loop_needs(self, operation, logical, period, residue, before):
        for need in self.needs[before:]:
            self.loop_carries.append(LoopCarry(operation, logical.count, period, residue, need))
            self.work.loop_carries += 1

    def _close(self, operation, body):
        logical = self.memory.control.loops[operation]
        result = {}
        for cell, cases in body.items():
            period, residue = self._pattern(cell, operation)
            if (
                any(
                    self._first_branch(guard, logical) is not None
                    or self._last_branch(guard, logical) is not None
                    for case in cases
                    for guard in case.guards
                )
                and (phase_cases := self._close_phases(operation, cell, cases)) is not None
            ):
                result[cell] = phase_cases
                continue
            closed = []
            for case in cases:
                if any(self._guard_depends(guard, logical) for guard in case.guards):
                    self.issue(
                        DependencyIssueKind.PARENT_ACTIVATION,
                        operation,
                        "parent-dependent conditional child needs an activation-family closure",
                        cell,
                    )
                    closed.append(case)
                    continue
                active = self._guards(
                    case.guards,
                    (LoopCountGuard(operation, logical.count, CountRelation.MORE, residue),),
                )
                empty = self._guards(
                    case.guards,
                    (LoopCountGuard(operation, logical.count, CountRelation.AT_MOST, residue),),
                )
                if empty is not None:
                    closed.append(GuardedCellSummary(empty, CellSummary(cell)))
                if active is None:
                    continue
                transfer = case.transfer
                family = LoopDomain(operation, logical.count, period, residue)
                if operation in cell.streaming or transfer.first_write is None:
                    # Distinct streaming cells or read-only native families:
                    # no final native end can retire all prior members.
                    mapped = self._map_transfer(
                        transfer,
                        lambda value, family=family: self._bind(
                            value, operation, logical.ordinal, family=family
                        ),
                        operation,
                        active,
                    )
                    closed.append(GuardedCellSummary(active, mapped))
                    continue
                carry_guards = self._guards(
                    active,
                    (
                        LoopCountGuard(
                            operation,
                            logical.count,
                            CountRelation.MORE,
                            residue + period
                            if type(residue) is int
                            else expression("add", residue, literal(period)),
                        ),
                    ),
                )
                if carry_guards is not None:
                    previous = self._map_transfer(
                        transfer,
                        lambda value, period=period: self._bind(
                            value, operation, expression("sub", logical.ordinal, literal(period))
                        ),
                        operation,
                        carry_guards,
                    )
                    current = self._map_transfer(
                        transfer,
                        lambda value: self._bind(value, operation, logical.ordinal),
                        operation,
                        carry_guards,
                    )
                    before = len(self.needs)
                    self._compose(
                        previous,
                        current,
                        operation,
                        carry_guards,
                        loop=operation,
                        distance=period,
                        period=period,
                        residue=residue,
                    )
                    self._record_loop_needs(operation, logical, period, residue, before)
                first = LoopResidueSelection(operation, period, residue, ResiduePosition.FIRST)
                last = LoopResidueSelection(operation, period, residue, ResiduePosition.LAST)
                head = self._map_transfer(
                    transfer,
                    lambda value, first=first: self._bind(value, operation, first),
                    operation,
                    active,
                )
                tail = self._map_transfer(
                    transfer,
                    lambda value, last=last: self._bind(value, operation, last),
                    operation,
                    active,
                )
                closed.append(
                    GuardedCellSummary(
                        active,
                        CellSummary(
                            cell,
                            head.head_reads,
                            head.first_write,
                            tail.last_write,
                            tail.tail_reads,
                            head.entry_carries,
                        ),
                    )
                )
            result[cell] = tuple(closed)
        return result

    def _node(self, operation):
        self.work.scope_visits += 1
        if operation in self.effects:
            cells = self._native(operation)
        elif (
            operation in self.memory.control.loops
            and _normalize(self.memory.control.loops[operation].count, self.facts).constant == 0
        ):
            cells = {}
        elif isinstance(operation, scf.IfOp):
            branches = []
            for region in operation.regions:
                branch = {}
                for block in region.blocks:
                    for child in block.ops:
                        branch = self._sequence(branch, self._node(child), operation)
                branches.append(branch)
            cells = {}
            for cell in dict.fromkeys(cell for branch in branches for cell in branch):
                cases = []
                for index, branch in enumerate(branches):
                    for case in branch.get(cell, (GuardedCellSummary((), CellSummary(cell)),)):
                        guards = self._guards(case.guards, (BranchGuard(operation, index),))
                        if guards is not None:
                            cases.append(GuardedCellSummary(guards, case.transfer))
                cells[cell] = tuple(cases)
        else:
            cells = {}
            if len(operation.regions) > 1:
                self.issue(
                    DependencyIssueKind.CONTROL,
                    operation,
                    "memory-bearing control needs an explicit branch/loop contract",
                )
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.ops:
                        cells = self._sequence(cells, self._node(child), operation)
            if operation in self.memory.control.loops:
                cells = self._close(operation, cells)
        if cells or operation in self.memory.control.loops or operation is self.memory.module:
            self.summaries[operation] = ScopeSummary(operation, MappingProxyType(cells))
        return cells

    def build(self):
        cells = self._node(self.memory.module)
        root = ScopeSummary(self.memory.module, MappingProxyType(cells))
        entries, exits = [], []
        for cell, cases in cells.items():
            for case in cases:
                transfer = case.transfer
                entries.extend(
                    replace(read.raw, guards=case.guards) for read in transfer.head_reads
                )
                if transfer.first_write is not None:
                    entries.append(
                        self._need(
                            cell,
                            HazardKind.WAW,
                            BoundaryPort(cell, BoundaryKind.WRITER),
                            transfer.first_write,
                            self.memory.module,
                            case.guards,
                            emit=False,
                        )
                    )
                    entries.append(
                        self._need(
                            cell,
                            HazardKind.WAR,
                            BoundaryPort(cell, BoundaryKind.READERS),
                            transfer.first_write,
                            self.memory.module,
                            case.guards,
                            emit=False,
                        )
                    )
                if transfer.last_write is not None:
                    exits.append(ExitObligation(cell, transfer.last_write, True, case.guards))
                exits.extend(
                    ExitObligation(cell, read.reader, False, case.guards)
                    for read in transfer.tail_reads
                )
        emitted = {id(need) for need in self.needs}

        def available(need):
            if id(need) in emitted:
                return True
            template = need.instance_of
            return (
                template is not None
                and id(template) in emitted
                and all(
                    not isinstance(guard, BranchGuard) or guard in need.guards
                    for guard in template.guards
                )
            )

        covered = {
            id(carry.waw)
            for carry in self.carries
            if available(carry.raw) and (carry.war is None or available(carry.war))
        }
        return StructuredDependencies(
            self.memory,
            self.admission,
            self.accesses,
            MappingProxyType(self.summaries),
            root,
            tuple(self.needs),
            tuple(need for need in self.needs if id(need) not in covered),
            tuple(self.carries),
            tuple(self.loop_carries),
            tuple(entries),
            tuple(exits),
            tuple(self.issues),
            self.work,
        )
