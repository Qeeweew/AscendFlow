"""Bind source reads to their exact logical cell definitions.

One recursive frontier owns source value versions, guarded writes and loop
bindings. It generates RAW bindings only. Actual dispatch storage lifetimes,
WAR/WAW ordering, completion and notification resources belong to later stages.
Runtime loop counts are never enumerated; child boundaries retain FIRST/LAST
selections and complete reader families rather than execution histories.
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
from ..indexing import equivalent
from ..scalar import Expr, expression, literal
from .access_frontier import AccessEndpoint, LoopResidueSelection, ResiduePosition
from .hazards import HazardKind
from .native import AccessMode
from .region_coordinates import RegionCoordinates, normalize_coordinate
from .structured_access import PartitionedAccesses, partition_accesses
from .structured_memory import NativeMemoryFacts
from .structured_regions import (
    AccessOccurrence,
    BoundaryPort,
    BranchGuard,
    CellSummary,
    CountRelation,
    DependencyIssue,
    DependencyIssueKind,
    DependencyWork,
    ExitObligation,
    GuardedCellSummary,
    LogicalWrite,
    LogicalWriteKind,
    LoopBinding,
    LoopCountGuard,
    LoopDomain,
    LoopOrdinalGuard,
    RegionConnection,
    ScopeSummary,
    StructuredNeed,
)

if TYPE_CHECKING:
    from .structured_admission import StructuredAdmission


@dataclass(frozen=True)
class LogicalValueFlow:
    memory: NativeMemoryFacts
    admission: StructuredAdmission | None
    accesses: PartitionedAccesses
    summaries: Mapping[Operation, ScopeSummary]
    root: ScopeSummary
    reads: tuple[StructuredNeed, ...]
    writes: tuple[LogicalWrite, ...]
    connections: tuple[RegionConnection, ...]
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
                phase="logical-valueflow",
                location=str(issue.operation.location),
                witness=(f"classification={issue.kind.value}",),
            )


def build_logical_valueflow(
    memory: NativeMemoryFacts, *, admission: StructuredAdmission | None = None
) -> LogicalValueFlow:
    """Bind source values once; callers reject unresolved closure."""
    if admission is not None:
        require(
            admission.memory is memory, "valueflow admission belongs to another memory snapshot"
        )
    return _Builder(memory, admission).build()


class _Builder:
    # Limits bound proof representation, never supply a synchronization fallback.
    MAX_CASES = 128

    def __init__(self, memory, admission):
        self.memory, self.admission = memory, admission
        self.facts = memory.control.environment.facts
        access = self.accesses = partition_accesses(memory, admission)
        self.work = access.work
        self.reads, self.writes, self.connections = [], [], []
        self.issues = list(access.issues)
        self.summaries = {}
        self.effects = access.effects
        self.coordinates = RegionCoordinates(memory, access.patterns, self.issue)
        self._issue_keys = {(issue.kind, issue.operation, issue.cell) for issue in access.issues}

    def issue(self, kind, operation, message, cell=None):
        key = (kind, operation, cell)
        if key not in self._issue_keys:
            self._issue_keys.add(key)
            self.issues.append(DependencyIssue(kind, operation, message, cell))

    def _read(self, cell, source, consumer, scope, guards, *, emit=True, **clock):
        need = self.coordinates.need(cell, HazardKind.RAW, source, consumer, scope, guards, **clock)
        if emit and isinstance(source, AccessOccurrence):
            self.reads.append(need)
            self.work.needs += 1
        return need

    def _effect_cases(self, effect):
        reads = tuple(access for access in effect.accesses if access.mode.reads)
        writes = any(access.mode.writes for access in effect.accesses)

        def mode(read):
            return (
                AccessMode.READ_WRITE
                if read and writes
                else AccessMode.READ
                if read
                else AccessMode.WRITE
                if writes
                else None
            )

        if not reads or any(access.read_condition is None for access in reads):
            return ((mode(bool(reads)), ()),)
        conditions = tuple(dict.fromkeys(access.read_condition for access in reads))
        if len(conditions) != 1:
            self.issue(
                DependencyIssueKind.CONTROL,
                effect.template.operation,
                "one native cell has distinct conditional read predicates; exact effect partition is not established",
            )
            return ()
        (condition,) = conditions
        return (
            (mode(True), (LoopOrdinalGuard(condition.loop, condition.read),)),
            (mode(False), (LoopOrdinalGuard(condition.loop, condition.kill),)),
        )

    def _native(self, operation):
        cells = {}
        for cell, effects in self.effects.get(operation, {}).items():
            cases = (GuardedCellSummary(cell.conditions, CellSummary(cell)),)
            for effect in effects:
                template, participant = effect.template, effect.participant
                occurrence = AccessOccurrence(
                    AccessEndpoint.from_template(template, participant),
                    tuple(
                        LoopBinding(loop.operation, loop.ordinal) for loop in template.frame.loops()
                    ),
                )
                alternatives = []
                for mode, predicates in self._effect_cases(effect):
                    guards = self.coordinates.guards(cell.conditions, predicates)
                    if guards is None:
                        continue
                    reads = ()
                    if mode is not None and mode.reads:
                        reads = (
                            self._read(
                                cell,
                                BoundaryPort(cell),
                                occurrence,
                                operation,
                                guards,
                                emit=False,
                            ),
                        )
                    write = occurrence if mode is not None and mode.writes else None
                    if write is not None:
                        activation = self.coordinates.guards(
                            guards,
                            tuple(
                                BranchGuard(frame.operation, frame.branch)
                                for frame in template.frame.ancestors()
                                if frame.branch is not None
                            ),
                        )
                        if activation is not None:
                            self.writes.append(
                                LogicalWrite(
                                    cell,
                                    occurrence,
                                    LogicalWriteKind.UPDATE
                                    if mode.reads
                                    else LogicalWriteKind.KILL,
                                    activation,
                                    self.coordinates.domains(cell, occurrence),
                                    tuple(
                                        access for access in effect.accesses if access.mode.writes
                                    ),
                                )
                            )
                    alternatives.append(
                        GuardedCellSummary(
                            guards,
                            CellSummary(
                                cell, reads, write, write, () if write is not None else reads
                            ),
                        )
                    )
                # Distinct participants execute together; read/kill predicates
                # of one native effect are alternatives, never a sequence.
                combined = []
                for left, right in product(cases, alternatives):
                    guards = self.coordinates.guards(left.guards, right.guards)
                    if guards is not None:
                        combined.append(
                            GuardedCellSummary(
                                guards,
                                self._compose(left.transfer, right.transfer, operation, guards),
                            )
                        )
                cases = tuple(combined)
            cells[cell] = cases
        return cells

    def _compose(self, left, right, scope, guards, **clock):
        # RAW binding below still visits every read. Only an incoming write
        # can consume a physical retirement splice in storage_lifetimes.
        if right.first_write is not None:
            self.connections.append(RegionConnection(scope, left, right, guards, **clock))
        self.work.boundary_visits += len(right.head_reads) + len(left.tail_reads)
        bound = []
        for read in right.head_reads:
            raw = self._read(left.cell, left.writer, read.consumer, scope, guards, **clock)
            bound.append(raw)
        head = left.head_reads + tuple(bound) if left.first_write is None else left.head_reads
        tail = right.tail_reads if right.first_write is not None else left.tail_reads + tuple(bound)
        return CellSummary(
            left.cell,
            head,
            left.first_write or right.first_write,
            right.last_write or left.last_write,
            tail,
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
                guards = self.coordinates.guards(a.guards, b.guards)
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

    def _guard_depends(self, guard, loop):
        if isinstance(guard, LoopOrdinalGuard):
            return any(
                not self.memory.control.invariant_over(value, loop.operation)
                for value in (guard.comparison.lhs, guard.comparison.rhs)
            )
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
        if isinstance(guard, LoopOrdinalGuard):
            if guard.loop is not logical.operation or guard.comparison.predicate not in {
                "==",
                "!=",
            }:
                return None
            test = guard.comparison
            if any(
                equivalent(a, logical.ordinal, self.facts) and equivalent(b, literal(0), self.facts)
                for a, b in ((test.lhs, test.rhs), (test.rhs, test.lhs))
            ):
                return test.predicate == "=="
            return None
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
        if isinstance(guard, LoopOrdinalGuard):
            if guard.loop is not logical.operation or guard.comparison.predicate not in {
                "==",
                "!=",
            }:
                return None
            test = guard.comparison
            last = expression("sub", logical.count, literal(1))
            if any(
                equivalent(a, logical.ordinal, self.facts) and equivalent(b, last, self.facts)
                for a, b in ((test.lhs, test.rhs), (test.rhs, test.lhs))
            ):
                return test.predicate == "=="
            return None
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
            return self.coordinates.map_summary(
                transfer,
                lambda value: self.coordinates.bind(
                    value, operation, logical.ordinal, family=family
                ),
                operation,
                guards,
            )
        repeated = self.coordinates.guards(
            guards,
            (
                LoopCountGuard(
                    operation, logical.count, CountRelation.MORE, lower + int(has_last) + 1
                ),
            ),
        )
        if repeated is not None:
            previous = self.coordinates.map_summary(
                transfer,
                lambda value: self.coordinates.bind(
                    value, operation, expression("sub", logical.ordinal, literal(1))
                ),
                operation,
                repeated,
            )
            current = self.coordinates.map_summary(
                transfer,
                lambda value: self.coordinates.bind(value, operation, logical.ordinal),
                operation,
                repeated,
            )
            before = len(self.reads)
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
            self.work.loop_carries += len(self.reads) - before
        head = self.coordinates.map_summary(
            transfer,
            lambda value: self.coordinates.bind(value, operation, literal(lower)),
            operation,
            guards,
        )
        tail = self.coordinates.map_summary(
            transfer,
            lambda value: self.coordinates.bind(
                value, operation, expression("sub", upper, literal(1))
            ),
            operation,
            guards,
        )
        return CellSummary(
            transfer.cell,
            head.head_reads,
            head.first_write,
            tail.last_write,
            tail.tail_reads,
        )

    def _close_phases(self, operation, cell, cases):
        logical = self.memory.control.loops[operation]
        phase_cases = self._phase_cases(cases, logical)
        if phase_cases is None or self.coordinates.pattern(cell, operation) != (1, 0):
            return None
        phases, has_first, has_last = phase_cases
        if not has_first and not has_last:
            return None
        result = []
        for case in phases["single"]:
            empty = self.coordinates.guards(
                case.guards, (LoopCountGuard(operation, logical.count, CountRelation.AT_MOST, 0),)
            )
            one = self.coordinates.guards(
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
                        self.coordinates.map_summary(
                            case.transfer,
                            lambda value: self.coordinates.bind(value, operation, literal(0)),
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
            guards = self.coordinates.guards(
                first.guards,
                middle.guards,
                last.guards,
                (LoopCountGuard(operation, logical.count, CountRelation.MORE, 1),),
            )
            if guards is None:
                continue
            for populated in (False, True):
                live = self.coordinates.guards(
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
                    transfer = self.coordinates.map_summary(
                        first.transfer,
                        lambda value: self.coordinates.bind(value, operation, literal(0)),
                        operation,
                        live,
                    )
                if populated:
                    repeat = self._repeat_window(
                        operation, middle.transfer, lower, upper, live, has_last
                    )
                    transfer = self._compose(transfer, repeat, operation, live, first=lower)
                if has_last:
                    suffix = self.coordinates.map_summary(
                        last.transfer,
                        lambda value: self.coordinates.bind(
                            value, operation, expression("sub", logical.count, literal(1))
                        ),
                        operation,
                        live,
                    )
                    transfer = self._compose(transfer, suffix, operation, live)
                result.append(GuardedCellSummary(live, transfer))
        return tuple(result)

    def _close(self, operation, body):
        logical = self.memory.control.loops[operation]
        result = {}
        for cell, cases in body.items():
            period, residue = self.coordinates.pattern(cell, operation)
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
                active = self.coordinates.guards(
                    case.guards,
                    (LoopCountGuard(operation, logical.count, CountRelation.MORE, residue),),
                )
                empty = self.coordinates.guards(
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
                    mapped = self.coordinates.map_summary(
                        transfer,
                        lambda value, family=family: self.coordinates.bind(
                            value, operation, logical.ordinal, family=family
                        ),
                        operation,
                        active,
                    )
                    closed.append(GuardedCellSummary(active, mapped))
                    continue
                carry_guards = self.coordinates.guards(
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
                    previous = self.coordinates.map_summary(
                        transfer,
                        lambda value, period=period: self.coordinates.bind(
                            value, operation, expression("sub", logical.ordinal, literal(period))
                        ),
                        operation,
                        carry_guards,
                    )
                    current = self.coordinates.map_summary(
                        transfer,
                        lambda value: self.coordinates.bind(value, operation, logical.ordinal),
                        operation,
                        carry_guards,
                    )
                    before = len(self.reads)
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
                    self.work.loop_carries += len(self.reads) - before
                first = LoopResidueSelection(operation, period, residue, ResiduePosition.FIRST)
                last = LoopResidueSelection(operation, period, residue, ResiduePosition.LAST)
                head = self.coordinates.map_summary(
                    transfer,
                    lambda value, first=first: self.coordinates.bind(value, operation, first),
                    operation,
                    active,
                )
                tail = self.coordinates.map_summary(
                    transfer,
                    lambda value, last=last: self.coordinates.bind(value, operation, last),
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
            and normalize_coordinate(
                self.memory.control.loops[operation].count, self.facts
            ).constant
            == 0
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
                        guards = self.coordinates.guards(
                            case.guards, (BranchGuard(operation, index),)
                        )
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
                entries.extend(replace(read, guards=case.guards) for read in transfer.head_reads)
                if transfer.last_write is not None:
                    exits.append(ExitObligation(cell, transfer.last_write, True, case.guards))
                exits.extend(
                    ExitObligation(cell, read.consumer, False, case.guards)
                    for read in transfer.tail_reads
                )
        return LogicalValueFlow(
            self.memory,
            self.admission,
            self.accesses,
            MappingProxyType(self.summaries),
            root,
            tuple(self.reads) + tuple(entries),
            tuple(self.writes),
            tuple(self.connections),
            tuple(entries),
            tuple(exits),
            tuple(self.issues),
            self.work,
        )
