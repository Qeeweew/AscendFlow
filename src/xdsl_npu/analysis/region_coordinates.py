"""Shared exact coordinate, guard and frontier-binding operations.

These operations construct immutable records under caller-owned facts. They do
not traverse a program, choose a writer, select storage reuse, emit a hazard,
or perform notification/lifetime analysis.
"""

from __future__ import annotations

from dataclasses import replace

from ..diagnostics import CompileError, require
from ..indexing import normalize
from ..scalar import Expr, expression, infer, literal, substitute
from .access_frontier import LoopResidueSelection, ResiduePosition
from .structured_regions import (
    AccessOccurrence,
    BoundaryPort,
    BranchGuard,
    CellSummary,
    CountRelation,
    DependencyIssueKind,
    LoopCountGuard,
    LoopDomain,
    LoopOrdinalGuard,
    StructuredNeed,
)
from .structured_schedule import Comparison


def normalize_coordinate(value, facts):
    # Declared uniform multiples can give the proof polynomial rational
    # coefficients (e.g. Sq/128). Keep its original exact quotient DAG when
    # that polynomial cannot be reconstructed with integer coefficients.
    try:
        return normalize(value, facts)
    except CompileError as error:
        if error.message != "runtime address retains a fractional coefficient":
            raise
        return value


class RegionCoordinates:
    """One owned coordinate environment shared by value and storage stages."""

    def __init__(self, memory, patterns, issue):
        self.memory, self.patterns, self.issue = memory, patterns, issue
        self.facts = memory.control.environment.facts

    def pattern(self, cell, loop):
        patterns = self.patterns.get((cell, loop), {(1, 0)})
        if len(patterns) != 1:
            self.issue(
                DependencyIssueKind.MULTI_RESIDUE,
                loop,
                "one cell has multiple activation residues; finite word closure is not implemented",
                cell,
            )
        return max(patterns, key=repr)

    def domains(self, cell, occurrence):
        return tuple(
            LoopDomain(loop.operation, loop.count, *self.pattern(cell, loop.operation))
            for loop in occurrence.frame.loops()
            if any(
                binding.loop is loop.operation and binding.ordinal == loop.ordinal
                for binding in occurrence.bindings
            )
        )

    def guards(self, *groups):
        guards = []
        for guard in (guard for group in groups for guard in group):
            if isinstance(guard, LoopCountGuard):
                count = normalize_coordinate(guard.count, self.facts)
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
            elif isinstance(guard, LoopOrdinalGuard):
                test = guard.comparison
                difference = normalize_coordinate(expression("sub", test.lhs, test.rhs), self.facts)
                if difference.constant is not None:
                    if not Comparison(difference, test.predicate, literal(0)).evaluate({}):
                        return None
                    continue
                opposite = {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
                if any(
                    isinstance(prior, LoopOrdinalGuard)
                    and prior.loop is guard.loop
                    and prior.comparison.lhs == test.lhs
                    and prior.comparison.rhs == test.rhs
                    and prior.comparison.predicate == opposite[test.predicate]
                    for prior in guards
                ):
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

    def bind(self, occurrence, loop, ordinal, *, family=None):
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

    def map_summary(self, transfer, transform, scope, guards):
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

        return CellSummary(
            transfer.cell,
            tuple(need(r) for r in transfer.head_reads),
            transform(transfer.first_write),
            transform(transfer.last_write),
            tuple(need(r) for r in transfer.tail_reads),
        )

    def need(
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
        first=0,
        limit=None,
    ):
        require(source != consumer, "distinct access occurrences need explicit loop bindings")
        domains = self.domains(cell, consumer)
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
                            else normalize_coordinate(
                                expression("sub", candidate.ordinal, source.bindings[-1].ordinal),
                                self.facts,
                            ).constant
                        )
            elif not consumer.bindings and (
                isinstance(source, BoundaryPort) or not source.bindings
            ):
                distance = 0
        if loop is not None:
            default_period, default_residue = self.pattern(cell, loop)
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
        return need
