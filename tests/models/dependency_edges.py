"""Finite obligation expansion and byte-trace hazards, independent of production ordering."""

from ascendflow.analysis.hazards import HazardKind
from ascendflow.analysis.structured_regions import (
    AccessOccurrence,
)
from tests.models.structured_domains import expression_value, guard_active, occurrence_key


def materialized_edges(result, values):
    """Small independent interpreter for quantified obligations, never production."""
    edges = set()
    loops = tuple(result.memory.control.loops.values())
    for need in result.needs:
        assert isinstance(need.source, AccessOccurrence)
        domains = need.domains + need.source.families + need.consumer.families
        required = {domain.loop for domain in domains}
        selected = [loop for loop in loops if loop.operation in required]

        def visit(index, bindings, selected=selected, need=need, domains=domains):
            if index == len(selected):
                if not all(guard_active(g, bindings, result.memory) for g in need.guards):
                    return
                edges.add(
                    (
                        need.kind,
                        occurrence_key(need.source, bindings, result.memory),
                        occurrence_key(need.consumer, bindings, result.memory),
                    )
                )
                return
            logical = selected[index]
            current = [domain for domain in domains if domain.loop is logical.operation]
            limit = max(domain.count.evaluate(bindings) for domain in current)
            for ordinal in range(limit):
                context = bindings | {logical.ordinal.args[0]: ordinal}
                if all(
                    ordinal >= max(domain.first, domain.distance)
                    and ordinal < domain.count.evaluate(context)
                    and ordinal % domain.period == expression_value(domain.residue, context)
                    for domain in current
                ):
                    visit(index + 1, context)

        visit(0, values)
    return edges


def trace_edges(events):
    writers, readers, edges = {}, {}, set()
    for endpoint, accesses in events:
        for cell, reads, writes in accesses:
            writer = writers.get(cell)
            if reads and writer is not None:
                edges.add((HazardKind.RAW, writer, endpoint))
            if writes:
                if writer is not None:
                    edges.add((HazardKind.WAW, writer, endpoint))
                edges.update((HazardKind.WAR, reader, endpoint) for reader in readers.get(cell, ()))
                writers[cell], readers[cell] = endpoint, []
            elif reads:
                readers.setdefault(cell, []).append(endpoint)
    return edges
