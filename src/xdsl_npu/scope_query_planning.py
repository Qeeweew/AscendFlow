"""Plan missing local query memory releases on the actual retained program.

This internal stage does not grant kernel verification: old child/parent
event resources, cross-counter backlog and full progress remain separate.
It never changes buffers, native instructions, or source scheduling policy.
"""

from dataclasses import dataclass, replace

from xdsl.dialects.builtin import ArrayAttr, IntAttr

from .budget import bounded, poll
from .completion import coalesce
from .ir import QueryChannelOp
from .model import require
from .scope_domains import affine
from .scope_query_checks import check_memory, prepare_memory_checks
from .scope_query_completion import QueryCompletion, QueryEvent
from .scope_query_protocols import (
    QueryProtocol,
    install_query_protocols,
    read_query_protocols,
    validate_query_protocol,
)
from .symbolic_completion import holds, meet


@dataclass(frozen=True)
class QueryMemoryPlan:
    protocols: tuple[QueryProtocol, ...]
    memory_checks: int
    initially_missing: int
    new_event_reuse_checks: int


def _candidates(checks):
    """Join count cases only when their actual endpoint expressions coincide.

    Short summaries may name LAST as constant 0 at N=1 or 1 at N=2. Match
    those to an existing symbolic boundary over the complete case domain;
    coalesce preserves count holes. This changes no native or waiting scope.
    """
    families, groups = {}, {}

    def identity(check):
        return (
            check.source.query,
            check.source.call,
            check.source.site,
            check.target.query,
            check.target.call,
            check.target.site,
        )

    for check in checks:
        if check.domain.exact is None:
            families.setdefault(identity(check), []).append((check.source, check.target))
    for check in checks:
        source, target = check.source, check.target
        if check.domain.exact is not None:
            for first, last in families.get(identity(check), ()):
                if all(
                    holds(affine(a.parent), affine(b.parent), check.domain)
                    and holds(affine(b.parent), affine(a.parent), check.domain)
                    for a, b in ((first, source), (last, target))
                ):
                    source, target = first, last
                    break
        groups.setdefault((source, target), []).append(check)
    result = []
    for (source, target), members in groups.items():
        for domain in coalesce(member.domain for member in members):
            result.append(
                (
                    QueryProtocol(f"query_c{len(result)}", source, target, domain),
                    tuple(member for member in members if meet(domain, member.domain) is not None),
                )
            )
    return result


def _query_width(parent, copies, candidate, maximum_width, memory_queries):
    """Try one actual ID before enlarging its static reuse proof window."""
    # The last represented query's outgoing Set is real under the outer
    # existence guard, even if its consumer is outside the proof window.
    proof = QueryCompletion(parent, copies=max(copies, candidate.target.query + 1))
    actor = candidate.source.site.task[0]
    source = QueryEvent(
        candidate.target.query, actor, candidate.name, "wait", candidate.target.parent
    )

    def reuse_query(width):
        return (
            source,
            QueryEvent(width, actor, candidate.name, "set", candidate.source.parent),
            candidate.domain,
        )

    # check_memory already validated every prepared allocation identity.
    # Keep its exact request order, registering all origins in one batch.
    single, *answers = proof.covered_many((reuse_query(1), *memory_queries))
    if single:
        return 1, answers, 1
    require(
        maximum_width > 1,
        f"{candidate.name}: available event widths cannot prove old Wait before re-Set",
    )
    if proof.copies < maximum_width + 1:
        proof = QueryCompletion(parent, copies=maximum_width + 1)
    resources = tuple(reuse_query(width) for width in range(2, maximum_width + 1))
    covered = proof.covered_many(resources)
    require(
        any(covered),
        f"{candidate.name}: available event widths cannot prove old Wait before re-Set",
    )
    return covered.index(True) + 2, answers, maximum_width


@bounded
def plan_query_memory(checked, parent) -> QueryMemoryPlan:
    """Materialize actual local protocols, then prove the new ID rotation.

    Selection queries completion first; already-covered memory creates no
    instruction. Candidates use distinct free IDs from the real physical bank.
    A finite search of widths 1..8 concerns hardware resources only, never
    runtime query/KV counts. Unsupported kinds or resource conflicts reject.
    """
    from .scope_sync import _leaf_banks

    require(
        checked.program is parent.program and checked.access is parent.access,
        "query planner needs the actual checked invocation proof",
    )
    require(
        not read_query_protocols(checked.program, checked.access),
        "query memory planning requires no pre-existing query channels",
    )
    checks = prepare_memory_checks(checked, parent)
    copies = len(checked.query_summary.parts)
    initial = QueryCompletion(parent, copies=copies)
    answers = check_memory(checks, initial)
    memory_queries = tuple((check.source, check.target, check.domain) for check in checks)
    validity = {
        (check.source, check.target, check.domain): valid
        for check, valid in zip(checks, answers, strict=True)
    }
    missing = [check for check, valid in zip(checks, answers, strict=True) if not valid]
    tasks = {
        (actor, task.uid): task
        for actor, data in checked.access.loop.actors.items()
        for task in data.program.tasks
    }
    owners = {
        (uid, replace(site, point="begin")): owner
        for owner, (uid, site) in checked.access.endpoints.items()
    }

    def bank(protocol):
        source, target = (
            owners[endpoint.call, replace(endpoint.site, point="begin")]
            for endpoint in (protocol.source, protocol.target)
        )
        return source[0], tasks[source].pipe, tasks[target].pipe

    # A supported release can also cover same-pipe/cross-actor obligations.
    # Decide those only after the actual release edges are present; do not
    # reject a redundant relation merely because it has no direct local form.
    candidates = sorted(
        _candidates(missing),
        key=lambda item: (
            item[0].source.site.task[0] != item[0].target.site.task[0]
            or bank(item[0])[1] == bank(item[0])[2],
            item[0].target.query,
        ),
    )
    used = _leaf_banks(checked.program, checked.analyses)
    for protocol in checked.channels.values():
        if protocol.kind == "local":
            source, target = protocol.sources[0], protocol.destinations[0]
            used[source[0], tasks[source].pipe, tasks[target].pipe].update(protocol.ids)
    selected, reuse_checks = [], 0
    for candidate, members in candidates:
        poll("query memory protocol selection", channel=candidate.name)
        # Earlier selected actual protocols may have covered this obligation.
        if all(validity[c.source, c.target, c.domain] for c in members):
            continue
        validate_query_protocol(checked.program, checked.access, candidate)
        resource = bank(candidate)
        free = tuple(event for event in range(8) if event not in used[resource])
        require(free, f"{resource}: query release has no free binary event ID")
        candidate = replace(candidate, ids=(free[0],))
        install_query_protocols(checked.program, checked.access, (candidate,))
        width, answers, attempted = _query_width(
            parent, copies, candidate, len(free), memory_queries
        )
        reuse_checks += attempted
        candidate = replace(candidate, ids=free[:width])
        declaration = next(
            op
            for op in checked.program.root.body.block.ops
            if isinstance(op, QueryChannelOp) and op.sym_name.data == candidate.name
        )
        declaration.attributes["event_ids"] = ArrayAttr(IntAttr(i) for i in candidate.ids)
        selected.append(candidate)
        used[resource].update(candidate.ids)
        # ID rotation adds no completion edges. The small-window memory
        # answers survive this ID-only update; do not use its stale snapshot.
        # Final verification independently rebuilds the actual resource proof.
        validity = {
            (check.source, check.target, check.domain): valid
            for check, valid in zip(checks, answers, strict=True)
        }
    require(all(answers), "actual query protocols leave memory hazards")
    return QueryMemoryPlan(tuple(selected), len(checks), len(missing), reuse_checks)
