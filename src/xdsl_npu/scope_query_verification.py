"""Compose owned query results, or independently verify imported actual IR."""

from .budget import bounded
from .diagnostics import require
from .scope_query_checks import check_memory, memory_check_reason, prepare_memory_checks
from .scope_query_completion import QueryCompletion
from .scope_query_protocols import read_query_protocols
from .scope_resources import invocation_resources


@bounded
def verify_query_invocations(checked, parent, *, resources=None, query_plan=None):
    from .scope_query_resources import _verify_query_resources

    require(
        checked.program is parent.program and checked.access is parent.access,
        "query verification needs the actual checked invocation proof",
    )
    established_here = resources is None and getattr(checked, "resource_proof", None) is None
    resources = invocation_resources(checked, parent, resources)
    protocols = read_query_protocols(checked.program, checked.access)
    if query_plan is None:
        copies = max(
            len(checked.query_summary.parts),
            max((max(p.target.query, len(p.ids)) + 1 for p in protocols), default=2),
        )
        query = QueryCompletion(parent, copies=copies)
        checks = prepare_memory_checks(checked, parent)
        answers = check_memory(checks, query)
    else:
        from .scope_query_planning import QueryMemoryPlan

        require(
            isinstance(query_plan, QueryMemoryPlan), "query verification needs an owned query plan"
        )
        query_plan._check_owned(checked, parent, resources)
        query, checks, answers = query_plan.completion, query_plan.checks, query_plan.answers
    for check, valid in zip(checks, answers, strict=True):
        require(
            valid,
            "missing cross-query memory and event lifetime synchronization: "
            f"{memory_check_reason(check, checked.access)}, "
            f"{check.source} -> {check.target}, {check.domain}",
        )
    resource_report = _verify_query_resources(checked, parent, query, resources, query_plan)
    if established_here:
        resource_report["invocation_resources"].update(
            source="rebuilt from actual IR by this entry", new_invocation_resource_solver_calls=3
        )
    # Parent progress was independently checked before this call. New local
    # waits only acquire from smaller query indices, at verified native cuts.
    # Query 0 has no such waits; assuming all earlier invocations finish, the
    # new waits of query q eventually finish too. This induction adds no
    # intra-query wait dependency and requires no runtime-M unrolling. Pairing
    # suppresses the last d Sets and first d Waits; resources were proved above.
    require(
        all(p.source.query == 0 < p.target.query for p in protocols),
        "query progress requires strictly positive inter-query dependence",
    )
    return {
        "verified": True,
        "memory_checks": len(checks),
        "memory_checks_reused": len(checks) if query_plan is not None else 0,
        "new_memory_checks": len(checks) if query_plan is None else 0,
        "protocols": len(protocols),
        "resources": resource_report,
        "completion": query.statistics(),
        "progress": "parent progress plus positive-distance local query waits at actual native cuts",
        "pairing": "Set q < M-d and Wait q >= d, matching source ordinal q and q-d",
        "runtime_query_expansion": False,
    }
