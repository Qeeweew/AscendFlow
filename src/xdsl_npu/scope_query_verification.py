"""Final serial-query verification rebuilt from the actual synchronized IR."""

from .budget import bounded
from .model import require
from .scope_query_checks import check_memory, prepare_memory_checks
from .scope_query_completion import QueryCompletion
from .scope_query_protocols import read_query_protocols


@bounded
def verify_query_invocations(checked, parent):
    from .scope_query_resources import verify_query_resources

    require(
        checked.program is parent.program and checked.access is parent.access,
        "query verification needs the actual checked invocation proof",
    )
    protocols = read_query_protocols(checked.program, checked.access)
    copies = max(
        len(checked.query_summary.parts),
        max((max(p.target.query, len(p.ids)) + 1 for p in protocols), default=2),
    )
    query = QueryCompletion(parent, copies=copies)
    checks = prepare_memory_checks(checked, parent)
    answers = check_memory(checks, query)
    for check, valid in zip(checks, answers, strict=True):
        obligation = check.obligations[0]
        roots = [
            name for name, root in checked.access.origins.items() if root is obligation.storage
        ]
        require(
            valid,
            "missing cross-query memory and event lifetime synchronization: "
            f"{obligation.hazard} {'/'.join(roots)}[{obligation.start}:{obligation.stop}], "
            f"{check.source} -> {check.target}, {check.domain}",
        )
    resources = verify_query_resources(checked, parent, query)
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
        "protocols": len(protocols),
        "resources": resources,
        "completion": query.statistics(),
        "progress": "parent progress plus positive-distance local query waits at actual native cuts",
        "pairing": "Set q < M-d and Wait q >= d, matching source ordinal q and q-d",
        "runtime_query_expansion": False,
    }
