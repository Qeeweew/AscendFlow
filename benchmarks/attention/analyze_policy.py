"""Bind fixed-policy reference profiling to source, work and conditional hardware bounds."""

import argparse
import hashlib
import json
from pathlib import Path

from ascendflow.diagnostics import require
from ascendflow.performance import profile_evidence, theoretical_metrics
from benchmarks.references.attention import NAME, reference_source


def policy_work(policy, sq, sk):
    require(
        sq > 0 and sq % 128 == 0 and sk > 0 and sk % 1024 == 0,
        "performance accounting requires positive complete tiles",
    )
    b, h, kh, groups = (policy[k] for k in ("batch", "heads", "kv_heads", "groups"))
    queries, kv = b * h * (sq // 128), sk // 1024
    tasks = queries * kv
    requests = {
        "Q_read": queries * 128 * 128 * 2,
        "K_read": tasks * 1024 * 128 * 2,
        "V_read": tasks * 1024 * 128 * 2,
        "Score_write": tasks * 128 * 1024 * 4,
        "Score_read": tasks * 128 * 1024 * 4,
        "P_write": tasks * 128 * 1024 * 2,
        "P_read": tasks * 128 * 1024 * 2,
        "PV_write": tasks * 128 * 128 * 4,
        "PV_read": tasks * 128 * 128 * 4,
        "O_write": queries * 128 * 128 * 2,
    }
    io = b * 128 * 2 * (2 * h * sq + 2 * kh * sk)
    workspace = groups * 2 * (128 * 1024 * 6 + 128 * 128 * 4)
    return {
        "native_cube_flops": 4 * b * h * sq * sk * 128,
        "useful_cube_flops": 4 * b * h * sq * sk * 128,
        "native_mmad_invocations": tasks * 16,
        "task_count": tasks,
        "query_count": queries,
        "active_aic_cores": min(groups, queries),
        "active_aiv_cores": 2 * min(groups, queries),
        "gm_instruction_request_bytes": sum(requests.values()),
        "gm_requests_by_role": requests,
        "unique_gm_io_allocation_bytes": io,
        "gm_workspace_bytes": workspace,
        "gm_allocation_bytes": io + workspace,
        "vector_probability_exp_elements": tasks * 128 * 1024,
        "vector_alpha_exp_elements": tasks * 128,
        "vector_r64_repeats_per_max_or_sum": tasks * 128 * 1024 // 64,
        "vector_merge_repeats_per_max_or_sum": tasks * 128,
        "vector_output_rescale_and_add_elements_each": tasks * 128 * 128,
        "local_allocation_bytes": {
            "L1": 224 * 1024,
            "L0A": 64 * 1024,
            "L0B": 64 * 1024,
            "L0C": 128 * 1024,
            "UB_per_AIV": 178304,
        },
        "scope": "fixed source algorithm/native block accounting; not measured memory transactions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = json.loads(args.run.read_text())
    require(len(device["cases"]) == 1, "one fixed case per profile required")
    case = device["cases"][0]
    summary = json.loads((args.profile / "summary.json").read_text())
    source = reference_source(**case["policy"])
    manifest = device["compilation"]
    library = args.run.parent / (NAME + ".so")
    require(
        hashlib.sha256(source.encode()).hexdigest() == manifest["source_sha256"]
        and hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
        "reference source/library no longer matches measured run",
    )
    evidence = profile_evidence(
        args.profile,
        summary,
        physical=case["health"]["physical_device"],
        warmup=case["warmup"],
        repeats=case["repeats"],
        kernel=NAME,
    )
    work = policy_work(case["policy"], case["sq"], case["sk"])
    metrics = theoretical_metrics(
        work, {**case["hardware"], "aic_cores": work["active_aic_cores"]}, summary["median_us"]
    )
    result = {
        "policy": case["policy"],
        "sq": case["sq"],
        "sk": case["sk"],
        "source_sha256": manifest["source_sha256"],
        "shared_sha256": manifest["shared_sha256"],
        "work": work,
        "hardware": case["hardware"],
        "profile": evidence,
        "metrics": metrics,
        "numerical_scope": case["oracle_scope"],
        "limits": "one profiling batch, warm repeated buffers; not a generated-kernel or ACLNN comparison; no measured HBM/Vector roof",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "activity": evidence["pipe_activity_medians"]}, indent=2))


if __name__ == "__main__":
    main()
