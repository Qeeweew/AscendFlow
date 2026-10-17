"""Qualify three independent Attention profile batches against ACLNN.

Uses device task time, identical inputs/contracts and artifact hashes. Failure
of the default 1.00 time ratio is a failed performance target, never a warning
that can silently become success. Overlapping batch ranges remain unresolved.
"""

import argparse
import json
from pathlib import Path
from statistics import median

from ascendflow.diagnostics import require
from ascendflow.performance import profile_evidence
from benchmarks.artifacts import verified_source
from benchmarks.attention.contract import AttentionPolicy
from benchmarks.attention.run import load_backend


def artifact_identity(record, case, variant):
    artifact = record["artifact"]
    require(
        artifact["backend"] == case["backend"] == variant
        and artifact["library"] == case["library"]
        and artifact["shared_sha256"] == case["shared_sha256"],
        "artifact/record mismatch",
    )
    if variant == "aclnn":
        # Installed ACLNN has library identities, not a generated ASC manifest.
        current = load_backend(None, "aclnn", policy=AttentionPolicy(**record["policy"]))
        require(artifact == current.identity, "ACLNN libraries or adapter changed")
        require(
            case["reference_opapi_sha256"] == artifact["shared_sha256"],
            "ACLNN library/record mismatch",
        )
        return None, artifact["shared_sha256"], artifact["adapter_sha256"]
    manifest = json.loads(Path(case["library"]).with_suffix(".compile.json").read_text())
    source = verified_source(manifest)
    require(
        str(source.resolve()) == artifact["compiled_source"]
        and manifest["source_sha256"]
        == artifact["compiled_source_sha256"]
        == case["compiled_source_sha256"]
        and manifest["shared_sha256"] == case["shared_sha256"],
        "artifact/record mismatch",
    )
    return case["compiled_source_sha256"], case["shared_sha256"], None


def qualification(generated, reference, *, max_ratio=1.0):
    require(
        len(generated) >= 3 and len(reference) == len(generated), "three paired batches required"
    )
    require(1 <= max_ratio <= 2, "invalid explicitly chosen performance threshold")
    ratio = median(generated) / median(reference)
    if max(generated) <= min(reference) * max_ratio:
        status = "passed"
    elif min(generated) > max(reference) * max_ratio:
        status = "failed"
    else:
        status = "unresolved"
    return {
        "status": status,
        "generated_over_aclnn": ratio,
        "max_ratio": max_ratio,
        "generated_batch_p50_us": generated,
        "aclnn_batch_p50_us": reference,
    }


def compare(directory, *, max_ratio=1.0):
    directory = Path(directory).resolve()
    trials = {"generated": [], "aclnn": []}
    identity, source_identity = None, {}
    batches = {name: set() for name in trials}
    for path in sorted(directory.glob("trial-*/*/summary.json")):
        variant = path.parent.name
        require(variant in trials, "unexpected profile backend")
        record = json.loads((path.parent / "device/device.json").read_text())
        summary = json.loads(path.read_text())
        require(record["backend"] == variant, "profile/backend mismatch")
        require(
            record["status"] == "passed" and len(record["cases"]) == 1,
            "each profiling run requires exactly one completed numerical case",
        )
        device = record["cases"][0]
        contract = tuple(
            json.dumps(device[key], sort_keys=True)
            for key in (
                "shape_b_hq_hkv_sq_sk_d",
                "dtype",
                "softmax_scale_f32",
                "causal",
                "query_start",
                "warmup",
                "repeats",
                "oracle_scope",
                "query_rows_per_head",
                "data_case",
            )
        )
        contract += (
            tuple(device["argument_sha256"][key] for key in ("q", "k", "v")),
            record["health"]["physical_device"],
            record["hardware"]["platform_sha256"],
        )
        require(
            identity is None or identity == contract, "profile input/math/hardware contracts differ"
        )
        identity = contract
        require(
            device["inputs_unchanged"] and device["guards_checked"], "missing input/guard checks"
        )
        key = artifact_identity(record, device, variant)
        require(
            source_identity.setdefault(variant, key) == key,
            "one backend changed code across batches",
        )
        require(len(summary["operators"]) == 1, "one device kernel per launch required")
        kernel, operator = next(iter(summary["operators"].items()))
        evidence = profile_evidence(
            path.parent,
            summary,
            physical=record["health"]["physical_device"],
            warmup=device["warmup"],
            repeats=device["repeats"],
            kernel=kernel,
            task_type=operator["task_type"],
        )
        samples = sorted(summary["durations_us"])
        trials[variant].append(
            {
                "p50_us": median(samples),
                "p90_us": samples[(9 * len(samples) + 9) // 10 - 1],
                "source_sha256": key[0],
                "shared_sha256": key[1],
                "kernel": kernel,
                "blocks": operator["blocks"],
                "evidence": evidence,
                "max_abs_error": device["max_abs_error"],
                "oracle_scope": device["oracle_scope"],
            }
        )
        batches[variant].add(path.parent.parent.name)
    require(batches["generated"] == batches["aclnn"], "profile batches are not paired")
    result = qualification(
        [t["p50_us"] for t in trials["generated"]],
        [t["p50_us"] for t in trials["aclnn"]],
        max_ratio=max_ratio,
    )
    return {
        **result,
        "trials": trials,
        "timing_scope": "device task durations; host gaps excluded",
        "limits": "fixed shapes/policy; sampled numerical evidence remains sampled",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ratio", type=float, default=1.0)
    args = parser.parse_args()
    result = compare(args.profiles, max_ratio=args.max_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
