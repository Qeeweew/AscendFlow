"""Bind fixed normalization profiles to code, footprint and hardware evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from ascendflow.diagnostics import require
from ascendflow.performance import profile_evidence
from benchmarks.artifacts import verified_source
from benchmarks.work import read_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-cwd", type=Path, help="Base for recorded relative compile paths")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["generated", "ascendc", "aclnn"],
        default=["generated", "ascendc"],
    )
    args = parser.parse_args()
    require(
        {"generated", "ascendc"} <= set(args.variants), "comparison needs generated and ascendc"
    )
    trials = {variant: [] for variant in args.variants}
    variant_identity = {}
    identity = None
    compiler_identity = None
    first_device = None
    for summary_path in sorted(args.profiles.glob("trial-*/*/summary.json")):
        variant = summary_path.parent.name
        require(variant in trials, "unexpected profiling variant")
        summary = json.loads(summary_path.read_text())
        device = json.loads((summary_path.parent / "device.json").read_text())
        require(len(device["cases"]) == 1, "one fixed case per profiling process required")
        case = device["cases"][0]
        work = read_work(
            case,
            required_counts=("add_mul_operations", "logical_input_bytes", "logical_output_bytes"),
        )
        groups = work["structure_inventory"]["groups"]
        manifest = device["variants"][variant]
        binary_identity = (
            manifest.get("source_sha256"),
            manifest.get("shared_sha256"),
            json.dumps(manifest.get("libraries"), sort_keys=True),
            case["output_sha256"],
        )
        require(
            variant_identity.setdefault(variant, binary_identity) == binary_identity,
            "one variant changed binary/library/output across trials",
        )
        source = None
        if variant != "aclnn":
            command = manifest["commands"][0]
            source = verified_source(manifest, args.build_cwd)
            compiled = (
                manifest["compiler_version"],
                tuple(command[1 : command.index("--cce-aicore-only")]),
            )
            require(
                compiler_identity is None or compiled == compiler_identity,
                "CANN flags/version differ",
            )
            compiler_identity = compiled
        else:
            for name, sha in manifest["libraries"].items():
                path = Path(manifest["cann"]) / "lib64" / name
                require(
                    hashlib.sha256(path.read_bytes()).hexdigest() == sha, "ACLNN library changed"
                )
        run_identity = (
            case["shape"],
            case["input_sha256"],
            case["warmup"],
            case["repeats"],
            case["cache_policy"],
            device["health"]["physical_device"],
            device["hardware"]["platform_sha256"],
            case["rows_per_group"],
            groups,
            case.get("operator", "rms"),
            case["dtype"],
            work,
        )
        require(
            len(summary["operators"]) == 1, "this comparison requires one device kernel per launch"
        )
        kernel, operator = next(iter(summary["operators"].items()))
        require(
            variant == "aclnn" or kernel == case["kernel"], "profile kernel differs from source"
        )
        require(
            variant == "aclnn" or operator["blocks"] == [groups],
            "profile block count differs from the fixed launch grid",
        )
        require(
            identity is None or identity == run_identity,
            "profiling contracts/hardware/input differ",
        )
        identity = run_identity
        first_device = first_device or device
        evidence = profile_evidence(
            summary_path.parent,
            summary,
            physical=device["health"]["physical_device"],
            warmup=case["warmup"],
            repeats=case["repeats"],
            kernel=kernel,
            task_type=operator["task_type"],
        )
        samples = sorted(summary["durations_us"])
        trials[variant].append(
            {
                "p50_us": median(samples),
                "p90_us": samples[(9 * len(samples) + 9) // 10 - 1],
                "min_us": min(samples),
                "max_us": max(samples),
                "source_sha256": manifest.get("source_sha256"),
                "shared_sha256": manifest.get("shared_sha256"),
                "compiler": manifest,
                "source_path": str(source) if source is not None else None,
                "kernel": kernel,
                "blocks": operator["blocks"],
                "output_sha256": case["output_sha256"],
                "max_abs_error": case["max_abs_error"],
                "evidence": evidence,
            }
        )
    require(
        all(len(values) >= 3 for values in trials.values()),
        "at least three complete runs per variant required",
    )
    medians = {
        variant: median(run["p50_us"] for run in values) for variant, values in trials.items()
    }
    rows, cols = identity[0]
    require(rows > 0 and cols > 0, "performance comparison requires nonempty shapes")
    is_softmax = case.get("operator") == "softmax"
    counts = work["algorithm_work"]["counts"]
    require(
        counts["add_mul_operations"] == rows * (2 * cols - 1 if is_softmax else 3 * cols + 1),
        "algorithm operation convention differs from the normalization problem",
    )
    result = {
        "shape": identity[0],
        "dtype": case["dtype"],
        "hardware": device["hardware"],
        "cache_policy": case["cache_policy"],
        "trials": trials,
        "median_of_run_medians_us": medians,
        "generated_over_reference": medians["generated"] / medians["ascendc"],
        "work": work,
        "algorithm_rates": {
            "add_mul_ops_s": counts["add_mul_operations"] / (medians["generated"] * 1e-6),
            "elements_s": rows * cols / (medians["generated"] * 1e-6),
            "scope": "scalar problem convention per measured time; not Vector recipe throughput",
        },
        "problem_io_bytes_over_l2": (counts["logical_input_bytes"] + counts["logical_output_bytes"])
        / first_device["hardware"]["l2_bytes"],
        "runtime_request_analysis": work["runtime_requests"],
        "hardware_roofline": {
            "status": "unavailable",
            "reason": "runtime GM/UB requests, HBM traffic, and sustained Vector/memory ceilings are not established",
        },
    }
    result["generated_over_variant"] = {
        name: medians["generated"] / t for name, t in medians.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "median_us": medians,
                "generated_over_reference": result["generated_over_reference"],
                "algorithm_rates": result["algorithm_rates"],
                "runtime_request_analysis": result["runtime_request_analysis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
