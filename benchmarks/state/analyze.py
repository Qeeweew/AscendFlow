"""Fixed-policy Vector recurrence, optionally repeated in a retained parent loop."""

import argparse
import json
from pathlib import Path
from statistics import median

from benchmarks.artifacts import verified_source
from benchmarks.work import read_work
from xdsl_npu.diagnostics import require
from xdsl_npu.performance import profile_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-cwd", type=Path, help="Explicit base for recorded relative compile paths"
    )
    args = parser.parse_args()
    trials = {"generated": [], "ascendc": []}
    identity = None
    for path in sorted(args.profiles.glob("trial-*/*/summary.json")):
        variant = path.parent.name
        require(variant in trials, "unknown profile variant")
        summary = json.loads(path.read_text())
        device = json.loads((path.parent / "device.json").read_text())
        require(len(device["cases"]) == 1, "one fixed case per profile required")
        case = device["cases"][0]
        work = read_work(case, required_counts=("fp32_add_operations",))
        require(
            work["structure_inventory"]["groups"] == device["policy"]["GROUPS"],
            "generated source grid differs from the benchmark policy",
        )
        require(not case["rounding_case"], "rounding counterexample is not the performance case")
        manifest = device["variants"][variant]
        command = manifest["commands"][0]
        source = verified_source(manifest, args.build_cwd)
        contract = (
            device["policy"],
            case.get("outputs", 1),
            case["tiles"],
            case["input_sha256"],
            case["output_sha256"],
            case["cache_policy"],
            case["warmup"],
            case["repeats"],
            device["health"]["physical_device"],
            device["hardware"]["platform_sha256"],
            manifest["compiler_version"],
            command[1 : command.index("--cce-aicore-only")],
            work,
        )
        require(
            identity is None or identity == contract, "profile contracts/hardware/output differ"
        )
        identity = contract
        require(
            summary["operators"][case["kernel"]]["blocks"] == [device["policy"]["GROUPS"]],
            "profile block count differs",
        )
        evidence = profile_evidence(
            path.parent,
            summary,
            physical=device["health"]["physical_device"],
            warmup=case["warmup"],
            repeats=case["repeats"],
            kernel=case["kernel"],
            task_type="AI_VECTOR_CORE",
        )
        samples = sorted(summary["durations_us"])
        trials[variant].append(
            {
                "p50_us": median(samples),
                "p90_us": samples[(9 * len(samples) + 9) // 10 - 1],
                "min_us": min(samples),
                "max_us": max(samples),
                "compiler": manifest,
                "source_path": str(source),
                "evidence": evidence,
            }
        )
    require(
        all(len(v) >= 3 for v in trials.values()), "three independent runs per variant required"
    )
    times = {variant: median(run["p50_us"] for run in values) for variant, values in trials.items()}
    ops = work["algorithm_work"]["counts"]["fp32_add_operations"]
    outputs = case.get("outputs", 1)
    adds_per_call = case["tiles"] + int(device["policy"].get("frame_snapshot", False))
    expected = device["policy"]["GROUPS"] * outputs * adds_per_call * device["policy"]["B"]
    require(ops == expected, "algorithm add count disagrees with the ordered recurrence")
    result = {
        "policy": device["policy"],
        "outputs": outputs,
        "tiles": case["tiles"],
        "hardware": device["hardware"],
        "build_cwd": str(args.build_cwd.resolve())
        if args.build_cwd is not None
        else "not supplied",
        "trials": trials,
        "median_of_run_medians_us": times,
        "generated_over_reference": times["generated"] / times["ascendc"],
        "work": work,
        "algorithm_rates": {
            "fp32_add_operations": ops,
            "fp32_add_ops_s": ops / (times["generated"] * 1e-6),
            "scope": "ordered problem additions per measured time; not Vector instruction throughput",
        },
        "runtime_request_analysis": work["runtime_requests"],
        "hardware_roofline": {
            "status": "unavailable",
            "reason": "runtime GM/UB requests and sustained memory/Vector ceilings are not established",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "median_of_run_medians_us",
                    "generated_over_reference",
                    "algorithm_rates",
                    "runtime_request_analysis",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
