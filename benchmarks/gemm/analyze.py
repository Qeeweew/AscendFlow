"""Bind native GEMM profile trials to source, numeric policy, grid and hardware."""

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
        "--reduction", action="store_true", help="runtime K panels and one terminal output"
    )
    parser.add_argument("--hierarchical", action="store_true", help="runtime outputs and K panels")
    parser.add_argument("--build-cwd", type=Path, help="Base for recorded relative compile paths")
    args = parser.parse_args()
    reduction = args.reduction or args.hierarchical
    trials = {"generated": [], "ascendc": []}
    identity = None
    for summary_path in sorted(args.profiles.glob("trial-*/*/summary.json")):
        variant = summary_path.parent.name
        require(variant in trials, "unexpected variant")
        summary = json.loads(summary_path.read_text())
        device = json.loads((summary_path.parent / "device.json").read_text())
        require(len(device["cases"]) == 1, "one case per profiling process required")
        case = device["cases"][0]
        work = read_work(case, required_counts=("fma_flops",))
        require(
            work["structure_inventory"]["groups"] == device["policy"]["GROUPS"],
            "generated source grid differs from the benchmark policy",
        )
        manifest = device["variants"][variant]
        command = manifest["commands"][0]
        source = verified_source(manifest, args.build_cwd)
        contract = (
            device["policy"],
            case["effective_gemm_shape"] if reduction else case["input_a_shape"],
            device["scope"] if reduction else case["input_b_shape"],
            case["input_sha256"],
            case["output_sha256"],
            case["warmup"],
            case["repeats"],
            case["cache_policy"],
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
            "profile grid differs",
        )
        evidence = profile_evidence(
            summary_path.parent,
            summary,
            physical=device["health"]["physical_device"],
            warmup=case["warmup"],
            repeats=case["repeats"],
            kernel=case["kernel"],
            task_type="AI_CORE",
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
        all(len(v) >= 3 for v in trials.values()),
        "at least three independent runs per variant required",
    )
    medians = {v: median(t["p50_us"] for t in ts) for v, ts in trials.items()}
    policy = device["policy"]
    count_key = "panels" if reduction else "tiles_per_group"
    outputs = case["outputs"] if args.hierarchical else 1
    products = case[count_key] * policy["GROUPS"] * outputs
    products *= 1 if reduction else 2 if policy["ACCUMULATE"] else 1
    flops = 2 * products * policy["M"] * policy["N"] * policy["BK" if reduction else "K"]
    require(
        work["algorithm_work"]["counts"]["fma_flops"] == flops,
        "algorithm FMA convention differs from the benchmark problem",
    )
    result = {
        "policy": policy,
        count_key: case[count_key],
        **({"outputs": outputs} if args.hierarchical else {}),
        "kind": "nested_panel_reduction"
        if args.hierarchical
        else "panel_reduction"
        if reduction
        else "native_batch",
        "hardware": device["hardware"],
        "trials": trials,
        "median_of_run_medians_us": medians,
        "generated_over_reference": medians["generated"] / medians["ascendc"],
        "work": work,
        "algorithm_rates": {
            "fma_tflops": flops / medians["generated"] / 1e6,
            "scope": "problem-size FLOPs per measured time; not native Cube throughput",
        },
        "runtime_request_analysis": work["runtime_requests"],
        "native_roofline": {
            "status": "unavailable",
            "reason": "runtime native Cube work and GM requests have not been established",
        },
        "reference_setup": "CANN9.1 TPipe seeds/drains M_MTE1 IDs0/1/2; explicit L0 ring uses3/4. Generated ASC has no TPipe",
        "limits": "fixed policy only; algorithm FLOPs are not a native instruction count; runtime traffic is unavailable; pipe ratios overlap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "median_us": medians,
                "ratio": result["generated_over_reference"],
                "algorithm_rates": result["algorithm_rates"],
                "runtime_request_analysis": result["runtime_request_analysis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
