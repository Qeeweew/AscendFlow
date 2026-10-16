"""Fixed-policy Vector recurrence, optionally repeated in a retained parent loop."""

import argparse
import json
from pathlib import Path
from statistics import median

from benchmarks.artifacts import verified_source
from xdsl_npu.diagnostics import require
from xdsl_npu.performance import profile_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-cwd", type=Path, help="Explicit base for legacy relative compile paths"
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
    work = case["requested_work"]
    gm = sum(
        n for key, n in work["memory_requested_bytes_by_pipe_bank_mode"].items() if ":GM:" in key
    )
    ub = sum(
        n for key, n in work["memory_requested_bytes_by_pipe_bank_mode"].items() if ":UB:" in key
    )
    ops = work["vector_requested_elements"]["add"]
    outputs = case.get("outputs", 1)
    adds_per_call = case["tiles"] + int(device["policy"].get("frame_snapshot", False))
    expected = device["policy"]["GROUPS"] * outputs * adds_per_call * device["policy"]["B"]
    require(ops == expected, "physical add work disagrees with the ordered recurrence")
    result = {
        "policy": device["policy"],
        "outputs": outputs,
        "tiles": case["tiles"],
        "hardware": device["hardware"],
        "legacy_build_cwd": None if args.build_cwd is None else str(args.build_cwd.resolve()),
        "trials": trials,
        "median_of_run_medians_us": times,
        "generated_over_reference": times["generated"] / times["ascendc"],
        "work": work,
        "theory_inputs": {
            "fp32_add_operations": ops,
            "gm_requested_bytes": gm,
            "ub_operand_requested_bytes": ub,
            "ops_per_gm_request_byte": ops / gm,
            "gm_requested_GBps": gm / (times["generated"] * 1000),
            "declared_working_set_over_l2": work["declared_gm_working_set_bytes"]
            / device["hardware"]["l2_bytes"],
            "sustained_hbm_GBps": None,
            "sustained_vector_add_ops_s": None,
            "lower_bound": "max(GM/HBM actual bytes / measured ceiling, UB port demand / ceiling, FP32 adds / ceiling, dependency critical path)",
            "limits": "GM request bytes are not HBM transactions; active ratios overlap. No numeric hardware roofline without measured bandwidth and Vector clock/rate.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("median_of_run_medians_us", "generated_over_reference", "theory_inputs")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
