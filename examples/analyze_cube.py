"""Bind native GEMM profile trials to source, numeric policy, grid and hardware."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from xdsl_npu.model import require
from xdsl_npu.performance import profile_evidence, theoretical_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reduction", action="store_true", help="runtime K panels and one terminal output"
    )
    parser.add_argument("--hierarchical", action="store_true", help="runtime outputs and K panels")
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
        manifest = device["variants"][variant]
        command = manifest["commands"][0]
        source = Path(command[command.index("-o") - 1])
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
            and hashlib.sha256(source.with_suffix(".so").read_bytes()).hexdigest()
            == manifest["shared_sha256"],
            "profile source/library changed",
        )
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
                "evidence": evidence,
            }
        )
    require(
        all(len(v) >= 3 for v in trials.values()),
        "at least three independent runs per variant required",
    )
    medians = {v: median(t["p50_us"] for t in ts) for v, ts in trials.items()}
    policy, requested = device["policy"], case["requested_work"]
    count_key = "panels" if reduction else "tiles_per_group"
    outputs = case["outputs"] if args.hierarchical else 1
    products = case[count_key] * policy["GROUPS"] * outputs
    mads = products if reduction else products * (2 if policy["ACCUMULATE"] else 1)
    require(
        requested["native_invocations"]["mad_first" if reduction else "mad"] == mads,
        "physical MAD count differs from policy",
    )
    if reduction:
        require(
            requested["native_invocations"]["fix"]
            == policy["GROUPS"] * outputs * int(case["panels"] > 0),
            "terminal output count differs",
        )
    flops = 2 * mads * policy["M"] * policy["N"] * policy["BK" if reduction else "K"]
    gm = sum(
        value
        for key, value in requested["memory_requested_bytes_by_pipe_bank_mode"].items()
        if ":GM:" in key
    )
    work = {
        "native_cube_flops": flops,
        "useful_cube_flops": flops,
        "gm_instruction_request_bytes": gm,
        "unique_gm_io_allocation_bytes": requested["declared_gm_working_set_bytes"],
        "gm_allocation_bytes": requested["declared_gm_working_set_bytes"],
    }
    theory = theoretical_metrics(work, device["hardware"], medians["generated"])
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
        "requested_work": requested,
        "theory": theory,
        "reference_setup": "CANN9.1 TPipe seeds/drains M_MTE1 IDs0/1/2; explicit L0 ring uses3/4. Generated ASC has no TPipe",
        "limits": "fixed policy only; no full GEMM/FA performance claim; GM requests are not HBM traffic; pipe ratios overlap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"median_us": medians, "ratio": result["generated_over_reference"], "theory": theory},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
