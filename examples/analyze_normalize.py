"""Bind fixed normalization profiles to code, footprint and hardware evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from xdsl_npu.model import require
from xdsl_npu.performance import profile_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = {"generated": [], "ascendc": []}
    identity = None
    for summary_path in sorted(args.profiles.glob("trial-*/*/summary.json")):
        variant = summary_path.parent.name
        require(variant in trials, "unexpected profiling variant")
        summary = json.loads(summary_path.read_text())
        device = json.loads((summary_path.parent / "device.json").read_text())
        require(len(device["cases"]) == 1, "one fixed case per profiling process required")
        case = device["cases"][0]
        manifest = device["variants"][variant]
        command = manifest["commands"][0]
        source = Path(command[command.index("-o") - 1])
        library = source.with_suffix(".so")
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
            and hashlib.sha256(library.read_bytes()).hexdigest() == manifest["shared_sha256"],
            "profiling sources/libraries changed",
        )
        run_identity = (
            case["shape"],
            case["input_sha256"],
            case["output_sha256"],
            case["warmup"],
            case["repeats"],
            case["cache_policy"],
            device["health"]["physical_device"],
            device["hardware"]["platform_sha256"],
            case["rows_per_group"],
            case["requested_work"]["groups"],
            manifest["compiler_version"],
            tuple(command[1 : command.index("--cce-aicore-only")]),
        )
        require(
            summary["operators"][case["kernel"]]["blocks"] == [case["requested_work"]["groups"]],
            "profile block count differs from the fixed launch grid",
        )
        require(
            identity is None or identity == run_identity,
            "profiling contracts/hardware/output differ",
        )
        identity = run_identity
        evidence = profile_evidence(
            summary_path.parent,
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
                "source_sha256": manifest["source_sha256"],
                "shared_sha256": manifest["shared_sha256"],
                "compiler": manifest,
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
    work = case["requested_work"]
    gm = sum(
        value
        for key, value in work["memory_requested_bytes_by_pipe_bank_mode"].items()
        if ":GM:" in key
    )
    ub = sum(
        value
        for key, value in work["memory_requested_bytes_by_pipe_bank_mode"].items()
        if ":UB:" in key
    )
    result = {
        "shape": identity[0],
        "dtype": case["dtype"],
        "hardware": device["hardware"],
        "cache_policy": case["cache_policy"],
        "trials": trials,
        "median_of_run_medians_us": medians,
        "generated_over_reference": medians["generated"] / medians["ascendc"],
        "work_from_generated_ir": work,
        "theory_inputs": {
            "gm_requested_bytes": gm,
            "ub_operand_requested_bytes": ub,
            "add_mul_operations": rows * (3 * cols + 1),
            "sqrt_elements": rows,
            "div_elements": rows,
            "cube_flops": 0,
            "add_mul_per_gm_request_byte": rows * (3 * cols + 1) / gm,
            "gm_effective_GBps": gm / (medians["generated"] * 1000),
            "ub_requested_effective_GBps": ub / (medians["generated"] * 1000),
            "hbm_bytes": None,
            "sustained_hbm_GBps": None,
            "sustained_ub_GBps": None,
            "lower_bound": "max(resource work / its sustained ceiling, dependency critical path); shared Vector work needs a mixed-instruction rate",
            "limits": "GM requests can hit L2; operand UB bytes can share physical reads. AIV clock/rates and HBM traffic are unmeasured, so no numeric hardware roofline is claimed",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "median_us": medians,
                "generated_over_reference": result["generated_over_reference"],
                "theory_inputs": result["theory_inputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
