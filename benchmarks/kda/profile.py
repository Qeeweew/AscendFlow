"""Collect msprof op counters for a published KDA kernel and preserve provenance."""

import argparse
import csv
import json
import math
import os
import shlex
import sys
from pathlib import Path
from statistics import mean, median

from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.kda.run import digest

COLUMNS = {
    "OpBasicInfo": ("Task Duration(us)", "Block Dim", "Current Freq"),
    "PipeUtilization": (
        "aiv_vec_ratio",
        "aiv_scalar_ratio",
        "aiv_mte2_ratio",
        "aiv_mte3_ratio",
        "aiv_scalar_wait_time(us)",
        "aiv_scalar_single_time(us)",
        "aiv_scalar_dual_time(us)",
    ),
    "Memory": (
        "read_main_memory_datas(KB)",
        "write_main_memory_datas(KB)",
        "aiv_gm_to_ub_bw(GB/s)",
        "aiv_ub_to_gm_bw(GB/s)",
    ),
    "ArithmeticUtilization": ("aiv_vec_fops",),
    "L2Cache": ("aiv_read_hit_rate(%)", "aiv_write_hit_rate(%)"),
    "ResourceConflictRatio": (
        "aiv_vec_total_cflt_ratio",
        "aiv_vec_wait_ratio",
        "aiv_mte3_wait_ratio",
    ),
}


def summarize(directory, backend="ascendflow", metrics="Default"):
    record = json.loads((directory / "device/results.json").read_text())
    require(record["status"] == "passed", "profiling runner failed numerical validation")
    kernel = (
        record["operator"]
        if backend == "ascendflow"
        else "fused_recurrent_gated_delta_rule_fwd_kernel"
    )
    require(set(record["results"]) == {backend}, "profile runner backend differs")
    result = {}
    for name, columns in COLUMNS.items():
        if metrics == "BasicInfo" and name != "OpBasicInfo":
            continue
        files = list((directory / "raw").rglob(name + ".csv"))
        require(len(files) == 1, f"expected one completed {name} CSV, found {len(files)}")
        path = files[0]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        require(rows, f"empty {name} CSV")
        if name == "OpBasicInfo":
            require(
                len(rows) == 1
                and rows[0]["Op Name"] == kernel
                and int(rows[0]["Device Id"]) == record["health"]["physical_device"],
                "profile kernel or physical device differs from the validated launch",
            )
        table = {"csv": str(path.resolve()), "csv_sha256": digest(path), "rows": len(rows)}
        for column in columns:
            values = [float(row[column]) for row in rows]
            require(all(math.isfinite(value) for value in values), f"nonfinite {column}")
            table[column] = {"mean": mean(values), "median": median(values), "sum": sum(values)}
        result[name] = table
    result["provenance"] = {
        "backend": backend,
        "aic_metrics": metrics,
        "device_record": str((directory / "device/results.json").resolve()),
        "device_record_sha256": digest(directory / "device/results.json"),
        "counter_scope": "means across logical blocks; pipe ratios are not whole-device utilization",
        "timing_scope": "one msprof op Task Duration; excludes host submission gaps",
        "artifact": record["generated"],
        "reference": record["reference"],
        "triton_cache": {
            str(path.relative_to(directory)): digest(path)
            for path in (directory / "device/triton-cache").rglob("*")
            if path.is_file()
        },
    }
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", required=True, type=int)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--backend", choices=("ascendflow", "triton"), default="ascendflow")
    parser.add_argument("--metrics", choices=("BasicInfo", "Default"), default="BasicInfo")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    if not args.summarize_only:
        require(not directory.exists(), "KDA profiling requires a new output directory")
        artifact = args.artifact.resolve()
        kernel = json.loads((artifact / "verification.json").read_text())["kernel"]
        if args.backend == "triton":
            kernel = "fused_recurrent_gated_delta_rule_fwd_kernel"
        root = Path(__file__).resolve().parents[2]
        application = [
            str(root / "benchmarks/profile.sh"),
            "benchmarks.kda.run",
            "--artifact",
            str(artifact),
            "--output",
            str(directory / "device"),
            "--physical",
            str(args.physical),
            "--backend",
            args.backend,
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--worker",
        ]
        if args.lengths is not None:
            application.extend(("--lengths", *(str(length) for length in args.lengths)))
        os.environ["ASCENDFLOW_PYTHON"] = sys.executable
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        command = [
            "msprof",
            "op",
            "--application=" + shlex.join(application),
            "--output=" + str(directory / "raw"),
            "--kernel-name=" + kernel,
            "--aic-metrics=" + args.metrics,
            "--launch-count=1",
            "--warm-up=0",
        ]
        bounded_process(command, directory / "watchdog", timeout_s=300)
    result = summarize(directory, args.backend, args.metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
