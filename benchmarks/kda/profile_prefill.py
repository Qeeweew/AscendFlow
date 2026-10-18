"""Collect BasicInfo for one numerically checked prefill invocation."""

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path

from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.kda.prefill import KERNELS, PHASES
from benchmarks.kda.run import digest


def summarize(directory):
    device_path = directory / "device/results.json"
    record = json.loads(device_path.read_text())
    require(record["status"] == "passed", "prefill numerical validation failed")
    rows = []
    for path in sorted((directory / "raw").rglob("OpBasicInfo*.csv")):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                require(
                    int(row["Device Id"]) == record["health"]["physical_device"], "wrong device"
                )
                duration = float(row["Task Duration(us)"])
                require(duration > 0, "invalid task duration")
                rows.append(
                    {
                        "name": row["Op Name"],
                        "kernel_us": duration,
                        "csv": str(path.resolve()),
                        "csv_sha256": digest(path),
                    }
                )
    require(rows, "no completed kernel profiles")
    collected = rows
    if set(record["results"]) == {"ascendflow"}:
        rows = [row for row in rows if row["name"].startswith("prefill_")]
        require(
            sorted(row["name"] for row in rows) == sorted(KERNELS[p] for p in PHASES),
            "expected exactly one collection of each prefill phase",
        )
    else:
        require(set(record["results"]) == {"triton"}, "expected one backend")
        # Inputs and metadata are constructed on the CPU. These two device
        # helpers belong to input reset and numerical validation, respectively.
        # Zero initialization and the original gate scaling remain timed.
        rows = [row for row in rows if not row["name"].startswith(("Cast_", "TensorMove_"))]
        require(
            any(row["name"] == "chunk_gla_fwd_kernel_o" for row in rows),
            "original output kernel was not collected",
        )
    result = {
        "status": "passed",
        "aic_metrics": "BasicInfo",
        "scope": "sum of prefill kernel Task Duration; excludes host gaps and validation kernels",
        "kernel_us": sum(row["kernel_us"] for row in rows),
        "kernels": rows,
        "all_collections": collected,
        "device_record": str(device_path.resolve()),
        "device_record_sha256": digest(device_path),
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
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", required=True, type=int)
    parser.add_argument("--backend", choices=("ascendflow", "triton"), default="ascendflow")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    if not args.summarize_only:
        require(not directory.exists(), "prefill profiling requires a new output directory")
        root = Path(__file__).resolve().parents[2]
        os.environ.update(
            ASCENDFLOW_PYTHON=sys.executable,
            ASCEND_RT_VISIBLE_DEVICES=str(args.physical),
            ASCEND_DEVICE_ID="0",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            TRITON_CACHE_DIR=str(directory / "device/triton-cache"),
        )
        application = [
            str(root / "benchmarks/profile.sh"),
            "benchmarks.kda.prefill",
            "--artifacts",
            str(args.artifacts.resolve()),
            "--output",
            str(directory / "device"),
            "--physical",
            str(args.physical),
            "--backend",
            args.backend,
            "--warmup",
            "0",
            "--repeats",
            "0",
            "--worker",
        ]
        command = [
            "msprof",
            "op",
            "--application=" + shlex.join(application),
            "--output=" + str(directory / "raw"),
            "--aic-metrics=BasicInfo",
            "--launch-count=100",
            "--warm-up=0",
        ]
        bounded_process(command, directory / "watchdog", timeout_s=300)
    print(json.dumps(summarize(directory), indent=2))


if __name__ == "__main__":
    main()
