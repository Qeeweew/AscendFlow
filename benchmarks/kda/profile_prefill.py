"""Collect msprof timing or PMU counters for a checked prefill invocation."""

import argparse
import csv
import json
import math
import os
import shlex
import sys
from pathlib import Path
from statistics import mean

from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.kda.prefill import GRAM_PHASES, KERNELS
from benchmarks.kda.run import digest


def counter_tables(path):
    tables = {}
    for csv_path in sorted(path.parent.glob("*.csv")):
        if csv_path.name.startswith("OpBasicInfo"):
            continue
        with csv_path.open() as stream:
            samples = list(csv.DictReader(stream))
        columns = {}
        if samples:
            for column in samples[0]:
                if not column or column in ("block_id", "sub_block_id"):
                    continue
                values = []
                for sample in samples:
                    try:
                        value = float(sample[column])
                    except (TypeError, ValueError):
                        continue
                    require(math.isfinite(value), f"nonfinite {column} in {csv_path}")
                    values.append(value)
                if values:
                    columns[column] = {
                        "samples": len(values),
                        "mean": mean(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                    }
        tables[csv_path.stem] = {
            "csv": str(csv_path.resolve()),
            "csv_sha256": digest(csv_path),
            "rows": len(samples),
            "columns": columns,
        }
    return tables


def summarize(directory, metrics="BasicInfo"):
    capture = json.loads((directory / "watchdog/build-result.json").read_text())
    require(capture["status"] == "passed", "profiling command did not complete")
    options = dict(
        argument.removeprefix("--").split("=", 1)
        for argument in capture["command"]
        if argument.startswith("--") and "=" in argument
    )
    require(options.get("replay-mode", "kernel") == "kernel", "expected kernel replay")
    require(options.get("aic-metrics") == metrics, "recorded PMU group differs")
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
                        "block_dim": row["Block Dim"],
                        "frequency_mhz": row["Current Freq"],
                        "counters": counter_tables(path) if metrics != "BasicInfo" else {},
                    }
                )
    require(rows, "no completed kernel profiles")
    collected = rows
    if set(record["results"]) == {"ascendflow"}:
        rows = [row for row in rows if row["name"].startswith("prefill_")]
        config = record["configuration"]
        slabs = (
            len(config["lengths"]) * config["chunks_per_sequence"] * config["heads"]
            + config.get("gram_slab_tasks", 1)
            - 1
        ) // config.get("gram_slab_tasks", 1)
        require(
            sorted(row["name"] for row in rows)
            == sorted(
                KERNELS.get(p, "prefill_" + p)
                for p in config["phases"]
                for _ in range(slabs if p in GRAM_PHASES else 1)
            ),
            "prefill phase launch multiplicities differ",
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
    if metrics != "BasicInfo":
        expected = (
            ("PipeUtilization", "ArithmeticUtilization") if metrics == "Default" else (metrics,)
        )
        for row in rows:
            require(
                all(
                    any(
                        name.startswith(prefix) and table["rows"]
                        for name, table in row["counters"].items()
                    )
                    for prefix in expected
                ),
                f"missing {metrics} counters for {row['name']}",
            )
    result = {
        "status": "passed",
        "aic_metrics": metrics,
        "replay_mode": "kernel",
        "scope": "sum of prefill kernel Task Duration; excludes host gaps and validation kernels",
        "counter_scope": "per-block samples; pipe ratios are fractions of each core's active interval, not whole-device utilization",
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
    cube_times = [
        table["columns"]["aic_cube_time(us)"]
        for row in rows
        for name, table in row["counters"].items()
        if name.startswith("PipeUtilization") and "aic_cube_time(us)" in table["columns"]
    ]
    if cube_times:
        result["cube_active_core_us"] = sum(item["sum"] for item in cube_times)
        result["device_cube_busy_fraction"] = result["cube_active_core_us"] / (
            record["hardware"]["aic_cores"] * result["kernel_us"]
        )
        result["device_cube_scope"] = (
            "sum of per-block Cube active time / (physical AIC cores * summed kernel durations); excludes host gaps"
        )
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", required=True, type=int)
    parser.add_argument("--backend", choices=("ascendflow", "triton"), default="ascendflow")
    parser.add_argument(
        "--metrics",
        choices=(
            "BasicInfo",
            "PipeUtilization",
            "ArithmeticUtilization",
            "L2Cache",
            "Memory",
            "Default",
        ),
        default="BasicInfo",
    )
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
            "--aic-metrics=" + args.metrics,
            "--launch-count=100",
            "--warm-up=0",
        ]
        bounded_process(command, directory / "watchdog", timeout_s=300)
    result = summarize(directory, args.metrics)
    print(json.dumps({k: result[k] for k in ("status", "aic_metrics", "kernel_us")}, indent=2))


if __name__ == "__main__":
    main()
