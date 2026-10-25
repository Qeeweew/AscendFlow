"""Collect msprof op PMU data for one validated grouped GEMM shape."""

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from statistics import mean

from ascendflow.backend.cann import CANN
from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.moe.run import ROOT, save, sha

TABLES = (
    "OpBasicInfo",
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
    "Occupancy",
    "Roofline",
)


def summarize(directory):
    path = directory / "device/device.json"
    record = json.loads(path.read_text())
    require(record["status"] == "passed", "PMU application failed numerical validation")
    command = json.loads((directory / "command.json").read_text())
    log = (directory / "watchdog/build.log").read_text()
    if "[ERROR]" in log:
        save(
            directory / "metrics.json",
            {
                "status": "failed",
                "command": command,
                "device_numerical_status": record["status"],
                "error": "msprof op reported a collection error; see watchdog/build.log",
            },
        )
        require(False, "msprof op collection failed; see watchdog/build.log")
    tables = {}
    for name in TABLES:
        files = sorted(
            p
            for p in (directory / "raw").rglob("*.csv")
            if p.stem == name or p.stem.startswith(name + "_")
        )
        if not files:
            continue
        rows = [r for p in files for r in csv.DictReader(p.open())]
        require(rows, f"empty PMU table: {name}")
        if name == "OpBasicInfo":
            accepted = {command["kernel_name"]}
            if command["backend"] == "original":
                accepted.add(command["kernel_name"] + "_mix_aic")
            require(
                len(rows) == 1 and rows[0]["Op Name"] in accepted,
                "PMU kernel differs from requested backend",
            )
            require(
                {int(r["Device Id"]) for r in rows} == {record["health"]["physical_device"]},
                "PMU physical device differs from validated application",
            )
        columns = {}
        for column in rows[0]:
            try:
                values = [float(r[column]) for r in rows]
            except (ValueError, TypeError):
                continue
            columns[column] = {"mean": mean(values), "min": min(values), "max": max(values)}
        tables[name] = {
            "rows": len(rows),
            "columns": columns,
            "files": {str(p.resolve()): sha(p) for p in files},
        }
    require("OpBasicInfo" in tables, "msprof op did not publish basic counters")
    if command.get("metrics", "Default") != "TimelineDetail":
        require("PipeUtilization" in tables, "msprof op did not publish pipe counters")
    result = {
        "status": "passed",
        "command": command,
        "tables": tables,
        "device_record": str(path.resolve()),
        "device_record_sha256": sha(path),
        "policy": record["policy"],
        "manifest": record["manifest"],
        "input_sha256": record["cases"][0]["input_sha256"],
        "weight_sha256": record["weight_sha256"],
        "health": record["health"],
        "scope": "PMU rows are per logical block; replay groups are not additive timing phases",
    }
    if command["backend"] == "original":
        binary = (
            CANN
            / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/grouped_matmul"
            / (command["kernel_name"].rsplit("_", 1)[0] + ".o")
        )
        metadata = binary.with_suffix(".json")
        require(binary.is_file() and metadata.is_file(), "captured original GMM binary is missing")
        require(
            any(
                k["kernelName"] == command["kernel_name"]
                for k in json.loads(metadata.read_text())["kernelList"]
            ),
            "captured original tiling key does not belong to the recorded binary",
        )
        result["captured_binary"] = {str(p): sha(p) for p in (binary, metadata)}
    else:
        result["captured_binary"] = {
            "library_sha256": record["manifest"]["shared_sha256"],
            "asc_sha256": record["manifest"]["source_sha256"],
        }
    save(directory / "metrics.json", result)
    print(json.dumps({"status": result["status"], "tables": list(tables)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical", type=int, default=0)
    parser.add_argument("--backend", choices=("generated", "original"), default="generated")
    parser.add_argument("--kernel-name", default="grouped_gemm")
    parser.add_argument("--rows", type=int, default=2048, help="rows per expert")
    parser.add_argument(
        "--metrics", choices=("Default", "PipeUtilization", "TimelineDetail"), default="Default"
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    require(
        args.backend != "original" or args.kernel_name != "grouped_gemm",
        "original PMU capture requires its exact kernel name",
    )
    directory = args.output.resolve()
    if not args.summarize_only:
        require(not directory.exists(), "PMU output directory already exists")
        directory.mkdir(parents=True)
        os.environ.update(
            ASCENDFLOW_PYTHON=sys.executable,
            ASCEND_RT_VISIBLE_DEVICES=str(args.physical),
            ASCEND_DEVICE_ID="0",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        application = [
            str(ROOT / "benchmarks/profile.sh"),
            "benchmarks.moe.run",
            "--artifact",
            str(args.artifact.resolve()),
            "--output",
            str(directory / "device"),
            "--physical",
            str(args.physical),
            "--rows",
            str(args.rows),
            "--data",
            "normal",
            "--warmup",
            "0",
            "--worker",
        ]
        command = [
            "msprof",
            "op",
            "--application=" + shlex.join(application),
            "--output=" + str(directory / "raw"),
            "--aic-metrics=" + args.metrics,
            "--kernel-name=" + args.kernel_name,
            "--launch-count=1",
            "--warm-up=5",
        ]
        save(
            directory / "command.json",
            {
                "backend": args.backend,
                "kernel_name": args.kernel_name,
                "metrics": args.metrics,
                "argv": command,
            },
        )
        bounded_process(command, directory / "watchdog", timeout_s=300)
    summarize(directory)


if __name__ == "__main__":
    main()
