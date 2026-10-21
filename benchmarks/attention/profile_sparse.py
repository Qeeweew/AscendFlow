"""Collect same-input sparse MLA decode measurements with msprof."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from ascendflow.diagnostics import require
from ascendflow.runtime.process import kill_process_group

ROOT = Path(__file__).resolve().parents[2]


def summarize(directory, warmup, repeats, backend):
    paths = list(directory.rglob("op_summary*.csv"))
    require(len(paths) == 1, "expected exactly one msprof operator summary")
    rows = list(csv.DictReader(paths[0].open()))
    expected = {"sparse_mla_decode"} if backend == "ascendflow" else {"SparseFlashMla"}
    groups = {}
    setup = []
    first_main = min(
        float(row["Task Start Time(us)"].strip()) for row in rows if row["OP Type"] in expected
    )
    for row in rows:
        name = row["OP Type"]
        if name not in expected:
            require(
                float(row["Task Start Time(us)"].strip()) < first_main,
                f"unexpected task during measured calls: {name}",
            )
            setup.append(
                {key: row[key] for key in ("Op Name", "OP Type", "Task Type", "Task Duration(us)")}
            )
            continue
        require(row["Task Type"] in {"AI_VECTOR_CORE", "MIX_AIC"}, "unexpected task type")
        groups.setdefault(name, []).append(row)
    require(groups, "no device tasks collected")
    samples, operators = [], {}
    for name, launches in groups.items():
        require(len(launches) == warmup + repeats, f"{name}: launch count differs")
        measured = launches[warmup:]
        times = [float(row["Task Duration(us)"]) for row in measured]
        samples.append(times)
        operators[name] = {
            "task_type": measured[0]["Task Type"],
            "block_num": measured[0]["Block Num"],
            "median_us": statistics.median(times),
            "min_us": min(times),
            "max_us": max(times),
            "samples_us": times,
            "metrics_median": {
                key: statistics.median(float(row[key]) for row in measured)
                for key in measured[0]
                if key.startswith(("aic_", "aiv_", "l2_", "hbm_", "ddr_"))
                and all(row[key] not in {"N/A", ""} for row in measured)
            },
        }
    totals = [sum(values) for values in zip(*samples, strict=True)]
    return {
        "scope": "sum of device task durations per call; excludes setup and host gaps",
        "setup_tasks_excluded": setup,
        "summary_csv": str(paths[0].resolve()),
        "summary_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "operators": operators,
        "median_us": statistics.median(totals),
        "samples_us": totals,
    }


def profile(args, backend, batch, context, trial):
    output = args.output / f"b{batch}-s{context}-t{trial}" / backend
    require(not output.exists(), f"measurement output already exists: {output}")
    output.mkdir(parents=True)
    artifact = (
        args.artifacts
        if (args.artifacts / "policy.json").exists()
        else args.artifacts / f"b{batch}"
    )
    policy = json.loads((artifact / "policy.json").read_text())
    worker = [
        str(ROOT / "benchmarks/profile.sh"),
        "benchmarks.attention.sparse_performance",
        "--backend",
        backend,
        "--physical",
        str(args.physical),
        "--batch",
        str(batch),
        "--query-tokens",
        str(policy["T"] // policy["B"]),
        "--groups",
        str(policy["GROUPS"]),
        "--context",
        str(context),
        "--max-seq",
        str(policy["MAX_SEQ"]),
        "--ratio",
        str(policy["RATIO"]),
        "--heads",
        str(policy["H"]),
        "--features",
        str(policy["D"]),
        "--topk",
        str(policy["TOPK"]),
        "--window",
        str(policy["WINDOW"]),
        "--head-tile",
        str(policy["BM"]),
        "--kv-parts",
        str(policy["PARTS"]),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--rotations",
        str(args.rotations),
        "--output",
        str(output / "device"),
    ]
    if backend == "ascendflow":
        worker += ["--artifacts", str(artifact)]
    else:
        worker += ["--reference-artifacts", str(args.reference_artifacts)]
    command = [
        "msprof",
        "--output=" + str(output / "profile"),
        "--aic-metrics=" + args.metrics,
        "--application=" + shlex.join(worker),
    ]
    env = os.environ | {
        "ASCENDFLOW_PYTHON": sys.executable,
        "ASCEND_RT_VISIBLE_DEVICES": str(args.physical),
        "ASCEND_DEVICE_ID": "0",
        "OPENBLAS_NUM_THREADS": "1",
    }
    (output / "command.json").write_text(
        json.dumps({"command": command, "worker": worker}, indent=2)
    )
    with (output / "msprof.log").open("w") as log:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = proc.wait(timeout=240)
        finally:
            kill_process_group(proc.pid)
        require(code == 0, f"msprof failed: {output / 'msprof.log'}")
    device = json.loads((output / "device/device.json").read_text())
    require(device["status"] == "passed", "device validation failed")
    result = summarize(output / "profile", args.warmup, args.repeats, backend)
    result.update(
        batch=batch,
        context=context,
        query_tokens=policy["T"] // policy["B"],
        total_tokens=policy["T"],
        trial=trial,
        backend=backend,
        device_record=str(output / "device/device.json"),
        input_identity=device["input_identity"],
        max_abs_error=device["max_abs_error"],
    )
    (output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("ascendflow", "vllm-ascend"),
        default=["ascendflow", "vllm-ascend"],
    )
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--contexts", nargs="+", type=int, default=[16384, 65536, 131072])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--metrics", default="PipeUtilization")
    parser.add_argument("--rotations", type=int, default=1)
    args = parser.parse_args(argv)
    require(args.trials > 0 and args.repeats > 0 and args.warmup >= 0, "invalid repetition counts")
    require(
        "vllm-ascend" not in args.backends or args.reference_artifacts is not None,
        "source reference artifacts required",
    )
    args.output = args.output.resolve()
    args.artifacts = args.artifacts.resolve()
    if args.reference_artifacts is not None:
        args.reference_artifacts = args.reference_artifacts.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []
    for batch in args.batches:
        for context in args.contexts:
            for trial in range(args.trials):
                pair = []
                # Alternate launch order to limit systematic thermal/frequency bias.
                backends = args.backends if trial % 2 == 0 else list(reversed(args.backends))
                for backend in backends:
                    row = profile(args, backend, batch, context, trial)
                    pair.append(row)
                    measurements.append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in ("backend", "batch", "context", "trial", "median_us")
                            }
                        ),
                        flush=True,
                    )
                    (args.output / "measurements.json").write_text(
                        json.dumps(measurements, indent=2) + "\n"
                    )
                require(
                    all(row["input_identity"] == pair[0]["input_identity"] for row in pair),
                    "backend input bytes differ",
                )
    return measurements


if __name__ == "__main__":
    main()
