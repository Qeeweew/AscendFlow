"""Summarize a single msprof run; retain task timings and provenance outside Git."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--repeats", type=int, default=30)
args = parser.parse_args()
files = list(args.directory.rglob("op_summary*.csv"))
if len(files) != 1 or args.warmup < 0 or args.repeats < 1:
    raise ValueError("require one op_summary CSV and valid fixed launch counts")
source = files[0]
groups = defaultdict(list)
for row in csv.DictReader(source.open()):
    if row["Task Type"] not in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}:
        raise ValueError(f"unaccounted task type: {row['Task Type']}")
    groups[row["Op Name"]].append(row)
total = args.warmup + args.repeats
durations = [0.0] * args.repeats
operators = {}
for name, rows in groups.items():
    rows.sort(key=lambda row: float(row["Task Start Time(us)"]))
    if len(rows) % total:
        raise ValueError(f"{name}: {len(rows)} tasks do not match {total} fixed launches")
    multiplicity = len(rows) // total
    samples = [
        sum(float(row["Task Duration(us)"]) for row in rows[i : i + multiplicity])
        for i in range(0, len(rows), multiplicity)
    ][args.warmup :]
    durations = [a + b for a, b in zip(durations, samples)]
    operators[name] = {
        "tasks_per_launch": multiplicity,
        "task_type": rows[0]["Task Type"],
        "blocks": sorted({int(row["Block Num"]) for row in rows}),
        "median_us": median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }
if not operators:
    raise ValueError("no device tasks found")
record = {
    "csv": str(source.resolve()),
    "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "warmup": args.warmup,
    "repeats": args.repeats,
    "timing_scope": "sum of device task durations per launch; excludes host/stream gaps",
    "sampling": f"{args.repeats} consecutive launches in one profiling run; not independent trials",
    "operators": operators,
    "median_us": median(durations),
    "min_us": min(durations),
    "max_us": max(durations),
    "durations_us": durations,
}
(args.directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
