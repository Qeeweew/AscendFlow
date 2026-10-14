"""Generic cold-process CPU compilation benchmark; no CANN or device timing."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter


def worker(path, output):
    started = perf_counter()
    from xdsl_npu import compile
    from xdsl_npu.ir import parse

    imported = perf_counter()
    source = path.read_text()
    module = parse(source)
    parsed = perf_counter()
    result = compile(module)
    timings = result.timings
    reuse = []
    for _ in range(100):
        start = perf_counter()
        _ = result.asc, result.report, result.schedule
        reuse.append(perf_counter() - start)
    if output:
        result.write(output)
    package = Path(sys.modules["xdsl_npu"].__file__).parent
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(str(file.relative_to(package)).encode())
        digest.update(file.read_bytes())
    return {
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "package_tree_sha256": digest.hexdigest(),
        "asc_sha256": hashlib.sha256(result.asc.encode()).hexdigest(),
        "import_s": imported - started,
        "parse_s": parsed - imported,
        "compilation": timings,
        "reuse_median_s": median(reuse),
        "reuse_max_s": max(reuse),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verification": result.report,
        "python": sys.version,
        "machine": platform.machine(),
        "scope": "fresh Python process; full finite verification; no cProfile, CANN, or NPU timing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.input[0], args.output)))
        return
    if args.trials < 3:
        parser.error("at least three independent trials required")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for number, source in enumerate(args.input):
        trials = []
        for trial in range(args.trials):
            start = perf_counter()
            raw = subprocess.check_output(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--input",
                    str(source.resolve()),
                    "--output",
                    str((args.output / f"case-{number}-trial-{trial}").resolve()),
                ],
                text=True,
            )
            result = json.loads(raw)
            result["process_wall_s"] = perf_counter() - start
            trials.append(result)
        if len({(trial["asc_sha256"], trial["package_tree_sha256"]) for trial in trials}) != 1:
            raise ValueError("code or compiler changed between benchmark trials")
        record = {
            "input": str(source.resolve()),
            "trials": trials,
            "total_median_s": median(trial["compilation"]["total_s"] for trial in trials),
            "total_max_s": max(trial["compilation"]["total_s"] for trial in trials),
            "reuse_max_s": max(trial["reuse_max_s"] for trial in trials),
            "peak_rss_kib": max(trial["peak_rss_kib"] for trial in trials),
        }
        cases.append(record)
        (args.output / "benchmark.json").write_text(json.dumps(cases, indent=2) + "\n")
        print(
            f"{source.parent.name}: median {record['total_median_s']:.4f}s, "
            f"max {record['total_max_s']:.4f}s, RSS {record['peak_rss_kib']} KiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
