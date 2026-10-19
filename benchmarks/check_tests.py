"""Run the CPU test suite in parallel file workers.

Each ordinary file owns a 300-second process-group deadline. Watchdog tests
and the CLI helper exercise that contract themselves and run without another
build-group wrapper.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from xml.etree import ElementTree

from ascendflow.runtime.process import MAX_BUILD_SECONDS, bounded_process

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = "tests/runtime/test_compile_watchdog.py"
CLI_HELPER = (
    "tests/frontend/test_frontend_helpers.py::"
    "test_cli_source_imports_local_dsl_helpers_without_pythonpath"
)


def run_file(path, output):
    directory = output / str(path).removesuffix(".py").replace("/", "__")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--junitxml=" + str(directory / "results.xml"),
        str(path),
        "--deselect=" + CLI_HELPER,
    ]
    started = monotonic()
    try:
        if str(path) == WATCHDOG:
            run_isolated(command, directory)
        else:
            bounded_process(command, directory)
        result = {"file": str(path), "status": "passed"}
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result = {"file": str(path), "status": "failed", "error": str(error)}
    result["seconds"] = monotonic() - started
    print(json.dumps(result), flush=True)
    return result


def run_isolated(command, directory):
    directory.mkdir(parents=True)
    with (directory / "build.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=MAX_BUILD_SECONDS,
        )
    if result.returncode:
        raise RuntimeError(f"isolated test failed; see {directory / 'build.log'}")


def summarize(output, results, jobs, started):
    cases, errors, skipped = set(), [], []
    for path in output.glob("*/results.xml"):
        for case in ElementTree.parse(path).getroot().iter("testcase"):
            key = case.get("classname", "") + "::" + case.get("name", "")
            if key in cases:
                raise RuntimeError(f"duplicate test result: {key}")
            cases.add(key)
            if case.find("failure") is not None or case.find("error") is not None:
                errors.append(key)
            if case.find("skipped") is not None:
                skipped.append(key)
    return {
        "status": "passed"
        if all(r["status"] == "passed" for r in results) and not errors
        else "failed",
        "workers": jobs,
        "wall_seconds": monotonic() - started,
        "unique_cases": len(cases),
        "failures_errors": errors,
        "skipped": skipped,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--paths", nargs="+", type=Path, default=[Path("tests")])
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    os.chdir(ROOT)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    paths = set()
    for selected in args.paths:
        if selected.is_dir():
            paths.update(selected.rglob("test_*.py"))
        elif selected.is_file() and selected.name.startswith("test_"):
            paths.add(selected)
        else:
            parser.error(f"test path does not exist: {selected}")
    if not paths:
        parser.error("no test files selected")
    try:
        paths = sorted({path.resolve().relative_to(ROOT) for path in paths})
    except ValueError:
        parser.error("test paths must belong to this repository")
    output.mkdir(parents=True, exist_ok=False)
    (output / "files.json").write_text(json.dumps([str(p) for p in paths], indent=2) + "\n")
    started = monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda p: run_file(p, output), paths))
    if Path(CLI_HELPER.split("::", 1)[0]) in paths:
        directory = output / "standalone-helper"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--junitxml=" + str(directory / "results.xml"),
            CLI_HELPER,
        ]
        try:
            run_isolated(command, directory)
            results.append({"file": CLI_HELPER, "status": "passed"})
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            results.append({"file": CLI_HELPER, "status": "failed", "error": str(error)})
    summary = summarize(output, results, args.jobs, started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return int(summary["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
