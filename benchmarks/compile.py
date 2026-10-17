"""Fresh-process CPU compilation measurements; no CANN or device execution."""

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import traceback
import uuid
from math import isfinite
from pathlib import Path
from statistics import median
from time import monotonic, perf_counter

MAX_SECONDS = 300.0


def _json_object(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _timeout(value):
    if type(value) not in {int, float} or not isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("worker timeout must be finite and in (0,300] seconds")
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _package_sha(package):
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _remaining(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("cold compilation worker deadline exceeded")
    return remaining


def _constexpr(value):
    return tuple(_constexpr(item) for item in value) if isinstance(value, list) else value


def worker(path, *, entry, parameters, mode, metadata, deadline, artifacts=None):
    """Measure all input work through immutable ASC/IR snapshot construction."""
    started = perf_counter()
    import ascendflow
    from ascendflow.dialects.io import parse

    imported = perf_counter()
    package = Path(ascendflow.__file__).parent
    identity_started = perf_counter()
    package_sha = _package_sha(package)
    identity_before_s = perf_counter() - identity_started
    source_started = perf_counter()
    source = path.read_bytes()
    read = perf_counter()
    if entry is None:
        if parameters:
            raise ValueError("constexpr parameters require an explicit AST --entry")
        module = parse(source.decode())
    else:
        import runpy

        from ascendflow.frontend import PrimFunction

        module = runpy.run_path(str(path))[entry]
        if not isinstance(module, PrimFunction):
            raise TypeError("--entry must name a captured AST prim_func")
        module = module.specialize(
            **{name: _constexpr(value) for name, value in parameters.items()}
        )
    constructed = perf_counter()
    result = ascendflow.compile(module, timeout_s=_remaining(deadline), debug_verify=mode == "deep")
    completed = perf_counter()
    # compile() already serializes every IR/report snapshot. Publication of
    # those immutable strings and identity recording happen after this boundary.
    identity_started = perf_counter()
    if path.read_bytes() != source or _package_sha(package) != package_sha:
        raise ValueError("source or compiler changed while this worker was compiling")
    metrics = {
        "status": "passed",
        "input": str(path),
        "input_kind": "serialized_ir" if entry is None else "python_ast_definition",
        "entry": entry,
        "parameters": parameters,
        "mode": mode,
        "metadata": metadata,
        "pid": os.getpid(),
        "input_sha256": _sha(source),
        "source_sha256": _sha(source),
        "package_tree_sha256": package_sha,
        "source_ir_sha256": _sha(result.source_ir.encode()),
        "before_ir_sha256": _sha(result.before_ir.encode()),
        "plan_ir_sha256": _sha(result.plan_ir.encode()),
        "asc_sha256": _sha(result.asc.encode()),
        "source_bytes": len(source),
        "asc_bytes": len(result.asc.encode()),
        "import_s": imported - started,
        "read_s": read - source_started,
        "construction_s": constructed - read,
        "source_to_asc_s": completed - source_started,
        "frontend_to_asc_s": completed - source_started,
        "compilation": result.timings,
        "verification": result.report,
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "identity_s": identity_before_s + perf_counter() - identity_started,
        "scope": "source reading/construction, AST, all analysis, ASC and immutable IR snapshots; import and artifact IO separate",
    }
    artifact_started = perf_counter()
    if artifacts is not None:
        result.write(artifacts)
        (artifacts / ("input_source" + path.suffix)).write_bytes(source)
    metrics["artifact_write_s"] = perf_counter() - artifact_started
    metrics["worker_s"] = perf_counter() - started
    metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics["rss_unit"] = "KiB"
    _remaining(deadline)
    return metrics


def _worker_main(args):
    started = monotonic()
    try:
        result = worker(
            args.input[0].resolve(),
            entry=args.entry,
            parameters=args.parameters,
            mode=args.mode,
            metadata=args.metadata,
            deadline=args.deadline,
            artifacts=args.artifacts,
        )
    except BaseException as error:  # noqa: BLE001 - worker failures must be persisted.
        result = {
            "status": "timeout"
            if isinstance(error, TimeoutError) or monotonic() >= args.deadline
            else "failed",
            "error": f"{type(error).__name__}: {error}",
            "worker_s": monotonic() - started,
            "pid": os.getpid(),
        }
        traceback.print_exc()
    _write_json(args.result, result)
    return 0 if result["status"] == "passed" else 1


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_trial(command, result_path, log_path, *, timeout_s, environment=None):
    """Bound one fresh worker group, including children surviving worker exit."""
    _timeout(timeout_s)
    started = monotonic()
    deadline = started + timeout_s
    process, timed_out, error = None, False, None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [*command, "--deadline", str(deadline)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            try:
                process.wait(timeout=max(0, deadline - monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    except BaseException as failure:  # noqa: BLE001 - every exit must stop the process group.
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if process is not None:
            _kill_group(process.pid)
            process.wait()
    wall = monotonic() - started
    try:
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if not isinstance(result, dict):
            raise TypeError("worker result must be an object")
    except (TypeError, ValueError, OSError) as failure:
        result, error = {}, f"invalid worker result: {failure}"
    required = {
        "source_sha256",
        "package_tree_sha256",
        "asc_sha256",
        "source_to_asc_s",
        "peak_rss_kib",
    }
    if result.get("status") == "passed" and not required <= result.keys():
        error = "worker result omitted required identity or timing metrics"
    code = None if process is None else process.returncode
    if timed_out or wall >= timeout_s:
        result.update(status="timeout", error=f"worker group exceeded {timeout_s}s and was stopped")
    elif error or code or result.get("status") != "passed":
        result.update(
            status="timeout" if result.get("status") == "timeout" else "failed",
            error=error
            or result.get("error")
            or f"worker exited {code} without a successful result",
        )
    result.update(process_wall_s=wall, returncode=code, timeout_s=timeout_s, log=str(log_path))
    _write_json(result_path, result)
    return result


def _summary(trials):
    return {
        "source_to_asc_median_s": median(row["source_to_asc_s"] for row in trials),
        "source_to_asc_max_s": max(row["source_to_asc_s"] for row in trials),
        "process_wall_median_s": median(row["process_wall_s"] for row in trials),
        "process_wall_max_s": max(row["process_wall_s"] for row in trials),
        "peak_rss_kib": max(row["peak_rss_kib"] for row in trials),
    }


def measure(args):
    """Persist each trial before scheduling the next; failure ends the sequence."""
    run_id = uuid.uuid4().hex
    directory = args.output.resolve() / ("run-" + run_id)
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "running",
        "measurement_kind": "probe" if args.trials == 1 else "cold_trials",
        "requested_trials": args.trials,
        "completed_trials": 0,
        "input_count": len(args.input),
        "mode": args.mode,
        "timeout_s": args.timeout,
        "entry": args.entry,
        "parameters": args.parameters,
        "metadata": args.metadata,
        "cases": [],
    }
    target = args.output.resolve() / "benchmark.json"
    _write_json(target, manifest)
    modes = ("normal", "deep") if args.mode == "both" else (args.mode,)
    for number, source in enumerate(args.input):
        for mode in modes:
            case_dir = directory / f"case-{number}-{mode}"
            case_dir.mkdir()
            case = {
                "input": str(source.resolve()),
                "mode": mode,
                "status": "running",
                "requested_trials": args.trials,
                "completed_trials": 0,
                "artifacts": str(case_dir / "artifacts"),
                "trials": [],
            }
            manifest["cases"].append(case)
            for trial in range(args.trials):
                result_path = case_dir / f"trial-{trial}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--input",
                    str(source.resolve()),
                    "--result",
                    str(result_path),
                    "--mode",
                    mode,
                    "--parameters",
                    json.dumps(args.parameters),
                    "--metadata",
                    json.dumps(args.metadata),
                ]
                if args.entry is not None:
                    command += ["--entry", args.entry]
                if trial == 0:
                    command += ["--artifacts", case["artifacts"]]
                result = _run_trial(
                    command, result_path, case_dir / f"trial-{trial}.log", timeout_s=args.timeout
                )
                result.update(
                    trial=trial,
                    mode=mode,
                    input=str(source.resolve()),
                    entry=args.entry,
                    parameters=args.parameters,
                    metadata=args.metadata,
                )
                if result["status"] == "passed" and case["trials"]:
                    prior = case["trials"][0]
                    keys = ("source_sha256", "package_tree_sha256", "asc_sha256")
                    if any(result[key] != prior[key] for key in keys):
                        result.update(
                            status="failed",
                            error="source/compiler/ASC identity changed between fresh trials",
                        )
                if (
                    result["status"] == "passed"
                    and args.mode == "both"
                    and mode == "deep"
                    and trial == 0
                ):
                    normal = manifest["cases"][-2]["trials"][0]
                    comparison = {
                        "input": str(source.resolve()),
                        "same_source": normal["source_sha256"] == result["source_sha256"],
                        "same_package": normal["package_tree_sha256"]
                        == result["package_tree_sha256"],
                        "same_asc": normal["asc_sha256"] == result["asc_sha256"],
                    }
                    manifest.setdefault("normal_deep", []).append(comparison)
                    if not all(
                        comparison[key] for key in ("same_source", "same_package", "same_asc")
                    ):
                        result.update(status="failed", error="normal/deep identity differs")
                case["trials"].append(result)
                _write_json(result_path, result)
                if result["status"] != "passed":
                    case["status"] = manifest["status"] = result["status"]
                    manifest.update(
                        stopped=True, failed_trial=trial, failed_input=str(source.resolve())
                    )
                    _write_json(target, manifest)
                    return manifest
                case["completed_trials"] += 1
                manifest["completed_trials"] += 1
                _write_json(target, manifest)
            case.update(status="passed", summary=_summary(case["trials"]))
            _write_json(target, manifest)
            stats = case["summary"]
            print(
                f"{source.name} [{mode}, {manifest['measurement_kind']}]: "
                f"source-to-ASC median {stats['source_to_asc_median_s']:.4f}s, "
                f"max {stats['source_to_asc_max_s']:.4f}s, RSS {stats['peak_rss_kib']} KiB",
                flush=True,
            )
    manifest.update(status="passed", stopped=False)
    _write_json(target, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entry", help="explicit captured AST prim_func; otherwise parse xDSL")
    parser.add_argument("--parameters", type=_json_object, default={})
    parser.add_argument("--metadata", type=_json_object, default={})
    parser.add_argument("--mode", choices=("normal", "deep", "both"), default="normal")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=MAX_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--artifacts", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        _timeout(args.timeout)
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        if (
            len(args.input) != 1
            or args.result is None
            or args.deadline is None
            or args.mode == "both"
        ):
            parser.error("worker requires one input, result, deadline and one mode")
        return _worker_main(args)
    if args.output is None:
        parser.error("--output is required")
    if args.trials != 1 and args.trials < 5:
        parser.error("use one probe trial or at least five independent trials")
    if args.parameters and args.entry is None:
        parser.error("--parameters requires an explicit --entry")
    if not all(isinstance(name, str) and name.isidentifier() for name in args.parameters):
        parser.error("constexpr parameter names must be identifiers")
    result = measure(args)
    if result["status"] != "passed":
        print(f"cold compilation stopped; see {args.output / 'benchmark.json'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
