"""Shared build deadlines, process-group lifetime, and failure records.

This layer owns the watchdog contract and stays independent of compiler IR,
build orchestration, and the CANN backend.
"""

import json
import os
import shutil
import signal
import subprocess
from math import isfinite
from pathlib import Path
from time import monotonic

MAX_BUILD_SECONDS = 300.0
_DEADLINE_ENV = "XDSL_NPU_BUILD_DEADLINE"
_GROUP_ENV = "XDSL_NPU_BUILD_GROUP"
_WORKER_ENV = "XDSL_NPU_BUILD_WORKER"


def deadline(timeout_s=MAX_BUILD_SECONDS, *, inherited=None):
    if (
        type(timeout_s) not in {int, float}
        or not isfinite(timeout_s)
        or not 0 < timeout_s <= MAX_BUILD_SECONDS
    ):
        raise ValueError("per-operator build timeout must be finite and in (0,300] seconds")
    end = monotonic() + timeout_s
    parent = os.environ.get(_DEADLINE_ENV)
    if parent is not None:
        end = min(end, float(parent))
    if inherited is not None:
        end = min(end, inherited)
    if in_build_group() and (parent is None or end != float(parent)):
        raise ValueError(
            "a shared-group build must inherit the outer deadline; "
            "set timeout_s on the outer bounded_process"
        )
    return end


def remaining(end, phase="build"):
    seconds = end - monotonic()
    if seconds <= 0:
        raise TimeoutError(f"operator compilation deadline exceeded during {phase}")
    return seconds


def in_build_group():
    return os.environ.get(_GROUP_ENV) == "1"


def reject_worker_nesting():
    if os.environ.get(_WORKER_ENV) == "1":
        raise ValueError("nested build/watchdog is unsupported; construct IR or ASC directly")


def enter_build_worker(end):
    """Join the inherited watchdog group or establish the worker's own group."""
    if not in_build_group():
        os.setsid()
    os.environ.update({_DEADLINE_ENV: str(end), _GROUP_ENV: "1", _WORKER_ENV: "1"})


def invalidate_build_artifacts(directory):
    # Source/IR and logs may remain useful diagnostics; none authorizes a launch.
    for pattern in ("*.compile.json", "*.o", "*.aibin", "*.so"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)
    for name in ("verification.json", "compilation.json", "build-result.json"):
        (directory / name).unlink(missing_ok=True)
    for path in directory.glob(".building-*"):
        if path.is_dir():
            shutil.rmtree(path)


def kill_process_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        # A very short timeout can expire before the worker calls setsid().
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_process(process, end, *, fork=False):
    try:
        if fork:
            process.join(timeout=remaining(end, "worker"))
            if process.is_alive():
                raise TimeoutError("operator compilation deadline exceeded during worker")
            return process.exitcode
        return process.wait(timeout=remaining(end, "worker"))
    except subprocess.TimeoutExpired:
        raise TimeoutError("operator compilation deadline exceeded during worker") from None
    finally:
        # Also reap descendants after normal/error worker exit. They cannot be
        # allowed to outlive a completed build, even if their parent exited.
        kill_process_group(process.pid)
        if fork:
            process.join()
        else:
            process.wait()


def record_build_result(directory, *, status, started, timeout_s, **details):
    result = {
        "status": status,
        "seconds": monotonic() - started,
        "deadline_s": timeout_s,
        **details,
    }
    (directory / "build-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def bounded_process(command, directory, *, timeout_s=MAX_BUILD_SECONDS):
    """Hard deadline for an isolated build command and all its compiler children."""
    reject_worker_nesting()
    if in_build_group():
        raise ValueError("nested command watchdog is unsupported; use the outer bounded_process")
    started, end = monotonic(), deadline(timeout_s)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    invalidate_build_artifacts(directory)
    environment = os.environ | {_DEADLINE_ENV: str(end), _GROUP_ENV: "1"}
    status, code = "failed", None
    try:
        with (directory / "build.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            code = wait_for_process(process, end)
        if code:
            raise RuntimeError(f"operator compilation failed; see {directory / 'build.log'}")
        remaining(end, "completion")
        status = "passed"
    except TimeoutError:
        status = "timeout"
        raise TimeoutError(
            f"operator compilation reached {timeout_s}s; stopped, see {directory}"
        ) from None
    finally:
        if code is None and "process" in locals():
            code = process.returncode
        if status != "passed":
            invalidate_build_artifacts(directory)
        record = record_build_result(
            directory,
            status=status,
            started=started,
            timeout_s=timeout_s,
            returncode=code,
            command=command,
        )
    return record
