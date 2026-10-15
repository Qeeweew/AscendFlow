"""One operator, one deadline, and publication only after the whole build passes.

The Linux worker owns input construction, Python compilation, ASC and CANN.
Compiler subprocesses stay in its process group, so a hard timeout cannot leave
an orphan compiler behind. ``compile()`` itself remains an in-process,
cooperatively bounded API.
"""

import contextlib
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import tempfile
import traceback
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


def _reject_worker_nesting():
    if os.environ.get(_WORKER_ENV) == "1":
        raise ValueError("nested build/watchdog is unsupported; construct IR or ASC directly")


def _invalidate(directory):
    # Source/IR and logs may remain useful diagnostics; none authorizes a launch.
    for pattern in ("*.compile.json", "*.o", "*.aibin", "*.so"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)
    for name in ("verification.json", "compilation.json", "build-result.json"):
        (directory / name).unlink(missing_ok=True)
    for path in directory.glob(".building-*"):
        if path.is_dir():
            shutil.rmtree(path)


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        # A very short timeout can expire before the worker calls setsid().
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait(process, end, *, fork=False):
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
        _kill_group(process.pid)
        if fork:
            process.join()
        else:
            process.wait()


def _record(directory, *, status, started, timeout_s, **details):
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
    _reject_worker_nesting()
    if in_build_group():
        raise ValueError("nested command watchdog is unsupported; use the outer bounded_process")
    started, end = monotonic(), deadline(timeout_s)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _invalidate(directory)
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
            code = _wait(process, end)
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
            _invalidate(directory)
        record = _record(
            directory,
            status=status,
            started=started,
            timeout_s=timeout_s,
            returncode=code,
            command=command,
        )
    return record


def _build_one(value, directory, options, end):
    from .compilation import compile
    from .toolchain import offline_compile

    value = value() if callable(value) else value
    remaining(end, "input construction")
    if isinstance(value, str):
        name = options["name"]
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("internal ASC reference builds require an identifier name")
        report = {"kind": "asc_reference", "kernel": name}
        source = value
        (directory / f"{name}.asc").write_text(source)
    else:
        result = compile(
            value,
            synchronize=options["synchronize"],
            load3d=options["load3d"],
            timeout_s=remaining(end, "Python compilation"),
        )
        report = result.report
        report["compilation"] = result.timings
        if options["check_interleavings"]:
            from dataclasses import asdict

            from .check import bounded_check
            from .ir import MixedOp
            from .mixed_check import bounded_mixed_check

            plan = result.plan
            scope = options["check_interleavings"]
            if isinstance(plan.body.block.first_op, MixedOp):
                checked = bounded_mixed_check(
                    plan, scope=scope, max_states=options["max_check_states"]
                )
            elif scope == "all":
                checked = bounded_check(plan, max_states=options["max_check_states"])
            else:
                raise ValueError("shared interleaving scope requires a mixed kernel")
            report["bounded_interleavings"] = asdict(checked)
        result.write(directory, diagnostics=options["diagnostics"])
        name, source = result.name, result.asc
    if options["compile_asc"] or options["shared"]:
        report["cann"] = offline_compile(
            source,
            directory,
            name,
            shared=options["shared"],
            _deadline=end,
        )
    remaining(end, "completed build")
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def _worker(value, directory, options, end, log):
    if not in_build_group():
        os.setsid()
    os.environ.update({_DEADLINE_ENV: str(end), _GROUP_ENV: "1", _WORKER_ENV: "1"})
    with (
        Path(log).open("w") as output,
        contextlib.redirect_stdout(output),
        contextlib.redirect_stderr(output),
    ):
        try:
            _build_one(value, Path(directory), options, end)
        except Exception as error:  # noqa: BLE001 - record arbitrary worker failures before exit
            (Path(directory) / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "timeout"
                        if isinstance(error, TimeoutError) or monotonic() >= end
                        else "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                + "\n"
            )
            traceback.print_exc()
            raise SystemExit(1) from None


def _relocate(value, staging, output):
    if isinstance(value, dict):
        return {key: _relocate(item, staging, output) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, staging, output) for item in value]
    if isinstance(value, str) and value.startswith(str(staging) + os.sep):
        return str(output / Path(value).relative_to(staging))
    return value


def build_module(
    module,
    output,
    *,
    synchronize=True,
    compile_asc=False,
    shared=False,
    diagnostics=False,
    check_interleavings=None,
    max_check_states=200_000,
    load3d=False,
    timeout_s=MAX_BUILD_SECONDS,
    name=None,
):
    """Internal Linux build tool; a callable constructs its input inside the worker.

    Already constructed inputs are accepted, but work before this call cannot
    belong to its deadline. Public file/AST builds should use ``build_file``.
    Each output directory belongs to one operator build.

    Builders cannot start another build/watchdog. When an outer command
    watchdog launches the CLI, its single file build inherits that command's
    deadline; an independent shorter deadline in the shared group is rejected.
    """
    _reject_worker_nesting()
    started, end = monotonic(), deadline(timeout_s)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _invalidate(output)
    options = {
        "synchronize": synchronize,
        "compile_asc": compile_asc,
        "shared": shared,
        "diagnostics": diagnostics,
        "check_interleavings": check_interleavings,
        "max_check_states": max_check_states,
        "load3d": load3d,
        "name": name,
    }
    status, code, error = "failed", None, None
    try:
        with tempfile.TemporaryDirectory(prefix=".building-", dir=output) as temporary:
            staging = Path(temporary)
            process = multiprocessing.get_context("fork").Process(
                target=_worker,
                args=(module, str(staging), options, end, str(output / "build.log")),
            )
            process.start()
            code = _wait(process, end, fork=True)
            if code:
                failure = staging / "failure.json"
                details = json.loads(failure.read_text()) if failure.exists() else {}
                if details.get("status") == "timeout":
                    raise TimeoutError(details["error"])
                raise RuntimeError(
                    f"operator compilation failed: {details.get('error', code)}; see {output / 'build.log'}"
                )
            report = _relocate(
                json.loads((staging / "verification.json").read_text()), staging, output
            )
            for path in staging.glob("*.compile.json"):
                manifest = _relocate(json.loads(path.read_text()), staging, output)
                path.write_text(json.dumps(manifest, indent=2) + "\n")
            (staging / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
            # Publish success manifests last; every path in them now names the
            # final ASC/object/library, never a removed staging directory.
            for path in sorted(staging.iterdir(), key=lambda p: p.name.endswith(".compile.json")):
                remaining(end, "artifact publication")
                path.replace(output / path.name)
            remaining(end, "completion")
            status = "passed"
        return report
    except TimeoutError as failure:
        status, error = "timeout", str(failure)
        raise TimeoutError(
            f"operator compilation reached {timeout_s}s; stopped, see {output}"
        ) from None
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        if code is None and "process" in locals():
            code = process.exitcode
        if status != "passed":
            _invalidate(output)
        _record(
            output,
            status=status,
            started=started,
            timeout_s=timeout_s,
            returncode=code,
            error=error,
        )


def build_file(path, output, *, entry=None, parameters=None, **options):
    """Read IR or capture a current AST PrimFunction inside the bounded worker."""
    _reject_worker_nesting()
    path = Path(path).resolve()
    parameters = {} if parameters is None else dict(parameters)
    if parameters and entry is None:
        raise ValueError("AST specialization parameters require --entry")

    def construct():
        if entry is None:
            from .ir import parse

            return parse(path.read_text())
        import runpy

        from .frontend import PrimFunction

        definition = runpy.run_path(str(path))[entry]
        if not isinstance(definition, PrimFunction):
            raise TypeError("--entry must name a captured AST prim_func")
        return definition.specialize(**parameters)

    return build_module(construct, output, **options)
