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
import tempfile
import traceback
from pathlib import Path
from time import monotonic

from .runtime import process as build_process


def _build_one(value, directory, options, end):
    from .backend.cann import offline_compile
    from .compilation import compile

    value = value() if callable(value) else value
    build_process.remaining(end, "input construction")
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
            timeout_s=build_process.remaining(end, "Python compilation"),
        )
        report = result.report
        report["compilation"] = result.timings
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
    build_process.remaining(end, "completed build")
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def _worker(value, directory, options, end, log):
    build_process.enter_build_worker(end)
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
    compile_asc=False,
    shared=False,
    diagnostics=False,
    timeout_s=build_process.MAX_BUILD_SECONDS,
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
    build_process.reject_worker_nesting()
    started, end = monotonic(), build_process.deadline(timeout_s)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_process.invalidate_build_artifacts(output)
    options = {
        "compile_asc": compile_asc,
        "shared": shared,
        "diagnostics": diagnostics,
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
            code = build_process.wait_for_process(process, end, fork=True)
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
                build_process.remaining(end, "artifact publication")
                path.replace(output / path.name)
            build_process.remaining(end, "completion")
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
            build_process.invalidate_build_artifacts(output)
        build_process.record_build_result(
            output,
            status=status,
            started=started,
            timeout_s=timeout_s,
            returncode=code,
            error=error,
        )


def build_file(path, output, *, entry=None, parameters=None, device_resources=None, **options):
    """Read IR or capture a current AST PrimFunction inside the bounded worker."""
    build_process.reject_worker_nesting()
    path = Path(path).resolve()
    parameters = {} if parameters is None else dict(parameters)
    if parameters and entry is None:
        raise ValueError("AST specialization parameters require --entry")
    if device_resources is not None and entry is None:
        raise ValueError("device resource binding requires a DSL --entry")

    def construct():
        if entry is None:
            from .dialects.io import parse

            return parse(path.read_text())
        import runpy
        import sys

        from .frontend import PrimFunction

        # Match `python -m`: source modules may import helpers from the
        # invocation directory even when the console script lives in bin/.
        search_path = sys.path[:]
        sys.path.insert(0, str(Path.cwd()))
        try:
            definition = runpy.run_path(str(path))[entry]
        finally:
            sys.path[:] = search_path
        if not isinstance(definition, PrimFunction):
            raise TypeError("--entry must name a captured AST prim_func")
        definition = definition.specialize(**parameters)
        return definition if device_resources is None else definition.bind_device(device_resources)

    return build_module(construct, output, **options)
