"""Direct CANN 9.1 ASC compilation. No TileLang/TVM/MLIR import or build."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import monotonic

from ..runtime.process import (
    MAX_BUILD_SECONDS,
    deadline,
    in_build_group,
    kill_process_group,
    remaining,
)

CANN = Path("/usr/local/Ascend/cann-9.1.0")
FLAGS = [
    "-O2",
    "-fPIC",
    "-std=c++20",
    "-mllvm",
    "-cce-aicore-dcpreload-args=false",
    "--npu-arch=dav-2201",
]


def offline_compile(
    source: str,
    directory: Path,
    name: str,
    *,
    shared=False,
    timeout_s=MAX_BUILD_SECONDS,
    _deadline=None,
) -> dict:
    """Compile within one wall-clock budget, killing the compiler process group on expiry."""
    started = monotonic()
    end = deadline(timeout_s, inherited=_deadline)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not name.isidentifier():
        raise ValueError("CANN build name must be an identifier")
    artifacts = [
        directory / f"{name}{suffix}" for suffix in (".o", ".aibin", ".so", ".compile.json")
    ]
    for path in artifacts:
        path.unlink(missing_ok=True)
    timeout_record = directory / f"{name}.timeout.json"
    timeout_record.unlink(missing_ok=True)
    src = directory / f"{name}.asc"
    src.write_text(source)
    binary = CANN / "tools/bisheng_compiler/bin/bisheng"
    linker = binary.with_name("ld.lld")
    obj, output = directory / f"{name}.o", directory / f"{name}.aibin"
    commands = [
        [str(binary), *FLAGS, "--cce-aicore-only", str(src), "-o", str(obj)],
        [
            str(linker),
            "-m",
            "aicorelinux",
            "-Ttext",
            "0",
            "--no-mmap-output-file",
            str(obj),
            "-o",
            str(output),
        ],
    ]
    if shared:
        commands.append(
            [
                str(binary),
                *FLAGS,
                "-shared",
                "-Wl,--no-mmap-output-file",
                str(src),
                "-o",
                str(directory / f"{name}.so"),
            ]
        )
    logs, timings = [], []
    own_group = not in_build_group()

    def stop(proc):
        try:
            if own_group:
                kill_process_group(proc.pid)
            else:
                # The outer watchdog owns the whole worker group and always
                # reaps it, including compiler descendants, when work ends.
                proc.kill()
        except ProcessLookupError:
            pass

    def run(command):
        remaining(end, "CANN command launch")
        command_started = monotonic()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=own_group,
        )
        try:
            stdout, _ = proc.communicate(timeout=remaining(end, "CANN command"))
        except subprocess.TimeoutExpired:
            stop(proc)
            stdout, _ = proc.communicate()
            logs.append(stdout)
            raise TimeoutError(
                f"CANN compilation exceeded the shared deadline; process group killed: {command}"
            ) from None
        except BaseException:
            stop(proc)
            proc.communicate()
            raise
        finally:
            # A compiler may exit successfully or with an error while a
            # background child has redirected its output and is still alive.
            # Always retire the process group this command actually owns.
            if own_group:
                stop(proc)
        logs.append(stdout)
        timings.append(monotonic() - command_started)
        (directory / f"{name}.compile.log").write_text("\n".join(logs))
        if proc.returncode:
            raise RuntimeError(f"CANN failed (exit {proc.returncode}): {command}\n{stdout}")
        return stdout

    try:
        for command in commands:
            run(command)
        version = run([str(binary), "--version"])
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        record = {
            "target": "dav-2201",
            "cann": str(CANN),
            "commands": commands,
            "compiler_version": version,
            "source_sha256": sha(src),
            "binary_sha256": sha(output),
            "binary_bytes": output.stat().st_size,
            "command_seconds": timings[: len(commands)],
            "compiler_version_seconds": timings[-1],
            "seconds": monotonic() - started,
        }
        if shared:
            record["shared_sha256"] = sha(directory / f"{name}.so")
        remaining(end, "CANN artifact validation")
        (directory / f"{name}.compile.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    except BaseException as error:
        for path in artifacts:
            path.unlink(missing_ok=True)
        (directory / f"{name}.compile.log").write_text("\n".join(logs))
        if isinstance(error, TimeoutError):
            timeout_record.write_text(
                json.dumps(
                    {
                        "status": "timeout",
                        "timeout_s": timeout_s,
                        "seconds": monotonic() - started,
                        "error": str(error),
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
        raise
