"""Direct CANN 9.1 ASC compilation. No TileLang/TVM/MLIR import or build."""

import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
from time import monotonic

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
    source: str, directory: Path, name: str, *, shared=False, timeout_s=300.0
) -> dict:
    """Compile within one wall-clock budget, killing the compiler process group on expiry."""
    if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 300:
        raise ValueError("CANN compilation timeout must be in (0,300] seconds")
    deadline = monotonic() + timeout_s
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.compile.json").unlink(missing_ok=True)
    (directory / f"{name}.timeout.json").unlink(missing_ok=True)
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
    logs = []
    for command in commands:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=max(0.001, deadline - monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, _ = proc.communicate()
            logs.append(stdout)
            (directory / f"{name}.compile.log").write_text("\n".join(logs))
            (directory / f"{name}.timeout.json").write_text(
                json.dumps(
                    {
                        "status": "timeout",
                        "timeout_s": timeout_s,
                        "command": command,
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
            raise TimeoutError(
                f"CANN compilation exceeded {timeout_s}s; process group killed"
            ) from None
        except BaseException:
            # The outer per-operator watchdog interrupts Python at its total
            # deadline; do not leave a detached CANN compiler/linker running.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        logs.append(stdout)
        (directory / f"{name}.compile.log").write_text("\n".join(logs))
        if proc.returncode:
            raise RuntimeError(f"CANN failed (exit {proc.returncode}): {command}\n{stdout}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "target": "dav-2201",
        "cann": str(CANN),
        "commands": commands,
        "compiler_version": subprocess.check_output(
            [str(binary), "--version"], text=True, timeout=max(0.001, deadline - monotonic())
        ),
        "source_sha256": sha(src),
        "binary_sha256": sha(output),
        "binary_bytes": output.stat().st_size,
    }
    if shared:
        record["shared_sha256"] = sha(directory / f"{name}.so")
    (directory / f"{name}.compile.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
