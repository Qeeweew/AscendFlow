"""Direct CANN 9.1 ASC compilation. No TileLang/TVM/MLIR import or build."""

import hashlib
import json
import subprocess
from pathlib import Path

CANN = Path("/usr/local/Ascend/cann-9.1.0")
FLAGS = [
    "-O2",
    "-fPIC",
    "-std=c++20",
    "-mllvm",
    "-cce-aicore-dcpreload-args=false",
    "--npu-arch=dav-2201",
]


def offline_compile(source: str, directory: Path, name: str, *, shared=False) -> dict:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
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
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
        logs.append(proc.stdout)
        (directory / f"{name}.compile.log").write_text("\n".join(logs))
        if proc.returncode:
            raise RuntimeError(f"CANN failed (exit {proc.returncode}): {command}\n{proc.stdout}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "target": "dav-2201",
        "cann": str(CANN),
        "commands": commands,
        "compiler_version": subprocess.check_output([str(binary), "--version"], text=True),
        "source_sha256": sha(src),
        "binary_sha256": sha(output),
        "binary_bytes": output.stat().st_size,
    }
    if shared:
        record["shared_sha256"] = sha(directory / f"{name}.so")
    (directory / f"{name}.compile.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
