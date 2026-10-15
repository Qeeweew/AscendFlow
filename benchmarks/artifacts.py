"""Verify the source/library identity recorded by a profiling run."""

import hashlib
from pathlib import Path

from xdsl_npu.model import require


def verified_source(manifest: dict, build_cwd: Path | None = None) -> Path:
    command = manifest["commands"][0]
    source = Path(command[command.index("-o") - 1])
    if not source.is_absolute():
        require(build_cwd is not None, "relative compile paths require --build-cwd")
        source = (build_cwd / source).resolve()
    require(
        hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
        and hashlib.sha256(source.with_suffix(".so").read_bytes()).hexdigest()
        == manifest["shared_sha256"],
        "profile source/library changed",
    )
    return source
