"""Verify the source/library identity recorded by a profiling run."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ascendflow.diagnostics import require
from ascendflow.dialects.io import parse


@dataclass(frozen=True)
class FrozenArtifact:
    """Published bytes and metadata for runners; never a verification authority."""

    name: str
    asc: str
    plan_ir: str | None
    _report_json: str
    _timings_json: str

    @property
    def report(self):
        return json.loads(self._report_json)

    @property
    def timings(self):
        return json.loads(self._timings_json)

    @property
    def plan(self):
        require(self.plan_ir is not None, "an ASC reference has no compiler PlanIR")
        # As with Compilation.plan, callers must pass the fresh mutable module
        # through the ordinary validators. JSON supplies no trusted status.
        return parse(self.plan_ir)


def read_artifact(directory):
    directory = Path(directory)
    report = json.loads((directory / "verification.json").read_text())
    name = report["kernel"]
    require(isinstance(name, str) and name.isidentifier(), "invalid artifact kernel name")
    plan = directory / "after.xdsl"
    return FrozenArtifact(
        name,
        (directory / f"{name}.asc").read_text(),
        plan.read_text() if report.get("kind") != "asc_reference" and plan.exists() else None,
        json.dumps({k: v for k, v in report.items() if k not in {"compilation", "cann"}}),
        json.dumps(report.get("compilation", {})),
    )


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
