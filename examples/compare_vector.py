"""Finite generated-vs-expert comparison. Does not search/tune configurations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pipeline_fixtures import vector_pipeline

from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.ir import dump
from xdsl_npu.toolchain import offline_compile

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--count", type=int, default=32)
parser.add_argument("--tile", type=int, default=128)
parser.add_argument("--physical", type=int, default=0)
parser.add_argument("--run", action="store_true")
args = parser.parse_args()
if args.count < 3 or args.tile <= 0 or args.tile % 8:
    parser.error("reference supports count>=3 and a positive multiple-of-8 tile")
module = vector_pipeline(args.count, args.tile)
plan = autosync(module)
generated = emit_asc(plan)
manual = Path(__file__).with_name("manual_vector.asc.in").read_text()
values = {
    "COUNT": args.count,
    "END": args.count + 2,
    "V_END": args.count + 1,
    "RELEASE_END": args.count - 2,
    "TILE": args.tile,
    "TILE_BYTES": args.tile * 4,
    "RING_BYTES": args.tile * 8,
}
for key, value in values.items():
    manual = manual.replace(f"@{key}@", str(value))
reports = {}
for variant, source in [("generated", generated), ("manual", manual)]:
    directory = (args.output / variant).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "before.xdsl").write_text(dump(module))
    # The manual source is an external reference; this PlanIR describes the
    # generated side's effects/issue organization, not a proof of manual C++.
    (directory / "after.xdsl").write_text(dump(plan))
    (directory / "verification.json").write_text(json.dumps(verify_plan(plan), indent=2))
    reports[variant] = offline_compile(source, directory, "vector_pipeline", shared=True)
    if args.run:
        runner = Path(__file__).with_name("run_device.py")
        subprocess.run(
            [sys.executable, str(runner), str(directory), "--physical", str(args.physical)]
            + (["--reference"] if variant == "manual" else []),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        reports[variant]["device"] = json.loads((directory / "device.json").read_text())
equal = reports["generated"]["binary_sha256"] == reports["manual"]["binary_sha256"]
result = {
    "shape": [args.count, args.tile],
    "dtype": "f32",
    "slots": 2,
    "stages": [0, 1, 1, 2],
    "same_device_elf": equal,
    "physical_ir_sha256": hashlib.sha256(dump(module).encode()).hexdigest(),
    "variants": reports,
    "scope": "single AIV toy pipeline; no claim about production GEMM/attention performance",
}
(args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"same_device_elf": equal, "output": str(args.output)}, indent=2))
