"""Hardware/work accounting for a completed msprof run; writes outside Git."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.validation.attention import attention_description
from xdsl_npu.codegen import emit_asc
from xdsl_npu.ir import parse
from xdsl_npu.performance import (
    hardware_snapshot,
    physical_work,
    profile_evidence,
    theoretical_metrics,
)

parser = argparse.ArgumentParser()
parser.add_argument("compiled", type=Path)
parser.add_argument("profile", type=Path)
parser.add_argument("--physical", type=int, required=True)
parser.add_argument("--hbm-gbps", type=float)
parser.add_argument("--l2-gbps", type=float)
args = parser.parse_args()
summary = json.loads((args.profile / "summary.json").read_text())
device = json.loads((args.profile / "device/device.json").read_text())
if device["backend"] != "generated" or device["health"]["physical_device"] != args.physical:
    raise ValueError("work accounting requires the generated kernel on the recorded physical card")
ir = (args.compiled / "after.xdsl").read_bytes()
plan = parse(ir.decode())
source = (args.compiled / "flash_attention.asc").read_bytes()
if (
    hashlib.sha256(source).hexdigest() != device["compiled_source_sha256"]
    or emit_asc(plan).encode() != source
    or hashlib.sha256((args.compiled / "flash_attention.so").read_bytes()).hexdigest()
    != device["shared_sha256"]
):
    raise ValueError("recorded performance does not match the current IR, source and library")
evidence = profile_evidence(
    args.profile,
    summary,
    physical=args.physical,
    warmup=device["warmup"],
    repeats=device["repeats"],
    kernel=device["kernel"],
)
_, attributes, _ = attention_description(plan)
b, h, _, sq, sk, d = [n.data for n in attributes["attention_shape"]]
cells = sq * sk
if attributes["attention_causal"].data:
    start = attributes["attention_start"].data
    cells = sum(min(sk, start + row + 1) for row in range(sq))
work = physical_work(plan, useful_cube_flops=4 * b * h * d * cells)
hardware = hardware_snapshot(args.physical)
record = {
    "hardware": hardware,
    "profile_evidence": evidence,
    "work": work,
    "metrics": theoretical_metrics(
        work, hardware, summary["median_us"], hbm_gbps=args.hbm_gbps, l2_gbps=args.l2_gbps
    ),
    "compiled_source_sha256": device["compiled_source_sha256"],
    "physical_ir_sha256": hashlib.sha256(ir).hexdigest(),
    "profile_csv_sha256": summary["csv_sha256"],
    "numerical_scope": device["oracle_scope"],
}
(args.profile / "performance.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record["metrics"], indent=2))
