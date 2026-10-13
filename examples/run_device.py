"""Run AFTER offline compile. All outputs belong in the external runs directory."""

import argparse
import json
from pathlib import Path

from xdsl_npu.device import run_example
from xdsl_npu.ir import parse

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--physical", type=int, default=0)
parser.add_argument("--repeats", type=int, default=30)
parser.add_argument("--late-reader", action="store_true")
parser.add_argument(
    "--reference", action="store_true", help="external hand-written comparison only"
)
args = parser.parse_args()
plan = parse((args.directory / "after.xdsl").read_text())
from xdsl_npu.model import extract

program = extract(plan)
result = run_example(
    plan,
    args.directory / f"{program.name}.so",
    args.directory,
    physical=args.physical,
    repeats=args.repeats,
    late_reader=args.late_reader,
    reference=args.reference,
)
print(json.dumps(result, indent=2))
