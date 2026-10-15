import argparse
import json
from pathlib import Path

from benchmarks.references.aclnn import run_reference
from xdsl_npu.ir import parse

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--physical", type=int, default=1)
parser.add_argument("--repeats", type=int, default=30)
parser.add_argument("--data-case", choices=["standard", "numerical-edges"], default="standard")
parser.add_argument("--record-output", type=Path)
args = parser.parse_args()
plan = parse((args.directory / "after.xdsl").read_text())
print(
    json.dumps(
        run_reference(
            plan,
            args.record_output or args.directory,
            physical=args.physical,
            repeats=args.repeats,
            data_case=args.data_case,
        ),
        indent=2,
    )
)
