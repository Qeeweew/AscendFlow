"""Reproducible FA correctness/timing; outputs must be outside product Git."""

import argparse
import json
from pathlib import Path

from attention_oracle import run_attention

from xdsl_npu.ir import parse

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--physical", type=int, required=True)
parser.add_argument("--repeats", type=int, default=30)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--oracle", choices=["full", "sampled"], default="full")
parser.add_argument("--data-case", choices=["standard", "sharp", "uniform"], default="standard")
parser.add_argument("--backend", choices=["generated", "aclnn"], default="generated")
parser.add_argument("--record-output", type=Path)
args = parser.parse_args()
plan = parse((args.directory / "after.xdsl").read_text())
result = run_attention(
    plan,
    args.directory / "flash_attention.so",
    args.record_output or args.directory,
    physical=args.physical,
    repeats=args.repeats,
    warmup=args.warmup,
    scope=args.oracle,
    data_case=args.data_case,
    backend=args.backend,
)
print(json.dumps(result, indent=2))
