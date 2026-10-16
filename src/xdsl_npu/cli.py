"""Source DSL/IR compiler; algorithm options belong to examples."""

import argparse
import json
from pathlib import Path

from .build import build_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="typed source xDSL")
    parser.add_argument("--entry", help="captured AST prim_func in a Python input file")
    parser.add_argument(
        "--parameters", type=json.loads, default={}, help="JSON AST constexpr specializations"
    )
    parser.add_argument("--output", type=Path, required=True, help="external run/build directory")
    parser.add_argument(
        "--timeout", type=float, default=300, help="total operator deadline, at most 300 seconds"
    )
    parser.add_argument("--compile", action="store_true", help="CANN offline compilation")
    parser.add_argument(
        "--shared", action="store_true", help="also produce the CANN shared launcher"
    )
    parser.add_argument(
        "--explain", action="store_true", help="expensive per-access sync witnesses"
    )
    args = parser.parse_args()
    report = build_file(
        args.input,
        args.output,
        entry=args.entry,
        parameters=args.parameters,
        timeout_s=args.timeout,
        compile_asc=args.compile,
        shared=args.shared,
        diagnostics=args.explain,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
