"""AscendFlow: compile multistage Ascend 910B source programs to ASC."""

import argparse
import json
from pathlib import Path

from .build import build_file
from .target.resources import DeviceResources


def main():
    parser = argparse.ArgumentParser(prog="ascendflow", description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Python DSL or serialized source IR"
    )
    parser.add_argument("--entry", help="captured AST prim_func in a Python input file")
    parser.add_argument(
        "--parameters", type=json.loads, default={}, help="JSON AST constexpr specializations"
    )
    parser.add_argument("--output", type=Path, required=True, help="external run/build directory")
    parser.add_argument(
        "--device-resources",
        type=json.loads,
        help='explicit target facts: {"cube_cores":20,"vector_cores":40}; no device is queried offline',
    )
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
        device_resources=None
        if args.device_resources is None
        else DeviceResources(**args.device_resources),
        timeout_s=args.timeout,
        compile_asc=args.compile,
        shared=args.shared,
        diagnostics=args.explain,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
