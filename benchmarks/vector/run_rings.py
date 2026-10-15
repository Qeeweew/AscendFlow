"""Build and validate independent input/output rings."""

import argparse
from pathlib import Path

from benchmarks.artifacts import read_artifact
from xdsl_npu.build import build_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    build_output = args.output if args.library is None else args.output / "source-check"
    build_file(
        Path(__file__).resolve().parents[2] / "examples/vector/independent_rings.py",
        build_output,
        entry="independent_rings",
        compile_asc=args.library is None and args.compile,
        shared=args.library is None and args.shared,
        timeout_s=args.timeout,
    )
    result = read_artifact(build_output)
    library = args.library
    if library is None and args.shared:
        library = args.output / f"{result.name}.so"
    if args.physical is not None:
        if library is None:
            parser.error("--physical requires --shared or --library")
        from benchmarks.vector.run import run

        run(
            result,
            library,
            args.output,
            args.physical,
            [0, 1, 2, 4, 5, 6, 7, 8, 14, 35, 65536, 262144],
            5,
        )
    print(result.timings)


if __name__ == "__main__":
    main()
