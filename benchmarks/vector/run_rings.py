"""Build and validate independent input/output rings."""

import argparse
from pathlib import Path

from examples.vector.independent_rings import independent_rings
from xdsl_npu import compile
from xdsl_npu.toolchain import offline_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--physical", type=int)
    args = parser.parse_args()
    result = compile(independent_rings)
    result.write(args.output)
    library = args.library
    if args.compile or args.shared:
        offline_compile(result.asc, args.output.resolve(), result.name, shared=args.shared)
        if args.shared:
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
