"""Generic source/physical IR compiler; algorithm options belong to examples."""

import argparse
import json
from pathlib import Path

from .compilation import compile
from .ir import parse
from .toolchain import offline_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="typed source or physical xDSL")
    parser.add_argument("--output", type=Path, required=True, help="external run/build directory")
    parser.add_argument(
        "--synchronized", action="store_true", help="check an expert-synchronized physical plan"
    )
    parser.add_argument("--compile", action="store_true", help="CANN offline compilation")
    parser.add_argument(
        "--shared", action="store_true", help="also produce the CANN shared launcher"
    )
    parser.add_argument(
        "--explain", action="store_true", help="expensive per-access sync witnesses"
    )
    parser.add_argument("--check-interleavings", choices=["all", "shared"])
    parser.add_argument("--max-check-states", type=int, default=200_000)
    args = parser.parse_args()
    report = build_module(
        parse(args.input.read_text()),
        args.output,
        synchronize=not args.synchronized,
        compile_asc=args.compile,
        shared=args.shared,
        diagnostics=args.explain,
        check_interleavings=args.check_interleavings,
        max_check_states=args.max_check_states,
    )
    print(json.dumps(report, indent=2))


def build_module(
    module,
    output,
    *,
    synchronize=True,
    compile_asc=False,
    shared=False,
    diagnostics=False,
    check_interleavings=None,
    max_check_states=200_000,
):
    """Compile generic IR once; optional diagnostics do not change its schedule."""
    output = Path(output)
    result = compile(module, synchronize=synchronize)
    report = result.report
    report["compilation"] = result.timings
    if check_interleavings:
        from dataclasses import asdict

        from .check import bounded_check
        from .ir import MixedOp
        from .mixed_check import bounded_mixed_check

        plan = result.plan
        if isinstance(plan.body.block.first_op, MixedOp):
            checked = bounded_mixed_check(
                plan, scope=check_interleavings, max_states=max_check_states
            )
        elif check_interleavings == "all":
            checked = bounded_check(plan, max_states=max_check_states)
        else:
            raise ValueError("shared interleaving scope requires a mixed kernel")
        report["bounded_interleavings"] = asdict(checked)
    result.write(output, diagnostics=diagnostics)
    if compile_asc or shared:
        report["cann"] = offline_compile(result.asc, output.resolve(), result.name, shared=shared)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
