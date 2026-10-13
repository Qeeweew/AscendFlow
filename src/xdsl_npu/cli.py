import argparse
import json
from pathlib import Path

from .codegen import emit_asc
from .examples import cube_pipeline, mixed_pipeline, scalar_pipeline, vector_pipeline
from .ir import dump, parse
from .kernels import batched_matmul_nt, gemm_bias_silu, gemm_nt, rms_norm, rope, softmax, swiglu
from .sync import autosync, explain_plan, verify_plan
from .toolchain import offline_compile


def main():
    parser = argparse.ArgumentParser(description="Physical xDSL NPU compiler for dav-2201")
    parser.add_argument(
        "example",
        choices=[
            "vector",
            "last-read",
            "cube",
            "scalar",
            "rms-norm",
            "rope",
            "softmax",
            "swiglu",
            "batched-matmul",
            "gemm",
            "gemm-bias-silu",
            "mixed",
            "ir",
        ],
    )
    parser.add_argument("--input", type=Path, help="physical unsynchronized xDSL for example=ir")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=128)
    parser.add_argument("--row-block", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--k-block", type=int, default=64)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--l1-k", type=int, default=128)
    parser.add_argument("--l0-k", type=int, default=64)
    parser.add_argument("--l1-slots", type=int, default=2)
    parser.add_argument("--l0-slots", type=int, default=2)
    parser.add_argument("--grid-m", type=int)
    parser.add_argument("--c-slots", type=int, default=1)
    parser.add_argument("--cache-b", action="store_true")
    parser.add_argument("--gm-slots", type=int, default=2)
    parser.add_argument("--ub-slots", type=int, default=2)
    parser.add_argument(
        "--aiv1-delay",
        type=int,
        default=0,
        help="mixed example: real private work before AIV1 import",
    )
    parser.add_argument("--blocks", type=int, default=1, help="independent physical cores")
    parser.add_argument(
        "--tail", type=int, default=0, help="final Vector tile element count, multiple of 8"
    )
    parser.add_argument("--output", type=Path, required=True, help="external run/build directory")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--check-interleavings", choices=["all", "shared"])
    parser.add_argument("--max-check-states", type=int, default=200_000)
    args = parser.parse_args()
    if args.example == "ir":
        if args.input is None:
            parser.error("ir requires --input")
        module = parse(args.input.read_text())
    elif args.example == "cube":
        module = cube_pipeline(args.count, args.slots)
    elif args.example == "mixed":
        module = mixed_pipeline(args.count, args.slots, aiv1_delay=args.aiv1_delay)
    elif args.example == "scalar":
        module = scalar_pipeline(args.count, args.blocks)
    elif args.example == "batched-matmul":
        module = batched_matmul_nt(
            args.batch, args.m, args.n, args.k, k_block=args.k_block, slots=args.slots
        )
    elif args.example == "gemm-bias-silu":
        module = gemm_bias_silu(
            args.m,
            args.n,
            args.k,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            l1_k=args.l1_k,
            l0_k=args.l0_k,
            l1_slots=args.l1_slots,
            l0_slots=args.l0_slots,
            grid_m=args.grid_m,
            gm_slots=args.gm_slots,
            ub_slots=args.ub_slots,
            c_slots=args.c_slots,
            cache_b=args.cache_b,
        )
    elif args.example == "gemm":
        module = gemm_nt(
            args.m,
            args.n,
            args.k,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            l1_k=args.l1_k,
            l0_k=args.l0_k,
            l1_slots=args.l1_slots,
            l0_slots=args.l0_slots,
            grid_m=args.grid_m,
            c_slots=args.c_slots,
            cache_b=args.cache_b,
        )
    elif args.example in {"rms-norm", "rope", "softmax", "swiglu"}:
        factory = {"rms-norm": rms_norm, "rope": rope, "softmax": softmax, "swiglu": swiglu}[
            args.example
        ]
        module = factory(
            args.rows, args.columns, blocks=args.blocks, row_block=args.row_block, slots=args.slots
        )
    else:
        module = vector_pipeline(
            args.count, slots=args.slots, late_reader=args.example == "last-read", tail=args.tail
        )
    plan = autosync(module)
    report = verify_plan(plan)
    if args.check_interleavings:
        from dataclasses import asdict

        from .check import bounded_check
        from .ir import MixedOp
        from .mixed_check import bounded_mixed_check

        if isinstance(plan.body.block.first_op, MixedOp):
            checked = bounded_mixed_check(
                plan, scope=args.check_interleavings, max_states=args.max_check_states
            )
        elif args.check_interleavings == "all":
            checked = bounded_check(plan, max_states=args.max_check_states)
        else:
            parser.error("shared interleaving scope requires a mixed kernel")
        report["bounded_interleavings"] = asdict(checked)
    source = emit_asc(plan)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "before.xdsl").write_text(dump(module))
    (args.output / "after.xdsl").write_text(dump(plan))
    (args.output / "witnesses.json").write_text(json.dumps(explain_plan(plan), indent=2) + "\n")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    name = plan.body.block.first_op.sym_name.data
    (args.output / f"{name}.asc").write_text(source)
    if args.compile or args.shared:
        report["cann"] = offline_compile(source, args.output.resolve(), name, shared=args.shared)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
