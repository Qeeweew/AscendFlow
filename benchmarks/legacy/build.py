"""Retained physical recipes for regression; new examples use language.py."""

import argparse
import json
from pathlib import Path

from examples.legacy.gdn import recurrent_gdn
from examples.legacy.operators import (
    batched_matmul_nt,
    gemm_bias_silu,
    gemm_nt,
    rms_norm,
    rope,
    softmax,
    swiglu,
)
from tests.fixtures.pipelines import (
    cube_pipeline,
    mixed_pipeline,
    phased_pipeline,
    scalar_pipeline,
    vector_pipeline,
)
from xdsl_npu.cli import build_module
from xdsl_npu.ir import parse


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
            "phased",
            "gdn",
            "ir",
        ],
    )
    parser.add_argument("--input", type=Path, help="physical unsynchronized xDSL for example=ir")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--phases", type=int, default=9)
    parser.add_argument("--phase-length", type=int, default=1)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--snapshot-slots", type=int, default=2)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--segments", type=int, default=1, help="native pieces per mixed GM epoch")
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
        module = mixed_pipeline(
            args.count, args.slots, aiv1_delay=args.aiv1_delay, segments=args.segments
        )
    elif args.example == "scalar":
        module = scalar_pipeline(args.count, args.blocks)
    elif args.example == "phased":
        module = phased_pipeline(args.phases, args.phase_length, args.columns)
    elif args.example == "gdn":
        module = recurrent_gdn(
            args.sequences,
            args.steps,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            slots=args.slots,
            snapshot_slots=args.snapshot_slots,
        )
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
    report = build_module(
        module,
        args.output,
        compile_asc=args.compile,
        shared=args.shared,
        check_interleavings=args.check_interleavings,
        max_check_states=args.max_check_states,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
