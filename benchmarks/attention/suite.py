"""Fixed Attention cases, with a hard per-operator compilation watchdog.

Current workers explicitly use the physical examples; they are not AST FA.
Stop the entire build sequence on a timeout. No device work or policy search is
started implicitly. Generated sources/manifests remain in the external output.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic

from xdsl_npu.model import require

# B, Hq, Hkv, Sq, Sk, D, dtype, causal. Full shapes, not tiny timing cases.
CASES = {
    "mha4k": (1, 8, 8, 4096, 4096, 128, "f16", False),
    "gqa8k": (1, 8, 2, 8192, 8192, 64, "bf16", False),
    "mqa4k": (1, 8, 1, 4096, 4096, 64, "f16", False),
    "cross4k8k": (1, 8, 2, 4096, 8192, 64, "bf16", False),
    "causal4k": (1, 8, 8, 4096, 4096, 128, "f16", True),
    "decode32k": (1, 32, 1, 1, 32768, 64, "bf16", False),
}


def bounded_process(command, directory, *, timeout_s=300):
    """A separate process covers construction, AutoSync, emission AND CANN."""
    require(
        type(timeout_s) in {int, float} and 0 < timeout_s <= 300,
        "per-operator compile timeout must be in (0,300] seconds",
    )
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    with (directory / "build.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        status = "failed"
        try:
            code = process.wait(timeout=timeout_s)
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            # Interrupt Python first so toolchain kills its detached compiler
            # process group. A short grace period is cleanup, never more work.
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = process.returncode
        record = {
            "status": status,
            "seconds": monotonic() - started,
            "deadline_s": timeout_s,
            "returncode": code,
            "command": command,
        }
    (directory / "build-result.json").write_text(json.dumps(record, indent=2) + "\n")
    if status == "timeout":
        raise TimeoutError(f"operator compilation reached {timeout_s}s; stopped, see {directory}")
    require(status == "passed", f"operator compilation failed; see {directory / 'build.log'}")
    return record


def build_case(name, directory, policy, *, load3d=False):
    from examples.flash_attention.legacy import flash_attention
    from examples.flash_attention.tiled_legacy import flash_attention_tiled
    from xdsl_npu.cli import build_module

    *shape, dtype, causal = CASES[name]
    if policy in {"ck256", "continuous", "tile-vector", "block-max"}:
        require(
            not causal and shape[-3] != 1,
            "tiled candidates require dense complete prefill/cross attention",
        )
        require(policy != "ck256" or shape[-1] == 64, "CK256 candidate requires D64")
        module = flash_attention_tiled(
            *shape,
            dtype=dtype,
            block_q=64 if policy == "ck256" else 128,
            cube_kv=256 if policy == "ck256" else 128,
            query_tiles_per_group=2 if policy == "ck256" or shape[-2] > 4096 else 4,
            overlap_queries=policy in {"continuous", "tile-vector", "block-max"},
            vector_update="tile" if policy in {"tile-vector", "block-max"} else "rows",
            max_reduce="block" if policy == "block-max" else "whole",
        )
    else:
        module = flash_attention(
            *shape,
            dtype=dtype,
            block_q=32 if shape[-3] == 1 else 128,
            q_cache="l1",
            reuse_ub=True,
            pipeline=True,
            packed_reductions=True,
            causal=causal,
        )
    build_module(module, directory, shared=True, load3d=load3d, timeout_s=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=["mha4k", "gqa8k", "mqa4k", "cross4k8k"]
    )
    parser.add_argument(
        "--policy",
        choices=["existing", "ck256", "continuous", "tile-vector", "block-max"],
        default="existing",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load3d", action="store_true")
    parser.add_argument("--compile-timeout", type=float, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(len(args.cases) == 1, "one operator per isolated worker")
        build_case(args.cases[0], args.output, args.policy, load3d=args.load3d)
        return
    for name in args.cases:
        directory = (args.output / name).resolve()
        command = [
            sys.executable,
            "-m",
            "benchmarks.attention.suite",
            "--worker",
            "--cases",
            name,
            "--policy",
            args.policy,
            "--output",
            str(directory),
        ]
        if args.load3d:
            command.append("--load3d")
        result = bounded_process(command, directory, timeout_s=args.compile_timeout)
        print(json.dumps({"case": name, "policy": args.policy, **result}), flush=True)


if __name__ == "__main__":
    main()
