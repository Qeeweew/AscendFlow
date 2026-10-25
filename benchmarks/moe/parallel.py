"""Single-device local EP/TP composition check; no communication framework."""

import argparse
import json
import os
import sys
from pathlib import Path

from ascendflow.build import build_file
from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process, in_build_group
from ascendflow.runtime.torch import TorchRuntime
from benchmarks.moe.run import SOURCE, record_build, save, sha
from examples.moe.torch import GroupedGemm

SHAPES = tuple((e, k, n) for e in (1, 2) for k, n in ((32, 64), (32, 32), (16, 32)))


def directory(root, shape):
    e, k, n = shape
    return root / f"e{e}-k{k}-n{n}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.build:
        for e, k, n in SHAPES:
            path = directory(output, (e, k, n))
            policy = {
                "E": e,
                "K": k,
                "N": n,
                "WEIGHT_LAYOUT": "NZ",
                "DTYPE": "bf16",
                "WEIGHT_CACHE": "auto",
                "BM": 16,
                "BN": 64,
                "BK": 32,
                "CK": 32,
                "GROUPS": 2,
                "SWIZZLE": 1,
                "UNIT_FLAG": True,
            }
            result = build_file(
                SOURCE,
                path,
                entry="grouped_gemm",
                shared=True,
                parameters=policy,
            )
            record_build(path, policy, result)
        return
    if not args.worker and not in_build_group():
        try:
            bounded_process(
                [sys.executable, "-m", "benchmarks.moe.parallel", *sys.argv[1:], "--worker"],
                output / "device-step",
                timeout_s=300,
            )
        except Exception as error:
            path = output / "device.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            save(path, record)
            raise
        return
    artifact = (args.artifact or output).resolve()
    health = health_snapshot(args.physical, output)
    os.chdir(output)
    import torch
    import torch_npu

    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(20261025)
    dtype = torch.bfloat16
    x = (torch.randn((7, 32), generator=rng) / 8).to(dtype)
    w13 = (torch.randn((2, 32, 64), generator=rng) / 8).to(dtype)
    w2 = (torch.randn((2, 32, 32), generator=rng) / 8).to(dtype)
    report = {
        "status": "running",
        "health": health,
        "cases": [],
        "runner_sha256": sha(Path(__file__)),
        "libraries": {},
    }
    save(output / "device.json", report)
    with TorchRuntime() as runtime:
        operators = {}
        for shape in SHAPES:
            path = directory(artifact, shape) / "grouped_gemm.so"
            operators[shape] = GroupedGemm(runtime, path)
            manifest = json.loads(path.with_suffix(".compile.json").read_text())
            require(sha(path) == manifest["shared_sha256"], "library changed")
            require(sha(path.with_suffix(".asc")) == manifest["source_sha256"], "ASC changed")
            report["libraries"][str(shape)] = {
                "path": str(path),
                "sha256": sha(path),
                "manifest": manifest,
                "identity": json.loads((path.parent / "identity.json").read_text()),
            }

        def gmm(a, weight, ends):
            report["phase"] = {"shape": list(weight.shape), "ends": ends, "step": "prepare"}
            save(output / "device.json", report)
            op = operators[tuple(weight.shape)]
            packed = torch_npu.npu_format_cast(weight.to("npu:0"), 29)
            report["phase"]["step"] = "launch"
            save(output / "device.json", report)
            result = op(
                a.to("npu:0"), packed, torch.tensor(ends, dtype=torch.int64, device="npu:0")
            )
            report["phase"]["step"] = "synchronize"
            save(output / "device.json", report)
            torch.npu.synchronize()
            actual = result.cpu()
            expected = torch.empty_like(actual)
            begin = 0
            for expert, end in enumerate(ends):
                expected[begin:end] = (a[begin:end].double() @ weight[expert].double()).to(dtype)
                begin = end
            require(
                torch.allclose(actual.float(), expected.float(), atol=0.0005, rtol=0.008),
                "local shard GEMM differs from CPU oracle",
            )
            return actual

        def moe(a, gate_up, down, ends):
            intermediate = gmm(a, gate_up, ends)
            gate, up = intermediate.float().chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate) * up).to(dtype)
            return gmm(activated, down, ends)

        full = moe(x, w13, w2, [3, 7])
        for ep, tp in ((1, 2), (2, 1), (2, 2)):
            result = torch.empty_like(full, dtype=torch.float32)
            expert_sets = (
                ((0, 2, 0, 7, [3, 7]),) if ep == 1 else ((0, 1, 0, 3, [3]), (1, 2, 3, 7, [4]))
            )
            for e0, e1, r0, r1, ends in expert_sets:
                parts = []
                for rank in range(tp):
                    j0, j1 = rank * (32 // tp), (rank + 1) * (32 // tp)
                    local13 = torch.cat(
                        (w13[e0:e1, :, j0:j1], w13[e0:e1, :, 32 + j0 : 32 + j1]), dim=-1
                    )
                    local2 = w2[e0:e1, j0:j1, :].contiguous()
                    parts.append(moe(x[r0:r1], local13, local2, ends).float())
                result[r0:r1] = sum(parts)
            error = (result - full.float()).abs()
            # Each TP contribution has its own final BF16 rounding boundary.
            require(
                torch.allclose(result, full.float(), atol=0.001, rtol=0.02),
                f"EP={ep}, TP={tp} composition differs from unsharded MoE",
            )
            row = {
                "ep": ep,
                "tp": tp,
                "max_abs_error": error.max().item(),
                "local_gemms_checked_against_cpu": True,
            }
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
    report["status"] = "passed"
    report["phase"] = "complete"
    save(output / "device.json", report)


if __name__ == "__main__":
    main()
