"""Build and check type-0 grouped GEMM, independently of any inference framework."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import median

from ascendflow.build import build_file
from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process, in_build_group
from ascendflow.runtime.torch import TorchRuntime

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/moe/grouped_gemm.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def record_build(directory, policy, verification):
    save(directory / "policy.json", policy)
    save(
        directory / "identity.json",
        {
            "policy": policy,
            "source_sha256": sha(SOURCE),
            "compiler_files": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))
            },
            "verification": verification,
        },
    )


def build(args, directory):
    capacity = args.capacity
    if capacity is None:
        capacity = (
            max(
                (int(ends.rsplit(",", 1)[-1]) for ends in args.ends),
                default=args.experts * args.rows,
            )
            + args.padding
        )
    bm = 16 if capacity <= 16 * args.experts else 128
    policy = {
        "E": args.experts,
        "K": args.k,
        "N": args.n,
        "DTYPE": args.dtype,
        "WEIGHT_LAYOUT": args.layout,
        "WEIGHT_CACHE": "auto",
        "BM": bm,
        "BN": 256,
        "BK": 256,
        "CK": 64,
        "GROUPS": 20,
        "SWIZZLE": 1 if bm == 16 else 8,
        "UNIT_FLAG": True,
    }
    result = build_file(SOURCE, directory, entry="grouped_gemm", parameters=policy, shared=True)
    record_build(directory, policy, result)


def profile(torch, launch, directory, repeats):
    """Measure device task durations; allocation and validation precede capture."""
    from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler

    require(not directory.exists(), f"profile output already exists: {directory}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(str(directory)),
    ):
        for _ in range(repeats):
            launch()
        torch.npu.synchronize()
    files = sorted(directory.rglob("task_time.csv"))
    require(len(files) == 1, f"expected one device task CSV in {directory}")
    rows = list(csv.DictReader(files[0].open()))
    require(rows, "profiler captured no tasks")
    # All tasks for an invocation count, including an original GMM's auxiliary tasks.
    require(len(rows) % repeats == 0, "profile tasks do not match launch count")
    multiplicity = len(rows) // repeats
    rows.sort(key=lambda r: float(r["task_start(us)"]))
    durations = [
        sum(float(r["task_time(us)"]) for r in rows[i : i + multiplicity])
        for i in range(0, len(rows), multiplicity)
    ]
    return {
        "csv": str(files[0].resolve()),
        "csv_sha256": sha(files[0]),
        "tasks_per_launch": multiplicity,
        "durations_us": durations,
        "median_us": median(durations),
        "operators": sorted({r["kernel_name"] for r in rows}),
        "timing_scope": "sum of device task durations per launch",
    }


def execute(args, artifact, output):
    health = health_snapshot(args.physical, output)
    policy = json.loads((artifact / "policy.json").read_text())
    identity = (
        json.loads((artifact / "identity.json").read_text())
        if (artifact / "identity.json").exists()
        else {}
    )
    manifest = json.loads((artifact / "grouped_gemm.compile.json").read_text())
    require(sha(artifact / "grouped_gemm.asc") == manifest["source_sha256"], "ASC changed")
    require(sha(artifact / "grouped_gemm.so") == manifest["shared_sha256"], "library changed")
    # CANN/Torch may create compilation metadata in the process directory.
    os.chdir(output)
    import numpy as np
    import torch
    import torch_npu

    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    e, k, n = (policy[key] for key in ("E", "K", "N"))
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[policy.get("DTYPE", "bf16")]
    ends_cases = [list(map(int, case.split(","))) for case in args.ends]
    if not ends_cases:
        ends_cases = [[(i + 1) * args.rows for i in range(e)]]
    generator = torch.Generator().manual_seed(args.seed)
    # Exact products/sums for short K also expose wrong expert/column packing.
    w_cpu = (
        torch.randint(-2, 3, (e, k, n), generator=generator).float() / 8
        if args.data == "exact"
        else torch.randn(e, k, n, generator=generator) / 8
    ).to(dtype)
    w = w_cpu.to("npu:0")
    if policy["WEIGHT_LAYOUT"] == "NZ":
        w = torch_npu.npu_format_cast(w, 29)
    record = {
        "status": "running",
        "policy": policy,
        "artifact": identity,
        "manifest": manifest,
        "health": health,
        "cases": [],
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "runner_sha256": sha(Path(__file__)),
        "data": args.data,
        "baseline": "torch_npu.npu_grouped_matmul split_item=2 group_type=0 group_list_type=0",
        "weight_format": torch_npu.get_npu_format(w),
        "weight_sha256": hashlib.sha256(w_cpu.view(torch.uint8).numpy()).hexdigest(),
        "cache_policy": "same input/weight addresses, warm allocations, no flush",
        "output_addresses": "generated reuses guarded out; original Torch allocates out",
        "max_slowdown_percent": 1.0,
    }
    save(output / "device.json", record)
    with TorchRuntime() as runtime:
        kernel = runtime.kernel(artifact / "grouped_gemm.so", "grouped_gemm")
        for number, ends in enumerate(ends_cases):
            require(
                len(ends) == e and ends == sorted(ends) and ends[0] >= 0,
                "--ends requires E nondecreasing cumulative ends",
            )
            capacity = args.capacity if args.capacity is not None else ends[-1] + args.padding
            require(capacity >= ends[-1], "capacity is smaller than last group end")
            x_cpu = (
                torch.randint(-2, 3, (capacity, k), generator=generator).float() / 8
                if args.data == "exact"
                else torch.randn(capacity, k, generator=generator) / 8
            ).to(dtype)
            x = x_cpu.to("npu:0")
            gl = torch.tensor(ends, dtype=torch.int64, device="npu:0")
            # Guard the exact output interval, including unaligned expert starts.
            storage = torch.full((capacity * n + 512,), 123, dtype=dtype, device="npu:0")
            y = storage[256:-256].view(capacity, n)
            invoke = kernel.bind(X=x, W=w, GroupList=gl, Y=y)
            row = {
                "ends": ends,
                "capacity": capacity,
                "status": "running",
                "input_sha256": hashlib.sha256(x_cpu.view(torch.uint8).numpy()).hexdigest(),
                "seed": args.seed,
                "pointers": {
                    "x": x.data_ptr(),
                    "weight": w.data_ptr(),
                    "group_list": gl.data_ptr(),
                    "generated_out": y.data_ptr(),
                },
            }
            record["cases"].append(row)
            record["phase"] = f"case {number}: generated launch"
            save(output / "device.json", record)
            invoke()
            record["phase"] = f"case {number}: generated synchronize"
            save(output / "device.json", record)
            torch.npu.synchronize()
            record["phase"] = f"case {number}: output readback"
            save(output / "device.json", record)
            actual = y.cpu().float().numpy()
            expected = np.zeros((capacity, n), dtype=np.float64)
            begin = 0
            for expert, end in enumerate(ends):
                expected[begin:end] = (
                    x_cpu[begin:end].double().numpy() @ w_cpu[expert].double().numpy()
                )
                begin = end
            rounded = torch.from_numpy(expected).to(dtype).float().numpy()
            tolerance = (0.0001 if dtype == torch.float16 else 0.0002) + np.abs(expected) * (
                0.0006 if dtype == torch.float16 else 0.004
            )
            errors = np.abs(actual - expected) > tolerance
            if args.data == "exact":
                errors = actual != rounded
            guards = storage[:256].eq(123).all().item() and storage[-256:].eq(123).all().item()
            row.update(
                {
                    "checked_elements": int(actual.size),
                    "errors": int(errors.sum()),
                    "guards_unchanged": guards,
                    "max_abs_error": float(np.abs(actual - expected).max(initial=0)),
                    "rounded_mismatches": int(np.count_nonzero(actual != rounded)),
                }
            )
            save(output / "device.json", record)
            require(
                np.isfinite(actual).all() and not errors.any() and guards,
                f"grouped GEMM numerical/guard failure: {row}",
            )
            row["numerical_status"] = "passed-cpu-oracle"
            save(output / "device.json", record)
            if ends[-1] > 0:
                # The original operator leaves capacity after S undefined; compare valid rows.
                baseline_result = [None]

                def original(x=x, gl=gl, baseline_result=baseline_result):
                    baseline_result[0] = torch_npu.npu_grouped_matmul(
                        x=[x],
                        weight=[w],
                        group_list=gl,
                        split_item=2,
                        group_type=0,
                        group_list_type=0,
                    )[0]

                record["phase"] = f"case {number}: original validation"
                save(output / "device.json", record)
                original()
                torch.npu.synchronize()
                reference = baseline_result[0][: ends[-1]].cpu().float().numpy()
                require(
                    np.all(np.abs(reference - expected[: ends[-1]]) <= tolerance[: ends[-1]]),
                    "original GMM disagrees with CPU oracle",
                )
                row["baseline_numerical_status"] = "passed-cpu-oracle"
                for _ in range(args.warmup):
                    invoke()
                    original()
                torch.npu.synchronize()
                if args.profile:
                    row["trials"] = []
                    for trial in range(args.trials):
                        profiles = {}
                        order = (
                            ("generated", "original")
                            if trial % 2 == 0
                            else ("original", "generated")
                        )
                        for backend in order:
                            record["phase"] = f"case {number}: {backend} trial {trial}"
                            save(output / "device.json", record)
                            profiles[backend] = profile(
                                torch,
                                invoke if backend == "generated" else original,
                                output / f"case-{number}-{backend}-{trial}",
                                args.repeats,
                            )
                        profiles["speedup"] = (
                            profiles["original"]["median_us"] / profiles["generated"]["median_us"]
                        )
                        row["trials"].append(profiles)
                        save(output / "device.json", record)
            require(
                torch.equal(x.cpu(), x_cpu)
                and torch.equal(w.cpu(), w_cpu)
                and gl.cpu().tolist() == ends,
                "operator modified an input",
            )
            row["inputs_unchanged"] = True
            row["status"] = "passed"
            print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
    record["status"] = "passed"
    record["phase"] = "complete"
    record["numerical_status"] = "passed"
    trials = [t for case in record["cases"] for t in case.get("trials", [])]
    record["performance_status"] = (
        "not-measured"
        if not trials
        else "regression"
        if any(t["generated"]["median_us"] > 1.01 * t["original"]["median_us"] for t in trials)
        else "passed-tested-cases"
        if args.trials >= 3
        else "probe"
    )
    save(output / "device.json", record)
    if args.require_performance:
        require(
            record["performance_status"] == "passed-tested-cases",
            f"performance acceptance failed: {record['performance_status']}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", type=int, default=0)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--layout", choices=("linear", "NZ"), default="NZ")
    parser.add_argument("--ends", nargs="*", default=[])
    parser.add_argument("--rows", type=int, default=7, help="rows per expert when --ends is absent")
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--data", choices=("exact", "normal"), default="exact")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--require-performance",
        action="store_true",
        help="require every trial within 1%% of original GMM",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.build:
        build(args, output)
    else:
        require(
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
            and os.environ.get("ASCEND_DEVICE_ID") == "0",
            "explicit physical mapping required",
        )
        try:
            if args.worker or in_build_group():
                execute(args, (args.artifact or output).resolve(), output)
            else:
                bounded_process(
                    [sys.executable, "-m", "benchmarks.moe.run", *sys.argv[1:], "--worker"],
                    output / "device-step",
                    timeout_s=300,
                )
        except Exception as error:
            path = output / "device.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(
                status="failed",
                failure_phase=record.get("phase"),
                error=f"{type(error).__name__}: {error}",
            )
            save(path, record)
            raise


if __name__ == "__main__":
    main()
