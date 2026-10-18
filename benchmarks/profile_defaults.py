"""Compare default-policy FA/GEMM shapes with ACLNN using one msprof run each."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from ascendflow.device import health_snapshot
from ascendflow.diagnostics import require
from ascendflow.runtime.process import bounded_process
from benchmarks.artifacts import verified_source
from benchmarks.check_examples import ROOT, run_device


@dataclass(frozen=True)
class ProfileCase:
    operator: str
    shape: tuple[int, ...]

    @property
    def name(self):
        return self.operator + "-" + "x".join(map(str, self.shape))

    @property
    def attention(self):
        return self.operator == "flash-attention"

    @property
    def module(self):
        return "benchmarks.attention.run" if self.attention else "benchmarks.gemm.run_matmul"

    @property
    def artifact_key(self):
        # Sq/Sk remain runtime parameters; all attention cases share one build.
        return self.operator if self.attention else self.name

    @property
    def build_options(self):
        if self.attention:
            return ()
        return tuple(
            value
            for key, size in zip(("m", "n", "k"), self.shape, strict=True)
            for value in ("--" + key, str(size))
        )

    @property
    def run_options(self):
        return (
            ("--cases", ":".join(map(str, self.shape))) if self.attention else ("--data", "normal")
        )

    @property
    def launch_counts(self):
        return (1, 3) if self.attention else (5, 30)


CASES = tuple(
    ProfileCase("flash-attention", shape)
    for shape in (
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (4096, 8192),
        (8192, 4096),
    )
) + tuple(
    ProfileCase("gemm-normal", shape)
    for shape in (
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (2048, 8192, 4096),
        (8192, 2048, 4096),
    )
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithm_flops(operator, shape):
    """Useful matrix FLOPs, FMA=2; FA excludes softmax and online-state work."""
    require(all(type(size) is int and size > 0 for size in shape), "invalid problem dimensions")
    if operator == "gemm-normal":
        require(len(shape) == 3, "GEMM shape must be M,N,K")
        m, n, k = shape
        return 2 * m * n * k
    require(operator == "flash-attention" and len(shape) == 6, "unknown FLOP contract")
    batch, heads, _, sq, sk, dim = shape
    return 4 * batch * heads * sq * sk * dim


def tflops(flops, duration_us):
    require(
        flops > 0 and math.isfinite(duration_us) and duration_us > 0,
        "FLOPs and device duration must be positive and finite",
    )
    return flops / (duration_us * 1e6)


def case_contract(spec, record, backend):
    """Read actual math, numerical checks and input identity for one backend."""
    require(
        record.get("status") == "passed" and len(record["cases"]) == 1,
        "profile needs one successful numerical case",
    )
    case = record["cases"][0]
    require(
        case["backend"] == backend and (case["warmup"], case["repeats"]) == spec.launch_counts,
        "backend or runner launch defaults changed",
    )
    require(
        case["inputs_unchanged"]
        and (case["guards_checked"] if spec.attention else case["guards_unchanged"]),
        "missing input/guard validation",
    )
    require(
        record["health"]["logical_device"] == 0
        and record["hardware"]["physical_device"] == record["health"]["physical_device"],
        "inconsistent physical/logical device mapping",
    )
    if spec.attention:
        shape = case["shape_b_hq_hkv_sq_sk_d"]
        require(
            shape == [1, 8, 8, *spec.shape, 128]
            and case["dtype"] == "f16"
            and case["layout"] == "BNSD"
            and not case["causal"]
            and case["query_start"] == 0
            and case["data_case"] == "standard"
            and case["oracle_scope"] == "sampled",
            "attention policy or shape differs",
        )
        inputs = {key: case["argument_sha256"][key] for key in ("q", "k", "v")}
        numerical = {
            key: case[key]
            for key in ("softmax_scale_f32_bits", "query_rows_per_head", "rtol", "atol")
        }
        identity = record["artifact"]
    else:
        shape = case["shape_mnk"]
        require(
            shape == list(spec.shape)
            and case["input_dtype"] == case["output_dtype"] == "f16"
            and case["layout"] == "NN contiguous"
            and case["data"] == "normal"
            and case["scope"] == "sampled"
            and case["finite_output"]
            and case["offset_elements"] == 0,
            "GEMM policy or shape differs",
        )
        inputs = case["input_sha256"]
        numerical = {key: case[key] for key in ("seed", "oracle_rows", "rtol", "atol")}
        identity = record["backend"]
    if backend == "generated":
        manifest = identity["compile_manifest" if spec.attention else "manifest"]
        verified_source(manifest)
    else:
        require(
            set(identity["libraries"]) == {"libopapi.so", "libnnopbase.so"},
            "missing installed ACLNN libraries",
        )
        for library in identity["libraries"].values():
            require(digest(library["path"]) == library["sha256"], "ACLNN library changed")
        if not spec.attention:
            require(
                identity["tensor_dtypes"] == {key: 1 for key in ("A", "B", "C")},
                "ACLNN actual GEMM dtype differs",
            )
    return {
        "shape": shape,
        "dtype": "f16",
        "inputs": inputs,
        "numerical": numerical,
        "physical": record["health"]["physical_device"],
        "platform_sha256": record["hardware"]["platform_sha256"],
    }


def timing_summary(profile, spec, physical):
    warmup, repeats = spec.launch_counts
    command = [
        sys.executable,
        "-m",
        "benchmarks.profile_summary",
        str(profile),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((profile / "summary.json").read_text())
    source = Path(summary["csv"]).resolve()
    require(
        source.is_relative_to(profile.resolve()) and digest(source) == summary["csv_sha256"],
        "timing CSV identity differs",
    )
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    require(
        all(int(row["Device_id"]) == physical for row in rows), "profiler physical card differs"
    )
    require(
        summary["warmup"] == warmup
        and summary["repeats"] == repeats
        and len(summary["durations_us"]) == repeats
        and all(math.isfinite(t) and t > 0 for t in summary["durations_us"])
        and summary["median_us"] == median(summary["durations_us"]),
        "invalid timing samples",
    )
    return summary


def compare_pair(spec, records, timings):
    generated = case_contract(spec, records["generated"], "generated")
    aclnn = case_contract(spec, records["aclnn"], "aclnn")
    require(generated == aclnn, "generated/ACLNN inputs, math, oracle or device differ")
    flops = algorithm_flops(spec.operator, generated["shape"])
    result = {
        "shape": generated["shape"],
        "flops": flops,
        "dtype": "f16",
        "input_contract": generated,
    }
    for backend in ("generated", "aclnn"):
        duration = timings[backend]["median_us"]
        result[backend + "_us"] = duration
        result[backend + "_tflops"] = tflops(flops, duration)
    result["speedup"] = result["aclnn_us"] / result["generated_us"]
    if not spec.attention:
        result["output_cache"] = records["generated"]["cases"][0][
            "generated_buffer_cache_policies"
        ]["C"]
    return result


def render_tables(report):
    parts = []
    for operator, label in (("flash-attention", "Flash Attention"), ("gemm-normal", "GEMM normal")):
        parts.extend(
            [
                f"## {label}",
                "",
                "| shape | AscendFlow µs | ACLNN µs | 加速比 | AscendFlow TFLOPS | ACLNN TFLOPS |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["cases"]:
            if row["operator"] != operator or row["status"] != "passed":
                continue
            shape = "×".join(map(str, row["problem_shape"]))
            parts.append(
                f"| {shape} | {row['generated_us']:.3f} | {row['aclnn_us']:.3f} | "
                f"{row['speedup']:.3f}× | {row['generated_tflops']:.2f} | {row['aclnn_tflops']:.2f} |"
            )
        parts.append("")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical", type=int)
    parser.add_argument("--only", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for spec in CASES:
            print(spec.name)
        return 0
    if args.output is None or args.physical is None:
        parser.error("--output and --physical are required for collection")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["ASCENDFLOW_PYTHON"] = sys.executable
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    report = {
        "status": "running",
        "cases": [],
        "builds": {},
        "date": datetime.now(UTC).date().isoformat(),
        "method": "one default msprof collection per shape/backend; generated then installed ACLNN",
        "tflops_convention": {
            "gemm-normal": "2*M*N*K",
            "flash-attention": "4*B*Hq*Sq*Sk*D (QK + PV)",
            "units": "FLOPs / (device_time_us * 1e6); FMA=2; useful matrix FLOPs",
        },
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {
            str(path.relative_to(ROOT)): digest(path)
            for folder in ("src", "examples", "benchmarks")
            for path in sorted((ROOT / folder).rglob("*.py"))
        },
    }

    def save():
        (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        report["health_before"] = health_snapshot(args.physical, directory)
        for spec in CASES:
            if args.only and spec.name not in args.only:
                continue
            artifact = directory / "artifacts" / spec.artifact_key
            row = {
                "name": spec.name,
                "operator": spec.operator,
                "problem_shape": spec.shape,
                "status": "building",
                "artifact": str(artifact),
                "backends": {},
            }
            report["cases"].append(row)
            save()
            if spec.artifact_key not in report["builds"]:
                command = [
                    sys.executable,
                    "-m",
                    spec.module,
                    "--output",
                    str(artifact),
                    "--build",
                    *spec.build_options,
                ]
                report["builds"][spec.artifact_key] = bounded_process(
                    command, directory / "builds" / spec.artifact_key, timeout_s=300
                )
            records, timings = {}, {}
            for backend in ("generated", "aclnn"):
                profile = directory / spec.name / backend
                profile.parent.mkdir(parents=True, exist_ok=True)
                application = [
                    str(ROOT / "benchmarks/profile.sh"),
                    spec.module,
                    "--output",
                    str(artifact),
                    "--physical",
                    str(args.physical),
                    "--backend",
                    backend,
                    "--scope",
                    "sampled",
                    "--record",
                    str(profile / "device.json"),
                    *spec.run_options,
                ]
                command = ["msprof", f"--output={profile}", *application]
                row["status"] = "profiling-" + backend
                row["backends"][backend] = {"profile_command": command}
                save()
                run_device(command, directory / spec.name / (backend + "-step"))
                records[backend] = json.loads((profile / "device.json").read_text())
                case_contract(spec, records[backend], backend)
                timings[backend] = timing_summary(profile, spec, args.physical)
                summary = timings[backend]
                row["backends"][backend].update(
                    status="passed",
                    kernel_time_us=summary["median_us"],
                    warmup=spec.launch_counts[0],
                    repeats=spec.launch_counts[1],
                    summary=str(profile / "summary.json"),
                    csv=summary["csv"],
                    csv_sha256=summary["csv_sha256"],
                    device_record=str(profile / "device.json"),
                    device_record_sha256=digest(profile / "device.json"),
                )
                print(f"{spec.name}/{backend}: {summary['median_us']:.3f} us", flush=True)
                save()
            row.update(compare_pair(spec, records, timings), status="passed")
            (directory / "tables.md").write_text(render_tables(report))
            save()
        report["health_after"] = health_snapshot(args.physical, directory / "after")
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - preserve failure and stop later cases
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["cases"]:
            report["cases"][-1]["status"] = "failed"
        print(report["error"], file=sys.stderr)
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
