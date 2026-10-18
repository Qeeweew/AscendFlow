"""Paired inputs, actual timing CSVs, FMA counts and one default collection."""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

from ascendflow.diagnostics import CompileError
from benchmarks import profile_defaults as profile


def device_record(spec, backend, tmp_path):
    library = tmp_path / "kernel.so"
    source = tmp_path / "kernel.asc"
    library.write_bytes(b"test binary")
    source.write_text("test ASC")
    manifest = {
        "commands": [["bisheng", str(source), "-o", "kernel.o"]],
        "source_sha256": profile.digest(source),
        "shared_sha256": profile.digest(library),
    }
    # verified_source derives the shared-library filename from the ASC path.
    identity = {"compile_manifest" if spec.attention else "manifest": manifest}
    if backend == "aclnn":
        identity = {
            "libraries": {
                name: {"path": str(library), "sha256": profile.digest(library)}
                for name in ("libopapi.so", "libnnopbase.so")
            },
            "tensor_dtypes": {key: 1 for key in ("A", "B", "C")},
        }
    common = {
        "backend": backend,
        "warmup": spec.launch_counts[0],
        "repeats": spec.launch_counts[1],
        "inputs_unchanged": True,
        "rtol": 0.002,
        "atol": 0.0002,
    }
    if spec.attention:
        common.update(
            shape_b_hq_hkv_sq_sk_d=[1, 8, 8, *spec.shape, 128],
            dtype="f16",
            layout="BNSD",
            causal=False,
            query_start=0,
            data_case="standard",
            oracle_scope="sampled",
            guards_checked=True,
            argument_sha256={key: "same " + key for key in ("q", "k", "v")},
            softmax_scale_f32_bits="constant",
            query_rows_per_head=[0, 7, 63],
        )
    else:
        common.update(
            shape_mnk=list(spec.shape),
            input_dtype="f16",
            output_dtype="f16",
            layout="NN contiguous",
            data="normal",
            scope="sampled",
            finite_output=True,
            offset_elements=0,
            guards_unchanged=True,
            input_sha256=["same A", "same B"],
            seed=20261017,
            oracle_rows=[0, 1],
            generated_buffer_cache_policies={"C": "normal"},
        )
    return {
        "status": "passed",
        "cases": [common],
        "health": {"physical_device": 2, "logical_device": 0},
        "hardware": {"physical_device": 2, "platform_sha256": "same platform"},
        "artifact" if spec.attention else "backend": identity,
    }


def test_algorithm_flops_and_microsecond_conversion():
    assert profile.algorithm_flops("gemm-normal", (4096, 4096, 4096)) == 137438953472
    assert profile.algorithm_flops("gemm-normal", (64, 128, 256)) == 4194304
    assert profile.algorithm_flops("flash-attention", (1, 8, 8, 4096, 4096, 128)) == 68719476736
    # Query heads, rather than KV heads, determine both dense matrix products.
    assert profile.algorithm_flops("flash-attention", (2, 16, 4, 256, 1024, 64)) == 2147483648
    assert profile.tflops(2_000_000_000, 20) == 100


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf])
def test_invalid_duration_cannot_produce_throughput(duration):
    with pytest.raises(CompileError, match="positive and finite"):
        profile.tflops(100, duration)


@pytest.mark.parametrize("change", ["input", "physical", "dtype", "shape", "guards"])
def test_comparison_rejects_unmatched_evidence(change, tmp_path):
    spec = profile.ProfileCase("gemm-normal", (1024, 1024, 1024))
    records = {
        backend: device_record(spec, backend, tmp_path) for backend in ("generated", "aclnn")
    }
    case = records["aclnn"]["cases"][0]
    if change == "input":
        case["input_sha256"][0] = "different input"
    elif change == "physical":
        records["aclnn"]["health"]["physical_device"] = 1
        records["aclnn"]["hardware"]["physical_device"] = 1
    elif change == "dtype":
        case["output_dtype"] = "bf16"
    elif change == "shape":
        case["shape_mnk"][0] = 2048
    else:
        case["guards_unchanged"] = False
    with pytest.raises(CompileError):
        profile.compare_pair(spec, records, {backend: {"median_us": 10} for backend in records})


def test_default_matrix_includes_square_and_rectangular_cases():
    assert len(profile.CASES) == len({case.name for case in profile.CASES}) == 12
    for operator in ("flash-attention", "gemm-normal"):
        shapes = [case.shape for case in profile.CASES if case.operator == operator]
        assert len(shapes) == 6
        assert sum(len(set(shape)) > 1 for shape in shapes) == 2


def test_build_failure_stops_before_profiling_or_next_shape(monkeypatch, tmp_path):
    attempts = []

    def fail(command, *_args, **_kwargs):
        attempts.append(command)
        raise TimeoutError("build deadline reached")

    monkeypatch.setattr(profile, "bounded_process", fail)
    monkeypatch.setattr(profile, "health_snapshot", lambda *_: {"healthy": True})
    monkeypatch.setattr(
        sys, "argv", ["profile_defaults", "--output", str(tmp_path / "failed"), "--physical", "2"]
    )
    assert profile.main() == 1
    assert len(attempts) == 1
    report = json.loads((tmp_path / "failed/results.json").read_text())
    assert report["status"] == "failed"
    assert len(report["cases"]) == 1
    assert report["cases"][0]["backends"] == {}


@pytest.mark.parametrize("physical", [1, 2])
def test_timing_counts_all_kernels_and_rejects_wrong_card(physical, tmp_path):
    spec = profile.ProfileCase("flash-attention", (1024, 1024))
    with (tmp_path / "op_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Device_id",
                "Task Type",
                "Op Name",
                "Task Start Time(us)",
                "Task Duration(us)",
                "Block Num",
            ],
        )
        writer.writeheader()
        for index in range(4):
            for name, duration in (("first", 5), ("second", 7)):
                writer.writerow(
                    {
                        "Device_id": physical,
                        "Task Type": "AI_CORE",
                        "Op Name": name,
                        "Task Start Time(us)": index * 1000 + duration,
                        "Task Duration(us)": 999 if index == 0 else duration,
                        "Block Num": 20,
                    }
                )
    if physical == 1:
        with pytest.raises(CompileError, match="physical card differs"):
            profile.timing_summary(tmp_path, spec, 2)
    else:
        summary = profile.timing_summary(tmp_path, spec, 2)
        assert summary["median_us"] == 12
        assert len(summary["operators"]) == 2


def test_paired_collections_reuse_runtime_build_and_extract_actual_csv(monkeypatch, tmp_path):
    specs = (
        profile.ProfileCase("flash-attention", (1024, 1024)),
        profile.ProfileCase("flash-attention", (2048, 2048)),
        profile.ProfileCase("gemm-normal", (1024, 1024, 1024)),
    )
    monkeypatch.setattr(profile, "CASES", specs)
    monkeypatch.setattr(profile, "health_snapshot", lambda *_: {"healthy": True})
    builds, collected = [], []

    def build(command, *_a, **_k):
        builds.append(command)
        return {"status": "passed"}

    def collect(command, directory):
        assert command[0] == "msprof" and command[1].startswith("--output=")
        assert Path(command[2]).name == "profile.sh"
        assert "--build" not in command
        destination = Path(command[1].split("=", 1)[1])
        backend = command[command.index("--backend") + 1]
        spec = next(spec for spec in specs if spec.name == destination.parent.name)
        collected.append((spec.name, backend))
        destination.mkdir()
        record = device_record(spec, backend, tmp_path)
        (destination / "device.json").write_text(json.dumps(record))
        warmup, repeats = spec.launch_counts
        with (destination / "op_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Device_id",
                    "Task Type",
                    "Op Name",
                    "Task Start Time(us)",
                    "Task Duration(us)",
                    "Block Num",
                ],
            )
            writer.writeheader()
            for i in range(warmup + repeats):
                writer.writerow(
                    {
                        "Device_id": 2,
                        "Task Type": "MIX_AIC" if spec.attention else "AI_CORE",
                        "Op Name": "test_kernel",
                        "Task Start Time(us)": i * 1000,
                        "Task Duration(us)": 999
                        if i < warmup
                        else 10
                        if backend == "generated"
                        else 20,
                        "Block Num": 20,
                    }
                )

    monkeypatch.setattr(profile, "bounded_process", build)
    monkeypatch.setattr(profile, "run_device", collect)
    monkeypatch.setattr(
        sys, "argv", ["profile_defaults", "--output", str(tmp_path / "profiles"), "--physical", "2"]
    )
    assert profile.main() == 0
    assert len(builds) == 2  # one runtime-parameterized FA, one constexpr GEMM
    assert collected == [
        (spec.name, backend) for spec in specs for backend in ("generated", "aclnn")
    ]
    report = json.loads((tmp_path / "profiles/results.json").read_text())
    assert report["status"] == "passed"
    assert all(
        row["generated_us"] == 10 and row["aclnn_us"] == 20 and row["speedup"] == 2
        for row in report["cases"]
    )
    assert all(row["generated_tflops"] == 2 * row["aclnn_tflops"] for row in report["cases"])
    assert "2.000×" in (tmp_path / "profiles/tables.md").read_text()
