"""Compare numerically validated msprof op records with identical decode inputs."""

import argparse
import json
from pathlib import Path

from ascendflow.diagnostics import require
from benchmarks.kda.run import digest


def read_profile(directory, backend):
    metrics = json.loads((directory / "metrics.json").read_text())
    device_path = directory / "device/results.json"
    device = json.loads(device_path.read_text())
    require(
        device["status"] == "passed" and set(device["results"]) == {backend},
        "invalid backend record",
    )
    require(
        metrics["provenance"]["device_record_sha256"] == digest(device_path),
        "numerical record changed after profiling",
    )
    for table in metrics.values():
        if isinstance(table, dict) and "csv" in table:
            require(digest(table["csv"]) == table["csv_sha256"], "profiling CSV changed")
    for relative, sha256 in metrics["provenance"]["triton_cache"].items():
        require(digest(directory / relative) == sha256, "Triton compilation artifact changed")
    numerical = device["results"][backend]["correctness"]
    require(
        all(v["rmse_ratio"] < device["tolerance_rmse_ratio"] for v in numerical.values()),
        "numerical acceptance failed",
    )
    duration = metrics["OpBasicInfo"]["Task Duration(us)"]["mean"]
    require(duration > 0, "invalid kernel duration")
    return device, {
        "kernel_us": duration,
        "correctness": numerical,
        "metrics": str((directory / "metrics.json").resolve()),
        "metrics_sha256": digest(directory / "metrics.json"),
        "device_record": str(device_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascendflow", required=True, type=Path)
    parser.add_argument("--triton", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-release", required=True, type=Path)
    args = parser.parse_args()
    generated, ours = read_profile(args.ascendflow, "ascendflow")
    original, theirs = read_profile(args.triton, "triton")
    keys = (
        "shapes",
        "lengths",
        "dtypes",
        "input_sha256",
        "runtime_parameters",
        "tolerance_rmse_ratio",
        "generated",
    )
    require(
        all(generated[key] == original[key] for key in keys),
        "decode inputs, contracts or artifact differ",
    )
    require(
        generated["health"]["physical_device"] == original["health"]["physical_device"]
        and generated["hardware"]["platform_sha256"] == original["hardware"]["platform_sha256"],
        "device or platform differs",
    )
    require(
        len(generated["lengths"]) >= 128 and set(generated["lengths"]) == {1},
        "comparison requires batch>=128 single-token decode",
    )
    artifact = generated["generated"]
    root = Path(artifact["directory"])
    kernel = generated["operator"]
    require(
        digest(root / f"{kernel}.asc") == artifact["asc_sha256"]
        and digest(root / f"{kernel}.so") == artifact["shared_sha256"],
        "compiled artifact changed",
    )
    release = json.loads(args.official_release.read_text())
    reference = original["reference"]
    for path, sha256 in reference["runtime_files"].items():
        require(digest(path) == sha256, "Triton runtime changed after profiling")
    for relative, sha256 in reference["files"].items():
        require(
            digest(Path(reference["checkout"]) / relative) == sha256, "reference source changed"
        )
    require(
        reference["status"] == ""
        and reference["distribution"] == release["tag"].removeprefix("v")
        and reference["installation"]["archive_info"]["hashes"]["sha256"] == release["sha256"]
        and release["asset"]["digest"] == "sha256:" + release["sha256"],
        "Triton does not match the official release or vllm-ascend checkout is modified",
    )
    report = {
        "status": "passed",
        "scope": "single-token decode; one msprof op Default collection per backend",
        "ascendflow": ours,
        "triton": theirs,
        "speedup": theirs["kernel_us"] / ours["kernel_us"],
        "configuration": {key: generated[key] for key in keys},
        "physical_device": generated["health"]["physical_device"],
        "reference": reference,
        "official_release": str(args.official_release.resolve()),
        "official_release_sha256": digest(args.official_release),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "ascendflow", "triton", "speedup")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
